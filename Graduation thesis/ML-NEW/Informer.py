"""
Daily PM2.5 prediction for BTH/YRD/PRD using Informer (AAAI 2021).
Informer: ProbSparse self-attention + self-attention distillation for long sequence forecasting.
Reuses daily_ml_pipeline for data preparation and sequence building; input is (seq_len, n_features).
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from daily_ml_pipeline import (
    DEFAULT_CITY_GEOJSON_DIR,
    DEFAULT_CORRELATION_DIR,
    DEFAULT_DATA_READ_DIR,
    DEFAULT_ERA5_DAY_DIR,
    DEFAULT_PM25_DAY_DIR,
    SCRIPT_DIR,
    build_pm25_nc_file_index,
    build_daily_features,
    build_prediction_frames,
    build_sequence_matrices,
    compute_metrics,
    export_generalization_artifacts,
    export_regression_artifacts,
    metrics_by_cluster,
    prepare_training_table,
    split_by_time,
)
from cluster_training_utils import prepare_training_table_with_fallback

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "informer_daily_pm25"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # pylint: disable=broad-except
        pass


def get_device(args: argparse.Namespace) -> "torch.device":
    if getattr(args, "device", None) and str(args.device).strip().lower() in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(args, "device", None) and str(args.device).strip().lower() == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp_dtype(
    device: "torch.device",
    enabled: bool,
    amp_dtype: str,
) -> tuple[bool, Any | None]:
    if torch is None or device.type != "cuda" or not enabled:
        return False, None
    mode = str(amp_dtype).strip().lower()
    if mode == "bf16":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16
        print("[WARN] CUDA bf16 is not supported on this GPU; fallback to fp16 AMP.")
        return True, torch.float16
    if mode in ("fp16", "float16"):
        return True, torch.float16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype}. Expected one of: bf16, fp16")


def _ensure_min_std(std_values: np.ndarray, min_std: float) -> np.ndarray:
    return np.where(std_values < float(min_std), float(min_std), std_values).astype(np.float32)


def fit_train_standardizers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    min_std: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training arrays must be non-empty for standardization.")
    x_mean = X_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    x_std = X_train.std(axis=(0, 1), keepdims=True).astype(np.float32)
    x_std = _ensure_min_std(x_std, min_std=min_std)
    y_mean = float(np.mean(y_train))
    y_std = max(float(np.std(y_train)), float(min_std))
    return x_mean, x_std, y_mean, y_std


def transform_inputs(X: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return X.astype(np.float32, copy=False)
    return ((X - x_mean) / x_std).astype(np.float32, copy=False)


def transform_targets(y: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    if len(y) == 0:
        return y.astype(np.float32, copy=False)
    return ((y - y_mean) / y_std).astype(np.float32, copy=False)


def inverse_transform_targets(y_scaled: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    if len(y_scaled) == 0:
        return y_scaled.astype(np.float32, copy=False)
    return (y_scaled * y_std + y_mean).astype(np.float32, copy=False)


def apply_target_transform(y: np.ndarray, mode: str) -> np.ndarray:
    mode_normalized = str(mode).strip().lower()
    values = y.astype(np.float32, copy=False)
    if mode_normalized == "none":
        return values
    if mode_normalized == "log1p":
        clipped = np.clip(values, a_min=-0.999999, a_max=None)
        return np.log1p(clipped).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported target transform mode: {mode}")


def inverse_target_transform(y_transformed: np.ndarray, mode: str) -> np.ndarray:
    mode_normalized = str(mode).strip().lower()
    values = y_transformed.astype(np.float32, copy=False)
    if mode_normalized == "none":
        return values
    if mode_normalized == "log1p":
        return np.expm1(values).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported target transform mode: {mode}")


class ProbSparseAttention(nn.Module):
    """
    ProbSparse self-attention (Informer): select top-u queries by sparsity measure M(q,K).
    M(q_i, K) = max_j(scores_ij) - mean_j(scores_ij) approximates query "activity".
    Top-u queries get full attention; others use mean(V). Reduces O(L^2) to O(L log L).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        factor: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: "torch.Tensor",
        mask: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        B, L, _ = x.shape
        q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        k = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, L, L)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))

        u = max(1, int(self.factor * math.log(max(L, 2))))
        u = min(u, L)
        m_i = scores.max(dim=-1).values - scores.mean(dim=-1)  # (B, H, L)
        _, top_indices = torch.topk(m_i, u, dim=-1)  # (B, H, u)

        v_mean = v.mean(dim=2, keepdim=True)  # (B, H, 1, d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        out_attn = torch.matmul(attn_weights, v)  # (B, H, L, d_k)
        out_attn = out_attn.reshape(B, self.n_heads, L, self.d_k)
        out_full = v_mean.expand(-1, -1, L, -1).clone()
        # 向量化替代 Python 双层 for 循环：gather + scatter
        top_idx_exp = top_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
        gathered = torch.gather(out_attn, 2, top_idx_exp)
        out_full.scatter_(2, top_idx_exp, gathered)
        out = out_full.permute(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.dropout(self.w_out(out))


class InformerEncoderLayer(nn.Module):
    """Informer encoder layer: ProbSparse attention + FFN + distillation (Conv1d + MaxPool)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, factor=factor, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.distill = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(
        self,
        x: "torch.Tensor",
        mask: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        attn_out = self.attn(self.norm1(x), mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        x = x.transpose(1, 2)
        x = self.distill(x)
        return x.transpose(1, 2)


class InformerRegressor(nn.Module):
    """
    Informer for sequence-to-one regression.
    Input (B, T, F) -> ProbSparse encoder with distillation -> mean pool -> FC -> scalar.
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        factor: int = 5,
        dropout: float = 0.2,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        assert d_model % n_heads == 0
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layers: list[nn.Module] = []
        cur_len = seq_len
        for _ in range(n_layers):
            layers.append(
                InformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    factor=factor,
                    dropout=dropout,
                )
            )
            cur_len = (cur_len + 1) // 2
        self.encoder_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        B, T, F = x.shape
        x = self.input_proj(x) + self.pos_enc[:, :T, :]
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.dropout(x.mean(dim=1))
        return self.fc(x).squeeze(-1)


def build_informer_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    batch_size: int,
    device: "torch.device",
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    pin_memory: bool | None = None,
) -> tuple["DataLoader", "DataLoader | None"]:
    use_cuda = device.type == "cuda"
    use_pin_memory = use_cuda if pin_memory is None else bool(pin_memory)
    kw = {
        "batch_size": batch_size,
        "pin_memory": use_pin_memory,
        "num_workers": max(0, int(num_workers)),
    }
    if int(num_workers) > 0:
        kw["persistent_workers"] = bool(persistent_workers)
        kw["prefetch_factor"] = max(1, int(prefetch_factor))
    X_tr = torch.from_numpy(X_train)
    y_tr = torch.from_numpy(y_train).unsqueeze(1)
    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    valid_loader = None
    if len(X_valid) > 0:
        X_va = torch.from_numpy(X_valid)
        y_va = torch.from_numpy(y_valid).unsqueeze(1)
        valid_ds = TensorDataset(X_va, y_va)
        valid_loader = DataLoader(valid_ds, shuffle=False, **kw)
    return train_loader, valid_loader


def train_informer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_cols: list[str],
    args: argparse.Namespace,
    device: "torch.device",
    output_dir: Path,
    save_checkpoint: bool = True,
    report_callback: Callable[[int, float], None] | None = None,
) -> tuple["nn.Module", float]:
    if torch is None or nn is None:
        raise ImportError("PyTorch is not installed. Please run: pip install torch")

    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    model = InformerRegressor(
        input_size=n_features,
        seq_len=seq_len,
        d_model=int(getattr(args, "d_model", 64)),
        n_heads=int(getattr(args, "n_heads", 4)),
        n_layers=int(getattr(args, "n_layers", 2)),
        dim_feedforward=int(getattr(args, "dim_feedforward", 256)),
        factor=int(getattr(args, "factor", 5)),
        dropout=float(getattr(args, "dropout", 0.2)),
        seed=args.seed,
    ).to(device)
    if bool(getattr(args, "compile", False)):
        if hasattr(torch, "compile"):
            compile_mode = str(getattr(args, "compile_mode", "default"))
            try:
                model = torch.compile(model, mode=compile_mode)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] torch.compile failed, fallback to eager mode: {exc}")
        else:
            print("[WARN] torch.compile is unavailable in current PyTorch version; skipping compile.")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=float(getattr(args, "weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    num_workers = int(getattr(args, "dataloader_workers", 0))
    prefetch_factor = int(getattr(args, "prefetch_factor", 2))
    persistent_workers = bool(getattr(args, "persistent_workers", True))
    pin_memory = bool(getattr(args, "pin_memory", device.type == "cuda"))
    train_loader, valid_loader = build_informer_dataloaders(
        X_train,
        y_train,
        X_valid,
        y_valid,
        args.batch_size,
        device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    amp_enabled, amp_dtype = resolve_amp_dtype(
        device=device,
        enabled=bool(getattr(args, "amp", False)),
        amp_dtype=str(getattr(args, "amp_dtype", "bf16")),
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=bool(amp_enabled and amp_dtype == torch.float16),
        )
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=bool(amp_enabled and amp_dtype == torch.float16))

    best_rmse = float("inf")
    patience = getattr(args, "early_stopping_patience", 15)
    epochs_no_improve = 0
    best_state: dict[str, Any] | None = None

    epoch_iter = range(1, args.epochs + 1)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc="Informer 训练轮次", dynamic_ncols=True)

    for epoch in epoch_iter:
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True).squeeze(1)
            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_enabled and amp_dtype is not None
                else nullcontext()
            )
            with autocast_ctx:
                out = model(X_b)
                loss = criterion(out, y_b)
            grad_clip_norm = float(getattr(args, "grad_clip_norm", 5.0))
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
            train_loss += float(loss.detach()) * X_b.size(0)
        train_loss /= len(X_train)
        train_rmse = float(np.sqrt(train_loss))

        if valid_loader is not None:
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in valid_loader:
                    X_b = X_b.to(device, non_blocking=True)
                    y_b = y_b.to(device, non_blocking=True).squeeze(1)
                    autocast_ctx = (
                        torch.autocast(device_type="cuda", dtype=amp_dtype)
                        if amp_enabled and amp_dtype is not None
                        else nullcontext()
                    )
                    with autocast_ctx:
                        out = model(X_b)
                        valid_batch_loss = criterion(out, y_b)
                    valid_loss += float(valid_batch_loss.detach()) * X_b.size(0)
            valid_loss /= len(X_valid)
            valid_rmse = float(np.sqrt(valid_loss))
            if report_callback is not None:
                report_callback(epoch, valid_rmse)
            scheduler.step(valid_rmse)
            if valid_rmse < best_rmse:
                best_rmse = valid_rmse
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
            if tqdm is not None:
                epoch_iter.set_postfix(
                    train_rmse=train_rmse,
                    valid_rmse=valid_rmse,
                    best_rmse=best_rmse,
                    no_improve=epochs_no_improve,
                )
            if epochs_no_improve >= patience:
                if best_state is not None:
                    model.load_state_dict(best_state)
                break
        else:
            if report_callback is not None:
                report_callback(epoch, train_rmse)
            scheduler.step(train_rmse)
            if train_rmse < best_rmse:
                best_rmse = train_rmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if tqdm is not None:
                epoch_iter.set_postfix(train_rmse=train_rmse, best_rmse=best_rmse)

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_checkpoint:
        torch.save(model.state_dict(), output_dir / "informer_best.pt")
    return model, best_rmse


def predict_numpy_seq(
    model: "nn.Module",
    X: np.ndarray,
    device: "torch.device",
    batch_size: int = 4096,
    use_amp: bool = False,
    amp_dtype: str = "bf16",
) -> np.ndarray:
    if len(X) == 0:
        return np.array([], dtype=np.float32)
    model.eval()
    preds: list[torch.Tensor] = []
    start_indices = range(0, len(X), batch_size)
    if tqdm is not None:
        total_batches = (len(X) + batch_size - 1) // batch_size
        start_indices = tqdm(start_indices, desc="批量预测", total=total_batches, leave=False, dynamic_ncols=True)
    amp_enabled, resolved_amp_dtype = resolve_amp_dtype(device=device, enabled=use_amp, amp_dtype=amp_dtype)
    with torch.no_grad():
        for i in start_indices:
            batch = torch.from_numpy(X[i : i + batch_size]).to(device, non_blocking=True)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=resolved_amp_dtype)
                if amp_enabled and resolved_amp_dtype is not None
                else nullcontext()
            )
            with autocast_ctx:
                out = model(batch)
            preds.append(out.float().cpu())
    return torch.cat(preds, dim=0).numpy().astype(np.float32)


def compute_feature_importance_last_step(
    model: "nn.Module",
    X: np.ndarray,
    feature_cols: list[str],
    device: "torch.device",
    n_samples: int = 2000,
) -> np.ndarray:
    if len(X) == 0 or len(feature_cols) == 0:
        return np.zeros(len(feature_cols), dtype=np.float64)
    was_training = model.training
    model.eval()
    n = min(n_samples, len(X))
    idx = np.random.RandomState(42).choice(len(X), size=n, replace=False)
    sample = X[idx]
    X_t = torch.from_numpy(sample).to(device)
    X_t.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    grad = None
    try:
        with torch.backends.cudnn.flags(enabled=False):
            out = model(X_t)
            out.sum().backward()
        grad = X_t.grad
    finally:
        if was_training:
            model.train()
    if grad is None:
        return np.zeros(len(feature_cols), dtype=np.float64)
    if grad.ndim != 3 or grad.shape[2] != len(feature_cols):
        return np.zeros(len(feature_cols), dtype=np.float64)
    last_step = grad[:, -1, :]
    imp = (X_t[:, -1, :].detach().abs() * last_step.abs()).mean(dim=0).cpu().numpy()
    return np.asarray(imp, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Daily PM2.5 prediction for BTH/YRD/PRD using Informer (AAAI 2021)."
    )
    parser.add_argument("--daily-input", nargs="+", default=None, help="Optional daily long-table file(s)/folder(s).")
    parser.add_argument("--pm25-day-dir", type=str, default=str(DEFAULT_PM25_DAY_DIR), help="PM2.5 daily NC directory.")
    parser.add_argument("--era5-day-dir", type=str, default=str(DEFAULT_ERA5_DAY_DIR), help="ERA5 daily NC directory.")
    parser.add_argument("--city-geojson-dir", type=str, default=str(DEFAULT_CITY_GEOJSON_DIR), help="City geojson directory.")
    parser.add_argument("--skip-era5", action="store_true", help="Skip merging ERA5 daily meteorological features.")
    parser.add_argument("--correlation-dir", type=str, default=str(DEFAULT_CORRELATION_DIR), help="Correlation directory.")
    parser.add_argument("--data-read-dir", type=str, default=str(DEFAULT_DATA_READ_DIR), help="Data Read directory.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--train-end-year", type=int, default=2021, help="Train years <= this year.")
    parser.add_argument("--valid-year", type=int, default=2022, help="Validation year.")
    parser.add_argument("--test-year", type=int, default=2023, help="Test year.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--seq-len", type=int, default=14, help="Sequence length (days) for input.")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=4096, help="Evaluation/prediction batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--grad-clip-norm", type=float, default=5.0, help="Gradient clipping; <=0 disables.")
    parser.add_argument("--dataloader-workers", type=int, default=0, help="DataLoader workers for training/validation.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when workers > 0.")
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=True)
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training/inference on CUDA.")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=("bf16", "fp16"), help="AMP dtype.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for model acceleration.")
    parser.add_argument("--compile-mode", type=str, default="default", help="torch.compile mode.")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/cudnn on CUDA for faster training.")
    parser.add_argument("--min-std", type=float, default=1e-6, help="Lower bound for std in scaling.")
    parser.add_argument("--d-model", type=int, default=64, help="Informer d_model.")
    parser.add_argument("--n-heads", type=int, default=4, help="Attention heads.")
    parser.add_argument("--n-layers", type=int, default=2, help="Encoder layers.")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="FFN dimension.")
    parser.add_argument("--factor", type=int, default=5, help="ProbSparse factor (u = factor * log(L)).")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument(
        "--target-transform",
        type=str,
        default="log1p",
        choices=("none", "log1p"),
        help="Target transform before standardization.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--device", type=str, default="", help="Device: 'cuda', 'gpu', or 'cpu'.")
    parser.add_argument("--pm25-workers", type=int, default=64, help="Worker count for PM2.5 NC reading.")
    parser.add_argument("--era5-workers", type=int, default=64, help="Worker count for ERA5 NC reading.")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_OUTPUT_DIR / "cache"), help="Cache directory.")
    parser.add_argument("--disable-data-cache", action="store_true", help="Disable NC aggregation cache.")
    parser.add_argument("--pm25-precropped", dest="pm25_precropped", action="store_true", default=True)
    parser.add_argument("--no-pm25-precropped", dest="pm25_precropped", action="store_false")
    return parser


def main() -> int:
    if torch is None:
        raise ImportError("PyTorch is not installed. Please run: pip install torch")
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    correlation_dir = Path(args.correlation_dir).expanduser().resolve()
    data_read_dir = Path(args.data_read_dir).expanduser().resolve()
    pm25_day_dir = Path(args.pm25_day_dir).expanduser().resolve()
    era5_day_dir = Path(args.era5_day_dir).expanduser().resolve()
    city_geojson_dir = Path(args.city_geojson_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_pbar = tqdm(total=6, desc="Informer 总进度", dynamic_ncols=True) if tqdm is not None else None

    device = get_device(args)
    if device.type == "cuda":
        use_tf32 = bool(getattr(args, "tf32", False))
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cudnn.benchmark = True
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] TF32 enabled: {use_tf32}")
    else:
        print("[INFO] Using CPU.")

    train_seconds_total = 0.0
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_enabled = not args.disable_data_cache
    full_begin = time.perf_counter()
    target_cluster_order = ["BTH", "YRD", "PRD"]
    cluster_results: dict[str, dict[str, Any]] = {}
    prebuilt_pm25_nc_index = build_pm25_nc_file_index(pm25_day_dir) if not args.daily_input else None

    try:
        (
            base_df,
            year_factor_cols,
            met_cols,
            prepare_stats_main,
            training_mode,
            fallback_years,
        ) = prepare_training_table_with_fallback(
            module_tag="informer",
            correlation_dir=correlation_dir,
            data_read_dir=data_read_dir,
            city_geojson_dir=city_geojson_dir,
            daily_input=args.daily_input,
            pm25_day_dir=pm25_day_dir,
            era5_day_dir=era5_day_dir,
            include_era5_daily=not args.skip_era5,
            pm25_workers=int(args.pm25_workers),
            era5_workers=int(args.era5_workers),
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            train_end_year=int(args.train_end_year),
            valid_year=int(args.valid_year),
            test_year=int(args.test_year),
            prepare_fn=prepare_training_table,
            prebuilt_pm25_nc_index=prebuilt_pm25_nc_index,
            use_year_factors=False,
            pm25_precropped=args.pm25_precropped,
        )
    except Exception:
        if stage_pbar is not None:
            stage_pbar.close()
        raise

    if stage_pbar is not None:
        stage_pbar.update(1)
        stage_pbar.set_postfix_str("数据读取完成")
    feature_df = build_daily_features(base_df)
    train_df, valid_df, test_df = split_by_time(
        feature_df,
        train_end_year=args.train_end_year,
        valid_year=args.valid_year,
        test_year=args.test_year,
    )
    available_clusters = sorted(
        set(train_df["cluster"].dropna().astype(str))
        | set(valid_df["cluster"].dropna().astype(str))
        | set(test_df["cluster"].dropna().astype(str))
    )
    clusters_to_train = [name for name in target_cluster_order if name in available_clusters]
    if not clusters_to_train:
        if stage_pbar is not None:
            stage_pbar.close()
        raise ValueError("No cluster samples found for BTH/YRD/PRD after split.")
    if stage_pbar is not None:
        stage_pbar.update(1)
        stage_pbar.set_postfix_str("序列特征构建完成")

    for cluster_name in clusters_to_train:
        cluster_train_df = train_df.loc[train_df["cluster"] == cluster_name].copy()
        cluster_valid_df = valid_df.loc[valid_df["cluster"] == cluster_name].copy()
        cluster_test_df = test_df.loc[test_df["cluster"] == cluster_name].copy()
        (
            X_train, y_train, meta_train,
            X_valid, y_valid, meta_valid,
            X_test, y_test, meta_test,
            feature_cols,
            city_categories,
        ) = build_sequence_matrices(cluster_train_df, cluster_valid_df, cluster_test_df, seq_len=args.seq_len)
        if len(X_train) == 0 or len(X_valid) == 0 or len(X_test) == 0:
            print(
                f"[WARN] 跳过城市群 {cluster_name}: "
                f"train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)}"
            )
            continue

        target_transform_mode = str(getattr(args, "target_transform", "log1p")).strip().lower()
        y_train_model = apply_target_transform(y_train, target_transform_mode)
        y_valid_model = apply_target_transform(y_valid, target_transform_mode)
        x_mean, x_std, y_mean, y_std = fit_train_standardizers(
            X_train=X_train,
            y_train=y_train_model,
            min_std=float(args.min_std),
        )
        X_train_scaled = transform_inputs(X_train, x_mean, x_std)
        X_valid_scaled = transform_inputs(X_valid, x_mean, x_std)
        X_test_scaled = transform_inputs(X_test, x_mean, x_std)
        y_train_scaled = transform_targets(y_train_model, y_mean, y_std)
        y_valid_scaled = transform_targets(y_valid_model, y_mean, y_std)

        cluster_output_dir = output_dir / cluster_name.lower()
        cluster_output_dir.mkdir(parents=True, exist_ok=True)

        train_begin = time.perf_counter()
        model, _ = train_informer(
            X_train_scaled,
            y_train_scaled,
            X_valid_scaled,
            y_valid_scaled,
            feature_cols,
            args,
            device,
            cluster_output_dir,
        )
        train_seconds = time.perf_counter() - train_begin
        train_seconds_total += train_seconds

        pred_train_scaled = predict_numpy_seq(
            model,
            X_train_scaled,
            device,
            batch_size=int(getattr(args, "eval_batch_size", 4096)),
            use_amp=bool(getattr(args, "amp", False)),
            amp_dtype=str(getattr(args, "amp_dtype", "bf16")),
        )
        pred_valid_scaled = predict_numpy_seq(
            model,
            X_valid_scaled,
            device,
            batch_size=int(getattr(args, "eval_batch_size", 4096)),
            use_amp=bool(getattr(args, "amp", False)),
            amp_dtype=str(getattr(args, "amp_dtype", "bf16")),
        )
        pred_test_scaled = predict_numpy_seq(
            model,
            X_test_scaled,
            device,
            batch_size=int(getattr(args, "eval_batch_size", 4096)),
            use_amp=bool(getattr(args, "amp", False)),
            amp_dtype=str(getattr(args, "amp_dtype", "bf16")),
        )
        pred_train = inverse_target_transform(
            inverse_transform_targets(pred_train_scaled, y_mean, y_std), target_transform_mode
        )
        pred_valid = inverse_target_transform(
            inverse_transform_targets(pred_valid_scaled, y_mean, y_std), target_transform_mode
        )
        pred_test = inverse_target_transform(
            inverse_transform_targets(pred_test_scaled, y_mean, y_std), target_transform_mode
        )

        metric_rows = [
            {"split": "train", **compute_metrics(pd.Series(y_train), pred_train), "n_samples": int(len(y_train))},
            {"split": "valid", **compute_metrics(pd.Series(y_valid), pred_valid), "n_samples": int(len(y_valid))},
            {"split": "test", **compute_metrics(pd.Series(y_test), pred_test), "n_samples": int(len(y_test))},
        ]
        metrics_df = pd.DataFrame(metric_rows)
        all_pred_df, test_pred_df = build_prediction_frames(
            train_df=meta_train,
            valid_df=meta_valid,
            test_df=meta_test,
            pred_train=pred_train,
            pred_valid=pred_valid,
            pred_test=pred_test,
        )
        importance = compute_feature_importance_last_step(
            model,
            X_valid_scaled if len(X_valid_scaled) > 0 else X_train_scaled,
            feature_cols,
            device,
        )
        importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": importance, "cluster": cluster_name}
        ).sort_values("importance", ascending=False, kind="mergesort")

        joblib.dump(
            {
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": float(y_mean),
                "y_std": float(y_std),
                "feature_cols": feature_cols,
                "seq_len": args.seq_len,
                "target_transform": target_transform_mode,
            },
            cluster_output_dir / "informer_scaler.joblib",
        )

        cluster_results[cluster_name] = {
            "model": model,
            "feature_cols": feature_cols,
            "all_pred_df": all_pred_df,
            "test_pred_df": test_pred_df,
            "metrics_df": metrics_df,
            "importance_df": importance_df,
            "train_rows": int(len(X_train)),
            "valid_rows": int(len(X_valid)),
            "test_rows": int(len(X_test)),
            "train_seconds": float(train_seconds),
        }

    if not cluster_results:
        if stage_pbar is not None:
            stage_pbar.close()
        raise ValueError("No cluster model was successfully trained.")
    if stage_pbar is not None:
        stage_pbar.update(1)
        stage_pbar.set_postfix_str("模型训练完成")
        stage_pbar.update(1)
        stage_pbar.set_postfix_str("预测完成")

    per_cluster_run_info: dict[str, Any] = {}
    all_pred_frames: list[pd.DataFrame] = []
    test_pred_frames: list[pd.DataFrame] = []
    metrics_by_cluster_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for cluster_name in target_cluster_order:
        if cluster_name not in cluster_results:
            continue
        cluster_output_dir = output_dir / cluster_name.lower()
        result = cluster_results[cluster_name]
        all_pred_df = result["all_pred_df"]
        test_pred_df = result["test_pred_df"]
        metrics_df = result["metrics_df"]
        importance_df = result["importance_df"]
        cluster_metrics_df = metrics_by_cluster(test_pred_df)
        generalization_df = export_generalization_artifacts(metrics_df, cluster_output_dir)
        export_regression_artifacts(
            all_pred_df=all_pred_df,
            output_dir=cluster_output_dir,
            model_name=f"Informer-{cluster_name}",
        )
        metrics_df.to_csv(cluster_output_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")
        cluster_metrics_df.to_csv(
            cluster_output_dir / "metrics_by_cluster_test.csv", index=False, encoding="utf-8-sig"
        )
        test_pred_df.sort_values(["date", "city"], kind="mergesort").to_csv(
            cluster_output_dir / "predictions_test.csv", index=False, encoding="utf-8-sig"
        )
        all_pred_df.sort_values(["split", "date", "city"], kind="mergesort").to_csv(
            cluster_output_dir / "predictions_all_splits.csv", index=False, encoding="utf-8-sig"
        )
        importance_df.to_csv(
            cluster_output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig"
        )
        metrics_by_cluster_frames.append(metrics_df.assign(cluster=cluster_name))
        all_pred_frames.append(all_pred_df.assign(model_cluster=cluster_name))
        test_pred_frames.append(test_pred_df.assign(model_cluster=cluster_name))
        importance_frames.append(importance_df)
        per_cluster_run_info[cluster_name] = {
            "output_dir": str(cluster_output_dir),
            "n_features": int(len(result["feature_cols"])),
            "train_rows": int(result["train_rows"]),
            "valid_rows": int(result["valid_rows"]),
            "test_rows": int(result["test_rows"]),
            "train_seconds": float(result["train_seconds"]),
            "generalization_level": str(generalization_df.loc[0, "generalization_level"])
            if not generalization_df.empty
            else "",
        }

    all_pred_df = pd.concat(all_pred_frames, ignore_index=True) if all_pred_frames else pd.DataFrame()
    test_pred_df = pd.concat(test_pred_frames, ignore_index=True) if test_pred_frames else pd.DataFrame()
    metrics_overall_by_cluster_df = (
        pd.concat(metrics_by_cluster_frames, ignore_index=True)
        if metrics_by_cluster_frames
        else pd.DataFrame()
    )
    feature_importance_by_cluster_df = (
        pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    )
    pooled_metric_rows: list[dict[str, Any]] = []
    for split_name in ("train", "valid", "test"):
        split_df = all_pred_df.loc[all_pred_df["split"] == split_name].copy()
        if split_df.empty:
            pooled_metric_rows.append(
                {"split": split_name, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n_samples": 0}
            )
            continue
        split_metrics = compute_metrics(split_df["y_true"], split_df["y_pred"].to_numpy())
        pooled_metric_rows.append({"split": split_name, **split_metrics, "n_samples": int(len(split_df))})
    metrics_df = pd.DataFrame(pooled_metric_rows)
    cluster_metrics_df = metrics_by_cluster(test_pred_df)
    pooled_generalization_df = export_generalization_artifacts(metrics_df, output_dir)
    export_regression_artifacts(
        all_pred_df=all_pred_df,
        output_dir=output_dir,
        model_name="Informer-ClusterModels",
    )
    if stage_pbar is not None:
        stage_pbar.update(1)
        stage_pbar.set_postfix_str("评估与图表完成")

    metrics_df.to_csv(output_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(
        output_dir / "metrics_overall_pooled_from_cluster_models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_overall_by_cluster_df.to_csv(
        output_dir / "metrics_overall_by_cluster.csv", index=False, encoding="utf-8-sig"
    )
    cluster_metrics_df.to_csv(
        output_dir / "metrics_by_cluster_test.csv", index=False, encoding="utf-8-sig"
    )
    test_pred_df.sort_values(["date", "city"], kind="mergesort").to_csv(
        output_dir / "predictions_test.csv", index=False, encoding="utf-8-sig"
    )
    all_pred_df.sort_values(["split", "date", "city"], kind="mergesort").to_csv(
        output_dir / "predictions_all_splits.csv", index=False, encoding="utf-8-sig"
    )
    all_pred_df.sort_values(["split", "date", "city"], kind="mergesort").to_csv(
        output_dir / "predictions_all_splits_with_cluster.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_pred_df.sort_values(["date", "city"], kind="mergesort").to_csv(
        output_dir / "predictions_test_with_cluster.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_importance_by_cluster_df.to_csv(
        output_dir / "feature_importance_by_cluster.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if not feature_importance_by_cluster_df.empty:
        feature_importance_by_cluster_df.groupby("feature", as_index=False)["importance"].mean().sort_values(
            "importance", ascending=False, kind="mergesort"
        ).reset_index(drop=True).to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    n_features_by_cluster = {
        cluster: int(info["n_features"]) for cluster, info in per_cluster_run_info.items()
    }
    run_info = {
        "model": "Informer",
        "training_granularity": "cluster",
        "clusters_trained": list(per_cluster_run_info.keys()),
        "per_cluster": per_cluster_run_info,
        "device": str(device),
        "train_end_year": args.train_end_year,
        "valid_year": args.valid_year,
        "test_year": args.test_year,
        "seq_len": args.seq_len,
        "n_features": int(max(n_features_by_cluster.values())) if n_features_by_cluster else 0,
        "n_features_by_cluster": n_features_by_cluster,
        "n_year_factor_features": len(year_factor_cols),
        "n_era5_daily_features": len(met_cols),
        "train_rows": int(sum(info["train_rows"] for info in per_cluster_run_info.values())),
        "valid_rows": int(sum(info["valid_rows"] for info in per_cluster_run_info.values())),
        "test_rows": int(sum(info["test_rows"] for info in per_cluster_run_info.values())),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": int(getattr(args, "eval_batch_size", 4096)),
        "lr": args.lr,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "factor": args.factor,
        "dropout": args.dropout,
        "dataloader_workers": int(getattr(args, "dataloader_workers", 0)),
        "prefetch_factor": int(getattr(args, "prefetch_factor", 2)),
        "persistent_workers": bool(getattr(args, "persistent_workers", True)),
        "pin_memory": bool(getattr(args, "pin_memory", device.type == "cuda")),
        "amp": bool(getattr(args, "amp", False)),
        "amp_dtype": str(getattr(args, "amp_dtype", "bf16")),
        "compile": bool(getattr(args, "compile", False)),
        "compile_mode": str(getattr(args, "compile_mode", "default")),
        "tf32": bool(getattr(args, "tf32", False)),
        "daily_input": args.daily_input or [],
        "pm25_day_dir": str(pm25_day_dir),
        "era5_day_dir": str(era5_day_dir),
        "generalization_level": str(pooled_generalization_df.loc[0, "generalization_level"])
        if not pooled_generalization_df.empty
        else "",
        "training_mode": training_mode,
        "fallback_years": fallback_years,
        "pm25_workers": int(args.pm25_workers),
        "era5_workers": int(args.era5_workers),
        "cache_dir": str(cache_dir),
        "cache_enabled": bool(cache_enabled),
        "cache_hit_pm25": bool(prepare_stats_main.get("cache_hit_pm25", False)),
        "cache_hit_era5": bool(prepare_stats_main.get("cache_hit_era5", False)),
        "data_prepare_seconds": float(prepare_stats_main.get("data_prepare_seconds", 0.0)),
        "pm25_read_seconds": float(prepare_stats_main.get("pm25_seconds", 0.0)),
        "era5_read_seconds": float(prepare_stats_main.get("era5_seconds", 0.0)),
        "train_seconds": float(train_seconds_total),
        "total_elapsed_seconds": float(time.perf_counter() - full_begin),
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    if stage_pbar is not None:
        stage_pbar.update(1)
        stage_pbar.set_postfix_str("结果导出完成")
        stage_pbar.close()

    print("=" * 90)
    print("[INFO] Informer daily PM2.5 training finished.")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Trained clusters: {', '.join(per_cluster_run_info.keys())}")
    for cluster_name in per_cluster_run_info:
        print(f"       - {cluster_name}: {output_dir / cluster_name.lower()}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
