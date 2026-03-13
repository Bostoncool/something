"""
Daily PM2.5 prediction for BTH/YRD/PRD using Temporal Fusion Transformer (TFT).
Reuses daily_ml_pipeline for data preparation and sequence building; input is (seq_len, n_features).
TFT components: variable selection, LSTM, interpretable multi-head attention, gated residual.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError:
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
    export_feature_quality_report,
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
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "tft_daily_pm25"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_device(args: argparse.Namespace) -> "torch.device":
    if getattr(args, "device", None) and str(args.device).strip().lower() in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(args, "device", None) and str(args.device).strip().lower() == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def describe_vector(name: str, values: np.ndarray) -> str:
    if values.size == 0:
        return f"{name}: empty"
    q = np.quantile(values, [0.1, 0.5, 0.9])
    return (
        f"{name}: mean={float(np.mean(values)):.4f}, std={float(np.std(values)):.4f}, "
        f"min={float(np.min(values)):.4f}, p10={float(q[0]):.4f}, p50={float(q[1]):.4f}, "
        f"p90={float(q[2]):.4f}, max={float(np.max(values)):.4f}"
    )


# ---------- TFT building blocks ----------


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) from TFT: suppresses unnecessary components."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int | None = None,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        if context_size is not None and context_size > 0:
            self.context_proj = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context_proj = None
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(output_size, output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.input_proj: nn.Module | None = None
        if input_size != output_size:
            self.input_proj = nn.Linear(input_size, output_size)

    def forward(
        self,
        x: "torch.Tensor",
        context: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        # x: (..., input_size), context: (..., context_size) or None
        hidden = self.elu(self.fc1(x))
        if self.context_proj is not None and context is not None:
            hidden = hidden + self.context_proj(context)
        out = self.fc2(hidden)
        gate = torch.sigmoid(self.gate(out))
        out = gate * out
        if self.input_proj is not None:
            out = out + self.input_proj(x)
        else:
            out = out + x[..., : self.output_size]
        return self.dropout(out)


class VariableSelectionNetwork(nn.Module):
    """Variable selection: static context weights the importance of each input feature."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        output_size: int,
        static_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.per_var = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(n_features)
        ])
        self.weights_net = nn.Sequential(
            nn.Linear(static_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, n_features),
        )
        self.output_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout,
            context_size=static_size,
        )

    def forward(
        self,
        x: "torch.Tensor",
        static: "torch.Tensor",
    ) -> "torch.Tensor":
        # x: (B, T, F), static: (B, static_dim)
        B, T, F = x.shape
        assert F == self.n_features
        var_embeds = []
        for i in range(self.n_features):
            var_embeds.append(self.per_var[i](x[..., i : i + 1]))  # (B, T, hidden)
        stacked = torch.stack(var_embeds, dim=2)  # (B, T, F, hidden)
        weights = torch.softmax(self.weights_net(static), dim=-1)  # (B, F)
        weights = weights.unsqueeze(1).unsqueeze(-1)  # (B, 1, F, 1)
        selected = (stacked * weights).sum(dim=2)  # (B, T, hidden)
        static_expand = static.unsqueeze(1).expand(-1, T, -1)  # (B, T, static)
        out = self.output_grn(selected, static_expand)  # (B, T, output_size)
        return out


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with shared value projection for interpretability (TFT-style)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: "torch.Tensor", mask: "torch.Tensor | None" = None) -> "torch.Tensor":
        # x: (B, T, d_model)
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_out(out)


class TFTRegressor(nn.Module):
    """
    Temporal Fusion Transformer for sequence-to-one regression.
    Input: (B, T, F) past features, city_id (B,) as static.
    Output: (B,) scalar prediction.
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        n_cities: int,
        hidden_size: int = 64,
        lstm_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.2,
        city_embed_dim: int = 16,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self._input_size = input_size
        self._seq_len = seq_len
        self._n_cities = max(1, n_cities)
        self.hidden_size = hidden_size
        if hidden_size % attention_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by attention_heads ({attention_heads}).")

        self.static_embed = nn.Sequential(
            nn.Embedding(max(1, n_cities), city_embed_dim),
            nn.Linear(city_embed_dim, hidden_size),
            nn.ELU(),
        )
        static_size = hidden_size

        self.variable_selection = VariableSelectionNetwork(
            n_features=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            static_size=static_size,
            dropout=dropout,
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.attention = InterpretableMultiHeadAttention(
            d_model=hidden_size,
            n_heads=attention_heads,
            dropout=dropout,
        )

        self.gate_grn = GatedResidualNetwork(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: "torch.Tensor", city_id: "torch.Tensor") -> "torch.Tensor":
        # x: (B, T, F), city_id: (B,) long
        B = x.size(0)
        static = self.static_embed(city_id.clamp(0, self._n_cities - 1))  # (B, hidden)
        selected = self.variable_selection(x, static)  # (B, T, hidden)
        lstm_out, _ = self.lstm(selected)  # (B, T, hidden)
        attn_out = self.attention(lstm_out)  # (B, T, hidden)
        combined = torch.cat([lstm_out, attn_out], dim=-1)  # (B, T, 2*hidden)
        gated = self.gate_grn(combined)  # (B, T, hidden)
        pooled = gated.mean(dim=1)  # (B, hidden)
        out = self.fc(self.dropout(pooled)).squeeze(-1)  # (B,)
        return out


def build_tft_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    city_id_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    city_id_valid: np.ndarray,
    batch_size: int,
    device: "torch.device",
) -> tuple[DataLoader, DataLoader | None]:
    X_tr = torch.from_numpy(X_train)
    y_tr = torch.from_numpy(y_train).unsqueeze(1)
    c_tr = torch.from_numpy(city_id_train).long()
    train_ds = TensorDataset(X_tr, c_tr, y_tr)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = None
    if len(X_valid) > 0:
        X_va = torch.from_numpy(X_valid)
        y_va = torch.from_numpy(y_valid).unsqueeze(1)
        c_va = torch.from_numpy(city_id_valid).long()
        valid_ds = TensorDataset(X_va, c_va, y_va)
        valid_loader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
    return train_loader, valid_loader


def train_tft(
    X_train: np.ndarray,
    y_train: np.ndarray,
    city_id_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    city_id_valid: np.ndarray,
    feature_cols: list[str],
    city_categories: list[str],
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
    n_cities = max(1, len(city_categories))
    model = TFTRegressor(
        input_size=n_features,
        seq_len=seq_len,
        n_cities=n_cities,
        hidden_size=getattr(args, "hidden_size", 64),
        lstm_layers=getattr(args, "lstm_layers", 2),
        attention_heads=getattr(args, "attention_heads", 4),
        dropout=getattr(args, "dropout", 0.2),
        city_embed_dim=getattr(args, "city_embed_dim", 16),
        seed=args.seed,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=float(getattr(args, "lr_scheduler_factor", 0.5)),
        patience=int(getattr(args, "lr_scheduler_patience", 5)),
        min_lr=float(getattr(args, "min_lr", 1e-6)),
    )

    train_loader, valid_loader = build_tft_dataloaders(
        X_train, y_train, city_id_train,
        X_valid, y_valid, city_id_valid,
        args.batch_size, device,
    )
    best_rmse = float("inf")
    patience = getattr(args, "early_stopping_patience", 15)
    epochs_no_improve = 0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, c_b, y_b in train_loader:
            X_b = X_b.to(device)
            c_b = c_b.to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b, c_b)
            loss = criterion(out, y_b.squeeze(1))
            loss.backward()
            grad_clip_norm = float(getattr(args, "grad_clip_norm", 0.0))
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            train_loss += loss.item() * X_b.size(0)
        train_loss /= len(X_train)
        train_rmse = float(np.sqrt(train_loss))

        if valid_loader is not None:
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for X_b, c_b, y_b in valid_loader:
                    X_b, c_b, y_b = X_b.to(device), c_b.to(device), y_b.to(device)
                    out = model(X_b, c_b)
                    valid_loss += criterion(out, y_b.squeeze(1)).item() * X_b.size(0)
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
            diag_interval = int(getattr(args, "diagnostic_interval", 0))
            if diag_interval > 0 and epoch % diag_interval == 0:
                n_diag = max(1, int(getattr(args, "diagnostic_samples", 512)))
                sample_X = X_valid[: min(n_diag, len(X_valid))]
                sample_c = city_id_valid[: min(n_diag, len(city_id_valid))]
                if len(sample_X) > 0:
                    with torch.no_grad():
                        pred_diag = model(
                            torch.from_numpy(sample_X).to(device),
                            torch.from_numpy(sample_c).long().to(device),
                        ).cpu().numpy().astype(np.float32)
                    print(
                        "[DIAG] "
                        f"epoch={epoch} {describe_vector('pred_scaled', pred_diag)} "
                        f"train_rmse={train_rmse:.4f} valid_rmse={valid_rmse:.4f} "
                        f"lr={float(optimizer.param_groups[0]['lr']):.6g}"
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
            diag_interval = int(getattr(args, "diagnostic_interval", 0))
            if diag_interval > 0 and epoch % diag_interval == 0:
                n_diag = max(1, int(getattr(args, "diagnostic_samples", 512)))
                sample_X = X_train[: min(n_diag, len(X_train))]
                sample_c = city_id_train[: min(n_diag, len(city_id_train))]
                if len(sample_X) > 0:
                    with torch.no_grad():
                        pred_diag = model(
                            torch.from_numpy(sample_X).to(device),
                            torch.from_numpy(sample_c).long().to(device),
                        ).cpu().numpy().astype(np.float32)
                    print(
                        "[DIAG] "
                        f"epoch={epoch} {describe_vector('pred_scaled', pred_diag)} "
                        f"train_rmse={train_rmse:.4f} lr={float(optimizer.param_groups[0]['lr']):.6g}"
                    )

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_checkpoint:
        torch.save(model.state_dict(), output_dir / "tft_best.pt")
    return model, best_rmse


def predict_numpy_tft(
    model: "nn.Module",
    X: np.ndarray,
    city_id: np.ndarray,
    device: "torch.device",
    batch_size: int = 4096,
) -> np.ndarray:
    if len(X) == 0:
        return np.array([], dtype=np.float32)
    model.eval()
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_b = torch.from_numpy(X[i : i + batch_size]).to(device)
            c_b = torch.from_numpy(city_id[i : i + batch_size]).long().to(device)
            out = model(X_b, c_b)
            preds.append(out.cpu())
    return torch.cat(preds, dim=0).numpy().astype(np.float32)


def _city_to_id(meta: pd.DataFrame, city_categories: list[str]) -> np.ndarray:
    city_to_idx = {c: i for i, c in enumerate(city_categories)}
    ids = meta["city"].astype(str).map(lambda c: city_to_idx.get(c, 0)).values
    return np.asarray(ids, dtype=np.int64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Daily PM2.5 prediction for BTH/YRD/PRD using Temporal Fusion Transformer (TFT)."
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
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for scheduler.")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5, help="ReduceLROnPlateau factor.")
    parser.add_argument("--lr-scheduler-patience", type=int, default=4, help="ReduceLROnPlateau patience.")
    parser.add_argument("--grad-clip-norm", type=float, default=5.0, help="Max norm for gradient clipping; <=0 disables.")
    parser.add_argument("--min-std", type=float, default=1e-6, help="Lower bound for standard deviation in scaling.")
    parser.add_argument("--diagnostic-interval", type=int, default=10, help="Epoch interval to print diagnostics; <=0 disables.")
    parser.add_argument("--diagnostic-samples", type=int, default=512, help="Max samples for per-epoch diagnostic stats.")
    parser.add_argument("--debug-overfit-n", type=int, default=0, help="If >0, run sanity overfit check on first N train samples.")
    parser.add_argument("--debug-overfit-epochs", type=int, default=200, help="Epochs used by overfit sanity check.")
    parser.add_argument("--debug-overfit-batch-size", type=int, default=64, help="Batch size for overfit sanity check.")
    # TFT-specific
    parser.add_argument("--hidden-size", type=int, default=64, help="TFT hidden size (LSTM, attention, GRN).")
    parser.add_argument("--lstm-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--attention-heads", type=int, default=4, help="Multi-head attention heads.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for Adam.")
    parser.add_argument("--city-embed-dim", type=int, default=16, help="City embedding dimension before projection.")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience (epochs).")
    parser.add_argument("--device", type=str, default="", help="Device: 'cuda', 'gpu', or 'cpu'. Default: auto-detect.")
    parser.add_argument("--pm25-workers", type=int, default=64, help="Worker count for PM2.5 NC reading.")
    parser.add_argument("--era5-workers", type=int, default=64, help="Worker count for ERA5 NC reading.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "cache"),
        help="Cache directory for aggregated daily PM2.5/ERA5 parquet.",
    )
    parser.add_argument("--disable-data-cache", action="store_true", help="Disable NC aggregation cache.")
    parser.add_argument("--pm25-precropped", dest="pm25_precropped", action="store_true", default=True, help="PM2.5 NC 已按城市群预裁剪（默认）.")
    parser.add_argument("--no-pm25-precropped", dest="pm25_precropped", action="store_false", help="PM2.5 为全局数据，读取时做空间裁剪.")
    parser.add_argument("--params-json", type=str, default=None, help="Load hyperparameters from JSON for full training.")
    return parser


def main() -> int:
    if torch is None:
        raise ImportError("PyTorch is not installed. Please run: pip install torch")
    args = build_parser().parse_args()
    params_json = getattr(args, "params_json", None)
    if params_json:
        path = Path(params_json).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Params JSON not found: {path}")
        with open(path, encoding="utf-8") as f:
            params = json.load(f)
        for key, value in params.items():
            if key == "best_value":
                continue
            if hasattr(args, key):
                setattr(args, key, value)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    correlation_dir = Path(args.correlation_dir).expanduser().resolve()
    data_read_dir = Path(args.data_read_dir).expanduser().resolve()
    pm25_day_dir = Path(args.pm25_day_dir).expanduser().resolve()
    era5_day_dir = Path(args.era5_day_dir).expanduser().resolve()
    city_geojson_dir = Path(args.city_geojson_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_pbar = tqdm(total=6, desc="TFT 总进度", dynamic_ncols=True) if tqdm is not None else None

    device = get_device(args)
    if device.type == "cuda":
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Using CPU (CUDA not available).")

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
            module_tag="pytorch",
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
    export_feature_quality_report(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        output_dir=output_dir,
        year_factor_cols=year_factor_cols,
        met_cols=met_cols,
    )

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
        print(f"[INFO] Cluster={cluster_name} {describe_vector('y_train_raw', y_train)}")
        print(f"[INFO] Cluster={cluster_name} {describe_vector('y_valid_raw', y_valid)}")
        x_mean, x_std, y_mean, y_std = fit_train_standardizers(
            X_train=X_train,
            y_train=y_train,
            min_std=float(args.min_std),
        )
        X_train_scaled = transform_inputs(X_train, x_mean, x_std)
        X_valid_scaled = transform_inputs(X_valid, x_mean, x_std)
        X_test_scaled = transform_inputs(X_test, x_mean, x_std)
        y_train_scaled = transform_targets(y_train, y_mean, y_std)
        y_valid_scaled = transform_targets(y_valid, y_mean, y_std)
        city_id_train = _city_to_id(meta_train, city_categories)
        city_id_valid = _city_to_id(meta_valid, city_categories)
        city_id_test = _city_to_id(meta_test, city_categories)
        cluster_output_dir = output_dir / cluster_name.lower()
        cluster_output_dir.mkdir(parents=True, exist_ok=True)
        if int(getattr(args, "debug_overfit_n", 0)) > 0:
            debug_n = min(int(args.debug_overfit_n), len(X_train_scaled))
            if debug_n > 1:
                print(f"[DEBUG] Running overfit sanity check on first {debug_n} samples for cluster={cluster_name}.")
                debug_args = argparse.Namespace(**vars(args))
                debug_args.epochs = int(getattr(args, "debug_overfit_epochs", 200))
                debug_args.batch_size = max(2, min(debug_n, int(getattr(args, "debug_overfit_batch_size", 64))))
                debug_args.early_stopping_patience = max(10, int(debug_args.epochs // 4))
                debug_model, _ = train_tft(
                    X_train=X_train_scaled[:debug_n],
                    y_train=y_train_scaled[:debug_n],
                    city_id_train=city_id_train[:debug_n],
                    X_valid=X_train_scaled[:debug_n],
                    y_valid=y_train_scaled[:debug_n],
                    city_id_valid=city_id_train[:debug_n],
                    feature_cols=feature_cols,
                    city_categories=city_categories,
                    args=debug_args,
                    device=device,
                    output_dir=cluster_output_dir,
                    save_checkpoint=False,
                )
                debug_pred_scaled = predict_numpy_tft(
                    debug_model,
                    X_train_scaled[:debug_n],
                    city_id_train[:debug_n],
                    device,
                )
                debug_pred = inverse_transform_targets(debug_pred_scaled, y_mean, y_std)
                debug_metrics = compute_metrics(pd.Series(y_train[:debug_n]), debug_pred)
                debug_target_std = float(np.std(y_train[:debug_n]))
                print(
                    "[DEBUG] overfit_check "
                    f"cluster={cluster_name} rmse={float(debug_metrics['rmse']):.4f} "
                    f"target_std={debug_target_std:.4f}"
                )
                if np.isfinite(debug_target_std) and debug_target_std > 0 and float(debug_metrics["rmse"]) >= 0.8 * debug_target_std:
                    print(
                        "[WARN] Overfit sanity check did not converge well. "
                        "Please verify feature-label alignment and sequence construction."
                    )

        train_begin = time.perf_counter()
        model, _ = train_tft(
            X_train_scaled,
            y_train_scaled,
            city_id_train,
            X_valid_scaled,
            y_valid_scaled,
            city_id_valid,
            feature_cols,
            city_categories,
            args,
            device,
            cluster_output_dir,
        )
        train_seconds = time.perf_counter() - train_begin
        train_seconds_total += train_seconds
        pred_train_scaled = predict_numpy_tft(model, X_train_scaled, city_id_train, device)
        pred_valid_scaled = predict_numpy_tft(model, X_valid_scaled, city_id_valid, device)
        pred_test_scaled = predict_numpy_tft(model, X_test_scaled, city_id_test, device)
        pred_train = inverse_transform_targets(pred_train_scaled, y_mean, y_std)
        pred_valid = inverse_transform_targets(pred_valid_scaled, y_mean, y_std)
        pred_test = inverse_transform_targets(pred_test_scaled, y_mean, y_std)
        metric_rows = [
            {"split": "train", **compute_metrics(pd.Series(y_train), pred_train), "n_samples": int(len(y_train))},
            {"split": "valid", **compute_metrics(pd.Series(y_valid), pred_valid), "n_samples": int(len(y_valid))},
            {"split": "test", **compute_metrics(pd.Series(y_test), pred_test), "n_samples": int(len(y_test))},
        ]
        baseline_value = float(np.mean(y_train))
        baseline_valid_pred = np.full(shape=y_valid.shape, fill_value=baseline_value, dtype=np.float32)
        baseline_test_pred = np.full(shape=y_test.shape, fill_value=baseline_value, dtype=np.float32)
        baseline_valid_metrics = compute_metrics(pd.Series(y_valid), baseline_valid_pred)
        baseline_test_metrics = compute_metrics(pd.Series(y_test), baseline_test_pred)
        model_valid_metrics = compute_metrics(pd.Series(y_valid), pred_valid)
        model_test_metrics = compute_metrics(pd.Series(y_test), pred_test)
        print(
            "[DIAG] "
            f"cluster={cluster_name} mean_baseline={baseline_value:.4f} "
            f"valid_rmse(model/baseline)={float(model_valid_metrics['rmse']):.4f}/{float(baseline_valid_metrics['rmse']):.4f} "
            f"test_rmse(model/baseline)={float(model_test_metrics['rmse']):.4f}/{float(baseline_test_metrics['rmse']):.4f}"
        )
        print(f"[DIAG] Cluster={cluster_name} {describe_vector('pred_test_raw', pred_test)}")
        metrics_df = pd.DataFrame(metric_rows)
        all_pred_df, test_pred_df = build_prediction_frames(
            train_df=meta_train,
            valid_df=meta_valid,
            test_df=meta_test,
            pred_train=pred_train,
            pred_valid=pred_valid,
            pred_test=pred_test,
        )
        cluster_results[cluster_name] = {
            "feature_cols": feature_cols,
            "city_categories": city_categories,
            "all_pred_df": all_pred_df,
            "test_pred_df": test_pred_df,
            "metrics_df": metrics_df,
            "train_rows": int(len(X_train)),
            "valid_rows": int(len(X_valid)),
            "test_rows": int(len(X_test)),
            "train_seconds": float(train_seconds),
            "target_mean_train": float(y_mean),
            "target_std_train": float(y_std),
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
    for cluster_name in target_cluster_order:
        if cluster_name not in cluster_results:
            continue
        cluster_output_dir = output_dir / cluster_name.lower()
        result = cluster_results[cluster_name]
        all_pred_df = result["all_pred_df"]
        test_pred_df = result["test_pred_df"]
        metrics_df = result["metrics_df"]
        cluster_metrics_df = metrics_by_cluster(test_pred_df)
        generalization_df = export_generalization_artifacts(metrics_df, cluster_output_dir)
        export_regression_artifacts(all_pred_df=all_pred_df, output_dir=cluster_output_dir, model_name=f"TFT-{cluster_name}")
        metrics_df.to_csv(cluster_output_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")
        cluster_metrics_df.to_csv(cluster_output_dir / "metrics_by_cluster_test.csv", index=False, encoding="utf-8-sig")
        test_pred_df.sort_values(["date", "city"], kind="mergesort").to_csv(
            cluster_output_dir / "predictions_test.csv", index=False, encoding="utf-8-sig"
        )
        all_pred_df.sort_values(["split", "date", "city"], kind="mergesort").to_csv(
            cluster_output_dir / "predictions_all_splits.csv", index=False, encoding="utf-8-sig"
        )
        metrics_by_cluster_frames.append(metrics_df.assign(cluster=cluster_name))
        all_pred_frames.append(all_pred_df.assign(model_cluster=cluster_name))
        test_pred_frames.append(test_pred_df.assign(model_cluster=cluster_name))
        per_cluster_run_info[cluster_name] = {
            "output_dir": str(cluster_output_dir),
            "n_features": int(len(result["feature_cols"])),
            "n_cities": int(len(result["city_categories"])),
            "train_rows": int(result["train_rows"]),
            "valid_rows": int(result["valid_rows"]),
            "test_rows": int(result["test_rows"]),
            "train_seconds": float(result["train_seconds"]),
            "generalization_level": str(generalization_df.loc[0, "generalization_level"]) if not generalization_df.empty else "",
        }

    all_pred_df = pd.concat(all_pred_frames, ignore_index=True) if all_pred_frames else pd.DataFrame()
    test_pred_df = pd.concat(test_pred_frames, ignore_index=True) if test_pred_frames else pd.DataFrame()
    metrics_overall_by_cluster_df = (
        pd.concat(metrics_by_cluster_frames, ignore_index=True) if metrics_by_cluster_frames else pd.DataFrame()
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
    export_regression_artifacts(all_pred_df=all_pred_df, output_dir=output_dir, model_name="TFT-ClusterModels")
    if stage_pbar is not None:
        stage_pbar.update(1)
        stage_pbar.set_postfix_str("评估与图表完成")

    metrics_df.to_csv(output_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(output_dir / "metrics_overall_pooled_from_cluster_models.csv", index=False, encoding="utf-8-sig")
    metrics_overall_by_cluster_df.to_csv(output_dir / "metrics_overall_by_cluster.csv", index=False, encoding="utf-8-sig")
    cluster_metrics_df.to_csv(output_dir / "metrics_by_cluster_test.csv", index=False, encoding="utf-8-sig")
    test_pred_df.sort_values(["date", "city"], kind="mergesort").to_csv(
        output_dir / "predictions_test.csv", index=False, encoding="utf-8-sig"
    )
    all_pred_df.sort_values(["split", "date", "city"], kind="mergesort").to_csv(
        output_dir / "predictions_all_splits.csv", index=False, encoding="utf-8-sig"
    )
    all_pred_df.sort_values(["split", "date", "city"], kind="mergesort").to_csv(
        output_dir / "predictions_all_splits_with_cluster.csv", index=False, encoding="utf-8-sig"
    )
    test_pred_df.sort_values(["date", "city"], kind="mergesort").to_csv(
        output_dir / "predictions_test_with_cluster.csv", index=False, encoding="utf-8-sig"
    )

    total_train_rows = int(sum(info["train_rows"] for info in per_cluster_run_info.values()))
    total_valid_rows = int(sum(info["valid_rows"] for info in per_cluster_run_info.values()))
    total_test_rows = int(sum(info["test_rows"] for info in per_cluster_run_info.values()))
    n_features_by_cluster = {cluster: int(info["n_features"]) for cluster, info in per_cluster_run_info.items()}
    run_info = {
        "model": "TFT",
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
        "train_seq_rows": total_train_rows,
        "valid_seq_rows": total_valid_rows,
        "test_seq_rows": total_test_rows,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "min_lr": args.min_lr,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "grad_clip_norm": args.grad_clip_norm,
        "min_std": args.min_std,
        "hidden_size": args.hidden_size,
        "lstm_layers": args.lstm_layers,
        "attention_heads": args.attention_heads,
        "dropout": args.dropout,
        "city_embed_dim": args.city_embed_dim,
        "early_stopping_patience": args.early_stopping_patience,
        "diagnostic_interval": int(args.diagnostic_interval),
        "diagnostic_samples": int(args.diagnostic_samples),
        "debug_overfit_n": int(args.debug_overfit_n),
        "debug_overfit_epochs": int(args.debug_overfit_epochs),
        "debug_overfit_batch_size": int(args.debug_overfit_batch_size),
        "daily_input": args.daily_input or [],
        "pm25_day_dir": str(pm25_day_dir),
        "era5_day_dir": str(era5_day_dir),
        "generalization_level": str(pooled_generalization_df.loc[0, "generalization_level"]) if not pooled_generalization_df.empty else "",
        "pm25_workers": int(args.pm25_workers),
        "era5_workers": int(args.era5_workers),
        "cache_dir": str(cache_dir),
        "cache_enabled": bool(cache_enabled),
        "training_mode": training_mode,
        "fallback_years": fallback_years,
        "cache_hit_pm25": bool(prepare_stats_main.get("cache_hit_pm25", False)),
        "cache_hit_era5": bool(prepare_stats_main.get("cache_hit_era5", False)),
        "data_prepare_seconds": float(prepare_stats_main.get("data_prepare_seconds", 0.0)),
        "pm25_read_seconds": float(prepare_stats_main.get("pm25_seconds", 0.0)),
        "era5_read_seconds": float(prepare_stats_main.get("era5_seconds", 0.0)),
        "year_factor_seconds": float(prepare_stats_main.get("year_factor_seconds", 0.0)),
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
    print("[INFO] TFT daily PM2.5 training finished.")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Trained clusters: {', '.join(per_cluster_run_info.keys())}")
    for cluster_name in per_cluster_run_info:
        print(f"       - {cluster_name}: {output_dir / cluster_name.lower()}")
    print("[INFO] Files:")
    print("       - metrics_overall.csv")
    print("       - metrics_overall_pooled_from_cluster_models.csv")
    print("       - metrics_overall_by_cluster.csv")
    print("       - metrics_by_cluster_test.csv")
    print("       - predictions_test.csv")
    print("       - predictions_test_with_cluster.csv")
    print("       - predictions_all_splits.csv")
    print("       - predictions_all_splits_with_cluster.csv")
    print("       - run_info.json")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
