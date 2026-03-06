"""
Daily PM2.5 prediction for BTH/YRD/PRD using Spatio-Temporal Transformer (time + city embedding).
Reuses daily_ml_pipeline for data preparation and sequence building; input is (seq_len, n_features).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_ml_pipeline import (
    DEFAULT_CITY_GEOJSON_DIR,
    DEFAULT_CORRELATION_DIR,
    DEFAULT_DATA_READ_DIR,
    DEFAULT_ERA5_DAY_DIR,
    DEFAULT_PM25_DAY_DIR,
    SCRIPT_DIR,
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

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "st_transformer_daily_pm25"

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


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class STTransformerRegressor(nn.Module):
    """Temporal Transformer + city embedding for sequence regression. Input (B, T, F)."""

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        n_cities: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        city_embed_dim: int = 16,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self._input_size = input_size
        self._seq_len = seq_len
        self.d_model = d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=seq_len + 10, dropout=0.0)
        self.city_embed = nn.Embedding(max(1, n_cities), city_embed_dim)
        self.embed_proj = nn.Linear(d_model + city_embed_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: "torch.Tensor", city_id: "torch.Tensor") -> "torch.Tensor":
        # x: (B, T, F), city_id: (B,) long
        B, T, _ = x.shape
        x = self.input_proj(x)
        x = self.pos_enc(x)
        city_emb = self.city_embed(city_id.clamp(0, self.city_embed.num_embeddings - 1))
        city_emb = city_emb.unsqueeze(1).expand(-1, T, -1)
        x = self.embed_proj(torch.cat([x, city_emb], dim=-1))
        x = self.transformer(x)
        out = self.dropout(x.mean(dim=1))
        return self.fc(out).squeeze(-1)


def build_st_dataloaders(
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


def train_st_transformer(
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
) -> "nn.Module":
    if torch is None or nn is None:
        raise ImportError("PyTorch is not installed. Please run: pip install torch")

    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    n_cities = max(1, len(city_categories))
    model = STTransformerRegressor(
        input_size=n_features,
        seq_len=seq_len,
        n_cities=n_cities,
        d_model=getattr(args, "d_model", 64),
        nhead=getattr(args, "nhead", 4),
        num_encoder_layers=getattr(args, "num_encoder_layers", 3),
        dim_feedforward=getattr(args, "dim_feedforward", 256),
        dropout=getattr(args, "dropout", 0.2),
        city_embed_dim=getattr(args, "city_embed_dim", 16),
        seed=args.seed,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader, valid_loader = build_st_dataloaders(
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
            optimizer.step()
            train_loss += loss.item() * X_b.size(0)
        train_loss /= len(X_train)

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
            if valid_rmse < best_rmse:
                best_rmse = valid_rmse
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if best_state is not None:
                    model.load_state_dict(best_state)
                break
        else:
            if train_loss < best_rmse:
                best_rmse = train_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "st_transformer_best.pt")
    return model


def predict_numpy_st(
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
        description="Daily PM2.5 prediction for BTH/YRD/PRD using ST-Transformer (time + city embedding)."
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
    parser.add_argument("--d-model", type=int, default=64, help="Transformer d_model.")
    parser.add_argument("--nhead", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--num-encoder-layers", type=int, default=3, help="Transformer encoder layers.")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="Transformer feedforward dim.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--city-embed-dim", type=int, default=16, help="City embedding dimension.")
    parser.add_argument("--early-stopping-patience", type=int, default=15, help="Early stopping patience (epochs).")
    parser.add_argument("--device", type=str, default="", help="Device: 'cuda', 'gpu', or 'cpu'. Default: auto-detect.")
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

    device = get_device(args)
    if device.type == "cuda":
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Using CPU (CUDA not available).")

    base_df, year_factor_cols, met_cols = prepare_training_table(
        module_tag="pytorch",
        correlation_dir=correlation_dir,
        data_read_dir=data_read_dir,
        city_geojson_dir=city_geojson_dir,
        daily_input=args.daily_input,
        pm25_day_dir=pm25_day_dir,
        era5_day_dir=era5_day_dir,
        include_era5_daily=not args.skip_era5,
    )
    feature_df = build_daily_features(base_df)
    train_df, valid_df, test_df = split_by_time(
        feature_df,
        train_end_year=args.train_end_year,
        valid_year=args.valid_year,
        test_year=args.test_year,
    )
    (
        X_train, y_train, meta_train,
        X_valid, y_valid, meta_valid,
        X_test, y_test, meta_test,
        feature_cols,
        city_categories,
    ) = build_sequence_matrices(train_df, valid_df, test_df, seq_len=args.seq_len)

    if len(X_train) == 0:
        raise ValueError("No training sequences; try smaller --seq-len or check data.")

    city_id_train = _city_to_id(meta_train, city_categories)
    city_id_valid = _city_to_id(meta_valid, city_categories)
    city_id_test = _city_to_id(meta_test, city_categories)

    model = train_st_transformer(
        X_train, y_train, city_id_train,
        X_valid, y_valid, city_id_valid,
        feature_cols, city_categories, args, device, output_dir,
    )

    pred_train = predict_numpy_st(model, X_train, city_id_train, device)
    pred_valid = predict_numpy_st(model, X_valid, city_id_valid, device)
    pred_test = predict_numpy_st(model, X_test, city_id_test, device)

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
    cluster_metrics_df = metrics_by_cluster(test_pred_df)
    generalization_df = export_generalization_artifacts(metrics_df, output_dir)
    export_regression_artifacts(all_pred_df=all_pred_df, output_dir=output_dir, model_name="ST_Transformer")

    metrics_df.to_csv(output_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")
    cluster_metrics_df.to_csv(output_dir / "metrics_by_cluster_test.csv", index=False, encoding="utf-8-sig")
    test_pred_df.sort_values(["date", "city"], kind="mergesort").to_csv(
        output_dir / "predictions_test.csv", index=False, encoding="utf-8-sig"
    )
    all_pred_df.sort_values(["split", "date", "city"], kind="mergesort").to_csv(
        output_dir / "predictions_all_splits.csv", index=False, encoding="utf-8-sig"
    )

    run_info = {
        "model": "ST_Transformer",
        "device": str(device),
        "train_end_year": args.train_end_year,
        "valid_year": args.valid_year,
        "test_year": args.test_year,
        "seq_len": args.seq_len,
        "n_features": len(feature_cols),
        "n_cities": len(city_categories),
        "n_year_factor_features": len(year_factor_cols),
        "n_era5_daily_features": len(met_cols),
        "train_seq_rows": int(len(X_train)),
        "valid_seq_rows": int(len(X_valid)),
        "test_seq_rows": int(len(X_test)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_encoder_layers": args.num_encoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "city_embed_dim": args.city_embed_dim,
        "early_stopping_patience": args.early_stopping_patience,
        "daily_input": args.daily_input or [],
        "pm25_day_dir": str(pm25_day_dir),
        "era5_day_dir": str(era5_day_dir),
        "generalization_level": str(generalization_df.loc[0, "generalization_level"]) if not generalization_df.empty else "",
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print("=" * 90)
    print("[INFO] ST-Transformer daily PM2.5 training finished.")
    print(f"[INFO] Output directory: {output_dir}")
    print("[INFO] Files: metrics_overall.csv, metrics_by_cluster_test.csv, predictions_test.csv,")
    print("       predictions_all_splits.csv, st_transformer_best.pt, run_info.json")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
