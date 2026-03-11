"""
Daily PM2.5 prediction for BTH/YRD/PRD using CNN-LSTM with GPU acceleration.
Reuses daily_ml_pipeline for data preparation and sequence building; input is (seq_len, n_features).
"""
from __future__ import annotations

import argparse
import json
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


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "cnn_lstm_daily_pm25"

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


class CNNLSTMRegressor(nn.Module):
    """2D CNN on (time, feature) grid then LSTM for sequence regression. Input (B, T, F)."""

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        cnn_channels: int = 64,
        cnn_kernel_time: int = 3,
        cnn_kernel_feature: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self._input_size = input_size
        self._seq_len = seq_len
        time_padding = cnn_kernel_time // 2
        feature_padding = cnn_kernel_feature // 2
        self.cnn = nn.Conv2d(
            in_channels=1,
            out_channels=cnn_channels,
            kernel_size=(cnn_kernel_time, cnn_kernel_feature),
            stride=(1, 1),
            padding=(time_padding, feature_padding),
        )
        self.lstm = nn.LSTM(
            cnn_channels,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(lstm_hidden, 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, T, F) -> (B, 1, T, F)
        x = x.unsqueeze(1)
        x = torch.relu(self.cnn(x))
        # (B, C, T, F) -> compress feature axis -> (B, C, T)
        x = x.mean(dim=3)
        # (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


def build_sequence_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    batch_size: int,
    device: "torch.device",
) -> tuple[DataLoader, DataLoader | None]:
    X_tr = torch.from_numpy(X_train)
    y_tr = torch.from_numpy(y_train).unsqueeze(1)
    train_ds = TensorDataset(X_tr, y_tr)
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
        valid_ds = TensorDataset(X_va, y_va)
        valid_loader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
    return train_loader, valid_loader


def train_cnn_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_cols: list[str],
    args: argparse.Namespace,
    device: "torch.device",
    output_dir: Path,
) -> "nn.Module":
    if torch is None or nn is None:
        raise ImportError("PyTorch is not installed. Please run: pip install torch")

    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    model = CNNLSTMRegressor(
        input_size=n_features,
        seq_len=seq_len,
        cnn_channels=getattr(args, "cnn_channels", 64),
        cnn_kernel_time=getattr(args, "cnn_kernel_time", 3),
        cnn_kernel_feature=getattr(args, "cnn_kernel_feature", 3),
        lstm_hidden=getattr(args, "lstm_hidden", 128),
        lstm_layers=getattr(args, "lstm_layers", 2),
        dropout=getattr(args, "dropout", 0.2),
        seed=args.seed,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader, valid_loader = build_sequence_dataloaders(
        X_train, y_train, X_valid, y_valid, args.batch_size, device
    )
    best_rmse = float("inf")
    patience = getattr(args, "early_stopping_patience", 15)
    epochs_no_improve = 0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b.squeeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_b.size(0)
        train_loss /= len(X_train)

        if valid_loader is not None:
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in valid_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    out = model(X_b)
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
    torch.save(model.state_dict(), output_dir / "cnn_lstm_best.pt")
    return model


def predict_numpy_seq(
    model: "nn.Module",
    X: np.ndarray,
    device: "torch.device",
    batch_size: int = 4096,
) -> np.ndarray:
    if len(X) == 0:
        return np.array([], dtype=np.float32)
    model.eval()
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i : i + batch_size]).to(device)
            out = model(batch)
            preds.append(out.cpu())
    return torch.cat(preds, dim=0).numpy().astype(np.float32)


def compute_feature_importance_last_step(
    model: "nn.Module",
    X: np.ndarray,
    feature_cols: list[str],
    device: "torch.device",
    n_samples: int = 2000,
) -> np.ndarray:
    """Feature importance via mean absolute (input * gradient) on last time step."""
    if len(X) == 0 or len(feature_cols) == 0:
        return np.zeros(len(feature_cols), dtype=np.float64)
    model.eval()
    n = min(n_samples, len(X))
    idx = np.random.RandomState(42).choice(len(X), size=n, replace=False)
    sample = X[idx]
    X_t = torch.from_numpy(sample).to(device)
    X_t.requires_grad_(True)
    out = model(X_t)
    out.sum().backward()
    grad = X_t.grad
    if grad is None:
        return np.zeros(len(feature_cols), dtype=np.float64)
    if grad.ndim != 3 or grad.shape[2] != len(feature_cols):
        raise ValueError(
            f"Unexpected gradient shape {tuple(grad.shape)} for {len(feature_cols)} features."
        )
    last_step = grad[:, -1, :]
    imp = (X_t[:, -1, :].detach().abs() * last_step.abs()).mean(dim=0).cpu().numpy()
    return np.asarray(imp, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Daily PM2.5 prediction for BTH/YRD/PRD using CNN-LSTM (GPU)."
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
    parser.add_argument("--cnn-channels", type=int, default=64, help="CNN 2D output channels.")
    parser.add_argument("--cnn-kernel-time", type=int, default=3, help="CNN kernel size on time axis.")
    parser.add_argument("--cnn-kernel-feature", type=int, default=3, help="CNN kernel size on feature axis.")
    parser.add_argument("--lstm-hidden", type=int, default=128, help="LSTM hidden size.")
    parser.add_argument("--lstm-layers", type=int, default=2, help="LSTM num layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
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

    model = train_cnn_lstm(
        X_train, y_train, X_valid, y_valid, feature_cols, args, device, output_dir
    )

    pred_train = predict_numpy_seq(model, X_train, device)
    pred_valid = predict_numpy_seq(model, X_valid, device)
    pred_test = predict_numpy_seq(model, X_test, device)

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
    export_regression_artifacts(all_pred_df=all_pred_df, output_dir=output_dir, model_name="CNN_LSTM")

    importance = compute_feature_importance_last_step(
        model, X_valid if len(X_valid) > 0 else X_train, feature_cols, device
    )
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importance}
    ).sort_values("importance", ascending=False, kind="mergesort")

    metrics_df.to_csv(output_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")
    cluster_metrics_df.to_csv(output_dir / "metrics_by_cluster_test.csv", index=False, encoding="utf-8-sig")
    test_pred_df.sort_values(["date", "city"], kind="mergesort").to_csv(
        output_dir / "predictions_test.csv", index=False, encoding="utf-8-sig"
    )
    all_pred_df.sort_values(["split", "date", "city"], kind="mergesort").to_csv(
        output_dir / "predictions_all_splits.csv", index=False, encoding="utf-8-sig"
    )
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    run_info = {
        "model": "CNN_LSTM",
        "device": str(device),
        "train_end_year": args.train_end_year,
        "valid_year": args.valid_year,
        "test_year": args.test_year,
        "seq_len": args.seq_len,
        "n_features": len(feature_cols),
        "n_year_factor_features": len(year_factor_cols),
        "n_era5_daily_features": len(met_cols),
        "train_seq_rows": int(len(X_train)),
        "valid_seq_rows": int(len(X_valid)),
        "test_seq_rows": int(len(X_test)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "cnn_channels": args.cnn_channels,
        "cnn_kernel_time": args.cnn_kernel_time,
        "cnn_kernel_feature": args.cnn_kernel_feature,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "dropout": args.dropout,
        "early_stopping_patience": args.early_stopping_patience,
        "daily_input": args.daily_input or [],
        "pm25_day_dir": str(pm25_day_dir),
        "era5_day_dir": str(era5_day_dir),
        "generalization_level": str(generalization_df.loc[0, "generalization_level"]) if not generalization_df.empty else "",
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print("=" * 90)
    print("[INFO] CNN-LSTM daily PM2.5 training finished.")
    print(f"[INFO] Output directory: {output_dir}")
    print("[INFO] Files: metrics_overall.csv, metrics_by_cluster_test.csv, predictions_test.csv,")
    print("       predictions_all_splits.csv, feature_importance.csv, cnn_lstm_best.pt, run_info.json")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
