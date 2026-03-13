"""
Daily PM2.5 prediction for BTH/YRD/PRD using a classic BP (fully connected) neural network with GPU acceleration.
Reuses daily_ml_pipeline for data preparation and evaluation; training/inference on GPU.
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
    build_model_matrices,
    build_prediction_frames,
    compute_metrics,
    export_generalization_artifacts,
    export_regression_artifacts,
    export_shap_artifacts,
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


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "bpnn_daily_pm25"

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


class BPNNRegressor(nn.Module):
    """Classic BP-style MLP for regression: input_dim -> hidden_layers -> 1."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        dropout: float = 0.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._input_dim = input_dim

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x).squeeze(-1)

    @property
    def input_dim(self) -> int:
        return self._input_dim


def build_dataloaders(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    batch_size: int,
    device: "torch.device",
) -> tuple["DataLoader", "DataLoader | None"]:
    X_tr = torch.from_numpy(x_train.values.astype(np.float32))
    y_tr = torch.from_numpy(y_train.values.astype(np.float32)).unsqueeze(1)
    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = None
    if not x_valid.empty and len(x_valid) > 0:
        X_va = torch.from_numpy(x_valid.values.astype(np.float32))
        y_va = torch.from_numpy(y_valid.values.astype(np.float32)).unsqueeze(1)
        valid_ds = TensorDataset(X_va, y_va)
        valid_loader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
    return train_loader, valid_loader


def train_bpnn(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    feature_cols: list[str],
    args: argparse.Namespace,
    device: "torch.device",
    output_dir: Path,
) -> "nn.Module":
    if torch is None or nn is None:
        raise ImportError("PyTorch is not installed. Please run: pip install torch")

    n_features = len(feature_cols)
    hidden_sizes = getattr(args, "hidden_sizes", [256, 128, 64])
    if isinstance(hidden_sizes, (int, float)):
        hidden_sizes = [int(hidden_sizes)]
    dropout = float(getattr(args, "dropout", 0.0))
    model = BPNNRegressor(
        input_dim=n_features,
        hidden_sizes=[int(h) for h in hidden_sizes],
        dropout=dropout,
        seed=args.seed,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
    )

    train_loader, valid_loader = build_dataloaders(
        x_train, y_train, x_valid, y_valid, args.batch_size, device
    )
    best_rmse = float("inf")
    patience = getattr(args, "early_stopping_patience", 10)
    epochs_no_improve = 0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(X_b)
            loss = criterion(out, y_b.squeeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_b.size(0)
        train_loss /= len(x_train)

        if valid_loader is not None:
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in valid_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    out = model(X_b)
                    valid_loss += criterion(out, y_b.squeeze(1)).item() * X_b.size(0)
            valid_loss /= len(x_valid)
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
            # no validation set, fall back to training loss
            train_rmse = float(np.sqrt(train_loss))
            if train_rmse < best_rmse:
                best_rmse = train_rmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "pytorch_bpnn_best.pt")
    return model


def predict_numpy(
    model: "nn.Module",
    x: pd.DataFrame,
    device: "torch.device",
    batch_size: int = 4096,
) -> np.ndarray:
    if x.empty:
        return np.array([], dtype=np.float32)
    model.eval()
    X = torch.from_numpy(x.values.astype(np.float32))
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size].to(device)
            out = model(batch)
            preds.append(out.cpu())
    return torch.cat(preds, dim=0).numpy().astype(np.float32)


def compute_feature_importance_gradient(
    model: "nn.Module",
    x: pd.DataFrame,
    feature_cols: list[str],
    device: "torch.device",
    n_samples: int = 2000,
) -> np.ndarray:
    """Feature importance via mean absolute (input * gradient) over samples."""
    if x.empty or len(feature_cols) == 0:
        return np.zeros(len(feature_cols), dtype=np.float64)
    model.eval()
    sample = x.sample(n=min(n_samples, len(x)), random_state=42) if len(x) > n_samples else x
    X = torch.from_numpy(sample.values.astype(np.float32)).to(device)
    X.requires_grad_(True)
    out = model(X)
    out.sum().backward()
    grad = X.grad
    if grad is None:
        return np.zeros(len(feature_cols), dtype=np.float64)
    imp = (X.detach().abs() * grad.abs()).mean(dim=0).cpu().numpy()
    return np.asarray(imp, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Daily PM2.5 prediction for BTH/YRD/PRD using BP neural network (GPU/CPU)."
    )
    parser.add_argument(
        "--daily-input",
        nargs="+",
        default=None,
        help="Optional daily long-table file(s)/folder(s), columns include city/date/pm25.",
    )
    parser.add_argument(
        "--pm25-day-dir",
        type=str,
        default=str(DEFAULT_PM25_DAY_DIR),
        help=f"PM2.5 daily NC directory (default: {DEFAULT_PM25_DAY_DIR})",
    )
    parser.add_argument(
        "--era5-day-dir",
        type=str,
        default=str(DEFAULT_ERA5_DAY_DIR),
        help=f"ERA5 daily NC directory (default: {DEFAULT_ERA5_DAY_DIR})",
    )
    parser.add_argument(
        "--city-geojson-dir",
        type=str,
        default=str(DEFAULT_CITY_GEOJSON_DIR),
        help=f"City geojson directory (default: {DEFAULT_CITY_GEOJSON_DIR})",
    )
    parser.add_argument(
        "--skip-era5",
        action="store_true",
        help="Skip merging ERA5 daily meteorological features.",
    )
    parser.add_argument(
        "--correlation-dir",
        type=str,
        default=str(DEFAULT_CORRELATION_DIR),
        help=f"Correlation directory (default: {DEFAULT_CORRELATION_DIR})",
    )
    parser.add_argument(
        "--data-read-dir",
        type=str,
        default=str(DEFAULT_DATA_READ_DIR),
        help=f"Data Read directory (default: {DEFAULT_DATA_READ_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--train-end-year", type=int, default=2021, help="Train years <= this year.")
    parser.add_argument("--valid-year", type=int, default=2022, help="Validation year.")
    parser.add_argument("--test-year", type=int, default=2023, help="Test year.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="Hidden layer sizes (default: 256 128 64).",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate applied after each hidden layer.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for Adam optimizer.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device: 'cuda', 'gpu', or 'cpu'. Default: auto-detect.",
    )
    parser.add_argument("--disable-shap", action="store_true", help="Disable SHAP explainability.")
    parser.add_argument("--shap-max-samples", type=int, default=3000, help="Max rows for SHAP.")
    parser.add_argument("--shap-max-display", type=int, default=20, help="Max displayed SHAP features.")
    parser.add_argument(
        "--pm25-precropped",
        dest="pm25_precropped",
        action="store_true",
        default=True,
        help="PM2.5 NC 已按城市群预裁剪（默认）.",
    )
    parser.add_argument(
        "--no-pm25-precropped",
        dest="pm25_precropped",
        action="store_false",
        help="PM2.5 为全局数据，读取时做空间裁剪.",
    )
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
        pm25_precropped=args.pm25_precropped,
    )
    feature_df = build_daily_features(base_df)

    train_df, valid_df, test_df = split_by_time(
        feature_df,
        train_end_year=args.train_end_year,
        valid_year=args.valid_year,
        test_year=args.test_year,
    )

    target_cluster_order = ["BTH", "YRD", "PRD"]
    available_clusters = sorted(
        set(train_df["cluster"].dropna().astype(str))
        | set(valid_df["cluster"].dropna().astype(str))
        | set(test_df["cluster"].dropna().astype(str))
    )
    clusters_to_train = [name for name in target_cluster_order if name in available_clusters]
    if not clusters_to_train:
        raise ValueError("No cluster samples found for BTH/YRD/PRD after split.")

    cluster_results: dict[str, Any] = {}

    for cluster_name in clusters_to_train:
        cluster_train_df = train_df.loc[train_df["cluster"] == cluster_name].copy()
        cluster_valid_df = valid_df.loc[valid_df["cluster"] == cluster_name].copy()
        cluster_test_df = test_df.loc[test_df["cluster"] == cluster_name].copy()

        (
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            y_test,
            feature_cols,
        ) = build_model_matrices(
            train_df=cluster_train_df,
            valid_df=cluster_valid_df,
            test_df=cluster_test_df,
        )

        if x_train.empty or x_valid.empty or x_test.empty:
            print(
                f"[WARN] 跳过城市群 {cluster_name}: "
                f"train={len(x_train)}, valid={len(x_valid)}, test={len(x_test)}"
            )
            continue

        cluster_output_dir = output_dir / cluster_name.lower()
        cluster_output_dir.mkdir(parents=True, exist_ok=True)

        model = train_bpnn(
            x_train, y_train, x_valid, y_valid, feature_cols, args, device, cluster_output_dir
        )

        pred_train = predict_numpy(model, x_train, device)
        pred_valid = predict_numpy(model, x_valid, device)
        pred_test = predict_numpy(model, x_test, device)

        metric_rows = [
            {
                "split": "train",
                **compute_metrics(y_train, pred_train),
                "n_samples": int(len(y_train)),
            },
            {
                "split": "valid",
                **compute_metrics(y_valid, pred_valid),
                "n_samples": int(len(y_valid)),
            },
            {
                "split": "test",
                **compute_metrics(y_test, pred_test),
                "n_samples": int(len(y_test)),
            },
        ]
        metrics_df = pd.DataFrame(metric_rows)

        all_pred_df, test_pred_df = build_prediction_frames(
            train_df=cluster_train_df,
            valid_df=cluster_valid_df,
            test_df=cluster_test_df,
            pred_train=pred_train,
            pred_valid=pred_valid,
            pred_test=pred_test,
        )
        cluster_metrics_df = metrics_by_cluster(test_pred_df)
        generalization_df = export_generalization_artifacts(metrics_df, cluster_output_dir)
        export_regression_artifacts(
            all_pred_df=all_pred_df,
            output_dir=cluster_output_dir,
            model_name=f"PyTorch_BPNN-{cluster_name}",
        )

        shap_status = "disabled"
        if not args.disable_shap:
            try:
                x_shap = x_test if not x_test.empty else (x_valid if not x_valid.empty else x_train)
                export_shap_artifacts(
                    model=model,
                    x_for_shap=x_shap,
                    output_dir=cluster_output_dir,
                    model_name=f"PyTorch_BPNN-{cluster_name}",
                    shap_max_samples=args.shap_max_samples,
                    shap_max_display=args.shap_max_display,
                    random_state=args.seed,
                )
                shap_status = "ok"
            except Exception as exc:  # pylint: disable=broad-except
                shap_status = f"failed: {exc}"
                print(f"[WARN] SHAP export failed for cluster {cluster_name}: {exc}")

        importance = compute_feature_importance_gradient(
            model, x_valid if not x_valid.empty else x_train, feature_cols, device
        )
        importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": importance, "cluster": cluster_name}
        ).sort_values("importance", ascending=False, kind="mergesort")

        metrics_df.to_csv(
            cluster_output_dir / "metrics_overall.csv",
            index=False,
            encoding="utf-8-sig",
        )
        cluster_metrics_df.to_csv(
            cluster_output_dir / "metrics_by_cluster_test.csv",
            index=False,
            encoding="utf-8-sig",
        )
        test_pred_df.sort_values(["date", "city"], kind="mergesort").to_csv(
            cluster_output_dir / "predictions_test.csv",
            index=False,
            encoding="utf-8-sig",
        )
        all_pred_df.sort_values(["split", "date", "city"], kind="mergesort").to_csv(
            cluster_output_dir / "predictions_all_splits.csv",
            index=False,
            encoding="utf-8-sig",
        )
        importance_df.to_csv(
            cluster_output_dir / "feature_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )

        cluster_results[cluster_name] = {
            "feature_cols": feature_cols,
            "all_pred_df": all_pred_df,
            "test_pred_df": test_pred_df,
            "metrics_df": metrics_df,
            "importance_df": importance_df,
            "train_rows": int(len(cluster_train_df)),
            "valid_rows": int(len(cluster_valid_df)),
            "test_rows": int(len(cluster_test_df)),
            "generalization_df": generalization_df,
            "shap_status": shap_status,
        }

    if not cluster_results:
        raise ValueError("No cluster model was successfully trained.")

    all_pred_frames: list[pd.DataFrame] = []
    test_pred_frames: list[pd.DataFrame] = []
    metrics_by_cluster_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    per_cluster_run_info: dict[str, Any] = {}

    for cluster_name in target_cluster_order:
        if cluster_name not in cluster_results:
            continue
        result = cluster_results[cluster_name]
        metrics_df = result["metrics_df"]
        all_pred_df = result["all_pred_df"]
        test_pred_df = result["test_pred_df"]
        importance_df = result["importance_df"]
        generalization_df = result["generalization_df"]

        metrics_by_cluster_frames.append(metrics_df.assign(cluster=cluster_name))
        all_pred_frames.append(all_pred_df.assign(model_cluster=cluster_name))
        test_pred_frames.append(test_pred_df.assign(model_cluster=cluster_name))
        importance_frames.append(importance_df)
        per_cluster_run_info[cluster_name] = {
            "output_dir": str(output_dir / cluster_name.lower()),
            "n_features": int(len(result["feature_cols"])),
            "train_rows": int(result["train_rows"]),
            "valid_rows": int(result["valid_rows"]),
            "test_rows": int(result["test_rows"]),
            "generalization_level": str(generalization_df.loc[0, "generalization_level"]) if not generalization_df.empty else "",
            "shap_status": result["shap_status"],
        }

    all_pred_df = pd.concat(all_pred_frames, ignore_index=True) if all_pred_frames else pd.DataFrame()
    test_pred_df = pd.concat(test_pred_frames, ignore_index=True) if test_pred_frames else pd.DataFrame()
    metrics_overall_by_cluster_df = (
        pd.concat(metrics_by_cluster_frames, ignore_index=True) if metrics_by_cluster_frames else pd.DataFrame()
    )
    feature_importance_by_cluster_df = (
        pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    )

    pooled_metric_rows: list[dict[str, Any]] = []
    for split_name in ("train", "valid", "test"):
        split_df = all_pred_df.loc[all_pred_df["split"] == split_name].copy()
        if split_df.empty:
            pooled_metric_rows.append(
                {
                    "split": split_name,
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "r2": float("nan"),
                    "n_samples": 0,
                }
            )
            continue
        split_metrics = compute_metrics(split_df["y_true"], split_df["y_pred"].to_numpy())
        pooled_metric_rows.append(
            {"split": split_name, **split_metrics, "n_samples": int(len(split_df))}
        )
    metrics_df = pd.DataFrame(pooled_metric_rows)
    cluster_metrics_df = metrics_by_cluster(test_pred_df)
    generalization_df = export_generalization_artifacts(metrics_df, output_dir)
    export_regression_artifacts(
        all_pred_df=all_pred_df,
        output_dir=output_dir,
        model_name="PyTorch_BPNN-ClusterModels",
    )

    metrics_df.to_csv(output_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(
        output_dir / "metrics_overall_pooled_from_cluster_models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_overall_by_cluster_df.to_csv(
        output_dir / "metrics_overall_by_cluster.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cluster_metrics_df.to_csv(
        output_dir / "metrics_by_cluster_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_pred_df.sort_values(["date", "city"], kind="mergesort").to_csv(
        output_dir / "predictions_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_pred_df.sort_values(["split", "date", "city"], kind="mergesort").to_csv(
        output_dir / "predictions_all_splits.csv",
        index=False,
        encoding="utf-8-sig",
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
        feature_importance_by_cluster_df.groupby("feature", as_index=False)[
            "importance"
        ].mean().sort_values(
            "importance",
            ascending=False,
            kind="mergesort",
        ).reset_index(drop=True).to_csv(
            output_dir / "feature_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )

    total_train_rows = int(sum(info["train_rows"] for info in per_cluster_run_info.values()))
    total_valid_rows = int(sum(info["valid_rows"] for info in per_cluster_run_info.values()))
    total_test_rows = int(sum(info["test_rows"] for info in per_cluster_run_info.values()))
    n_features_by_cluster = {
        cluster: int(info["n_features"]) for cluster, info in per_cluster_run_info.items()
    }

    run_info = {
        "model": "PyTorch_BPNN",
        "training_granularity": "cluster",
        "clusters_trained": list(per_cluster_run_info.keys()),
        "per_cluster": per_cluster_run_info,
        "device": str(device),
        "train_end_year": args.train_end_year,
        "valid_year": args.valid_year,
        "test_year": args.test_year,
        "n_features": int(max(n_features_by_cluster.values())) if n_features_by_cluster else 0,
        "n_features_by_cluster": n_features_by_cluster,
        "n_year_factor_features": len(year_factor_cols),
        "n_era5_daily_features": len(met_cols),
        "train_rows": total_train_rows,
        "valid_rows": total_valid_rows,
        "test_rows": total_test_rows,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_sizes": getattr(args, "hidden_sizes", [256, 128, 64]),
        "dropout": getattr(args, "dropout", 0.2),
        "weight_decay": getattr(args, "weight_decay", 1e-4),
        "early_stopping_patience": getattr(args, "early_stopping_patience", 10),
        "daily_input": args.daily_input or [],
        "pm25_day_dir": str(pm25_day_dir),
        "era5_day_dir": str(era5_day_dir),
        "year_factor_rule": "keep_yearly_value",
        "monthly_factor_rule": "divide_by_days_in_month",
        "generalization_level": str(generalization_df.loc[0, "generalization_level"]) if not generalization_df.empty else "",
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print("=" * 90)
    print("[INFO] PyTorch BPNN daily PM2.5 training finished.")
    print(f"[INFO] Output directory: {output_dir}")
    print("[INFO] Files:")
    print("       - metrics_overall.csv")
    print("       - metrics_by_cluster_test.csv")
    print("       - predictions_test.csv")
    print("       - predictions_all_splits.csv")
    print("       - feature_importance.csv")
    print("       - pytorch_bpnn_best.pt")
    print("       - generalization_assessment.csv")
    print("       - generalization_plot_data.csv")
    print("       - regression_all_splits_data.csv")
    print("       - regression_test_data.csv")
    print("       - shap_sample_features.csv (if SHAP enabled)")
    print("       - shap_values_wide.csv")
    print("       - shap_beeswarm_data_long.csv")
    print("       - shap_importance_bar_data.csv")
    print("       - run_info.json")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

