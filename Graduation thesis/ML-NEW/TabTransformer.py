"""
Daily PM2.5 prediction for BTH/YRD/PRD using TabTransformer.
TabTransformer: Column embeddings for categorical (city, cluster) + Transformer + MLP.
Reuses daily_ml_pipeline for data preparation; uses build_model_matrices + categorical split.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

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
    build_model_matrices,
    build_prediction_frames,
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


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "tabtransformer_daily_pm25"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # pylint: disable=broad-except
        pass


def _append_city_cluster_codes(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Append city_code and cluster_code for categorical embedding (same logic as daily_ml_pipeline)."""
    city_categories = sorted(train["city"].dropna().astype(str).unique().tolist())
    cluster_categories = sorted(train["cluster"].dropna().astype(str).unique().tolist())
    for frame in (train, valid, test):
        frame["city_code"] = pd.Categorical(frame["city"], categories=city_categories).codes
        frame["cluster_code"] = pd.Categorical(frame["cluster"], categories=cluster_categories).codes
        frame["city_code"] = frame["city_code"].replace(-1, 0).clip(lower=0)
        frame["cluster_code"] = frame["cluster_code"].replace(-1, 0).clip(lower=0)


def build_tabtransformer_matrices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[int],
]:
    """
    Build (cat, cont, y) per split for TabTransformer.
    Categorical: city_code, cluster_code.
    Continuous: feature_cols from build_model_matrices.
    Returns n_categories_per_col for embedding sizes.
    """
    train = train_df.copy()
    valid = valid_df.copy()
    test = test_df.copy()
    _append_city_cluster_codes(train, valid, test)

    non_feature_cols = {"date", "city", "cluster", "pm25", "city_code", "cluster_code"}
    candidate_cols = [col for col in train.columns if col not in non_feature_cols]
    feature_cols = [col for col in candidate_cols if pd.api.types.is_numeric_dtype(train[col])]
    feature_cols = [col for col in feature_cols if train[col].notna().any()]
    if not feature_cols:
        raise ValueError("No usable numerical features detected for TabTransformer.")

    fill_values = train[feature_cols].median(numeric_only=True)
    for frame in (train, valid, test):
        frame[feature_cols] = frame[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(fill_values)

    cat_cols = ["city_code", "cluster_code"]
    n_city = int(train["city_code"].max()) + 1
    n_cluster = int(train["cluster_code"].max()) + 1
    n_categories_per_col = [max(1, n_city), max(1, n_cluster)]

    def _to_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cat = df[cat_cols].to_numpy(dtype=np.int64)
        cont = df[feature_cols].to_numpy(dtype=np.float32)
        y = df["pm25"].astype(float).to_numpy(dtype=np.float32)
        return cat, cont, y

    cat_train, cont_train, y_train = _to_arrays(train)
    cat_valid, cont_valid, y_valid = _to_arrays(valid)
    cat_test, cont_test, y_test = _to_arrays(test)

    return (
        cat_train,
        cont_train,
        y_train,
        cat_valid,
        cont_valid,
        y_valid,
        cat_test,
        cont_test,
        y_test,
        feature_cols,
        n_categories_per_col,
    )


def get_device(args: argparse.Namespace) -> "torch.device":
    if getattr(args, "device", None) and str(args.device).strip().lower() in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(args, "device", None) and str(args.device).strip().lower() == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabTransformerRegressor(nn.Module):
    """
    TabTransformer for tabular regression.
    Categorical: column embeddings -> Transformer -> contextual embeddings.
    Concat with continuous -> MLP -> scalar.
    """

    def __init__(
        self,
        n_categories_per_col: list[int],
        n_continuous: int,
        embed_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 3,
        mlp_hidden: list[int] | None = None,
        dropout: float = 0.2,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.n_cat_cols = len(n_categories_per_col)
        self.embed_dim = embed_dim
        self.embeddings = nn.ModuleList([
            nn.Embedding(max(1, n), embed_dim) for n in n_categories_per_col
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        cat_out_dim = self.n_cat_cols * embed_dim
        mlp_hidden = mlp_hidden or [128, 64]
        layers: list[nn.Module] = []
        prev = cat_out_dim + n_continuous
        for h in mlp_hidden:
            layers.extend([nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        cat: "torch.Tensor",
        cont: "torch.Tensor",
    ) -> "torch.Tensor":
        # cat: (B, n_cat_cols), cont: (B, n_cont)
        B = cat.size(0)
        embeds = []
        for i in range(self.n_cat_cols):
            embeds.append(self.embeddings[i](cat[:, i].clamp(0)))
        x_cat = torch.stack(embeds, dim=1)  # (B, n_cat_cols, embed_dim)
        x_cat = self.transformer(x_cat)  # (B, n_cat_cols, embed_dim)
        x_cat = x_cat.reshape(B, -1)  # (B, n_cat_cols * embed_dim)
        x = torch.cat([x_cat, cont], dim=-1)
        return self.mlp(x).squeeze(-1)


def build_tabtransformer_dataloaders(
    cat_train: np.ndarray,
    cont_train: np.ndarray,
    y_train: np.ndarray,
    cat_valid: np.ndarray,
    cont_valid: np.ndarray,
    y_valid: np.ndarray,
    batch_size: int,
    device: "torch.device",
) -> tuple["DataLoader", "DataLoader | None"]:
    cat_tr = torch.from_numpy(cat_train)
    cont_tr = torch.from_numpy(cont_train)
    y_tr = torch.from_numpy(y_train)
    train_ds = TensorDataset(cat_tr, cont_tr, y_tr)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = None
    if len(cat_valid) > 0:
        cat_va = torch.from_numpy(cat_valid)
        cont_va = torch.from_numpy(cont_valid)
        y_va = torch.from_numpy(y_valid)
        valid_ds = TensorDataset(cat_va, cont_va, y_va)
        valid_loader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
    return train_loader, valid_loader


def fit_cont_standardizer(
    cont_train: np.ndarray,
    min_std: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    mean = cont_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = cont_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < min_std, min_std, std).astype(np.float32)
    return mean, std


def transform_cont(cont: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((cont - mean) / std).astype(np.float32, copy=False)


def train_tabtransformer(
    cat_train: np.ndarray,
    cont_train: np.ndarray,
    y_train: np.ndarray,
    cat_valid: np.ndarray,
    cont_valid: np.ndarray,
    y_valid: np.ndarray,
    n_categories_per_col: list[int],
    feature_cols: list[str],
    args: argparse.Namespace,
    device: "torch.device",
    output_dir: Path,
) -> tuple["nn.Module", float]:
    if torch is None or nn is None:
        raise ImportError("PyTorch is not installed. Please run: pip install torch")

    cont_mean, cont_std = fit_cont_standardizer(cont_train)
    cont_train_norm = transform_cont(cont_train, cont_mean, cont_std)
    cont_valid_norm = transform_cont(cont_valid, cont_mean, cont_std)

    model = TabTransformerRegressor(
        n_categories_per_col=n_categories_per_col,
        n_continuous=len(feature_cols),
        embed_dim=int(getattr(args, "embed_dim", 32)),
        n_heads=int(getattr(args, "n_heads", 4)),
        n_layers=int(getattr(args, "n_layers", 3)),
        dropout=float(getattr(args, "dropout", 0.2)),
        seed=args.seed,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=float(getattr(args, "weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    train_loader, valid_loader = build_tabtransformer_dataloaders(
        cat_train,
        cont_train_norm,
        y_train,
        cat_valid,
        cont_valid_norm,
        y_valid,
        args.batch_size,
        device,
    )

    best_rmse = float("inf")
    patience = getattr(args, "early_stopping_patience", 15)
    epochs_no_improve = 0
    best_state: dict[str, Any] | None = None

    epoch_iter = range(1, args.epochs + 1)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc="TabTransformer 训练轮次", dynamic_ncols=True)

    for epoch in epoch_iter:
        model.train()
        train_loss = 0.0
        for cat_b, cont_b, y_b in train_loader:
            cat_b = cat_b.to(device)
            cont_b = cont_b.to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(cat_b, cont_b)
            loss = criterion(out, y_b)
            loss.backward()
            if float(getattr(args, "grad_clip_norm", 5.0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm))
            optimizer.step()
            train_loss += loss.item() * cat_b.size(0)
        train_loss /= len(cat_train)
        train_rmse = float(np.sqrt(train_loss))

        if valid_loader is not None:
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for cat_b, cont_b, y_b in valid_loader:
                    cat_b = cat_b.to(device)
                    cont_b = cont_b.to(device)
                    y_b = y_b.to(device)
                    out = model(cat_b, cont_b)
                    valid_loss += float(criterion(out, y_b)) * cat_b.size(0)
            valid_loss /= len(cat_valid)
            valid_rmse = float(np.sqrt(valid_loss))
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
            if train_rmse < best_rmse:
                best_rmse = train_rmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if tqdm is not None:
                epoch_iter.set_postfix(train_rmse=train_rmse, best_rmse=best_rmse)

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "tabtransformer_best.pt")
    joblib.dump(
        {"cont_mean": cont_mean, "cont_std": cont_std, "feature_cols": feature_cols},
        output_dir / "tabtransformer_scaler.joblib",
    )
    return model, best_rmse


def predict_tabtransformer(
    model: "nn.Module",
    cat: np.ndarray,
    cont: np.ndarray,
    cont_mean: np.ndarray,
    cont_std: np.ndarray,
    device: "torch.device",
) -> np.ndarray:
    model.eval()
    cont_norm = transform_cont(cont, cont_mean, cont_std)
    cat_t = torch.from_numpy(cat).to(device)
    cont_t = torch.from_numpy(cont_norm).to(device)
    with torch.no_grad():
        out = model(cat_t, cont_t)
    return out.cpu().numpy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Daily PM2.5 prediction for BTH/YRD/PRD using TabTransformer."
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
    parser.add_argument("--epochs", type=int, default=150, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--grad-clip-norm", type=float, default=5.0, help="Gradient clipping; <=0 disables.")
    parser.add_argument("--embed-dim", type=int, default=32, help="TabTransformer embedding dimension.")
    parser.add_argument("--n-heads", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--n-layers", type=int, default=3, help="Transformer encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
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
    stage_pbar = tqdm(total=6, desc="TabTransformer 总进度", dynamic_ncols=True) if tqdm is not None else None

    device = get_device(args)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
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
            module_tag="tabtransformer",
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
        stage_pbar.set_postfix_str("特征构建完成")

    for cluster_name in clusters_to_train:
        cluster_train_df = train_df.loc[train_df["cluster"] == cluster_name].copy()
        cluster_valid_df = valid_df.loc[valid_df["cluster"] == cluster_name].copy()
        cluster_test_df = test_df.loc[test_df["cluster"] == cluster_name].copy()
        if cluster_train_df.empty or cluster_valid_df.empty or cluster_test_df.empty:
            print(
                f"[WARN] 跳过城市群 {cluster_name}: "
                f"train={len(cluster_train_df)}, valid={len(cluster_valid_df)}, test={len(cluster_test_df)}"
            )
            continue

        (
            cat_train,
            cont_train,
            y_train,
            cat_valid,
            cont_valid,
            y_valid,
            cat_test,
            cont_test,
            y_test,
            feature_cols,
            n_categories_per_col,
        ) = build_tabtransformer_matrices(
            cluster_train_df,
            cluster_valid_df,
            cluster_test_df,
        )

        cluster_output_dir = output_dir / cluster_name.lower()
        cluster_output_dir.mkdir(parents=True, exist_ok=True)

        train_begin = time.perf_counter()
        try:
            model, _ = train_tabtransformer(
                cat_train,
                cont_train,
                y_train,
                cat_valid,
                cont_valid,
                y_valid,
                n_categories_per_col,
                feature_cols,
                args,
                device,
                cluster_output_dir,
            )
        except Exception as exc:
            print(f"[WARN] 城市群 {cluster_name} TabTransformer 训练失败: {exc}")
            continue
        train_seconds = time.perf_counter() - train_begin
        train_seconds_total += train_seconds

        meta = joblib.load(cluster_output_dir / "tabtransformer_scaler.joblib")
        cont_mean = meta["cont_mean"]
        cont_std = meta["cont_std"]

        pred_train = predict_tabtransformer(model, cat_train, cont_train, cont_mean, cont_std, device)
        pred_valid = predict_tabtransformer(model, cat_valid, cont_valid, cont_mean, cont_std, device)
        pred_test = predict_tabtransformer(model, cat_test, cont_test, cont_mean, cont_std, device)

        metric_rows = [
            {"split": "train", **compute_metrics(pd.Series(y_train), pred_train), "n_samples": int(len(y_train))},
            {"split": "valid", **compute_metrics(pd.Series(y_valid), pred_valid), "n_samples": int(len(y_valid))},
            {"split": "test", **compute_metrics(pd.Series(y_test), pred_test), "n_samples": int(len(y_test))},
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

        importance = np.abs(np.mean(cont_train, axis=0))
        importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": importance, "cluster": cluster_name}
        ).sort_values("importance", ascending=False, kind="mergesort")

        cluster_results[cluster_name] = {
            "model": model,
            "feature_cols": feature_cols,
            "all_pred_df": all_pred_df,
            "test_pred_df": test_pred_df,
            "metrics_df": metrics_df,
            "importance_df": importance_df,
            "train_rows": int(len(y_train)),
            "valid_rows": int(len(y_valid)),
            "test_rows": int(len(y_test)),
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
            model_name=f"TabTransformer-{cluster_name}",
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
        model_name="TabTransformer-ClusterModels",
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
        "model": "TabTransformer",
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
        "train_rows": int(sum(info["train_rows"] for info in per_cluster_run_info.values())),
        "valid_rows": int(sum(info["valid_rows"] for info in per_cluster_run_info.values())),
        "test_rows": int(sum(info["test_rows"] for info in per_cluster_run_info.values())),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "embed_dim": args.embed_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
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
    print("[INFO] TabTransformer daily PM2.5 training finished.")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Trained clusters: {', '.join(per_cluster_run_info.keys())}")
    for cluster_name in per_cluster_run_info:
        print(f"       - {cluster_name}: {output_dir / cluster_name.lower()}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
