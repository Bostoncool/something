"""
Daily PM2.5 prediction for BTH/YRD/PRD using GCN (Graph Convolutional Network).
Treats cities as nodes, builds fully-connected graph within each cluster per date.
Reuses daily_ml_pipeline for data preparation; builds graph structure from (date, city, features).
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
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    Dataset = None


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "gcn_daily_pm25"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # pylint: disable=broad-except
        pass


def _build_adjacency_normalized(n_nodes: int, device: "torch.device") -> "torch.Tensor":
    """Fully connected graph with self-loops, normalized: D^{-1/2} (A+I) D^{-1/2}."""
    adj = np.ones((n_nodes, n_nodes), dtype=np.float32)
    adj = adj + np.eye(n_nodes, dtype=np.float32)
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    adj_norm = deg_inv_sqrt[:, np.newaxis] * adj * deg_inv_sqrt[np.newaxis, :]
    return torch.from_numpy(adj_norm.astype(np.float32)).to(device)


def _build_graph_batches(
    df: pd.DataFrame,
    feature_cols: list[str],
    city_order: list[str],
    fill_values: pd.Series,
    cluster_name: str = "",
) -> tuple[list[np.ndarray], list[np.ndarray], list[pd.DataFrame]]:
    """For each date, build (x, y, meta) where x=(N,F), y=(N,), meta has date,city,cluster,pm25 for each node."""
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    meta_list: list[pd.DataFrame] = []
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing in df: {missing}")
    fill_vals = fill_values.reindex(feature_cols).fillna(0.0)
    by_date = df.groupby("date", observed=True, sort=True)
    for date, g in by_date:
        g = g.copy()
        g[feature_cols] = g[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(fill_vals)
        city_to_row = g.set_index("city").to_dict("index")
        x_rows = []
        y_rows = []
        meta_rows = []
        for city in city_order:
            if city in city_to_row:
                row = city_to_row[city]
                x_rows.append([float(row.get(c, 0.0)) for c in feature_cols])
                y_rows.append(float(row.get("pm25", 0.0)))
                meta_rows.append({
                    "date": date,
                    "city": city,
                    "cluster": row.get("cluster", cluster_name),
                    "pm25": row.get("pm25", 0.0),
                })
            else:
                x_rows.append([float(fill_vals.get(c, 0.0)) for c in feature_cols])
                y_rows.append(0.0)
                meta_rows.append({
                    "date": date,
                    "city": city,
                    "cluster": cluster_name,
                    "pm25": 0.0,
                })
        if not x_rows:
            continue
        x_list.append(np.array(x_rows, dtype=np.float32))
        y_list.append(np.array(y_rows, dtype=np.float32))
        meta_list.append(pd.DataFrame(meta_rows))
    return x_list, y_list, meta_list


class GraphDataset(Dataset):
    """Dataset of (x, y) graph samples."""

    def __init__(
        self,
        x_list: list[np.ndarray],
        y_list: list[np.ndarray],
    ) -> None:
        self.x_list = x_list
        self.y_list = y_list

    def __len__(self) -> int:
        return len(self.x_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.x_list[idx]),
            torch.from_numpy(self.y_list[idx]),
        )


class _GroupedBatchSampler:
    """Yields batches of indices where each batch has samples with same shape (n_nodes, n_features)."""

    def __init__(
        self,
        x_list: list[np.ndarray],
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        idx_by_shape: dict[tuple[int, ...], list[int]] = {}
        for idx, x in enumerate(x_list):
            shp = tuple(int(s) for s in (x.shape if x.ndim >= 1 else (0,)))
            if shp not in idx_by_shape:
                idx_by_shape[shp] = []
            idx_by_shape[shp].append(idx)
        self._batches = []
        rng = np.random.default_rng(seed)
        for _shp, indices in idx_by_shape.items():
            if shuffle:
                indices = rng.permutation(indices).tolist()
            for i in range(0, len(indices), batch_size):
                self._batches.append(indices[i : i + batch_size])
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        batches = list(self._batches)
        if self._shuffle:
            self._rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches)


def collate_graph_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    device: "torch.device",
    adj_cache: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack graphs with same N into batch. Returns (x, y, adj)."""
    x_batch = torch.stack([b[0] for b in batch])
    y_batch = torch.stack([b[1] for b in batch])
    n_nodes = x_batch.shape[1]
    if n_nodes not in adj_cache:
        adj_cache[n_nodes] = _build_adjacency_normalized(n_nodes, device)
    adj = adj_cache[n_nodes]
    return x_batch.to(device), y_batch.to(device), adj


class GCNLayer(nn.Module):
    """GCN layer: H' = σ(A_norm @ H @ W). Supports batch (B, N, F)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight, gain=1.0)

    def forward(self, adj: "torch.Tensor", h: "torch.Tensor") -> "torch.Tensor":
        if h.dim() == 3:
            h = torch.einsum("ij,bjf->bif", adj, h)
        else:
            h = torch.matmul(adj, h)
        h = torch.matmul(h, self.weight)
        return h


class GCNRegressor(nn.Module):
    """2-layer GCN for node-level regression: (N, F) or (B, N, F) -> GCN -> ReLU -> GCN -> FC -> (N,) or (B, N)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        dropout: float = 0.3,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.gcn1 = GCNLayer(input_size, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: "torch.Tensor", adj: "torch.Tensor") -> "torch.Tensor":
        h = torch.relu(self.gcn1(adj, x))
        h = self.dropout(h)
        h = torch.relu(self.gcn2(adj, h))
        h = self.dropout(h)
        out = self.fc(h).squeeze(-1)
        return out


def train_gcn(
    x_list: list[np.ndarray],
    y_list: list[np.ndarray],
    x_valid_list: list[np.ndarray],
    y_valid_list: list[np.ndarray],
    feature_cols: list[str],
    args: argparse.Namespace,
    device: "torch.device",
    output_dir: Path,
) -> tuple["nn.Module", float]:
    if torch is None or nn is None:
        raise ImportError("PyTorch is not installed. Please run: pip install torch")

    n_features = len(feature_cols)
    n_nodes = x_list[0].shape[0] if x_list else 0
    if n_nodes == 0:
        raise ValueError("No graph samples.")

    model = GCNRegressor(
        input_size=n_features,
        hidden_size=int(getattr(args, "gcn_hidden", 64)),
        dropout=float(getattr(args, "dropout", 0.3)),
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

    train_ds = GraphDataset(x_list, y_list)
    adj_cache: dict[int, torch.Tensor] = {}
    batch_size = min(args.batch_size, len(train_ds))

    def _collate_fn(batch: list) -> tuple:
        return collate_graph_batch(batch, device, adj_cache)

    # 始终使用 _GroupedBatchSampler，按 x 形状分组，确保同 batch 内样本形状一致
    batch_sampler = _GroupedBatchSampler(
        x_list, batch_size, shuffle=True, seed=args.seed
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=0,
        collate_fn=_collate_fn,
    )

    best_rmse = float("inf")
    patience = getattr(args, "early_stopping_patience", 15)
    epochs_no_improve = 0
    best_state: dict[str, Any] | None = None

    epoch_iter = range(1, args.epochs + 1)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc="GCN 训练轮次", dynamic_ncols=True)

    for epoch in epoch_iter:
        model.train()
        train_loss = 0.0
        for x_b, y_b, adj in train_loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(x_b, adj)
            loss = criterion(out, y_b)
            loss.backward()
            grad_clip = float(getattr(args, "grad_clip_norm", 5.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_loss += loss.item() * x_b.size(0)
        train_loss /= len(x_list) if x_list else 1
        train_rmse = float(np.sqrt(train_loss))

        if x_valid_list:
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for x_v, y_v in zip(x_valid_list, y_valid_list):
                    x_t = torch.from_numpy(x_v).unsqueeze(0).to(device)
                    y_t = torch.from_numpy(y_v).float().to(device)
                    adj = _build_adjacency_normalized(x_v.shape[0], device)
                    out = model(x_t, adj).squeeze(0)
                    valid_loss += float(torch.nn.functional.mse_loss(out, y_t)) * x_v.shape[0]
            valid_loss /= sum(x.shape[0] for x in x_valid_list)
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
    torch.save(model.state_dict(), output_dir / "gcn_best.pt")
    return model, best_rmse


def predict_gcn(
    model: "nn.Module",
    x_list: list[np.ndarray],
    device: "torch.device",
) -> np.ndarray:
    """Predict for each graph, return flattened predictions in order."""
    model.eval()
    preds = []
    with torch.no_grad():
        for x in x_list:
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            adj = _build_adjacency_normalized(x.shape[0], device)
            out = model(x_t, adj).squeeze(0).cpu().numpy()
            preds.append(out)
    return np.concatenate(preds, axis=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Daily PM2.5 prediction for BTH/YRD/PRD using GCN (Graph Convolutional Network)."
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
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for graph training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--grad-clip-norm", type=float, default=5.0, help="Gradient clipping; <=0 disables.")
    parser.add_argument("--gcn-hidden", type=int, default=64, help="GCN hidden size.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")
    parser.add_argument("--early-stopping-patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--device", type=str, default="", help="Device: 'cuda', 'gpu', or 'cpu'.")
    parser.add_argument("--pm25-workers", type=int, default=64, help="Worker count for PM2.5 NC reading.")
    parser.add_argument("--era5-workers", type=int, default=64, help="Worker count for ERA5 NC reading.")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_OUTPUT_DIR / "cache"), help="Cache directory.")
    parser.add_argument("--disable-data-cache", action="store_true", help="Disable NC aggregation cache.")
    parser.add_argument("--pm25-precropped", dest="pm25_precropped", action="store_true", default=True)
    parser.add_argument("--no-pm25-precropped", dest="pm25_precropped", action="store_false")
    return parser


def get_device(args: argparse.Namespace) -> "torch.device":
    if getattr(args, "device", None) and str(args.device).strip().lower() in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(args, "device", None) and str(args.device).strip().lower() == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    stage_pbar = tqdm(total=6, desc="GCN 总进度", dynamic_ncols=True) if tqdm is not None else None

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
            module_tag="gcn",
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

        _, _, _, _, _, _, feature_cols = build_model_matrices(
            cluster_train_df, cluster_valid_df, cluster_test_df
        )
        city_order = sorted(
            set(cluster_train_df["city"].dropna().astype(str).unique())
            | set(cluster_valid_df["city"].dropna().astype(str).unique())
            | set(cluster_test_df["city"].dropna().astype(str).unique())
        )
        if len(city_order) < 2:
            print(f"[WARN] 跳过城市群 {cluster_name}: 城市数 < 2，无法构建图")
            continue

        fill_values = cluster_train_df[feature_cols].median(numeric_only=True)
        cluster_train_df["pm25"] = cluster_train_df["pm25"].astype(float)
        cluster_valid_df["pm25"] = cluster_valid_df["pm25"].astype(float)
        cluster_test_df["pm25"] = cluster_test_df["pm25"].astype(float)

        x_train_list, y_train_list, meta_train_list = _build_graph_batches(
            cluster_train_df, feature_cols, city_order, fill_values, cluster_name
        )
        x_valid_list, y_valid_list, meta_valid_list = _build_graph_batches(
            cluster_valid_df, feature_cols, city_order, fill_values, cluster_name
        )
        x_test_list, y_test_list, meta_test_list = _build_graph_batches(
            cluster_test_df, feature_cols, city_order, fill_values, cluster_name
        )

        if not x_train_list or not x_valid_list or not x_test_list:
            print(
                f"[WARN] 跳过城市群 {cluster_name}: "
                f"train={len(x_train_list)}, valid={len(x_valid_list)}, test={len(x_test_list)}"
            )
            continue

        cluster_output_dir = output_dir / cluster_name.lower()
        cluster_output_dir.mkdir(parents=True, exist_ok=True)

        train_begin = time.perf_counter()
        try:
            model, _ = train_gcn(
                x_train_list,
                y_train_list,
                x_valid_list,
                y_valid_list,
                feature_cols,
                args,
                device,
                cluster_output_dir,
            )
        except Exception as exc:
            print(f"[WARN] 城市群 {cluster_name} GCN 训练失败: {exc}")
            continue
        train_seconds = time.perf_counter() - train_begin
        train_seconds_total += train_seconds

        pred_train = predict_gcn(model, x_train_list, device)
        pred_valid = predict_gcn(model, x_valid_list, device)
        pred_test = predict_gcn(model, x_test_list, device)

        meta_train = pd.concat(meta_train_list, ignore_index=True)
        meta_valid = pd.concat(meta_valid_list, ignore_index=True)
        meta_test = pd.concat(meta_test_list, ignore_index=True)

        y_train_flat = np.concatenate(y_train_list, axis=0)
        y_valid_flat = np.concatenate(y_valid_list, axis=0)
        y_test_flat = np.concatenate(y_test_list, axis=0)

        metric_rows = [
            {"split": "train", **compute_metrics(pd.Series(y_train_flat), pred_train), "n_samples": int(len(y_train_flat))},
            {"split": "valid", **compute_metrics(pd.Series(y_valid_flat), pred_valid), "n_samples": int(len(y_valid_flat))},
            {"split": "test", **compute_metrics(pd.Series(y_test_flat), pred_test), "n_samples": int(len(y_test_flat))},
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

        try:
            importance = np.abs(np.mean(np.stack(x_train_list, axis=0), axis=(0, 1)))
        except ValueError:
            importance = np.mean([np.abs(x).mean(axis=0) for x in x_train_list], axis=0)
        importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": importance, "cluster": cluster_name}
        ).sort_values("importance", ascending=False, kind="mergesort")

        joblib.dump(
            {"feature_cols": feature_cols, "city_order": city_order},
            cluster_output_dir / "gcn_meta.joblib",
        )

        cluster_results[cluster_name] = {
            "model": model,
            "feature_cols": feature_cols,
            "all_pred_df": all_pred_df,
            "test_pred_df": test_pred_df,
            "metrics_df": metrics_df,
            "importance_df": importance_df,
            "train_rows": int(len(y_train_flat)),
            "valid_rows": int(len(y_valid_flat)),
            "test_rows": int(len(y_test_flat)),
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
            model_name=f"GCN-{cluster_name}",
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
        model_name="GCN-ClusterModels",
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
        "model": "GCN",
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
        "gcn_hidden": args.gcn_hidden,
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
    print("[INFO] GCN daily PM2.5 training finished.")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Trained clusters: {', '.join(per_cluster_run_info.keys())}")
    for cluster_name in per_cluster_run_info:
        print(f"       - {cluster_name}: {output_dir / cluster_name.lower()}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
