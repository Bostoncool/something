"""
Daily PM2.5 prediction for BTH/YRD/PRD using Gradient Boosting Decision Tree (GDBT).
Reuses daily_ml_pipeline for data preparation and sequence building; sequence input
is flattened to (n_samples, seq_len * n_features + 1) with city_id as extra feature.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

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

import joblib
from sklearn.ensemble import GradientBoostingRegressor


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "gdbt_daily_pm25"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _ensure_min_std(std_values: np.ndarray, min_std: float) -> np.ndarray:
    return np.where(std_values < float(min_std), float(min_std), std_values).astype(np.float32)


def fit_train_standardizers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    min_std: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training arrays must be non-empty for standardization.")
    x_mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = X_train.std(axis=0, keepdims=True).astype(np.float32)
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


def _city_to_id(meta: pd.DataFrame, city_categories: list[str]) -> np.ndarray:
    city_to_idx = {c: i for i, c in enumerate(city_categories)}
    ids = meta["city"].astype(str).map(lambda c: city_to_idx.get(c, 0)).values
    return np.asarray(ids, dtype=np.int64)


def flatten_sequence_for_gbdt(
    X: np.ndarray,
    city_id: np.ndarray,
) -> np.ndarray:
    """
    Flatten (n_samples, seq_len, n_features) to (n_samples, seq_len * n_features + 1).
    Appends city_id as last column for tree models.
    """
    if len(X) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    n = X.shape[0]
    X_flat = X.reshape(n, -1).astype(np.float32)
    city_col = city_id.astype(np.float32).reshape(-1, 1)
    return np.hstack([X_flat, city_col])


def train_gbdt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[GradientBoostingRegressor, float]:
    """Train GradientBoostingRegressor; return model and best valid RMSE."""
    model = GradientBoostingRegressor(
        n_estimators=getattr(args, "n_estimators", 200),
        max_depth=getattr(args, "max_depth", 6),
        learning_rate=getattr(args, "learning_rate", 0.1),
        subsample=getattr(args, "subsample", 0.8),
        min_samples_split=getattr(args, "min_samples_split", 5),
        min_samples_leaf=getattr(args, "min_samples_leaf", 2),
        max_features=getattr(args, "max_features", "sqrt"),
        random_state=args.seed,
        validation_fraction=0.1,  # 必须在 (0,1) 内，用于早停验证
    )
    model.fit(X_train, y_train)
    valid_pred = model.predict(X_valid)
    valid_rmse = float(np.sqrt(np.mean((y_valid - valid_pred) ** 2)))
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "gdbt_model.joblib")
    return model, valid_rmse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Daily PM2.5 prediction for BTH/YRD/PRD using GDBT (Gradient Boosting Decision Tree)."
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
    parser.add_argument("--min-std", type=float, default=1e-6, help="Lower bound for standard deviation in scaling.")
    # GDBT 超参数
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of boosting stages.")
    parser.add_argument("--max-depth", type=int, default=6, help="Max depth of trees.")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate for boosting.")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio of samples for each tree.")
    parser.add_argument("--min-samples-split", type=int, default=5, help="Min samples required to split a node.")
    parser.add_argument("--min-samples-leaf", type=int, default=2, help="Min samples in leaf.")
    parser.add_argument("--max-features", type=str, default="sqrt", help="Max features per split: int, float, 'sqrt', 'log2'.")
    parser.add_argument("--pm25-workers", type=int, default=64, help="Worker count for PM2.5 NC reading.")
    parser.add_argument("--era5-workers", type=int, default=64, help="Worker count for ERA5 NC reading.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "cache"),
        help="Cache directory for aggregated daily PM2.5/ERA5 parquet.",
    )
    parser.add_argument("--disable-data-cache", action="store_true", help="Disable NC aggregation cache.")
    parser.add_argument("--pm25-precropped", dest="pm25_precropped", action="store_true", default=True,
                        help="PM2.5 NC 已按城市群预裁剪（默认）.")
    parser.add_argument("--no-pm25-precropped", dest="pm25_precropped", action="store_false",
                        help="PM2.5 为全局数据，读取时做空间裁剪.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    np.random.seed(args.seed)

    correlation_dir = Path(args.correlation_dir).expanduser().resolve()
    data_read_dir = Path(args.data_read_dir).expanduser().resolve()
    pm25_day_dir = Path(args.pm25_day_dir).expanduser().resolve()
    era5_day_dir = Path(args.era5_day_dir).expanduser().resolve()
    city_geojson_dir = Path(args.city_geojson_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_pbar = tqdm(total=6, desc="GDBT 总进度", dynamic_ncols=True) if tqdm is not None else None

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
            module_tag="gdbt",
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

        city_id_train = _city_to_id(meta_train, city_categories)
        city_id_valid = _city_to_id(meta_valid, city_categories)
        city_id_test = _city_to_id(meta_test, city_categories)

        X_train_flat = flatten_sequence_for_gbdt(X_train, city_id_train)
        X_valid_flat = flatten_sequence_for_gbdt(X_valid, city_id_valid)
        X_test_flat = flatten_sequence_for_gbdt(X_test, city_id_test)

        x_mean, x_std, y_mean, y_std = fit_train_standardizers(
            X_train=X_train_flat,
            y_train=y_train,
            min_std=float(args.min_std),
        )
        X_train_scaled = transform_inputs(X_train_flat, x_mean, x_std)
        X_valid_scaled = transform_inputs(X_valid_flat, x_mean, x_std)
        X_test_scaled = transform_inputs(X_test_flat, x_mean, x_std)
        y_train_scaled = transform_targets(y_train, y_mean, y_std)
        y_valid_scaled = transform_targets(y_valid, y_mean, y_std)

        cluster_output_dir = output_dir / cluster_name.lower()
        cluster_output_dir.mkdir(parents=True, exist_ok=True)

        train_begin = time.perf_counter()
        model, _ = train_gbdt(
            X_train_scaled,
            y_train_scaled,
            X_valid_scaled,
            y_valid_scaled,
            args,
            cluster_output_dir,
        )
        train_seconds = time.perf_counter() - train_begin
        train_seconds_total += train_seconds

        pred_train_scaled = model.predict(X_train_scaled).astype(np.float32)
        pred_valid_scaled = model.predict(X_valid_scaled).astype(np.float32)
        pred_test_scaled = model.predict(X_test_scaled).astype(np.float32)
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
            "model": model,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
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
        export_regression_artifacts(
            all_pred_df=all_pred_df,
            output_dir=cluster_output_dir,
            model_name=f"GDBT-{cluster_name}",
        )
        scaler_params = {
            "y_mean": float(result["y_mean"]),
            "y_std": float(result["y_std"]),
            "x_mean": result["x_mean"].tolist(),
            "x_std": result["x_std"].tolist(),
            "feature_cols": result["feature_cols"],
            "city_categories": result["city_categories"],
            "seq_len": args.seq_len,
        }
        with open(cluster_output_dir / "scaler_params.json", "w", encoding="utf-8") as f:
            json.dump(scaler_params, f, ensure_ascii=False, indent=2)
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
    export_regression_artifacts(
        all_pred_df=all_pred_df,
        output_dir=output_dir,
        model_name="GDBT-ClusterModels",
    )
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
    n_features_by_cluster = {c: int(info["n_features"]) for c, info in per_cluster_run_info.items()}
    run_info = {
        "model": "GDBT",
        "training_granularity": "cluster",
        "clusters_trained": list(per_cluster_run_info.keys()),
        "per_cluster": per_cluster_run_info,
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
        "min_std": args.min_std,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
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
    print("[INFO] GDBT daily PM2.5 training finished.")
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
