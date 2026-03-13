"""
Daily PM2.5 prediction for BTH/YRD/PRD using Prophet with per-cluster training.
Time series model: aggregates by (date, cluster), trains Prophet with regressors, broadcasts predictions.
Reuses daily_ml_pipeline for data preparation and evaluation.
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
    from prophet import Prophet  # type: ignore[import-untyped]
except ImportError as _e:
    Prophet = None  # type: ignore[misc, assignment]
    _prophet_import_error = _e

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
    compute_metrics,
    export_generalization_artifacts,
    export_regression_artifacts,
    metrics_by_cluster,
    prepare_training_table,
    split_by_time,
)
from cluster_training_utils import prepare_training_table_with_fallback


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "prophet_daily_pm25"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # pylint: disable=broad-except
        pass


def _aggregate_cluster_by_date(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """按 (date, cluster) 聚合：pm25 与特征取均值，保证每个 cluster 每个 date 一行"""
    agg_dict: dict[str, str] = {"pm25": "mean"}
    for c in feature_cols:
        if c in df.columns:
            agg_dict[c] = "mean"
    return df.groupby(["date", "cluster"], as_index=False).agg(agg_dict).sort_values("date")


def _broadcast_predictions_to_cities(df: pd.DataFrame, pred_by_date: pd.Series) -> np.ndarray:
    """pred_by_date.index = date, 返回与 df 行数一致的预测数组"""
    return df["date"].map(pred_by_date).to_numpy()


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Extract numeric feature columns (same logic as build_model_matrices)."""
    non_feature_cols = {"date", "city", "cluster", "pm25"}
    candidate_cols = [col for col in df.columns if col not in non_feature_cols]
    feature_cols = [col for col in candidate_cols if pd.api.types.is_numeric_dtype(df[col])]
    feature_cols = [col for col in feature_cols if df[col].notna().any()]
    return feature_cols


def train_and_predict_prophet(
    train_agg: pd.DataFrame,
    valid_agg: pd.DataFrame,
    test_agg: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> tuple[Any, pd.Series, pd.Series, pd.Series]:
    """Fit Prophet on train, predict train/valid/test. Returns (model, pred_train, pred_valid, pred_test)."""
    if Prophet is None:
        raise ImportError("Prophet 需要 prophet，请安装: pip install prophet") from getattr(
            sys.modules.get(__name__), "_prophet_import_error", None
        )

    fill_values = train_agg[feature_cols].median(numeric_only=True)
    train_agg = train_agg.copy()
    valid_agg = valid_agg.copy()
    test_agg = test_agg.copy()
    for c in feature_cols:
        fv = float(fill_values[c]) if c in fill_values.index else 0.0
        train_agg[c] = pd.to_numeric(train_agg[c], errors="coerce").fillna(fv)
        valid_agg[c] = pd.to_numeric(valid_agg[c], errors="coerce").fillna(fv)
        test_agg[c] = pd.to_numeric(test_agg[c], errors="coerce").fillna(fv)

    train_prophet = train_agg.rename(columns={"date": "ds", "pm25": "y"})[["ds", "y"] + feature_cols].copy()
    valid_prophet = valid_agg.rename(columns={"date": "ds", "pm25": "y"})[["ds"] + feature_cols].copy()
    test_prophet = test_agg.rename(columns={"date": "ds", "pm25": "y"})[["ds"] + feature_cols].copy()

    train_prophet["ds"] = pd.to_datetime(train_prophet["ds"])
    valid_prophet["ds"] = pd.to_datetime(valid_prophet["ds"])
    test_prophet["ds"] = pd.to_datetime(test_prophet["ds"])

    m = Prophet(
        yearly_seasonality=args.yearly_seasonality,
        weekly_seasonality=args.weekly_seasonality,
        daily_seasonality=args.daily_seasonality,
        seasonality_mode=args.seasonality_mode,
        changepoint_prior_scale=args.changepoint_prior_scale,
        seasonality_prior_scale=args.seasonality_prior_scale,
    )
    for col in feature_cols:
        m.add_regressor(col)

    m.fit(train_prophet)

    pred_train_df = m.predict(train_prophet[["ds"] + feature_cols])
    pred_valid_df = m.predict(valid_prophet[["ds"] + feature_cols])
    pred_test_df = m.predict(test_prophet[["ds"] + feature_cols])

    pred_train_series = pd.Series(pred_train_df["yhat"].values, index=train_agg["date"].values)
    pred_valid_series = pd.Series(pred_valid_df["yhat"].values, index=valid_agg["date"].values)
    pred_test_series = pd.Series(pred_test_df["yhat"].values, index=test_agg["date"].values)

    return m, pred_train_series, pred_valid_series, pred_test_series


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Daily PM2.5 prediction for BTH/YRD/PRD using Prophet."
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
    parser.add_argument("--yearly-seasonality", action="store_true", default=True, help="Enable yearly seasonality.")
    parser.add_argument("--no-yearly-seasonality", dest="yearly_seasonality", action="store_false")
    parser.add_argument("--weekly-seasonality", action="store_true", default=True, help="Enable weekly seasonality.")
    parser.add_argument("--no-weekly-seasonality", dest="weekly_seasonality", action="store_false")
    parser.add_argument("--daily-seasonality", action="store_false", default=False, help="Enable daily seasonality.")
    parser.add_argument(
        "--seasonality-mode",
        type=str,
        default="additive",
        choices=("additive", "multiplicative"),
        help="Seasonality mode.",
    )
    parser.add_argument("--changepoint-prior-scale", type=float, default=0.05, help="Changepoint prior scale.")
    parser.add_argument("--seasonality-prior-scale", type=float, default=10.0, help="Seasonality prior scale.")
    parser.add_argument("--pm25-workers", type=int, default=64, help="Worker count for PM2.5 NC reading.")
    parser.add_argument("--era5-workers", type=int, default=64, help="Worker count for ERA5 NC reading.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "cache"),
        help="Cache directory for aggregated daily PM2.5/ERA5 parquet.",
    )
    parser.add_argument("--disable-data-cache", action="store_true", help="Disable NC aggregation cache.")
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
    args = build_parser().parse_args()
    correlation_dir = Path(args.correlation_dir).expanduser().resolve()
    data_read_dir = Path(args.data_read_dir).expanduser().resolve()
    pm25_day_dir = Path(args.pm25_day_dir).expanduser().resolve()
    era5_day_dir = Path(args.era5_day_dir).expanduser().resolve()
    city_geojson_dir = Path(args.city_geojson_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_pbar = tqdm(total=6, desc="Prophet 总进度", dynamic_ncols=True) if tqdm is not None else None
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
            module_tag="prophet",
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

        feature_cols = _get_feature_cols(cluster_train_df)
        if not feature_cols:
            print(f"[WARN] 跳过城市群 {cluster_name}: 无可用特征列")
            continue

        train_agg = _aggregate_cluster_by_date(cluster_train_df, feature_cols)
        valid_agg = _aggregate_cluster_by_date(cluster_valid_df, feature_cols)
        test_agg = _aggregate_cluster_by_date(cluster_test_df, feature_cols)

        train_begin = time.perf_counter()
        try:
            model, pred_train_series, pred_valid_series, pred_test_series = train_and_predict_prophet(
                train_agg, valid_agg, test_agg, feature_cols, args
            )
        except Exception as exc:
            print(f"[WARN] 城市群 {cluster_name} Prophet 训练失败: {exc}")
            continue
        train_seconds = time.perf_counter() - train_begin
        train_seconds_total += train_seconds

        pred_train = _broadcast_predictions_to_cities(cluster_train_df, pred_train_series)
        pred_valid = _broadcast_predictions_to_cities(cluster_valid_df, pred_valid_series)
        pred_test = _broadcast_predictions_to_cities(cluster_test_df, pred_test_series)

        y_train = cluster_train_df["pm25"].astype(float).values
        y_valid = cluster_valid_df["pm25"].astype(float).values
        y_test = cluster_test_df["pm25"].astype(float).values

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

        regressor_coefs: dict[str, float] = {}
        try:
            if hasattr(model, "params") and model.params is not None:
                params = model.params
                if hasattr(params, "get"):
                    for c in feature_cols:
                        key = f"beta_{c}" if f"beta_{c}" in str(params.index) else c
                        if key in params.index:
                            regressor_coefs[c] = float(params.loc[key, 0])
        except Exception:  # pylint: disable=broad-except
            pass
        for c in feature_cols:
            if c not in regressor_coefs:
                regressor_coefs[c] = 1.0

        cluster_results[cluster_name] = {
            "model": model,
            "feature_cols": feature_cols,
            "all_pred_df": all_pred_df,
            "test_pred_df": test_pred_df,
            "metrics_df": metrics_df,
            "regressor_coefs": regressor_coefs,
            "train_rows": int(len(cluster_train_df)),
            "valid_rows": int(len(cluster_valid_df)),
            "test_rows": int(len(cluster_test_df)),
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
        cluster_output_dir.mkdir(parents=True, exist_ok=True)
        result = cluster_results[cluster_name]
        model = result["model"]
        feature_cols = result["feature_cols"]
        all_pred_df = result["all_pred_df"]
        test_pred_df = result["test_pred_df"]
        metrics_df = result["metrics_df"]
        regressor_coefs = result.get("regressor_coefs", {})

        joblib.dump(
            {"model": model, "feature_cols": feature_cols},
            cluster_output_dir / "prophet_best.joblib",
        )

        cluster_metrics_df = metrics_by_cluster(test_pred_df)
        generalization_df = export_generalization_artifacts(metrics_df, cluster_output_dir)
        export_regression_artifacts(
            all_pred_df=all_pred_df,
            output_dir=cluster_output_dir,
            model_name=f"Prophet-{cluster_name}",
        )

        importance_vals = [abs(regressor_coefs.get(c, 0.0)) for c in feature_cols]
        importance_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": importance_vals,
                "cluster": cluster_name,
            }
        ).sort_values("importance", ascending=False, kind="mergesort")

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
            "n_features": int(len(feature_cols)),
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
        model_name="Prophet-ClusterModels",
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
        feature_importance_by_cluster_df.groupby("feature", as_index=False)[
            "importance"
        ].mean().sort_values("importance", ascending=False, kind="mergesort").reset_index(
            drop=True
        ).to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    n_features_by_cluster = {
        cluster: int(info["n_features"]) for cluster, info in per_cluster_run_info.items()
    }
    run_info = {
        "model": "Prophet",
        "training_granularity": "cluster",
        "clusters_trained": list(per_cluster_run_info.keys()),
        "per_cluster": per_cluster_run_info,
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
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as file:
        json.dump(run_info, file, ensure_ascii=False, indent=2)
    if stage_pbar is not None:
        stage_pbar.update(1)
        stage_pbar.set_postfix_str("结果导出完成")
        stage_pbar.close()

    print("=" * 90)
    print("[INFO] Prophet daily PM2.5 training finished.")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Trained clusters: {', '.join(per_cluster_run_info.keys())}")
    for cluster_name in per_cluster_run_info:
        print(f"       - {cluster_name}: {output_dir / cluster_name.lower()}")
    print("[INFO] Files:")
    print("       - metrics_overall.csv")
    print("       - predictions_test.csv")
    print("       - predictions_all_splits.csv")
    print("       - feature_importance.csv")
    print("       - run_info.json")
    print("       - prophet_best.joblib (per cluster)")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
