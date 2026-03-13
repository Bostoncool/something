from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from daily_ml_pipeline import (
    DEFAULT_CITY_GEOJSON_DIR,
    DEFAULT_CORRELATION_DIR,
    DEFAULT_DATA_READ_DIR,
    DEFAULT_ERA5_DAY_DIR,
    DEFAULT_PM25_DAY_DIR,
    SCRIPT_DIR,
    build_model_matrices,
    build_pm25_nc_file_index,
    build_prediction_frames,
    compute_metrics,
    export_feature_quality_report,
    export_generalization_artifacts,
    export_regression_artifacts,
    export_shap_artifacts,
    prepare_training_table,
    split_by_time,
)

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "lightgbm_bth_seasonal_daily_pm25"
SEASON_ORDER = ("winter", "non_winter")
DEFAULT_WINTER_MONTHS = (11, 12, 1, 2, 3)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover  # pylint: disable=broad-except
        pass


def _load_bth_module() -> Any:
    module_path = SCRIPT_DIR / "Lightgbm-BTH.py"
    spec = importlib.util.spec_from_file_location("lightgbm_bth_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BTH_MODULE = _load_bth_module()
train_lightgbm = _BTH_MODULE.train_lightgbm
is_memory_pressure_error = _BTH_MODULE.is_memory_pressure_error
discover_pm25_years = _BTH_MODULE.discover_pm25_years
build_fallback_feature_cache = _BTH_MODULE.build_fallback_feature_cache
build_bth_feature_frame = _BTH_MODULE.build_bth_feature_frame


def parse_winter_months(raw_values: list[int] | None) -> list[int]:
    values = list(DEFAULT_WINTER_MONTHS if not raw_values else raw_values)
    months = sorted({int(value) for value in values})
    if not months or any(month < 1 or month > 12 for month in months):
        raise ValueError("winter_months must be integers in 1..12")
    return months


def add_season_partition(df: pd.DataFrame, winter_months: list[int]) -> pd.DataFrame:
    out = df.copy()
    winter_set = set(int(month) for month in winter_months)
    out["is_winter_month"] = out["month"].isin(winter_set).astype(int)
    out["season_partition"] = np.where(out["is_winter_month"] == 1, "winter", "non_winter")
    return out


def split_frames_by_season(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
    season_frames: dict[str, dict[str, pd.DataFrame]] = {}
    for season_name in SEASON_ORDER:
        season_frames[season_name] = {
            "train": train_df.loc[train_df["season_partition"] == season_name].copy(),
            "valid": valid_df.loc[valid_df["season_partition"] == season_name].copy(),
            "test": test_df.loc[test_df["season_partition"] == season_name].copy(),
        }
    return season_frames


def validate_season_frames(season_frames: dict[str, dict[str, pd.DataFrame]]) -> None:
    for season_name, frames in season_frames.items():
        for split_name in ("train", "valid", "test"):
            frame = frames[split_name]
            if frame.empty:
                raise ValueError(f"Season '{season_name}' has empty {split_name} split.")


def _metric_rows_from_predictions(
    *,
    train_pred_df: pd.DataFrame,
    valid_pred_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
) -> pd.DataFrame:
    split_map = {
        "train": train_pred_df,
        "valid": valid_pred_df,
        "test": test_pred_df,
    }
    rows: list[dict[str, Any]] = []
    for split_name, pred_df in split_map.items():
        if pred_df.empty:
            rows.append(
                {"split": split_name, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n_samples": 0}
            )
            continue
        metrics = compute_metrics(pred_df["y_true"], pred_df["y_pred"].to_numpy())
        rows.append({"split": split_name, **metrics, "n_samples": int(len(pred_df))})
    return pd.DataFrame(rows)


def train_one_season_model(
    *,
    season_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    x_train, y_train, x_valid, y_valid, x_test, y_test, feature_cols = build_model_matrices(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )
    train_begin = time.perf_counter()
    try:
        model, grid_search_info = train_lightgbm(x_train, y_train, x_valid, y_valid, args)
    except Exception as exc:
        if getattr(args, "enable_bayes_search", False):
            print(f"[WARN] {season_name} 贝叶斯搜索失败，回退默认参数训练: {exc}")
            old_flag = args.enable_bayes_search
            args.enable_bayes_search = False
            try:
                model, grid_search_info = train_lightgbm(x_train, y_train, x_valid, y_valid, args)
            finally:
                args.enable_bayes_search = old_flag
            grid_search_info["bayes_fallback_reason"] = str(exc)
        else:
            raise
    train_seconds = time.perf_counter() - train_begin

    pred_train = model.predict(x_train)
    pred_valid = model.predict(x_valid)
    pred_test = model.predict(x_test)
    all_pred_df, test_pred_df = build_prediction_frames(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        pred_train=pred_train,
        pred_valid=pred_valid,
        pred_test=pred_test,
    )
    all_pred_df["season_partition"] = season_name
    all_pred_df["route_model"] = f"lightgbm_{season_name}"
    test_pred_df["season_partition"] = season_name
    test_pred_df["route_model"] = f"lightgbm_{season_name}"
    metrics_df = _metric_rows_from_predictions(
        train_pred_df=all_pred_df.loc[all_pred_df["split"] == "train"].copy(),
        valid_pred_df=all_pred_df.loc[all_pred_df["split"] == "valid"].copy(),
        test_pred_df=all_pred_df.loc[all_pred_df["split"] == "test"].copy(),
    )
    x_shap = x_test if not x_test.empty else (x_valid if not x_valid.empty else x_train)

    return {
        "model": model,
        "feature_cols": feature_cols,
        "all_pred_df": all_pred_df,
        "test_pred_df": test_pred_df,
        "metrics_df": metrics_df,
        "x_shap": x_shap,
        "grid_search_info": grid_search_info,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "train_seconds": float(train_seconds),
    }


def _merge_prepare_stats(total: dict[str, Any], one: dict[str, Any]) -> None:
    total["cache_hit_pm25"] = bool(total.get("cache_hit_pm25", False) or one.get("cache_hit_pm25", False))
    total["cache_hit_era5"] = bool(total.get("cache_hit_era5", False) or one.get("cache_hit_era5", False))
    for key in ("pm25_seconds", "era5_seconds", "year_factor_seconds", "data_prepare_seconds"):
        total[key] = float(total.get(key, 0.0)) + float(one.get(key, 0.0))


def _build_split_frames_from_cache(
    feature_by_year: dict[int, pd.DataFrame],
    *,
    train_end_year: int,
    valid_year: int,
    test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_frames = [df for year, df in feature_by_year.items() if year <= train_end_year]
    valid_frames = [df for year, df in feature_by_year.items() if year == valid_year]
    test_frames = [df for year, df in feature_by_year.items() if year == test_year]
    train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    valid_df = pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame()
    test_df = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    return train_df, valid_df, test_df


def run_yearly_fallback_training_by_season(
    *,
    season_name: str,
    args: argparse.Namespace,
    seasonal_feature_by_year: dict[int, pd.DataFrame],
    available_years: list[int],
) -> dict[str, Any]:
    if LGBMRegressor is None:
        raise ImportError("lightgbm is not installed. Please run: pip install lightgbm")

    train_years = [year for year in available_years if year <= args.train_end_year]
    valid_years = [args.valid_year]
    test_years = [args.test_year]
    if not train_years:
        raise ValueError(f"Fallback mode has no training years for season {season_name}.")

    use_gpu = bool(getattr(args, "use_gpu", False))
    per_year_estimators = max(50, int(args.n_estimators // max(1, len(train_years))))
    base_params = dict(
        objective="regression",
        random_state=args.seed,
        n_jobs=1 if use_gpu else args.n_jobs,
    )
    if use_gpu:
        base_params["device"] = "gpu"
    train_params = {
        "n_estimators": per_year_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }

    feature_cols: list[str] | None = None
    fill_values: pd.Series | None = None
    model: Any = None

    year_iter: Any = train_years
    if tqdm is not None:
        year_iter = tqdm(train_years, desc=f"按年份训练 {season_name}", dynamic_ncols=True)

    train_begin = time.perf_counter()
    for year in year_iter:
        chunk_df = seasonal_feature_by_year.get(year, pd.DataFrame()).copy()
        chunk_df = chunk_df.loc[chunk_df["season_partition"] == season_name].copy()
        if chunk_df.empty:
            continue

        if feature_cols is None:
            non_feature_cols = {"date", "city", "cluster", "pm25", "season_partition"}
            candidate_cols = [col for col in chunk_df.columns if col not in non_feature_cols]
            feature_cols = [col for col in candidate_cols if pd.api.types.is_numeric_dtype(chunk_df[col])]
            feature_cols = [col for col in feature_cols if chunk_df[col].notna().any()]
            if not feature_cols:
                raise ValueError(f"Fallback mode found no usable numerical features for {season_name}.")

        x_raw = chunk_df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")
        chunk_fill = x_raw.median(numeric_only=True)
        fill_values = chunk_fill if fill_values is None else fill_values.combine_first(chunk_fill)
        x_chunk = x_raw.fillna(fill_values).fillna(0.0)
        y_chunk = chunk_df["pm25"].astype(float)

        if model is None:
            model = LGBMRegressor(**base_params, **train_params)
            model.fit(x_chunk, y_chunk)
        else:
            model.fit(x_chunk, y_chunk, init_model=model.booster_)

    if model is None or feature_cols is None or fill_values is None:
        raise ValueError(f"Fallback mode could not train a valid {season_name} model.")
    train_seconds = time.perf_counter() - train_begin

    def _predict_one_year(one_year: int, split: str) -> pd.DataFrame:
        one_df = seasonal_feature_by_year.get(one_year, pd.DataFrame()).copy()
        one_df = one_df.loc[one_df["season_partition"] == season_name].copy()
        if one_df.empty:
            return pd.DataFrame(columns=["date", "city", "cluster", "y_true", "y_pred", "split"])
        x_raw = one_df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")
        x_mat = x_raw.fillna(fill_values).fillna(0.0)
        pred = model.predict(x_mat)
        out = one_df[["date", "city", "cluster", "pm25"]].copy()
        out = out.rename(columns={"pm25": "y_true"})
        out["y_pred"] = pred
        out["split"] = split
        out["season_partition"] = season_name
        out["route_model"] = f"lightgbm_{season_name}"
        return out[["date", "city", "cluster", "y_true", "y_pred", "split", "season_partition", "route_model"]]

    train_frames = [one for year in train_years if not (one := _predict_one_year(year, "train")).empty]
    valid_frames = [one for year in valid_years if not (one := _predict_one_year(year, "valid")).empty]
    test_frames = [one for year in test_years if not (one := _predict_one_year(year, "test")).empty]
    train_pred_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    valid_pred_df = pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame()
    test_pred_df = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    all_pred_df = pd.concat([train_pred_df, valid_pred_df, test_pred_df], ignore_index=True)
    metrics_df = _metric_rows_from_predictions(
        train_pred_df=train_pred_df,
        valid_pred_df=valid_pred_df,
        test_pred_df=test_pred_df,
    )

    x_shap = pd.DataFrame(columns=feature_cols)
    test_frame = seasonal_feature_by_year.get(args.test_year, pd.DataFrame()).copy()
    test_frame = test_frame.loc[test_frame["season_partition"] == season_name].copy()
    if not test_frame.empty:
        x_shap = (
            test_frame.reindex(columns=feature_cols)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(fill_values)
            .fillna(0.0)
        )

    return {
        "model": model,
        "feature_cols": feature_cols,
        "all_pred_df": all_pred_df,
        "test_pred_df": test_pred_df,
        "metrics_df": metrics_df,
        "x_shap": x_shap,
        "grid_search_info": {"enabled": False},
        "train_rows": int(len(train_pred_df)),
        "valid_rows": int(len(valid_pred_df)),
        "test_rows": int(len(test_pred_df)),
        "train_seconds": float(train_seconds),
        "yearly_batches": list(train_years),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seasonal LightGBM for BTH PM2.5 regression.")
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
    parser.add_argument("--skip-era5", action="store_true", help="Skip merging ERA5 daily meteorological features.")
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
    parser.add_argument("--n-estimators", type=int, default=1200, help="Number of boosting trees.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Learning rate.")
    parser.add_argument("--num-leaves", type=int, default=31, help="Maximum tree leaves.")
    parser.add_argument("--subsample", type=float, default=0.8, help="Row sampling ratio.")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Column sampling ratio.")
    parser.add_argument("--early-stopping-rounds", type=int, default=50, help="Early stopping rounds.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for model fitting.")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for LightGBM (device=gpu).")
    parser.add_argument(
        "--enable-bayes-search",
        dest="enable_bayes_search",
        action="store_true",
        default=False,
        help="Enable Bayesian hyperparameter search.",
    )
    parser.add_argument(
        "--no-enable-bayes-search",
        dest="enable_bayes_search",
        action="store_false",
        help="Disable Bayesian search.",
    )
    parser.add_argument("--bayes-n-iter", type=int, default=50, help="Bayesian search iterations.")
    parser.add_argument("--cv-splits", type=int, default=5, help="TimeSeriesSplit folds.")
    parser.add_argument("--bayes-n-jobs", type=int, default=1, help="Bayesian search CV workers.")
    parser.add_argument(
        "--bayes-scoring",
        type=str,
        default="neg_root_mean_squared_error",
        help="Bayesian search scoring metric.",
    )
    parser.add_argument("--disable-shap", action="store_true", help="Disable SHAP explainability outputs.")
    parser.add_argument("--shap-max-samples", type=int, default=3000, help="Max rows used for SHAP.")
    parser.add_argument("--shap-max-display", type=int, default=20, help="Max displayed SHAP features.")
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
    parser.add_argument(
        "--winter-months",
        type=int,
        nargs="+",
        default=list(DEFAULT_WINTER_MONTHS),
        help="Months treated as winter, e.g. 11 12 1 2 3.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    winter_months = parse_winter_months(args.winter_months)
    correlation_dir = Path(args.correlation_dir).expanduser().resolve()
    data_read_dir = Path(args.data_read_dir).expanduser().resolve()
    pm25_day_dir = Path(args.pm25_day_dir).expanduser().resolve()
    era5_day_dir = Path(args.era5_day_dir).expanduser().resolve()
    city_geojson_dir = Path(args.city_geojson_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_pbar = tqdm(total=6, desc="BTH 季节双模型总进度", dynamic_ncols=True) if tqdm is not None else None
    full_begin = time.perf_counter()
    prepare_stats_main: dict[str, Any] = {}
    train_seconds_total = 0.0
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_enabled = not args.disable_data_cache
    prebuilt_year_factor_df: pd.DataFrame | None = None
    prebuilt_pm25_nc_index: dict[str, Any] | None = None
    training_mode = "full"
    fallback_triggered = False
    year_factor_cols: list[str] = []
    met_cols: list[str] = []
    season_results: dict[str, dict[str, Any]] = {}
    yearly_batches_by_season: dict[str, list[int]] = {}
    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    test_df = pd.DataFrame()

    if not args.daily_input:
        prebuilt_pm25_nc_index = build_pm25_nc_file_index(pm25_day_dir)

    try:
        base_df, year_factor_cols, met_cols = prepare_training_table(
            module_tag="lgb_bth_seasonal",
            correlation_dir=correlation_dir,
            data_read_dir=data_read_dir,
            city_geojson_dir=city_geojson_dir,
            daily_input=args.daily_input,
            pm25_day_dir=pm25_day_dir,
            era5_day_dir=era5_day_dir,
            include_era5_daily=not args.skip_era5,
            pm25_workers=args.pm25_workers,
            era5_workers=args.era5_workers,
            cache_dir=cache_dir,
            enable_cache=cache_enabled,
            prebuilt_year_factor_df=prebuilt_year_factor_df,
            prebuilt_pm25_nc_index=prebuilt_pm25_nc_index,
            prepare_stats=prepare_stats_main,
            use_year_factors=False,
            pm25_precropped=args.pm25_precropped,
        )
        if stage_pbar is not None:
            stage_pbar.update(1)
            stage_pbar.set_postfix_str("数据读取完成")

        feature_df = build_bth_feature_frame(base_df)
        feature_df = add_season_partition(feature_df, winter_months)
        train_df, valid_df, test_df = split_by_time(
            feature_df,
            train_end_year=args.train_end_year,
            valid_year=args.valid_year,
            test_year=args.test_year,
        )
        export_feature_quality_report(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            output_dir=output_dir,
            year_factor_cols=year_factor_cols,
            met_cols=met_cols,
        )
        if stage_pbar is not None:
            stage_pbar.update(1)
            stage_pbar.set_postfix_str("BTH 特征与季节划分完成")

        season_frames = split_frames_by_season(train_df, valid_df, test_df)
        validate_season_frames(season_frames)
        for season_name in SEASON_ORDER:
            season_results[season_name] = train_one_season_model(
                season_name=season_name,
                train_df=season_frames[season_name]["train"],
                valid_df=season_frames[season_name]["valid"],
                test_df=season_frames[season_name]["test"],
                args=args,
            )
            train_seconds_total += float(season_results[season_name]["train_seconds"])
        if stage_pbar is not None:
            stage_pbar.update(2)
            stage_pbar.set_postfix_str("季节双模型训练完成")
    except Exception as exc:
        if not is_memory_pressure_error(exc):
            if stage_pbar is not None:
                stage_pbar.close()
            raise

        print(f"[WARN] 全量读取/训练触发内存压力，切换到按年份季节降级模式: {exc}")
        fallback_triggered = True
        training_mode = "yearly_fallback"
        fallback_available_years = discover_pm25_years(pm25_day_dir, pm25_nc_index=prebuilt_pm25_nc_index)
        fallback_years_needed = sorted(
            {
                year
                for year in fallback_available_years
                if year <= args.train_end_year or year == args.valid_year or year == args.test_year
            }
        )
        if not fallback_years_needed:
            raise ValueError("Fallback mode has no available years for train/valid/test.")

        (
            fallback_feature_by_year,
            fallback_prepare_stats,
            fallback_year_factor_cols,
            fallback_met_cols,
        ) = build_fallback_feature_cache(
            args=args,
            correlation_dir=correlation_dir,
            data_read_dir=data_read_dir,
            city_geojson_dir=city_geojson_dir,
            pm25_day_dir=pm25_day_dir,
            era5_day_dir=era5_day_dir,
            years=fallback_years_needed,
            pm25_nc_index=prebuilt_pm25_nc_index,
            prebuilt_year_factor_df=prebuilt_year_factor_df,
        )
        _merge_prepare_stats(prepare_stats_main, fallback_prepare_stats)
        year_factor_cols = fallback_year_factor_cols
        met_cols = fallback_met_cols
        seasonal_feature_by_year = {
            year: add_season_partition(frame, winter_months)
            for year, frame in fallback_feature_by_year.items()
        }
        train_df, valid_df, test_df = _build_split_frames_from_cache(
            seasonal_feature_by_year,
            train_end_year=args.train_end_year,
            valid_year=args.valid_year,
            test_year=args.test_year,
        )
        export_feature_quality_report(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            output_dir=output_dir,
            year_factor_cols=year_factor_cols,
            met_cols=met_cols,
        )

        for season_name in SEASON_ORDER:
            season_results[season_name] = run_yearly_fallback_training_by_season(
                season_name=season_name,
                args=args,
                seasonal_feature_by_year=seasonal_feature_by_year,
                available_years=fallback_available_years,
            )
            yearly_batches_by_season[season_name] = list(season_results[season_name].get("yearly_batches", []))
            train_seconds_total += float(season_results[season_name]["train_seconds"])
        if stage_pbar is not None:
            target_stage = 4
            if stage_pbar.n < target_stage:
                stage_pbar.update(target_stage - stage_pbar.n)
            stage_pbar.set_postfix_str("按年份季节训练完成")

    if not season_results:
        if stage_pbar is not None:
            stage_pbar.close()
        raise ValueError("No seasonal model was successfully trained.")

    all_pred_df = pd.concat(
        [season_results[season_name]["all_pred_df"] for season_name in SEASON_ORDER],
        ignore_index=True,
    ).sort_values(["split", "date", "city"], kind="mergesort")
    test_pred_df = pd.concat(
        [season_results[season_name]["test_pred_df"] for season_name in SEASON_ORDER],
        ignore_index=True,
    ).sort_values(["date", "city"], kind="mergesort")
    metrics_by_season_df = pd.concat(
        [season_results[season_name]["metrics_df"].assign(season=season_name) for season_name in SEASON_ORDER],
        ignore_index=True,
    )

    overall_rows: list[dict[str, Any]] = []
    for split_name in ("train", "valid", "test"):
        split_df = all_pred_df.loc[all_pred_df["split"] == split_name].copy()
        if split_df.empty:
            overall_rows.append(
                {"split": split_name, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n_samples": 0}
            )
            continue
        split_metrics = compute_metrics(split_df["y_true"], split_df["y_pred"].to_numpy())
        overall_rows.append({"split": split_name, **split_metrics, "n_samples": int(len(split_df))})
    metrics_df = pd.DataFrame(overall_rows)

    generalization_df = export_generalization_artifacts(metrics_df, output_dir)
    export_regression_artifacts(
        all_pred_df=all_pred_df,
        output_dir=output_dir,
        model_name="LightGBM-BTH-SeasonalRouter",
    )
    if stage_pbar is not None:
        stage_pbar.update(1)
        stage_pbar.set_postfix_str("总体评估完成")

    per_season_run_info: dict[str, Any] = {}
    shap_status_by_season: dict[str, str] = {}
    for season_name in SEASON_ORDER:
        result = season_results[season_name]
        season_output_dir = output_dir / season_name
        season_output_dir.mkdir(parents=True, exist_ok=True)
        season_metrics_df = result["metrics_df"]
        season_all_pred_df = result["all_pred_df"]
        season_test_pred_df = result["test_pred_df"]
        season_generalization_df = export_generalization_artifacts(season_metrics_df, season_output_dir)
        export_regression_artifacts(
            all_pred_df=season_all_pred_df,
            output_dir=season_output_dir,
            model_name=f"LightGBM-BTH-{season_name}",
        )

        shap_status = "disabled"
        if not args.disable_shap:
            try:
                export_shap_artifacts(
                    model=result["model"],
                    x_for_shap=result["x_shap"],
                    output_dir=season_output_dir,
                    model_name=f"LightGBM-BTH-{season_name}",
                    shap_max_samples=args.shap_max_samples,
                    shap_max_display=args.shap_max_display,
                    random_state=args.seed,
                )
                shap_status = "ok"
            except Exception as exc:  # pragma: no cover  # pylint: disable=broad-except
                shap_status = f"failed: {exc}"
                print(f"[WARN] SHAP export failed for {season_name}: {exc}")
        shap_status_by_season[season_name] = shap_status

        importance_df = pd.DataFrame(
            {
                "feature": result["feature_cols"],
                "importance": result["model"].feature_importances_,
            }
        ).sort_values("importance", ascending=False, kind="mergesort")

        season_metrics_df.to_csv(season_output_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")
        season_test_pred_df.sort_values(["date", "city"], kind="mergesort").to_csv(
            season_output_dir / "predictions_test.csv",
            index=False,
            encoding="utf-8-sig",
        )
        season_all_pred_df.sort_values(["split", "date", "city"], kind="mergesort").to_csv(
            season_output_dir / "predictions_all_splits.csv",
            index=False,
            encoding="utf-8-sig",
        )
        importance_df.to_csv(season_output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
        importance_df.to_csv(output_dir / f"feature_importance_{season_name}.csv", index=False, encoding="utf-8-sig")
        if result["grid_search_info"].get("enabled") and isinstance(result["grid_search_info"].get("results_df"), pd.DataFrame):
            gdf = result["grid_search_info"]["results_df"]
            keep_cols = [c for c in ["rank_test_score", "mean_test_score", "std_test_score", "params"] if c in gdf.columns]
            gdf.loc[:, keep_cols].to_csv(season_output_dir / "bayes_search_results.csv", index=False, encoding="utf-8-sig")

        per_season_run_info[season_name] = {
            "output_dir": str(season_output_dir),
            "n_features": int(len(result["feature_cols"])),
            "train_rows": int(result["train_rows"]),
            "valid_rows": int(result["valid_rows"]),
            "test_rows": int(result["test_rows"]),
            "train_seconds": float(result["train_seconds"]),
            "generalization_level": str(season_generalization_df.loc[0, "generalization_level"]) if not season_generalization_df.empty else "",
            "shap_status": shap_status,
            "bayes_search_enabled": bool(result["grid_search_info"].get("enabled", False)),
            "bayes_search_best_score": result["grid_search_info"].get("best_score"),
            "bayes_search_best_params": result["grid_search_info"].get("best_params", {}),
            "yearly_batches": yearly_batches_by_season.get(season_name, []),
        }

    metrics_df.to_csv(output_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")
    metrics_by_season_df.to_csv(output_dir / "metrics_by_season.csv", index=False, encoding="utf-8-sig")
    test_pred_df.to_csv(output_dir / "predictions_test.csv", index=False, encoding="utf-8-sig")
    all_pred_df.to_csv(output_dir / "predictions_all_splits.csv", index=False, encoding="utf-8-sig")

    run_info = {
        "model": "LightGBM",
        "cluster": "BTH",
        "training_granularity": "season_router",
        "season_mode": "dual_router",
        "winter_months": winter_months,
        "train_end_year": args.train_end_year,
        "valid_year": args.valid_year,
        "test_year": args.test_year,
        "n_features_max": int(max(info["n_features"] for info in per_season_run_info.values())),
        "n_year_factor_features": len(year_factor_cols),
        "n_era5_daily_features": len(met_cols),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "daily_input": args.daily_input or [],
        "pm25_day_dir": str(pm25_day_dir),
        "era5_day_dir": str(era5_day_dir),
        "output_dir": str(output_dir),
        "generalization_level": str(generalization_df.loc[0, "generalization_level"]) if not generalization_df.empty else "",
        "season_models": per_season_run_info,
        "shap_status": shap_status_by_season,
        "enable_bayes_search": bool(getattr(args, "enable_bayes_search", False)),
        "use_gpu": bool(getattr(args, "use_gpu", False)),
        "bayes_n_iter": int(getattr(args, "bayes_n_iter", 50)),
        "training_mode": training_mode,
        "fallback_triggered": fallback_triggered,
        "yearly_batches_by_season": yearly_batches_by_season,
        "pm25_workers": int(args.pm25_workers),
        "era5_workers": int(args.era5_workers),
        "cache_dir": str(cache_dir),
        "cache_enabled": bool(cache_enabled),
        "cache_hit_pm25": bool(prepare_stats_main.get("cache_hit_pm25", False)),
        "cache_hit_era5": bool(prepare_stats_main.get("cache_hit_era5", False)),
        "data_prepare_seconds": float(prepare_stats_main.get("data_prepare_seconds", 0.0)),
        "pm25_read_seconds": float(prepare_stats_main.get("pm25_seconds", 0.0)),
        "era5_read_seconds": float(prepare_stats_main.get("era5_seconds", 0.0)),
        "year_factor_seconds": float(prepare_stats_main.get("year_factor_seconds", 0.0)),
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
    print("[INFO] Seasonal LightGBM for BTH finished.")
    print(f"[INFO] Output directory: {output_dir}")
    print("[INFO] Files:")
    print("       - metrics_overall.csv")
    print("       - metrics_by_season.csv")
    print("       - predictions_test.csv")
    print("       - predictions_all_splits.csv")
    print("       - feature_importance_winter.csv")
    print("       - feature_importance_non_winter.csv")
    print("       - generalization_assessment.csv")
    print("       - generalization_plot_data.csv")
    print("       - regression_all_splits_data.csv")
    print("       - regression_test_data.csv")
    print("       - run_info.json")
    print("       - [winter|non_winter]/metrics_overall.csv")
    print("       - [winter|non_winter]/feature_importance.csv")
    print("       - [winter|non_winter]/shap_*")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
