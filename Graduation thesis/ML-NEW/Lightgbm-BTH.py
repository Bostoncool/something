from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

try:
    from skopt import BayesSearchCV  # type: ignore[import-untyped]
    from skopt.space import Integer, Real  # type: ignore[import-untyped]
except ImportError as _e:
    BayesSearchCV = None  # type: ignore[misc, assignment]
    Integer = Real = None  # type: ignore[misc, assignment]
    _skopt_import_error = _e

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
    build_daily_features,
    build_model_matrices,
    build_pm25_nc_file_index,
    build_prediction_frames,
    compute_metrics,
    export_feature_quality_report,
    export_generalization_artifacts,
    export_regression_artifacts,
    export_shap_artifacts,
    infer_year_from_filename,
    infer_year_from_nc_file,
    prepare_training_table,
    split_by_time,
)

try:
    from lightgbm import LGBMRegressor, early_stopping
except ImportError:  # pragma: no cover
    LGBMRegressor = None
    early_stopping = None


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "lightgbm_bth_daily_pm25"
BTH_CLUSTER_NAME = "BTH"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover  # pylint: disable=broad-except
        pass


def _log_bs_msg(msg: str) -> None:
    if tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def _bayesian_search_progress_callback(n_iter: int, start_time: float):
    def callback(res: Any) -> None:
        func_vals = getattr(res, "func_vals", None) or []
        cur = len(func_vals)
        elapsed = time.perf_counter() - start_time
        eta_s = (elapsed / cur) * (n_iter - cur) if cur > 0 else 0
        pct = 100.0 * cur / n_iter if n_iter else 0
        best_score = float(np.min(func_vals)) if func_vals else float("nan")
        _log_bs_msg(
            f"[BayesSearch] 进度 {cur}/{n_iter} ({pct:.1f}%) "
            f"当前最佳得分 {best_score:.6f} 已用 {elapsed:.0f}s 预计剩余 {eta_s:.0f}s"
        )

    return callback


def _native_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if hasattr(value, "item"):
            out[key] = value.item()
        elif isinstance(value, (np.integer, np.int64)):
            out[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def train_lightgbm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    if LGBMRegressor is None:
        raise ImportError("lightgbm is not installed. Please run: pip install lightgbm")

    use_gpu = getattr(args, "use_gpu", False)
    base_params = dict(
        objective="regression",
        random_state=args.seed,
        n_jobs=1 if use_gpu else args.n_jobs,
    )
    if use_gpu:
        base_params["device"] = "gpu"
    train_params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }

    if getattr(args, "enable_bayes_search", False):
        if BayesSearchCV is None or Real is None or Integer is None:
            raise ImportError(
                "贝叶斯搜索需要 scikit-optimize，请安装: pip install scikit-optimize"
            ) from getattr(sys.modules.get(__name__), "_skopt_import_error", None)

        search_spaces = {
            "n_estimators": Integer(500, 2000),
            "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
            "num_leaves": Integer(15, 127),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
        }
        cv_splits = getattr(args, "cv_splits", 5)
        n_iter = getattr(args, "bayes_n_iter", 50)
        start_time = time.perf_counter()
        start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        _log_bs_msg(
            f"[BayesSearch] 迭代数 {n_iter}, CV 折数 {cv_splits}, GPU={use_gpu}, 开始时间 {start_str}"
        )
        _log_bs_msg("[BayesSearch] 每组参数先跑时序 CV，再汇报当前最优结果。")
        base_est = LGBMRegressor(
            **base_params,
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        search = BayesSearchCV(
            estimator=base_est,
            search_spaces=search_spaces,
            n_iter=n_iter,
            scoring=getattr(args, "bayes_scoring", "neg_root_mean_squared_error"),
            cv=TimeSeriesSplit(n_splits=cv_splits),
            n_jobs=getattr(args, "bayes_n_jobs", 1),
            refit=True,
            random_state=args.seed,
            verbose=0,
        )
        progress_cb = _bayesian_search_progress_callback(n_iter, start_time)
        search.fit(x_train, y_train, callback=progress_cb)
        elapsed = time.perf_counter() - start_time
        best_params = _native_params(search.best_params_)
        _log_bs_msg(
            f"[BayesSearch] 完成 总耗时 {elapsed:.1f}s 最佳得分 {search.best_score_} 最佳参数 {best_params}"
        )
        return search.best_estimator_, {
            "enabled": True,
            "search_method": "bayesian",
            "best_params": best_params,
            "best_score": float(search.best_score_),
            "cv_splits": cv_splits,
            "n_iter": n_iter,
            "results_df": pd.DataFrame(search.cv_results_).sort_values("rank_test_score", kind="mergesort"),
            "use_gpu": use_gpu,
        }

    model = LGBMRegressor(**base_params, **train_params)
    fit_kwargs: dict[str, Any] = {}
    if not x_valid.empty and args.early_stopping_rounds > 0:
        fit_kwargs["eval_set"] = [(x_valid, y_valid)]
        fit_kwargs["eval_metric"] = "l2"
        if early_stopping is not None:
            fit_kwargs["callbacks"] = [early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False)]
    model.fit(x_train, y_train, **fit_kwargs)
    return model, {"enabled": False}


def is_memory_pressure_error(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    memory_tokens = [
        "out of memory",
        "cannot allocate memory",
        "unable to allocate",
        "std::bad_alloc",
        "cuda out of memory",
    ]
    return any(token in text for token in memory_tokens)


def discover_pm25_years(pm25_day_dir: Path, pm25_nc_index: dict[str, Any] | None = None) -> list[int]:
    if pm25_nc_index is not None:
        year_to_files = pm25_nc_index.get("year_to_files", {})
        return sorted(int(year) for year in year_to_files.keys())
    years: set[int] = set()
    for nc_file in pm25_day_dir.rglob("*.nc"):
        if not nc_file.is_file():
            continue
        year_val = infer_year_from_filename(nc_file.name)
        if year_val is None:
            year_val = infer_year_from_nc_file(nc_file)
        if year_val is None:
            match = re.search(r"(20\d{2})", nc_file.name)
            if match:
                year_val = int(match.group(1))
        if year_val is not None:
            years.add(int(year_val))
    return sorted(years)


def filter_to_bth(frame: pd.DataFrame) -> pd.DataFrame:
    if "cluster" not in frame.columns:
        raise ValueError("Expected 'cluster' column in prepared data.")
    out = frame.loc[frame["cluster"].astype(str) == BTH_CLUSTER_NAME].copy()
    if out.empty:
        raise ValueError("No BTH rows found after cluster filtering.")
    return out


def add_bth_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["city", "date"], kind="mergesort").reset_index(drop=True).copy()

    u_col = "met_10m_u_component_of_wind"
    v_col = "met_10m_v_component_of_wind"
    t_col = "met_2m_temperature"
    d_col = "met_2m_dewpoint_temperature"
    p_col = "met_mean_sea_level_pressure"
    tp_col = "met_total_precipitation"

    if u_col in out.columns and v_col in out.columns:
        u = pd.to_numeric(out[u_col], errors="coerce")
        v = pd.to_numeric(out[v_col], errors="coerce")
        out["wind_speed"] = np.sqrt(np.square(u) + np.square(v))
        wind_dir = np.arctan2(v, u)
        out["wind_dir_sin"] = np.sin(wind_dir)
        out["wind_dir_cos"] = np.cos(wind_dir)
    else:
        out["wind_speed"] = np.nan
        out["wind_dir_sin"] = np.nan
        out["wind_dir_cos"] = np.nan

    if t_col in out.columns and d_col in out.columns:
        t2m = pd.to_numeric(out[t_col], errors="coerce")
        d2m = pd.to_numeric(out[d_col], errors="coerce")
        out["dewpoint_depression"] = t2m - d2m
    else:
        out["dewpoint_depression"] = np.nan

    grouped = out.groupby("city", observed=True, sort=False)
    if p_col in out.columns:
        out["pressure_change_1d"] = grouped[p_col].diff()
    else:
        out["pressure_change_1d"] = np.nan

    if "wind_speed" in out.columns:
        out["wind_speed_3d_mean"] = grouped["wind_speed"].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window=3, min_periods=1).mean()
        )
    else:
        out["wind_speed_3d_mean"] = np.nan

    if tp_col in out.columns:
        out["precip_3d_sum"] = grouped[tp_col].transform(
            lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(window=3, min_periods=1).sum()
        )
    else:
        out["precip_3d_sum"] = np.nan

    heating_months = {11, 12, 1, 2, 3}
    cold_halfyear_months = {10, 11, 12, 1, 2, 3}
    out["is_heating_season"] = out["month"].isin(heating_months).astype(int)
    out["cold_halfyear"] = out["month"].isin(cold_halfyear_months).astype(int)

    wind_ok = pd.to_numeric(out["wind_speed_3d_mean"], errors="coerce").le(2.5)
    pressure_ok = pd.to_numeric(out["pressure_change_1d"], errors="coerce").ge(0.0)
    precip_ok = pd.to_numeric(out["precip_3d_sum"], errors="coerce").fillna(0.0).le(1.0)
    out["stagnation_flag"] = (wind_ok & pressure_ok.fillna(False) & precip_ok).astype(int)

    lag_1 = pd.to_numeric(out.get("lag_1"), errors="coerce")
    out["lag_1_x_heating"] = lag_1 * out["is_heating_season"]
    out["lag_1_x_stagnation"] = lag_1 * out["stagnation_flag"]

    return out


def build_bth_feature_frame(base_df: pd.DataFrame) -> pd.DataFrame:
    bth_df = filter_to_bth(base_df)
    feature_df = build_daily_features(bth_df)
    return add_bth_features(feature_df)


def build_fallback_feature_cache(
    *,
    args: argparse.Namespace,
    correlation_dir: Path,
    data_read_dir: Path,
    city_geojson_dir: Path,
    pm25_day_dir: Path,
    era5_day_dir: Path,
    years: list[int],
    pm25_nc_index: dict[str, Any] | None = None,
    prebuilt_year_factor_df: pd.DataFrame | None = None,
) -> tuple[dict[int, pd.DataFrame], dict[str, Any], list[str], list[str]]:
    feature_by_year: dict[int, pd.DataFrame] = {}
    prepare_acc = {
        "cache_hit_pm25": False,
        "cache_hit_era5": False,
        "pm25_seconds": 0.0,
        "era5_seconds": 0.0,
        "year_factor_seconds": 0.0,
        "data_prepare_seconds": 0.0,
    }
    year_factor_col_set: set[str] = set()
    met_col_set: set[str] = set()

    year_iter: Any = years
    if tqdm is not None:
        year_iter = tqdm(years, desc="预加载 BTH 年份特征(降级模式)", dynamic_ncols=True)

    for year in year_iter:
        prepare_stats_year: dict[str, Any] = {}
        base_df, year_factor_cols, met_cols = prepare_training_table(
            module_tag="lgb_bth",
            correlation_dir=correlation_dir,
            data_read_dir=data_read_dir,
            city_geojson_dir=city_geojson_dir,
            daily_input=args.daily_input,
            pm25_day_dir=pm25_day_dir,
            era5_day_dir=era5_day_dir,
            include_era5_daily=not args.skip_era5,
            allowed_years=[year],
            pm25_workers=args.pm25_workers,
            era5_workers=args.era5_workers,
            cache_dir=Path(args.cache_dir),
            enable_cache=not args.disable_data_cache,
            prebuilt_year_factor_df=prebuilt_year_factor_df,
            prebuilt_pm25_nc_index=pm25_nc_index,
            prepare_stats=prepare_stats_year,
            use_year_factors=False,
            pm25_precropped=args.pm25_precropped,
        )
        prepare_acc["cache_hit_pm25"] = prepare_acc["cache_hit_pm25"] or bool(prepare_stats_year.get("cache_hit_pm25", False))
        prepare_acc["cache_hit_era5"] = prepare_acc["cache_hit_era5"] or bool(prepare_stats_year.get("cache_hit_era5", False))
        prepare_acc["pm25_seconds"] += float(prepare_stats_year.get("pm25_seconds", 0.0))
        prepare_acc["era5_seconds"] += float(prepare_stats_year.get("era5_seconds", 0.0))
        prepare_acc["year_factor_seconds"] += float(prepare_stats_year.get("year_factor_seconds", 0.0))
        prepare_acc["data_prepare_seconds"] += float(prepare_stats_year.get("data_prepare_seconds", 0.0))
        year_factor_col_set.update(year_factor_cols)
        met_col_set.update(met_cols)
        feature_df = build_bth_feature_frame(base_df)
        feature_by_year[year] = feature_df.loc[feature_df["year"] == year].copy()

    return feature_by_year, prepare_acc, sorted(year_factor_col_set), sorted(met_col_set)


def run_yearly_fallback_training(
    *,
    args: argparse.Namespace,
    correlation_dir: Path,
    data_read_dir: Path,
    city_geojson_dir: Path,
    pm25_day_dir: Path,
    era5_day_dir: Path,
    prebuilt_year_factor_df: pd.DataFrame | None = None,
    prebuilt_pm25_nc_index: dict[str, Any] | None = None,
    prebuilt_feature_by_year: dict[int, pd.DataFrame] | None = None,
    prebuilt_available_years: list[int] | None = None,
    prebuilt_year_factor_cols: list[str] | None = None,
    prebuilt_met_cols: list[str] | None = None,
) -> dict[str, Any]:
    if LGBMRegressor is None:
        raise ImportError("lightgbm is not installed. Please run: pip install lightgbm")

    available_years = (
        list(prebuilt_available_years)
        if prebuilt_available_years is not None
        else discover_pm25_years(pm25_day_dir, pm25_nc_index=prebuilt_pm25_nc_index)
    )
    if not available_years:
        raise ValueError("Fallback mode cannot detect years from PM2.5 NC filenames.")

    train_years = [year for year in available_years if year <= args.train_end_year]
    valid_years = [args.valid_year]
    test_years = [args.test_year]
    if not train_years:
        raise ValueError("Fallback mode has no available training years.")

    print(f"[INFO] Fallback cluster: {BTH_CLUSTER_NAME}")
    print(f"[INFO] Fallback train years: {train_years}")
    print(f"[INFO] Fallback valid/test years: {valid_years + test_years}")

    per_year_estimators = max(50, int(args.n_estimators // max(1, len(train_years))))
    base_params = dict(
        objective="regression",
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
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
    year_factor_col_set: set[str] = set(prebuilt_year_factor_cols or [])
    met_col_set: set[str] = set(prebuilt_met_cols or [])
    prepare_acc = {
        "cache_hit_pm25": False,
        "cache_hit_era5": False,
        "pm25_seconds": 0.0,
        "era5_seconds": 0.0,
        "year_factor_seconds": 0.0,
        "data_prepare_seconds": 0.0,
    }

    year_iter: Any = train_years
    if tqdm is not None:
        year_iter = tqdm(train_years, desc="按年份训练 BTH(降级模式)", dynamic_ncols=True)

    train_begin = time.perf_counter()
    for year in year_iter:
        if prebuilt_feature_by_year is not None and year in prebuilt_feature_by_year:
            chunk_df = prebuilt_feature_by_year[year].copy()
        else:
            prepare_stats_year: dict[str, Any] = {}
            base_df, year_factor_cols, met_cols = prepare_training_table(
                module_tag="lgb_bth",
                correlation_dir=correlation_dir,
                data_read_dir=data_read_dir,
                city_geojson_dir=city_geojson_dir,
                daily_input=args.daily_input,
                pm25_day_dir=pm25_day_dir,
                era5_day_dir=era5_day_dir,
                include_era5_daily=not args.skip_era5,
                allowed_years=[year],
                pm25_workers=args.pm25_workers,
                era5_workers=args.era5_workers,
                cache_dir=Path(args.cache_dir),
                enable_cache=not args.disable_data_cache,
                prebuilt_year_factor_df=prebuilt_year_factor_df,
                prebuilt_pm25_nc_index=prebuilt_pm25_nc_index,
                prepare_stats=prepare_stats_year,
                use_year_factors=False,
                pm25_precropped=args.pm25_precropped,
            )
            prepare_acc["cache_hit_pm25"] = prepare_acc["cache_hit_pm25"] or bool(prepare_stats_year.get("cache_hit_pm25", False))
            prepare_acc["cache_hit_era5"] = prepare_acc["cache_hit_era5"] or bool(prepare_stats_year.get("cache_hit_era5", False))
            prepare_acc["pm25_seconds"] += float(prepare_stats_year.get("pm25_seconds", 0.0))
            prepare_acc["era5_seconds"] += float(prepare_stats_year.get("era5_seconds", 0.0))
            prepare_acc["year_factor_seconds"] += float(prepare_stats_year.get("year_factor_seconds", 0.0))
            prepare_acc["data_prepare_seconds"] += float(prepare_stats_year.get("data_prepare_seconds", 0.0))
            year_factor_col_set.update(year_factor_cols)
            met_col_set.update(met_cols)
            chunk_df = build_bth_feature_frame(base_df)
            chunk_df = chunk_df.loc[chunk_df["year"] == year].copy()

        if chunk_df.empty:
            continue

        if feature_cols is None:
            non_feature_cols = {"date", "city", "cluster", "pm25"}
            candidate_cols = [col for col in chunk_df.columns if col not in non_feature_cols]
            feature_cols = [col for col in candidate_cols if pd.api.types.is_numeric_dtype(chunk_df[col])]
            feature_cols = [col for col in feature_cols if chunk_df[col].notna().any()]
            if not feature_cols:
                raise ValueError("Fallback mode found no usable numerical features.")

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
        raise ValueError("Fallback mode could not train a valid BTH model.")
    train_seconds = time.perf_counter() - train_begin

    def _predict_one_year(one_year: int, split: str) -> pd.DataFrame:
        if prebuilt_feature_by_year is not None and one_year in prebuilt_feature_by_year:
            one_df = prebuilt_feature_by_year[one_year].copy()
        else:
            base_df, _, _ = prepare_training_table(
                module_tag="lgb_bth",
                correlation_dir=correlation_dir,
                data_read_dir=data_read_dir,
                city_geojson_dir=city_geojson_dir,
                daily_input=args.daily_input,
                pm25_day_dir=pm25_day_dir,
                era5_day_dir=era5_day_dir,
                include_era5_daily=not args.skip_era5,
                allowed_years=[one_year],
                pm25_workers=args.pm25_workers,
                era5_workers=args.era5_workers,
                cache_dir=Path(args.cache_dir),
                enable_cache=not args.disable_data_cache,
                prebuilt_year_factor_df=prebuilt_year_factor_df,
                prebuilt_pm25_nc_index=prebuilt_pm25_nc_index,
                use_year_factors=False,
                pm25_precropped=args.pm25_precropped,
            )
            one_df = build_bth_feature_frame(base_df)
            one_df = one_df.loc[one_df["year"] == one_year].copy()

        if one_df.empty:
            return pd.DataFrame(columns=["date", "city", "cluster", "y_true", "y_pred", "split"])

        x_raw = one_df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")
        x_mat = x_raw.fillna(fill_values).fillna(0.0)
        pred = model.predict(x_mat)
        out = one_df[["date", "city", "cluster", "pm25"]].copy()
        out = out.rename(columns={"pm25": "y_true"})
        out["y_pred"] = pred
        out["split"] = split
        return out[["date", "city", "cluster", "y_true", "y_pred", "split"]]

    train_frames = [one for year in train_years if not (one := _predict_one_year(year, "train")).empty]
    valid_frames = [one for year in valid_years if not (one := _predict_one_year(year, "valid")).empty]
    test_frames = [one for year in test_years if not (one := _predict_one_year(year, "test")).empty]

    train_pred_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame(
        columns=["date", "city", "cluster", "y_true", "y_pred", "split"]
    )
    valid_pred_df = pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame(
        columns=["date", "city", "cluster", "y_true", "y_pred", "split"]
    )
    test_pred_df = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame(
        columns=["date", "city", "cluster", "y_true", "y_pred", "split"]
    )
    all_pred_df = pd.concat([train_pred_df, valid_pred_df, test_pred_df], ignore_index=True)

    def _metric_row(split: str, pred_df: pd.DataFrame) -> dict[str, Any]:
        if pred_df.empty:
            return {"split": split, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n_samples": 0}
        metrics = compute_metrics(pred_df["y_true"], pred_df["y_pred"].to_numpy())
        return {"split": split, **metrics, "n_samples": int(len(pred_df))}

    metrics_df = pd.DataFrame(
        [
            _metric_row("train", train_pred_df),
            _metric_row("valid", valid_pred_df),
            _metric_row("test", test_pred_df),
        ]
    )

    x_shap = pd.DataFrame(columns=feature_cols)
    if not test_pred_df.empty:
        if prebuilt_feature_by_year is not None and args.test_year in prebuilt_feature_by_year:
            test_frame = prebuilt_feature_by_year[args.test_year].copy()
        else:
            base_df_test, _, _ = prepare_training_table(
                module_tag="lgb_bth",
                correlation_dir=correlation_dir,
                data_read_dir=data_read_dir,
                city_geojson_dir=city_geojson_dir,
                daily_input=args.daily_input,
                pm25_day_dir=pm25_day_dir,
                era5_day_dir=era5_day_dir,
                include_era5_daily=not args.skip_era5,
                allowed_years=[args.test_year],
                pm25_workers=args.pm25_workers,
                era5_workers=args.era5_workers,
                cache_dir=Path(args.cache_dir),
                enable_cache=not args.disable_data_cache,
                prebuilt_year_factor_df=prebuilt_year_factor_df,
                prebuilt_pm25_nc_index=prebuilt_pm25_nc_index,
                use_year_factors=False,
                pm25_precropped=args.pm25_precropped,
            )
            test_frame = build_bth_feature_frame(base_df_test)
            test_frame = test_frame.loc[test_frame["year"] == args.test_year].copy()
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
        "year_factor_cols": sorted(year_factor_col_set),
        "met_cols": sorted(met_col_set),
        "train_rows": int(len(train_pred_df)),
        "valid_rows": int(len(valid_pred_df)),
        "test_rows": int(len(test_pred_df)),
        "yearly_batches": train_years,
        "prepare_stats": prepare_acc,
        "train_seconds": float(train_seconds),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily PM2.5 prediction for BTH using LightGBM.")
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
    parser.add_argument("--bayes-n-iter", type=int, default=50, help="贝叶斯搜索迭代次数.")
    parser.add_argument("--cv-splits", type=int, default=5, help="TimeSeriesSplit 折数.")
    parser.add_argument("--bayes-n-jobs", type=int, default=1, help="贝叶斯搜索 CV 并行数.")
    parser.add_argument(
        "--bayes-scoring",
        type=str,
        default="neg_root_mean_squared_error",
        help="贝叶斯搜索 scoring 指标.",
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
    return parser


def _merge_prepare_stats(total: dict[str, Any], one: dict[str, Any]) -> None:
    total["cache_hit_pm25"] = bool(total.get("cache_hit_pm25", False) or one.get("cache_hit_pm25", False))
    total["cache_hit_era5"] = bool(total.get("cache_hit_era5", False) or one.get("cache_hit_era5", False))
    for key in ("pm25_seconds", "era5_seconds", "year_factor_seconds", "data_prepare_seconds"):
        total[key] = float(total.get(key, 0.0)) + float(one.get(key, 0.0))


def _build_bth_split_frames_from_cache(
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


def main() -> int:
    args = build_parser().parse_args()
    correlation_dir = Path(args.correlation_dir).expanduser().resolve()
    data_read_dir = Path(args.data_read_dir).expanduser().resolve()
    pm25_day_dir = Path(args.pm25_day_dir).expanduser().resolve()
    era5_day_dir = Path(args.era5_day_dir).expanduser().resolve()
    city_geojson_dir = Path(args.city_geojson_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_pbar = tqdm(total=6, desc="LightGBM-BTH 总进度", dynamic_ncols=True) if tqdm is not None else None
    prepare_stats_main: dict[str, Any] = {}
    full_begin = time.perf_counter()
    train_seconds_total = 0.0
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_enabled = not args.disable_data_cache
    prebuilt_year_factor_df: pd.DataFrame | None = None
    prebuilt_pm25_nc_index: dict[str, Any] | None = None
    training_mode = "full"
    fallback_triggered = False
    yearly_batches: list[int] = []
    year_factor_cols: list[str] = []
    met_cols: list[str] = []
    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    test_df = pd.DataFrame()

    if not args.daily_input:
        prebuilt_pm25_nc_index = build_pm25_nc_file_index(pm25_day_dir)

    try:
        base_df, year_factor_cols, met_cols = prepare_training_table(
            module_tag="lgb_bth",
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
        train_df, valid_df, test_df = split_by_time(
            feature_df,
            train_end_year=args.train_end_year,
            valid_year=args.valid_year,
            test_year=args.test_year,
        )
        if stage_pbar is not None:
            stage_pbar.update(1)
            stage_pbar.set_postfix_str("BTH 特征构建完成")

        export_feature_quality_report(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            output_dir=output_dir,
            year_factor_cols=year_factor_cols,
            met_cols=met_cols,
        )

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
                print(f"[WARN] BTH 贝叶斯搜索失败，回退默认参数训练: {exc}")
                old_flag = args.enable_bayes_search
                args.enable_bayes_search = False
                try:
                    model, grid_search_info = train_lightgbm(x_train, y_train, x_valid, y_valid, args)
                finally:
                    args.enable_bayes_search = old_flag
                grid_search_info["bayes_fallback_reason"] = str(exc)
            else:
                raise
        train_seconds_total += time.perf_counter() - train_begin

        pred_train = model.predict(x_train)
        pred_valid = model.predict(x_valid)
        pred_test = model.predict(x_test)
        metrics_df = pd.DataFrame(
            [
                {"split": "train", **compute_metrics(y_train, pred_train), "n_samples": int(len(y_train))},
                {"split": "valid", **compute_metrics(y_valid, pred_valid), "n_samples": int(len(y_valid))},
                {"split": "test", **compute_metrics(y_test, pred_test), "n_samples": int(len(y_test))},
            ]
        )
        all_pred_df, test_pred_df = build_prediction_frames(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            pred_train=pred_train,
            pred_valid=pred_valid,
            pred_test=pred_test,
        )
        x_shap = x_test if not x_test.empty else (x_valid if not x_valid.empty else x_train)
        if stage_pbar is not None:
            stage_pbar.update(2)
            stage_pbar.set_postfix_str("模型训练与预测完成")
    except Exception as exc:
        if not is_memory_pressure_error(exc):
            if stage_pbar is not None:
                stage_pbar.close()
            raise

        print(f"[WARN] 全量读取/训练触发内存压力，切换到 BTH 按年份降级模式: {exc}")
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
        train_df, valid_df, test_df = _build_bth_split_frames_from_cache(
            fallback_feature_by_year,
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

        fallback = run_yearly_fallback_training(
            args=args,
            correlation_dir=correlation_dir,
            data_read_dir=data_read_dir,
            city_geojson_dir=city_geojson_dir,
            pm25_day_dir=pm25_day_dir,
            era5_day_dir=era5_day_dir,
            prebuilt_year_factor_df=prebuilt_year_factor_df,
            prebuilt_pm25_nc_index=prebuilt_pm25_nc_index,
            prebuilt_feature_by_year=fallback_feature_by_year,
            prebuilt_available_years=fallback_available_years,
            prebuilt_year_factor_cols=fallback_year_factor_cols,
            prebuilt_met_cols=fallback_met_cols,
        )
        model = fallback["model"]
        feature_cols = fallback["feature_cols"]
        all_pred_df = fallback["all_pred_df"]
        test_pred_df = fallback["test_pred_df"]
        metrics_df = fallback["metrics_df"]
        x_shap = fallback["x_shap"]
        grid_search_info = fallback["grid_search_info"]
        train_seconds_total += float(fallback.get("train_seconds", 0.0))
        yearly_batches = list(fallback.get("yearly_batches", []))
        if stage_pbar is not None:
            target_stage = 4
            if stage_pbar.n < target_stage:
                stage_pbar.update(target_stage - stage_pbar.n)
            stage_pbar.set_postfix_str("BTH 按年份降级训练完成")

    generalization_df = export_generalization_artifacts(metrics_df, output_dir)
    export_regression_artifacts(all_pred_df=all_pred_df, output_dir=output_dir, model_name="LightGBM-BTH")
    shap_status = "disabled"
    if not args.disable_shap:
        try:
            export_shap_artifacts(
                model=model,
                x_for_shap=x_shap,
                output_dir=output_dir,
                model_name="LightGBM-BTH",
                shap_max_samples=args.shap_max_samples,
                shap_max_display=args.shap_max_display,
                random_state=args.seed,
            )
            shap_status = "ok"
        except Exception as exc:  # pragma: no cover  # pylint: disable=broad-except
            shap_status = f"failed: {exc}"
            print(f"[WARN] SHAP export failed for BTH: {exc}")

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, kind="mergesort")

    metrics_df.to_csv(output_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")
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
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    if grid_search_info.get("enabled") and isinstance(grid_search_info.get("results_df"), pd.DataFrame):
        gdf = grid_search_info["results_df"]
        keep_cols = [c for c in ["rank_test_score", "mean_test_score", "std_test_score", "params"] if c in gdf.columns]
        gdf.loc[:, keep_cols].to_csv(output_dir / "bayes_search_results.csv", index=False, encoding="utf-8-sig")

    run_info = {
        "model": "LightGBM",
        "cluster": BTH_CLUSTER_NAME,
        "training_granularity": "single_cluster",
        "train_end_year": args.train_end_year,
        "valid_year": args.valid_year,
        "test_year": args.test_year,
        "n_features": int(len(feature_cols)),
        "n_year_factor_features": len(year_factor_cols),
        "n_era5_daily_features": len(met_cols),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "daily_input": args.daily_input or [],
        "pm25_day_dir": str(pm25_day_dir),
        "era5_day_dir": str(era5_day_dir),
        "output_dir": str(output_dir),
        "year_factor_rule": "keep_yearly_value",
        "monthly_factor_rule": "divide_by_days_in_month",
        "generalization_level": str(generalization_df.loc[0, "generalization_level"]) if not generalization_df.empty else "",
        "shap_status": shap_status,
        "enable_bayes_search": bool(getattr(args, "enable_bayes_search", False)),
        "use_gpu": bool(getattr(args, "use_gpu", False)),
        "bayes_n_iter": int(getattr(args, "bayes_n_iter", 50)),
        "bayes_search_best_params": grid_search_info.get("best_params", {}),
        "training_mode": training_mode,
        "fallback_triggered": fallback_triggered,
        "yearly_batches": yearly_batches,
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
        "bth_feature_blocks": [
            "wind_speed",
            "wind_dir_sin",
            "wind_dir_cos",
            "dewpoint_depression",
            "pressure_change_1d",
            "wind_speed_3d_mean",
            "precip_3d_sum",
            "stagnation_flag",
            "is_heating_season",
            "cold_halfyear",
            "lag_1_x_heating",
            "lag_1_x_stagnation",
        ],
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as file:
        json.dump(run_info, file, ensure_ascii=False, indent=2)

    if stage_pbar is not None:
        stage_pbar.update(2)
        stage_pbar.set_postfix_str("结果导出完成")
        stage_pbar.close()

    if prebuilt_pm25_nc_index is not None:
        prebuilt_pm25_nc_index.clear()
        del prebuilt_pm25_nc_index

    print("=" * 90)
    print("[INFO] LightGBM-BTH daily PM2.5 training finished.")
    print(f"[INFO] Output directory: {output_dir}")
    print("[INFO] Files:")
    print("       - metrics_overall.csv")
    print("       - predictions_test.csv")
    print("       - predictions_all_splits.csv")
    print("       - feature_importance.csv")
    print("       - feature_quality_report.csv")
    print("       - generalization_assessment.csv")
    print("       - generalization_plot_data.csv")
    print("       - regression_all_splits_data.csv")
    print("       - regression_test_data.csv")
    print("       - shap_*")
    print("       - run_info.json")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
