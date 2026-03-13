"""
Daily PM2.5 classification for BTH/YRD/PRD using Logistic Regression with per-cluster training.
Converts continuous PM2.5 to bins (优/良/轻度污染/中重度污染) and predicts pollution level.
Reuses daily_ml_pipeline for data preparation; adds classification-specific evaluation.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    from skopt import BayesSearchCV  # type: ignore[import-untyped]
    from skopt.space import Categorical, Real  # type: ignore[import-untyped]
except ImportError as _e:
    BayesSearchCV = None  # type: ignore[misc, assignment]
    Categorical = Real = None  # type: ignore[misc, assignment]
    _skopt_import_error = _e

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
    export_shap_artifacts,
    metrics_by_cluster,
    prepare_training_table,
    split_by_time,
)
from cluster_training_utils import prepare_training_table_with_fallback


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "logistic_daily_pm25"

# PM2.5 bins (μg/m³): [0,35) 优, [35,75) 良, [75,115) 轻度污染, [115,+∞) 中重度污染
PM25_BINS = np.array([35.0, 75.0, 115.0])
PM25_MIDPOINTS = np.array([17.5, 55.0, 95.0, 207.5])
CLASS_LABELS = ["优", "良", "轻度污染", "中重度污染"]

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # pylint: disable=broad-except
        pass


def _pm25_to_class(pm25: np.ndarray | pd.Series) -> np.ndarray:
    """Convert continuous PM2.5 to class indices 0..3."""
    arr = np.asarray(pm25, dtype=np.float64)
    return np.clip(np.digitize(arr, PM25_BINS), 0, 3)


def _class_to_midpoint(class_indices: np.ndarray) -> np.ndarray:
    """Map class indices to PM2.5 midpoint values for pipeline compatibility."""
    idx = np.clip(class_indices.astype(int), 0, len(PM25_MIDPOINTS) - 1)
    return PM25_MIDPOINTS[idx].astype(np.float64)


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
        best_score = float(np.max(func_vals)) if func_vals else float("nan")
        _log_bs_msg(
            f"[BayesSearch] 进度 {cur}/{n_iter} ({pct:.1f}%) "
            f"当前最佳得分 {best_score:.6f} 已用 {elapsed:.0f}s 预计剩余 {eta_s:.0f}s"
        )

    return callback


def _native_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in params.items():
        if hasattr(v, "item"):
            out[k] = v.item()
        elif isinstance(v, (np.integer, np.int64)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _get_logistic_feature_importance(model: LogisticRegression, feature_cols: list[str]) -> np.ndarray:
    """Multiclass LogisticRegression: coef_ is (n_classes, n_features), use mean abs across classes."""
    coef = model.coef_
    if coef.ndim == 1:
        return np.abs(coef[: len(feature_cols)]).astype(np.float64)
    return np.mean(np.abs(coef), axis=0)[: len(feature_cols)].astype(np.float64)


def compute_classification_metrics(
    y_true_class: np.ndarray, y_pred_class: np.ndarray
) -> dict[str, float]:
    """Compute accuracy, F1-macro, precision-macro, recall-macro."""
    if len(y_true_class) == 0:
        return {"accuracy": np.nan, "f1_macro": np.nan, "precision_macro": np.nan, "recall_macro": np.nan}
    return {
        "accuracy": float(accuracy_score(y_true_class, y_pred_class)),
        "f1_macro": float(f1_score(y_true_class, y_pred_class, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true_class, y_pred_class, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_class, y_pred_class, average="macro", zero_division=0)),
    }


def export_classification_artifacts(
    *,
    y_true_class: np.ndarray,
    y_pred_class: np.ndarray,
    output_dir: Path,
    model_name: str,
) -> None:
    """Export confusion matrix and classification report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true_class, y_pred_class, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, index=CLASS_LABELS, columns=CLASS_LABELS)
    cm_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    report = classification_report(
        y_true_class, y_pred_class, labels=[0, 1, 2, 3], target_names=CLASS_LABELS, zero_division=0
    )
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"{model_name}\n\n{report}")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"label": "count"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()


def train_logistic(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    args: argparse.Namespace,
) -> tuple[tuple[Any, StandardScaler], dict[str, Any]]:
    y_train_class = _pm25_to_class(y_train.values)

    scaler_x = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler_x.fit_transform(x_train),
        columns=x_train.columns,
        index=x_train.index,
    )

    train_params = {
        "C": args.C,
        "solver": args.solver,
        "max_iter": args.max_iter,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
    }

    if getattr(args, "enable_bayes_search", False):
        if BayesSearchCV is None or Real is None or Categorical is None:
            raise ImportError(
                "贝叶斯搜索需要 scikit-optimize，请安装: pip install scikit-optimize"
            ) from getattr(sys.modules.get(__name__), "_skopt_import_error", None)
        search_spaces = {
            "C": Real(0.01, 100.0, prior="log-uniform"),
            "solver": Categorical(["lbfgs", "saga"]),
        }
        cv_splits = getattr(args, "cv_splits", 5)
        n_iter = getattr(args, "bayes_n_iter", 30)
        start_time = time.perf_counter()
        start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        _log_bs_msg(
            f"[BayesSearch] 迭代数 {n_iter}, CV 折数 {cv_splits}, 开始时间 {start_str}"
        )
        _log_bs_msg(
            f"[BayesSearch] 每组参数会先跑 {cv_splits} 折 CV，每组完成后会打印进度，请耐心等待。"
        )
        base_est = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=args.max_iter, random_state=args.seed, n_jobs=args.n_jobs
        )
        search = BayesSearchCV(
            estimator=base_est,
            search_spaces=search_spaces,
            n_iter=n_iter,
            scoring=getattr(args, "bayes_scoring", "f1_macro"),
            cv=TimeSeriesSplit(n_splits=cv_splits),
            n_jobs=getattr(args, "bayes_n_jobs", 1),
            refit=True,
            random_state=args.seed,
            verbose=0,
        )
        progress_cb = _bayesian_search_progress_callback(n_iter, start_time)
        search.fit(x_train_scaled, y_train_class, callback=progress_cb)
        elapsed = time.perf_counter() - start_time
        best_params = _native_params(search.best_params_)
        _log_bs_msg(
            f"[BayesSearch] 完成 总耗时 {elapsed:.1f}s 最佳得分 {search.best_score_} 最佳参数 {best_params}"
        )
        model = search.best_estimator_
        results_df = pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_score", kind="mergesort"
        )
        search_info = {
            "enabled": True,
            "search_method": "bayesian",
            "best_params": best_params,
            "best_score": float(search.best_score_),
            "cv_splits": cv_splits,
            "n_iter": n_iter,
            "results_df": results_df,
        }
        return (model, scaler_x), search_info

    model = LogisticRegression(**train_params)
    model.fit(x_train_scaled, y_train_class)
    return (model, scaler_x), {"enabled": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Daily PM2.5 classification for BTH/YRD/PRD using Logistic Regression (PM2.5 bins)."
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
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength.")
    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"),
        help="Solver for LogisticRegression.",
    )
    parser.add_argument("--max-iter", type=int, default=1000, help="Max iterations for solver.")
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=0,
        help="Ignored (Logistic has no early stopping); kept for API compatibility.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs.")
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
    parser.add_argument("--bayes-n-iter", type=int, default=30, help="贝叶斯搜索迭代次数.")
    parser.add_argument("--cv-splits", type=int, default=5, help="TimeSeriesSplit 折数.")
    parser.add_argument(
        "--bayes-n-jobs",
        type=int,
        default=1,
        help="贝叶斯搜索 CV 并行数.",
    )
    parser.add_argument(
        "--bayes-scoring",
        type=str,
        default="f1_macro",
        help="贝叶斯搜索 scoring 指标 (e.g. f1_macro, accuracy).",
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


def main() -> int:
    args = build_parser().parse_args()
    correlation_dir = Path(args.correlation_dir).expanduser().resolve()
    data_read_dir = Path(args.data_read_dir).expanduser().resolve()
    pm25_day_dir = Path(args.pm25_day_dir).expanduser().resolve()
    era5_day_dir = Path(args.era5_day_dir).expanduser().resolve()
    city_geojson_dir = Path(args.city_geojson_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_pbar = tqdm(total=6, desc="Logistic 总进度", dynamic_ncols=True) if tqdm is not None else None
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
            module_tag="logistic",
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

        x_train, y_train, x_valid, y_valid, x_test, y_test, feature_cols = build_model_matrices(
            train_df=cluster_train_df,
            valid_df=cluster_valid_df,
            test_df=cluster_test_df,
        )
        train_begin = time.perf_counter()
        try:
            (model, scaler_x), grid_search_info = train_logistic(
                x_train, y_train, x_valid, y_valid, args
            )
        except Exception as exc:
            if getattr(args, "enable_bayes_search", False):
                print(f"[WARN] 城市群 {cluster_name} 贝叶斯搜索失败，回退默认参数训练: {exc}")
                old_flag = args.enable_bayes_search
                args.enable_bayes_search = False
                try:
                    (model, scaler_x), grid_search_info = train_logistic(
                        x_train, y_train, x_valid, y_valid, args
                    )
                finally:
                    args.enable_bayes_search = old_flag
                grid_search_info["bayes_fallback_reason"] = str(exc)
            else:
                raise
        train_seconds = time.perf_counter() - train_begin
        train_seconds_total += train_seconds

        x_train_scaled = scaler_x.transform(x_train)
        x_valid_scaled = scaler_x.transform(x_valid)
        x_test_scaled = scaler_x.transform(x_test)
        pred_train_class = model.predict(x_train_scaled)
        pred_valid_class = model.predict(x_valid_scaled)
        pred_test_class = model.predict(x_test_scaled)
        pred_train = _class_to_midpoint(pred_train_class)
        pred_valid = _class_to_midpoint(pred_valid_class)
        pred_test = _class_to_midpoint(pred_test_class)

        y_train_class = _pm25_to_class(y_train.values)
        y_valid_class = _pm25_to_class(y_valid.values)
        y_test_class = _pm25_to_class(y_test.values)

        reg_metrics_train = compute_metrics(y_train, pred_train)
        reg_metrics_valid = compute_metrics(y_valid, pred_valid)
        reg_metrics_test = compute_metrics(y_test, pred_test)
        clf_train = compute_classification_metrics(y_train_class, pred_train_class)
        clf_valid = compute_classification_metrics(y_valid_class, pred_valid_class)
        clf_test = compute_classification_metrics(y_test_class, pred_test_class)

        metric_rows = [
            {
                "split": "train",
                **reg_metrics_train,
                **clf_train,
                "n_samples": int(len(y_train)),
            },
            {
                "split": "valid",
                **reg_metrics_valid,
                **clf_valid,
                "n_samples": int(len(y_valid)),
            },
            {
                "split": "test",
                **reg_metrics_test,
                **clf_test,
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
        x_shap = x_test if not x_test.empty else (x_valid if not x_valid.empty else x_train)
        x_shap_scaled = pd.DataFrame(
            scaler_x.transform(x_shap),
            columns=x_shap.columns,
            index=x_shap.index,
        )
        cluster_results[cluster_name] = {
            "model": model,
            "scaler_x": scaler_x,
            "feature_cols": feature_cols,
            "all_pred_df": all_pred_df,
            "test_pred_df": test_pred_df,
            "metrics_df": metrics_df,
            "x_shap": x_shap_scaled,
            "x_train": x_train,
            "y_train": y_train,
            "y_test_class": y_test_class,
            "pred_test_class": pred_test_class,
            "grid_search_info": grid_search_info,
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
        scaler_x = result["scaler_x"]
        feature_cols = result["feature_cols"]
        all_pred_df = result["all_pred_df"]
        test_pred_df = result["test_pred_df"]
        metrics_df = result["metrics_df"]
        x_shap = result["x_shap"]
        x_train = result["x_train"]
        y_train = result["y_train"]
        y_test_class = result["y_test_class"]
        pred_test_class = result["pred_test_class"]
        grid_search_info = result["grid_search_info"]

        joblib.dump(
            {
                "model": model,
                "scaler_x": scaler_x,
                "pm25_bins": PM25_BINS,
                "pm25_midpoints": PM25_MIDPOINTS,
            },
            cluster_output_dir / "logistic_best.joblib",
        )

        cluster_metrics_df = metrics_by_cluster(test_pred_df)
        generalization_df = export_generalization_artifacts(metrics_df, cluster_output_dir)
        export_regression_artifacts(
            all_pred_df=all_pred_df,
            output_dir=cluster_output_dir,
            model_name=f"Logistic-{cluster_name}",
        )
        export_classification_artifacts(
            y_true_class=y_test_class,
            y_pred_class=pred_test_class,
            output_dir=cluster_output_dir,
            model_name=f"Logistic-{cluster_name}",
        )

        shap_status = "disabled"
        if not args.disable_shap:
            try:
                export_shap_artifacts(
                    model=model,
                    x_for_shap=x_shap,
                    output_dir=cluster_output_dir,
                    model_name=f"Logistic-{cluster_name}",
                    shap_max_samples=args.shap_max_samples,
                    shap_max_display=args.shap_max_display,
                    random_state=args.seed,
                )
                shap_status = "ok"
            except Exception as exc:  # pylint: disable=broad-except
                shap_status = f"failed: {exc}"
                print(f"[WARN] SHAP export failed for {cluster_name}: {exc}")

        importance = _get_logistic_feature_importance(model, feature_cols)
        importance_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": importance,
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
        if grid_search_info.get("enabled") and isinstance(
            grid_search_info.get("results_df"), pd.DataFrame
        ):
            gdf = grid_search_info["results_df"]
            keep_cols = [
                c
                for c in ["rank_test_score", "mean_test_score", "std_test_score", "params"]
                if c in gdf.columns
            ]
            gdf.loc[:, keep_cols].to_csv(
                cluster_output_dir / "bayes_search_results.csv",
                index=False,
                encoding="utf-8-sig",
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
            "shap_status": shap_status,
            "bayes_search_enabled": bool(grid_search_info.get("enabled", False)),
            "bayes_search_best_score": grid_search_info.get("best_score"),
            "bayes_search_best_params": grid_search_info.get("best_params", {}),
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
                {
                    "split": split_name,
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "r2": float("nan"),
                    "n_samples": 0,
                }
            )
            continue
        split_reg = compute_metrics(split_df["y_true"], split_df["y_pred"].to_numpy())
        pooled_metric_rows.append(
            {"split": split_name, **split_reg, "n_samples": int(len(split_df))}
        )
    metrics_df = pd.DataFrame(pooled_metric_rows)
    cluster_metrics_df = metrics_by_cluster(test_pred_df)
    pooled_generalization_df = export_generalization_artifacts(metrics_df, output_dir)
    export_regression_artifacts(
        all_pred_df=all_pred_df,
        output_dir=output_dir,
        model_name="Logistic-ClusterModels",
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

    total_train_rows = int(sum(info["train_rows"] for info in per_cluster_run_info.values()))
    total_valid_rows = int(sum(info["valid_rows"] for info in per_cluster_run_info.values()))
    total_test_rows = int(sum(info["test_rows"] for info in per_cluster_run_info.values()))
    n_features_by_cluster = {
        cluster: int(info["n_features"]) for cluster, info in per_cluster_run_info.items()
    }
    run_info = {
        "model": "Logistic",
        "task": "classification",
        "pm25_bins": PM25_BINS.tolist(),
        "pm25_midpoints": PM25_MIDPOINTS.tolist(),
        "class_labels": CLASS_LABELS,
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
        "train_rows": total_train_rows,
        "valid_rows": total_valid_rows,
        "test_rows": total_test_rows,
        "daily_input": args.daily_input or [],
        "pm25_day_dir": str(pm25_day_dir),
        "era5_day_dir": str(era5_day_dir),
        "year_factor_rule": "keep_yearly_value",
        "monthly_factor_rule": "divide_by_days_in_month",
        "generalization_level": str(pooled_generalization_df.loc[0, "generalization_level"])
        if not pooled_generalization_df.empty
        else "",
        "shap_status": {
            cluster: info["shap_status"] for cluster, info in per_cluster_run_info.items()
        },
        "enable_bayes_search": bool(getattr(args, "enable_bayes_search", False)),
        "bayes_n_iter": int(getattr(args, "bayes_n_iter", 30)),
        "bayes_search_best_params": {
            c: info.get("bayes_search_best_params", {})
            for c, info in per_cluster_run_info.items()
        },
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
    print("[INFO] Logistic Regression daily PM2.5 classification finished.")
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
    print("       - feature_importance.csv")
    print("       - feature_importance_by_cluster.csv")
    print("       - confusion_matrix.csv / confusion_matrix.png")
    print("       - classification_report.txt")
    print("       - run_info.json")
    print("       - logistic_best.joblib (per cluster: model + scaler + bins)")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
