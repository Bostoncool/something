from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "xgboost_daily_pm25"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # pylint: disable=broad-except
        pass


def train_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    args: argparse.Namespace,
) -> Any:
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed. Please run: pip install xgboost")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )

    fit_kwargs: dict[str, Any] = {}
    if not x_valid.empty:
        fit_kwargs["eval_set"] = [(x_valid, y_valid)]
        fit_kwargs["verbose"] = False

    try:
        if args.early_stopping_rounds > 0 and not x_valid.empty:
            model.fit(x_train, y_train, early_stopping_rounds=args.early_stopping_rounds, **fit_kwargs)
        else:
            model.fit(x_train, y_train, **fit_kwargs)
    except TypeError:
        if args.early_stopping_rounds > 0 and not x_valid.empty:
            model.set_params(early_stopping_rounds=args.early_stopping_rounds)
        model.fit(x_train, y_train, **fit_kwargs)
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily PM2.5 prediction for BTH/YRD/PRD using XGBoost.")
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
    parser.add_argument("--n-estimators", type=int, default=1500, help="Number of boosting trees.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Learning rate.")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth.")
    parser.add_argument("--subsample", type=float, default=0.9, help="Row sampling ratio.")
    parser.add_argument("--colsample-bytree", type=float, default=0.9, help="Column sampling ratio.")
    parser.add_argument("--reg-alpha", type=float, default=0.0, help="L1 regularization.")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="L2 regularization.")
    parser.add_argument("--early-stopping-rounds", type=int, default=100, help="Early stopping rounds.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for model fitting.")
    parser.add_argument("--disable-shap", action="store_true", help="Disable SHAP explainability outputs.")
    parser.add_argument("--shap-max-samples", type=int, default=3000, help="Max rows used for SHAP.")
    parser.add_argument("--shap-max-display", type=int, default=20, help="Max displayed SHAP features.")
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

    base_df, year_factor_cols, met_cols = prepare_training_table(
        module_tag="xgb",
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
    x_train, y_train, x_valid, y_valid, x_test, y_test, feature_cols = build_model_matrices(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )

    model = train_xgboost(x_train, y_train, x_valid, y_valid, args)
    pred_train = model.predict(x_train)
    pred_valid = model.predict(x_valid)
    pred_test = model.predict(x_test)

    metric_rows = [
        {"split": "train", **compute_metrics(y_train, pred_train), "n_samples": int(len(y_train))},
        {"split": "valid", **compute_metrics(y_valid, pred_valid), "n_samples": int(len(y_valid))},
        {"split": "test", **compute_metrics(y_test, pred_test), "n_samples": int(len(y_test))},
    ]
    metrics_df = pd.DataFrame(metric_rows)

    all_pred_df, test_pred_df = build_prediction_frames(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        pred_train=pred_train,
        pred_valid=pred_valid,
        pred_test=pred_test,
    )
    cluster_metrics_df = metrics_by_cluster(test_pred_df)
    generalization_df = export_generalization_artifacts(metrics_df, output_dir)
    export_regression_artifacts(all_pred_df=all_pred_df, output_dir=output_dir, model_name="XGBoost")

    shap_status = "disabled"
    if not args.disable_shap:
        try:
            x_shap = x_test if not x_test.empty else (x_valid if not x_valid.empty else x_train)
            export_shap_artifacts(
                model=model,
                x_for_shap=x_shap,
                output_dir=output_dir,
                model_name="XGBoost",
                shap_max_samples=args.shap_max_samples,
                shap_max_display=args.shap_max_display,
                random_state=args.seed,
            )
            shap_status = "ok"
        except Exception as exc:  # pylint: disable=broad-except
            shap_status = f"failed: {exc}"
            print(f"[WARN] SHAP export failed: {exc}")

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, kind="mergesort")

    metrics_df.to_csv(output_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")
    cluster_metrics_df.to_csv(output_dir / "metrics_by_cluster_test.csv", index=False, encoding="utf-8-sig")
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

    run_info = {
        "model": "XGBoost",
        "train_end_year": args.train_end_year,
        "valid_year": args.valid_year,
        "test_year": args.test_year,
        "n_features": len(feature_cols),
        "n_year_factor_features": len(year_factor_cols),
        "n_era5_daily_features": len(met_cols),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "daily_input": args.daily_input or [],
        "pm25_day_dir": str(pm25_day_dir),
        "era5_day_dir": str(era5_day_dir),
        "year_factor_rule": "divide_by_365",
        "monthly_factor_rule": "divide_by_days_in_month",
        "generalization_level": str(generalization_df.loc[0, "generalization_level"]) if not generalization_df.empty else "",
        "shap_status": shap_status,
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as file:
        json.dump(run_info, file, ensure_ascii=False, indent=2)

    print("=" * 90)
    print("[INFO] XGBoost daily PM2.5 training finished.")
    print(f"[INFO] Output directory: {output_dir}")
    print("[INFO] Files:")
    print("       - metrics_overall.csv")
    print("       - metrics_by_cluster_test.csv")
    print("       - predictions_test.csv")
    print("       - predictions_all_splits.csv")
    print("       - feature_importance.csv")
    print("       - generalization_assessment.csv")
    print("       - generalization_plot_data.csv")
    print("       - regression_all_splits_data.csv")
    print("       - regression_test_data.csv")
    print("       - shap_sample_features.csv")
    print("       - shap_values_wide.csv")
    print("       - shap_beeswarm_data_long.csv")
    print("       - shap_importance_bar_data.csv")
    print("       - run_info.json")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
