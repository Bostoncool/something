from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark_config import BenchmarkConfig, ClusterInputConfig, DatasetTypeConfig, build_default_config, load_config_from_dict, resolve_paths
from daily_ml_pipeline import (
    CITY_ALIASES,
    DATE_ALIASES,
    PM25_ALIASES,
    DEFAULT_CITY_GEOJSON_DIR,
    DEFAULT_CORRELATION_DIR,
    DEFAULT_DATA_READ_DIR,
    DEFAULT_ERA5_DAY_DIR,
    DEFAULT_PM25_DAY_DIR,
    SCRIPT_DIR,
    detect_column,
    load_region_configs,
    normalize_city_name,
    read_table,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "benchmark"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 3 urban clusters x 4 models benchmark with unified data ingestion and report export."
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default="",
        help="Optional JSON file for cluster/model config overrides.",
    )
    parser.add_argument(
        "--daily-input",
        nargs="+",
        default=None,
        help="Optional shared daily long-table file(s)/folder(s), used for all clusters unless cluster override exists.",
    )
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Benchmark output root directory.")
    parser.add_argument("--run-tag", type=str, default="", help="Custom run tag; default is timestamp.")
    parser.add_argument("--correlation-dir", type=str, default=str(DEFAULT_CORRELATION_DIR), help="Correlation directory.")
    parser.add_argument("--data-read-dir", type=str, default=str(DEFAULT_DATA_READ_DIR), help="Data Read directory.")
    parser.add_argument("--pm25-day-dir", type=str, default=str(DEFAULT_PM25_DAY_DIR), help="PM2.5 daily NC directory.")
    parser.add_argument("--era5-day-dir", type=str, default=str(DEFAULT_ERA5_DAY_DIR), help="ERA5 daily NC directory.")
    parser.add_argument("--city-geojson-dir", type=str, default=str(DEFAULT_CITY_GEOJSON_DIR), help="City geojson directory.")
    parser.add_argument("--python-exe", type=str, default=sys.executable, help="Python executable for subprocess model runs.")
    parser.add_argument("--train-end-year", type=int, default=None, help="Override train end year from config.")
    parser.add_argument("--valid-year", type=int, default=None, help="Override valid year from config.")
    parser.add_argument("--test-year", type=int, default=None, help="Override test year from config.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed from config.")
    parser.add_argument("--device", type=str, default="", help="Override deep model device: cuda/gpu/cpu.")
    parser.add_argument("--skip-era5-daily", action="store_true", help="Force disable ERA5 daily feature merging.")
    parser.add_argument("--disable-shap", action="store_true", help="Disable SHAP outputs for tree models.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue benchmark when one model run fails.")
    parser.add_argument(
        "--strict-core",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require all core dataset paths to be provided and readable.",
    )
    parser.add_argument(
        "--allow-missing-aux",
        action="store_true",
        help="Allow auxiliary dataset missing paths/files without warning escalation.",
    )
    return parser


def load_benchmark_config(config_json: str) -> BenchmarkConfig:
    if not config_json.strip():
        return build_default_config()
    config_path = Path(config_json).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("Config JSON must be an object/dict.")
    return load_config_from_dict(raw)


def _detect_or_fallback_column(
    raw_df: pd.DataFrame,
    aliases: list[str],
    contains_tokens: list[str],
    fallback: str,
) -> str:
    detected = detect_column(raw_df, aliases=aliases, contains_tokens=contains_tokens)
    if detected is not None:
        return detected
    if fallback and fallback in raw_df.columns:
        return fallback
    raise ValueError(f"Column detect failed, fallback={fallback!r} not found.")


def _standardize_daily_file(path: Path, column_map: dict[str, str]) -> pd.DataFrame:
    raw = read_table(path).copy()
    raw.columns = [str(col).strip() for col in raw.columns]
    city_col = _detect_or_fallback_column(
        raw_df=raw,
        aliases=CITY_ALIASES,
        contains_tokens=["city", "城市"],
        fallback=str(column_map.get("city", "")).strip(),
    )
    date_col = _detect_or_fallback_column(
        raw_df=raw,
        aliases=DATE_ALIASES,
        contains_tokens=["date", "日期", "time"],
        fallback=str(column_map.get("date", "")).strip(),
    )
    pm25_col = _detect_or_fallback_column(
        raw_df=raw,
        aliases=PM25_ALIASES,
        contains_tokens=["pm25", "pm2.5"],
        fallback=str(column_map.get("pm25", "")).strip(),
    )
    out = raw.rename(columns={city_col: "city", date_col: "date", pm25_col: "pm25"})[["city", "date", "pm25"]].copy()
    date_series = pd.to_datetime(out["date"], errors="coerce")
    if date_series.isna().all():
        date_series = pd.to_datetime(out["date"].astype(str), format="%Y%m%d", errors="coerce")
    if getattr(date_series.dt, "tz", None) is not None:
        date_series = date_series.dt.tz_localize(None)
    out["date"] = date_series.dt.normalize()
    out["city"] = out["city"].map(normalize_city_name)
    out["pm25"] = pd.to_numeric(out["pm25"], errors="coerce")
    out = out.dropna(subset=["city", "date", "pm25"]).copy()
    out["city"] = out["city"].astype(str).str.strip()
    return out


def _expand_data_files(path_str: str) -> list[Path]:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    return sorted([*path.rglob("*.csv"), *path.rglob("*.parquet"), *path.rglob("*.xlsx"), *path.rglob("*.xls")])


def _validate_dataset_source(
    *,
    cluster_cfg: ClusterInputConfig,
    dataset_cfg: DatasetTypeConfig,
    strict_core: bool,
    allow_missing_aux: bool,
) -> tuple[list[Path], str, int]:
    raw_paths = resolve_paths(dataset_cfg.paths)
    if dataset_cfg.key == "pm25" and not raw_paths:
        raw_paths = resolve_paths(cluster_cfg.daily_input)

    if dataset_cfg.priority == "unused" or not dataset_cfg.enabled:
        return [], "skipped_unused", 0

    if not raw_paths:
        if dataset_cfg.priority == "core" and strict_core:
            raise ValueError(f"{cluster_cfg.key}:{dataset_cfg.key} is core but has no configured paths.")
        return [], "missing_path", 0

    data_files: list[Path] = []
    for one_path in raw_paths:
        data_files.extend(_expand_data_files(one_path))
    if not data_files:
        if dataset_cfg.priority == "core" and strict_core:
            raise FileNotFoundError(f"{cluster_cfg.key}:{dataset_cfg.key} core paths have no readable files.")
        if dataset_cfg.priority == "aux" and not allow_missing_aux:
            print(f"[WARN] {cluster_cfg.key}:{dataset_cfg.key} aux paths have no readable files.")
        return [], "missing_file", int(len(raw_paths))
    return data_files, "ok", int(len(raw_paths))


def build_cluster_daily_input_with_priority(
    *,
    cluster_cfg: ClusterInputConfig,
    allowed_cities: set[str],
    run_dir: Path,
    strict_core: bool,
    allow_missing_aux: bool,
) -> tuple[Path, pd.DataFrame]:
    if "pm25" not in cluster_cfg.dataset_types:
        raise ValueError(f"{cluster_cfg.key} missing required dataset_types.pm25 config.")

    coverage_rows: list[dict[str, Any]] = []
    pm25_records: list[pd.DataFrame] = []
    pm25_dataset = cluster_cfg.dataset_types["pm25"]

    for dataset_key, dataset_cfg in cluster_cfg.dataset_types.items():
        files, status, n_paths = _validate_dataset_source(
            cluster_cfg=cluster_cfg,
            dataset_cfg=dataset_cfg,
            strict_core=strict_core,
            allow_missing_aux=allow_missing_aux,
        )
        if status != "ok":
            coverage_rows.append(
                {
                    "cluster": cluster_cfg.key,
                    "cluster_name": cluster_cfg.display_name,
                    "dataset_type": dataset_key,
                    "priority": dataset_cfg.priority,
                    "enabled": bool(dataset_cfg.enabled),
                    "status": status,
                    "n_paths": int(n_paths),
                    "n_files": 0,
                    "n_rows": 0,
                }
            )
            continue

        row_count = 0
        for one_file in files:
            if dataset_key == "pm25":
                standardized = _standardize_daily_file(one_file, column_map=dataset_cfg.column_map or cluster_cfg.column_map)
                pm25_records.append(standardized)
                row_count += len(standardized)
            else:
                # Non-pm25 datasets are validated/read for coverage and consistency checks.
                row_count += len(read_table(one_file))

        coverage_rows.append(
            {
                "cluster": cluster_cfg.key,
                "cluster_name": cluster_cfg.display_name,
                "dataset_type": dataset_key,
                "priority": dataset_cfg.priority,
                "enabled": bool(dataset_cfg.enabled),
                "status": "loaded",
                "n_paths": int(n_paths),
                "n_files": int(len(files)),
                "n_rows": int(row_count),
            }
        )

    if not pm25_records:
        if strict_core or pm25_dataset.priority == "core":
            raise ValueError(f"{cluster_cfg.display_name}({cluster_cfg.key}) has no usable PM2.5 records.")
        raise ValueError(f"{cluster_cfg.display_name}({cluster_cfg.key}) PM2.5 source missing.")

    daily_df = pd.concat(pm25_records, ignore_index=True)
    daily_df = daily_df.loc[daily_df["city"].isin(allowed_cities)].copy()
    if daily_df.empty:
        raise ValueError(f"{cluster_cfg.display_name}({cluster_cfg.key}) has no rows after city filtering.")
    daily_df = (
        daily_df.groupby(["city", "date"], as_index=False)["pm25"]
        .mean()
        .sort_values(["city", "date"], kind="mergesort")
        .reset_index(drop=True)
    )

    prep_dir = run_dir / "prepared_inputs"
    prep_dir.mkdir(parents=True, exist_ok=True)
    output_path = prep_dir / f"{cluster_cfg.key.lower()}_daily_input.csv"
    daily_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path, pd.DataFrame(coverage_rows)


def _to_cli_args(extra_args: dict[str, Any]) -> list[str]:
    cli_args: list[str] = []
    for key, value in extra_args.items():
        cli_key = f"--{str(key).strip()}"
        if isinstance(value, bool):
            if value:
                cli_args.append(cli_key)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            cli_args.append(cli_key)
            cli_args.extend([str(item) for item in value])
            continue
        cli_args.extend([cli_key, str(value)])
    return cli_args


def run_one_model(
    *,
    python_exe: str,
    script_path: Path,
    output_dir: Path,
    daily_input_path: Path,
    common_args: dict[str, Any],
    extra_args: dict[str, Any],
    disable_shap: bool,
    device: str,
) -> None:
    cmd = [
        python_exe,
        str(script_path),
        "--daily-input",
        str(daily_input_path),
        "--output-dir",
        str(output_dir),
        "--correlation-dir",
        str(common_args["correlation_dir"]),
        "--data-read-dir",
        str(common_args["data_read_dir"]),
        "--pm25-day-dir",
        str(common_args["pm25_day_dir"]),
        "--era5-day-dir",
        str(common_args["era5_day_dir"]),
        "--city-geojson-dir",
        str(common_args["city_geojson_dir"]),
        "--train-end-year",
        str(common_args["train_end_year"]),
        "--valid-year",
        str(common_args["valid_year"]),
        "--test-year",
        str(common_args["test_year"]),
        "--seed",
        str(common_args["seed"]),
    ]
    if not common_args["include_era5_daily"]:
        cmd.append("--skip-era5")
    if disable_shap and script_path.name in {"XGBOOST.py", "LightGBM.py"}:
        cmd.append("--disable-shap")
    if device and script_path.name in {"CNN-LSTM.py", "ST-Transformer.py"}:
        cmd.extend(["--device", device])
    cmd.extend(_to_cli_args(extra_args))
    subprocess.run(cmd, check=True)


def collect_metrics(model_output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_path = model_output_dir / "metrics_overall.csv"
    cluster_path = model_output_dir / "metrics_by_cluster_test.csv"
    if not overall_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {overall_path}")
    overall_df = pd.read_csv(overall_path)
    cluster_df = pd.read_csv(cluster_path) if cluster_path.exists() else pd.DataFrame()
    return overall_df, cluster_df


def export_benchmark_reports(metrics_long: pd.DataFrame, output_dir: Path, coverage_df: pd.DataFrame) -> None:
    metrics_dir = output_dir / "metrics"
    plot_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics_long = metrics_long.sort_values(["cluster", "model", "split"], kind="mergesort").reset_index(drop=True)
    metrics_long.to_csv(metrics_dir / "benchmark_metrics_long.csv", index=False, encoding="utf-8-sig")

    test_df = metrics_long.loc[metrics_long["split"] == "test"].copy()
    metrics_wide = test_df.pivot_table(index=["cluster", "model"], values=["rmse", "mae", "r2"], aggfunc="first").reset_index()
    metrics_wide.to_csv(metrics_dir / "benchmark_metrics_wide.csv", index=False, encoding="utf-8-sig")

    ranking_df = (
        test_df.sort_values(["cluster", "rmse"], kind="mergesort")
        .groupby("cluster", as_index=False, group_keys=False)
        .apply(lambda part: part.assign(rmse_rank=np.arange(1, len(part) + 1)))
        .reset_index(drop=True)
    )
    ranking_df.to_csv(metrics_dir / "benchmark_test_rmse_ranking.csv", index=False, encoding="utf-8-sig")
    if not coverage_df.empty:
        coverage_sorted = coverage_df.sort_values(
            ["cluster", "priority", "dataset_type"], kind="mergesort"
        ).reset_index(drop=True)
        coverage_sorted.to_csv(metrics_dir / "data_coverage_by_cluster.csv", index=False, encoding="utf-8-sig")

    if plt is None or test_df.empty:
        return

    metrics = ["rmse", "mae", "r2"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    models = sorted(test_df["model"].unique().tolist())
    clusters = sorted(test_df["cluster"].unique().tolist())
    x = np.arange(len(clusters))
    width = 0.8 / max(1, len(models))

    for idx_metric, metric in enumerate(metrics):
        ax = axes[idx_metric]
        for idx_model, model in enumerate(models):
            sub = test_df.loc[test_df["model"] == model].set_index("cluster").reindex(clusters)
            y = sub[metric].to_numpy(dtype=float)
            pos = x - 0.4 + width / 2 + idx_model * width
            ax.bar(pos, y, width=width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels(clusters)
        ax.set_title(f"Test {metric.upper()} by Cluster/Model")
        ax.set_xlabel("cluster")
        ax.set_ylabel(metric.upper())
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "benchmark_test_metrics_grouped_bar.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    split_order = ["train", "valid", "test"]
    for idx_metric, metric in enumerate(metrics):
        ax = axes[idx_metric]
        for name, part in metrics_long.groupby(["cluster", "model"], observed=True):
            label = f"{name[0]}-{name[1]}"
            ordered = part.set_index("split").reindex(split_order)
            ax.plot(split_order, ordered[metric], marker="o", linewidth=1.2, alpha=0.9, label=label)
        ax.set_title(f"{metric.upper()} Across Splits")
        ax.set_xlabel("split")
        ax.set_ylabel(metric.upper())
    axes[0].legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(plot_dir / "benchmark_split_stability.png", dpi=300)
    plt.close()


def main() -> int:
    args = build_parser().parse_args()
    config = load_benchmark_config(args.config_json)
    if args.train_end_year is not None:
        config.train_end_year = int(args.train_end_year)
    if args.valid_year is not None:
        config.valid_year = int(args.valid_year)
    if args.test_year is not None:
        config.test_year = int(args.test_year)
    if args.seed is not None:
        config.seed = int(args.seed)
    if args.device.strip():
        config.device = args.device.strip()
    if args.skip_era5_daily:
        config.include_era5_daily = False

    if args.daily_input:
        shared_paths = resolve_paths(args.daily_input)
        for cluster_cfg in config.clusters:
            pm25_cfg = cluster_cfg.dataset_types.get("pm25")
            if pm25_cfg is not None and not pm25_cfg.paths:
                pm25_cfg.paths = shared_paths
            if not cluster_cfg.daily_input:
                cluster_cfg.daily_input = shared_paths
    for cluster_cfg in config.clusters:
        cluster_cfg.daily_input = resolve_paths(cluster_cfg.daily_input)
        for dataset_cfg in cluster_cfg.dataset_types.values():
            dataset_cfg.paths = resolve_paths(dataset_cfg.paths)

    correlation_dir = Path(args.correlation_dir).expanduser().resolve()
    common_args = {
        "correlation_dir": correlation_dir,
        "data_read_dir": Path(args.data_read_dir).expanduser().resolve(),
        "pm25_day_dir": Path(args.pm25_day_dir).expanduser().resolve(),
        "era5_day_dir": Path(args.era5_day_dir).expanduser().resolve(),
        "city_geojson_dir": Path(args.city_geojson_dir).expanduser().resolve(),
        "train_end_year": config.train_end_year,
        "valid_year": config.valid_year,
        "test_year": config.test_year,
        "seed": config.seed,
        "include_era5_daily": config.include_era5_daily,
    }

    run_tag = args.run_tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root).expanduser().resolve() / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    _, region_configs = load_region_configs(correlation_dir=correlation_dir, module_tag=f"benchmark_{run_tag}")
    region_city_map = {cfg.name: {normalize_city_name(city) for city in cfg.cities} for cfg in region_configs}
    available_clusters = set(region_city_map.keys())

    metrics_rows: list[pd.DataFrame] = []
    coverage_rows: list[pd.DataFrame] = []
    failed_runs: list[dict[str, str]] = []

    for cluster_cfg in config.clusters:
        if cluster_cfg.key not in available_clusters:
            raise ValueError(f"Unknown cluster key: {cluster_cfg.key}. Available={sorted(available_clusters)}")
        print("=" * 90)
        print(f"[INFO] Preparing cluster input: {cluster_cfg.display_name} ({cluster_cfg.key})")
        cluster_input_path, cluster_coverage_df = build_cluster_daily_input_with_priority(
            cluster_cfg=cluster_cfg,
            allowed_cities=region_city_map[cluster_cfg.key],
            run_dir=run_dir,
            strict_core=bool(args.strict_core),
            allow_missing_aux=bool(args.allow_missing_aux),
        )
        coverage_rows.append(cluster_coverage_df)

        for model_cfg in config.models:
            if not model_cfg.enabled:
                continue
            script_path = SCRIPT_DIR / model_cfg.script_name
            if not script_path.exists():
                message = f"Script not found: {script_path}"
                if args.continue_on_error:
                    failed_runs.append({"cluster": cluster_cfg.key, "model": model_cfg.key, "error": message})
                    print(f"[WARN] {message}")
                    continue
                raise FileNotFoundError(message)

            model_output_dir = run_dir / "artifacts" / cluster_cfg.key.lower() / model_cfg.key
            model_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Running {cluster_cfg.key} - {model_cfg.key}")
            try:
                use_era5 = common_args["include_era5_daily"]
                era5_cfg = cluster_cfg.dataset_types.get("era5")
                if era5_cfg is not None:
                    use_era5 = use_era5 and era5_cfg.enabled and era5_cfg.priority != "unused"

                run_one_model(
                    python_exe=args.python_exe,
                    script_path=script_path,
                    output_dir=model_output_dir,
                    daily_input_path=cluster_input_path,
                    common_args={**common_args, "include_era5_daily": use_era5},
                    extra_args=model_cfg.extra_args,
                    disable_shap=args.disable_shap,
                    device=config.device,
                )
                overall_df, _ = collect_metrics(model_output_dir)
                overall_df["cluster"] = cluster_cfg.key
                overall_df["cluster_name"] = cluster_cfg.display_name
                overall_df["model"] = model_cfg.key
                metrics_rows.append(overall_df[["cluster", "cluster_name", "model", "split", "rmse", "mae", "r2", "n_samples"]])
            except Exception as exc:  # pylint: disable=broad-except
                message = str(exc)
                failed_runs.append({"cluster": cluster_cfg.key, "model": model_cfg.key, "error": message})
                print(f"[ERROR] Run failed: cluster={cluster_cfg.key}, model={model_cfg.key}, error={message}")
                if not args.continue_on_error:
                    raise

    if not metrics_rows:
        raise RuntimeError("No successful model runs. Please check inputs/config.")

    metrics_long = pd.concat(metrics_rows, ignore_index=True)
    coverage_df = pd.concat(coverage_rows, ignore_index=True) if coverage_rows else pd.DataFrame()
    export_benchmark_reports(metrics_long=metrics_long, output_dir=run_dir, coverage_df=coverage_df)

    run_meta = {
        "run_tag": run_tag,
        "output_dir": str(run_dir),
        "clusters": [cfg.key for cfg in config.clusters],
        "models_enabled": [cfg.key for cfg in config.models if cfg.enabled],
        "train_end_year": config.train_end_year,
        "valid_year": config.valid_year,
        "test_year": config.test_year,
        "seed": config.seed,
        "include_era5_daily": config.include_era5_daily,
        "disable_shap": bool(args.disable_shap),
        "strict_core": bool(args.strict_core),
        "allow_missing_aux": bool(args.allow_missing_aux),
        "failed_runs": failed_runs,
    }
    with open(run_dir / "run_info.json", "w", encoding="utf-8") as file:
        json.dump(run_meta, file, ensure_ascii=False, indent=2)

    print("=" * 90)
    print("[INFO] Benchmark finished.")
    print(f"[INFO] Run directory: {run_dir}")
    print("[INFO] Generated:")
    print("       - metrics/benchmark_metrics_long.csv")
    print("       - metrics/benchmark_metrics_wide.csv")
    print("       - metrics/benchmark_test_rmse_ranking.csv")
    print("       - metrics/data_coverage_by_cluster.csv")
    print("       - plots/benchmark_test_metrics_grouped_bar.png")
    print("       - plots/benchmark_split_stability.png")
    print("       - run_info.json")
    if failed_runs:
        print(f"[WARN] Failed runs: {len(failed_runs)}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
