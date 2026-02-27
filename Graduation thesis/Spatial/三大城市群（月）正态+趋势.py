from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyproj import Transformer
from scipy import stats


DEFAULT_GEOJSON_DIRS = {
    "BTH": r"H:\DATA Science\大论文Result\大论文图\2.京津冀\具体城市",
    "YRD": r"H:\DATA Science\大论文Result\大论文图\3.长三角\具体城市",
    "PRD": r"H:\DATA Science\大论文Result\大论文图\4.珠三角\具体城市",
}
DEFAULT_PM25_CSV = r"H:\DATA Science\大论文Result\三大城市群（市）月均PM2.5浓度\合并数据_2018-2023.csv"
FIG_DPI = 300


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")


def normalize_city_name(name: str) -> str:
    if pd.isna(name):
        return ""
    cleaned = str(name).strip().replace(" ", "")
    for token in ["特别行政区", "自治州", "地区", "盟", "市", "县", "区"]:
        cleaned = cleaned.replace(token, "")
    return cleaned


def flatten_points(coords: Iterable) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if isinstance(coords, (list, tuple)) and len(coords) == 2 and all(
        isinstance(v, (int, float)) for v in coords
    ):
        return [(float(coords[0]), float(coords[1]))]
    if isinstance(coords, (list, tuple)):
        for item in coords:
            points.extend(flatten_points(item))
    return points


def read_geojson_city_centroids(geojson_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for file_path in sorted(geojson_dir.glob("*.geojson")):
        with file_path.open("r", encoding="utf-8") as file_obj:
            geo = json.load(file_obj)
        features = geo.get("features", [])
        if not features:
            continue

        feature = features[0]
        props = feature.get("properties", {})
        city_name = props.get("name", file_path.stem)
        centroid = props.get("centroid")
        center = props.get("center")

        lon_lat: Optional[tuple[float, float]] = None
        if isinstance(centroid, list) and len(centroid) == 2:
            lon_lat = (float(centroid[0]), float(centroid[1]))
        elif isinstance(center, list) and len(center) == 2:
            lon_lat = (float(center[0]), float(center[1]))
        else:
            points = flatten_points(feature.get("geometry", {}).get("coordinates", []))
            if points:
                lon_lat = (
                    float(np.mean([point[0] for point in points])),
                    float(np.mean([point[1] for point in points])),
                )

        if lon_lat is None:
            continue

        records.append(
            {
                "city": city_name,
                "city_norm": normalize_city_name(city_name),
                "lon": lon_lat[0],
                "lat": lon_lat[1],
            }
        )

    if not records:
        raise ValueError(f"未在目录读取到有效 GeoJSON 坐标信息: {geojson_dir}")
    return pd.DataFrame(records)


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise ValueError(f"CSV 读取失败: {csv_path}") from last_error


def detect_city_column(df: pd.DataFrame) -> str:
    priority_patterns = [r"city", r"城市", r"地市", r"地区", r"region", r"name"]
    for pattern in priority_patterns:
        candidates = [col for col in df.columns if pd.Series([col]).str.contains(pattern, case=False, regex=True)[0]]
        if candidates:
            return candidates[0]
    return df.columns[0]


def detect_pm_column(df: pd.DataFrame, city_col: str) -> Optional[str]:
    pm_candidates = [
        col
        for col in df.columns
        if col != city_col and pd.Series([col]).str.contains(r"pm\s*2\.?5|pm25|pm_?2_?5", case=False, regex=True)[0]
    ]
    return pm_candidates[0] if pm_candidates else None


def detect_period_column(df: pd.DataFrame, city_col: str, pm_col: Optional[str]) -> Optional[str]:
    excluded = {city_col}
    if pm_col is not None:
        excluded.add(pm_col)
    period_patterns = [r"month", r"年月", r"月份", r"date", r"time", r"period", r"year", r"年份"]
    for pattern in period_patterns:
        candidates = [
            col for col in df.columns if col not in excluded and pd.Series([col]).str.contains(pattern, case=False, regex=True)[0]
        ]
        if candidates:
            return candidates[0]
    return None


def reshape_pm25_table(pm_df: pd.DataFrame) -> pd.DataFrame:
    city_col = detect_city_column(pm_df)
    pm_col = detect_pm_column(pm_df, city_col)
    period_col = detect_period_column(pm_df, city_col, pm_col)

    if pm_col is not None:
        if period_col is None:
            long_df = pm_df[[city_col, pm_col]].copy()
            long_df["period"] = "ALL"
        else:
            long_df = pm_df[[city_col, period_col, pm_col]].copy()
            long_df = long_df.rename(columns={period_col: "period"})
        long_df = long_df.rename(columns={city_col: "city", pm_col: "pm25"})
    else:
        value_cols = [col for col in pm_df.columns if col != city_col]
        long_df = pm_df.melt(id_vars=[city_col], value_vars=value_cols, var_name="period", value_name="pm25")
        long_df = long_df.rename(columns={city_col: "city"})

    long_df["pm25"] = pd.to_numeric(long_df["pm25"], errors="coerce")
    long_df = long_df.dropna(subset=["city", "pm25"]).copy()
    long_df["city_norm"] = long_df["city"].map(normalize_city_name)
    long_df["period"] = long_df["period"].astype(str)
    return long_df


def choose_period(data_long: pd.DataFrame, period: Optional[str]) -> str:
    available = sorted(data_long["period"].dropna().unique().tolist())
    if not available:
        raise ValueError("PM2.5 数据中未识别到 period。")
    if period is None:
        return available[-1]
    if str(period) not in available:
        raise ValueError(f"指定 period={period} 不存在, 可选值: {available}")
    return str(period)


def select_pm25_for_analysis(pm_long: pd.DataFrame, period: Optional[str]) -> tuple[pd.DataFrame, str]:
    if period is None:
        grouped = (
            pm_long.groupby("city_norm", as_index=False)
            .agg(city=("city", "first"), pm25=("pm25", "mean"))
            .dropna(subset=["city_norm", "pm25"])
        )
        return grouped, "ALL_MONTHLY_MEAN"

    selected_period = choose_period(pm_long, period)
    period_df = pm_long.loc[pm_long["period"] == selected_period, ["city", "city_norm", "pm25"]].copy()
    return period_df, selected_period


def choose_utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def project_to_planar(df: pd.DataFrame) -> pd.DataFrame:
    mean_lon = df["lon"].mean()
    mean_lat = df["lat"].mean()
    epsg = choose_utm_epsg(mean_lon, mean_lat)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    out = df.copy()
    out["x"] = x
    out["y"] = y
    out["projected_crs"] = f"EPSG:{epsg}"
    return out


def shapiro_wilk_test(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return np.nan, np.nan
    if arr.size > 5000:
        rng = np.random.default_rng(42)
        arr = rng.choice(arr, size=5000, replace=False)
    stat, p_value = stats.shapiro(arr)
    return float(stat), float(p_value)


def normality_report(values: np.ndarray, alpha: float) -> dict:
    skew = float(stats.skew(values, bias=False))
    kurt = float(stats.kurtosis(values, fisher=True, bias=False))
    sw_stat, sw_p = shapiro_wilk_test(values)
    pass_skew_kurt = (abs(skew) < 2.0) and (abs(kurt) < 2.0)
    pass_sw = bool(sw_p >= alpha) if not np.isnan(sw_p) else False
    return {
        "skewness": skew,
        "kurtosis_excess": kurt,
        "sw_stat": sw_stat,
        "sw_p": sw_p,
        "pass_skew_kurt": pass_skew_kurt,
        "pass_sw": pass_sw,
        "pass_overall": pass_skew_kurt and pass_sw,
    }


def transform_and_recheck(values: np.ndarray, alpha: float) -> tuple[np.ndarray, str, dict]:
    base_report = normality_report(values, alpha)
    if base_report["pass_overall"]:
        return values, "original", base_report

    log_values = np.log1p(np.maximum(values, 0))
    log_report = normality_report(log_values, alpha)
    if log_report["pass_overall"]:
        return log_values, "log1p", log_report

    shift = 0.0
    if np.min(values) <= 0:
        shift = abs(float(np.min(values))) + 1e-6
    boxcox_values, boxcox_lambda = stats.boxcox(values + shift)
    boxcox_report = normality_report(boxcox_values, alpha)
    boxcox_report["boxcox_lambda"] = float(boxcox_lambda)
    boxcox_report["boxcox_shift"] = float(shift)

    candidates = [("log1p", log_values, log_report), ("boxcox", boxcox_values, boxcox_report)]
    best_name, best_values, best_report = max(
        candidates,
        key=lambda item: item[2]["sw_p"] if not np.isnan(item[2]["sw_p"]) else -1,
    )
    return best_values, best_name, best_report


def fit_linear_trend_plane(df: pd.DataFrame, value_col: str) -> dict:
    z = df[value_col].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    n = z.size
    p = 2
    if n <= p + 1:
        raise ValueError("样本量不足，无法做一阶趋势面 F 检验（至少 4 个空间单元）。")

    x_matrix = np.column_stack([np.ones(n), x, y])
    beta, _, _, _ = np.linalg.lstsq(x_matrix, z, rcond=None)
    y_hat = x_matrix @ beta
    residuals = z - y_hat

    sse = float(np.sum((z - y_hat) ** 2))
    sst = float(np.sum((z - np.mean(z)) ** 2))
    ssr = sst - sse
    r2 = 0.0 if sst == 0 else ssr / sst

    df_reg = p
    df_res = n - p - 1
    msr = ssr / df_reg
    mse = sse / df_res
    f_stat = np.inf if mse == 0 else msr / mse
    p_value = float(stats.f.sf(f_stat, df_reg, df_res))

    return {
        "beta0": float(beta[0]),
        "beta1_x": float(beta[1]),
        "beta2_y": float(beta[2]),
        "r2": float(r2),
        "f_stat": float(f_stat),
        "f_p": p_value,
        "residuals": residuals,
    }


def save_qq_plot(values: np.ndarray, title: str, save_path: Path) -> None:
    fig = plt.figure(figsize=(6, 6))
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0.0)
    stats.probplot(values, dist="norm", plot=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, transparent=True)
    plt.close(fig)


def save_trend_plots(df: pd.DataFrame, value_col: str, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_alpha(0.0)
    for ax, coord_col, label in zip(axes, ["x", "y"], ["X direction", "Y direction"]):
        ax.patch.set_alpha(0.0)
        sns.scatterplot(data=df, x=coord_col, y=value_col, ax=ax, s=55, color="#2c7fb8")
        coef = np.polyfit(df[coord_col], df[value_col], 1)
        x_line = np.linspace(df[coord_col].min(), df[coord_col].max(), 200)
        y_line = coef[0] * x_line + coef[1]
        ax.plot(x_line, y_line, color="#d7301f", linewidth=2)
        ax.set_box_aspect(1)
        ax.set_xlabel(f"{label} projected coordinate (m)")
        ax.set_ylabel("PM2.5 (transformed)" if value_col == "pm25_for_test" else "PM2.5")
        ax.set_title(f"{label} trend")
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, transparent=True)
    plt.close(fig)


def run_single_cluster(
    cluster_key: str,
    geojson_dir: Path,
    base_output_dir: Path,
    analysis_df: pd.DataFrame,
    period_label: str,
    alpha: float,
) -> dict:
    cluster_output = base_output_dir / cluster_key
    cluster_output.mkdir(parents=True, exist_ok=True)

    geo_df = read_geojson_city_centroids(geojson_dir)
    merged = geo_df.merge(analysis_df, on="city_norm", how="inner").drop_duplicates(subset=["city_norm"], keep="first")
    if merged.shape[0] < 4:
        raise ValueError(f"{cluster_key} 匹配后的有效城市数不足 4。")

    merged = project_to_planar(merged)
    original_values = merged["pm25"].to_numpy(dtype=float)
    transformed_values, transform_name, transform_report = transform_and_recheck(original_values, alpha)
    merged["pm25_for_test"] = transformed_values

    normality_before = normality_report(original_values, alpha)
    trend_result = fit_linear_trend_plane(merged, "pm25_for_test")
    trend_sig = trend_result["f_p"] < alpha
    merged["trend_residual"] = trend_result["residuals"]
    merged["moran_input"] = merged["trend_residual"] if trend_sig else merged["pm25_for_test"]
    merged["cluster"] = cluster_key
    merged["period"] = period_label

    qq_orig = cluster_output / "qqplot_original.svg"
    qq_final = cluster_output / "qqplot_for_trend.svg"
    trend_plot = cluster_output / "trend_xy.svg"
    save_qq_plot(original_values, f"Q-Q Plot ({cluster_key} Original)", qq_orig)
    save_qq_plot(transformed_values, f"Q-Q Plot ({cluster_key} {transform_name})", qq_final)
    save_trend_plots(merged, "pm25_for_test", trend_plot)

    merged_path = cluster_output / f"{cluster_key}_normality_trend_result.csv"
    report_path = cluster_output / f"{cluster_key}_report.json"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

    report = {
        "cluster": cluster_key,
        "period": period_label,
        "alpha": alpha,
        "n_units": int(merged.shape[0]),
        "normality_before": normality_before,
        "transform_selected": transform_name,
        "normality_after": transform_report,
        "trend_linear_plane": {
            "beta0": trend_result["beta0"],
            "beta1_x": trend_result["beta1_x"],
            "beta2_y": trend_result["beta2_y"],
            "r2": trend_result["r2"],
            "f_stat": trend_result["f_stat"],
            "f_p": trend_result["f_p"],
            "trend_significant": bool(trend_sig),
            "trend_decision": "trend significant, use residual for moran input"
            if trend_sig
            else "trend not significant, use transformed/original value for moran input",
        },
        "projected_crs": merged["projected_crs"].iloc[0],
        "outputs": {
            "merged_data": str(merged_path),
            "report_json": str(report_path),
            "qqplot_original": str(qq_orig),
            "qqplot_final": str(qq_final),
            "trend_plot": str(trend_plot),
        },
    }
    with report_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)

    return {
        "cluster": cluster_key,
        "period": period_label,
        "n_units": int(merged.shape[0]),
        "transform_selected": transform_name,
        "normality_sw_p": transform_report["sw_p"],
        "trend_f_p": trend_result["f_p"],
        "trend_r2": trend_result["r2"],
        "trend_significant": bool(trend_sig),
        "moran_input_field": "trend_residual" if trend_sig else "pm25_for_test",
        "output_dir": str(cluster_output),
    }


def run_all_clusters(
    pm25_csv: Path,
    output_dir: Path,
    period: Optional[str],
    alpha: float,
    selected_clusters: list[str],
) -> None:
    setup_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    pm_raw = read_csv_with_fallback(pm25_csv)
    pm_long = reshape_pm25_table(pm_raw)
    analysis_df, period_label = select_pm25_for_analysis(pm_long, period)

    summaries: list[dict] = []
    for cluster_key in selected_clusters:
        geojson_dir = Path(DEFAULT_GEOJSON_DIRS[cluster_key])
        summary = run_single_cluster(
            cluster_key=cluster_key,
            geojson_dir=geojson_dir,
            analysis_df=analysis_df,
            base_output_dir=output_dir,
            period_label=period_label,
            alpha=alpha,
        )
        summaries.append(summary)
        print(
            f"{cluster_key} completed | n={summary['n_units']} | "
            f"transform={summary['transform_selected']} | trend_p={summary['trend_f_p']:.6f}"
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "three_clusters_summary.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "three_clusters_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summaries, file_obj, ensure_ascii=False, indent=2)

    print("All clusters completed.")
    print(f"period: {period_label}")
    print("summary files have been written.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="三大城市群 PM2.5: 正态性检验 + 非正态校正 + 一阶空间趋势面分析"
    )
    parser.add_argument("--pm25-csv", type=Path, default=Path(DEFAULT_PM25_CSV), help="PM2.5 CSV 路径")
    parser.add_argument(
        "--period",
        type=str,
        default=None,
        help="指定时间字段, 例如 202312；不指定时默认使用全部月份的城市均值",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平")
    parser.add_argument(
        "--clusters",
        type=str,
        default="BTH,YRD,PRD",
        help="要运行的城市群, 逗号分隔, 可选 BTH,YRD,PRD",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "三大城市群_正态趋势结果",
        help="输出目录",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clusters = [item.strip().upper() for item in args.clusters.split(",") if item.strip()]
    invalid = [item for item in clusters if item not in DEFAULT_GEOJSON_DIRS]
    if invalid:
        raise ValueError(f"无效城市群标识: {invalid}. 可选值为 BTH,YRD,PRD")
    run_all_clusters(
        pm25_csv=args.pm25_csv,
        output_dir=args.output_dir,
        period=args.period,
        alpha=args.alpha,
        selected_clusters=clusters,
    )
