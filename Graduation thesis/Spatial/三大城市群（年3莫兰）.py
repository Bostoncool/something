from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda import Moran, Moran_BV, Moran_Local
from libpysal.weights import KNN, Queen


DEFAULT_GEOJSON_DIRS = {
    "BTH": r"H:\DATA Science\大论文Result\大论文图\2.京津冀\具体城市",
    "YRD": r"H:\DATA Science\大论文Result\大论文图\3.长三角\具体城市",
    "PRD": r"H:\DATA Science\大论文Result\大论文图\4.珠三角\具体城市",
}
DEFAULT_PM25_CSV = r"H:\DATA Science\大论文Result\三大城市群（市）月均PM2.5浓度\合并数据_2018-2023.csv"
FIG_DPI = 300


def normalize_city_name(name: str) -> str:
    text = str(name).strip().replace(" ", "")
    for token in ["特别行政区", "自治州", "地区", "盟", "市", "县", "区"]:
        text = text.replace(token, "")
    return text


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception:  # noqa: BLE001
            continue
    raise ValueError(f"CSV读取失败: {csv_path}")


def reshape_pm25_table(pm_df: pd.DataFrame) -> pd.DataFrame:
    city_col = next((c for c in pm_df.columns if "城市" in c or c.lower() == "city"), pm_df.columns[0])
    value_cols = [col for col in pm_df.columns if col != city_col]
    long_df = pm_df.melt(id_vars=[city_col], value_vars=value_cols, var_name="period", value_name="pm25")
    long_df = long_df.rename(columns={city_col: "city"})
    long_df["pm25"] = pd.to_numeric(long_df["pm25"], errors="coerce")
    long_df = long_df.dropna(subset=["city", "pm25"]).copy()
    long_df["city_norm"] = long_df["city"].map(normalize_city_name)
    long_df["period"] = long_df["period"].astype(str)
    return long_df


def build_annual_pm25(pm_long: pd.DataFrame) -> pd.DataFrame:
    annual = pm_long.copy()
    annual["year"] = annual["period"].str.extract(r"(\d{4})", expand=False)
    annual = annual.dropna(subset=["year"]).copy()
    annual = (
        annual.groupby(["city_norm", "year"], as_index=False)
        .agg(city=("city", "first"), pm25=("pm25", "mean"))
        .rename(columns={"year": "period"})
    )
    annual["period"] = annual["period"].astype(str)
    return annual


def read_cluster_geojson(cluster_key: str, geojson_dir: Path) -> gpd.GeoDataFrame:
    files = sorted(geojson_dir.glob("*.geojson"))
    if not files:
        raise ValueError(f"{cluster_key} 未找到 GeoJSON 文件: {geojson_dir}")

    frames: list[gpd.GeoDataFrame] = []
    for file_path in files:
        gdf = gpd.read_file(file_path)
        if gdf.empty:
            continue
        if "name" in gdf.columns:
            gdf["city"] = gdf["name"]
        else:
            gdf["city"] = file_path.stem
        gdf["city_norm"] = gdf["city"].map(normalize_city_name)
        frames.append(gdf[["city", "city_norm", "geometry"]].copy())

    if not frames:
        raise ValueError(f"{cluster_key} GeoJSON 无有效几何: {geojson_dir}")
    merged = pd.concat(frames, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=frames[0].crs)
    merged = merged.drop_duplicates(subset=["city_norm"], keep="first")
    return merged


def build_weights(gdf: gpd.GeoDataFrame) -> Queen:
    w = Queen.from_dataframe(gdf, ids=gdf["city_norm"].tolist(), silence_warnings=True)
    islands = list(w.islands)
    if islands:
        knn = KNN.from_dataframe(gdf, k=min(3, len(gdf) - 1), ids=gdf["city_norm"].tolist())
        for island in islands:
            w.neighbors[island] = knn.neighbors[island]
            w.weights[island] = knn.weights[island]
    w.transform = "r"
    return w


def safe_stat(value: float, default: float) -> float:
    return default if (pd.isna(value) or np.isinf(value)) else float(value)


def sort_periods(periods: list[str]) -> list[str]:
    def _key(p: str) -> tuple[int, str]:
        return (0, f"{int(p):010d}") if p.isdigit() else (1, p)

    return sorted(periods, key=_key)


def lisa_type(q: int, p_val: float, alpha: float) -> str:
    if p_val >= alpha:
        return "不显著"
    mapping = {1: "高-高", 2: "低-高", 3: "低-低", 4: "高-低"}
    return mapping.get(int(q), "不显著")


def plot_lisa_cluster_map(gdf_plot: gpd.GeoDataFrame, cluster_key: str, period: str, save_path: Path) -> None:
    color_map = {
        "高-高": "#d7191c",
        "低-低": "#2c7bb6",
        "高-低": "#fdae61",
        "低-高": "#abd9e9",
        "不显著": "#d9d9d9",
    }
    gdf_plot["plot_color"] = gdf_plot["lisa_type"].map(color_map).fillna("#d9d9d9")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    gdf_plot.plot(color=gdf_plot["plot_color"], edgecolor="#333333", linewidth=0.8, ax=ax)
    ax.set_axis_off()
    ax.set_title(f"{cluster_key} LISA聚类图 ({period})", fontsize=12)

    for key in ["高-高", "低-低", "高-低", "低-高", "不显著"]:
        ax.scatter([], [], color=color_map[key], label=key, s=80)
    ax.legend(loc="lower left", frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, format="svg", dpi=FIG_DPI, transparent=True)
    plt.close(fig)


def plot_pm25_map_with_global_text(
    gdf_plot: gpd.GeoDataFrame,
    cluster_key: str,
    period: str,
    global_i: float,
    global_z: float,
    global_p: float,
    stmi_lag1: Optional[dict],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    gdf_plot.plot(column="pm25", cmap="YlOrRd", edgecolor="#333333", linewidth=0.8, legend=True, ax=ax)
    ax.set_axis_off()
    ax.set_title(f"{cluster_key} PM2.5空间分布 ({period})", fontsize=12)

    text_lines = [
        f"Global Moran I = {global_i:.4f}",
        f"Global Z = {global_z:.4f}",
        f"Global P = {global_p:.4f}",
    ]
    if stmi_lag1 is not None:
        text_lines.extend(
            [
                f"STMI(lag=1) I = {stmi_lag1['stmi_i']:.4f}",
                f"STMI(lag=1) Z = {stmi_lag1['stmi_z']:.4f}",
                f"STMI(lag=1) P = {stmi_lag1['stmi_p']:.4f}",
            ]
        )
    ax.text(
        0.02,
        0.02,
        "\n".join(text_lines),
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    fig.tight_layout()
    fig.savefig(save_path, format="svg", dpi=FIG_DPI, transparent=True)
    plt.close(fig)


def compute_cluster_metrics(
    cluster_key: str,
    cluster_gdf: gpd.GeoDataFrame,
    pm_long: pd.DataFrame,
    periods_sorted: list[str],
    alpha: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cluster_dir = output_dir / cluster_key
    cluster_dir.mkdir(parents=True, exist_ok=True)

    global_records: list[dict] = []
    local_records: list[dict] = []
    stmi_records: list[dict] = []

    w = build_weights(cluster_gdf)
    city_order = cluster_gdf["city_norm"].tolist()

    period_to_vec: dict[str, np.ndarray] = {}
    period_to_gdf: dict[str, gpd.GeoDataFrame] = {}

    for period in periods_sorted:
        period_df = pm_long.loc[pm_long["period"] == period, ["city_norm", "pm25"]].copy()
        merged = cluster_gdf.merge(period_df, on="city_norm", how="left")
        merged = merged.dropna(subset=["pm25"]).copy()
        if merged.shape[0] < 4:
            continue
        if sorted(merged["city_norm"].tolist()) != sorted(city_order):
            merged = merged.set_index("city_norm").reindex(city_order).reset_index()
            merged["city"] = merged["city"].fillna(merged["city_norm"])
        if merged["pm25"].isna().any():
            continue

        y = merged["pm25"].to_numpy(dtype=float)
        period_to_vec[period] = y
        period_to_gdf[period] = merged

        moran = Moran(y, w, permutations=999, two_tailed=True)
        global_z = safe_stat(getattr(moran, "z_sim", np.nan), safe_stat(getattr(moran, "z_norm", np.nan), 0.0))
        global_p = safe_stat(getattr(moran, "p_sim", np.nan), safe_stat(getattr(moran, "p_norm", np.nan), 1.0))
        global_records.append(
            {
                "cluster": cluster_key,
                "period": period,
                "n_units": int(len(y)),
                "global_i": float(moran.I),
                "global_z": global_z,
                "global_p": global_p,
                "global_ei": float(moran.EI),
                "global_z_norm": float(moran.z_norm),
                "global_p_norm": float(moran.p_norm),
            }
        )

        lisa = Moran_Local(y, w, permutations=999)
        for idx, row in merged.reset_index(drop=True).iterrows():
            p_val = safe_stat(lisa.p_sim[idx], 1.0)
            z_val = safe_stat(lisa.z_sim[idx], 0.0)
            q_val = int(lisa.q[idx])
            local_records.append(
                {
                    "cluster": cluster_key,
                    "period": period,
                    "city": row["city"],
                    "city_norm": row["city_norm"],
                    "pm25": float(row["pm25"]),
                    "local_i": float(lisa.Is[idx]),
                    "local_z": z_val,
                    "local_p": p_val,
                    "quadrant": q_val,
                    "lisa_type": lisa_type(q_val, p_val, alpha),
                    "significant": bool(p_val < alpha),
                }
            )

    local_df_all = pd.DataFrame(local_records)
    global_df_all = pd.DataFrame(global_records)

    for lag in range(1, 8):
        for idx in range(lag, len(periods_sorted)):
            period_t = periods_sorted[idx]
            period_tlag = periods_sorted[idx - lag]
            if period_t not in period_to_vec or period_tlag not in period_to_vec:
                continue
            x_t = period_to_vec[period_t]
            y_tlag = period_to_vec[period_tlag]
            moran_bv = Moran_BV(x_t, y_tlag, w, permutations=999)
            stmi_z = safe_stat(getattr(moran_bv, "z_sim", np.nan), 0.0)
            stmi_p = safe_stat(getattr(moran_bv, "p_sim", np.nan), 1.0)
            stmi_records.append(
                {
                    "cluster": cluster_key,
                    "period_t": period_t,
                    "period_t_lag": period_tlag,
                    "lag": lag,
                    "n_units": int(len(x_t)),
                    "stmi_i": float(moran_bv.I),
                    "stmi_z": stmi_z,
                    "stmi_p": stmi_p,
                    "stmi_ei": float(moran_bv.EI_sim),
                }
            )

    stmi_df_all = pd.DataFrame(stmi_records)

    if not global_df_all.empty:
        for period in periods_sorted:
            if period not in period_to_gdf:
                continue

            gdf_period = period_to_gdf[period].copy()
            local_period = local_df_all.loc[
                (local_df_all["cluster"] == cluster_key) & (local_df_all["period"] == period),
                ["city_norm", "lisa_type"],
            ].copy()
            gdf_period = gdf_period.merge(local_period, on="city_norm", how="left")
            gdf_period["lisa_type"] = gdf_period["lisa_type"].fillna("不显著")

            stmi_lag1 = None
            if not stmi_df_all.empty:
                stmi_lag1_df = stmi_df_all.loc[
                    (stmi_df_all["cluster"] == cluster_key)
                    & (stmi_df_all["period_t"] == period)
                    & (stmi_df_all["lag"] == 1)
                ]
                if not stmi_lag1_df.empty:
                    stmi_lag1 = stmi_lag1_df.iloc[0].to_dict()

            g_row = global_df_all.loc[
                (global_df_all["cluster"] == cluster_key) & (global_df_all["period"] == period)
            ].iloc[0]

            plot_lisa_cluster_map(
                gdf_plot=gdf_period,
                cluster_key=cluster_key,
                period=period,
                save_path=cluster_dir / f"{cluster_key}_LISA_cluster_map_{period}.svg",
            )
            plot_pm25_map_with_global_text(
                gdf_plot=gdf_period,
                cluster_key=cluster_key,
                period=period,
                global_i=float(g_row["global_i"]),
                global_z=float(g_row["global_z"]),
                global_p=float(g_row["global_p"]),
                stmi_lag1=stmi_lag1,
                save_path=cluster_dir / f"{cluster_key}_PM25_map_with_global_STMI_{period}.svg",
            )

    return global_df_all, local_df_all, stmi_df_all


def run_pipeline(pm25_csv: Path, output_dir: Path, alpha: float) -> None:
    setup_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    pm_raw = read_csv_with_fallback(pm25_csv)
    pm_long = reshape_pm25_table(pm_raw)
    pm_annual = build_annual_pm25(pm_long)
    periods_sorted = sort_periods(pm_annual["period"].unique().tolist())

    global_list: list[pd.DataFrame] = []
    local_list: list[pd.DataFrame] = []
    stmi_list: list[pd.DataFrame] = []

    for cluster_key, geo_dir in DEFAULT_GEOJSON_DIRS.items():
        cluster_gdf = read_cluster_geojson(cluster_key, Path(geo_dir))
        global_df, local_df, stmi_df = compute_cluster_metrics(
            cluster_key=cluster_key,
            cluster_gdf=cluster_gdf,
            pm_long=pm_annual,
            periods_sorted=periods_sorted,
            alpha=alpha,
            output_dir=output_dir,
        )
        global_list.append(global_df)
        local_list.append(local_df)
        stmi_list.append(stmi_df)
        print(
            f"{cluster_key} completed | Global={len(global_df)} rows | "
            f"Local={len(local_df)} rows | STMI={len(stmi_df)} rows"
        )

    global_all = pd.concat(global_list, ignore_index=True) if global_list else pd.DataFrame()
    local_all = pd.concat(local_list, ignore_index=True) if local_list else pd.DataFrame()
    stmi_all = pd.concat(stmi_list, ignore_index=True) if stmi_list else pd.DataFrame()

    global_all.to_csv(output_dir / "global_moran_summary.csv", index=False, encoding="utf-8-sig")
    local_all.to_csv(output_dir / "local_moran_summary.csv", index=False, encoding="utf-8-sig")
    stmi_all.to_csv(output_dir / "stmi_summary.csv", index=False, encoding="utf-8-sig")

    summary_json = {
        "alpha": alpha,
        "global_rows": int(len(global_all)),
        "local_rows": int(len(local_all)),
        "stmi_rows": int(len(stmi_all)),
        "outputs": {
            "global": str(output_dir / "global_moran_summary.csv"),
            "local": str(output_dir / "local_moran_summary.csv"),
            "stmi": str(output_dir / "stmi_summary.csv"),
        },
    }
    with (output_dir / "moran_summary_meta.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary_json, file_obj, ensure_ascii=False, indent=2)

    print("All clusters finished.")
    print(f"period count: {len(periods_sorted)}")
    print("Outputs written.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="三大城市群年度PM2.5 全局莫兰 + 局部莫兰 + 时空莫兰计算与可视化"
    )
    parser.add_argument("--pm25-csv", type=Path, default=Path(DEFAULT_PM25_CSV), help="PM2.5数据CSV路径（可为月度，脚本自动聚合到年度）")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "三大城市群_（年）莫兰结果",
        help="输出目录",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(pm25_csv=args.pm25_csv, output_dir=args.output_dir, alpha=args.alpha)
