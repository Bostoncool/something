from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import cm, colors
from esda import Moran, Moran_BV, Moran_Local
from libpysal.weights import KNN, Queen
from shapely.geometry import box


DEFAULT_GEOJSON_DIRS = {
    "BTH": r"H:\DATA Science\大论文Result\大论文图\2.京津冀\具体城市（区分辨率）",
    "YRD": r"H:\DATA Science\大论文Result\大论文图\3.长三角\具体城市（区分辨率）",
    "PRD": r"H:\DATA Science\大论文Result\大论文图\4.珠三角\具体城市（区分辨率）",
}
DEFAULT_CHINA_GEOJSON = r"H:\DATA Science\大论文Result\大论文图\1.总图\中国（市）.geojson"
DEFAULT_PM25_NC_DIR = r"G:\2000-2023[PM2.5-china]\Year"
DEFAULT_YEARS = list(range(2018, 2024))
FIG_DPI = 300
CHINA_LCC_PROJ4 = (
    "+proj=lcc +lat_1=25 +lat_2=47 +lat_0=35 +lon_0=105 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
# Keep main map focused on mainland China, but leave extra southern margin
# so Hainan is never clipped at the frame edge.
MAIN_CHINA_EXTENT_WGS84 = (73, 14, 136, 54)


def normalize_city_name(name: str) -> str:
    text = str(name).strip().replace(" ", "")
    for token in ["特别行政区", "自治州", "地区", "盟", "市", "县", "区"]:
        text = text.replace(token, "")
    return text


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


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


def read_china_geojson(china_geojson_path: Path) -> gpd.GeoDataFrame:
    if not china_geojson_path.exists():
        raise FileNotFoundError(f"中国底图不存在: {china_geojson_path}")
    china_gdf = gpd.read_file(china_geojson_path)
    if china_gdf.empty:
        raise ValueError(f"中国底图为空: {china_geojson_path}")
    if china_gdf.crs is None:
        china_gdf = china_gdf.set_crs("EPSG:4326", allow_override=True)
    return china_gdf


def _to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs("EPSG:4326", allow_override=True)
    if str(gdf.crs).upper() == "EPSG:4326":
        return gdf
    return gdf.to_crs("EPSG:4326")


def _prepare_projected_layers(
    china_gdf: gpd.GeoDataFrame,
    focus_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    china_wgs84 = _to_wgs84(china_gdf)
    focus_wgs84 = _to_wgs84(focus_gdf)
    china_proj = china_wgs84.to_crs(CHINA_LCC_PROJ4)
    focus_proj = focus_wgs84.to_crs(CHINA_LCC_PROJ4)
    return china_wgs84, focus_wgs84, china_proj, focus_proj


def _set_china_main_extent(ax: plt.Axes) -> None:
    extent_polygon = gpd.GeoDataFrame(
        geometry=[box(*MAIN_CHINA_EXTENT_WGS84)],
        crs="EPSG:4326",
    ).to_crs(CHINA_LCC_PROJ4)
    min_x, min_y, max_x, max_y = extent_polygon.total_bounds
    if not np.isfinite([min_x, min_y, max_x, max_y]).all():
        # Fallback to a safe projected extent if geometry transformation fails unexpectedly.
        extent_polygon = gpd.GeoDataFrame(
            geometry=[box(73, 14, 136, 54)],
            crs="EPSG:4326",
        ).to_crs(CHINA_LCC_PROJ4)
        min_x, min_y, max_x, max_y = extent_polygon.total_bounds
    pad_x = (max_x - min_x) * 0.03
    pad_y = (max_y - min_y) * 0.03
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)


def _add_south_china_sea_inset(
    fig: plt.Figure,
    china_wgs84: gpd.GeoDataFrame,
    focus_wgs84: gpd.GeoDataFrame,
    mode: str,
    color_map: Optional[dict[str, str]] = None,
    pm25_vmin: Optional[float] = None,
    pm25_vmax: Optional[float] = None,
) -> None:
    south_china_bbox = box(105, 3, 125, 25)
    china_south = gpd.clip(china_wgs84, south_china_bbox)
    focus_south = gpd.clip(focus_wgs84, south_china_bbox)
    if china_south.empty and focus_south.empty:
        return

    china_south_proj = china_south.to_crs(CHINA_LCC_PROJ4) if not china_south.empty else china_south
    focus_south_proj = focus_south.to_crs(CHINA_LCC_PROJ4) if not focus_south.empty else focus_south

    inset_ax = fig.add_axes([0.08, 0.11, 0.14, 0.21])
    inset_ax.set_facecolor("white")

    if not china_south_proj.empty:
        china_south_proj.plot(ax=inset_ax, color="#f7f7f7", edgecolor="#9f9f9f", linewidth=0.3, zorder=1)

    if not focus_south_proj.empty:
        if mode == "lisa":
            focus_draw = focus_south_proj.copy()
            focus_draw["plot_color"] = focus_draw["lisa_type"].map(color_map or {}).fillna("#d0d0d0")
            focus_draw.plot(ax=inset_ax, color=focus_draw["plot_color"], edgecolor="#2f2f2f", linewidth=0.45, zorder=2)
        else:
            focus_south_proj.plot(
                ax=inset_ax,
                column="pm25",
                cmap="YlOrRd",
                vmin=pm25_vmin,
                vmax=pm25_vmax,
                edgecolor="#2f2f2f",
                linewidth=0.45,
                zorder=2,
            )

    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_title("South China Sea", fontsize=7, pad=2)
    for spine in inset_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#666666")


def export_lisa_legend_svg(save_path: Path) -> None:
    color_map = {
        "高-高": "#b2182b",
        "低-低": "#2166ac",
        "高-低": "#ef8a62",
        "低-高": "#67a9cf",
        "不显著": "#d0d0d0",
    }
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.set_axis_off()
    for key in ["高-高", "低-低", "高-低", "低-高", "不显著"]:
        ax.scatter([], [], color=color_map[key], label=key, s=80)
    legend = ax.legend(
        loc="center",
        frameon=True,
        title="LISA 类型",
        borderpad=0.6,
        labelspacing=0.5,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor("#c5c5c5")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    fig.savefig(save_path, format="svg", dpi=FIG_DPI, transparent=True)
    plt.close(fig)


def export_pm25_colorbar_svg(save_path: Path, pm25_vmin: float, pm25_vmax: float) -> None:
    fig = plt.figure(figsize=(1.4, 4.2))
    cax = fig.add_axes([0.42, 0.07, 0.22, 0.9])
    norm = colors.Normalize(vmin=pm25_vmin, vmax=pm25_vmax)
    sm = cm.ScalarMappable(norm=norm, cmap="YlOrRd")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("PM2.5 浓度")
    fig.savefig(save_path, format="svg", dpi=FIG_DPI, transparent=True)
    plt.close(fig)


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


def find_yearly_nc_file(nc_dir: Path, year: int) -> Path:
    exact_name = nc_dir / f"CHAP_PM2.5_Y1K_{year}_V4.nc"
    if exact_name.exists():
        return exact_name
    candidates = sorted(nc_dir.glob(f"*{year}*.nc"))
    if not candidates:
        raise FileNotFoundError(f"未找到年份 {year} 的 NC 文件: {nc_dir}")
    return candidates[0]


def _standardize_nc_coords(ds: xr.Dataset) -> xr.Dataset:
    rename_map: dict[str, str] = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)
    try:
        ds = xr.decode_cf(ds)
    except Exception:  # noqa: BLE001
        pass
    return ds


def _choose_pm25_var(ds: xr.Dataset) -> str:
    var_names = list(ds.data_vars)
    if not var_names:
        raise ValueError("NC 文件中未找到数据变量。")
    for name in var_names:
        if re.search(r"pm\s*2\.?5|pm25", name, flags=re.IGNORECASE):
            return name
    for name in var_names:
        dims = set(ds[name].dims)
        if {"latitude", "longitude"}.issubset(dims):
            return name
    return var_names[0]


def _extract_2d_pm25(da: xr.DataArray) -> xr.DataArray:
    data = da
    while data.ndim > 2:
        dim_name = data.dims[0]
        data = data.mean(dim=dim_name, skipna=True)
    if data.ndim != 2:
        raise ValueError(f"PM2.5变量维度异常: {da.dims}")
    if "latitude" not in data.dims or "longitude" not in data.dims:
        raise ValueError(f"PM2.5变量缺少经纬度维度: {data.dims}")
    return data


def extract_pm25_by_city_from_nc(nc_file: Path, cluster_gdf_wgs84: gpd.GeoDataFrame) -> pd.DataFrame:
    with xr.open_dataset(nc_file, engine="netcdf4", decode_times=True) as ds:
        ds = _standardize_nc_coords(ds)
        pm_var = _choose_pm25_var(ds)
        da2d = _extract_2d_pm25(ds[pm_var]).load()

    lon = da2d["longitude"].to_numpy()
    lat = da2d["latitude"].to_numpy()
    val = da2d.to_numpy().astype(float, copy=False)

    lat_is_desc = lat[0] > lat[-1]
    lat_min = float(cluster_gdf_wgs84.total_bounds[1])
    lat_max = float(cluster_gdf_wgs84.total_bounds[3])
    lon_min = float(cluster_gdf_wgs84.total_bounds[0])
    lon_max = float(cluster_gdf_wgs84.total_bounds[2])

    if lat_is_desc:
        lat_idx = np.where((lat <= lat_max) & (lat >= lat_min))[0]
    else:
        lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(f"NC与城市范围无重叠: {nc_file.name}")

    lat_sub = lat[lat_idx]
    lon_sub = lon[lon_idx]
    val_sub = val[np.ix_(lat_idx, lon_idx)]

    points = gpd.GeoDataFrame(
        {"pm25": val_sub.ravel()},
        geometry=gpd.points_from_xy(
            np.tile(lon_sub, len(lat_sub)),
            np.repeat(lat_sub, len(lon_sub)),
        ),
        crs="EPSG:4326",
    )
    points = points.dropna(subset=["pm25"]).copy()
    joined = gpd.sjoin(
        points,
        cluster_gdf_wgs84[["cluster", "city", "city_norm", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        raise ValueError(f"NC像元未匹配到城市几何: {nc_file.name}")

    city_pm = (
        joined.groupby(["cluster", "city_norm", "city"], as_index=False)
        .agg(pm25=("pm25", "mean"))
        .dropna(subset=["pm25"])
    )
    return city_pm


def build_pm25_long_from_nc(nc_dir: Path, years: list[int], cluster_geo: dict[str, gpd.GeoDataFrame]) -> pd.DataFrame:
    all_clusters = pd.concat(
        [gdf[["city", "city_norm", "geometry"]].assign(cluster=key) for key, gdf in cluster_geo.items()],
        ignore_index=True,
    )
    all_clusters = gpd.GeoDataFrame(all_clusters, geometry="geometry", crs=next(iter(cluster_geo.values())).crs)
    all_clusters_wgs84 = all_clusters.to_crs("EPSG:4326")

    frames: list[pd.DataFrame] = []
    for year in years:
        nc_file = find_yearly_nc_file(nc_dir, year)
        city_pm = extract_pm25_by_city_from_nc(nc_file, all_clusters_wgs84)
        city_pm["period"] = str(year)
        frames.append(city_pm[["cluster", "city", "city_norm", "period", "pm25"]].copy())
        print(f"PM2.5 NC loaded | year={year} | file={nc_file.name} | cities={len(city_pm)}")

    if not frames:
        raise ValueError("未生成任何年度 PM2.5 数据。")
    return pd.concat(frames, ignore_index=True)


def compute_cluster_metrics(
    cluster_key: str,
    cluster_gdf: gpd.GeoDataFrame,
    pm_long: pd.DataFrame,
    periods_sorted: list[str],
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_records: list[dict] = []
    local_records: list[dict] = []
    stmi_records: list[dict] = []

    w = build_weights(cluster_gdf)
    city_order = cluster_gdf["city_norm"].tolist()

    period_to_vec: dict[str, np.ndarray] = {}

    for period in periods_sorted:
        period_df = pm_long.loc[
            (pm_long["period"] == period) & (pm_long["cluster"] == cluster_key),
            ["city_norm", "pm25"],
        ].copy()
        period_df = period_df.groupby("city_norm", as_index=False).agg(pm25=("pm25", "mean"))
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
    return global_df_all, local_df_all, stmi_df_all


def plot_yearly_lisa_on_china(
    gdf_period: gpd.GeoDataFrame,
    china_gdf: gpd.GeoDataFrame,
    period: str,
    save_path: Path,
) -> None:
    color_map = {
        "高-高": "#b2182b",
        "低-低": "#2166ac",
        "高-低": "#ef8a62",
        "低-高": "#67a9cf",
        "不显著": "#d0d0d0",
    }
    gdf_plot = gdf_period.copy()
    gdf_plot["plot_color"] = gdf_plot["lisa_type"].map(color_map).fillna("#d0d0d0")
    china_wgs84, focus_wgs84, china_proj, gdf_plot_proj = _prepare_projected_layers(china_gdf, gdf_plot)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    china_proj.plot(ax=ax, color="#f7f7f7", edgecolor="#b0b0b0", linewidth=0.35, zorder=1)
    gdf_plot_proj.plot(ax=ax, color=gdf_plot_proj["plot_color"], edgecolor="#2f2f2f", linewidth=0.75, zorder=2)
    _set_china_main_extent(ax)
    ax.set_axis_off()
    ax.set_title(f"三大城市群 LISA 聚类分布（{period}）", fontsize=16, pad=5, fontweight="bold")
    _add_south_china_sea_inset(
        fig=fig,
        china_wgs84=china_wgs84,
        focus_wgs84=focus_wgs84,
        mode="lisa",
        color_map=color_map,
    )

    fig.subplots_adjust(left=0.01, right=0.995, top=0.962, bottom=0.01)
    fig.savefig(save_path, format="svg", dpi=FIG_DPI, transparent=True)
    plt.close(fig)


def plot_yearly_pm25_on_china(
    gdf_period: gpd.GeoDataFrame,
    china_gdf: gpd.GeoDataFrame,
    period: str,
    pm25_vmin: float,
    pm25_vmax: float,
    save_path: Path,
) -> None:
    china_wgs84, focus_wgs84, china_proj, gdf_period_proj = _prepare_projected_layers(china_gdf, gdf_period)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    china_proj.plot(ax=ax, color="#f7f7f7", edgecolor="#b0b0b0", linewidth=0.35, zorder=1)
    gdf_period_proj.plot(
        column="pm25",
        cmap="YlOrRd",
        vmin=pm25_vmin,
        vmax=pm25_vmax,
        edgecolor="#2f2f2f",
        linewidth=0.75,
        legend=False,
        ax=ax,
        zorder=2,
    )
    _set_china_main_extent(ax)
    ax.set_axis_off()
    ax.set_title(f"三大城市群 PM2.5 空间分布（{period}）", fontsize=16, pad=5, fontweight="bold")
    _add_south_china_sea_inset(
        fig=fig,
        china_wgs84=china_wgs84,
        focus_wgs84=focus_wgs84,
        mode="pm25",
        pm25_vmin=pm25_vmin,
        pm25_vmax=pm25_vmax,
    )

    fig.subplots_adjust(left=0.01, right=0.995, top=0.962, bottom=0.01)
    fig.savefig(save_path, format="svg", dpi=FIG_DPI, transparent=True)
    plt.close(fig)


def _build_all_cluster_geometry(cluster_geo: dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    merged = pd.concat(
        [gdf[["city", "city_norm", "geometry"]].assign(cluster=key) for key, gdf in cluster_geo.items()],
        ignore_index=True,
    )
    all_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=next(iter(cluster_geo.values())).crs)
    return all_gdf


def plot_yearly_combined_maps(
    output_dir: Path,
    china_gdf: gpd.GeoDataFrame,
    all_clusters_gdf: gpd.GeoDataFrame,
    pm_long: pd.DataFrame,
    local_all: pd.DataFrame,
    global_all: pd.DataFrame,
    stmi_all: pd.DataFrame,
    periods_sorted: list[str],
) -> None:
    yearly_dir = output_dir / "China_combined_yearly_maps"
    yearly_dir.mkdir(parents=True, exist_ok=True)
    pm25_non_null = pm_long["pm25"].dropna()
    if pm25_non_null.empty:
        raise ValueError("PM2.5 数据为空，无法计算统一色带范围。")
    pm25_global_vmin = float(pm25_non_null.min())
    pm25_global_vmax = float(pm25_non_null.max())
    export_lisa_legend_svg(yearly_dir / "LISA_legend.svg")
    export_pm25_colorbar_svg(
        save_path=yearly_dir / "PM25_colorbar.svg",
        pm25_vmin=pm25_global_vmin,
        pm25_vmax=pm25_global_vmax,
    )

    if china_gdf.crs != all_clusters_gdf.crs:
        china_gdf = china_gdf.to_crs(all_clusters_gdf.crs)

    for period in periods_sorted:
        pm_period = (
            pm_long.loc[pm_long["period"] == period, ["cluster", "city_norm", "pm25"]]
            .groupby(["cluster", "city_norm"], as_index=False)
            .agg(pm25=("pm25", "mean"))
        )
        local_period = local_all.loc[
            local_all["period"] == period,
            ["cluster", "city_norm", "lisa_type"],
        ].copy()

        gdf_period = all_clusters_gdf.merge(pm_period, on=["cluster", "city_norm"], how="left")
        gdf_period = gdf_period.merge(local_period, on=["cluster", "city_norm"], how="left")
        gdf_period = gdf_period.dropna(subset=["pm25"]).copy()
        if gdf_period.empty:
            continue
        gdf_period["lisa_type"] = gdf_period["lisa_type"].fillna("不显著")

        plot_yearly_lisa_on_china(
            gdf_period=gdf_period,
            china_gdf=china_gdf,
            period=period,
            save_path=yearly_dir / f"China_3clusters_LISA_{period}.svg",
        )
        plot_yearly_pm25_on_china(
            gdf_period=gdf_period,
            china_gdf=china_gdf,
            period=period,
            pm25_vmin=pm25_global_vmin,
            pm25_vmax=pm25_global_vmax,
            save_path=yearly_dir / f"China_3clusters_PM25_with_global_STMI_{period}.svg",
        )


def run_pipeline(
    pm25_nc_dir: Path,
    china_geojson_path: Path,
    output_dir: Path,
    alpha: float,
    years: list[int],
) -> None:
    setup_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_geo = {
        cluster_key: read_cluster_geojson(cluster_key, Path(geo_dir))
        for cluster_key, geo_dir in DEFAULT_GEOJSON_DIRS.items()
    }
    china_gdf = read_china_geojson(china_geojson_path)

    pm_long = build_pm25_long_from_nc(pm25_nc_dir, years, cluster_geo)
    periods_sorted = sort_periods(pm_long["period"].unique().tolist())

    global_list: list[pd.DataFrame] = []
    local_list: list[pd.DataFrame] = []
    stmi_list: list[pd.DataFrame] = []

    for cluster_key, cluster_gdf in cluster_geo.items():
        global_df, local_df, stmi_df = compute_cluster_metrics(
            cluster_key=cluster_key,
            cluster_gdf=cluster_gdf,
            pm_long=pm_long,
            periods_sorted=periods_sorted,
            alpha=alpha,
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

    all_clusters_gdf = _build_all_cluster_geometry(cluster_geo)
    plot_yearly_combined_maps(
        output_dir=output_dir,
        china_gdf=china_gdf,
        all_clusters_gdf=all_clusters_gdf,
        pm_long=pm_long,
        local_all=local_all,
        global_all=global_all,
        stmi_all=stmi_all,
        periods_sorted=periods_sorted,
    )

    summary_json = {
        "alpha": alpha,
        "years": years,
        "china_geojson": str(china_geojson_path),
        "global_rows": int(len(global_all)),
        "local_rows": int(len(local_all)),
        "stmi_rows": int(len(stmi_all)),
        "outputs": {
            "global": str(output_dir / "global_moran_summary.csv"),
            "local": str(output_dir / "local_moran_summary.csv"),
            "stmi": str(output_dir / "stmi_summary.csv"),
            "yearly_maps_dir": str(output_dir / "China_combined_yearly_maps"),
        },
    }
    with (output_dir / "moran_summary_meta.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary_json, file_obj, ensure_ascii=False, indent=2)

    print("All clusters finished.")
    print(f"period count: {len(periods_sorted)}")
    print("Outputs written.")


def parse_years(years_text: str) -> list[int]:
    years: list[int] = []
    for item in str(years_text).split(","):
        val = item.strip()
        if not val:
            continue
        years.append(int(val))
    if not years:
        raise ValueError("years 为空。")
    return sorted(set(years))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="三大城市群年度 PM2.5 (NC) 全局莫兰 + 局部莫兰 + 时空莫兰计算，按年份输出中国底图总图"
    )
    parser.add_argument("--pm25-nc-dir", type=Path, default=Path(DEFAULT_PM25_NC_DIR), help="年度 PM2.5 NC 目录")
    parser.add_argument(
        "--china-geojson-path",
        type=Path,
        default=Path(DEFAULT_CHINA_GEOJSON),
        help="中国区域底图 GeoJSON 路径",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2018,2019,2020,2021,2022,2023",
        help="分析年份，逗号分隔",
    )
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
    run_pipeline(
        pm25_nc_dir=args.pm25_nc_dir,
        china_geojson_path=args.china_geojson_path,
        output_dir=args.output_dir,
        alpha=args.alpha,
        years=parse_years(args.years),
    )
