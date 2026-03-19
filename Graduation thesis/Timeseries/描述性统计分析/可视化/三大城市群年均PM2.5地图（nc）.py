from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


# 与参考脚本保持一致：不改变数据读取来源
DEFAULT_GEOJSON_DIRS = {
    "BTH": r"H:\大论文Result\大论文图\2.京津冀\具体城市（区分辨率）",
    "YRD": r"H:\大论文Result\大论文图\3.长三角\具体城市（区分辨率）",
    "PRD": r"H:\大论文Result\大论文图\4.珠三角\具体城市（区分辨率）",
}
DEFAULT_PM25_NC_DIR = r"G:\2000-2023[PM2.5-china]\Year"
DEFAULT_YEARS = list(range(2018, 2024))

FIG_DPI = 300

# 参考 Ribbon.py 的 12 色梯度（深蓝 -> 深红）
RIBBON_COLORS = [
    "#003366",
    "#3366CC",
    "#6699CC",
    "#99CCFF",
    "#CCFFFF",
    "#CCFFCC",
    "#99CC99",
    "#99CC66",
    "#FFFF99",
    "#FFCC66",
    "#FF6666",
    "#CC3333",
]

CLUSTER_ORDER = ["BTH", "YRD", "PRD"]
CLUSTER_NAME_ZH = {"BTH": "京津冀", "YRD": "长三角", "PRD": "珠三角"}


def setup_matplotlib() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False


def normalize_city_name(name: str) -> str:
    text = str(name).strip().replace(" ", "")
    for token in ["特别行政区", "自治州", "地区", "盟", "市", "县", "区"]:
        text = text.replace(token, "")
    return text


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
        data = data.mean(dim=data.dims[0], skipna=True)
    if data.ndim != 2:
        raise ValueError(f"PM2.5变量维度异常: {da.dims}")
    if "latitude" not in data.dims or "longitude" not in data.dims:
        raise ValueError(f"PM2.5变量缺少经纬度维度: {data.dims}")
    return data


def extract_pm25_by_city_from_nc(nc_file: Path, all_clusters_wgs84: gpd.GeoDataFrame) -> pd.DataFrame:
    with xr.open_dataset(nc_file, engine="netcdf4", decode_times=True) as ds:
        ds = _standardize_nc_coords(ds)
        pm_var = _choose_pm25_var(ds)
        da2d = _extract_2d_pm25(ds[pm_var]).load()

    lon = da2d["longitude"].to_numpy()
    lat = da2d["latitude"].to_numpy()
    val = da2d.to_numpy().astype(float, copy=False)

    lat_is_desc = lat[0] > lat[-1]
    lon_min, lat_min, lon_max, lat_max = all_clusters_wgs84.total_bounds

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
        all_clusters_wgs84[["cluster", "city", "city_norm", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        raise ValueError(f"NC像元未匹配到城市几何: {nc_file.name}")

    return (
        joined.groupby(["cluster", "city_norm", "city"], as_index=False)
        .agg(pm25=("pm25", "mean"))
        .dropna(subset=["pm25"])
    )


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
        city_pm["year"] = int(year)
        frames.append(city_pm[["cluster", "city", "city_norm", "year", "pm25"]].copy())
        print(f"PM2.5 NC loaded | year={year} | file={nc_file.name} | cities={len(city_pm)}")

    if not frames:
        raise ValueError("未生成任何年度 PM2.5 数据。")
    return pd.concat(frames, ignore_index=True)


def _merge_plot_data(cluster_geo: dict[str, gpd.GeoDataFrame], pm_long: pd.DataFrame, year: int) -> dict[str, gpd.GeoDataFrame]:
    result: dict[str, gpd.GeoDataFrame] = {}
    year_df = pm_long.loc[pm_long["year"] == year, ["cluster", "city_norm", "pm25"]].copy()
    year_df = year_df.groupby(["cluster", "city_norm"], as_index=False).agg(pm25=("pm25", "mean"))

    for cluster_key, gdf in cluster_geo.items():
        merged = gdf.merge(
            year_df.loc[year_df["cluster"] == cluster_key, ["city_norm", "pm25"]],
            on="city_norm",
            how="left",
        )
        result[cluster_key] = merged
    return result


def _compute_global_norm(
    all_yearly_gdf: list[dict[str, gpd.GeoDataFrame]],
) -> tuple[float, float]:
    """从所有年度数据计算全局 vmin/vmax，保证色标一致。"""
    all_vals: list[np.ndarray] = []
    for yearly_gdf_map in all_yearly_gdf:
        for key in CLUSTER_ORDER:
            if key in yearly_gdf_map:
                values = yearly_gdf_map[key]["pm25"].to_numpy(dtype=float)
                valid = values[~np.isnan(values)]
                if len(valid) > 0:
                    all_vals.append(valid)
    if not all_vals:
        return 0.0, 1.0
    concat = np.concatenate(all_vals)
    vmin = float(np.nanmin(concat))
    vmax = float(np.nanmax(concat))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


def plot_all_years_combined(
    years: list[int],
    all_yearly_gdf: list[dict[str, gpd.GeoDataFrame]],
    cmap: LinearSegmentedColormap,
    save_path: Path,
) -> None:
    """将多年地图绘制到同一画布，从上到下紧密排列。"""
    n_rows = len(years)
    if n_rows == 0:
        return

    vmin, vmax = _compute_global_norm(all_yearly_gdf)
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(18, 4 * n_rows),
        gridspec_kw={"hspace": 0.02, "wspace": 0.02},
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (year, yearly_gdf_map) in enumerate(zip(years, all_yearly_gdf)):
        row_axes = axes[row_idx]
        for col_idx, cluster_key in enumerate(CLUSTER_ORDER):
            ax = row_axes[col_idx]
            gdf_plot = yearly_gdf_map.get(cluster_key)
            if gdf_plot is None or gdf_plot.empty:
                ax.set_axis_off()
                continue

            gdf_plot.plot(
                column="pm25",
                cmap=cmap,
                norm=norm,
                edgecolor="#4d4d4d",
                linewidth=0.6,
                ax=ax,
                missing_kwds={"color": "#e6e6e6", "label": "无数据"},
            )
            ax.set_axis_off()

        row_axes[0].text(
            -0.02, 0.5, str(year),
            transform=row_axes[0].transAxes,
            fontsize=26, va="center", ha="right", clip_on=False,
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.02, pad=0.06)
    cbar.set_label(r"Annual mean PM$_{2.5}$ concentration", fontsize=30)

    fig.tight_layout(rect=[0.05, 0.02, 1, 0.98], h_pad=0.02, w_pad=0.02)
    fig.savefig(save_path, format="svg", dpi=FIG_DPI, transparent=True)
    plt.close(fig)


def run_pipeline(pm25_nc_dir: Path, output_dir: Path, years: list[int]) -> None:
    setup_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_geo = {
        cluster_key: read_cluster_geojson(cluster_key, Path(geo_dir))
        for cluster_key, geo_dir in DEFAULT_GEOJSON_DIRS.items()
    }
    pm_long = build_pm25_long_from_nc(pm25_nc_dir, years, cluster_geo)
    pm_long.to_csv(output_dir / "yearly_city_pm25_summary.csv", index=False, encoding="utf-8-sig")

    cmap = LinearSegmentedColormap.from_list("pm25_ribbon", RIBBON_COLORS, N=256)
    all_yearly_gdf = [_merge_plot_data(cluster_geo, pm_long, year) for year in years]
    save_path = output_dir / "三大城市群年均PM2.5浓度分布图_全部年份.svg"
    plot_all_years_combined(years, all_yearly_gdf, cmap=cmap, save_path=save_path)
    print(f"Combined map saved: {save_path.name}")

    print("全部年度地图已输出。")
    print(f"输出目录: {output_dir}")


def parse_years(years_text: str) -> list[int]:
    years = []
    for item in str(years_text).split(","):
        value = item.strip()
        if value:
            years.append(int(value))
    if not years:
        raise ValueError("years 为空。")
    return sorted(set(years))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 NC 数据输出三大城市群每年 PM2.5 年均浓度分布图")
    parser.add_argument("--pm25-nc-dir", type=Path, default=Path(DEFAULT_PM25_NC_DIR), help="年度 PM2.5 NC 目录")
    parser.add_argument(
        "--years",
        type=str,
        default=",".join(str(y) for y in DEFAULT_YEARS),
        help="分析年份，逗号分隔，如 2018,2019,2020",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "三大城市群年均PM2.5地图（nc）结果",
        help="输出目录",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        pm25_nc_dir=args.pm25_nc_dir,
        output_dir=args.output_dir,
        years=parse_years(args.years),
    )
