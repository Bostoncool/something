"""
珠三角地区分区图 - Python 复现
基于 R 代码 PRD-area.R 的功能实现
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString

# LAEA 投影：中心点珠三角地区
LAEA_CRS = CRS.from_proj4(
    "+proj=laea +lat_0=23 +lon_0=113.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
WGS84 = CRS.from_epsg(4326)
BASE_PATH = Path("H:/大论文Result/大论文图")

GUANGDONG_FILE = BASE_PATH / "4.珠三角/广东省 (市).geojson"
REGION_FILES = [
    BASE_PATH / "4.珠三角/具体城市/广州市.geojson",
    BASE_PATH / "4.珠三角/具体城市/深圳市.geojson",
    BASE_PATH / "4.珠三角/具体城市/佛山市.geojson",
    BASE_PATH / "4.珠三角/具体城市/东莞市.geojson",
    BASE_PATH / "4.珠三角/具体城市/中山市.geojson",
    BASE_PATH / "4.珠三角/具体城市/惠州市.geojson",
    BASE_PATH / "4.珠三角/具体城市/珠海市.geojson",
    BASE_PATH / "4.珠三角/具体城市/江门市.geojson",
    BASE_PATH / "4.珠三角/具体城市/肇庆市.geojson",
]
REGION_COLOR = "lightcoral"


def read_and_transform(file_path: Path, crs: CRS = LAEA_CRS) -> gpd.GeoDataFrame:
    """读取 GeoJSON 并转换到指定投影"""
    return gpd.read_file(file_path).to_crs(crs)


def create_graticule(
    gdf: gpd.GeoDataFrame, lat_step: int = 2, lon_step: int = 2
) -> tuple[gpd.GeoDataFrame, np.ndarray, np.ndarray]:
    """创建经纬度网格线"""
    gdf_wgs84 = gdf.to_crs(WGS84)
    xmin, ymin, xmax, ymax = gdf_wgs84.total_bounds

    lats = np.arange(np.floor(ymin), np.ceil(ymax) + 1, lat_step)
    lons = np.arange(np.floor(xmin), np.ceil(xmax) + 1, lon_step)

    lines = []
    for lat in lats:
        lines.append(LineString([(xmin, lat), (xmax, lat)]))
    for lon in lons:
        lines.append(LineString([(lon, ymin), (lon, ymax)]))

    graticule_gdf = gpd.GeoDataFrame(geometry=lines, crs=WGS84).to_crs(LAEA_CRS)
    return graticule_gdf, lats, lons


def add_graticule_labels(
    ax,
    lats: np.ndarray,
    lons: np.ndarray,
    bounds_wgs84: tuple[float, float, float, float],
    bounds_laea: tuple[float, float, float, float],
) -> None:
    """在外边框添加经纬度标签"""
    xmin_w, ymin_w, _, _ = bounds_wgs84
    xmin_l, ymin_l, xmax_l, ymax_l = bounds_laea
    transformer = Transformer.from_crs(WGS84, LAEA_CRS, always_xy=True)

    for lon in lons:
        x_proj, _ = transformer.transform(lon, ymin_w)
        if xmin_l <= x_proj <= xmax_l:
            ax.text(
                x_proj,
                ymin_l - (ymax_l - ymin_l) * 0.02,
                f"{int(lon)}°E",
                ha="center",
                va="top",
                fontsize=16,
            )

    for lat in lats:
        _, y_proj = transformer.transform(xmin_w, lat)
        if ymin_l <= y_proj <= ymax_l:
            ax.text(
                xmin_l - (xmax_l - xmin_l) * 0.02,
                y_proj,
                f"{int(lat)}°N",
                ha="right",
                va="center",
                fontsize=16,
            )

    ax.set_xticks([])
    ax.set_yticks([])


def draw_north_arrow(ax, x: float = 0.92, y: float = 0.96, size: float = 0.08) -> None:
    """绘制指北针"""
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - size),
        fontsize=24,
        fontweight="bold",
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        xycoords="axes fraction",
    )


def draw_scalebar(ax, gdf: gpd.GeoDataFrame, length_km: float = 100) -> None:
    """手动绘制比例尺（右下角）"""
    xmin, ymin, xmax, ymax = gdf.total_bounds
    bar_length_m = length_km * 1000
    pad_x = (xmax - xmin) * 0.04
    pad_y = (ymax - ymin) * 0.04
    x1 = xmax - pad_x
    x0 = x1 - bar_length_m
    y0 = ymin + pad_y

    ax.plot([x0, x1], [y0, y0], color="black", linewidth=2)
    ax.plot([x0, x0], [y0, y0 + pad_y * 0.3], color="black", linewidth=1)
    ax.plot([x1, x1], [y0, y0 + pad_y * 0.3], color="black", linewidth=1)
    ax.text((x0 + x1) / 2, y0 + pad_y * 0.45, f"{int(length_km)} km", ha="center", va="bottom", fontsize=16)


def load_region_data(file_paths: list[Path]) -> gpd.GeoDataFrame:
    """读取珠三角城市边界并合并"""
    gdfs = []
    for file_path in file_paths:
        gdf = read_and_transform(file_path)[["geometry"]].copy()
        gdf["region"] = "珠三角地区"
        gdf = gdf[["region", "geometry"]]
        gdfs.append(gdf)

    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), geometry="geometry", crs=LAEA_CRS)


def main() -> None:
    """主函数：绘制珠三角地区分区图"""
    plt.rcParams["font.family"] = "Times New Roman"

    guangdong_shp = read_and_transform(GUANGDONG_FILE)
    region_data = load_region_data(REGION_FILES)

    graticule_gdf, lats, lons = create_graticule(region_data, lat_step=2, lon_step=2)
    bounds_wgs84 = tuple(region_data.to_crs(WGS84).total_bounds)
    bounds_laea = tuple(region_data.total_bounds)
    xmin, ymin, xmax, ymax = bounds_laea

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    graticule_gdf.plot(ax=ax, color="#cccccc", linewidth=0.3, linestyle="--")
    guangdong_shp.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=0.5, alpha=0.3)
    region_data.plot(ax=ax, facecolor=REGION_COLOR, edgecolor="#4d4d4d", linewidth=0.4)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    draw_scalebar(ax, region_data, length_km=100)
    draw_north_arrow(ax)
    add_graticule_labels(ax, lats, lons, bounds_wgs84, bounds_laea)

    ax.set_title("PRD", fontsize=32, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=24, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=24, fontweight="bold")
    ax.set_aspect("equal")
    plt.tight_layout()

    plt.savefig("珠三角地区放大图.svg", format="svg", dpi=300, bbox_inches=None)
    plt.show()


if __name__ == "__main__":
    main()
