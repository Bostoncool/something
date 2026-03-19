"""
长三角地区分区图 - Python 复现
基于 R 代码 YRD-area.R 的功能实现
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString

# LAEA 投影：中心点长三角地区
LAEA_CRS = CRS.from_proj4(
    "+proj=laea +lat_0=31 +lon_0=120 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
WGS84 = CRS.from_epsg(4326)
BASE_PATH = Path("H:/大论文Result/大论文图")

PROVINCE_FILES = [
    BASE_PATH / "3.长三角/上海市 (划分).geojson",
    BASE_PATH / "3.长三角/安徽省 (市).geojson",
    BASE_PATH / "3.长三角/江苏省 (市).geojson",
    BASE_PATH / "3.长三角/浙江省 (市).geojson",
]
REGION_FILES = [
    BASE_PATH / "3.长三角/上海市.geojson",
    BASE_PATH / "3.长三角/具体城市/南京市.geojson",
    BASE_PATH / "3.长三角/具体城市/无锡市.geojson",
    BASE_PATH / "3.长三角/具体城市/南通市.geojson",
    BASE_PATH / "3.长三角/具体城市/盐城市.geojson",
    BASE_PATH / "3.长三角/具体城市/扬州市.geojson",
    BASE_PATH / "3.长三角/具体城市/镇江市.geojson",
    BASE_PATH / "3.长三角/具体城市/常州市.geojson",
    BASE_PATH / "3.长三角/具体城市/苏州市.geojson",
    BASE_PATH / "3.长三角/具体城市/泰州市.geojson",
    BASE_PATH / "3.长三角/具体城市/杭州市.geojson",
    BASE_PATH / "3.长三角/具体城市/宁波市.geojson",
    BASE_PATH / "3.长三角/具体城市/嘉兴市.geojson",
    BASE_PATH / "3.长三角/具体城市/湖州市.geojson",
    BASE_PATH / "3.长三角/具体城市/绍兴市.geojson",
    BASE_PATH / "3.长三角/具体城市/金华市.geojson",
    BASE_PATH / "3.长三角/具体城市/舟山市.geojson",
    BASE_PATH / "3.长三角/具体城市/台州市.geojson",
    BASE_PATH / "3.长三角/具体城市/温州市.geojson",
    BASE_PATH / "3.长三角/具体城市/合肥市.geojson",
    BASE_PATH / "3.长三角/具体城市/芜湖市.geojson",
    BASE_PATH / "3.长三角/具体城市/马鞍山市.geojson",
    BASE_PATH / "3.长三角/具体城市/铜陵市.geojson",
    BASE_PATH / "3.长三角/具体城市/安庆市.geojson",
    BASE_PATH / "3.长三角/具体城市/滁州市.geojson",
    BASE_PATH / "3.长三角/具体城市/池州市.geojson",
    BASE_PATH / "3.长三角/具体城市/宣城市.geojson",
]
REGION_COLOR = "turquoise"


def read_and_transform(file_path: Path, crs: CRS = LAEA_CRS) -> gpd.GeoDataFrame:
    """读取 GeoJSON 并转换到指定投影"""
    return gpd.read_file(file_path).to_crs(crs)


def load_geometries(file_paths: list[Path], label: str | None = None) -> gpd.GeoDataFrame:
    """读取多个矢量文件并合并"""
    gdfs = []
    for file_path in file_paths:
        gdf = read_and_transform(file_path)[["geometry"]].copy()
        if label is not None:
            gdf["region"] = label
            gdf = gdf[["region", "geometry"]]
        gdfs.append(gdf)

    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), geometry="geometry", crs=LAEA_CRS)


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


def draw_scalebar(ax, gdf: gpd.GeoDataFrame, length_km: float = 200) -> None:
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


def main() -> None:
    """主函数：绘制长三角地区分区图"""
    plt.rcParams["font.family"] = "Times New Roman"

    province_boundary_data = load_geometries(PROVINCE_FILES)
    region_data = load_geometries(REGION_FILES, label="长三角地区")

    graticule_gdf, lats, lons = create_graticule(region_data, lat_step=2, lon_step=2)
    bounds_wgs84 = tuple(region_data.to_crs(WGS84).total_bounds)
    bounds_laea = tuple(region_data.total_bounds)
    xmin, ymin, xmax, ymax = bounds_laea

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    graticule_gdf.plot(ax=ax, color="#cccccc", linewidth=0.3, linestyle="--")
    province_boundary_data.boundary.plot(ax=ax, color="black", linewidth=0.8, alpha=0.8)
    region_data.plot(ax=ax, facecolor=REGION_COLOR, edgecolor="gray", linewidth=0.3)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    draw_scalebar(ax, region_data, length_km=200)
    draw_north_arrow(ax)
    add_graticule_labels(ax, lats, lons, bounds_wgs84, bounds_laea)

    ax.set_title("YRD", fontsize=32, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=24, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=24, fontweight="bold")
    ax.set_aspect("equal")
    plt.tight_layout()

    plt.savefig("长三角地区放大图.svg", format="svg", dpi=300, bbox_inches=None)
    plt.show()


if __name__ == "__main__":
    main()
