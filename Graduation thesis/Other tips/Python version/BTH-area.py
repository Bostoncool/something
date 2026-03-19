"""
京津冀地区分区图 - Python 复现
基于 R 代码 BTH-area.R 的功能实现

用法:
    python BTH-area.py [数据目录]
    或设置环境变量: BTH_AREA_DATA=你的数据路径
"""

import argparse
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString

# LAEA 投影：中心点京津冀地区
LAEA_CRS = CRS.from_proj4(
    "+proj=laea +lat_0=39 +lon_0=116 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
WGS84 = CRS.from_epsg(4326)

# 默认数据根目录，可通过环境变量 BTH_AREA_DATA 或命令行参数覆盖
DEFAULT_BASE_PATH = Path(os.environ.get("BTH_AREA_DATA", "H:/大论文Result/大论文图"))


def get_province_files(base_path: Path) -> list[Path]:
    """根据根目录生成 GeoJSON 文件路径"""
    return [
        base_path / "2.京津冀/北京市 (划分).geojson",
        base_path / "2.京津冀/天津市 (划分).geojson",
        base_path / "2.京津冀/河北省 (市).geojson",
    ]


REGION_COLOR = "lightyellow"


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

    combined = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        geometry="geometry",
        crs=LAEA_CRS,
    )
    return combined


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
    xmin_w, ymin_w, xmax_w, ymax_w = bounds_wgs84
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


def main(base_path: Path | None = None) -> None:
    """主函数：绘制京津冀地区分区图"""
    plt.rcParams["font.family"] = "Times New Roman"

    base = base_path or DEFAULT_BASE_PATH
    province_files = get_province_files(base)
    if base_path:
        print(f"使用数据目录: {base}")

    province_boundary_data = load_geometries(province_files)
    region_data = load_geometries(province_files, label="京津冀地区")

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

    draw_scalebar(ax, region_data, length_km=100)
    draw_north_arrow(ax)
    add_graticule_labels(ax, lats, lons, bounds_wgs84, bounds_laea)

    ax.set_title("BTH", fontsize=32, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=24, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=24, fontweight="bold")
    ax.set_aspect("equal")
    plt.tight_layout()

    plt.savefig("京津冀地区放大图.svg", format="svg", dpi=300, bbox_inches=None)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="京津冀地区分区图绘制")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="数据根目录（包含 2.京津冀 的上级目录）",
    )
    args = parser.parse_args()
    base_path = Path(args.data_dir) if args.data_dir else None
    main(base_path)
