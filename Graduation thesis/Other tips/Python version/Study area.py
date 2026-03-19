"""
研究区域地图绘制 - Python 复现
基于 R 代码 Study area.R 的功能实现
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pyproj import CRS, Transformer
from shapely.geometry import LineString
import numpy as np

# LAEA 投影：中心点中国 (105°E, 35°N)
LAEA_CRS = CRS.from_proj4(
    "+proj=laea +lat_0=35 +lon_0=105 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
WGS84 = CRS.from_epsg(4326)

# 数据路径（可配置）
BASE_PATH = "H:/大论文Result/大论文图"

# 区域配置：路径、颜色与英文标签
REGION_CONFIG = {
    "京津冀地区": {
        "label_en": "BTH",
        "files": [
            f"{BASE_PATH}/2.京津冀/北京市.geojson",
            f"{BASE_PATH}/2.京津冀/天津市.geojson",
            f"{BASE_PATH}/2.京津冀/河北省 (市).geojson",
        ],
        "color": "lightyellow",
    },
    "长三角地区": {
        "label_en": "YRD",
        "files": [
            f"{BASE_PATH}/3.长三角/上海市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/南京市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/无锡市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/南通市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/盐城市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/扬州市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/镇江市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/常州市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/苏州市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/泰州市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/杭州市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/宁波市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/嘉兴市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/湖州市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/绍兴市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/金华市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/舟山市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/台州市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/温州市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/合肥市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/芜湖市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/马鞍山市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/铜陵市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/安庆市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/滁州市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/池州市.geojson",
            f"{BASE_PATH}/3.长三角/具体城市/宣城市.geojson",
        ],
        "color": "turquoise",
    },
    "珠三角地区": {
        "label_en": "PRD",
        "files": [
            f"{BASE_PATH}/4.珠三角/具体城市/广州市.geojson",
            f"{BASE_PATH}/4.珠三角/具体城市/深圳市.geojson",
            f"{BASE_PATH}/4.珠三角/具体城市/佛山市.geojson",
            f"{BASE_PATH}/4.珠三角/具体城市/东莞市.geojson",
            f"{BASE_PATH}/4.珠三角/具体城市/中山市.geojson",
            f"{BASE_PATH}/4.珠三角/具体城市/惠州市.geojson",
            f"{BASE_PATH}/4.珠三角/具体城市/珠海市.geojson",
            f"{BASE_PATH}/4.珠三角/具体城市/江门市.geojson",
            f"{BASE_PATH}/4.珠三角/具体城市/肇庆市.geojson",
        ],
        "color": "lightcoral",
    },
}


def read_and_transform(file_path: str, crs=LAEA_CRS) -> gpd.GeoDataFrame:
    """读取 GeoJSON 并转换到指定投影"""
    gdf = gpd.read_file(file_path)
    return gdf.to_crs(crs)


def create_graticule(
    gdf: gpd.GeoDataFrame, lat_step: int = 10, lon_step: int = 10
) -> tuple[gpd.GeoDataFrame, np.ndarray, np.ndarray]:
    """创建经纬度网格线（在 LAEA 投影下）"""
    gdf_wgs84 = gdf.to_crs(WGS84)
    bounds = gdf_wgs84.total_bounds
    xmin, ymin, xmax, ymax = bounds

    lats = np.arange(np.floor(ymin), np.ceil(ymax) + 1, lat_step)
    lons = np.arange(np.floor(xmin), np.ceil(xmax) + 1, lon_step)

    lines = []
    for lat in lats:
        line = LineString([(xmin, lat), (xmax, lat)])
        lines.append(line)
    for lon in lons:
        line = LineString([(lon, ymin), (lon, ymax)])
        lines.append(line)

    graticule_gdf = gpd.GeoDataFrame(geometry=lines, crs=WGS84).to_crs(LAEA_CRS)
    return graticule_gdf, lats, lons


def _add_graticule_labels(
    ax, lats: np.ndarray, lons: np.ndarray, bounds_wgs84: tuple, bounds_laea: tuple
) -> None:
    """在经纬网外边框添加经度/纬度度数标签（仅外边框，不显示坐标轴刻度）"""
    xmin_w, ymin_w, xmax_w, ymax_w = bounds_wgs84
    xmin_l, ymin_l, xmax_l, ymax_l = bounds_laea
    transformer = Transformer.from_crs(WGS84, LAEA_CRS, always_xy=True)

    # 在底边和左边外边框添加度数标签
    for lon in lons:
        x_proj, _ = transformer.transform(lon, ymin_w)
        if xmin_l <= x_proj <= xmax_l:
            label = f"{int(lon)}°E" if lon >= 0 else f"{int(-lon)}°W"
            ax.text(x_proj, ymin_l - (ymax_l - ymin_l) * 0.015, label, ha="center", va="top", fontsize=16)
    for lat in lats:
        _, y_proj = transformer.transform(xmin_w, lat)
        if ymin_l <= y_proj <= ymax_l:
            label = f"{int(lat)}°N" if lat >= 0 else f"{int(-lat)}°S"
            ax.text(xmin_l - (xmax_l - xmin_l) * 0.015, y_proj, label, ha="right", va="center", fontsize=16)

    # 隐藏坐标轴刻度及刻度标签（仅保留外边框度数标签）
    ax.set_xticks([])
    ax.set_yticks([])


def load_all_regions() -> gpd.GeoDataFrame:
    """批量读取各区域数据并合并"""
    all_gdfs = []
    for region_name, config in REGION_CONFIG.items():
        for fp in config["files"]:
            try:
                gdf = read_and_transform(fp)
                gdf = gdf[["geometry"]].copy()
                gdf["region"] = region_name
                gdf = gdf[["region", "geometry"]]
                all_gdfs.append(gdf)
            except Exception as e:
                print(f"Skipped {fp}: {e}")
    if not all_gdfs:
        return gpd.GeoDataFrame(columns=["region", "geometry"], crs=LAEA_CRS)
    combined = pd.concat(all_gdfs, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=LAEA_CRS)


def draw_north_arrow(ax, x: float = 0.95, y: float = 0.95, size: float = 0.08) -> None:
    """在图上绘制指北针（左上角）"""
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


def _draw_scalebar(ax, gdf: gpd.GeoDataFrame, length_km: float = 500) -> None:
    """手动绘制比例尺（LAEA 单位为米），置于左上角"""
    xmin, ymin, xmax, ymax = gdf.total_bounds
    bar_length_m = length_km * 1000
    pad_x = (xmax - xmin) * 0.05
    pad_y = (ymax - ymin) * 0.05
    # 左上角：从左往右绘制
    x0 = xmin + pad_x
    x1 = x0 + bar_length_m
    y0 = ymax - pad_y
    ax.plot([x0, x1], [y0, y0], "k-", linewidth=2, solid_capstyle="butt")
    ax.plot([x0, x0], [y0, y0 - pad_y * 0.3], "k-", linewidth=1)
    ax.plot([x1, x1], [y0, y0 - pad_y * 0.3], "k-", linewidth=1)
    ax.text((x0 + x1) / 2, y0 - pad_y * 0.5, f"{length_km} km", ha="center", va="top", fontsize=16)


def main() -> None:
    """主函数：绘制中国研究区域地图"""
    plt.rcParams["font.family"] = "Times New Roman"

    # 读取中国边界
    china_shp = read_and_transform(f"{BASE_PATH}/1.总图/中华人民共和国.geojson")
    china_province_shp = read_and_transform(f"{BASE_PATH}/1.总图/中国（省）.geojson")

    # 加载研究区域
    all_regions_data = load_all_regions()

    # 创建经纬网
    graticule_gdf, lats, lons = create_graticule(china_shp)
    bounds_wgs84 = tuple(china_shp.to_crs(WGS84).total_bounds)
    bounds_laea = tuple(china_shp.total_bounds)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    # 1. 经纬网（底层）
    graticule_gdf.plot(
        ax=ax, color="#cccccc", linewidth=0.3, linestyle="--"
    )

    # 2. 中国国界
    china_shp.plot(ax=ax, facecolor="white", edgecolor="black", linewidth=0.8)

    # 3. 省界
    china_province_shp.plot(
        ax=ax, facecolor="lightgray", edgecolor="gray", linewidth=0.3, alpha=0.5
    )

    # 4. 研究区域（按颜色分组绘制）
    for region_name, config in REGION_CONFIG.items():
        region_gdf = all_regions_data[all_regions_data["region"] == region_name]
        if not region_gdf.empty:
            region_gdf.plot(
                ax=ax,
                facecolor=config["color"],
                edgecolor="gray",
                linewidth=0.3,
                label=config["label_en"],
            )

    # 5. 比例尺（左下角，手动绘制，LAEA 单位为米）
    _draw_scalebar(ax, china_shp)

    # 6. 指北针（左上角）
    draw_north_arrow(ax, x=0.02, y=0.98)

    # 7. 经纬度标签（在坐标轴边缘显示度数）
    _add_graticule_labels(ax, lats, lons, bounds_wgs84, bounds_laea)

    # 8. 图例（手动创建确保显示）
    legend_handles = [
        Patch(facecolor=config["color"], edgecolor="gray", label=config["label_en"])
        for config in REGION_CONFIG.values()
    ]
    ax.legend(
        handles=legend_handles,
        title="Study area",
        loc="lower left",
        fontsize=28,
        title_fontsize=32,
        frameon=True,
        fancybox=True,
    )

    ax.set_title("Map of China (with provincial boundaries)", fontsize=32, fontweight="bold")
    ax.set_xlabel("Longitude (°E)", fontsize=24, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=24, fontweight="bold")
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=22)
    plt.tight_layout()

    # 保存 SVG
    plt.savefig(
        "China_map_study_area.svg",
        format="svg",
        bbox_inches=None,  # 不使用 tight，严格按 figsize(12,12) 输出正方形
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":
    main()
