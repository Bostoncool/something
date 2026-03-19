"""
长三角地区 GDP 分布图 - Python 复现
基于 R 代码 YRD-GDP.R 的功能实现
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import CRS, Transformer
from shapely.geometry import LineString

LAEA_CRS = CRS.from_proj4(
    "+proj=laea +lat_0=32.0 +lon_0=118.8 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
WGS84 = CRS.from_epsg(4326)
BASE_PATH = Path("H:/大论文Result/大论文图")
OUTPUT_SVG = "长三角_GDP_2024.svg"
OUTPUT_PNG = "长三角_GDP_2024.png"
MISSING_OUTPUT = "长三角_缺失GDP数据区域.csv"

CITY_GEO_FILES = [
    BASE_PATH / "3.长三角/具体城市（区分辨率）/南京市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/无锡市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/南通市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/盐城市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/扬州市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/镇江市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/常州市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/苏州市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/泰州市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/杭州市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/宁波市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/嘉兴市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/湖州市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/绍兴市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/金华市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/舟山市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/台州市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/温州市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/合肥市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/芜湖市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/马鞍山市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/铜陵市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/安庆市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/滁州市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/池州市 (区).geojson",
    BASE_PATH / "3.长三角/具体城市（区分辨率）/宣城市 (区).geojson",
]

# 城市 -> 省份 映射，用于按省份合并避免重名区错配
CITY_TO_PROVINCE = {
    "南京市": "江苏省", "无锡市": "江苏省", "南通市": "江苏省", "盐城市": "江苏省",
    "扬州市": "江苏省", "镇江市": "江苏省", "常州市": "江苏省", "苏州市": "江苏省",
    "泰州市": "江苏省",
    "杭州市": "浙江省", "宁波市": "浙江省", "嘉兴市": "浙江省", "湖州市": "浙江省",
    "绍兴市": "浙江省", "金华市": "浙江省", "舟山市": "浙江省", "台州市": "浙江省",
    "温州市": "浙江省",
    "合肥市": "安徽省", "芜湖市": "安徽省", "马鞍山市": "安徽省", "铜陵市": "安徽省",
    "安庆市": "安徽省", "滁州市": "安徽省", "池州市": "安徽省", "宣城市": "安徽省",
}

def read_and_transform(file_path: Path, crs: CRS = LAEA_CRS) -> gpd.GeoDataFrame:
    """读取 GeoJSON 并转换到指定投影"""
    return gpd.read_file(file_path).to_crs(crs)


def find_column(columns: list[str], candidates: list[str], fallback_idx: int | None = None) -> str:
    """按候选名称或索引寻找列名"""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if fallback_idx is not None and 0 <= fallback_idx < len(columns):
        return columns[fallback_idx]
    raise KeyError(f"无法在列名中找到候选字段: {candidates}")


def clean_gdp_data(file_path: Path, province_name: str) -> pd.DataFrame:
    """清理 GDP 数据"""
    df = pd.read_csv(file_path)
    columns = list(df.columns)

    year_col = find_column(columns, ["年份"])
    level_col = find_column(columns, ["level"])
    region_col = find_column(columns, ["县市区", "区域名称"], fallback_idx=2)
    gdp_col = find_column(columns, ["GDP.亿.", "GDP(亿)", "GDP（亿）", "GDP_亿"], fallback_idx=5)

    cleaned = (
        df.loc[(df[year_col] == 2024) & (df[level_col] == 2), [region_col, gdp_col]]
        .rename(columns={region_col: "区域名称", gdp_col: "GDP_2024"})
        .assign(省份=province_name)
    )
    cleaned["GDP_2024"] = pd.to_numeric(cleaned["GDP_2024"], errors="coerce")
    cleaned = cleaned.loc[cleaned["GDP_2024"].notna() & (cleaned["GDP_2024"] > 0)].copy()
    return cleaned


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

    return gpd.GeoDataFrame(geometry=lines, crs=WGS84).to_crs(LAEA_CRS), lats, lons


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
            ax.text(x_proj, ymin_l - (ymax_l - ymin_l) * 0.02, f"{int(lon)}°E", ha="center", va="top", fontsize=16)

    for lat in lats:
        _, y_proj = transformer.transform(xmin_w, lat)
        if ymin_l <= y_proj <= ymax_l:
            ax.text(xmin_l - (xmax_l - xmin_l) * 0.02, y_proj, f"{int(lat)}°N", ha="right", va="center", fontsize=16)

    ax.set_xticks([])
    ax.set_yticks([])


def draw_north_arrow(ax, x: float = 0.05, y: float = 0.95, size: float = 0.08) -> None:
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
    """手动绘制比例尺"""
    xmin, ymin, xmax, ymax = gdf.total_bounds
    bar_length_m = length_km * 1000
    pad_x = (xmax - xmin) * 0.05
    pad_y = (ymax - ymin) * 0.05
    x1 = xmax - pad_x
    x0 = x1 - bar_length_m
    y0 = ymin + pad_y

    ax.plot([x0, x1], [y0, y0], color="black", linewidth=2)
    ax.plot([x0, x0], [y0, y0 + pad_y * 0.3], color="black", linewidth=1)
    ax.plot([x1, x1], [y0, y0 + pad_y * 0.3], color="black", linewidth=1)
    ax.text((x0 + x1) / 2, y0 + pad_y * 0.45, f"{int(length_km)} km", ha="center", va="bottom", fontsize=16)


def load_geo_data() -> gpd.GeoDataFrame:
    """读取长三角城市和上海分区数据，为每行添加省份用于合并"""
    city_frames = []
    for file_path in CITY_GEO_FILES:
        city_name = file_path.stem.replace(" (区)", "")  # 如 "南京市 (区)" -> "南京市"
        province = CITY_TO_PROVINCE.get(city_name, "江苏省")
        gdf = read_and_transform(file_path)[["name", "level", "geometry"]].assign(省份=province)
        city_frames.append(gdf)
    shanghai_geo = read_and_transform(BASE_PATH / "3.长三角/上海市 (划分).geojson")
    shanghai_geo = shanghai_geo.loc[shanghai_geo["level"] == "district", ["name", "level", "geometry"]].assign(省份="上海市")

    geo_data = pd.concat([*city_frames, shanghai_geo], ignore_index=True)
    return gpd.GeoDataFrame(geo_data, geometry="geometry", crs=LAEA_CRS)


def merge_with_mapping(geo_data: gpd.GeoDataFrame, gdp_data: pd.DataFrame) -> gpd.GeoDataFrame:
    """按 (name, 省份) 合并，避免跨省重名区错配；重名区需名称映射（鼓楼区、普陀区等）"""
    geo_with_mapping = geo_data.copy()
    # 名称映射：GeoJSON 与 GDP 数据命名不一致的重名区
    mask_鼓楼区 = (geo_with_mapping["name"] == "鼓楼区") & (geo_with_mapping["省份"] == "江苏省")
    mask_普陀区 = (geo_with_mapping["name"] == "普陀区") & (geo_with_mapping["省份"] == "浙江省")
    mask_郊区 = (geo_with_mapping["name"] == "郊区") & (geo_with_mapping["省份"] == "安徽省")
    geo_with_mapping["name_clean"] = np.select(
        [mask_鼓楼区, mask_普陀区, mask_郊区],
        ["南京鼓楼区", "舟山普陀区", "铜陵郊区"],
        default=geo_with_mapping["name"],
    )
    merged = gpd.GeoDataFrame(
        geo_with_mapping.merge(gdp_data, left_on=["name_clean", "省份"], right_on=["区域名称", "省份"], how="left"),
        geometry="geometry",
        crs=LAEA_CRS,
    ).drop(columns=["name_clean"])
    return merged


def main() -> None:
    """主函数：绘制长三角地区 GDP 分布图"""
    plt.rcParams["font.family"] = "Times New Roman"

    gdp_data = pd.concat(
        [
            clean_gdp_data(BASE_PATH / "三大城市群的具体GDP/上海市各区GDP（2023-2024）.csv", "上海市"),
            clean_gdp_data(BASE_PATH / "三大城市群的具体GDP/江苏省各区GDP（2023-2024）.csv", "江苏省"),
            clean_gdp_data(BASE_PATH / "三大城市群的具体GDP/浙江省各区GDP（2023-2024）.csv", "浙江省"),
            clean_gdp_data(BASE_PATH / "三大城市群的具体GDP/安徽省各区GDP（2023-2024）.csv", "安徽省"),
        ],
        ignore_index=True,
    )

    geo_data = load_geo_data()
    merged_data = merge_with_mapping(geo_data, gdp_data)

    graticule_gdf, lats, lons = create_graticule(merged_data)
    bounds_wgs84 = tuple(merged_data.to_crs(WGS84).total_bounds)
    bounds_laea = tuple(merged_data.total_bounds)
    valid_gdp = merged_data["GDP_2024"].dropna()
    color_norm = LogNorm(vmin=float(valid_gdp.min()), vmax=float(valid_gdp.max()))

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    graticule_gdf.plot(ax=ax, color="#cccccc", linewidth=0.3, linestyle="--")
    merged_data.plot(
        ax=ax,
        column="GDP_2024",
        cmap="Blues",
        norm=color_norm,
        edgecolor="black",
        linewidth=0.3,
        legend=False,
        missing_kwds={"color": "#e6e6e6", "label": "No data"},
    )
    # 色带与主图方框等高，字体 28 号
    fig = ax.get_figure()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    sm = ScalarMappable(cmap="Blues", norm=color_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("GDP (100 million CNY, log scale)", fontsize=28)
    cbar.ax.tick_params(labelsize=28)

    draw_scalebar(ax, merged_data, length_km=200)
    draw_north_arrow(ax)
    add_graticule_labels(ax, lats, lons, bounds_wgs84, bounds_laea)

    ax.set_title("YRD 2024 GDP Distribution", fontsize=32, fontweight="bold")
    ax.set_xlabel("Longitude (°E)", fontsize=24, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=24, fontweight="bold")
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=22)
    plt.tight_layout()

    plt.savefig(OUTPUT_SVG, format="svg", dpi=300, bbox_inches=None)
    plt.savefig(OUTPUT_PNG, format="png", dpi=300, bbox_inches=None)
    plt.show()

    missing = merged_data.loc[merged_data["GDP_2024"].isna(), ["name", "level"]].sort_values("name").reset_index(drop=True)
    print("\n=== 数据统计信息 ===")
    print(f"总区域数量: {len(merged_data)}")
    print(f"有GDP数据的区域数量: {merged_data['GDP_2024'].notna().sum()}")
    print(f"缺失GDP数据的区域数量: {len(missing)}")

    if not missing.empty:
        missing.to_csv(MISSING_OUTPUT, index=False, encoding="utf-8-sig")
        print(f"\n缺失GDP数据的区域已保存到: {MISSING_OUTPUT}")
        for name in missing["name"]:
            print(name)
    else:
        print("所有区域都有对应的GDP数据！")

    if not valid_gdp.empty:
        print("\n=== GDP数据统计 ===")
        print(f"GDP数据范围: {valid_gdp.min()} - {valid_gdp.max()} 亿元")
        print(f"GDP数据平均值: {valid_gdp.mean():.2f} 亿元")
        print(f"GDP数据中位数: {valid_gdp.median():.2f} 亿元")

        summary = (
            gdp_data.groupby("省份")["GDP_2024"]
            .agg(区域数量="count", 平均GDP="mean", 最大GDP="max", 最小GDP="min")
            .reset_index()
        )
        print("\n=== 长三角地区GDP数据概览 ===")
        print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
