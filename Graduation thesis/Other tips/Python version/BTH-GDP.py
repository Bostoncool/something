"""
京津冀地区 GDP 分布图 - Python 复现
基于 R 代码 BTH-GDP.R 的功能实现
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
    "+proj=laea +lat_0=39.9 +lon_0=116.4 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
WGS84 = CRS.from_epsg(4326)
BASE_PATH = Path("H:/大论文Result/大论文图")
OUTPUT_SVG = "京津冀_GDP_2024.svg"
OUTPUT_PNG = "京津冀_GDP_2024.png"
MISSING_OUTPUT = "京津冀_缺失GDP数据区域.csv"

GDP_COLORS = [
    "#f7fbff",
    "#deebf7",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#08519c",
    "#08306b",
]

# 河北省 adcode 前4位 -> 城市名（用于 GeoJSON parent 与 CSV 区域名称映射）
HEBEI_ADCODE_TO_CITY: dict[int, str] = {
    130100: "石家庄",
    130200: "唐山",
    130300: "秦皇岛",
    130400: "邯郸",
    130500: "邢台",
    130600: "保定",
    130700: "张家口",
    130800: "承德",
    130900: "沧州",
    131000: "廊坊",
    131100: "衡水",
}

# GeoJSON name (短名) + 城市 -> CSV 区域名称（CSV 对重名区县使用 城市+区名 格式）
HEBEI_NAME_MAP: dict[tuple[str, str], str] = {
    ("桥东区", "张家口"): "张家口桥东区",
    ("桥西区", "张家口"): "张家口桥西区",
    ("新华区", "沧州"): "沧州新华区",
    ("长安区", "石家庄"): "长安市",  # CSV 中为「长安市」
}


def read_and_transform(file_path: Path, crs: CRS = LAEA_CRS) -> gpd.GeoDataFrame:
    """读取 GeoJSON 并转换到指定投影"""
    return gpd.read_file(file_path).to_crs(crs)


def hebei_geo_to_region_name(row) -> str:
    """河北省 GeoJSON：根据 name 和 parent 解析出与 CSV 匹配的区域名称"""
    name = str(row.get("name", "")).strip()
    parent = row.get("parent")
    city = ""
    if isinstance(parent, dict) and "adcode" in parent:
        adcode = parent["adcode"]
        city = HEBEI_ADCODE_TO_CITY.get(adcode, "")
    elif hasattr(parent, "get") and parent:
        adcode = parent.get("adcode")
        if adcode is not None:
            city = HEBEI_ADCODE_TO_CITY.get(adcode, "")
    mapped = HEBEI_NAME_MAP.get((name, city))
    return mapped if mapped else name


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
    return cleaned.loc[cleaned["GDP_2024"].notna() & (cleaned["GDP_2024"] > 0)].copy()


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


def draw_scalebar(ax, gdf: gpd.GeoDataFrame, length_km: float = 100) -> None:
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


def build_missing_regions(merged_data: gpd.GeoDataFrame) -> pd.DataFrame:
    """生成缺失 GDP 数据的区域列表"""
    beijing_regions = {
        "东城区", "西城区", "朝阳区", "丰台区", "石景山区", "海淀区", "门头沟区", "房山区",
        "通州区", "顺义区", "昌平区", "大兴区", "怀柔区", "平谷区", "密云区", "延庆区",
    }
    tianjin_regions = {
        "和平区", "河东区", "河西区", "南开区", "河北区", "红桥区", "东丽区", "西青区",
        "津南区", "北辰区", "武清区", "宝坻区", "滨海新区", "宁河区", "静海区", "蓟州区",
    }

    missing = merged_data.loc[merged_data["GDP_2024"].isna(), ["name"]].copy()
    missing["省份"] = np.where(
        missing["name"].isin(beijing_regions),
        "北京市",
        np.where(missing["name"].isin(tianjin_regions), "天津市", "河北省"),
    )
    return missing[["name", "省份"]].sort_values(["省份", "name"]).reset_index(drop=True)


def main() -> None:
    """主函数：绘制京津冀地区 GDP 分布图"""
    plt.rcParams["font.family"] = "Times New Roman"

    gdp_data = pd.concat(
        [
            clean_gdp_data(BASE_PATH / "三大城市群的具体GDP/北京市各区GDP（2023-2024）.csv", "北京市"),
            clean_gdp_data(BASE_PATH / "三大城市群的具体GDP/天津市各区GDP（2023-2024）.csv", "天津市"),
            clean_gdp_data(BASE_PATH / "三大城市群的具体GDP/河北省各区GDP（2023-2024）.csv", "河北省"),
        ],
        ignore_index=True,
    )

    beijing_geo = read_and_transform(BASE_PATH / "2.京津冀/北京市 (划分).geojson")
    tianjin_geo = read_and_transform(BASE_PATH / "2.京津冀/天津市 (划分).geojson")
    hebei_geo = read_and_transform(BASE_PATH / "2.京津冀/河北省 (区).geojson")

    beijing_geo_sub = beijing_geo.loc[beijing_geo["level"] == "district", ["name", "geometry"]].assign(
        省份="北京市", 区域名称=lambda d: d["name"]
    )
    tianjin_geo_sub = tianjin_geo.loc[tianjin_geo["level"] == "district", ["name", "geometry"]].assign(
        省份="天津市", 区域名称=lambda d: d["name"]
    )
    hebei_district = hebei_geo.loc[hebei_geo["level"] == "district", ["name", "parent", "geometry"]].copy()
    hebei_district["省份"] = "河北省"
    hebei_district["区域名称"] = hebei_district.apply(hebei_geo_to_region_name, axis=1)
    hebei_geo_sub = hebei_district[["name", "geometry", "省份", "区域名称"]]

    geo_data = pd.concat([beijing_geo_sub, tianjin_geo_sub, hebei_geo_sub], ignore_index=True)
    merged_data = gpd.GeoDataFrame(
        geo_data.merge(gdp_data, left_on=["区域名称", "省份"], right_on=["区域名称", "省份"], how="left"),
        geometry="geometry",
        crs=LAEA_CRS,
    )

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

    draw_scalebar(ax, merged_data, length_km=100)
    draw_north_arrow(ax)
    add_graticule_labels(ax, lats, lons, bounds_wgs84, bounds_laea)

    ax.set_title("BTH 2024 GDP Distribution", fontsize=32, fontweight="bold")
    ax.set_xlabel("Longitude (°E)", fontsize=24, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=24, fontweight="bold")
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=22)
    plt.tight_layout()

    plt.savefig(OUTPUT_SVG, format="svg", dpi=300, bbox_inches=None)
    plt.savefig(OUTPUT_PNG, format="png", dpi=300, bbox_inches=None)
    plt.show()

    missing_gdp_regions = build_missing_regions(merged_data)
    print("\n=== 数据统计信息 ===")
    print(f"总区域数量: {len(merged_data)}")
    print(f"有GDP数据的区域数量: {merged_data['GDP_2024'].notna().sum()}")
    print(f"缺失GDP数据的区域数量: {len(missing_gdp_regions)}")

    if not missing_gdp_regions.empty:
        missing_gdp_regions.to_csv(MISSING_OUTPUT, index=False, encoding="utf-8-sig")
        print(f"\n缺失GDP数据的区域已保存到: {MISSING_OUTPUT}")
        for _, row in missing_gdp_regions.iterrows():
            print(f"{row['省份']}: {row['name']}")
    else:
        print("所有区域都有对应的GDP数据！")

    if not valid_gdp.empty:
        print("\n=== GDP数据统计 ===")
        print(f"GDP数据范围: {valid_gdp.min()} - {valid_gdp.max()} 亿元")
        print(f"GDP数据平均值: {valid_gdp.mean():.2f} 亿元")
        print(f"GDP数据中位数: {valid_gdp.median():.2f} 亿元")


if __name__ == "__main__":
    main()
