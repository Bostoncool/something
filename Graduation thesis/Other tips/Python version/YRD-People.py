"""
长三角地区人口密度分布图 - Python 复现
基于 R 代码 YRD-People.R 的功能实现
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
    "+proj=laea +lat_0=31.2 +lon_0=121.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
WGS84 = CRS.from_epsg(4326)
# 数据根路径，按实际数据位置修改
BASE_PATH = Path("H:/大论文Result/大论文图")
POPULATION_FOLDER = "三大城市群的具体人口与占地面积"


OUTPUT_SVG = "长三角_人口密度_2024.svg"
OUTPUT_PNG = "长三角_人口密度_2024.png"
MISSING_OUTPUT = "长三角_缺失人口数据区域.csv"

SELECTED_CITIES = {
    "南京市", "无锡市", "南通市", "盐城市", "扬州市", "镇江市", "常州市", "苏州市", "泰州市",
    "杭州市", "宁波市", "嘉兴市", "湖州市", "绍兴市", "金华市", "舟山市", "台州市", "温州市",
    "合肥市", "芜湖市", "马鞍山市", "铜陵市", "安庆市", "滁州市", "池州市", "宣城市",
    "黄浦区", "徐汇区", "长宁区", "静安区", "普陀区", "虹口区", "杨浦区", "闵行区", "宝山区",
    "嘉定区", "浦东新区", "金山区", "松江区", "青浦区", "奉贤区", "崇明区",
}


def read_and_transform(file_path: Path, crs: CRS = LAEA_CRS) -> gpd.GeoDataFrame:
    """读取 GeoJSON 并转换到指定投影"""
    return gpd.read_file(file_path).to_crs(crs)


def parse_numeric(series, suffix: str) -> pd.Series:
    """去除单位并转换为数值"""
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series.astype(str).str.replace(suffix, "", regex=False), errors="coerce")


def _get_people_cols(df: pd.DataFrame) -> tuple[str, str, str]:
    """获取区域名称、人口、面积列名。5 列结构（含 href）时：区域=2, 人口=3, 面积=4"""
    cols = list(df.columns)
    if len(cols) >= 5:
        return cols[2], cols[3], cols[4]
    region = cols[1] if len(cols) > 1 else cols[0]
    pop = cols[2] if len(cols) > 2 else cols[1]
    area = cols[3] if len(cols) > 3 else cols[2]
    for c in cols:
        if str(c) in ("区域名称", "区域", "name", "column.2", "column-2"):
            region = c
            break
    for c in cols:
        if "人口" in str(c) or "万" in str(c):
            pop = c
            break
    for c in cols:
        if "面积" in str(c) or "km" in str(c).lower():
            area = c
            break
    return region, pop, area


def clean_people_data(file_path: Path, province_name: str) -> pd.DataFrame:
    """清理人口和面积数据并计算人口密度"""
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    columns = list(df.columns)
    region_col, pop_col, area_col = _get_people_cols(df)

    cleaned = (
        df.loc[df[columns[0]].astype(str) != "-", [region_col, pop_col, area_col]]
        .rename(columns={region_col: "区域名称", pop_col: "人口_万人", area_col: "面积_km2"})
        .assign(省份=province_name)
    )
    cleaned["人口_万人"] = parse_numeric(cleaned["人口_万人"], "万")
    cleaned["面积_km2"] = parse_numeric(cleaned["面积_km2"], "km²")
    cleaned = cleaned.loc[cleaned["面积_km2"].notna() & (cleaned["面积_km2"] > 0)].copy()
    cleaned["人口密度_人每平方公里"] = cleaned["人口_万人"] * 10000 / cleaned["面积_km2"]
    cleaned = cleaned.loc[
        cleaned["人口密度_人每平方公里"].notna()
        & (cleaned["人口密度_人每平方公里"] > 0)
        & np.isfinite(cleaned["人口密度_人每平方公里"])
    ].copy()
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
            ax.text(x_proj, ymin_l - (ymax_l - ymin_l) * 0.015, f"{int(lon)}°E", ha="center", va="top", fontsize=16)

    for lat in lats:
        _, y_proj = transformer.transform(xmin_w, lat)
        if ymin_l <= y_proj <= ymax_l:
            ax.text(xmin_l - (xmax_l - xmin_l) * 0.015, y_proj, f"{int(lat)}°N", ha="right", va="center", fontsize=16)

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
    x0 = xmin + pad_x
    x1 = x0 + bar_length_m
    y0 = ymin + pad_y

    ax.plot([x0, x1], [y0, y0], color="black", linewidth=2)
    ax.plot([x0, x0], [y0, y0 + pad_y * 0.3], color="black", linewidth=1)
    ax.plot([x1, x1], [y0, y0 + pad_y * 0.3], color="black", linewidth=1)
    ax.text((x0 + x1) / 2, y0 + pad_y * 0.45, f"{int(length_km)} km", ha="center", va="bottom", fontsize=16)


def infer_province(name: str) -> str:
    """根据名称推断所属省份"""
    shanghai_regions = {
        "黄浦区", "徐汇区", "长宁区", "静安区", "普陀区", "虹口区", "杨浦区",
        "闵行区", "宝山区", "嘉定区", "浦东新区", "金山区", "松江区", "青浦区", "奉贤区", "崇明区",
    }
    jiangsu_regions = {"南京市", "无锡市", "南通市", "盐城市", "扬州市", "镇江市", "常州市", "苏州市", "泰州市"}
    anhui_regions = {"合肥市", "芜湖市", "马鞍山市", "铜陵市", "安庆市", "滁州市", "池州市", "宣城市"}
    zhejiang_regions = {"杭州市", "宁波市", "嘉兴市", "湖州市", "绍兴市", "金华市", "舟山市", "台州市", "温州市"}

    if name in shanghai_regions:
        return "上海市"
    if name in jiangsu_regions:
        return "江苏省"
    if name in anhui_regions:
        return "安徽省"
    if name in zhejiang_regions:
        return "浙江省"
    return "其他"


def main() -> None:
    """主函数：绘制长三角地区人口密度分布图"""
    plt.rcParams["font.family"] = "Times New Roman"

    people_data = pd.concat(
        [
            clean_people_data(BASE_PATH / f"{POPULATION_FOLDER}/江苏省各区人口与占地面积（2024）.csv", "江苏省"),
            clean_people_data(BASE_PATH / f"{POPULATION_FOLDER}/安徽省各区人口与占地面积（2024）.csv", "安徽省"),
            clean_people_data(BASE_PATH / f"{POPULATION_FOLDER}/上海市各区人口与占地面积（2024）.csv", "上海市"),
            clean_people_data(BASE_PATH / f"{POPULATION_FOLDER}/浙江省各区人口与占地面积（2024）.csv", "浙江省"),
        ],
        ignore_index=True,
    )

    shanghai_geo = read_and_transform(BASE_PATH / "3.长三角/上海市 (划分).geojson")
    anhui_geo = read_and_transform(BASE_PATH / "3.长三角/安徽省 (市).geojson")
    jiangsu_geo = read_and_transform(BASE_PATH / "3.长三角/江苏省 (市).geojson")
    zhejiang_geo = read_and_transform(BASE_PATH / "3.长三角/浙江省 (市).geojson")

    geo_data = pd.concat(
        [
            shanghai_geo.loc[shanghai_geo["level"] == "district", ["name", "geometry"]],
            anhui_geo.loc[anhui_geo["level"] == "city", ["name", "geometry"]],
            jiangsu_geo.loc[jiangsu_geo["level"] == "city", ["name", "geometry"]],
            zhejiang_geo.loc[zhejiang_geo["level"] == "city", ["name", "geometry"]],
        ],
        ignore_index=True,
    )
    geo_data = geo_data.loc[geo_data["name"].isin(SELECTED_CITIES)].copy()
    # 诊断：若合并后全为缺失，检查两边名称是否一致
    people_names = set(people_data["区域名称"].dropna().astype(str).str.strip())
    geo_names = set(geo_data["name"].astype(str).str.strip())
    overlap = people_names & geo_names
    if len(overlap) == 0 and len(people_data) > 0 and len(geo_data) > 0:
        print("\n[诊断] 人口数据与地理数据名称无交集，无法匹配。")
        print(f"  CSV 区域名称示例: {list(people_names)[:10]}")
        print(f"  GeoJSON name 示例: {list(geo_names)[:10]}")
    merged_data = gpd.GeoDataFrame(
        geo_data.merge(people_data, left_on="name", right_on="区域名称", how="left"),
        geometry="geometry",
        crs=LAEA_CRS,
    )

    graticule_gdf, lats, lons = create_graticule(merged_data)
    bounds_wgs84 = tuple(merged_data.to_crs(WGS84).total_bounds)
    bounds_laea = tuple(merged_data.total_bounds)
    valid_density = merged_data["人口密度_人每平方公里"].dropna()
    color_norm = LogNorm(
        vmin=float(valid_density.min()) if len(valid_density) > 0 else 1,
        vmax=float(valid_density.max()) if len(valid_density) > 0 else 10000,
    )

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    graticule_gdf.plot(ax=ax, color="#cccccc", linewidth=0.3, linestyle="--")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.5)
    merged_data.plot(
        ax=ax,
        column="人口密度_人每平方公里",
        cmap="Reds",
        norm=color_norm,
        edgecolor="black",
        linewidth=0.3,
        legend=False,
        missing_kwds={"color": "#e6e6e6", "label": "No data"},
    )
    sm = ScalarMappable(cmap="Reds", norm=color_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Population density (persons/km², log scale)", fontsize=28, fontweight="bold")
    cbar.ax.tick_params(labelsize=28)

    draw_scalebar(ax, merged_data, length_km=200)
    draw_north_arrow(ax)
    add_graticule_labels(ax, lats, lons, bounds_wgs84, bounds_laea)

    ax.set_title("YRD Population Density 2024", fontsize=32, fontweight="bold")
    ax.set_xlabel("Longitude (°E)", fontsize=24, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=24, fontweight="bold")
    ax.set_aspect("equal")
    plt.tight_layout()

    plt.savefig(OUTPUT_SVG, format="svg", dpi=300, bbox_inches=None)
    plt.savefig(OUTPUT_PNG, format="png", dpi=300, bbox_inches=None)
    plt.show()

    missing = merged_data.loc[merged_data["人口密度_人每平方公里"].isna(), ["name"]].copy()
    missing["省份"] = missing["name"].map(infer_province)
    missing = missing[["name", "省份"]].sort_values(["省份", "name"]).reset_index(drop=True)

    print("\n=== 数据统计信息 ===")
    print(f"总区域数量: {len(merged_data)}")
    print(f"有人口数据的区域数量: {merged_data['人口密度_人每平方公里'].notna().sum()}")
    print(f"缺失人口数据的区域数量: {len(missing)}")

    if not missing.empty:
        missing.to_csv(MISSING_OUTPUT, index=False, encoding="utf-8-sig")
        print(f"\n缺失人口数据的区域已保存到: {MISSING_OUTPUT}")
        for _, row in missing.iterrows():
            print(f"{row['省份']}: {row['name']}")
    else:
        print("所有区域都有对应的人口数据！")

    valid_density = merged_data["人口密度_人每平方公里"].dropna()
    if not valid_density.empty:
        print("\n=== 人口密度数据统计 ===")
        print(f"人口密度范围: {valid_density.min():.2f} - {valid_density.max():.2f} 人/平方公里")
        print(f"人口密度平均值: {valid_density.mean():.2f} 人/平方公里")
        print(f"人口密度中位数: {valid_density.median():.2f} 人/平方公里")


if __name__ == "__main__":
    main()
