import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path


def normalize_city_name(city_name: str) -> str:
    """统一城市名称，便于城市匹配。"""
    city_name = str(city_name).strip()
    if city_name.endswith("市"):
        city_name = city_name[:-1]
    return city_name


def build_cluster_year_mean_df(city_pm25_csv_path: str, cluster_cities: dict[str, list[str]]) -> pd.DataFrame:
    """根据城市年度 PM2.5 数据，计算三大城市群每年的平均浓度。"""
    city_df = pd.read_csv(city_pm25_csv_path).copy()
    if "城市" not in city_df.columns:
        raise ValueError("CSV 中未找到 `城市` 列。")

    city_df["城市"] = city_df["城市"].map(normalize_city_name)
    year_cols = [col for col in city_df.columns if col != "城市"]
    if not year_cols:
        raise ValueError("CSV 中未找到年份列。")

    city_df[year_cols] = city_df[year_cols].apply(pd.to_numeric, errors="coerce")
    city_df = city_df.set_index("城市")

    cluster_series = {}
    for cluster_name, cities in cluster_cities.items():
        normalized_cities = [normalize_city_name(city) for city in cities]
        matched_cities = [city for city in normalized_cities if city in city_df.index]
        if not matched_cities:
            cluster_series[cluster_name] = pd.Series(np.nan, index=year_cols)
            print(f"[警告] {cluster_name} 未匹配到任何城市。")
            continue

        missing_cities = sorted(set(normalized_cities) - set(matched_cities))
        if missing_cities:
            print(f"[提示] {cluster_name} 有 {len(missing_cities)} 个城市未在 CSV 中找到：{missing_cities}")

        cluster_series[cluster_name] = city_df.loc[matched_cities, year_cols].mean(axis=0)

    plot_df = pd.DataFrame(cluster_series).T
    plot_df = plot_df.dropna(how="all")
    if plot_df.empty:
        raise ValueError("城市群均值为空，请检查 CSV 数据与城市群城市列表。")
    return plot_df


def safe_print(message: str) -> None:
    """在 Windows 终端编码不支持中文时，降级输出。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


# 图内 PM₂.₅：Unicode 下标（₂、₅），符合颗粒物粒径标注习惯，与 Times New Roman 混排无 mathtext 冲突
PM25_AXIS_LABEL = "PM\u2082.\u2085/(μg/m³)"

# 图例与柱序标签用英文缩写（数据仍按原城市群键名聚合）
CLUSTER_DISPLAY_NAMES = {
    "京津冀城市群": "BTH",
    "长江三角洲城市群": "YRD",
    "珠江三角洲城市群": "PRD",
}


def parse_year(col) -> int | None:
    """从列名解析年份：支持 2018、2018.0、2018年、Y2018 等常见导出格式。"""
    s = str(col).strip()
    try:
        v = float(s)
        y = int(round(v))
        if 1900 <= y <= 2100:
            return y
    except (TypeError, ValueError):
        pass
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        return int(m.group(0))
    return None


def main() -> None:
    city_pm25_csv_path = r"H:\大论文Result\三大城市群（市）年度PM2.5浓度.csv"

    cluster_cities = {
        "京津冀城市群": [
            "北京",
            "天津",
            "石家庄",
            "唐山",
            "秦皇岛",
            "邯郸",
            "邢台",
            "保定",
            "张家口",
            "承德",
            "沧州",
            "廊坊",
            "衡水",
        ],
        "长江三角洲城市群": [
            "上海",
            "南京",
            "无锡",
            "南通",
            "盐城",
            "扬州",
            "镇江",
            "常州",
            "苏州",
            "泰州",
            "杭州",
            "宁波",
            "嘉兴",
            "湖州",
            "绍兴",
            "金华",
            "舟山",
            "台州",
            "温州",
            "合肥",
            "芜湖",
            "马鞍山",
            "铜陵",
            "安庆",
            "滁州",
            "池州",
            "宣城",
        ],
        "珠江三角洲城市群": [
            "广州",
            "深圳",
            "佛山",
            "东莞",
            "中山",
            "惠州",
            "珠海",
            "江门",
            "肇庆",
        ],
    }

    plot_df = build_cluster_year_mean_df(city_pm25_csv_path, cluster_cities)

    year_cols: list = []
    for c in plot_df.columns:
        y = parse_year(c)
        if y is not None and 2018 <= y <= 2023:
            year_cols.append(c)
    year_cols = sorted(year_cols, key=lambda c: parse_year(c))
    if not year_cols:
        hint = ", ".join(f"{c!r}→{parse_year(c)}" for c in plot_df.columns)
        raise ValueError(
            "2018–2023 年无可用列。请检查 CSV 年份列名是否与数据一致。各列解析: " + hint
        )

    plot_df = plot_df[year_cols]
    clusters = plot_df.index.tolist()
    n_cluster = len(clusters)

    # 字体与画布设置（与《三大城市群年均PM2.5时序（新）.py》一致）
    mpl.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    mpl.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    x = np.arange(len(year_cols))
    bar_width = 0.24
    offsets = (np.arange(n_cluster) - (n_cluster - 1) / 2) * bar_width
    # 柔和粉彩：蓝 / 玫瑰 / 青绿，饱和度与对比度略降
    cluster_colors = ["#a8c4dd", "#e8b8bc", "#a5cfc4"][:n_cluster]

    bar_containers = []
    for i, cluster in enumerate(clusters):
        display_name = CLUSTER_DISPLAY_NAMES.get(cluster, cluster)
        bar_containers.append(
            ax.bar(
                x + offsets[i],
                plot_df.loc[cluster, year_cols].values,
                width=bar_width,
                color=cluster_colors[i],
                edgecolor="#a8a8a8",
                linewidth=0.7,
                zorder=3,
                label=display_name,
            )
        )

    for container in bar_containers:
        ax.bar_label(
            container,
            fmt="%.1f",
            padding=3,
            fontsize=11,
            fontfamily="Times New Roman",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(parse_year(c)) for c in year_cols])
    ax.set_xlabel("Year", fontsize=14, fontfamily="Times New Roman")
    ax.set_ylabel(PM25_AXIS_LABEL, fontsize=14, fontfamily="Times New Roman")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", width=1.0, length=5)
    # 刻度数字、纵轴英文单位等：Times New Roman
    plt.setp(ax.get_xticklabels(), fontfamily="Times New Roman", fontsize=13)
    plt.setp(ax.get_yticklabels(), fontfamily="Times New Roman", fontsize=13)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(
        loc="upper right",
        frameon=False,
        prop={"family": "Times New Roman", "size": 13},
    )

    output_path = Path(__file__).with_name("三大城市群_PM25_按年三城市群并列柱状图.svg")
    output_data_path = Path(__file__).with_name("三大城市群_PM25_按年三城市群_2018-2023均值.csv")
    plot_df.to_csv(output_data_path, encoding="utf-8-sig")
    plt.savefig(output_path, format="svg", transparent=True)
    plt.close(fig)

    safe_print(f"图片已保存到: {output_path}")
    safe_print(f"汇总数据已保存到: {output_data_path}")


if __name__ == "__main__":
    main()
