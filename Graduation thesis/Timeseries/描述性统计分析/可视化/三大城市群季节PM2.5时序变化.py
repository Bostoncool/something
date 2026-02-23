import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


SEASON_ORDER = {
    "春": 1,
    "夏": 2,
    "秋": 3,
    "冬": 4,
    "spring": 1,
    "summer": 2,
    "autumn": 3,
    "fall": 3,
    "winter": 4,
    "q1": 1,
    "q2": 2,
    "q3": 3,
    "q4": 4,
}


def safe_print(message: str) -> None:
    """在终端编码不支持中文时，降级输出。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


def normalize_season_token(raw_token: str) -> str | None:
    """将季节标识归一化为 春/夏/秋/冬。"""
    token = str(raw_token).strip().lower()
    mapping = {
        "春": "春",
        "夏": "夏",
        "秋": "秋",
        "冬": "冬",
        "spring": "春",
        "summer": "夏",
        "autumn": "秋",
        "fall": "秋",
        "winter": "冬",
        "q1": "春",
        "q2": "夏",
        "q3": "秋",
        "q4": "冬",
    }
    return mapping.get(token)


def parse_season_label(label: str) -> tuple[int, int, str] | None:
    """
    解析季度列名，返回 (year, season_order, normalized_label)。
    支持示例：2018-春, 2018-冬, 2018-Q1, 2018-spring。
    """
    text = str(label).strip()

    match = re.match(r"^\s*(\d{4})\s*[-_/]\s*([A-Za-z0-9\u4e00-\u9fff]+)\s*$", text)
    if not match:
        return None

    year = int(match.group(1))
    season_token = normalize_season_token(match.group(2))
    if season_token is None:
        return None

    order = SEASON_ORDER[season_token]
    normalized_label = f"{year}-{season_token}"
    return year, order, normalized_label


def build_cluster_seasonal_mean_series(csv_path: str, pollutant: str = "PM2.5") -> pd.Series:
    """
    读取单个城市群的 Seasonal_Means.csv，筛选 PM2.5，并计算该城市群季节均值时序。
    返回索引为季度标签（YYYY-春/夏/秋/冬）的 Series。
    """
    df = pd.read_csv(csv_path).copy()
    required_cols = {"City", "Pollutant"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{csv_path} 缺少必要列：{required_cols}")

    df = df.loc[df["Pollutant"].astype(str).str.strip() == pollutant].copy()
    if df.empty:
        raise ValueError(f"{csv_path} 中未找到污染物 {pollutant} 的记录。")

    candidate_cols = [col for col in df.columns if col not in ("City", "Pollutant")]
    parsed = []
    for col in candidate_cols:
        parsed_info = parse_season_label(col)
        if parsed_info is not None:
            parsed.append((col, *parsed_info))

    if not parsed:
        raise ValueError(f"{csv_path} 中未识别到可用季节列。")

    parsed.sort(key=lambda x: (x[1], x[2]))  # year, season_order
    season_cols = [item[0] for item in parsed]
    season_labels = [item[3] for item in parsed]

    season_df = df[season_cols].apply(pd.to_numeric, errors="coerce")
    seasonal_mean = season_df.mean(axis=0)
    seasonal_mean.index = season_labels
    seasonal_mean.name = "PM2.5"
    return seasonal_mean


def plot_single_cluster_seasonal_series(
    seasonal_series: pd.Series,
    cluster_name: str,
    output_dir: Path,
    line_color: str,
) -> Path:
    """绘制单个城市群季节平均 PM2.5 时序图，并保存为 SVG。"""
    fig, ax = plt.subplots(figsize=(13.5, 5.8), dpi=150)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    x_positions = range(len(seasonal_series))
    ax.plot(
        x_positions,
        seasonal_series.values,
        color=line_color,
        linewidth=2.0,
        marker="o",
        markersize=3.2,
        alpha=0.95,
        zorder=3,
    )

    ax.set_title(f"{cluster_name}季节平均PM2.5时序变化", fontsize=14, pad=10)
    ax.set_xlabel("时间", fontsize=12)
    ax.set_ylabel("PM2.5/(μg/m³)", fontsize=12, fontfamily="Times New Roman")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(seasonal_series.index, rotation=45, ha="right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", width=1.0, length=5, labelsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    output_path = output_dir / f"{cluster_name}_季节平均PM2.5时序变化.svg"
    fig.savefig(output_path, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    csv_paths = {
        "京津冀城市群(BTH)": r"H:\DATA Science\大论文Result\BTH\描述性统计分析\Seasonal_Means.csv",
        "长江三角洲城市群(YZD)": r"H:\DATA Science\大论文Result\YZD\描述性统计分析\Seasonal_Means.csv",
        "珠江三角洲城市群(PRD)": r"H:\DATA Science\大论文Result\PRD\描述性统计分析\Seasonal_Means.csv",
    }

    mpl.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    mpl.rcParams["axes.unicode_minus"] = False

    color_map = {
        "京津冀城市群(BTH)": "#d62828",
        "长江三角洲城市群(YZD)": "#1d3557",
        "珠江三角洲城市群(PRD)": "#2a9d8f",
    }

    output_dir = Path(__file__).resolve().parent

    for cluster_name, csv_path in csv_paths.items():
        seasonal_series = build_cluster_seasonal_mean_series(csv_path, pollutant="PM2.5")
        output_figure_path = plot_single_cluster_seasonal_series(
            seasonal_series=seasonal_series,
            cluster_name=cluster_name,
            output_dir=output_dir,
            line_color=color_map[cluster_name],
        )

        output_data_path = output_dir / f"{cluster_name}_季节平均PM2.5序列.csv"
        seasonal_series.to_frame().to_csv(output_data_path, encoding="utf-8-sig")

        safe_print(f"{cluster_name} 图片已保存到: {output_figure_path}")
        safe_print(f"{cluster_name} 数据已保存到: {output_data_path}")


if __name__ == "__main__":
    main()
