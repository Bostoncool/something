import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


def safe_print(message: str) -> None:
    """在 Windows 终端编码不支持中文时，降级输出。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


def build_cluster_monthly_mean_series(csv_path: str, pollutant: str = "PM2.5") -> pd.Series:
    """
    读取单个城市群的 Monthly_Means.csv，筛选 PM2.5，并计算该城市群月均值时序。
    返回索引为 datetime（按月）的 Series。
    """
    df = pd.read_csv(csv_path).copy()
    required_cols = {"City", "Pollutant"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{csv_path} 缺少必要列：{required_cols}")

    df = df.loc[df["Pollutant"].astype(str).str.strip() == pollutant].copy()
    if df.empty:
        raise ValueError(f"{csv_path} 中未找到污染物 {pollutant} 的记录。")

    month_cols = [col for col in df.columns if col not in ("City", "Pollutant")]
    if not month_cols:
        raise ValueError(f"{csv_path} 中未找到月份列。")

    month_df = df[month_cols].apply(pd.to_numeric, errors="coerce")
    monthly_mean = month_df.mean(axis=0)

    monthly_mean.index = pd.to_datetime(monthly_mean.index, format="%Y-%m", errors="coerce")
    monthly_mean = monthly_mean[~monthly_mean.index.isna()]
    monthly_mean = monthly_mean.sort_index()
    monthly_mean.name = "PM2.5"
    return monthly_mean


def plot_single_cluster_monthly_series(
    monthly_series: pd.Series,
    cluster_name: str,
    output_dir: Path,
    line_color: str,
) -> Path:
    """绘制单个城市群月均 PM2.5 时序图，并保存为 SVG。"""
    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=150)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    ax.plot(
        monthly_series.index,
        monthly_series.values,
        color=line_color,
        linewidth=2.0,
        marker="o",
        markersize=3.0,
        alpha=0.95,
        zorder=3,
    )

    ax.set_title(f"{cluster_name}月均PM2.5时序变化", fontsize=14, pad=10)
    ax.set_xlabel("时间", fontsize=12)
    ax.set_ylabel("PM2.5/(μg/m³)", fontsize=12, fontfamily="Times New Roman")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", width=1.0, length=5, labelsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    output_path = output_dir / f"{cluster_name}_月均PM2.5时序变化.svg"
    fig.savefig(output_path, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    # 数据路径
    csv_paths = {
        "京津冀城市群(BTH)": r"H:\DATA Science\大论文Result\BTH\描述性统计分析\Monthly_Means.csv",
        "长江三角洲城市群(YZD)": r"H:\DATA Science\大论文Result\YZD\描述性统计分析\Monthly_Means.csv",
        "珠江三角洲城市群(PRD)": r"H:\DATA Science\大论文Result\PRD\描述性统计分析\Monthly_Means.csv",
    }

    # 字体设置
    mpl.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    mpl.rcParams["axes.unicode_minus"] = False

    color_map = {
        "京津冀城市群(BTH)": "#d62828",
        "长江三角洲城市群(YZD)": "#1d3557",
        "珠江三角洲城市群(PRD)": "#2a9d8f",
    }

    output_dir = Path(__file__).resolve().parent

    for cluster_name, csv_path in csv_paths.items():
        monthly_series = build_cluster_monthly_mean_series(csv_path, pollutant="PM2.5")
        output_figure_path = plot_single_cluster_monthly_series(
            monthly_series=monthly_series,
            cluster_name=cluster_name,
            output_dir=output_dir,
            line_color=color_map[cluster_name],
        )

        output_data_path = output_dir / f"{cluster_name}_月均PM2.5序列.csv"
        monthly_series.rename("PM2.5").to_frame().to_csv(output_data_path, encoding="utf-8-sig")

        safe_print(f"{cluster_name} 图片已保存到: {output_figure_path}")
        safe_print(f"{cluster_name} 数据已保存到: {output_data_path}")


if __name__ == "__main__":
    main()
