"""
三大城市群年均 PM2.5 可视化（只读版）
直接读取预计算的城市群年度均值 CSV，绘制分组柱状图。
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path


def safe_print(message: str) -> None:
    """在 Windows 终端编码不支持中文时，降级输出。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


def main() -> None:
    # 数据路径（预计算的城市群年度均值）
    data_path = r"h:\大论文Result\三大城市群_PM25_城市群年度均值.csv"

    # 读取数据（保持 BTH, YRD, PRD 英文缩写）
    plot_df = pd.read_csv(data_path, index_col=0)
    plot_df.index = plot_df.index.astype(str).str.strip()

    years = plot_df.columns.astype(str).tolist()

    # 字体设置：全部使用 Times New Roman（参考 Study area.py）
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=150)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    x = np.arange(len(plot_df.index))
    bar_width = 0.11
    offsets = (np.arange(len(years)) - (len(years) - 1) / 2) * bar_width

    year_colors = [
        "#8ecae6",  # 浅天蓝
        "#ffafcc",  # 浅粉
        "#bde0fe",  # 冰蓝
        "#cdb4db",  # 浅紫
        "#ffd6a5",  # 浅橙
        "#caffbf",  # 浅绿
    ][: len(years)]

    for i, year in enumerate(years):
        ax.bar(
            x + offsets[i],
            plot_df[year].values,
            width=bar_width,
            color=year_colors[i],
            edgecolor="#3f3f3f",
            linewidth=0.7,
            zorder=3,
            label=year,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, fontsize=30)
    ax.set_ylabel(r"PM$_{2.5}$/(μg/m³)", fontsize=24, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", width=1.0, length=5, labelsize=22)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    ax.legend(loc="upper right", frameon=False, fontsize=14)

    output_path = Path(__file__).parent / "三大城市群_PM25_分组柱状图_只读.svg"
    plt.savefig(output_path, format="svg", transparent=True)
    plt.close(fig)

    safe_print(f"图片已保存到: {output_path}")


if __name__ == "__main__":
    main()
