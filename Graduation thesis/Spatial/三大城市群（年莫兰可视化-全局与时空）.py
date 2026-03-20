"""
三大城市群年度莫兰指数可视化脚本

读取 三大城市群（nc年3莫兰）.py 生成的 CSV 结果，绘制：
- 全局莫兰：折线图、柱状图
- 时空莫兰（STMI）：热力图、多线折线图
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DPI = 300
DEFAULT_INPUT_DIR = Path(r"H:\大论文Result\大论文图\三大城市群\三大城市群_（年）莫兰结果（英文）")


def setup_matplotlib() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
    plt.rcParams["axes.unicode_minus"] = False


def load_csv(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载 global_moran_summary.csv 和 stmi_summary.csv"""
    global_path = input_dir / "global_moran_summary.csv"
    stmi_path = input_dir / "stmi_summary.csv"
    if not global_path.exists():
        raise FileNotFoundError(
            f"未找到 {global_path}，请先运行 三大城市群（nc年3莫兰）.py 生成数据。"
        )
    if not stmi_path.exists():
        raise FileNotFoundError(
            f"未找到 {stmi_path}，请先运行 三大城市群（nc年3莫兰）.py 生成数据。"
        )
    global_df = pd.read_csv(global_path, encoding="utf-8-sig")
    stmi_df = pd.read_csv(stmi_path, encoding="utf-8-sig")
    return global_df, stmi_df


def plot_global_moran_timeseries(
    global_df: pd.DataFrame,
    output_path: Path,
    fmt: str = "svg",
) -> None:
    """全局莫兰 I 时间序列折线图"""
    if global_df.empty:
        print("警告: global_moran_summary 为空，跳过折线图。")
        return
    global_df = global_df.sort_values(["cluster", "period"])
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"BTH": "#1f77b4", "YRD": "#2ca02c", "PRD": "#ff7f0e"}
    for cluster in ["BTH", "YRD", "PRD"]:
        sub = global_df[global_df["cluster"] == cluster]
        if sub.empty:
            continue
        sub = sub.sort_values("period")
        ax.plot(
            sub["period"].astype(str),
            sub["global_i"],
            marker="o",
            label=cluster,
            linewidth=2,
            markersize=8,
            color=colors.get(cluster, None),
        )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    ymin, ymax = global_df["global_i"].min(), global_df["global_i"].max()
    margin = max((ymax - ymin) * 0.15, 0.03)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Global Moran's I", fontsize=12)
    ax.set_title("Global Moran's I Temporal Trend by City Cluster", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {output_path}")


def plot_global_moran_bar(
    global_df: pd.DataFrame,
    output_path: Path,
    fmt: str = "svg",
) -> None:
    """全局莫兰 I 分组柱状图"""
    if global_df.empty:
        print("警告: global_moran_summary 为空，跳过柱状图。")
        return
    pivot = global_df.pivot_table(
        index="period", columns="cluster", values="global_i"
    )
    pivot = pivot.reindex(columns=["BTH", "YRD", "PRD"])
    if pivot.empty:
        print("警告: pivot 为空，跳过柱状图。")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(pivot.index))
    width = 0.25
    multipliers = [-1, 0, 1]
    colors = {"BTH": "#1f77b4", "YRD": "#2ca02c", "PRD": "#ff7f0e"}
    for i, cluster in enumerate(["BTH", "YRD", "PRD"]):
        if cluster not in pivot.columns:
            continue
        offset = width * multipliers[i]
        ax.bar(
            x + offset,
            pivot[cluster],
            width,
            label=cluster,
            color=colors.get(cluster, None),
        )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    bar_vals = pivot.values.flatten()
    bar_vals = bar_vals[np.isfinite(bar_vals)]
    if len(bar_vals) > 0:
        ymin, ymax = np.min(bar_vals), np.max(bar_vals)
        margin = max((ymax - ymin) * 0.15, 0.03)
        ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Global Moran's I", fontsize=12)
    ax.set_title("Global Moran's I by Year and City Cluster", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.astype(str))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {output_path}")


def plot_stmi_heatmap(
    stmi_df: pd.DataFrame,
    cluster_key: str,
    output_path: Path,
    fmt: str = "svg",
) -> None:
    """STMI 热力图：行=period_t，列=lag，值=stmi_i"""
    sub = stmi_df[stmi_df["cluster"] == cluster_key].copy()
    if sub.empty:
        print(f"警告: {cluster_key} 无 STMI 数据，跳过热力图。")
        return
    pivot = sub.pivot_table(
        index="period_t", columns="lag", values="stmi_i", aggfunc="mean"
    )
    if pivot.empty:
        print(f"警告: {cluster_key} pivot 为空，跳过热力图。")
        return
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 0.01)
    im = ax.imshow(
        pivot.values,
        cmap="RdYlBu_r",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    ax.set_xlabel("Temporal Lag (years)")
    ax.set_ylabel("Period (year)")
    ax.set_title(f"STMI Heatmap - {cluster_key}\n(Spatiotemporal Moran's I)")
    plt.colorbar(im, ax=ax, label="STMI")
    fig.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {output_path}")


def plot_stmi_by_lag(
    stmi_df: pd.DataFrame,
    cluster_key: str,
    output_path: Path,
    fmt: str = "svg",
) -> None:
    """STMI 多线折线图：每条线代表一个 lag"""
    sub = stmi_df[stmi_df["cluster"] == cluster_key].copy()
    if sub.empty:
        print(f"警告: {cluster_key} 无 STMI 数据，跳过多线折线图。")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    lag_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2",
    ]
    for i, lag in enumerate(sorted(sub["lag"].unique())):
        sub_lag = sub[sub["lag"] == lag].sort_values("period_t")
        ax.plot(
            sub_lag["period_t"].astype(str),
            sub_lag["stmi_i"],
            marker="o",
            label=f"Lag {lag}",
            linewidth=2,
            color=lag_colors[i % len(lag_colors)],
        )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    ymin, ymax = sub["stmi_i"].min(), sub["stmi_i"].max()
    margin = max((ymax - ymin) * 0.15, 0.05)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlabel("Year (period_t)", fontsize=12)
    ax.set_ylabel("STMI", fontsize=12)
    ax.set_title(f"Spatiotemporal Moran's Index by Lag - {cluster_key}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, format=fmt, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="三大城市群全局莫兰与时空莫兰可视化"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="CSV 所在目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认与 input-dir 一致",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="svg",
        choices=["svg", "png"],
        help="输出格式",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format

    setup_matplotlib()
    global_df, stmi_df = load_csv(input_dir)

    plot_global_moran_timeseries(
        global_df,
        output_dir / f"global_moran_timeseries.{fmt}",
        fmt=fmt,
    )
    plot_global_moran_bar(
        global_df,
        output_dir / f"global_moran_bar.{fmt}",
        fmt=fmt,
    )

    for cluster in ["BTH", "YRD", "PRD"]:
        plot_stmi_heatmap(
            stmi_df,
            cluster,
            output_dir / f"stmi_heatmap_{cluster}.{fmt}",
            fmt=fmt,
        )
        plot_stmi_by_lag(
            stmi_df,
            cluster,
            output_dir / f"stmi_by_lag_{cluster}.{fmt}",
            fmt=fmt,
        )

    print("全部可视化完成。")


if __name__ == "__main__":
    main()
