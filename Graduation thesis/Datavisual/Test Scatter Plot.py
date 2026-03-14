"""
PM2.5 Actual vs Predicted Scatter Plot
符合 Scatter Plot.md 规范：测试集散点、y=x 理想线、拟合回归线、95% 置信区间、边缘直方图
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 配置路径
DATA_DIR = Path(r"h:\DATA Science\大论文Result\大论文图\机器学习结果\adaboost_daily_pm25\bth")
REGRESSION_FILE = DATA_DIR / "regression_test_data.csv"
METRICS_FILE = DATA_DIR / "metrics_overall.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "scatter_pm25_bth.svg"

# 配色：柔和蓝绿色系
COLOR_SCATTER = "#2E86AB"  # 墨兰蓝
COLOR_YEQX = "#1B4965"  # 深蓝
COLOR_REGRESSION = "#5FA8D3"  # 柔和蓝
COLOR_CI = "#BEE9E8"  # 浅青
COLOR_MARGINAL = "#62B6CB"  # 蓝绿


def load_data(
    regression_path: Path = REGRESSION_FILE,
    metrics_path: Path = METRICS_FILE,
) -> tuple[pd.DataFrame, dict]:
    """加载散点数据与指标。"""
    df = pd.read_csv(regression_path)
    df = df[df["split"] == "test"].copy()
    metrics_df = pd.read_csv(metrics_path)
    test_metrics = metrics_df[metrics_df["split"] == "test"].iloc[0].to_dict()
    return df, test_metrics


def compute_regression_ci(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    线性回归拟合与 95% 预测区间。
    返回: x_sorted, y_fit, y_lower, y_upper, slope, intercept
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_fit = slope * x + intercept
    residuals = y - y_fit
    n = len(x)
    mse = np.sum(residuals**2) / (n - 2)
    x_mean = np.mean(x)
    s_xx = np.sum((x - x_mean) ** 2)
    t_crit = stats.t.ppf(1 - alpha / 2, n - 2)
    se_pred = np.sqrt(mse * (1 + 1 / n + (x - x_mean) ** 2 / s_xx))
    y_lower = y_fit - t_crit * se_pred
    y_upper = y_fit + t_crit * se_pred
    idx = np.argsort(x)
    return x[idx], y_fit[idx], y_lower[idx], y_upper[idx], slope, intercept


def plot_scatter_with_marginals(
    df: pd.DataFrame,
    metrics: dict,
    output_path: Path = OUTPUT_FILE,
) -> None:
    """绘制带边缘直方图的散点图。"""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    n = len(y_true)

    # 指标（优先用 CSV，否则重算）
    r2 = metrics.get("r2", r2_score(y_true, y_pred))
    rmse = metrics.get("rmse", np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = metrics.get("mae", mean_absolute_error(y_true, y_pred))
    n_samples = int(metrics.get("n_samples", n))

    # 回归与 95% 置信区间
    x_sorted, y_fit, y_lower, y_upper, slope, intercept = compute_regression_ci(
        y_true, y_pred
    )

    # 画布：正方形，GridSpec 布局
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # 主散点图
    ax_main.scatter(
        y_true,
        y_pred,
        c=COLOR_SCATTER,
        alpha=0.5,
        s=20,
        edgecolors="none",
        label="Test samples",
    )

    # 坐标范围（正方形）
    all_vals = np.concatenate([y_true, y_pred])
    lim_min = max(0, np.floor(np.min(all_vals) / 10) * 10)
    lim_max = np.ceil(np.max(all_vals) / 10) * 10
    ax_main.set_xlim(lim_min, lim_max)
    ax_main.set_ylim(lim_min, lim_max)
    ax_main.set_aspect("equal")

    # 95% 置信区间
    ax_main.fill_between(
        x_sorted,
        y_lower,
        y_upper,
        color=COLOR_CI,
        alpha=0.6,
        label="95% confidence interval",
    )

    # y = x 理想线
    ax_main.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        color=COLOR_YEQX,
        linestyle="-",
        linewidth=2,
        label="y = x (ideal)",
    )

    # 拟合回归线
    ax_main.plot(
        x_sorted,
        y_fit,
        color=COLOR_REGRESSION,
        linestyle="--",
        linewidth=2,
        label=f"Fitted line (y = {slope:.3f}x + {intercept:.2f})",
    )

    # 网格
    ax_main.grid(True, linestyle="--", color="gray", linewidth=0.5, alpha=0.7)

    # 坐标轴标签（PM2.5 下标）
    ax_main.set_xlabel(r"Actual PM$_{2.5}$ ($\mu$g/m$^3$)", fontsize=14)
    ax_main.set_ylabel(r"Predicted PM$_{2.5}$ ($\mu$g/m$^3$)", fontsize=14)

    # 指标注释
    metrics_text = (
        f"$R^2$ = {r2:.4f}\n"
        f"RMSE = {rmse:.2f}\n"
        f"MAE = {mae:.2f}\n"
        f"N = {n_samples}"
    )
    ax_main.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax_main.transAxes,
        fontsize=16,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 图例
    ax_main.legend(loc="lower right", fontsize=12)

    # 隐藏顶部、右侧子图的刻度
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # 顶部边缘直方图（actual / y_true）
    ax_top.hist(
        y_true,
        bins=40,
        range=(lim_min, lim_max),
        color=COLOR_MARGINAL,
        alpha=0.7,
        edgecolor="white",
    )
    ax_top.set_frame_on(False)
    ax_top.set_xlim(lim_min, lim_max)
    ax_top.set_yticks([])
    ax_top.tick_params(axis="both", which="both", length=0)

    # 右侧边缘直方图（predicted / y_pred）
    ax_right.hist(
        y_pred,
        bins=40,
        range=(lim_min, lim_max),
        color=COLOR_MARGINAL,
        alpha=0.7,
        edgecolor="white",
        orientation="horizontal",
    )
    ax_right.set_frame_on(False)
    ax_right.set_ylim(lim_min, lim_max)
    ax_right.set_xticks([])
    ax_right.tick_params(axis="both", which="both", length=0)

    for ax in [ax_main, ax_top, ax_right]:
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(11)

    # 保存 SVG（正方形画布，无标题）
    fig.savefig(
        output_path,
        format="svg",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=False,
    )
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    df, metrics = load_data()
    plot_scatter_with_marginals(df, metrics)


if __name__ == "__main__":
    main()
