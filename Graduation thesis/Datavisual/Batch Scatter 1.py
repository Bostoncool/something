"""
Batch PM2.5 Actual vs Predicted Scatter Plot
遍历指定根目录的一级子文件夹，读取 predictions_test.csv，按 Scatter Plot.md 规范批量绘制散点图。
"""

import os
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 配置路径
ROOT_DIR = Path(r"H:\DATA Science\大论文Result\大论文图\机器学习结果")
OUTPUT_DIR = ROOT_DIR / "三大城市群散点图"

# 配色：柔和蓝绿色系
COLOR_SCATTER = "#2E86AB"  # 墨兰蓝
COLOR_YEQX = "#1B4965"  # 深蓝
COLOR_REGRESSION = "#5FA8D3"  # 柔和蓝
COLOR_CI = "#BEE9E8"  # 浅青
COLOR_MARGINAL = "#62B6CB"  # 蓝绿

# 列名映射（兼容不同 CSV 格式）
COLUMN_ALIASES = {
    "y_true": ["y_true", "Actual_PM25", "actual", "actual_pm25"],
    "y_pred": ["y_pred", "Predicted_PM25", "predicted", "predicted_pm25"],
}


def collect_valid_paths(root: Path) -> dict[str, dict[str, Path]]:
    """快速遍历，收集含 predictions_test.csv 的子文件夹路径。"""
    result: dict[str, dict[str, Path]] = {}
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue
            pred_path = Path(entry.path) / "predictions_test.csv"
            if pred_path.exists():
                result[entry.name] = {
                    "pred": pred_path,
                    "metrics": Path(entry.path) / "metrics_overall.csv",
                }
    return result


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将别名列映射为 y_true, y_pred。"""
    out = df.copy()
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in out.columns and target not in out.columns:
                out[target] = out[alias]
                break
    return out


def load_data(
    predictions_path: Path,
    metrics_path: Path | None = None,
) -> tuple[pd.DataFrame, dict] | None:
    """
    加载 predictions_test.csv 与可选 metrics_overall.csv。
    返回 (df, metrics_dict)，若数据无效则返回 None。
    """
    try:
        df = pd.read_csv(predictions_path, encoding="utf-8-sig")
    except Exception as e:
        print(f"  [跳过] 无法读取 {predictions_path}: {e}")
        return None

    df = _resolve_columns(df)
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        print(f"  [跳过] 缺少 y_true 或 y_pred 列: {predictions_path}")
        return None

    df = df[["y_true", "y_pred"]].dropna()
    if df.empty or len(df) < 2:
        print(f"  [跳过] 数据为空或样本不足: {predictions_path}")
        return None

    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    metrics: dict = {}
    if metrics_path is not None and metrics_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_path, encoding="utf-8-sig")
            if "split" in metrics_df.columns:
                test_row = metrics_df[metrics_df["split"] == "test"]
                if not test_row.empty:
                    metrics = test_row.iloc[0].to_dict()
        except Exception:
            pass

    if not metrics:
        metrics = {
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "n_samples": int(len(df)),
        }
    else:
        metrics.setdefault("r2", r2_score(y_true, y_pred))
        metrics.setdefault("rmse", np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics.setdefault("mae", mean_absolute_error(y_true, y_pred))
        metrics.setdefault("n_samples", len(df))

    return df, metrics


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
    output_path: Path,
    *,
    verbose: bool = True,
) -> None:
    """绘制带边缘直方图的散点图。"""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    n = len(y_true)

    r2 = metrics.get("r2", r2_score(y_true, y_pred))
    rmse = metrics.get("rmse", np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = metrics.get("mae", mean_absolute_error(y_true, y_pred))
    n_samples = int(metrics.get("n_samples", n))

    x_sorted, y_fit, y_lower, y_upper, slope, intercept = compute_regression_ci(
        y_true, y_pred
    )

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.scatter(
        y_true,
        y_pred,
        c=COLOR_SCATTER,
        alpha=0.5,
        s=20,
        edgecolors="none",
        label="Test samples",
    )

    all_vals = np.concatenate([y_true, y_pred])
    lim_min = max(0, np.floor(np.min(all_vals) / 10) * 10)
    lim_max = np.ceil(np.max(all_vals) / 10) * 10
    ax_main.set_xlim(lim_min, lim_max)
    ax_main.set_ylim(lim_min, lim_max)
    ax_main.set_aspect("equal")

    ax_main.fill_between(
        x_sorted,
        y_lower,
        y_upper,
        color=COLOR_CI,
        alpha=0.6,
        label="95% confidence interval",
    )

    ax_main.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        color=COLOR_YEQX,
        linestyle="-",
        linewidth=2,
        label="y = x (ideal)",
    )

    ax_main.plot(
        x_sorted,
        y_fit,
        color=COLOR_REGRESSION,
        linestyle="--",
        linewidth=2,
        label=f"Fitted line (y = {slope:.3f}x + {intercept:.2f})",
    )

    ax_main.grid(True, linestyle="--", color="gray", linewidth=0.5, alpha=0.7)
    ax_main.set_xlabel(r"Actual PM$_{2.5}$ ($\mu$g/m$^3$)", fontsize=14)
    ax_main.set_ylabel(r"Predicted PM$_{2.5}$ ($\mu$g/m$^3$)", fontsize=14)

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

    ax_main.legend(loc="lower right", fontsize=12)

    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

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

    fig.savefig(
        output_path,
        format="svg",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=False,
    )
    plt.close()
    if verbose:
        if tqdm:
            tqdm.write(f"Saved: {output_path}")
        else:
            print(f"Saved: {output_path}")


def _format_elapsed(seconds: float) -> str:
    """格式化耗时: 秒 -> 分:秒 或 时:分:秒"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.1f}s"
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {s:.1f}s"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    total_start = time.perf_counter()

    if not ROOT_DIR.exists():
        print(f"根目录不存在: {ROOT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    path_dict = collect_valid_paths(ROOT_DIR)
    scan_elapsed = time.perf_counter() - t0
    print(f"发现 {len(path_dict)} 个有效子文件夹 (耗时 {_format_elapsed(scan_elapsed)})")

    processed = 0
    items = sorted(path_dict.items())

    if tqdm:
        pbar = tqdm(items, desc="绘制散点图", unit="个", dynamic_ncols=True)
    else:
        pbar = items

    for name, paths in pbar:
        if tqdm:
            pbar.set_postfix_str(name, refresh=False)
        result = load_data(paths["pred"], paths["metrics"])
        if result is None:
            continue

        df, metrics = result
        base_name = name.replace("_daily_pm25", "")
        out_path = OUTPUT_DIR / f"{base_name}.svg"

        plot_scatter_with_marginals(df, metrics, out_path)
        processed += 1

    total_elapsed = time.perf_counter() - total_start
    print(f"\n完成，共处理 {processed} 个子文件夹，输出目录: {OUTPUT_DIR}")
    print(f"总耗时: {_format_elapsed(total_elapsed)}")


if __name__ == "__main__":
    main()
