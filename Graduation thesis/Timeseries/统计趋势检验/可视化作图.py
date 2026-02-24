"""
三大城市群统计趋势检验 —— 绘图模块
从检验结果 CSV 与聚合序列 CSV 生成所有可视化图表。
需要先运行 三大城市群统计趋势检验（新）.py 生成结果文件。
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# =========================
# 路径配置（按需修改）
# =========================
OUTPUT_DIR = r"H:\DATA Science\大论文Result\大论文图\三大城市群\统计趋势检验"


def safe_print(*args, **kwargs) -> None:
    """兼容终端编码差异的安全输出。"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        message = sep.join(str(x) for x in args)
        fallback = message.encode("ascii", errors="backslashreplace").decode("ascii")
        sys.stdout.write(fallback + end)


def configure_chinese_font() -> None:
    """配置中文字体，避免图中出现方框。"""
    chinese_font_chain = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = chinese_font_chain
    mpl.rcParams["axes.unicode_minus"] = False


def save_figure_dual(fig: plt.Figure, save_path_png: str, dpi: int = 300) -> None:
    """同时保存 PNG 与 SVG。"""
    fig.savefig(save_path_png, dpi=dpi, bbox_inches="tight")
    save_path_svg = os.path.splitext(save_path_png)[0] + ".svg"
    fig.savefig(save_path_svg, format="svg", bbox_inches="tight")


def read_csv_flexible(path: str) -> pd.DataFrame:
    """兼容多种编码读取 CSV。"""
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312"]
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"无法读取文件: {path}") from last_error


def _sort_groups(groups: List[str]) -> List[str]:
    group_order = ["京津冀", "长三角", "珠三角"]
    return sorted(groups, key=lambda g: group_order.index(g) if g in group_order else 99)


def _extract_change_label(row: pd.Series) -> Optional[str]:
    change_cols = [col for col in row.index if str(col).startswith("Change_")]
    for col in change_cols:
        value = row.get(col)
        if pd.notna(value):
            return str(value)
    return None


def _annotate_bar_values(ax: plt.Axes, bars: List[object], decimals: int = 3) -> None:
    y_min, y_max = ax.get_ylim()
    y_range = max(y_max - y_min, 1e-9)
    # 文本上边界安全距离，避免越过坐标框并与标题区域重叠。
    y_top_margin = 0.02 * y_range
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        raw_y = height + (0.005 if height <= 0.1 else 0.02)
        inside_top = y_max - y_top_margin
        text_y = min(raw_y, inside_top)
        va_style = "bottom" if text_y >= raw_y - 1e-12 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            f"{height:.{decimals}f}",
            ha="center",
            va=va_style,
            fontsize=8,
            clip_on=True,
        )


def _pick_annotation_offset(
    ax: plt.Axes,
    x_val: float,
    y_val: float,
    all_line_points: np.ndarray,
) -> Tuple[int, int]:
    """在多个候选位置中选择更不遮挡折线的标注偏移（单位：points）。"""
    candidate_offsets = [
        (10, 12), (10, -18), (-56, 12), (-56, -18),
        (20, 24), (20, -28), (-72, 24), (-72, -28),
        (0, 30), (0, -30), (34, 0), (-70, 0),
    ]
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    best_score = -np.inf
    best_offset = candidate_offsets[0]
    has_inside_candidate = False

    for dx, dy in candidate_offsets:
        anchor_disp = ax.transData.transform((x_val, y_val)) + np.array([dx, dy], dtype=float)
        anchor_x, anchor_y = ax.transData.inverted().transform(anchor_disp)
        inside_axes = (x_min <= anchor_x <= x_max) and (y_min <= anchor_y <= y_max)
        if not inside_axes:
            continue
        has_inside_candidate = True

        if all_line_points.size > 0:
            norm_dist2 = (
                ((all_line_points[:, 0] - anchor_x) / x_range) ** 2
                + ((all_line_points[:, 1] - anchor_y) / y_range) ** 2
            )
            min_dist_score = float(np.min(norm_dist2))
        else:
            min_dist_score = 1.0

        from_point_score = float(((anchor_x - x_val) / x_range) ** 2 + ((anchor_y - y_val) / y_range) ** 2)
        score = min_dist_score + 0.25 * from_point_score

        if score > best_score:
            best_score = score
            best_offset = (dx, dy)
    if not has_inside_candidate:
        return (-56, -18)
    return best_offset


def _choose_sparse_ticks(x_pos: np.ndarray, x_labels: List[str], max_ticks: int = 18) -> tuple[np.ndarray, List[str]]:
    """当横轴点过多时，自动抽稀刻度标签，避免重叠。"""
    if len(x_pos) <= max_ticks:
        return x_pos, x_labels
    step = int(np.ceil(len(x_pos) / max_ticks))
    tick_idx = np.arange(0, len(x_pos), step, dtype=int)
    if tick_idx[-1] != len(x_pos) - 1:
        tick_idx = np.append(tick_idx, len(x_pos) - 1)
    tick_pos = x_pos[tick_idx]
    tick_labels = [x_labels[i] for i in tick_idx]
    return tick_pos, tick_labels


def plot_group_series(
    group_series: Dict[str, pd.Series],
    title: str,
    xlabel: str,
    output_png_path: str,
    annual_mk_df: Optional[pd.DataFrame] = None,
    pettitt_df: Optional[pd.DataFrame] = None,
    sliding_t_df: Optional[pd.DataFrame] = None,
    annotate_sen_slope: bool = False,
    annotate_change_points: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6))
    group_line_data = {}
    trend_line_points = []
    for group in _sort_groups(list(group_series.keys())):
        s = group_series[group]
        x_labels = [str(i) for i in s.index.tolist()]
        x_pos = np.arange(len(x_labels))
        y = s.values.astype(float)
        line, = ax.plot(x_pos, y, marker="o", linewidth=2, label=group)
        group_line_data[group] = {
            "x_labels": x_labels,
            "x_pos": x_pos,
            "y": y,
            "color": line.get_color(),
        }
    if annotate_sen_slope and annual_mk_df is not None:
        for group in _sort_groups(list(group_line_data.keys())):
            row = annual_mk_df[annual_mk_df["城市群"] == group]
            if row.empty:
                continue
            slope = pd.to_numeric(row.iloc[0]["Sen_Slope_μg_m3_per_year"], errors="coerce")
            if pd.isna(slope):
                continue
            line_data = group_line_data[group]
            x_pos = line_data["x_pos"]
            y_vals = line_data["y"]
            if len(x_pos) < 2:
                continue
            trend_y = y_vals[0] + float(slope) * (x_pos - x_pos[0])
            ax.plot(
                x_pos,
                trend_y,
                linestyle=(0, (4, 3)),
                linewidth=1.4,
                color=line_data["color"],
                alpha=0.7,
                label=f"{group}趋势线",
            )
            trend_line_points.append(np.column_stack((x_pos, trend_y)))

        all_line_points_for_sen = np.concatenate(
            [
                np.column_stack((line_data["x_pos"], line_data["y"]))
                for line_data in group_line_data.values()
            ] + trend_line_points,
            axis=0,
        ) if group_line_data else np.empty((0, 2), dtype=float)
        for group in _sort_groups(list(group_line_data.keys())):
            row = annual_mk_df[annual_mk_df["城市群"] == group]
            if row.empty:
                continue
            slope = pd.to_numeric(row.iloc[0]["Sen_Slope_μg_m3_per_year"], errors="coerce")
            if pd.isna(slope):
                continue
            line_data = group_line_data[group]
            x_pos = line_data["x_pos"]
            y_vals = line_data["y"]
            if len(x_pos) < 1:
                continue
            trend_end_y = y_vals[0] + float(slope) * (x_pos[-1] - x_pos[0])
            text_dx, text_dy = _pick_annotation_offset(
                ax,
                float(x_pos[-1]),
                float(trend_end_y),
                all_line_points_for_sen,
            )
            ax.annotate(
                f"Sen斜率: {slope:.2f}",
                xy=(x_pos[-1], trend_end_y),
                xytext=(text_dx, text_dy),
                textcoords="offset points",
                color=line_data["color"],
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=line_data["color"], alpha=0.55),
            )

    if annotate_change_points and sliding_t_df is not None and not sliding_t_df.empty:
        all_line_points = np.concatenate(
            [
                np.column_stack((line_data["x_pos"], line_data["y"]))
                for line_data in group_line_data.values()
            ],
            axis=0,
        ) if group_line_data else np.empty((0, 2), dtype=float)
        used_group_point_labels = set()
        for _, row in sliding_t_df.iterrows():
            if not bool(row.get("Significant", False)):
                continue
            group = str(row.get("城市群", ""))
            if group not in group_line_data:
                continue
            change_label = _extract_change_label(row)
            if not change_label:
                continue
            x_labels = group_line_data[group]["x_labels"]
            if change_label not in x_labels:
                continue
            t_stat = pd.to_numeric(row.get("T_Statistic"), errors="coerce")
            if pd.isna(t_stat):
                continue
            change_idx = x_labels.index(change_label)
            x_pos = group_line_data[group]["x_pos"][change_idx]
            y_val = group_line_data[group]["y"][change_idx]
            point_label = f"{group}突变点"
            label = point_label if point_label not in used_group_point_labels else None
            used_group_point_labels.add(point_label)

            ax.scatter(
                x_pos,
                y_val,
                marker="*",
                s=120,
                color=group_line_data[group]["color"],
                edgecolors="black",
                linewidths=0.8,
                zorder=7,
                label=label,
            )
            text_dx, text_dy = _pick_annotation_offset(ax, float(x_pos), float(y_val), all_line_points)
            ax.annotate(
                f"{change_label}\nT={t_stat:.2f}",
                xy=(x_pos, y_val),
                xytext=(text_dx, text_dy),
                textcoords="offset points",
                fontsize=8,
                color=group_line_data[group]["color"],
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    fc="white",
                    ec=group_line_data[group]["color"],
                    alpha=0.55,
                ),
            )

    if group_line_data:
        ref_group = _sort_groups(list(group_line_data.keys()))[0]
        ref_x = group_line_data[ref_group]["x_pos"]
        ref_labels = group_line_data[ref_group]["x_labels"]
        tick_pos, tick_labels = _choose_sparse_ticks(ref_x, ref_labels, max_ticks=18)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("PM2.5浓度 (μg/m³)", fontsize=12)
    ax.grid(alpha=0.3, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), fontsize=10, loc="best")
    plt.tight_layout()
    save_figure_dual(fig, output_png_path, dpi=300)
    plt.close(fig)
    safe_print(f"已保存图像: {output_png_path} 及同名 SVG")


def plot_test_comparison_2x2(
    annual_results: Dict[str, pd.DataFrame],
    monthly_results: Dict[str, pd.DataFrame],
    output_png_path: str,
) -> None:
    groups = sorted(
        annual_results["annual_mann_kendall"]["城市群"].astype(str).unique().tolist(),
        key=lambda g: ["京津冀", "长三角", "珠三角"].index(g) if g in ["京津冀", "长三角", "珠三角"] else 99,
    )
    x = np.arange(len(groups))
    width = 0.36
    alpha_line = 0.05

    annual_mk = annual_results["annual_mann_kendall"].set_index("城市群")
    annual_lr = annual_results["annual_linear_regression"].set_index("城市群")
    annual_pt = annual_results["annual_pettitt"].set_index("城市群")
    annual_st = annual_results["annual_sliding_t"].set_index("城市群")
    monthly_pt = monthly_results["month_pettitt"].set_index("城市群")
    monthly_st = monthly_results["month_sliding_t"].set_index("城市群")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("三大城市群PM2.5浓度统计检验结果对比", fontsize=16, y=0.98)

    ax1 = axes[0, 0]
    mk_vals = np.array([pd.to_numeric(annual_mk.loc[g, "P_Value"], errors="coerce") for g in groups], dtype=float)
    lr_vals = np.array([pd.to_numeric(annual_lr.loc[g, "P_Value"], errors="coerce") for g in groups], dtype=float)
    bars1 = ax1.bar(x - width / 2, mk_vals, width, label="Mann-Kendall检验", color="#e76f51", alpha=0.82)
    bars2 = ax1.bar(x + width / 2, lr_vals, width, label="线性回归检验", color="#f2a8a0", alpha=0.82)
    ax1.axhline(alpha_line, color="#d62828", linestyle="--", linewidth=1.5, label="α=0.05")
    ax1.set_title("年度趋势检验P值对比", fontsize=12)
    ax1.set_ylabel("P值")
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.grid(axis="y", alpha=0.25, linestyle="--")
    _annotate_bar_values(ax1, list(bars1) + list(bars2), decimals=3)
    ax1.legend(fontsize=9, loc="upper right")

    ax2 = axes[0, 1]
    apt_vals = np.array([pd.to_numeric(annual_pt.loc[g, "P_Value"], errors="coerce") for g in groups], dtype=float)
    ast_vals = np.array([pd.to_numeric(annual_st.loc[g, "P_Value"], errors="coerce") for g in groups], dtype=float)
    bars3 = ax2.bar(x - width / 2, apt_vals, width, label="Pettitt检验", color="#4ea8de", alpha=0.82)
    bars4 = ax2.bar(x + width / 2, ast_vals, width, label="滑动T检验", color="#9ecae1", alpha=0.82)
    ax2.axhline(alpha_line, color="#d62828", linestyle="--", linewidth=1.5, label="α=0.05")
    ax2.set_title("年度突变检验P值对比", fontsize=12)
    ax2.set_ylabel("P值")
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups)
    ax2.grid(axis="y", alpha=0.25, linestyle="--")
    _annotate_bar_values(ax2, list(bars3) + list(bars4), decimals=3)
    ax2.legend(fontsize=9, loc="upper right")

    ax3 = axes[1, 0]
    mpt_vals = np.array([pd.to_numeric(monthly_pt.loc[g, "P_Value"], errors="coerce") for g in groups], dtype=float)
    mst_vals = np.array([pd.to_numeric(monthly_st.loc[g, "P_Value"], errors="coerce") for g in groups], dtype=float)
    bars5 = ax3.bar(x - width / 2, mpt_vals, width, label="Pettitt检验", color="#52b788", alpha=0.85)
    bars6 = ax3.bar(x + width / 2, mst_vals, width, label="滑动T检验", color="#95d5b2", alpha=0.85)
    ax3.axhline(alpha_line, color="#d62828", linestyle="--", linewidth=1.5, label="α=0.05")
    ax3.set_title("月度突变检验P值对比", fontsize=12)
    ax3.set_ylabel("P值")
    ax3.set_xticks(x)
    ax3.set_xticklabels(groups)
    ax3.grid(axis="y", alpha=0.25, linestyle="--")
    _annotate_bar_values(ax3, list(bars5) + list(bars6), decimals=4)
    ax3.legend(fontsize=9, loc="center right", bbox_to_anchor=(0.98, 0.6))

    ax4 = axes[1, 1]
    sen_vals = np.array(
        [abs(pd.to_numeric(annual_mk.loc[g, "Sen_Slope_μg_m3_per_year"], errors="coerce")) for g in groups],
        dtype=float,
    )
    lr_slope_vals = np.array(
        [abs(pd.to_numeric(annual_lr.loc[g, "Slope_μg_m3_per_year"], errors="coerce")) for g in groups],
        dtype=float,
    )
    bars7 = ax4.bar(x - width / 2, sen_vals, width, label="Sen斜率", color="#f77f00", alpha=0.8)
    bars8 = ax4.bar(x + width / 2, lr_slope_vals, width, label="线性回归斜率", color="#ffb703", alpha=0.8)
    ax4.set_title("年度PM2.5减少速率对比", fontsize=12)
    ax4.set_ylabel("PM2.5减少速率 (μg/m³/年)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(groups)
    ax4.grid(axis="y", alpha=0.25, linestyle="--")
    _annotate_bar_values(ax4, list(bars7) + list(bars8), decimals=2)
    ax4.legend(fontsize=9, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_dual(fig, output_png_path, dpi=300)
    plt.close(fig)
    safe_print(f"已保存图像: {output_png_path} 及同名 SVG")


# =========================
# 数据加载（从已保存的 CSV）
# =========================

def load_series_from_csv(csv_path: str, index_col: str) -> Dict[str, pd.Series]:
    """从聚合序列 CSV 重建 {城市群: pd.Series} 结构。"""
    df = read_csv_flexible(csv_path)
    series_map: Dict[str, pd.Series] = {}
    for group, sub in df.groupby("城市群"):
        s = sub.set_index(index_col)["PM2.5"].sort_index()
        series_map[group] = s
    return series_map


def _sliding_t_change_point_for_plot(
    data: np.ndarray,
    min_window: int = 3,
    min_each_side: int = 3,
) -> Dict[str, object]:
    """用于绘图标注的滑动T突变点：返回 |t| 最大位置（不以显著性作为是否返回的条件）。"""
    n = len(data)
    if n < max(2 * min_each_side, 6):
        return {"change_point": None, "t_statistic": np.nan, "p_value": np.nan, "significant": False}
    window_size = max(min_window, n // 3)
    window_size = min(window_size, n - min_each_side)
    t_stats, p_values, positions = [], [], []
    for i in range(window_size, n - window_size + 1):
        before, after = data[i - window_size:i], data[i:i + window_size]
        if len(before) < 2 or len(after) < 2:
            continue
        try:
            t_stat, p_val = stats.ttest_ind(before, after, equal_var=False)
            if not (np.isnan(t_stat) or np.isnan(p_val)):
                t_stats.append(abs(float(t_stat)))
                p_values.append(float(p_val))
                positions.append(i)
        except Exception:
            continue
    if not t_stats:
        return {"change_point": None, "t_statistic": np.nan, "p_value": np.nan, "significant": False}
    max_idx = int(np.argmax(t_stats))
    return {
        "change_point": positions[max_idx],
        "t_statistic": t_stats[max_idx],
        "p_value": p_values[max_idx],
        "significant": bool(p_values[max_idx] < 0.05),
    }


def build_change_points_df_for_plot(
    group_series: Dict[str, pd.Series],
    time_name: str,
    min_window: int,
    min_each_side: int,
) -> pd.DataFrame:
    """为绘图构建突变点表，包含 Change_* 与 T_Statistic。"""
    rows = []
    for group, s in group_series.items():
        s = s.dropna().sort_index()
        times = s.index.tolist()
        y = s.values.astype(float)
        cp = _sliding_t_change_point_for_plot(y, min_window=min_window, min_each_side=min_each_side)
        cp_idx = cp["change_point"]
        change_label = times[cp_idx] if cp_idx is not None and 0 <= cp_idx < len(times) else None
        rows.append(
            {
                "城市群": group,
                f"Change_{time_name}": change_label,
                "T_Statistic": cp["t_statistic"],
                "P_Value": cp["p_value"],
                "Significant": cp["significant"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    configure_chinese_font()

    safe_print("加载聚合序列...")
    annual_group_series = load_series_from_csv(
        os.path.join(OUTPUT_DIR, "城市群_年度PM2.5聚合序列.csv"), "Year",
    )
    monthly_group_series = load_series_from_csv(
        os.path.join(OUTPUT_DIR, "城市群_月度PM2.5聚合序列.csv"), "Month",
    )
    seasonal_group_series = load_series_from_csv(
        os.path.join(OUTPUT_DIR, "城市群_季度PM2.5聚合序列.csv"), "Season",
    )

    safe_print("加载检验结果...")
    annual_results = {
        "annual_mann_kendall": read_csv_flexible(os.path.join(OUTPUT_DIR, "年度_Mann_Kendall_城市群.csv")),
        "annual_pettitt": read_csv_flexible(os.path.join(OUTPUT_DIR, "年度_Pettitt_城市群.csv")),
        "annual_sliding_t": read_csv_flexible(os.path.join(OUTPUT_DIR, "年度_滑动T检验_城市群.csv")),
        "annual_linear_regression": read_csv_flexible(os.path.join(OUTPUT_DIR, "年度_线性回归_城市群.csv")),
    }
    monthly_results = {
        "month_pettitt": read_csv_flexible(os.path.join(OUTPUT_DIR, "月度_Pettitt_城市群.csv")),
        "month_sliding_t": read_csv_flexible(os.path.join(OUTPUT_DIR, "月度_滑动T检验_城市群.csv")),
    }
    seasonal_results = {
        "season_pettitt": read_csv_flexible(os.path.join(OUTPUT_DIR, "季度_Pettitt_城市群.csv")),
        "season_sliding_t": read_csv_flexible(os.path.join(OUTPUT_DIR, "季度_滑动T检验_城市群.csv")),
    }
    seasonal_plot_change_df = build_change_points_df_for_plot(
        seasonal_group_series,
        "Season",
        min_window=4,
        min_each_side=4,
    )

    safe_print("开始绘图...")
    plot_group_series(
        annual_group_series,
        "三大城市群年度PM2.5浓度变化",
        "年份",
        os.path.join(OUTPUT_DIR, "三大城市群_年度PM2.5时序.png"),
        annual_mk_df=annual_results["annual_mann_kendall"],
        annotate_sen_slope=True,
    )
    plot_group_series(
        monthly_group_series,
        "三大城市群月均PM2.5浓度变化（2018-2023）",
        "月份",
        os.path.join(OUTPUT_DIR, "三大城市群_月度PM2.5时序.png"),
    )
    plot_group_series(
        seasonal_group_series,
        "三大城市群季度PM2.5浓度变化（由月均聚合）",
        "季度",
        os.path.join(OUTPUT_DIR, "三大城市群_季度PM2.5时序.png"),
        pettitt_df=seasonal_results["season_pettitt"],
        sliding_t_df=seasonal_plot_change_df,
        annotate_change_points=True,
    )
    plot_test_comparison_2x2(
        annual_results,
        monthly_results,
        os.path.join(OUTPUT_DIR, "三大城市群_统计检验结果对比.png"),
    )

    safe_print("\n全部绘图完成。")
    safe_print("输出目录:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
