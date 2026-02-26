"""
三大城市群统计趋势检验（按城市群聚合）
- 年度：Mann-Kendall、Pettitt、滑动T、线性回归
- 月度：Pettitt、滑动T
- 季度：Pettitt、滑动T（由月均聚合）
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

# =========================
# 路径配置（按需修改）
# =========================
ANNUAL_DATA_PATH = r"H:\DATA Science\大论文Result\三大城市群（市）年度PM2.5浓度.csv"
MONTHLY_DATA_PATH = r"H:\DATA Science\大论文Result\三大城市群（市）月均PM2.5浓度\合并数据_2018-2023.csv"
CITY_GROUP_PATH = r"H:\DATA Science\大论文Result\大论文图\三大城市群\MSTL时间序列分解\城市归属_三大城市群.csv"
OUTPUT_DIR = r"H:\DATA Science\大论文Result\大论文图\三大城市群\统计趋势检验"

ALPHA = 0.05
ZERO_AS_MISSING = True

# 城市群名称别名：年度数据可能已是城市群级，首列为城市群名，需映射到与归属表一致
GROUP_NAME_ALIASES = {
    "京津冀城市群": "京津冀", "京津冀": "京津冀", "京津冀城市群(BTH)": "京津冀",
    "长江三角洲城市群": "长三角", "长三角": "长三角", "长江三角洲": "长三角",
    "长江三角洲城市群(YRD)": "长三角",
    "珠江三角洲城市群": "珠三角", "珠三角": "珠三角", "珠江三角洲": "珠三角",
    "珠江三角洲城市群(PRD)": "珠三角",
}


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


def choose_column(df: pd.DataFrame, candidates: List[str], desc: str) -> str:
    """在候选列名中选择存在的一列。若首列为 Unnamed: 0 且无标准列，则视为城市/索引列。"""
    for col in candidates:
        if col in df.columns:
            return col
    if "Unnamed: 0" in df.columns and desc == "城市":
        return "Unnamed: 0"
    raise ValueError(f"无法在数据中识别{desc}列，候选: {candidates}，实际列: {list(df.columns)}")


def extract_columns_by_regex(columns: List[str], pattern: str) -> List[str]:
    """按正则筛选列名。"""
    import re
    return [col for col in columns if re.match(pattern, col)]


def load_city_group_mapping(mapping_path: str) -> Dict[str, str]:
    """加载城市到城市群映射。"""
    df = read_csv_flexible(mapping_path)
    city_col = choose_column(df, ["城市", "city", "City"], "城市")
    group_col = choose_column(df, ["城市群", "group", "Group", "RegionGroup"], "城市群")
    mapping = (
        df[[city_col, group_col]]
        .dropna()
        .assign(**{city_col: lambda x: x[city_col].astype(str).str.strip()})
        .set_index(city_col)[group_col]
        .to_dict()
    )
    return mapping


def load_annual_city_data(annual_path: str) -> pd.DataFrame:
    """读取年度城市 PM2.5 数据并转换成长表。"""
    df = read_csv_flexible(annual_path)
    city_col = choose_column(df, ["城市", "city", "City"], "城市")
    year_cols = extract_columns_by_regex(list(df.columns), r"^\d{4}$")
    if not year_cols:
        raise ValueError("年度数据中未找到年份列（如 2018、2019）")
    long_df = df[[city_col] + year_cols].melt(
        id_vars=[city_col],
        var_name="year",
        value_name="pm25",
    )
    long_df = long_df.rename(columns={city_col: "城市"})
    long_df["城市"] = long_df["城市"].astype(str).str.strip()
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df["pm25"] = pd.to_numeric(long_df["pm25"], errors="coerce")
    long_df = long_df.dropna(subset=["year", "pm25"])
    long_df["year"] = long_df["year"].astype(int)
    if ZERO_AS_MISSING:
        long_df.loc[long_df["pm25"] <= 0, "pm25"] = np.nan
        long_df = long_df.dropna(subset=["pm25"])
    return long_df


def load_monthly_city_data(monthly_path: str) -> pd.DataFrame:
    """读取月均城市 PM2.5 数据并转换成长表。"""
    df = read_csv_flexible(monthly_path)
    city_col = choose_column(df, ["城市", "city", "City"], "城市")
    month_cols = extract_columns_by_regex(list(df.columns), r"^\d{6}$")
    if not month_cols:
        month_cols = extract_columns_by_regex(list(df.columns), r"^\d{4}-\d{2}$")
    if not month_cols:
        raise ValueError("月度数据中未找到月份列（如 201801 或 2018-01）")
    long_df = df[[city_col] + month_cols].melt(
        id_vars=[city_col],
        var_name="month_key",
        value_name="pm25",
    )
    long_df = long_df.rename(columns={city_col: "城市"})
    long_df["城市"] = long_df["城市"].astype(str).str.strip()
    long_df["pm25"] = pd.to_numeric(long_df["pm25"], errors="coerce")
    mk = long_df["month_key"].astype(str)
    as_ym = pd.to_datetime(mk, format="%Y%m", errors="coerce")
    as_ym_dash = pd.to_datetime(mk, format="%Y-%m", errors="coerce")
    long_df["month"] = as_ym.fillna(as_ym_dash)
    long_df = long_df.dropna(subset=["month", "pm25"])
    if ZERO_AS_MISSING:
        long_df.loc[long_df["pm25"] <= 0, "pm25"] = np.nan
        long_df = long_df.dropna(subset=["pm25"])
    long_df["month"] = long_df["month"].dt.to_period("M").dt.to_timestamp()
    return long_df[["城市", "month", "pm25"]]


def attach_group_and_aggregate(
    long_df: pd.DataFrame,
    city_group_map: Dict[str, str],
    time_col: str,
) -> pd.DataFrame:
    """将城市映射到城市群，并按城市群+时间取均值。支持城市级与城市群级数据。"""
    df = long_df.copy()
    df["城市群"] = df["城市"].map(city_group_map)
    # 若直接映射失败，尝试城市群名称别名（年度数据可能已是城市群级）
    still_missing = df["城市群"].isna()
    if still_missing.any():
        df.loc[still_missing, "城市群"] = df.loc[still_missing, "城市"].map(GROUP_NAME_ALIASES)
    missing_city = df[df["城市群"].isna()]["城市"].drop_duplicates().tolist()
    if missing_city:
        preview = missing_city[:10]
        safe_print(f"警告：有 {len(missing_city)} 个城市/城市群未匹配，已跳过。示例: {preview}")
    df = df.dropna(subset=["城市群"])
    agg = (
        df.groupby(["城市群", time_col], as_index=False)["pm25"]
        .mean()
        .sort_values(["城市群", time_col])
    )
    return agg


def month_to_season_label(ts: pd.Timestamp) -> str:
    """春3-5 夏6-8 秋9-11 冬12-2（1-2 归上一年冬季）"""
    year, month = ts.year, ts.month
    if month in [3, 4, 5]:
        return f"{year}春"
    if month in [6, 7, 8]:
        return f"{year}夏"
    if month in [9, 10, 11]:
        return f"{year}秋"
    if month == 12:
        return f"{year}冬"
    return f"{year - 1}冬"


def season_sort_key(season_label: str) -> Tuple[int, int]:
    order_map = {"春": 1, "夏": 2, "秋": 3, "冬": 4}
    year = int(season_label[:4])
    season = season_label[4:]
    return year, order_map.get(season, 99)


def build_group_series_map(
    agg_df: pd.DataFrame,
    time_col: str,
) -> Dict[str, pd.Series]:
    """将聚合表转为 {城市群: 时间序列} 结构。"""
    series_map: Dict[str, pd.Series] = {}
    for group, sub in agg_df.groupby("城市群"):
        s = sub.set_index(time_col)["pm25"].sort_index()
        series_map[group] = s
    return series_map


def calculate_sen_slope(data: np.ndarray) -> float:
    n = len(data)
    if n < 2:
        return np.nan
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes.append((data[j] - data[i]) / (j - i))
    return float(np.median(slopes)) if slopes else np.nan


def mann_kendall_test(data: np.ndarray, alpha: float = ALPHA) -> Dict[str, object]:
    n = len(data)
    if n < 3:
        return {
            "S": np.nan, "var_S": np.nan, "Z": np.nan, "p_value": np.nan,
            "trend": "数据不足", "significant": False, "sen_slope": np.nan,
        }
    s_stat = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s_stat += np.sign(data[j] - data[i])
    unique, counts = np.unique(data, return_counts=True)
    if len(unique) == n:
        var_s = n * (n - 1) * (2 * n + 5) / 18
    else:
        tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
        var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18
    if s_stat > 0:
        z_stat = (s_stat - 1) / np.sqrt(var_s)
    elif s_stat < 0:
        z_stat = (s_stat + 1) / np.sqrt(var_s)
    else:
        z_stat = 0.0
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    significant = bool(p_value < alpha)
    trend = "增加" if (significant and z_stat > 0) else "减少" if (significant and z_stat < 0) else "无显著趋势"
    return {
        "S": float(s_stat), "var_S": float(var_s), "Z": float(z_stat),
        "p_value": float(p_value), "trend": trend, "significant": significant,
        "sen_slope": float(calculate_sen_slope(data)),
    }


def pettitt_test(data: np.ndarray, alpha: float = ALPHA) -> Dict[str, object]:
    n = len(data)
    if n < 4:
        return {
            "change_point": None, "K": np.nan, "p_value": np.nan,
            "significant": False, "trend_before": None, "trend_after": None,
        }
    u_values = np.zeros(n - 1)
    for k in range(1, n):
        u_sum = sum(np.sign(data[i] - data[j]) for i in range(k) for j in range(k, n))
        u_values[k - 1] = u_sum
    abs_u = np.abs(u_values)
    max_idx = int(np.argmax(abs_u))
    k_stat = float(abs_u[max_idx])
    change_point = max_idx + 1
    p_value = float(2 * np.exp(-6 * k_stat**2 / (n**3 + n**2))) if k_stat > 0 else 1.0
    significant = bool(p_value < alpha)
    trend_before = trend_after = None
    if significant and 1 < change_point < n - 1:
        before, after = data[:change_point], data[change_point:]
        if len(before) >= 2:
            slope_b = np.polyfit(range(len(before)), before, 1)[0]
            trend_before = "增加" if slope_b > 0 else "减少" if slope_b < 0 else "稳定"
        if len(after) >= 2:
            slope_a = np.polyfit(range(len(after)), after, 1)[0]
            trend_after = "增加" if slope_a > 0 else "减少" if slope_a < 0 else "稳定"
    return {
        "change_point": change_point if significant else None,
        "K": k_stat, "p_value": p_value, "significant": significant,
        "trend_before": trend_before, "trend_after": trend_after,
    }


def sliding_t_test(
    data: np.ndarray,
    window_size: Optional[int] = None,
    min_window: int = 3,
    min_each_side: int = 3,
    alpha: float = ALPHA,
) -> Dict[str, object]:
    n = len(data)
    if n < max(2 * min_each_side, 6):
        return {
            "change_point": None, "t_statistic": np.nan, "p_value": np.nan,
            "significant": False, "trend_before": None, "trend_after": None,
            "window_size": window_size,
        }
    if window_size is None:
        window_size = max(min_window, n // 3)
    window_size = min(window_size, n - min_each_side)
    t_stats, p_values, positions = [], [], []
    for i in range(window_size, n - window_size + 1):
        before, after = data[i - window_size : i], data[i : i + window_size]
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
        return {
            "change_point": None, "t_statistic": np.nan, "p_value": np.nan,
            "significant": False, "trend_before": None, "trend_after": None,
            "window_size": window_size,
        }
    max_idx = int(np.argmax(t_stats))
    change_point = positions[max_idx]
    t_stat_max, p_value = t_stats[max_idx], p_values[max_idx]
    significant = bool(p_value < alpha)
    trend_before = trend_after = None
    if significant and change_point > window_size and change_point < n - window_size:
        before = data[max(0, change_point - window_size) : change_point]
        after = data[change_point : min(n, change_point + window_size)]
        if len(before) >= 2:
            slope_b = np.polyfit(range(len(before)), before, 1)[0]
            trend_before = "增加" if slope_b > 0 else "减少" if slope_b < 0 else "稳定"
        if len(after) >= 2:
            slope_a = np.polyfit(range(len(after)), after, 1)[0]
            trend_after = "增加" if slope_a > 0 else "减少" if slope_a < 0 else "稳定"
    return {
        "change_point": change_point if significant else None,
        "t_statistic": t_stat_max, "p_value": p_value, "significant": significant,
        "trend_before": trend_before, "trend_after": trend_after,
        "window_size": window_size,
    }


def linear_regression_test(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    if len(x) < 3:
        return {
            "slope": np.nan, "intercept": np.nan, "r_squared": np.nan,
            "p_value": np.nan, "std_err": np.nan, "trend": "数据不足",
        }
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    trend = "增加" if slope > 0 else "减少" if slope < 0 else "无变化"
    return {
        "slope": float(slope), "intercept": float(intercept),
        "r_squared": float(r_value**2), "p_value": float(p_value),
        "std_err": float(std_err), "trend": trend,
    }


def add_series_values_to_row(row: Dict[str, object], index_values: List[object], series_values: np.ndarray) -> None:
    for idx, val in zip(index_values, series_values):
        row[f"PM2.5_{idx}"] = float(val)


def run_annual_analyses(annual_group_series: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
    mk_rows, pettitt_rows, st_rows, lr_rows = [], [], [], []
    for group, s in annual_group_series.items():
        s = s.dropna().sort_index()
        years = s.index.astype(int).tolist()
        y = s.values.astype(float)
        base = {"城市群": group, "N_Years": len(years)}
        mk = mann_kendall_test(y)
        mk_row = {**base, "S_Statistic": mk["S"], "Variance_S": mk["var_S"], "Z_Statistic": mk["Z"],
                  "P_Value": mk["p_value"], "Trend": mk["trend"],
                  "Sen_Slope_μg_m3_per_year": mk["sen_slope"], "Significant": mk["significant"]}
        mk_row["Percent_Change_Per_Year"] = float(mk["sen_slope"] / y[0] * 100) if len(y) > 0 and not np.isnan(mk["sen_slope"]) and y[0] != 0 else np.nan
        add_series_values_to_row(mk_row, years, y)
        mk_rows.append(mk_row)
        pt = pettitt_test(y)
        change_year = years[pt["change_point"] - 1] if pt["change_point"] is not None else None
        pt_row = {**base, "Change_Point_Index": pt["change_point"], "Change_Year": change_year,
                  "K_Statistic": pt["K"], "P_Value": pt["p_value"], "Significant": pt["significant"],
                  "Trend_Before": pt["trend_before"], "Trend_After": pt["trend_after"]}
        add_series_values_to_row(pt_row, years, y)
        pettitt_rows.append(pt_row)
        st = sliding_t_test(y, window_size=None, min_window=3, min_each_side=3)
        change_year_st = years[st["change_point"]] if st["change_point"] is not None else None
        st_row = {**base, "Change_Point_Index": st["change_point"], "Change_Year": change_year_st,
                  "T_Statistic": st["t_statistic"], "P_Value": st["p_value"], "Significant": st["significant"],
                  "Trend_Before": st["trend_before"], "Trend_After": st["trend_after"],
                  "Window_Size": st["window_size"]}
        add_series_values_to_row(st_row, years, y)
        st_rows.append(st_row)
        lr = linear_regression_test(np.array(years, dtype=float), y)
        lr_row = {**base, "Slope_μg_m3_per_year": lr["slope"], "Intercept": lr["intercept"],
                  "R_Squared": lr["r_squared"], "P_Value": lr["p_value"], "Std_Error": lr["std_err"],
                  "Trend": lr["trend"]}
        lr_row["Percent_Change_Per_Year"] = float(lr["slope"] / y[0] * 100) if len(y) > 0 and not np.isnan(lr["slope"]) and y[0] != 0 else np.nan
        add_series_values_to_row(lr_row, years, y)
        lr_rows.append(lr_row)
    return {
        "annual_mann_kendall": pd.DataFrame(mk_rows),
        "annual_pettitt": pd.DataFrame(pettitt_rows),
        "annual_sliding_t": pd.DataFrame(st_rows),
        "annual_linear_regression": pd.DataFrame(lr_rows),
    }


def run_change_point_analyses(
    group_series: Dict[str, pd.Series],
    time_name: str,
    min_window: int,
    min_each_side: int,
) -> Dict[str, pd.DataFrame]:
    pettitt_rows, st_rows = [], []
    for group, s in group_series.items():
        s = s.dropna().sort_index()
        times = s.index.tolist()
        y = s.values.astype(float)
        base = {"城市群": group, f"N_{time_name}": len(times)}
        pt = pettitt_test(y)
        change_label = times[pt["change_point"] - 1] if pt["change_point"] is not None else None
        pt_row = {**base, "Change_Point_Index": pt["change_point"], f"Change_{time_name}": change_label,
                  "K_Statistic": pt["K"], "P_Value": pt["p_value"], "Significant": pt["significant"],
                  "Trend_Before": pt["trend_before"], "Trend_After": pt["trend_after"]}
        add_series_values_to_row(pt_row, [str(x) for x in times], y)
        pettitt_rows.append(pt_row)
        st = sliding_t_test(y, window_size=None, min_window=min_window, min_each_side=min_each_side)
        change_label_st = times[st["change_point"]] if st["change_point"] is not None else None
        st_row = {**base, "Change_Point_Index": st["change_point"], f"Change_{time_name}": change_label_st,
                  "T_Statistic": st["t_statistic"], "P_Value": st["p_value"], "Significant": st["significant"],
                  "Trend_Before": st["trend_before"], "Trend_After": st["trend_after"],
                  "Window_Size": st["window_size"]}
        add_series_values_to_row(st_row, [str(x) for x in times], y)
        st_rows.append(st_row)
    return {
        f"{time_name.lower()}_pettitt": pd.DataFrame(pettitt_rows),
        f"{time_name.lower()}_sliding_t": pd.DataFrame(st_rows),
    }


def save_df(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    safe_print(f"已保存: {output_path}")


def save_group_series_table(group_series: Dict[str, pd.Series], output_path: str, index_name: str) -> None:
    rows = []
    for group, s in group_series.items():
        for idx, val in s.items():
            rows.append({"城市群": group, index_name: idx, "PM2.5": float(val)})
    save_df(pd.DataFrame(rows), output_path)


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
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (0.005 if height <= 0.1 else 0.02),
            f"{height:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


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

        if annotate_sen_slope and annual_mk_df is not None and len(x_pos) > 0:
            row = annual_mk_df[annual_mk_df["城市群"] == group]
            if not row.empty:
                slope = pd.to_numeric(row.iloc[0]["Sen_Slope_μg_m3_per_year"], errors="coerce")
                if pd.notna(slope):
                    # 在右侧末端标注 Sen 斜率，便于与对应曲线直接关联。
                    ax.annotate(
                        f"Sen斜率: {slope:.2f}",
                        xy=(x_pos[-1], y[-1]),
                        xytext=(12, 0),
                        textcoords="offset points",
                        color=line.get_color(),
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=line.get_color(), alpha=0.55),
                    )

    if annotate_change_points and sliding_t_df is not None and not sliding_t_df.empty:
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
            # 标注样式按示例图：显示时间与 T 值。
            ax.annotate(
                f"{change_label}\nT={t_stat:.2f}",
                xy=(x_pos, y_val),
                xytext=(6, 8),
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
        ax.set_xticks(ref_x)
        ax.set_xticklabels(ref_labels)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("PM2.5浓度 (μg/m³)", fontsize=12)
    ax.grid(alpha=0.3, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), fontsize=10, loc="best")
    if annotate_sen_slope:
        fig.subplots_adjust(right=0.84)
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

    # 子图1：年度趋势检验 P 值
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

    # 子图2：年度突变检验 P 值
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

    # 子图3：月度突变检验 P 值
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
    ax3.legend(fontsize=9, loc="upper right")

    # 子图4：年度 PM2.5 减少速率（绝对值）
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


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    configure_chinese_font()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_print("加载城市归属映射...")
    city_group_map = load_city_group_mapping(CITY_GROUP_PATH)
    safe_print(f"城市归属映射条目数: {len(city_group_map)}")
    safe_print("加载年度数据...")
    annual_city_long = load_annual_city_data(ANNUAL_DATA_PATH)
    annual_group_agg = attach_group_and_aggregate(annual_city_long, city_group_map, "year")
    annual_group_series = build_group_series_map(annual_group_agg, "year")
    safe_print("加载月度数据...")
    monthly_city_long = load_monthly_city_data(MONTHLY_DATA_PATH)
    monthly_group_agg = attach_group_and_aggregate(monthly_city_long, city_group_map, "month")
    monthly_group_agg["month_str"] = monthly_group_agg["month"].dt.strftime("%Y-%m")
    monthly_group_series = {}
    for g, sub in monthly_group_agg.groupby("城市群"):
        s = sub.set_index("month_str")["pm25"].sort_index()
        monthly_group_series[g] = s
    seasonal_group_agg = monthly_group_agg.copy()
    seasonal_group_agg["season"] = seasonal_group_agg["month"].apply(month_to_season_label)
    seasonal_group_agg = (
        seasonal_group_agg.groupby(["城市群", "season"], as_index=False)["pm25"]
        .mean()
        .sort_values(["城市群", "season"], key=lambda col: col.map(season_sort_key) if col.name == "season" else col)
    )
    seasonal_group_series = {}
    for g, sub in seasonal_group_agg.groupby("城市群"):
        s = sub.set_index("season")["pm25"]
        s = s.reindex(sorted(s.index.tolist(), key=season_sort_key))
        seasonal_group_series[g] = s
    safe_print("执行年度检验（Mann-Kendall / Pettitt / 滑动T / 线性回归）...")
    annual_results = run_annual_analyses(annual_group_series)
    safe_print("执行月度检验（Pettitt / 滑动T）...")
    monthly_results = run_change_point_analyses(monthly_group_series, "Month", min_window=6, min_each_side=6)
    safe_print("执行季度检验（Pettitt / 滑动T）...")
    seasonal_results = run_change_point_analyses(seasonal_group_series, "Season", 4, 4)
    save_group_series_table(annual_group_series, os.path.join(OUTPUT_DIR, "城市群_年度PM2.5聚合序列.csv"), "Year")
    save_group_series_table(monthly_group_series, os.path.join(OUTPUT_DIR, "城市群_月度PM2.5聚合序列.csv"), "Month")
    save_group_series_table(seasonal_group_series, os.path.join(OUTPUT_DIR, "城市群_季度PM2.5聚合序列.csv"), "Season")
    save_df(annual_results["annual_mann_kendall"], os.path.join(OUTPUT_DIR, "年度_Mann_Kendall_城市群.csv"))
    save_df(annual_results["annual_pettitt"], os.path.join(OUTPUT_DIR, "年度_Pettitt_城市群.csv"))
    save_df(annual_results["annual_sliding_t"], os.path.join(OUTPUT_DIR, "年度_滑动T检验_城市群.csv"))
    save_df(annual_results["annual_linear_regression"], os.path.join(OUTPUT_DIR, "年度_线性回归_城市群.csv"))
    save_df(monthly_results["month_pettitt"], os.path.join(OUTPUT_DIR, "月度_Pettitt_城市群.csv"))
    save_df(monthly_results["month_sliding_t"], os.path.join(OUTPUT_DIR, "月度_滑动T检验_城市群.csv"))
    save_df(seasonal_results["season_pettitt"], os.path.join(OUTPUT_DIR, "季度_Pettitt_城市群.csv"))
    save_df(seasonal_results["season_sliding_t"], os.path.join(OUTPUT_DIR, "季度_滑动T检验_城市群.csv"))
    plot_group_series(
        annual_group_series,
        "三大城市群年度PM2.5浓度变化",
        "年份",
        os.path.join(OUTPUT_DIR, "三大城市群_年度PM2.5时序.png"),
        annual_mk_df=annual_results["annual_mann_kendall"],
        annotate_sen_slope=True,
    )
    plot_group_series(monthly_group_series, "三大城市群月均PM2.5浓度变化（2018-2023）", "月份",
                     os.path.join(OUTPUT_DIR, "三大城市群_月度PM2.5时序.png"))
    plot_group_series(
        seasonal_group_series,
        "三大城市群季度PM2.5浓度变化（由月均聚合）",
        "季度",
        os.path.join(OUTPUT_DIR, "三大城市群_季度PM2.5时序.png"),
        pettitt_df=seasonal_results["season_pettitt"],
        sliding_t_df=seasonal_results["season_sliding_t"],
        annotate_change_points=True,
    )
    plot_test_comparison_2x2(
        annual_results,
        monthly_results,
        os.path.join(OUTPUT_DIR, "三大城市群_统计检验结果对比.png"),
    )
    safe_print("\n全部完成。")
    safe_print("输出目录:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
