import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def _safe_print(*args, **kwargs):
    """兼容终端编码差异的安全输出。"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        msg = " ".join(str(x) for x in args)
        sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))


def _configure_chinese_font():
    """按排查指南配置中文字体，避免方框。"""
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


def _save_figure_dual(fig, save_path_png: str, dpi: int = 300) -> None:
    """同时保存 PNG 与 SVG（论文推荐）。"""
    fig.savefig(save_path_png, dpi=dpi, bbox_inches="tight")
    save_path_svg = os.path.splitext(save_path_png)[0] + ".svg"
    fig.savefig(save_path_svg, format="svg", bbox_inches="tight")

# 城市群名称别名
GROUP_NAME_ALIASES = {
    "京津冀城市群": "京津冀", "京津冀": "京津冀", "京津冀城市群(BTH)": "京津冀",
    "长江三角洲城市群": "长三角", "长三角": "长三角", "长江三角洲": "长三角",
    "长江三角洲城市群(YRD)": "长三角",
    "珠江三角洲城市群": "珠三角", "珠三角": "珠三角", "珠江三角洲": "珠三角",
    "珠江三角洲城市群(PRD)": "珠三角",
}


def _read_csv_flexible(path):
    """兼容多种编码读取 CSV。"""
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"无法读取文件: {path}")


def _load_city_group_mapping(mapping_path):
    """加载城市到城市群映射。"""
    df = _read_csv_flexible(mapping_path)
    city_col = "城市" if "城市" in df.columns else df.columns[0]
    group_col = "城市群" if "城市群" in df.columns else df.columns[1]
    mapping = df.set_index(city_col)[group_col].to_dict()
    return mapping


def load_and_preprocess_data(file_path, city_group_path=None):
    """
    加载并预处理PM2.5月度数据。
    支持两种格式：
    1) 已聚合格式：month, 京津冀, 长三角, 珠三角
    2) 城市级宽表：城市 + 201801/2018-01 等月份列（需提供 city_group_path）
    """
    df = _read_csv_flexible(file_path)

    # 格式1：已有 month 和 京津冀/长三角/珠三角 列
    if "month" in df.columns and all(g in df.columns for g in ["京津冀", "长三角", "珠三角"]):
        df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
        if df.isnull().sum().any():
            df = df.ffill()
        return df.sort_values("month").reset_index(drop=True)

    # 格式2：城市级宽表，需聚合
    city_col = "城市" if "城市" in df.columns else df.columns[0]
    month_cols = [c for c in df.columns if re.match(r"^\d{6}$", str(c))]
    if not month_cols:
        month_cols = [c for c in df.columns if re.match(r"^\d{4}-\d{2}$", str(c))]
    if not month_cols:
        raise ValueError(
            f"数据中未找到月份列（如 201801 或 2018-01）。实际列: {list(df.columns)}"
        )

    if city_group_path is None:
        raise ValueError(
            "城市级数据需要 city_group_path 进行城市群聚合。"
            "请传入城市归属文件路径。"
        )

    city_group_map = _load_city_group_mapping(city_group_path)
    long_df = df[[city_col] + month_cols].melt(
        id_vars=[city_col], var_name="month_key", value_name="pm25"
    )
    long_df[city_col] = long_df[city_col].astype(str).str.strip()
    long_df["pm25"] = pd.to_numeric(long_df["pm25"], errors="coerce")
    mk = long_df["month_key"].astype(str)
    as_ym = pd.to_datetime(mk, format="%Y%m", errors="coerce")
    as_ym_dash = pd.to_datetime(mk, format="%Y-%m", errors="coerce")
    long_df["month"] = as_ym.fillna(as_ym_dash)
    long_df = long_df.dropna(subset=["month", "pm25"])

    long_df["城市群"] = long_df[city_col].map(city_group_map)
    still_missing = long_df["城市群"].isna()
    if still_missing.any():
        long_df.loc[still_missing, "城市群"] = long_df.loc[still_missing, city_col].map(
            GROUP_NAME_ALIASES
        )
    long_df = long_df.dropna(subset=["城市群"])

    agg = (
        long_df.groupby(["城市群", "month"], as_index=False)["pm25"]
        .mean()
        .sort_values(["城市群", "month"])
    )
    pivot_df = agg.pivot(index="month", columns="城市群", values="pm25").reset_index()
    city_groups = ["京津冀", "长三角", "珠三角"]
    for g in city_groups:
        if g not in pivot_df.columns:
            pivot_df[g] = np.nan
    pivot_df = pivot_df[["month"] + city_groups]
    if pivot_df.isnull().any().any():
        pivot_df = pivot_df.ffill()
    return pivot_df.sort_values("month").reset_index(drop=True)

# ====================== 2. Mann-Kendall趋势检验 ======================
def mann_kendall_test(series):
    """
    Mann-Kendall趋势检验
    返回: Z统计量, p值, 趋势判断
    """
    n = len(series)
    if n < 10:
        return np.nan, np.nan, "样本量不足"
    
    # 计算S统计量
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(series[j] - series[i])
    
    # 计算方差
    var_s = n*(n-1)*(2*n+5)/18
    
    # 计算Z统计量
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    # 计算p值（双侧检验）
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # 判断趋势
    if p < 0.05:
        trend = "显著下降" if z < 0 else "显著上升"
    else:
        trend = "下降趋势" if z < 0 else "上升趋势" if z > 0 else "无明显趋势"
    
    return z, p, trend

# ====================== 3. Pettitt突变检验 ======================
def pettitt_test(series):
    """
    Pettitt突变检验
    返回: 突变点位置, p值, 突变时间
    """
    n = len(series)
    t = np.arange(1, n+1)
    k = np.arange(1, n)
    
    # 计算U统计量
    u = np.zeros(n)
    for i in k:
        u[i] = 2 * np.sum([np.sign(series[i] - series[j]) for j in range(n)])
    
    # 找到最大U值的位置
    u_abs = np.abs(u)
    k_p = np.argmax(u_abs)
    u_max = u_abs[k_p]
    
    # 计算p值
    p = 2 * np.exp(-6 * u_max**2 / (n**3 + n**2))
    
    return k_p, p

# ====================== 4. 滑动T检验 ======================
def sliding_t_test(series, split_point):
    """
    滑动T检验验证突变点
    返回: T统计量, p值
    """
    if split_point <= 0 or split_point >= len(series)-1:
        return np.nan, np.nan
    
    # 分割序列
    series1 = series[:split_point]
    series2 = series[split_point:]
    
    # 进行t检验（不等方差）
    t_stat, p_val = stats.ttest_ind(series1, series2, equal_var=False)
    
    return abs(t_stat), p_val

# ====================== 5. 月度特征分析主函数 ======================
def monthly_analysis_main(file_path, city_group_path=None):
    """
    月度PM2.5浓度变化特征与突变点识别主函数
    """
    _configure_chinese_font()

    # 1. 加载数据
    df = load_and_preprocess_data(file_path, city_group_path)
    _safe_print("数据加载完成，数据维度:", df.shape)

    city_groups = ['京津冀', '长三角', '珠三角']
    results = []
    month_series = df['month']

    # 颜色与线型，便于区分三条线
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝、橙、绿

    # 2. 单图绘制三条线
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle('三大城市群PM2.5月度浓度变化趋势及突变点识别 (2018-2023)', fontsize=14, y=1.02)

    for idx, city in enumerate(city_groups):
        pm_series = df[city].values

        mk_z, mk_p, mk_trend = mann_kendall_test(pm_series)
        pet_k, pet_p = pettitt_test(pm_series)
        mutation_month = month_series.iloc[pet_k] if pet_k < len(month_series) else np.nan
        t_stat, t_p = sliding_t_test(pm_series, pet_k)

        results.append({
            '城市群': city,
            'MK_Z统计量': round(mk_z, 3),
            'MK_P值': round(mk_p, 3),
            '趋势判断': mk_trend,
            'Pettitt_P值': round(pet_p, 3),
            '滑动T统计量': round(t_stat, 3),
            '滑动T_P值': round(t_p, 3),
            '突变点位置': pet_k,
            '突变月份': mutation_month.strftime('%Y-%m') if pd.notna(mutation_month) else '无'
        })

        ax.plot(month_series, pm_series, color=colors[idx], linewidth=1.8, label=city)

    ax.set_xlabel('月份')
    ax.set_ylabel('PM2.5浓度 (μg/m³)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save_figure_dual(fig, '三大城市群PM2.5月度趋势及突变点.png', dpi=300)
    plt.show()
    
    # 8. 输出结果表格
    results_df = pd.DataFrame(results)
    _safe_print("\n=== 月度PM2.5浓度变化特征与突变点识别结果 ===")
    _safe_print(results_df)
    
    # 保存结果到CSV
    results_df.to_csv('月度PM2.5突变点分析结果.csv', index=False, encoding='utf-8-sig')
    _safe_print("\n结果已保存到: 月度PM2.5突变点分析结果.csv")
    
    return results_df

# ====================== 6. 运行分析 ======================
if __name__ == "__main__":
    # 数据文件路径（城市级月均PM2.5宽表：城市 + 201801/2018-01 等月份列）
    data_file_path = r"H:\DATA Science\大论文Result\三大城市群（市）月均PM2.5浓度\合并数据_2018-2023.csv"
    # 城市归属文件（城市 -> 城市群 映射，用于聚合）
    city_group_path = r"H:\DATA Science\大论文Result\大论文图\三大城市群\MSTL时间序列分解\城市归属_三大城市群.csv"

    # 执行分析
    analysis_results = monthly_analysis_main(data_file_path, city_group_path)