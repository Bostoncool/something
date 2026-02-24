import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")


def safe_print(message: str) -> None:
    """安全中文输出，避免控制台编码导致的 UnicodeEncodeError。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


def configure_plot_fonts():
    """固定中文字体链，避免中文渲染为方框。"""
    # 使用与“可正常中文输出脚本”一致的优先字体逻辑
    chinese_font_chain = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    available_in_chain = [name for name in chinese_font_chain if name in available]
    if not available_in_chain:
        safe_print("警告: 未检测到常用中文字体，图片中文可能显示异常。")

    # 先设置 seaborn 主题，再强制覆盖字体相关 rc，避免被 theme 重置。
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = chinese_font_chain
    plt.rcParams["axes.unicode_minus"] = False

    if available_in_chain:
        safe_print(f"可用中文字体: {', '.join(available_in_chain)}")


def save_figure_dual(fig, save_path_png: str, dpi: int = 300) -> None:
    """同时保存 PNG 与 SVG 版本。"""
    fig.savefig(save_path_png, dpi=dpi, bbox_inches="tight")
    save_path_svg = os.path.splitext(save_path_png)[0] + ".svg"
    fig.savefig(save_path_svg, format="svg", bbox_inches="tight")


def get_city_cluster_map():
    """三大城市群-城市映射。"""
    return {
        "京津冀": {
            "北京", "天津",
            "石家庄", "唐山", "秦皇岛", "邯郸", "邢台", "保定",
            "张家口", "承德", "沧州", "廊坊", "衡水",
        },
        "长三角": {
            "上海",
            "南京", "无锡", "常州", "苏州", "南通", "盐城", "扬州", "镇江", "泰州",
            "杭州", "宁波", "温州", "嘉兴", "湖州", "绍兴", "金华", "舟山", "台州",
            "合肥", "芜湖", "马鞍山", "铜陵", "安庆", "滁州", "池州", "宣城",
        },
        "珠三角": {
            "广州", "深圳", "珠海", "佛山", "江门", "肇庆", "惠州", "东莞", "中山",
        },
    }


def load_and_transform_data(csv_path):
    """读取宽表并转为长表: 城市, Month, PM2.5。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"输入文件不存在: {csv_path}")

    try:
        df_wide = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df_wide = pd.read_csv(csv_path, encoding="gbk")

    df_wide.columns = [str(col).strip() for col in df_wide.columns]
    if "城市" not in df_wide.columns:
        raise ValueError("数据缺少 `城市` 列")

    df_wide["城市"] = df_wide["城市"].astype(str).str.strip()
    month_cols = [c for c in df_wide.columns if str(c).isdigit() and len(str(c)) == 6]
    if not month_cols:
        raise ValueError("未识别到 YYYYMM 格式月份列")

    df_long = (
        df_wide.melt(
            id_vars="城市",
            value_vars=month_cols,
            var_name="Month",
            value_name="PM2.5",
        )
        .assign(
            Month=lambda d: pd.to_datetime(d["Month"], format="%Y%m"),
            **{"PM2.5": lambda d: pd.to_numeric(d["PM2.5"], errors="coerce")},
        )
        .dropna(subset=["PM2.5"])
        .sort_values(["城市", "Month"])
        .reset_index(drop=True)
    )
    return df_long


def assign_city_cluster(df_long, city_cluster_map):
    """识别每个城市属于哪个城市群，并返回带城市群的数据。"""
    city_to_cluster = {
        city: cluster
        for cluster, cities in city_cluster_map.items()
        for city in cities
    }
    city_ref = (
        pd.DataFrame({"城市": sorted(df_long["城市"].unique())})
        .assign(城市群=lambda d: d["城市"].map(city_to_cluster))
    )

    unknown = city_ref[city_ref["城市群"].isna()]["城市"].tolist()
    if unknown:
        raise ValueError(f"以下城市未识别到城市群: {unknown}")

    merged = df_long.merge(city_ref, on="城市", how="left")
    return merged, city_ref


def build_cluster_monthly_series(df_with_cluster):
    """按城市群聚合到月均 PM2.5 序列。"""
    return (
        df_with_cluster
        .groupby(["城市群", "Month"], as_index=False)["PM2.5"]
        .mean()
        .sort_values(["城市群", "Month"])
    )


def perform_stl_decomposition(ts_data, seasonal=13):
    """执行 STL 分解；月度序列使用 13 作为 seasonal(奇数)。"""
    if len(ts_data) < 24:
        raise ValueError(f"序列长度不足，至少需要24个月，当前: {len(ts_data)}")
    result = STL(ts_data, seasonal=seasonal, robust=True).fit()
    return result


def analyze_stl_result(result, cluster_name):
    """输出每个城市群的 STL 统计指标。"""
    observed_var = np.var(result.observed)
    trend_var = np.var(result.trend)
    seasonal_var = np.var(result.seasonal)
    resid_var = np.var(result.resid)

    trend_start = result.trend.iloc[0]
    trend_end = result.trend.iloc[-1]
    if trend_end > trend_start * 1.05:
        trend_direction = "上升"
    elif trend_end < trend_start * 0.95:
        trend_direction = "下降"
    else:
        trend_direction = "平稳"

    seasonal_by_month = result.seasonal.groupby(result.seasonal.index.month).mean()

    return {
        "城市群": cluster_name,
        "趋势强度": trend_var / observed_var if observed_var > 0 else np.nan,
        "季节强度": seasonal_var / observed_var if observed_var > 0 else np.nan,
        "残差强度": resid_var / observed_var if observed_var > 0 else np.nan,
        "趋势方向": trend_direction,
        "季节峰值月": int(seasonal_by_month.idxmax()),
        "季节谷值月": int(seasonal_by_month.idxmin()),
        "观测值范围": f"{result.observed.min():.2f} - {result.observed.max():.2f}",
    }


def plot_stl_components(result, cluster_name, save_path):
    """绘制 STL 四分量图。"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    axes[0].plot(result.observed, color="#1f77b4", lw=1.5)
    axes[0].set_title(f"{cluster_name} PM2.5 原始序列", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("浓度")

    axes[1].plot(result.trend, color="#d62728", lw=1.5)
    axes[1].set_title("趋势项", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("趋势")

    axes[2].plot(result.seasonal, color="#2ca02c", lw=1.5)
    axes[2].set_title("季节项", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("季节")

    axes[3].plot(result.resid, color="#ff7f0e", lw=1.0)
    axes[3].axhline(0, ls="--", c="black", alpha=0.6)
    axes[3].set_title("残差项", fontsize=12, fontweight="bold")
    axes[3].set_ylabel("残差")
    axes[3].set_xlabel("时间")

    for ax in axes:
        ax.grid(alpha=0.25)
    plt.tight_layout()
    save_figure_dual(fig, save_path_png=save_path, dpi=300)
    plt.close(fig)


def plot_seasonal_analysis(result, cluster_name, save_path):
    """绘制季节性细节图。"""
    seasonal_df = pd.DataFrame(
        {
            "月份": result.seasonal.index.month,
            "季节项": result.seasonal.values,
        }
    )
    trend_change = result.trend.pct_change() * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.boxplot(data=seasonal_df, x="月份", y="季节项", ax=axes[0, 0], color="#91c8f6")
    axes[0, 0].set_title(f"{cluster_name} 季节项(月)箱线图", fontweight="bold")

    sns.violinplot(data=seasonal_df, x="月份", y="季节项", ax=axes[0, 1], color="#b8e986")
    axes[0, 1].set_title("季节项分布(小提琴图)", fontweight="bold")

    axes[1, 0].plot(trend_change.index, trend_change.values, color="#1f77b4", lw=1.2)
    axes[1, 0].axhline(0, ls="--", c="red", alpha=0.6)
    axes[1, 0].set_title("趋势月变化率(%)", fontweight="bold")

    axes[1, 1].hist(result.resid.values, bins=20, color="#f7b267", edgecolor="black", alpha=0.85)
    axes[1, 1].axvline(0, ls="--", c="red", alpha=0.7)
    axes[1, 1].set_title("残差分布直方图", fontweight="bold")

    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    plt.tight_layout()
    save_figure_dual(fig, save_path_png=save_path, dpi=300)
    plt.close(fig)


def plot_cluster_raw_comparison(cluster_monthly_df, save_path):
    """三大城市群月均 PM2.5 对比图。"""
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(
        data=cluster_monthly_df,
        x="Month",
        y="PM2.5",
        hue="城市群",
        marker="o",
        linewidth=1.8,
        ax=ax,
    )
    ax.set_title("三大城市群月均 PM2.5 浓度对比 (2018-2023)", fontsize=13, fontweight="bold")
    ax.set_xlabel("时间")
    ax.set_ylabel("PM2.5")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_figure_dual(fig, save_path_png=save_path, dpi=300)
    plt.close(fig)


def plot_cluster_trend_comparison(stl_result_map, save_path):
    """三大城市群 STL 趋势项对比图。"""
    fig, ax = plt.subplots(figsize=(14, 6))
    for cluster_name, result in stl_result_map.items():
        ax.plot(result.trend.index, result.trend.values, lw=2.0, label=cluster_name)
    ax.set_title("三大城市群 PM2.5 STL 趋势项对比", fontsize=13, fontweight="bold")
    ax.set_xlabel("时间")
    ax.set_ylabel("趋势项")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_figure_dual(fig, save_path_png=save_path, dpi=300)
    plt.close(fig)


def main():
    configure_plot_fonts()
    safe_print("=" * 60)
    safe_print("三大城市群（月）PM2.5 STL 分解分析")
    safe_print("=" * 60)

    input_csv = r"H:\DATA Science\大论文Result\三大城市群（市）月均PM2.5浓度\合并数据_2018-2023.csv"
    output_dir = r"H:\DATA Science\大论文Result\大论文图\三大城市群\STL时间序列分解"
    os.makedirs(output_dir, exist_ok=True)

    city_cluster_map = get_city_cluster_map()
    df_long = load_and_transform_data(input_csv)
    df_with_cluster, city_ref = assign_city_cluster(df_long, city_cluster_map)
    cluster_monthly_df = build_cluster_monthly_series(df_with_cluster)

    # 输出城市归属表，满足“先识别城市归属”的要求
    city_ref_path = os.path.join(output_dir, "城市归属_三大城市群.csv")
    city_ref.sort_values(["城市群", "城市"]).to_csv(city_ref_path, index=False, encoding="utf-8-sig")
    safe_print(f"城市归属表已保存: {city_ref_path}")

    # 输出聚合后月度数据
    monthly_output_path = os.path.join(output_dir, "城市群月均PM25_2018_2023.csv")
    cluster_monthly_df.to_csv(monthly_output_path, index=False, encoding="utf-8-sig")
    safe_print(f"城市群月度数据已保存: {monthly_output_path}")

    # 总体对比图
    plot_cluster_raw_comparison(
        cluster_monthly_df=cluster_monthly_df,
        save_path=os.path.join(output_dir, "三大城市群_PM25月均对比.png"),
    )

    all_analysis = []
    stl_result_map = {}
    for cluster_name in cluster_monthly_df["城市群"].unique():
        sub_df = cluster_monthly_df[cluster_monthly_df["城市群"] == cluster_name].copy()
        ts_data = pd.Series(sub_df["PM2.5"].values, index=sub_df["Month"], name=cluster_name).asfreq("MS")
        ts_data = ts_data.interpolate(method="linear").ffill().bfill()

        stl_result = perform_stl_decomposition(ts_data, seasonal=13)
        stl_result_map[cluster_name] = stl_result
        all_analysis.append(analyze_stl_result(stl_result, cluster_name))

        cluster_output_dir = os.path.join(output_dir, cluster_name)
        os.makedirs(cluster_output_dir, exist_ok=True)

        plot_stl_components(
            result=stl_result,
            cluster_name=cluster_name,
            save_path=os.path.join(cluster_output_dir, f"{cluster_name}_STL分解.png"),
        )
        plot_seasonal_analysis(
            result=stl_result,
            cluster_name=cluster_name,
            save_path=os.path.join(cluster_output_dir, f"{cluster_name}_季节性分析.png"),
        )
        safe_print(f"{cluster_name} 图表输出完成: {cluster_output_dir}")

    # 趋势对比图
    plot_cluster_trend_comparison(
        stl_result_map=stl_result_map,
        save_path=os.path.join(output_dir, "三大城市群_STL趋势对比.png"),
    )

    # STL 指标汇总
    analysis_df = pd.DataFrame(all_analysis).sort_values("城市群")
    analysis_path = os.path.join(output_dir, "三大城市群_STL分析汇总.csv")
    analysis_df.to_csv(analysis_path, index=False, encoding="utf-8-sig")
    safe_print(f"STL 分析汇总已保存: {analysis_path}")

    safe_print("\n分析完成。")
    safe_print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
