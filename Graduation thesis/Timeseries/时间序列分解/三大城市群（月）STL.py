import os
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

# 与 main() 中 output_dir 一致；导出的城市归属表等同目录，便于与其它脚本对齐
DEFAULT_STL_OUTPUT_DIR = r"H:\大论文Result\大论文图\三大城市群\STL时间序列分解"
DEFAULT_STL_INPUT_CSV = r"H:\大论文Result\三大城市群（市）月均PM2.5浓度\合并数据_2018-2023.csv"

# 图题/轴标签用 Unicode 下标（PM₂.₅），勿与 mathtext 的 $...$ 混排中文（见《中文字体显示问题排查指南》§2.4）
PM25_UNICODE = "PM\u2082.\u2085"

# 与 Other tips/Python version/Study area.py 对齐的论文字号（单图主标题 32；子图标题与轴标签 24；刻度 22；图例 28）
THESIS_MAIN_TITLE_SIZE = 32
THESIS_SUBPLOT_TITLE_SIZE = 24
THESIS_LABEL_SIZE = 24
THESIS_TICK_SIZE = 22
THESIS_LEGEND_SIZE = 28
THESIS_LEGEND_TITLE_SIZE = 32

# Windows 下将 simkai 等注册进 matplotlib（见《中文字体显示问题排查指南》§3.4）
_WINDOWS_FONT_FILES_REGISTERED = False


def _register_windows_font_files_once() -> None:
    global _WINDOWS_FONT_FILES_REGISTERED
    if _WINDOWS_FONT_FILES_REGISTERED or os.name != "nt":
        return
    fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for fname in (
        "simkai.ttf",
        "simkai.ttc",
        "msyh.ttc",
        "msyhbd.ttc",
        "simhei.ttf",
        "simsun.ttc",
        "simsunb.ttf",
    ):
        fp = fonts_dir / fname
        if not fp.is_file():
            continue
        try:
            mpl.font_manager.fontManager.addfont(str(fp))
        except (OSError, ValueError, RuntimeError):
            continue
    _WINDOWS_FONT_FILES_REGISTERED = True


def _font_family_resolves(family: str) -> bool:
    fm = mpl.font_manager.fontManager
    try:
        path = fm.findfont(
            mpl.font_manager.FontProperties(family=family),
            fallback_to_default=False,
        )
    except ValueError:
        return False
    return bool(path and os.path.isfile(path))


def _build_thesis_serif_font_chain() -> list[str]:
    """英文 Times New Roman，中文 KaiTi（及回退），与三大城市群（日）MSTL.py / AQI VS PM2.5.py 一致。"""
    _register_windows_font_files_once()
    ordered: list[str] = []

    def append_if_new(name: str) -> None:
        if name not in ordered and _font_family_resolves(name):
            ordered.append(name)

    for name in ("Times New Roman", "Times"):
        append_if_new(name)
        if ordered:
            break
    if not ordered:
        for name in ("DejaVu Serif", "DejaVu Sans", "Arial"):
            append_if_new(name)
            if ordered:
                break

    for name in (
        "KaiTi",
        "STKaiti",
        "DFKai-SB",
        "FZKai-Z03",
        "楷体",
        "Kaiti SC",
        "SimKai",
        "KaiTi_GB2312",
        "华文楷体",
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ):
        append_if_new(name)

    if not any(
        x in ordered
        for x in ("KaiTi", "STKaiti", "DFKai-SB", "FZKai-Z03", "楷体", "Kaiti SC", "SimKai", "KaiTi_GB2312")
    ):
        for font in mpl.font_manager.fontManager.ttflist:
            if "simkai" in font.fname.replace("\\", "/").lower():
                if font.name not in ordered:
                    insert_at = 1 if len(ordered) > 1 else len(ordered)
                    ordered.insert(insert_at, font.name)
                break

    append_if_new("DejaVu Sans")
    return ordered if ordered else ["DejaVu Sans"]


def safe_print(message: str) -> None:
    """安全中文输出，避免控制台编码导致的 UnicodeEncodeError。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


def configure_plot_fonts() -> None:
    """英文 Times New Roman，中文 KaiTi（楷体），按字形回退。"""
    serif_chain = _build_thesis_serif_font_chain()
    if not any(
        x in serif_chain
        for x in ("KaiTi", "STKaiti", "DFKai-SB", "FZKai-Z03", "楷体", "Kaiti SC", "SimKai", "KaiTi_GB2312")
    ):
        safe_print("警告: 未解析到 KaiTi/楷体，中文将使用链中后续字体（如雅黑/宋体）。")

    sns.set_theme(style="whitegrid")
    mpl.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = serif_chain
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.titlesize"] = THESIS_SUBPLOT_TITLE_SIZE
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = THESIS_LABEL_SIZE
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["xtick.labelsize"] = THESIS_TICK_SIZE
    plt.rcParams["ytick.labelsize"] = THESIS_TICK_SIZE
    plt.rcParams["legend.fontsize"] = THESIS_LEGEND_SIZE
    if "legend.title_fontsize" in plt.rcParams:
        plt.rcParams["legend.title_fontsize"] = THESIS_LEGEND_TITLE_SIZE

    safe_print(
        "绘图字体链(西文优先 Times New Roman，中文回退楷体及备用): "
        + ", ".join(serif_chain)
    )


def _svg_font_reorder_disabled() -> bool:
    for key in ("STL_SVG_NO_FONT_REORDER", "MSTL_SVG_NO_FONT_REORDER"):
        if os.environ.get(key, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def _patch_svg_font_family_for_weak_viewers(svg_path: str) -> None:
    """Word 等若只认 font-family 首项，TNR 会导致中文方框；保存后重写为中文优先（见排查指南 §2.3.1）。

    设 STL_SVG_NO_FONT_REORDER=1 或 MSTL_SVG_NO_FONT_REORDER=1 可关闭。
    """
    if _svg_font_reorder_disabled():
        return
    try:
        with open(svg_path, encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return
    office_first = (
        "'Microsoft YaHei', 'KaiTi', 'SimHei', 'SimSun', "
        "'Times New Roman', 'DejaVu Sans', serif"
    )
    patched, n = re.subn(
        r"font-family:\s*[^;]+;",
        f"font-family: {office_first};",
        content,
    )
    if n and patched != content:
        try:
            with open(svg_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(patched)
        except OSError:
            return


def save_figure_dual(fig, save_path_png: str, dpi: int = 300) -> None:
    """同时保存 PNG 与 SVG 版本。"""
    fig.savefig(save_path_png, dpi=dpi, bbox_inches="tight")
    save_path_svg = os.path.splitext(save_path_png)[0] + ".svg"
    fig.savefig(save_path_svg, format="svg", bbox_inches="tight")
    _patch_svg_font_family_for_weak_viewers(save_path_svg)


def resolve_stl_input_csv_path() -> str:
    """宽表输入：环境变量 STL_INPUT_CSV > 脚本同目录合并数据 > 默认 H: 路径。"""
    script_dir = Path(__file__).resolve().parent
    candidates: list[str] = []
    env = os.environ.get("STL_INPUT_CSV", "").strip()
    if env:
        candidates.append(env)
    candidates.append(str(script_dir / "合并数据_2018-2023.csv"))
    candidates.append(DEFAULT_STL_INPUT_CSV)

    tried: list[str] = []
    for p in candidates:
        if not p or p in tried:
            continue
        tried.append(p)
        if os.path.isfile(p):
            safe_print(f"使用月均输入文件: {p}")
            return os.path.abspath(p)

    msg_lines = ["未找到月均宽表 CSV（需含 `城市` 与 YYYYMM 列）。已尝试:"]
    msg_lines.extend(f"  - {p}" for p in tried)
    msg_lines.append(
        "请将 `合并数据_2018-2023.csv` 放在本脚本同目录，或设置环境变量 STL_INPUT_CSV=绝对路径。"
    )
    raise FileNotFoundError("\n".join(msg_lines))


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


def load_and_transform_data(csv_path: str) -> pd.DataFrame:
    """读取宽表并转为长表: 城市, Month, PM2.5。"""
    if not os.path.isfile(csv_path):
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
    axes[0].set_title(
        f"{cluster_name} {PM25_UNICODE} 原始序列",
        fontsize=THESIS_SUBPLOT_TITLE_SIZE,
        fontweight="bold",
    )
    axes[0].set_ylabel("浓度")

    axes[1].plot(result.trend, color="#d62728", lw=1.5)
    axes[1].set_title("趋势项", fontsize=THESIS_SUBPLOT_TITLE_SIZE, fontweight="bold")
    axes[1].set_ylabel("趋势")

    axes[2].plot(result.seasonal, color="#2ca02c", lw=1.5)
    axes[2].set_title("季节项", fontsize=THESIS_SUBPLOT_TITLE_SIZE, fontweight="bold")
    axes[2].set_ylabel("季节")

    axes[3].plot(result.resid, color="#ff7f0e", lw=1.0)
    axes[3].axhline(0, ls="--", c="black", alpha=0.6)
    axes[3].set_title("残差项", fontsize=THESIS_SUBPLOT_TITLE_SIZE, fontweight="bold")
    axes[3].set_ylabel("残差")
    axes[3].set_xlabel("时间")

    for ax in axes:
        ax.tick_params(axis="both", labelsize=THESIS_TICK_SIZE)
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
    axes[0, 0].set_title(
        f"{cluster_name} 季节项(月)箱线图",
        fontsize=THESIS_SUBPLOT_TITLE_SIZE,
        fontweight="bold",
    )

    sns.violinplot(data=seasonal_df, x="月份", y="季节项", ax=axes[0, 1], color="#b8e986")
    axes[0, 1].set_title(
        "季节项分布(小提琴图)",
        fontsize=THESIS_SUBPLOT_TITLE_SIZE,
        fontweight="bold",
    )

    axes[1, 0].plot(trend_change.index, trend_change.values, color="#1f77b4", lw=1.2)
    axes[1, 0].axhline(0, ls="--", c="red", alpha=0.6)
    axes[1, 0].set_title(
        "趋势月变化率(%)",
        fontsize=THESIS_SUBPLOT_TITLE_SIZE,
        fontweight="bold",
    )

    axes[1, 1].hist(result.resid.values, bins=20, color="#f7b267", edgecolor="black", alpha=0.85)
    axes[1, 1].axvline(0, ls="--", c="red", alpha=0.7)
    axes[1, 1].set_title(
        "残差分布直方图",
        fontsize=THESIS_SUBPLOT_TITLE_SIZE,
        fontweight="bold",
    )

    for ax in axes.ravel():
        ax.tick_params(axis="both", labelsize=THESIS_TICK_SIZE)
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
    ax.set_title(
        f"三大城市群月均 {PM25_UNICODE} 浓度对比 (2018-2023)",
        fontsize=THESIS_MAIN_TITLE_SIZE,
        fontweight="bold",
    )
    ax.set_xlabel("时间")
    ax.set_ylabel(PM25_UNICODE)
    ax.tick_params(axis="both", labelsize=THESIS_TICK_SIZE)
    ax.grid(alpha=0.25)
    leg = ax.get_legend()
    if leg is not None:
        plt.setp(leg.get_texts(), fontsize=THESIS_LEGEND_SIZE)
        t = leg.get_title()
        if t is not None and t.get_text():
            t.set_fontsize(THESIS_LEGEND_TITLE_SIZE)
    plt.tight_layout()
    save_figure_dual(fig, save_path_png=save_path, dpi=300)
    plt.close(fig)


def plot_cluster_trend_comparison(stl_result_map, save_path):
    """三大城市群 STL 趋势项对比图。"""
    fig, ax = plt.subplots(figsize=(14, 6))
    for cluster_name, result in stl_result_map.items():
        ax.plot(result.trend.index, result.trend.values, lw=2.0, label=cluster_name)
    ax.set_title(
        f"三大城市群 {PM25_UNICODE} STL 趋势项对比",
        fontsize=THESIS_MAIN_TITLE_SIZE,
        fontweight="bold",
    )
    ax.set_xlabel("时间")
    ax.set_ylabel("趋势项")
    ax.legend(fontsize=THESIS_LEGEND_SIZE)
    ax.tick_params(axis="both", labelsize=THESIS_TICK_SIZE)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_figure_dual(fig, save_path_png=save_path, dpi=300)
    plt.close(fig)


def main():
    configure_plot_fonts()
    safe_print("=" * 60)
    safe_print("三大城市群（月）PM2.5 STL 分解分析")
    safe_print("=" * 60)

    input_csv = resolve_stl_input_csv_path()
    output_dir = DEFAULT_STL_OUTPUT_DIR
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
