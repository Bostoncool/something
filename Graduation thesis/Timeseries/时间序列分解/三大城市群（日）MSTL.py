import os
import re
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import MSTL
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 与 main() 中 output_dir 一致；上次运行导出的城市归属表通常在此路径
DEFAULT_MSTL_OUTPUT_DIR = r"H:\大论文Result\大论文图\三大城市群\MSTL时间序列分解"

# 图题/轴标签用 Unicode 下标（PM₂.₅），勿与 mathtext 的 $...$ 混排中文（见《中文字体显示问题排查指南》§2.4）
PM25_UNICODE = "PM\u2082.\u2085"

# 与 Other tips/Python version/Study area.py 对齐的论文字号（单图主标题 32；子图标题与轴标签 24；刻度 22；图例 28）
THESIS_MAIN_TITLE_SIZE = 32
THESIS_SUBPLOT_TITLE_SIZE = 24
THESIS_LABEL_SIZE = 24
THESIS_TICK_SIZE = 22
THESIS_LEGEND_SIZE = 28
THESIS_LEGEND_TITLE_SIZE = 32

# Windows 下将 simkai 等注册进 matplotlib，避免 ttflist 里无「KaiTi」名称（见《中文字体显示问题排查指南》§3.4）
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
    """英文 Times New Roman，中文 KaiTi（及回退），与 AQI VS PM2.5.py 思路一致。"""
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
    """安全中文输出，避免控制台编码导致 UnicodeEncodeError。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


def configure_plot_fonts() -> None:
    """英文以 Times New Roman 显示，中文以 KaiTi（楷体）显示（按字形回退）。"""
    serif_chain = _build_thesis_serif_font_chain()
    if not any(
        x in serif_chain
        for x in ("KaiTi", "STKaiti", "DFKai-SB", "FZKai-Z03", "楷体", "Kaiti SC", "SimKai", "KaiTi_GB2312")
    ):
        safe_print("警告: 未解析到 KaiTi/楷体，中文将使用链中后续字体（如雅黑/宋体）。")

    sns.set_theme(style="whitegrid")
    # SVG 默认 path 模式下，西文字体可能对汉字生成错误占位轮廓（指南 §2.3）；保留 <text> 由查看器按本机字体绘制
    mpl.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = serif_chain
    plt.rcParams["axes.unicode_minus"] = False
    # 字号与 Study area.py 一致；子图默认标题用 24，单图总览在绘图函数中显式 32
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


def _patch_svg_font_family_for_weak_viewers(svg_path: str) -> None:
    """部分查看器（尤其 Word）几乎只用 font-family 的第一项；若首项为 Times New Roman，中文会显示为方框。

    导出仍为 Matplotlib 默认的「西文优先」列表；保存后对 SVG 内 style 做一次性替换，使中文字体排在前面。
    详见《中文字体显示问题排查指南》§2.3.1。若需保留原顺序（如仅在浏览器核对西文），可设置环境变量
    MSTL_SVG_NO_FONT_REORDER=1。
    """
    if os.environ.get("MSTL_SVG_NO_FONT_REORDER", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    try:
        with open(svg_path, encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return
    # 与 Windows / Office 常见环境一致；拉丁字母仍可由这些字体中的西文字形显示
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


def normalize_city_name(city_name: str) -> str:
    """标准化城市名，用于稳健匹配城市归属。"""
    name = str(city_name).strip()
    name = re.sub(r"\s+", "", name)
    if name.endswith("市"):
        name = name[:-1]
    return name


def read_csv_with_fallback(csv_path: str) -> pd.DataFrame:
    """按常见编码兜底读取 CSV。"""
    for encoding in ["utf-8-sig", "gbk", "utf-8"]:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(csv_path)


def parse_date_from_filename(file_path: str) -> pd.Timestamp | None:
    """从文件名提取 YYYYMMDD 日期。"""
    stem = Path(file_path).stem
    match = re.search(r"(\d{8})", stem)
    if not match:
        return None
    try:
        return pd.to_datetime(match.group(1), format="%Y%m%d")
    except ValueError:
        return None


def process_single_file(file_path: str) -> list[dict]:
    """处理单个日文件并返回长表记录: 日期, 城市, PM2.5。"""
    try:
        df = read_csv_with_fallback(file_path)
        df.columns = [str(col).strip() for col in df.columns]

        date_val = parse_date_from_filename(file_path)
        if date_val is None and "date" in df.columns:
            date_series = pd.to_datetime(df["date"], errors="coerce")
            date_series = date_series.dropna()
            if not date_series.empty:
                date_val = date_series.iloc[0].normalize()

        if date_val is None:
            safe_print(f"跳过文件(无法识别日期): {file_path}")
            return []

        if "type" not in df.columns:
            safe_print(f"跳过文件(缺少 type 列): {file_path}")
            return []

        city_cols = [
            col
            for col in df.columns
            if col not in {"date", "hour", "type", "__file__", "__missing_cols__"}
            and not str(col).startswith("Unnamed")
        ]
        if not city_cols:
            return []

        df_pm25 = df[df["type"].astype(str).str.strip().str.upper() == "PM2.5"].copy()
        if df_pm25.empty:
            return []

        city_values = (
            df_pm25[city_cols]
            .apply(pd.to_numeric, errors="coerce")
            .mean(axis=0, skipna=True)
            .dropna()
        )
        if city_values.empty:
            return []

        records = [
            {"日期": date_val, "城市": str(city).strip(), "PM2.5": float(value)}
            for city, value in city_values.items()
        ]
        return records

    except Exception as exc:
        safe_print(f"文件处理失败: {file_path} | {exc}")
        return []


def load_daily_city_data(input_folders: list[str], n_processes: int | None = None) -> pd.DataFrame:
    """读取多个目录下所有日文件，汇总为城市日均 PM2.5 长表。"""
    all_files: list[str] = []
    for folder in input_folders:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"输入目录不存在: {folder}")
        folder_files = sorted(str(path) for path in Path(folder).glob("*.csv"))
        safe_print(f"{folder} -> 识别到 {len(folder_files)} 个 CSV 文件")
        all_files.extend(folder_files)

    if not all_files:
        raise ValueError("未找到任何 CSV 文件")

    if n_processes is None:
        n_processes = max(cpu_count() - 1, 1)
    safe_print(f"并行进程数: {n_processes}")

    with Pool(processes=n_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_file, all_files),
                total=len(all_files),
                desc="读取日文件",
                unit="file",
            )
        )

    records = [record for sublist in results for record in sublist]
    if not records:
        raise ValueError("未从 CSV 中解析到有效 PM2.5 数据")

    daily_city_df = pd.DataFrame(records)
    daily_city_df["日期"] = pd.to_datetime(daily_city_df["日期"], errors="coerce")
    daily_city_df["城市"] = daily_city_df["城市"].astype(str).str.strip()
    daily_city_df = daily_city_df.dropna(subset=["日期", "城市", "PM2.5"])

    # 多目录可能存在同一天同城市重复记录，这里做均值去重。
    daily_city_df = (
        daily_city_df.groupby(["日期", "城市"], as_index=False)["PM2.5"]
        .mean()
        .sort_values(["城市", "日期"])
        .reset_index(drop=True)
    )
    return daily_city_df


def resolve_city_cluster_csv_path() -> str:
    """城市归属表路径：环境变量 > MSTL 输出目录 > 脚本同目录 > STL 目录下默认文件。"""
    script_dir = Path(__file__).resolve().parent
    default_h = r"H:\大论文Result\大论文图\三大城市群\STL\城市归属_三大城市群.csv"
    candidates: list[str] = []
    env = os.environ.get("MSTL_CITY_CLUSTER_CSV", "").strip()
    if env:
        candidates.append(env)
    candidates.append(str(Path(DEFAULT_MSTL_OUTPUT_DIR) / "城市归属_三大城市群.csv"))
    candidates.append(str(script_dir / "城市归属_三大城市群.csv"))
    candidates.append(default_h)

    tried = []
    for p in candidates:
        if not p or p in tried:
            continue
        tried.append(p)
        if os.path.isfile(p):
            safe_print(f"使用城市归属文件: {p}")
            return os.path.abspath(p)

    msg_lines = ["未找到城市归属 CSV（城市, 城市群）。已尝试:"]
    msg_lines.extend(f"  - {p}" for p in tried)
    msg_lines.append(
        "请将 `城市归属_三大城市群.csv` 放在本脚本同目录，或设置环境变量 "
        "MSTL_CITY_CLUSTER_CSV=该文件的绝对路径。"
    )
    raise FileNotFoundError("\n".join(msg_lines))


def load_city_cluster_reference(city_cluster_path: str) -> pd.DataFrame:
    """读取城市归属表，并标准化为 `城市`,`城市群`。"""
    if not os.path.isfile(city_cluster_path):
        raise FileNotFoundError(f"城市归属文件不存在: {city_cluster_path}")

    ref = read_csv_with_fallback(city_cluster_path)
    ref.columns = [str(col).strip() for col in ref.columns]

    city_col_candidates = ["城市", "city", "City"]
    cluster_col_candidates = ["城市群", "cluster", "Cluster"]

    city_col = next((col for col in city_col_candidates if col in ref.columns), None)
    cluster_col = next((col for col in cluster_col_candidates if col in ref.columns), None)
    if city_col is None or cluster_col is None:
        raise ValueError(
            f"城市归属文件列名不符合预期。当前列: {list(ref.columns)}，"
            "至少需要包含 `城市` 与 `城市群`。"
        )

    ref = (
        ref[[city_col, cluster_col]]
        .rename(columns={city_col: "城市", cluster_col: "城市群"})
        .dropna(subset=["城市", "城市群"])
        .assign(
            城市=lambda d: d["城市"].astype(str).str.strip(),
            城市群=lambda d: d["城市群"].astype(str).str.strip(),
            城市标准名=lambda d: d["城市"].map(normalize_city_name),
        )
    )
    ref = ref.drop_duplicates(subset=["城市标准名"])
    return ref


def assign_city_cluster(daily_city_df: pd.DataFrame, city_ref_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """将城市日序列映射到城市群，并校验未归属城市。"""
    city_ref = city_ref_df[["城市", "城市群", "城市标准名"]].copy()
    daily_df = daily_city_df.copy()
    daily_df["城市标准名"] = daily_df["城市"].map(normalize_city_name)

    merged = daily_df.merge(city_ref[["城市标准名", "城市群"]], on="城市标准名", how="left")

    unknown_cities = (
        merged[merged["城市群"].isna()]["城市"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if unknown_cities:
        raise ValueError(
            "以下城市未在城市归属文件中匹配到城市群，请先补充映射: "
            f"{unknown_cities}"
        )
    return merged, city_ref[["城市", "城市群"]].drop_duplicates()


def build_cluster_daily_series(df_with_cluster: pd.DataFrame) -> pd.DataFrame:
    """按城市群-日期聚合得到城市群日均 PM2.5 序列。"""
    cluster_daily_df = (
        df_with_cluster.groupby(["城市群", "日期"], as_index=False)["PM2.5"]
        .mean()
        .sort_values(["城市群", "日期"])
        .reset_index(drop=True)
    )
    return cluster_daily_df


def _validate_periods(data_length: int, seasonal_periods: tuple[int, ...]) -> tuple[int, ...]:
    valid_periods = []
    for period in seasonal_periods:
        if period < 2:
            continue
        if data_length >= (2 * period):
            valid_periods.append(int(period))
        else:
            safe_print(f"  跳过周期 {period}: 数据长度 {data_length} < 2x周期")
    return tuple(sorted(set(valid_periods)))


def perform_mstl_decomposition(ts_data: pd.Series, seasonal_periods: tuple[int, ...]) -> tuple | None:
    """执行 MSTL 分解，返回 (result, valid_periods)。"""
    ts_data = ts_data.dropna()
    if ts_data.empty:
        return None

    valid_periods = _validate_periods(len(ts_data), seasonal_periods)
    if not valid_periods:
        return None

    result = MSTL(ts_data, periods=valid_periods, stl_kwargs={"robust": True}).fit()
    return result, valid_periods


def _get_seasonal_df(result) -> pd.DataFrame:
    seasonal = result.seasonal
    if isinstance(seasonal, pd.Series):
        return seasonal.to_frame(name="seasonal")
    return seasonal


def analyze_mstl_result(result, cluster_name: str, valid_periods: tuple[int, ...]) -> dict:
    """提取每个城市群的 MSTL 指标。"""
    seasonal_df = _get_seasonal_df(result)
    seasonal_total = seasonal_df.sum(axis=1)

    observed_var = np.var(result.observed)
    trend_var = np.var(result.trend)
    seasonal_var = np.var(seasonal_total)
    resid_var = np.var(result.resid)

    trend_start = result.trend.iloc[0]
    trend_end = result.trend.iloc[-1]
    if trend_end > trend_start * 1.05:
        trend_direction = "上升"
    elif trend_end < trend_start * 0.95:
        trend_direction = "下降"
    else:
        trend_direction = "平稳"

    seasonal_by_month = seasonal_total.groupby(seasonal_total.index.month).mean()
    return {
        "城市群": cluster_name,
        "趋势强度": trend_var / observed_var if observed_var > 0 else np.nan,
        "季节强度": seasonal_var / observed_var if observed_var > 0 else np.nan,
        "残差强度": resid_var / observed_var if observed_var > 0 else np.nan,
        "趋势方向": trend_direction,
        "季节峰值月": int(seasonal_by_month.idxmax()),
        "季节谷值月": int(seasonal_by_month.idxmin()),
        "数据点数": int(len(result.observed)),
        "生效季节周期": ",".join(str(p) for p in valid_periods),
        "观测值范围": f"{result.observed.min():.2f} - {result.observed.max():.2f}",
    }


def plot_mstl_components(result, cluster_name: str, save_path: str) -> None:
    """绘制 MSTL 分解分量图（支持多季节项）。"""
    seasonal_df = _get_seasonal_df(result)
    n_seasonal = seasonal_df.shape[1]
    n_rows = 3 + n_seasonal

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.1 * n_rows), sharex=True)
    axes = np.atleast_1d(axes)

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

    for idx, col in enumerate(seasonal_df.columns, start=2):
        axes[idx].plot(seasonal_df[col], color="#2ca02c", lw=1.2)
        axes[idx].set_title(
            f"季节项({col})",
            fontsize=THESIS_SUBPLOT_TITLE_SIZE,
            fontweight="bold",
        )
        axes[idx].set_ylabel("季节")

    resid_axis = 2 + n_seasonal
    axes[resid_axis].plot(result.resid, color="#ff7f0e", lw=1.0)
    axes[resid_axis].axhline(0, ls="--", c="black", alpha=0.6)
    axes[resid_axis].set_title(
        "残差项", fontsize=THESIS_SUBPLOT_TITLE_SIZE, fontweight="bold"
    )
    axes[resid_axis].set_ylabel("残差")
    axes[resid_axis].set_xlabel("时间")

    for ax in axes:
        ax.tick_params(axis="both", labelsize=THESIS_TICK_SIZE)
        ax.grid(alpha=0.25)
    plt.tight_layout()
    save_figure_dual(fig, save_path_png=save_path, dpi=300)
    plt.close(fig)


def plot_seasonal_analysis(result, cluster_name: str, save_path: str) -> None:
    """绘制季节性细节图（基于总季节项）。"""
    seasonal_df = _get_seasonal_df(result)
    seasonal_total = seasonal_df.sum(axis=1)
    trend_change = result.trend.pct_change() * 100

    seasonal_plot_df = pd.DataFrame(
        {"月份": seasonal_total.index.month, "季节项": seasonal_total.values}
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.boxplot(data=seasonal_plot_df, x="月份", y="季节项", ax=axes[0, 0], color="#91c8f6")
    axes[0, 0].set_title(
        f"{cluster_name} 总季节项(月)箱线图",
        fontsize=THESIS_SUBPLOT_TITLE_SIZE,
        fontweight="bold",
    )

    sns.violinplot(data=seasonal_plot_df, x="月份", y="季节项", ax=axes[0, 1], color="#b8e986")
    axes[0, 1].set_title(
        "总季节项分布(小提琴图)",
        fontsize=THESIS_SUBPLOT_TITLE_SIZE,
        fontweight="bold",
    )

    axes[1, 0].plot(trend_change.index, trend_change.values, color="#1f77b4", lw=1.2)
    axes[1, 0].axhline(0, ls="--", c="red", alpha=0.6)
    axes[1, 0].set_title(
        "趋势日变化率(%)",
        fontsize=THESIS_SUBPLOT_TITLE_SIZE,
        fontweight="bold",
    )

    axes[1, 1].hist(result.resid.values, bins=30, color="#f7b267", edgecolor="black", alpha=0.85)
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


def plot_cluster_raw_comparison(cluster_daily_df: pd.DataFrame, save_path: str) -> None:
    """绘制三大城市群日均 PM2.5 对比图。"""
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.lineplot(
        data=cluster_daily_df,
        x="日期",
        y="PM2.5",
        hue="城市群",
        linewidth=1.2,
        ax=ax,
    )
    ax.set_title(
        f"三大城市群日均 {PM25_UNICODE} 浓度对比",
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


def plot_cluster_trend_comparison(mstl_result_map: dict, save_path: str) -> None:
    """绘制三大城市群 MSTL 趋势项对比图。"""
    fig, ax = plt.subplots(figsize=(15, 6))
    for cluster_name, result in mstl_result_map.items():
        ax.plot(result.trend.index, result.trend.values, lw=2.0, label=cluster_name)
    ax.set_title(
        f"三大城市群 {PM25_UNICODE} MSTL 趋势项对比",
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


def main() -> None:
    configure_plot_fonts()
    safe_print("=" * 60)
    safe_print("三大城市群（日）PM2.5 MSTL 分解分析")
    safe_print("=" * 60)

    input_folders = [
        r"H:\大论文Result\BTH\filtered_daily",
        r"H:\大论文Result\PRD\filtered_daily",
        r"H:\大论文Result\YRD\filtered_daily",
    ]
    city_cluster_path = resolve_city_cluster_csv_path()
    output_dir = DEFAULT_MSTL_OUTPUT_DIR
    seasonal_periods = (7, 365)

    os.makedirs(output_dir, exist_ok=True)

    safe_print("步骤1/6: 读取三大区域日度数据...")
    daily_city_df = load_daily_city_data(input_folders=input_folders, n_processes=None)
    safe_print(f"城市日序列记录数: {len(daily_city_df)}")

    safe_print("步骤2/6: 读取城市归属并执行映射校验...")
    city_ref_df = load_city_cluster_reference(city_cluster_path)
    df_with_cluster, city_ref = assign_city_cluster(daily_city_df, city_ref_df)

    city_ref_out = os.path.join(output_dir, "城市归属_三大城市群.csv")
    city_ref.sort_values(["城市群", "城市"]).to_csv(city_ref_out, index=False, encoding="utf-8-sig")
    safe_print(f"城市归属表已输出: {city_ref_out}")

    safe_print("步骤3/6: 按城市群聚合日均 PM2.5 序列...")
    cluster_daily_df = build_cluster_daily_series(df_with_cluster)
    cluster_daily_path = os.path.join(output_dir, "城市群日均PM25_2018_2023.csv")
    cluster_daily_df.to_csv(cluster_daily_path, index=False, encoding="utf-8-sig")
    safe_print(f"城市群日度序列已输出: {cluster_daily_path}")

    safe_print("步骤4/6: 绘制城市群日均对比图...")
    plot_cluster_raw_comparison(
        cluster_daily_df=cluster_daily_df,
        save_path=os.path.join(output_dir, "三大城市群_PM25日均对比.png"),
    )

    safe_print("步骤5/6: 对每个城市群执行 MSTL 分解并输出图表...")
    all_analysis = []
    mstl_result_map = {}

    for cluster_name in cluster_daily_df["城市群"].drop_duplicates():
        sub_df = cluster_daily_df[cluster_daily_df["城市群"] == cluster_name].copy()
        ts_data = pd.Series(sub_df["PM2.5"].values, index=sub_df["日期"], name=cluster_name)
        ts_data = ts_data.sort_index().asfreq("D")
        ts_data = ts_data.interpolate(method="time").ffill().bfill()

        decomposition_result = perform_mstl_decomposition(ts_data, seasonal_periods=seasonal_periods)
        if decomposition_result is None:
            safe_print(f"{cluster_name}: 有效季节周期不足，跳过 MSTL 分解")
            continue

        mstl_result, valid_periods = decomposition_result
        mstl_result_map[cluster_name] = mstl_result
        all_analysis.append(analyze_mstl_result(mstl_result, cluster_name, valid_periods))

        cluster_output_dir = os.path.join(output_dir, cluster_name)
        os.makedirs(cluster_output_dir, exist_ok=True)

        plot_mstl_components(
            result=mstl_result,
            cluster_name=cluster_name,
            save_path=os.path.join(cluster_output_dir, f"{cluster_name}_MSTL分解.png"),
        )
        plot_seasonal_analysis(
            result=mstl_result,
            cluster_name=cluster_name,
            save_path=os.path.join(cluster_output_dir, f"{cluster_name}_季节性分析.png"),
        )
        safe_print(f"{cluster_name} 图表输出完成: {cluster_output_dir}")

    safe_print("步骤6/6: 输出趋势对比图和 MSTL 指标汇总...")
    if mstl_result_map:
        plot_cluster_trend_comparison(
            mstl_result_map=mstl_result_map,
            save_path=os.path.join(output_dir, "三大城市群_MSTL趋势对比.png"),
        )

    if all_analysis:
        analysis_df = pd.DataFrame(all_analysis).sort_values("城市群")
        analysis_path = os.path.join(output_dir, "三大城市群_MSTL分析汇总.csv")
        analysis_df.to_csv(analysis_path, index=False, encoding="utf-8-sig")
        safe_print(f"MSTL 分析汇总已保存: {analysis_path}")
    else:
        safe_print("未产生有效 MSTL 分析结果，请检查序列长度或数据完整性。")

    safe_print("\n分析完成。")
    safe_print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
