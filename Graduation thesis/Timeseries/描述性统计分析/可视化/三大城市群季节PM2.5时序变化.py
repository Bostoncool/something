import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

# 图内 PM₂.₅：Unicode 下标，勿与中文同串使用 mathtext 的 $...$（《中文字体显示问题排查指南》§2.4）
PM25_UNICODE = "PM\u2082.\u2085"

SEASON_ORDER = {
    "春": 1,
    "夏": 2,
    "秋": 3,
    "冬": 4,
    "spring": 1,
    "summer": 2,
    "autumn": 3,
    "fall": 3,
    "winter": 4,
    "q1": 1,
    "q2": 2,
    "q3": 3,
    "q4": 4,
}

# 相对此前版式整体缩小 4 号（与 Study area.py 对齐后再减 4）
TITLE_FS = 28
AXIS_LABEL_FS = 20
TICK_FS = 18
LEGEND_FS = 18


def safe_print(message: str) -> None:
    """在终端编码不支持中文时，降级输出。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


def normalize_season_token(raw_token: str) -> str | None:
    """将季节标识归一化为 春/夏/秋/冬。"""
    token = str(raw_token).strip().lower()
    mapping = {
        "春": "春",
        "夏": "夏",
        "秋": "秋",
        "冬": "冬",
        "spring": "春",
        "summer": "夏",
        "autumn": "秋",
        "fall": "秋",
        "winter": "冬",
        "q1": "春",
        "q2": "夏",
        "q3": "秋",
        "q4": "冬",
    }
    return mapping.get(token)


def parse_season_label(label: str) -> tuple[int, int, str] | None:
    """
    解析季度列名，返回 (year, season_order, normalized_label)。
    支持示例：2018-春, 2018-冬, 2018-Q1, 2018-spring。
    """
    text = str(label).strip()

    match = re.match(r"^\s*(\d{4})\s*[-_/]\s*([A-Za-z0-9\u4e00-\u9fff]+)\s*$", text)
    if not match:
        return None

    year = int(match.group(1))
    season_token = normalize_season_token(match.group(2))
    if season_token is None:
        return None

    order = SEASON_ORDER[season_token]
    normalized_label = f"{year}-{season_token}"
    return year, order, normalized_label


def sort_season_index_labels(labels: list[str]) -> list[str]:
    """按年份与季节顺序排序季度标签。"""
    items: list[tuple[int, int, str]] = []
    for lab in labels:
        p = parse_season_label(lab)
        if p is not None:
            items.append((p[0], p[1], lab))
        else:
            items.append((9999, 99, lab))
    items.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in items]


def build_cluster_seasonal_mean_series(csv_path: str, pollutant: str = "PM2.5") -> pd.Series:
    """
    读取单个城市群的 Seasonal_Means.csv，筛选 PM2.5，并计算该城市群季节均值时序。
    返回索引为季度标签（YYYY-春/夏/秋/冬）的 Series。
    """
    df = pd.read_csv(csv_path).copy()
    required_cols = {"City", "Pollutant"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{csv_path} 缺少必要列：{required_cols}")

    df = df.loc[df["Pollutant"].astype(str).str.strip() == pollutant].copy()
    if df.empty:
        raise ValueError(f"{csv_path} 中未找到污染物 {pollutant} 的记录。")

    candidate_cols = [col for col in df.columns if col not in ("City", "Pollutant")]
    parsed = []
    for col in candidate_cols:
        parsed_info = parse_season_label(col)
        if parsed_info is not None:
            parsed.append((col, *parsed_info))

    if not parsed:
        raise ValueError(f"{csv_path} 中未识别到可用季节列。")

    parsed.sort(key=lambda x: (x[1], x[2]))  # year, season_order
    season_cols = [item[0] for item in parsed]
    season_labels = [item[3] for item in parsed]

    season_df = df[season_cols].apply(pd.to_numeric, errors="coerce")
    seasonal_mean = season_df.mean(axis=0)
    seasonal_mean.index = season_labels
    seasonal_mean.name = "PM2.5"
    return seasonal_mean


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
            font_manager.FontProperties(family=family),
            fallback_to_default=False,
        )
    except ValueError:
        return False
    return bool(path and os.path.isfile(path))


def _build_thesis_serif_font_chain() -> list[str]:
    """与 MSTL / 月均 PM2.5 / AQI 脚本一致：TNR + 楷体链；须为列表才按字形回退。"""
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
        for x in (
            "KaiTi",
            "STKaiti",
            "DFKai-SB",
            "FZKai-Z03",
            "楷体",
            "Kaiti SC",
            "SimKai",
            "KaiTi_GB2312",
        )
    ):
        for font in mpl.font_manager.fontManager.ttflist:
            if "simkai" in font.fname.replace("\\", "/").lower():
                if font.name not in ordered:
                    insert_at = 1 if len(ordered) > 1 else len(ordered)
                    ordered.insert(insert_at, font.name)
                break

    append_if_new("DejaVu Sans")
    return ordered if ordered else ["DejaVu Sans"]


def configure_plot_fonts() -> None:
    """svg.fonttype=none + font.family 列表，避免 SVG path 伪轮廓与整段绑 TNR 致中文方框（排查指南 §2.3、§6.2）。"""
    serif_chain = _build_thesis_serif_font_chain()
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = serif_chain
    mpl.rcParams["font.serif"] = serif_chain
    mpl.rcParams["axes.unicode_minus"] = False


def _svg_font_reorder_disabled() -> bool:
    for key in (
        "SEASONAL_PM25_SVG_NO_FONT_REORDER",
        "MONTHLY_PM25_SVG_NO_FONT_REORDER",
        "STL_SVG_NO_FONT_REORDER",
        "MSTL_SVG_NO_FONT_REORDER",
    ):
        if os.environ.get(key, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def _patch_svg_font_family_for_weak_viewers(svg_path: str) -> None:
    """Word 等只认 font-family 首项时，将中文字体提前（排查指南 §2.3.1）。"""
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


def plot_combined_cluster_seasonal_series(
    series_by_cluster: dict[str, pd.Series],
    color_map: dict[str, str],
    output_path: Path,
) -> tuple[Path, Path]:
    """将 BTH / YRD / PRD 三条季节均值曲线绘制在同一坐标系内；返回 (svg 路径, png 路径)。"""
    all_labels: set[str] = set()
    for s in series_by_cluster.values():
        all_labels.update(s.index)
    sorted_labels = sort_season_index_labels(list(all_labels))
    x_positions = np.arange(len(sorted_labels))

    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=150)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    legend_prop = font_manager.FontProperties(family="Times New Roman", size=LEGEND_FS)

    for cluster_name in ("BTH", "YRD", "PRD"):
        s = series_by_cluster[cluster_name].reindex(sorted_labels)
        ax.plot(
            x_positions,
            s.values.astype(float),
            color=color_map[cluster_name],
            linewidth=2.0,
            marker="o",
            markersize=3.2,
            alpha=0.95,
            zorder=3,
            label=cluster_name,
        )

    ax.set_title(
        f"三大城市群季节平均{PM25_UNICODE}时序变化",
        fontsize=TITLE_FS,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Season", fontsize=AXIS_LABEL_FS, fontweight="bold")
    # 单位用 Unicode，整段无 $，避免 STIX / ¤（排查指南 §2.4、§6.4）
    ax.set_ylabel(
        f"{PM25_UNICODE} /(\u03bcg/m\u00b3)",
        fontsize=AXIS_LABEL_FS,
        fontweight="bold",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_labels, rotation=45, ha="right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", width=1.0, length=5, labelsize=TICK_FS)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper right", frameon=True, fancybox=True, prop=legend_prop)

    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, transparent=True, bbox_inches="tight")
    fig.savefig(output_path, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    _patch_svg_font_family_for_weak_viewers(str(output_path))
    return output_path, png_path


def main() -> None:
    csv_paths = {
        "BTH": r"H:\大论文Result\BTH\描述性统计分析\Seasonal_Means.csv",
        "YRD": r"H:\大论文Result\YRD\描述性统计分析\Seasonal_Means.csv",
        "PRD": r"H:\大论文Result\PRD\描述性统计分析\Seasonal_Means.csv",
    }

    configure_plot_fonts()

    color_map = {
        "BTH": "#d62828",
        "YRD": "#1d3557",
        "PRD": "#2a9d8f",
    }

    output_dir = Path(__file__).resolve().parent

    series_by_cluster: dict[str, pd.Series] = {}
    for cluster_name, csv_path in csv_paths.items():
        series_by_cluster[cluster_name] = build_cluster_seasonal_mean_series(
            csv_path, pollutant="PM2.5"
        )
        output_data_path = output_dir / f"{cluster_name}_季节平均PM2.5序列.csv"
        series_by_cluster[cluster_name].to_frame().to_csv(
            output_data_path, encoding="utf-8-sig"
        )
        safe_print(f"{cluster_name} 数据已保存到: {output_data_path}")

    combined_svg = output_dir / "三大城市群_季节平均PM2.5时序变化_合并.svg"
    out_svg, out_png = plot_combined_cluster_seasonal_series(
        series_by_cluster=series_by_cluster,
        color_map=color_map,
        output_path=combined_svg,
    )
    safe_print(f"合并图 SVG 已保存到: {out_svg}")
    safe_print(f"合并图 PNG 已保存到: {out_png}")


if __name__ == "__main__":
    main()
