"""
三大城市群统计趋势检验 —— 绘图模块
从检验结果 CSV 与聚合序列 CSV 生成所有可视化图表。
需要先运行 三大城市群统计趋势检验（新）.py 生成结果文件。
"""
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd
from scipy import stats

# =========================
# 路径配置（按需修改）
# =========================
OUTPUT_DIR = r"H:\大论文Result\大论文图\三大城市群\统计趋势检验"

# 论文字号（在 Study area 对齐基础上整体减小 4pt：标题、轴标题、轴刻度）
FONT_TITLE = 28
FONT_AXIS_LABEL = 20
FONT_TICK = 18
# 2×2 检验对比图柱顶标注：先尝试的最大/最小字号，再按柱宽像素收缩以免宽于柱子、并排重叠
FONT_BAR_COMPARISON_MAX = 9
FONT_BAR_COMPARISON_MIN = 5
FONT_NOTE = 16  # 图中说明性小字（对齐 Study area 经纬度外围标签）
FONT_SEN_SLOPE = 22  # 二号字约 22pt，用于 Sen 斜率标注框

# 图内化学式：与中文同串勿用 $...$（mathtext 会把相邻中文走 STIX，SVG 里变成 ¤）。
PM25_PLAIN = "PM\u2082.\u2085"
LABEL_UNIT_UGM3 = "(\u03bcg/m\u00b3)"

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
            fm.fontManager.addfont(str(fp))
        except (OSError, ValueError, RuntimeError):
            continue
    _WINDOWS_FONT_FILES_REGISTERED = True


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


def _thesis_font_family_list() -> List[str]:
    """西文优先 TNR，中文优先楷体；仅加入 fontManager 已登记的名称，避免无效 findfont 刷屏。

    本机若无楷体（或注册名非 KaiTi），仍会有雅黑/黑体等可显示汉字。
    """
    _register_windows_font_files_once()
    by_lower: Dict[str, str] = {}
    for info in fm.fontManager.ttflist:
        k = info.name.lower()
        if k not in by_lower:
            by_lower[k] = info.name

    want_order = [
        "Times New Roman",
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
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    seen = set()
    resolved: List[str] = []
    for name in want_order:
        got = by_lower.get(name.lower())
        if got is not None and got not in seen:
            resolved.append(got)
            seen.add(got)

    if not resolved:
        return ["DejaVu Sans"]
    if resolved[0].lower() != "times new roman":
        tnr = by_lower.get("times new roman")
        if tnr is not None:
            resolved = [tnr] + [x for x in resolved if x.lower() != "times new roman"]
        else:
            resolved = ["Times New Roman"] + [x for x in resolved if x.lower() != "times new roman"]

    has_kai = any(
        "kaiti" in x.lower() or "kai" in x.lower() or "楷" in x
        for x in resolved
    )
    if not has_kai:
        for info in fm.fontManager.ttflist:
            if "simkai" in info.fname.replace("\\", "/").lower():
                if info.name not in seen:
                    insert_at = 1 if len(resolved) > 1 else len(resolved)
                    resolved.insert(insert_at, info.name)
                    seen.add(info.name)
                break
    return resolved


def configure_thesis_fonts() -> None:
    """西文/数字用 Times New Roman，中文优先楷体，否则雅黑/黑体/宋体（按字形在列表中回退）。

    svg.fonttype=none 避免把文字烧成 path（TNR 对汉字易产生占位轮廓导致「乱码」）。
    """
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = _thesis_font_family_list()
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Times New Roman"
    mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
    mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"


def _svg_font_reorder_disabled() -> bool:
    for key in (
        "TREND_SVG_NO_FONT_REORDER",
        "STL_SVG_NO_FONT_REORDER",
        "MSTL_SVG_NO_FONT_REORDER",
        "MONTHLY_PM25_SVG_NO_FONT_REORDER",
    ):
        if os.environ.get(key, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def _patch_svg_font_family_for_weak_viewers(svg_path: str) -> None:
    """Word 等只认 font-family 首项时避免中文方框（排查指南 §2.3.1）。

    旧版使用 [^;]+ 跨行匹配：若某处 font-family 后缺少分号，会一路吞到文件后部，
    删掉中间的 </g> 等标签，浏览器报「g 与 svg 标签不匹配」。现改为单行匹配并在
    写入前做 XML 校验，失败则保留 Matplotlib 原文件。
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
        r"font-family:\s*[^\r\n;]+;",
        f"font-family: {office_first};",
        content,
    )
    if not n or patched == content:
        return
    try:
        ET.fromstring(patched)
    except ET.ParseError:
        return
    try:
        with open(svg_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(patched)
    except OSError:
        return


def _save_svg_enabled() -> bool:
    """设为 0/false/no 时跳过 SVG，仅用 PNG（Word 对 Matplotlib 导出的 SVG 支持不完整，宜插 PNG）。"""
    v = os.environ.get("TREND_SAVE_SVG", "").strip().lower()
    return v not in ("0", "false", "no")


def _save_figure_notice(output_png_path: str) -> str:
    if _save_svg_enabled():
        return f"已保存图像: {output_png_path} 及同名 SVG"
    return f"已保存图像: {output_png_path}（未写 SVG，TREND_SAVE_SVG=0）"


def save_figure_dual(fig: plt.Figure, save_path_png: str, dpi: int = 300) -> None:
    """同时保存 PNG 与 SVG（SVG 可由环境变量 TREND_SAVE_SVG=0 关闭）。"""
    fig.savefig(save_path_png, dpi=dpi, bbox_inches="tight")
    if not _save_svg_enabled():
        return
    save_path_svg = os.path.splitext(save_path_png)[0] + ".svg"
    fig.savefig(save_path_svg, format="svg", bbox_inches="tight")
    _patch_svg_font_family_for_weak_viewers(save_path_svg)


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


_GROUP_DISPLAY_LABELS: Dict[str, str] = {
    "京津冀": "BTH",
    "长三角": "YRD",
    "珠三角": "PRD",
}


def _group_display_label(group: str) -> str:
    """图例、坐标轴等展示用英文缩写；CSV 与列名仍用中文城市群名。"""
    return _GROUP_DISPLAY_LABELS.get(str(group), str(group))


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


def _comparison_bar_label_text(height: float, *, is_pvalue: bool, decimals: int) -> str:
    """柱顶字符串：P 值用有效数字以缩短宽度；斜率面板用固定小数位。"""
    if np.isnan(height):
        return ""
    h = float(height)
    if not is_pvalue:
        return f"{h:.{decimals}f}"
    sig = max(1, min(int(decimals), 6))
    return f"{h:.{sig}g}"


def _annotate_bar_values_comparison(
    ax: plt.Axes,
    bars: List[object],
    *,
    decimals: int,
    is_pvalue: bool,
) -> None:
    """检验对比图专用：小号字体 + 按柱宽收缩字号 + 成对柱子轻微上下错开，减轻并排遮盖。"""
    y_min, y_max = ax.get_ylim()
    y_range = max(y_max - y_min, 1e-9)
    y_top_margin = 0.02 * y_range
    inside_top = y_max - y_top_margin

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    half = len(bars) // 2
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if np.isnan(height):
            continue
        label = _comparison_bar_label_text(height, is_pvalue=is_pvalue, decimals=decimals)
        if not label:
            continue
        x0 = bar.get_x()
        w = bar.get_width()
        xc = x0 + w / 2
        dy_base = (0.005 if height <= 0.1 else 0.015) * y_range
        # bars 顺序为 [左组全部柱, 右组全部柱]：右组略抬高，减轻同一城市群两根柱顶数字挤碰
        stagger = (0.012 * y_range) if half > 0 and i >= half else 0.0
        raw_y = float(height) + dy_base + stagger
        text_y = min(raw_y, inside_top)
        va_style = "bottom" if text_y >= raw_y - 1e-12 else "top"

        x_pix_lo, _ = ax.transData.transform((x0, 0))
        x_pix_hi, _ = ax.transData.transform((x0 + w, 0))
        bar_px_w = max(abs(x_pix_hi - x_pix_lo), 1.0)

        fs = float(FONT_BAR_COMPARISON_MAX)
        while fs >= FONT_BAR_COMPARISON_MIN:
            t = ax.text(
                xc,
                text_y,
                label,
                ha="center",
                va=va_style,
                fontsize=fs,
                alpha=0.0,
            )
            bb = t.get_window_extent(renderer=renderer)
            t.remove()
            if bb.width <= bar_px_w * 0.88:
                break
            fs -= 0.5

        ax.text(
            xc,
            text_y,
            label,
            ha="center",
            va=va_style,
            fontsize=fs,
            clip_on=True,
        )


def _pick_annotation_offset_for_sen(
    ax: plt.Axes,
    x_val: float,
    y_val: float,
    all_line_points: np.ndarray,
) -> Tuple[int, int]:
    """仅用于 Sen 斜率标注：与突变点用的 _pick_annotation_offset 解耦，保持早期候选与打分逻辑。"""
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
        return (0, 22)
    return best_offset


# Sen 斜率标注：在「首选偏移 + 常用偏移」中尝试，保证文字+bbox 整体落在坐标轴绘图区内（非仅锚点）。
_SEN_SLOPE_OFFSET_TRY_BASES: List[Tuple[int, int]] = [
    (10, 12),
    (10, -18),
    (-56, 12),
    (-56, -18),
    (20, 24),
    (20, -28),
    (-72, 24),
    (-72, -28),
    (0, 30),
    (0, -30),
    (34, 0),
    (-70, 0),
    (0, 40),
    (0, 48),
    (-44, 30),
    (-60, 26),
    (-80, 30),
    (-36, 20),
    (-24, 34),
    (-52, 18),
    (12, 28),
    (-68, 14),
    (-90, 22),
    (6, 16),
    (-48, 8),
]


def _fig_refresh_layout(fig: plt.Figure) -> None:
    """更新布局以便测量文字窗口范围；优先用轻量 API。"""
    if hasattr(fig, "draw_without_rendering"):
        fig.draw_without_rendering()
    else:
        fig.canvas.draw()


def _sen_slope_bbox_inside_axes(
    ax: plt.Axes,
    fig: plt.Figure,
    renderer,
    x_val: float,
    y_val: float,
    dx_pt: float,
    dy_pt: float,
    text: str,
    fontsize: float,
    bbox_dict: Optional[dict] = None,
    *,
    pad_px: float = 2.5,
) -> bool:
    """Sen 斜率纯文字（可选 bbox）是否完全落在当前坐标轴窗口矩形内。"""
    ann_kw: dict = dict(
        xy=(x_val, y_val),
        xytext=(dx_pt, dy_pt),
        textcoords="offset points",
        fontsize=fontsize,
        ha="center",
        va="center",
        alpha=0.0,
        clip_on=False,
    )
    if bbox_dict is not None:
        ann_kw["bbox"] = bbox_dict
    ann = ax.annotate(text, **ann_kw)
    _fig_refresh_layout(fig)
    tb = ann.get_window_extent(renderer=renderer)
    ann.remove()
    abb = ax.get_window_extent(renderer=renderer)
    return (
        tb.x0 >= abb.x0 + pad_px
        and tb.x1 <= abb.x1 - pad_px
        and tb.y0 >= abb.y0 + pad_px
        and tb.y1 <= abb.y1 - pad_px
    )


def _pick_sen_slope_offset_inside_axes(
    ax: plt.Axes,
    fig: plt.Figure,
    renderer,
    x_val: float,
    y_val: float,
    all_line_points: np.ndarray,
    text: str,
    fontsize: float,
    bbox_dict: Optional[dict] = None,
    *,
    dy_nudge: int = 6,
) -> Tuple[int, int]:
    pref_dx, pref_dy = _pick_annotation_offset_for_sen(ax, x_val, y_val, all_line_points)
    seen: Set[Tuple[int, int]] = set()
    ordered: List[Tuple[int, int]] = []

    def add(dx: int, dy: int) -> None:
        p = (dx, dy)
        if p not in seen:
            seen.add(p)
            ordered.append(p)

    add(int(pref_dx), int(pref_dy + dy_nudge))
    for dx, dy in _SEN_SLOPE_OFFSET_TRY_BASES:
        add(int(dx), int(dy + dy_nudge))
        add(int(dx), int(dy))

    for dx, dy in ordered:
        if _sen_slope_bbox_inside_axes(
            ax, fig, renderer, x_val, y_val, float(dx), float(dy), text, fontsize, bbox_dict
        ):
            return dx, dy

    for dy in range(8, 72, 6):
        for dx in range(-96, 44, 8):
            if (dx, dy) in seen:
                continue
            if _sen_slope_bbox_inside_axes(
                ax, fig, renderer, x_val, y_val, float(dx), float(dy), text, fontsize, bbox_dict
            ):
                return dx, dy

    return 0, 24


def _xytext_offset_to_anchor_data(
    ax: plt.Axes, x_val: float, y_val: float, dx_pt: float, dy_pt: float
) -> Tuple[float, float]:
    """offset points 相对数据点 (x_val,y_val) 时，文字锚点在数据坐标系中的近似位置。"""
    fig = ax.figure
    scale = fig.dpi / 72.0
    xd, yd = ax.transData.transform((x_val, y_val))
    return tuple(ax.transData.inverted().transform((xd + dx_pt * scale, yd + dy_pt * scale)))


# 突变点标注排序：优先把文字框放到离折线更远的位置；框与框避让次之（避免为互让而压回折线附近）
_CP_ANNOT_LINE_CLEAR_WEIGHT = 12.0
_CP_ANNOT_FROM_POINT_WEIGHT = 0.15
_CP_ANNOT_SEP_WEIGHT = 2.0


def _annotation_anchor_separation_penalty(
    anchor_x: float,
    anchor_y: float,
    placed: List[Tuple[float, float]],
    x_range: float,
    y_range: float,
    min_sep_x_frac: float = 0.22,
    min_sep_y_frac: float = 0.14,
) -> float:
    """已放置标注锚点过近时增大惩罚，使多框错开（同季多线时主要靠纵向/斜向分离）。"""
    if not placed:
        return 0.0
    min_sx = max(min_sep_x_frac * x_range, 1e-6)
    min_sy = max(min_sep_y_frac * y_range, 1e-6)
    pen = 0.0
    for px, py in placed:
        rx = abs(anchor_x - px) / min_sx
        ry = abs(anchor_y - py) / min_sy
        if rx < 1.0 and ry < 1.0:
            pen += (1.0 - rx) ** 2 + (1.0 - ry) ** 2
    return pen


def _iter_change_point_annotation_candidates(group_for_bias: Optional[str]) -> List[Tuple[int, int]]:
    base_candidates = [
        (10, 12), (10, -18), (-56, 12), (-56, -18),
        (20, 24), (20, -28), (-72, 24), (-72, -28),
        (0, 30), (0, -30), (34, 0), (-70, 0),
        (14, 42), (-68, 38), (28, -36), (-80, -32),
        (0, 48), (0, -42), (88, 22), (-92, 20), (72, -28), (-78, -26),
        (0, 58), (0, -52), (0, 68), (0, -62),
        (105, 18), (-105, 18), (95, -24), (-95, -24),
        (44, 52), (-44, 52), (44, -48), (-44, -48),
    ]
    group_bias: Dict[str, List[Tuple[int, int]]] = {
        "京津冀": [(16, 34), (-62, 30), (24, -32), (-76, -24), (0, 46)],
        "长三角": [(-68, 32), (12, 38), (-80, -28), (22, -30), (-92, 18)],
        "珠三角": [(0, -38), (18, 36), (-58, -34), (32, 28), (0, 50)],
    }
    preferred = group_bias.get(str(group_for_bias or ""), [])
    seen: Set[Tuple[int, int]] = set()
    out: List[Tuple[int, int]] = []
    for t in preferred + base_candidates:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _rank_change_point_annotation_offsets(
    ax: plt.Axes,
    x_val: float,
    y_val: float,
    all_line_points: np.ndarray,
    *,
    placed_text_anchors: Optional[List[Tuple[float, float]]] = None,
    group_for_bias: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """与 _pick_annotation_offset 相同打分，按分数从高到低排序；同分保留候选先后顺序。"""
    candidate_offsets = _iter_change_point_annotation_candidates(group_for_bias)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    scored: List[Tuple[float, int, Tuple[int, int]]] = []
    for i, (dx, dy) in enumerate(candidate_offsets):
        anchor_x, anchor_y = _xytext_offset_to_anchor_data(ax, x_val, y_val, float(dx), float(dy))
        inside_axes = (x_min <= anchor_x <= x_max) and (y_min <= anchor_y <= y_max)
        if not inside_axes:
            continue

        if all_line_points.size > 0:
            norm_dist2 = (
                ((all_line_points[:, 0] - anchor_x) / x_range) ** 2
                + ((all_line_points[:, 1] - anchor_y) / y_range) ** 2
            )
            min_dist_score = float(np.min(norm_dist2))
        else:
            min_dist_score = 1.0

        from_point_score = float(((anchor_x - x_val) / x_range) ** 2 + ((anchor_y - y_val) / y_range) ** 2)
        overlap_pen = _annotation_anchor_separation_penalty(
            anchor_x, anchor_y, placed_text_anchors or [], x_range, y_range
        )
        score = (
            _CP_ANNOT_LINE_CLEAR_WEIGHT * min_dist_score
            + _CP_ANNOT_FROM_POINT_WEIGHT * from_point_score
            - _CP_ANNOT_SEP_WEIGHT * overlap_pen
        )
        scored.append((score, i, (dx, dy)))

    if not scored:
        return [(-56, -18)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [off for _, _, off in scored]


def _pick_annotation_offset(
    ax: plt.Axes,
    x_val: float,
    y_val: float,
    all_line_points: np.ndarray,
    *,
    placed_text_anchors: Optional[List[Tuple[float, float]]] = None,
    group_for_bias: Optional[str] = None,
) -> Tuple[int, int]:
    """在多个候选位置中选择更不遮挡折线的标注偏移（单位：points）。

    placed_text_anchors: 已画出的突变点标注锚点（数据坐标），用于减轻方框互遮。
    group_for_bias: 城市群名，用于优先尝试不同象限的偏移，同季多点时更易散开。
    """
    ranked = _rank_change_point_annotation_offsets(
        ax,
        x_val,
        y_val,
        all_line_points,
        placed_text_anchors=placed_text_anchors,
        group_for_bias=group_for_bias,
    )
    return ranked[0]


def _display_bboxes_overlap(
    a: mpl.transforms.BboxBase,
    b: mpl.transforms.BboxBase,
    pad_px: float,
) -> bool:
    """屏幕像素坐标下，两矩形外扩 pad 后是否相交。"""
    return not (
        a.x1 + pad_px < b.x0 - pad_px
        or b.x1 + pad_px < a.x0 - pad_px
        or a.y1 + pad_px < b.y0 - pad_px
        or b.y1 + pad_px < a.y0 - pad_px
    )


def _point_axis_aligned_rect_dist_sq(px: float, py: float, bbox: mpl.transforms.BboxBase) -> float:
    """点到轴对齐矩形边界的距离平方；在矩形内部时为 0。"""
    qx = float(np.clip(px, bbox.x0, bbox.x1))
    qy = float(np.clip(py, bbox.y0, bbox.y1))
    dx = px - qx
    dy = py - qy
    return dx * dx + dy * dy


def _cp_line_segments_display(
    ax: plt.Axes, group_line_data: Dict[str, dict]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """三条城市群折线的线段端点（显示坐标 / 像素），用于判断注释框是否与折线重叠。"""
    segs: List[Tuple[np.ndarray, np.ndarray]] = []
    for gd in group_line_data.values():
        x_pos = gd["x_pos"]
        y = gd["y"]
        if len(x_pos) < 2:
            continue
        for i in range(len(x_pos) - 1):
            a = np.asarray(ax.transData.transform((float(x_pos[i]), float(y[i]))), dtype=float).ravel()[:2]
            b = np.asarray(ax.transData.transform((float(x_pos[i + 1]), float(y[i + 1]))), dtype=float).ravel()[:2]
            segs.append((a, b))
    return segs


def _cp_estimate_ann_pixel_size(text: str, fontsize: float) -> Tuple[float, float]:
    """粗估注释框在屏幕上的宽高（像素），略放大以包住中文与 round bbox。"""
    lines = text.split("\n")
    n = max(len(lines), 1)
    max_chars = max((len(line) for line in lines), default=1)
    pad = fontsize * 0.38
    w = max_chars * fontsize * 0.74 + 2 * pad
    h = n * fontsize * 1.22 + 2 * pad
    return float(w), float(h)


def _cp_synthetic_bbox_axes_frac(
    ax: plt.Axes, fx: float, fy: float, w_px: float, h_px: float
) -> mpl.transforms.Bbox:
    ctr = ax.transAxes.transform(np.array([[fx, fy]], dtype=float))[0]
    cx, cy = float(ctr[0]), float(ctr[1])
    hw, hh = w_px * 0.5, h_px * 0.5
    return mpl.transforms.Bbox.from_extents(cx - hw, cy - hh, cx + hw, cy + hh)


def _cp_line_swath_hits_bbox(
    bbox: mpl.transforms.BboxBase,
    p0: np.ndarray,
    p1: np.ndarray,
    half_width_px: float,
    extra_pad_px: float,
) -> bool:
    """折线视作有宽度的条带：若与矩形 bbox 距离小于 half_width+pad 则视为压住。"""
    thresh_sq = (half_width_px + extra_pad_px) ** 2
    leng = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    n = max(6, int(leng / 3.0) + 1)
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        px = p0[0] + t * (p1[0] - p0[0])
        py = p0[1] + t * (p1[1] - p0[1])
        if _point_axis_aligned_rect_dist_sq(px, py, bbox) <= thresh_sq:
            return True
    return False


def _cp_min_clearance_line_to_bbox(
    bbox: mpl.transforms.BboxBase,
    segs: List[Tuple[np.ndarray, np.ndarray]],
) -> float:
    """线段上采样点到注释框边界的最近距离（像素），用于在多个空白候选间择优。"""
    best = float("inf")
    for p0, p1 in segs:
        leng = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        n = max(5, int(leng / 2.5) + 1)
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.0
            px = p0[0] + t * (p1[0] - p0[0])
            py = p0[1] + t * (p1[1] - p0[1])
            d = float(np.sqrt(_point_axis_aligned_rect_dist_sq(px, py, bbox)))
            best = min(best, d)
    return best


def _place_change_point_annotations_bbox_avoid(
    ax: plt.Axes,
    fig: plt.Figure,
    cp_tasks: List[Dict[str, object]],
    all_line_points: np.ndarray,
    group_line_data: Dict[str, dict],
    *,
    pad_px: float = 8.0,
) -> None:
    """在坐标轴内用「空白网格」统一找注释位置，箭头连到突变星标。

    原先用 offset points + 锚点到数据点的距离打分，容易压线的原因主要是：
    1) 只比到折线「顶点」的距离，不比到线段，斜线段中间可能穿过文字框；
    2) 打分用的是文字锚点，不是带 bbox 的整块矩形；
    3) 多框避让时被迫选回折线密集区。

    现改为：在 axes fraction 平面上枚举候选，用像素空间线段条带与矩形相交判定，
    并避开图例框；通过后以真实 renderer 测量 bbox 再校验一次。
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    segs = _cp_line_segments_display(ax, group_line_data)
    halfw_px = 2.0 * (fig.dpi / 72.0) * 0.5
    line_pad_px = 5.0

    leg = ax.get_legend()
    leg_bbox = leg.get_window_extent(renderer).expanded(1.16, 1.16) if leg is not None else None

    frac_x = np.linspace(0.10, 0.90, 19)
    frac_y = np.linspace(0.12, 0.90, 17)

    placed_bboxes: List[mpl.transforms.BboxBase] = []
    placed_anchors: List[Tuple[float, float]] = []

    for t in cp_tasks:
        group = str(t["group"])
        x_pos = float(t["x_pos"])
        y_val = float(t["y_val"])
        color = group_line_data[group]["color"]
        text = str(t["ann"])
        w_px, h_px = _cp_estimate_ann_pixel_size(text, float(FONT_NOTE))
        w_px *= 1.1
        h_px *= 1.1

        candidates: List[Tuple[float, float, float]] = []
        for fx in frac_x:
            for fy in frac_y:
                syn = _cp_synthetic_bbox_axes_frac(ax, float(fx), float(fy), w_px, h_px)
                if leg_bbox is not None and syn.overlaps(leg_bbox):
                    continue
                if any(_display_bboxes_overlap(syn, pb, pad_px) for pb in placed_bboxes):
                    continue
                hit = False
                for p0, p1 in segs:
                    if _cp_line_swath_hits_bbox(syn, p0, p1, halfw_px, line_pad_px):
                        hit = True
                        break
                if hit:
                    continue
                clr = _cp_min_clearance_line_to_bbox(syn, segs)
                star = ax.transData.transform(np.array([[x_pos, y_val]], dtype=float))[0]
                ctr = ax.transAxes.transform(np.array([[fx, fy]], dtype=float))[0]
                alen = float(np.hypot(star[0] - ctr[0], star[1] - ctr[1]))
                score = clr - 0.0018 * alen
                candidates.append((score, float(fx), float(fy)))

        candidates.sort(key=lambda c: -c[0])
        bbox_style = dict(
            boxstyle="round,pad=0.18",
            fc="white",
            ec=color,
            alpha=0.55,
        )
        placed_this = False
        for _sc, fx, fy in candidates:
            ann = ax.annotate(
                text,
                xy=(x_pos, y_val),
                xycoords="data",
                xytext=(fx, fy),
                textcoords="axes fraction",
                ha="center",
                va="center",
                fontsize=FONT_NOTE,
                color=color,
                bbox=bbox_style,
                arrowprops=dict(arrowstyle="-", color=color, lw=1.0, shrinkA=3, shrinkB=4),
                zorder=9,
                clip_on=False,
            )
            fig.canvas.draw()
            ext = ann.get_window_extent(renderer)
            bad = False
            if leg_bbox is not None and ext.overlaps(leg_bbox):
                bad = True
            if not bad:
                for p0, p1 in segs:
                    if _cp_line_swath_hits_bbox(ext, p0, p1, halfw_px, line_pad_px * 0.9):
                        bad = True
                        break
            if not bad and any(_display_bboxes_overlap(ext, pb, pad_px) for pb in placed_bboxes):
                bad = True
            if not bad:
                placed_bboxes.append(ext)
                ctr_d = ax.transAxes.transform(np.array([[fx, fy]], dtype=float))[0]
                ax_d = ax.transData.inverted().transform(ctr_d)
                placed_anchors.append((float(ax_d[0]), float(ax_d[1])))
                placed_this = True
                break
            ann.remove()

        if not placed_this:
            ranked = _rank_change_point_annotation_offsets(
                ax,
                x_pos,
                y_val,
                all_line_points,
                placed_text_anchors=placed_anchors,
                group_for_bias=group,
            )
            dx, dy = ranked[0]
            ann = ax.annotate(
                text,
                xy=(x_pos, y_val),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=FONT_NOTE,
                color=color,
                bbox=bbox_style,
            )
            fig.canvas.draw()
            placed_bboxes.append(ann.get_window_extent(renderer))
            placed_anchors.append(_xytext_offset_to_anchor_data(ax, x_pos, y_val, float(dx), float(dy)))


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
    change_point_bbox_avoid: bool = False,
    change_point_annotation_text: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6))
    group_line_data = {}
    cp_annotate_bbox_deferred: Optional[Tuple[List[Dict[str, object]], np.ndarray]] = None
    trend_line_points: List[np.ndarray] = []
    all_line_points_for_sen = np.empty((0, 2), dtype=float)
    for group in _sort_groups(list(group_series.keys())):
        s = group_series[group]
        x_labels = [str(i) for i in s.index.tolist()]
        x_pos = np.arange(len(x_labels))
        y = s.values.astype(float)
        line, = ax.plot(x_pos, y, marker="o", linewidth=2, label=_group_display_label(group))
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
                label=f"{_group_display_label(group)} trend line",
            )
            trend_line_points.append(np.column_stack((x_pos, trend_y)))

        all_line_points_for_sen = np.concatenate(
            [
                np.column_stack((line_data["x_pos"], line_data["y"]))
                for line_data in group_line_data.values()
            ] + trend_line_points,
            axis=0,
        ) if group_line_data else np.empty((0, 2), dtype=float)

    if annotate_change_points and sliding_t_df is not None and not sliding_t_df.empty:
        all_line_points = np.concatenate(
            [
                np.column_stack((line_data["x_pos"], line_data["y"]))
                for line_data in group_line_data.values()
            ],
            axis=0,
        ) if group_line_data else np.empty((0, 2), dtype=float)
        used_group_point_labels = set()
        _group_rank = {"京津冀": 0, "长三角": 1, "珠三角": 2}
        cp_tasks: List[Dict[str, object]] = []
        for _, row in sliding_t_df.iterrows():
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
            x_pos = float(group_line_data[group]["x_pos"][change_idx])
            y_val = float(group_line_data[group]["y"][change_idx])
            point_label = f"{_group_display_label(group)}突变点"
            legend_label = point_label if point_label not in used_group_point_labels else None
            used_group_point_labels.add(point_label)
            p_val = pd.to_numeric(row.get("P_Value"), errors="coerce")
            ann_lines = [str(change_label), f"|t|={t_stat:.2f}"]
            if pd.notna(p_val):
                ann_lines.append(f"p={p_val:.3f}")
            if not bool(row.get("Significant", False)):
                ann_lines.append("(α=0.05未显著)")
            cp_tasks.append(
                {
                    "group": group,
                    "x_pos": x_pos,
                    "y_val": y_val,
                    "ann": "\n".join(ann_lines),
                    "legend_label": legend_label,
                    "change_idx": change_idx,
                }
            )

        cp_tasks.sort(key=lambda d: (int(d["change_idx"]), _group_rank.get(str(d["group"]), 99)))

        if change_point_bbox_avoid and change_point_annotation_text and cp_tasks:
            cp_annotate_bbox_deferred = (cp_tasks, all_line_points)

        placed_cp_anchors: List[Tuple[float, float]] = []
        for t in cp_tasks:
            group = str(t["group"])
            x_pos = float(t["x_pos"])
            y_val = float(t["y_val"])
            color = group_line_data[group]["color"]
            ax.scatter(
                x_pos,
                y_val,
                marker="*",
                s=120,
                color=color,
                edgecolors="black",
                linewidths=0.8,
                zorder=7,
                label=t["legend_label"],
            )
            if not change_point_annotation_text:
                continue
            if cp_annotate_bbox_deferred is not None:
                continue
            text_dx, text_dy = _pick_annotation_offset(
                ax,
                x_pos,
                y_val,
                all_line_points,
                placed_text_anchors=placed_cp_anchors,
                group_for_bias=group,
            )
            ax.annotate(
                str(t["ann"]),
                xy=(x_pos, y_val),
                xytext=(text_dx, text_dy),
                textcoords="offset points",
                fontsize=FONT_NOTE,
                color=color,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    fc="white",
                    ec=color,
                    alpha=0.55,
                ),
            )
            placed_cp_anchors.append(_xytext_offset_to_anchor_data(ax, x_pos, y_val, float(text_dx), float(text_dy)))

    if group_line_data:
        ref_group = _sort_groups(list(group_line_data.keys()))[0]
        ref_x = group_line_data[ref_group]["x_pos"]
        ref_labels = group_line_data[ref_group]["x_labels"]
        tick_pos, tick_labels = _choose_sparse_ticks(ref_x, ref_labels, max_ticks=18)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)

    if title:
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=FONT_AXIS_LABEL, fontweight="bold")
    ax.set_ylabel(PM25_PLAIN  + LABEL_UNIT_UGM3, fontsize=FONT_AXIS_LABEL, fontweight="bold")
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(alpha=0.3, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), fontsize=10, loc="best", frameon=False)
    plt.tight_layout()
    if cp_annotate_bbox_deferred is not None and change_point_annotation_text:
        _tasks, _lines = cp_annotate_bbox_deferred
        _place_change_point_annotations_bbox_avoid(ax, fig, _tasks, _lines, group_line_data)
    if annotate_sen_slope and annual_mk_df is not None and group_line_data:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
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
            label = f"Sen斜率: {slope:.2f}"
            text_dx, text_dy = _pick_sen_slope_offset_inside_axes(
                ax,
                fig,
                renderer,
                float(x_pos[-1]),
                float(trend_end_y),
                all_line_points_for_sen,
                label,
                FONT_SEN_SLOPE,
                dy_nudge=6,
            )
            ax.annotate(
                label,
                xy=(x_pos[-1], trend_end_y),
                xytext=(text_dx, text_dy),
                textcoords="offset points",
                color=line_data["color"],
                fontsize=FONT_SEN_SLOPE,
                ha="center",
                va="center",
            )

    save_figure_dual(fig, output_png_path, dpi=300)
    plt.close(fig)
    safe_print(_save_figure_notice(output_png_path))


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
    width = 0.26
    alpha_line = 0.05

    annual_mk = annual_results["annual_mann_kendall"].set_index("城市群")
    annual_lr = annual_results["annual_linear_regression"].set_index("城市群")
    annual_pt = annual_results["annual_pettitt"].set_index("城市群")
    annual_st = annual_results["annual_sliding_t"].set_index("城市群")
    monthly_pt = monthly_results["month_pettitt"].set_index("城市群")
    monthly_st = monthly_results["month_sliding_t"].set_index("城市群")

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10))

    ax1 = axes[0, 0]
    mk_vals = np.array([pd.to_numeric(annual_mk.loc[g, "P_Value"], errors="coerce") for g in groups], dtype=float)
    lr_vals = np.array([pd.to_numeric(annual_lr.loc[g, "P_Value"], errors="coerce") for g in groups], dtype=float)
    bars1 = ax1.bar(x - width / 2, mk_vals, width, label="Mann-Kendall检验", color="#e76f51", alpha=0.82)
    bars2 = ax1.bar(x + width / 2, lr_vals, width, label="线性回归检验", color="#f2a8a0", alpha=0.82)
    ax1.axhline(alpha_line, color="#d62828", linestyle="--", linewidth=1.5, label="α=0.05")
    ax1.set_title("年度趋势检验P值对比", fontsize=FONT_TITLE - 2, fontweight="bold")
    ax1.set_ylabel("P值", fontsize=FONT_AXIS_LABEL, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([_group_display_label(g) for g in groups])
    ax1.tick_params(axis="both", labelsize=FONT_TICK)
    ax1.grid(axis="y", alpha=0.25, linestyle="--")
    _annotate_bar_values_comparison(ax1, list(bars1) + list(bars2), decimals=3, is_pvalue=True)
    ax1.legend(fontsize=13, loc="upper right", bbox_to_anchor=(0.98, 0.75), frameon=False)

    ax2 = axes[0, 1]
    apt_vals = np.array([pd.to_numeric(annual_pt.loc[g, "P_Value"], errors="coerce") for g in groups], dtype=float)
    ast_vals = np.array([pd.to_numeric(annual_st.loc[g, "P_Value"], errors="coerce") for g in groups], dtype=float)
    bars3 = ax2.bar(x - width / 2, apt_vals, width, label="Pettitt检验", color="#4ea8de", alpha=0.82)
    bars4 = ax2.bar(x + width / 2, ast_vals, width, label="滑动T检验", color="#9ecae1", alpha=0.82)
    ax2.axhline(alpha_line, color="#d62828", linestyle="--", linewidth=1.5, label="α=0.05")
    ax2.set_title("年度突变检验P值对比", fontsize=FONT_TITLE - 2, fontweight="bold")
    ax2.set_ylabel("P值", fontsize=FONT_AXIS_LABEL, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([_group_display_label(g) for g in groups])
    ax2.tick_params(axis="both", labelsize=FONT_TICK)
    ax2.grid(axis="y", alpha=0.25, linestyle="--")
    ax2.set_ylim(0, 0.4)
    _annotate_bar_values_comparison(ax2, list(bars3) + list(bars4), decimals=3, is_pvalue=True)
    ax2.legend(fontsize=13, loc="upper right", frameon=False)

    ax3 = axes[1, 0]
    mpt_vals = np.array([pd.to_numeric(monthly_pt.loc[g, "P_Value"], errors="coerce") for g in groups], dtype=float)
    mst_vals = np.array([pd.to_numeric(monthly_st.loc[g, "P_Value"], errors="coerce") for g in groups], dtype=float)
    bars5 = ax3.bar(x - width / 2, mpt_vals, width, label="Pettitt检验", color="#52b788", alpha=0.85)
    bars6 = ax3.bar(x + width / 2, mst_vals, width, label="滑动T检验", color="#95d5b2", alpha=0.85)
    ax3.axhline(alpha_line, color="#d62828", linestyle="--", linewidth=1.5, label="α=0.05")
    ax3.set_title("月度突变检验P值对比", fontsize=FONT_TITLE - 2, fontweight="bold")
    ax3.set_ylabel("P值", fontsize=FONT_AXIS_LABEL, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([_group_display_label(g) for g in groups])
    ax3.tick_params(axis="both", labelsize=FONT_TICK)
    ax3.grid(axis="y", alpha=0.25, linestyle="--")
    _annotate_bar_values_comparison(ax3, list(bars5) + list(bars6), decimals=4, is_pvalue=True)
    ax3.legend(fontsize=13, loc="center right", bbox_to_anchor=(0.98, 0.6), frameon=False)

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
    ax4.set_title("年度" + PM25_PLAIN + "减少速率对比", fontsize=FONT_TITLE - 2, fontweight="bold")
    ax4.set_ylabel(
        PM25_PLAIN + "减少速率 " + LABEL_UNIT_UGM3 + "/年",
        fontsize=FONT_AXIS_LABEL,
        fontweight="bold",
    )
    ax4.set_xticks(x)
    ax4.set_xticklabels([_group_display_label(g) for g in groups])
    ax4.tick_params(axis="both", labelsize=FONT_TICK)
    ax4.grid(axis="y", alpha=0.25, linestyle="--")
    ax4.set_ylim(0, 5)
    _annotate_bar_values_comparison(ax4, list(bars7) + list(bars8), decimals=2, is_pvalue=False)
    ax4.legend(fontsize=13, loc="upper right", frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    save_figure_dual(fig, output_png_path, dpi=300)
    plt.close(fig)
    safe_print(_save_figure_notice(output_png_path))


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
    """用于绘图标注的滑动T：返回 |t| 最大位置；显著性仅写入 Significant，作图是否标注由 plot 决定。"""
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

    configure_thesis_fonts()

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
        "三大城市群年度" + PM25_PLAIN + "浓度变化",
        "Year",
        os.path.join(OUTPUT_DIR, "三大城市群_年度PM2.5时序.png"),
        annual_mk_df=annual_results["annual_mann_kendall"],
        annotate_sen_slope=True,
    )
    plot_group_series(
        monthly_group_series,
        "三大城市群月均" + PM25_PLAIN + "浓度变化（2018-2023）",
        "Month",
        os.path.join(OUTPUT_DIR, "三大城市群_月度PM2.5时序.png"),
    )
    plot_group_series(
        seasonal_group_series,
        "三大城市群季度" + PM25_PLAIN + "浓度变化（由月均聚合）",
        "Season",
        os.path.join(OUTPUT_DIR, "三大城市群_季度PM2.5时序.png"),
        pettitt_df=seasonal_results["season_pettitt"],
        sliding_t_df=seasonal_plot_change_df,
        annotate_change_points=True,
        change_point_bbox_avoid=True,
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
