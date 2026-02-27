"""
SVG 智能裁剪工具

功能：
- 自动识别 SVG 中的空白边缘
- 在不破坏原图内容的前提下，最大化裁剪为严格正方形
- 输入输出均为 SVG 格式

依赖：pip install svgpathtools
"""

from pathlib import Path
from io import StringIO
import xml.etree.ElementTree as ET
import re
from typing import Tuple, Optional

try:
    from svgpathtools import svg2paths
except ImportError:
    raise ImportError("请先安装 svgpathtools: pip install svgpathtools")


# ============ 在此直接配置要裁切的路径 ============
INPUT_FOLDER = r"H:\DATA Science\大论文Result\大论文图\三大城市群\三大城市群_（年）莫兰结果\China_combined_yearly_maps"  # 输入文件夹   
OUTPUT_FOLDER = r"H:\DATA Science\大论文Result\大论文图\三大城市群\三大城市群_（年）莫兰结果\China_combined_yearly_maps_cropped"  # 输出文件夹，None 表示与输入相同（覆盖原文件）
PADDING = 0 # 内边距比例，0 表示无内边距
# ================================================


# SVG 命名空间
SVG_NS = "http://www.w3.org/2000/svg"
NS_MAP = {"svg": SVG_NS}


def _parse_viewbox(svg_elem: ET.Element) -> Tuple[float, float, float, float]:
    """解析 SVG 的 viewBox，返回 (x, y, width, height)。"""
    viewbox = svg_elem.get("viewBox") or svg_elem.get("{http://www.w3.org/2000/svg}viewBox")
    if not viewbox:
        width = _parse_length(svg_elem.get("width", "100"))
        height = _parse_length(svg_elem.get("height", "100"))
        return 0.0, 0.0, width, height
    parts = [float(x) for x in viewbox.strip().replace(",", " ").split()]
    if len(parts) != 4:
        return 0.0, 0.0, 100.0, 100.0
    return parts[0], parts[1], parts[2], parts[3]


def _parse_length(s: str) -> float:
    """解析 SVG 长度值（如 '100pt', '100', '100px'）。"""
    if not s:
        return 100.0
    s = s.strip()
    match = re.match(r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*(pt|px|em|%|)?", s)
    if match:
        return float(match.group(1))
    try:
        return float(s)
    except ValueError:
        return 100.0


def _svg_remove_paths_without_d(svg_path: str) -> str:
    """移除没有 d 属性的 path 元素，避免 svgpathtools 解析报错。"""
    with open(svg_path, encoding="utf-8") as f:
        content = f.read()
    # 匹配自闭合的 <path ... /> 且无 d 属性，或 <path ...></path> 无 d
    # 简单策略：移除 <path ... /> 其中不包含 d=
    def remove_empty_path(m):
        tag = m.group(0)
        if " d=" in tag or " d =" in tag:
            return tag
        return ""
    content = re.sub(r"<path[^>]*/>", remove_empty_path, content)
    return content


def get_content_bbox(svg_path: str) -> Optional[Tuple[float, float, float, float]]:
    """
    获取 SVG 内容的边界框 (xmin, xmax, ymin, ymax)。
    自动排除覆盖全画布的空白背景矩形。
    """
    try:
        paths, _, svg_attrs = svg2paths(svg_path, return_svg_attributes=True)
    except KeyError:
        # 部分 path 缺少 d 属性（如空格字形），先预处理移除
        svg_str = _svg_remove_paths_without_d(svg_path)
        paths, _, svg_attrs = svg2paths(StringIO(svg_str), return_svg_attributes=True)

    if not paths:
        return None

    # 解析原始 viewBox
    vb_x, vb_y, vb_w, vb_h = 0.0, 0.0, 100.0, 100.0
    if "viewBox" in svg_attrs:
        parts = [float(x) for x in svg_attrs["viewBox"].strip().replace(",", " ").split()]
        if len(parts) == 4:
            vb_x, vb_y, vb_w, vb_h = parts[0], parts[1], parts[2], parts[3]
    vb_area = vb_w * vb_h
    if vb_area <= 0:
        vb_area = 1.0

    xmin = xmax = ymin = ymax = None

    for path in paths:
        try:
            b = path.bbox()
            if b is None:
                continue
            # svgpathtools bbox: (xmin, xmax, ymin, ymax)
            pxmin, pxmax, pymin, pymax = b
            p_area = (pxmax - pxmin) * (pymax - pymin)

            # 排除覆盖 >90% 画布的背景矩形（通常是空白背景）
            if p_area > 0.9 * vb_area:
                continue

            if xmin is None:
                xmin, xmax, ymin, ymax = pxmin, pxmax, pymin, pymax
            else:
                xmin = min(xmin, pxmin)
                xmax = max(xmax, pxmax)
                ymin = min(ymin, pymin)
                ymax = max(ymax, pymax)
        except (ValueError, TypeError):
            continue

    if xmin is None:
        return None
    return (xmin, xmax, ymin, ymax)


def crop_svg_to_square(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    padding_ratio: float = 0.0,
) -> Path:
    """
    将 SVG 裁剪为严格正方形，自动去除空白边缘。

    参数
    -----
    input_path : str | Path
        输入 SVG 文件路径
    output_path : str | Path, optional
        输出路径，默认在输入文件名后加 _cropped
    padding_ratio : float, default 0.0
        相对边长的内边距比例，0 表示无内边距，最大化裁剪

    返回
    -----
    Path
        输出文件路径
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_path = Path(output_path) if output_path else input_path.with_stem(
        f"{input_path.stem}_cropped"
    )
    output_path = output_path.with_suffix(".svg")

    # 获取内容边界
    bbox = get_content_bbox(str(input_path))

    # 解析原始 SVG
    tree = ET.parse(input_path)
    root = tree.getroot()

    # 注册命名空间，避免写入时添加 ns0 等前缀
    ET.register_namespace("", SVG_NS)
    for prefix, uri in [("svg", SVG_NS), ("xlink", "http://www.w3.org/1999/xlink")]:
        ET.register_namespace(prefix, uri)

    vb_x, vb_y, vb_w, vb_h = _parse_viewbox(root)

    if bbox is None:
        # 无法检测内容，保持原样，仅确保输出为正方形
        side = max(vb_w, vb_h)
        cx, cy = vb_x + vb_w / 2, vb_y + vb_h / 2
        new_vb_x = cx - side / 2
        new_vb_y = cy - side / 2
    else:
        xmin, xmax, ymin, ymax = bbox
        content_w = xmax - xmin
        content_h = ymax - ymin
        side = max(content_w, content_h)

        if padding_ratio > 0:
            side *= 1 + padding_ratio

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        new_vb_x = cx - side / 2
        new_vb_y = cy - side / 2

    new_viewbox = f"{new_vb_x:.6g} {new_vb_y:.6g} {side:.6g} {side:.6g}"

    # 更新根元素属性
    root.set("viewBox", new_viewbox)
    root.set("width", f"{side:.6g}pt")
    root.set("height", f"{side:.6g}pt")

    # 保持内容比例正确显示
    if root.get("preserveAspectRatio") is None:
        root.set("preserveAspectRatio", "xMidYMid meet")

    tree.write(
        str(output_path),
        encoding="utf-8",
        xml_declaration=True,
        method="xml",
    )

    return output_path


def batch_crop_svg(
    input_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    pattern: str = "*.svg",
    padding_ratio: float = 0.0,
) -> list[Path]:
    """
    批量裁剪目录下的 SVG 文件。

    参数
    -----
    input_dir : str | Path
        输入目录
    output_dir : str | Path, optional
        输出目录，默认与输入相同
    pattern : str
        文件名匹配模式，默认 "*.svg"

    返回
    -----
    list[Path]
        输出文件路径列表
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for f in input_dir.glob(pattern):
        if f.is_file():
            out = output_dir / f.name
            results.append(crop_svg_to_square(f, out, padding_ratio=padding_ratio))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SVG 智能裁剪为严格正方形")
    parser.add_argument("input", nargs="?", help="输入 SVG 文件或目录")
    parser.add_argument("-o", "--output", help="输出路径（文件或目录）")
    parser.add_argument("-b", "--batch", action="store_true", help="批量处理目录下所有 SVG")
    parser.add_argument("-p", "--padding", type=float, default=None, help="内边距比例，0~1")
    args = parser.parse_args()

    # 未传参时使用代码顶部配置的路径
    input_path = args.input or INPUT_FOLDER
    output_path = args.output if args.output is not None else OUTPUT_FOLDER
    padding = args.padding if args.padding is not None else PADDING

    if not input_path:
        parser.print_help()
        print("\n示例:")
        print("  python \"SVG crop.py\" image.svg")
        print("  python \"SVG crop.py\" image.svg -o cropped.svg")
        print("  或在代码顶部修改 INPUT_FOLDER 后直接运行")
    elif Path(input_path).is_dir():
        for p in batch_crop_svg(input_path, output_path, padding_ratio=padding):
            print(f"Saved: {p}")
    else:
        out = crop_svg_to_square(input_path, output_path, padding)
        print(f"Saved: {out}")
