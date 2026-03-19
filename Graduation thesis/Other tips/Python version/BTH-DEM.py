"""
京津冀地区高程数据可视化 - Python 复现
基于 R 代码 BTH-DEM.R 的功能实现

用法:
    python BTH-DEM.py [数据目录]
    或设置环境变量: BTH_DEM_DATA=你的数据路径
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import rasterio
from rasterio.windows import Window
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

# 默认数据根目录，可通过环境变量 BTH_DEM_DATA 或命令行参数覆盖
DEFAULT_BASE_PATH = Path(os.environ.get("BTH_DEM_DATA", "F:/"))


def get_file_paths(base_path: Path) -> dict:
    """根据根目录生成各区域 DEM 文件路径"""
    return {
        "Beijing": base_path / "4.分城市的数据/北京市.tif",
        "Tianjin": base_path / "4.分城市的数据/天津市.tif",
        "Hebei": base_path / "3.分省份的数据/河北省/dem地形.tif",
    }


OUTPUT_FILE = "京津冀地区_DEM_统一图.png"
ELEVATION_COLORS = ["blue", "cyan", "green", "yellow", "orange", "red", "darkred"]
MAX_DISPLAY_SIZE = 8000


def downsample_for_display(
    data: np.ndarray, extent: tuple[float, float, float, float]
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """对大栅格进行下采样，避免 imshow 时内存溢出"""
    h, w = data.shape
    if h <= MAX_DISPLAY_SIZE and w <= MAX_DISPLAY_SIZE:
        return data, extent
    scale = min(MAX_DISPLAY_SIZE / h, MAX_DISPLAY_SIZE / w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    step_h, step_w = max(1, h // new_h), max(1, w // new_w)
    downsampled = data[::step_h, ::step_w]
    return downsampled, extent


def load_dem(file_path: Path) -> tuple[np.ndarray, tuple[float, float, float, float], dict, tuple[float, float, float, float]] | None:
    """读取单个 DEM 栅格，返回 (data, extent, stats, data_bounds)"""
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return None

    with rasterio.open(file_path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        bounds = src.bounds
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
        valid_mask = ~np.isnan(data)
        if valid_mask.any():
            rows, cols = np.where(valid_mask)
            w = Window(cols.min(), rows.min(), cols.max() - cols.min() + 1, rows.max() - rows.min() + 1)
            wb = src.window_bounds(w)
            data_bounds = (wb[0], wb[2], wb[1], wb[3])
        else:
            data_bounds = extent

    valid = data[~np.isnan(data)]
    stats = {
        "min": float(valid.min()) if valid.size else np.nan,
        "max": float(valid.max()) if valid.size else np.nan,
        "mean": float(valid.mean()) if valid.size else np.nan,
        "median": float(np.median(valid)) if valid.size else np.nan,
        "count": int(valid.size),
    }
    return data, extent, stats, data_bounds


def main(base_path: Path | None = None) -> None:
    """主函数：绘制京津冀地区 DEM 图"""
    plt.rcParams["font.family"] = "Times New Roman"

    files = get_file_paths(base_path or DEFAULT_BASE_PATH)
    if base_path:
        print(f"使用数据目录: {base_path}")

    dem_items = []
    all_values = []

    print("读取高程数据...")
    for name, file_path in files.items():
        result = load_dem(file_path)
        if result is None:
            continue
        data, extent, stats, data_bounds = result
        dem_items.append({"name": name, "data": data, "extent": extent, "stats": stats, "data_bounds": data_bounds})
        all_values.append(data[~np.isnan(data)])
        print(f"读取{name}...")
        print(f"{name}: {stats['count']} 个数据点")

    if not dem_items:
        raise FileNotFoundError("没有找到任何数据文件!")

    valid_values = np.concatenate(all_values)
    vmin = float(valid_values.min())
    vmax = float(valid_values.max())
    cmap = LinearSegmentedColormap.from_list("elevation_map", ELEVATION_COLORS)
    norm = Normalize(vmin=vmin, vmax=vmax)

    x_min = min(e["data_bounds"][0] for e in dem_items)
    x_max = max(e["data_bounds"][1] for e in dem_items)
    y_min = min(e["data_bounds"][2] for e in dem_items)
    y_max = max(e["data_bounds"][3] for e in dem_items)
    dx, dy = (x_max - x_min) * 0.02, (y_max - y_min) * 0.02
    x_min, x_max = x_min - dx, x_max + dx
    y_min, y_max = y_min - dy, y_max + dy

    fig, ax = plt.subplots(figsize=(10, 10), dpi=600)
    for item in dem_items:
        disp_data, disp_extent = downsample_for_display(item["data"], item["extent"])
        ax.imshow(
            disp_data,
            extent=disp_extent,
            origin="upper",
            cmap=cmap,
            norm=norm,
            interpolation="bilinear",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    colorbar.set_label("DEM", fontsize=24, fontweight="bold")
    colorbar.ax.tick_params(labelsize=22)

    ax.set_title("BTH DEM", fontsize=32, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=24, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=24, fontweight="bold")
    ax.tick_params(axis="both", labelsize=22)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=600)
    plt.show()

    print(f"图像已保存为: {OUTPUT_FILE}")
    summary = pd.DataFrame(
        [
            {
                "region": item["name"],
                "min": item["stats"]["min"],
                "max": item["stats"]["max"],
                "mean": item["stats"]["mean"],
                "median": item["stats"]["median"],
            }
            for item in dem_items
        ]
    )
    print("\n统计信息:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="京津冀地区 DEM 高程图绘制")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="数据根目录（包含 4.分城市的数据 和 3.分省份的数据 的上级目录）",
    )
    args = parser.parse_args()
    base_path = Path(args.data_dir) if args.data_dir else None
    main(base_path)
