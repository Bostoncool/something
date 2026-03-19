"""
长三角地区高程数据可视化 - Python 复现
基于 R 代码 YRD-DEM.R 的功能实现
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import rasterio
from rasterio.windows import Window
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

BASE_PATH = Path("F:/")
FILES = {
    "Nanjing": BASE_PATH / "4.分城市的数据/南京市.tif",
    "Wuxi": BASE_PATH / "4.分城市的数据/无锡市.tif",
    "Nantong": BASE_PATH / "4.分城市的数据/南通市.tif",
    "Yancheng": BASE_PATH / "4.分城市的数据/盐城市.tif",
    "Yangzhou": BASE_PATH / "4.分城市的数据/扬州市.tif",
    "Zhenjiang": BASE_PATH / "4.分城市的数据/镇江市.tif",
    "Changzhou": BASE_PATH / "4.分城市的数据/常州市.tif",
    "Suzhou": BASE_PATH / "4.分城市的数据/苏州市.tif",
    "Taizhou_JS": BASE_PATH / "4.分城市的数据/泰州市.tif",
    "Hangzhou": BASE_PATH / "4.分城市的数据/杭州市.tif",
    "Ningbo": BASE_PATH / "4.分城市的数据/宁波市.tif",
    "Jiaxing": BASE_PATH / "4.分城市的数据/嘉兴市.tif",
    "Huzhou": BASE_PATH / "4.分城市的数据/湖州市.tif",
    "Shaoxing": BASE_PATH / "4.分城市的数据/绍兴市.tif",
    "Jinhua": BASE_PATH / "4.分城市的数据/金华市.tif",
    "Zhoushan": BASE_PATH / "4.分城市的数据/舟山市.tif",
    "Taizhou_ZJ": BASE_PATH / "4.分城市的数据/台州市.tif",
    "Wenzhou": BASE_PATH / "4.分城市的数据/温州市.tif",
    "Hefei": BASE_PATH / "4.分城市的数据/合肥市.tif",
    "Wuhu": BASE_PATH / "4.分城市的数据/芜湖市.tif",
    "Maanshan": BASE_PATH / "4.分城市的数据/马鞍山市.tif",
    "Tongling": BASE_PATH / "4.分城市的数据/铜陵市.tif",
    "Anqing": BASE_PATH / "4.分城市的数据/安庆市.tif",
    "Chuzhou": BASE_PATH / "4.分城市的数据/滁州市.tif",
    "Chizhou": BASE_PATH / "4.分城市的数据/池州市.tif",
    "Xuancheng": BASE_PATH / "4.分城市的数据/宣城市.tif",
    "Shanghai": BASE_PATH / "4.分城市的数据/上海市.tif",
}
OUTPUT_FILE = "长三角地区_DEM_统一图.png"
ELEVATION_COLORS = ["blue", "cyan", "green", "yellow", "orange", "red", "darkred"]
# 单边最大像素数，超过则下采样以避免内存溢出。越大越细腻，内存占用越高
MAX_DISPLAY_SIZE = 4000


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
    # 使用步长切片进行简单下采样（避免额外依赖）
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


def main() -> None:
    """主函数：绘制长三角地区 DEM 图"""
    plt.rcParams["font.family"] = "Times New Roman"

    dem_items = []
    all_values = []

    print("读取长三角地区高程数据...")
    for name, file_path in FILES.items():
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

    ax.set_title("YRD DEM", fontsize=32, fontweight="bold")
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
    main()
