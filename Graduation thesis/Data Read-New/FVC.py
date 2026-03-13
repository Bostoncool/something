"""
读取 2018-2023 年植被覆盖度（FVC）数据（.tif 格式）
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

# FVC 数据文件夹路径
_DATA_ROOT = Path(__file__).resolve().parent.parent / "1.模型要用的"
FVC_PATH = _DATA_ROOT / "2000-2023[250米分辨率植被FVC]" / "2018-2023"
YEAR_PATTERN = re.compile(r"(201[89]|202[0-3])")


def find_fvc_tif_files(data_dir: Path) -> list[tuple[int, Path]]:
    """
    扫描目录中的 .tif/.tiff 文件，并按年份排序。

    Parameters
    ----------
    data_dir : Path
        FVC 数据所在目录

    Returns
    -------
    list[tuple[int, Path]]
        (年份, 文件路径) 的列表，按年份升序
    """
    year_files: list[tuple[int, Path]] = []
    for path in data_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
            match = YEAR_PATTERN.search(path.stem)
            if match:
                year = int(match.group(1))
                if 2018 <= year <= 2023:
                    year_files.append((year, path))
    return sorted(year_files, key=lambda x: x[0])


def load_fvc_tif(file_path: Path, masked: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """
    读取单个 FVC .tif 文件。

    Parameters
    ----------
    file_path : Path
        .tif 文件路径
    masked : bool, default True
        是否将 NoData 转为 masked 数组（便于后续计算时忽略）

    Returns
    -------
    tuple[np.ndarray, dict]
        (栅格数据数组, 元数据字典)
    """
    with rasterio.open(file_path) as src:
        data = src.read(1, masked=masked)
        meta = {
            "height": src.height,
            "width": src.width,
            "crs": src.crs,
            "transform": src.transform,
            "nodata": src.nodata,
            "dtype": str(src.dtypes[0]),
        }
    return data, meta


def load_fvc_data(
    data_dir: Path | str | None = None,
) -> dict[int, tuple[np.ndarray, dict[str, Any]]]:
    """
    读取 2018-2023 年全部 FVC 数据。

    Parameters
    ----------
    data_dir : Path or str, optional
        数据目录，默认使用 FVC_PATH

    Returns
    -------
    dict[int, tuple[np.ndarray, dict]]
        以年份为键，值为 (栅格数组, 元数据) 的字典
    """
    data_dir = Path(data_dir) if data_dir else FVC_PATH
    year_files = find_fvc_tif_files(data_dir)

    if not year_files:
        raise FileNotFoundError(
            f"在目录中未找到 2018-2023 年的 .tif 文件: {data_dir}"
        )

    result: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
    for year, path in year_files:
        data, meta = load_fvc_tif(path)
        result[year] = (data, meta)

    return result


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        fvc_data = load_fvc_data()
        print("FVC 数据读取成功")
        print("=" * 60)
        for year, (arr, meta) in fvc_data.items():
            valid = arr.compressed() if hasattr(arr, "compressed") else arr[~np.isnan(arr)]
            print(
                f"  {year}: 形状 {arr.shape}, "
                f"有效像元 {len(valid)}, "
                f"范围 [{np.nanmin(valid):.4f}, {np.nanmax(valid):.4f}]"
            )
        print("=" * 60)
        print(f"共读取 {len(fvc_data)} 年数据: {sorted(fvc_data.keys())}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
