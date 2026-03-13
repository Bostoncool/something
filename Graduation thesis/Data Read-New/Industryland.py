"""
读取 2018-2023 年中国工业用地数据

数据目录结构:
    <workspace>/1.模型要用的/2018-2023[工业用地]/
    ├── 2018/  Industrial_land_CHN_{城市ID}_{年份}.tif
    ├── 2019/
    ├── ...
    └── 2023/
    └── 1093_city_information_china_only.xlsx  # 城市信息表（ID_HDC_G0 对应 tif 中的城市代码）
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

# 解决 Windows 控制台中文输出编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 默认数据路径
_DATA_ROOT = Path(__file__).resolve().parent.parent / "1.模型要用的"
DEFAULT_INPUT_DIR = _DATA_ROOT / "2018-2023[工业用地]"
DEFAULT_CITY_INFO_PATH = DEFAULT_INPUT_DIR / "1093_city_information_china_only.xlsx"
YEAR_PATTERN = re.compile(r"(20\d{2})")
CITY_ID_PATTERN = re.compile(r"Industrial_land_CHN_(\d+)_\d{4}")


@dataclass
class IndustrialLandReadResult:
    """单幅工业用地 tif 的读取结果"""

    file_path: str
    year: str
    city_id: int
    status: str
    height: int
    width: int
    pixel_count: int
    valid_pixel_count: int
    industrial_pixel_count: int
    min_value: float
    max_value: float
    nodata: str
    crs: str
    error: str = ""


def find_tif_files(root_dir: Path) -> list[Path]:
    """递归扫描所有工业用地 tif 文件。"""
    return sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def infer_year(file_path: Path) -> str:
    """从路径或文件名中提取年份。"""
    search_text = f"{file_path.parent.name}_{file_path.stem}"
    match = YEAR_PATTERN.search(search_text)
    return match.group(1) if match else ""


def infer_city_id(file_path: Path) -> int:
    """从文件名中提取城市 ID。Industrial_land_CHN_10319_2018.tif -> 10319"""
    match = CITY_ID_PATTERN.search(file_path.stem)
    return int(match.group(1)) if match else 0


def read_one_tif(file_path: Path) -> IndustrialLandReadResult:
    """读取单个工业用地 tif，提取基础统计。"""
    year = infer_year(file_path)
    city_id = infer_city_id(file_path)

    try:
        with rasterio.open(file_path) as src:
            band = src.read(1, masked=True)
            valid_values = band.compressed()

            # 工业用地通常为二值或分类栅格，统计非零/有效像素
            industrial_mask = valid_values > 0
            industrial_count = int(industrial_mask.sum())

            return IndustrialLandReadResult(
                file_path=str(file_path),
                year=year,
                city_id=city_id,
                status="ok",
                height=int(src.height),
                width=int(src.width),
                pixel_count=int(src.height * src.width),
                valid_pixel_count=int(valid_values.size),
                industrial_pixel_count=industrial_count,
                min_value=float(np.min(valid_values)) if valid_values.size > 0 else float("nan"),
                max_value=float(np.max(valid_values)) if valid_values.size > 0 else float("nan"),
                nodata=str(src.nodata) if src.nodata is not None else "",
                crs=str(src.crs) if src.crs is not None else "",
                error="",
            )
    except Exception as exc:  # pylint: disable=broad-except
        return IndustrialLandReadResult(
            file_path=str(file_path),
            year=year,
            city_id=city_id,
            status="failed",
            height=0,
            width=0,
            pixel_count=0,
            valid_pixel_count=0,
            industrial_pixel_count=0,
            min_value=float("nan"),
            max_value=float("nan"),
            nodata="",
            crs="",
            error=str(exc),
        )


def load_industrial_land_summary(
    input_dir: Path | str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    读取所有工业用地 tif 并汇总为 DataFrame。

    Parameters
    ----------
    input_dir : Path or str, optional
        工业用地数据根目录，默认使用 DEFAULT_INPUT_DIR。
    show_progress : bool, default True
        是否显示 tqdm 进度条。

    Returns
    -------
    pd.DataFrame
        汇总表，列包括：file_path, year, city_id, status, height, width,
        pixel_count, valid_pixel_count, industrial_pixel_count, min_value,
        max_value, nodata, crs, error
    """
    root = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    if not root.exists():
        raise FileNotFoundError(f"输入目录不存在: {root}")

    tif_files = find_tif_files(root)
    if not tif_files:
        return pd.DataFrame()

    iterator = tqdm(tif_files, desc="读取工业用地 tif", unit="file") if show_progress else tif_files
    results = [read_one_tif(p) for p in iterator]
    df = pd.DataFrame([vars(r) for r in results])
    return df.sort_values(["year", "city_id"], kind="mergesort").reset_index(drop=True)


def load_city_info(city_info_path: Path | str | None = None) -> pd.DataFrame:
    """
    读取城市信息表，用于将 city_id 映射到城市名称等。

    Parameters
    ----------
    city_info_path : Path or str, optional
        Excel 路径，默认使用 1093_city_information_china_only.xlsx。

    Returns
    -------
    pd.DataFrame
        城市信息，含 ID_HDC_G0, CTR_MN_NM, UC_NM_MN, IND_2018~IND_2023 等列。
    """
    path = Path(city_info_path) if city_info_path else DEFAULT_CITY_INFO_PATH
    if not path.exists():
        return pd.DataFrame()

    return pd.read_excel(path, engine="openpyxl")


def load_industrial_land_with_city_info(
    input_dir: Path | str | None = None,
    city_info_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    读取工业用地汇总并与城市信息表合并。

    Returns
    -------
    pd.DataFrame
        合并后的数据，包含 tif 统计与城市名称等信息。
    """
    df_summary = load_industrial_land_summary(input_dir)
    df_city = load_city_info(city_info_path)

    if df_summary.empty:
        return df_summary
    if df_city.empty:
        return df_summary

    df_city = df_city.rename(columns={"ID_HDC_G0": "city_id"})
    return df_summary.merge(
        df_city,
        on="city_id",
        how="left",
        suffixes=("", "_city"),
    )


def iter_tif_by_year(
    input_dir: Path | str | None = None,
) -> Iterator[tuple[str, list[Path]]]:
    """
    按年份迭代 tif 文件列表。

    Yields
    ------
    tuple[str, list[Path]]
        (年份, 该年份下所有 tif 路径列表)
    """
    root = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    tif_files = find_tif_files(root)

    by_year: dict[str, list[Path]] = {}
    for p in tif_files:
        year = infer_year(p)
        if year not in by_year:
            by_year[year] = []
        by_year[year].append(p)

    for year in sorted(by_year.keys()):
        yield year, sorted(by_year[year])


if __name__ == "__main__":
    input_dir = DEFAULT_INPUT_DIR

    if not input_dir.exists():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        sys.exit(1)

    print("=" * 60)
    print("工业用地数据读取")
    print("=" * 60)

    # 1. 读取汇总
    df = load_industrial_land_summary(input_dir)
    if df.empty:
        print("[WARN] 未找到 .tif 文件")
        sys.exit(2)

    print(f"\n汇总数据形状: {df.shape}")
    print(f"年份: {sorted(df['year'].dropna().unique().tolist())}")
    print(f"城市数量: {df['city_id'].nunique()}")
    print(f"\n前 10 行:\n{df.head(10).to_string()}")

    # 2. 合并城市信息
    df_merged = load_industrial_land_with_city_info(input_dir)
    if not df_merged.empty and "UC_NM_MN" in df_merged.columns:
        print("\n" + "=" * 60)
        print("合并城市信息后（含城市名称）:")
        print(df_merged[["year", "city_id", "UC_NM_MN", "industrial_pixel_count", "status"]].head(15).to_string())

    print("\n" + "=" * 60)
