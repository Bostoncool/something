from __future__ import annotations

import argparse
import multiprocessing as mp
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm


DEFAULT_INPUT_DIR = Path(r"F:\1.模型要用的\2018-2023[土地利用数据]")
DEFAULT_OUTPUT_CSV = Path(r"F:\1.模型要用的\landuse_tif_summary_2018_2023.csv")
YEAR_PATTERN = re.compile(r"(20\d{2})")
PROVINCE_PATTERN = re.compile(r"albert_([^._]+)", re.IGNORECASE)


@dataclass(frozen=True)
class LandUseReadResult:
    file_path: str
    year: str
    province: str
    status: str
    height: int
    width: int
    pixel_count: int
    valid_pixel_count: int
    unique_class_count: int
    min_value: float
    max_value: float
    nodata: str
    crs: str
    transform: str
    error: str = ""


def find_tif_files(root_dir: Path) -> list[Path]:
    """递归扫描所有 tif 文件。"""
    return sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def infer_year(file_path: Path) -> str:
    """从路径中提取年份。"""
    search_text = f"{file_path.parent.name}_{file_path.stem}"
    match = YEAR_PATTERN.search(search_text)
    return match.group(1) if match else ""


def infer_province(file_path: Path) -> str:
    """从文件名中提取省份名。"""
    match = PROVINCE_PATTERN.search(file_path.stem)
    if match:
        return match.group(1).strip().lower()
    # 回退：截取最后一段，尽量保证可追踪
    parts = file_path.stem.split("_")
    return parts[-1].strip().lower() if parts else ""


def _safe_min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size > 0 else float("nan")


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size > 0 else float("nan")


def read_one_tif(file_path_str: str) -> LandUseReadResult:
    """读取单个土地利用 tif，并提取基础统计。"""
    file_path = Path(file_path_str)
    year = infer_year(file_path)
    province = infer_province(file_path)

    try:
        with rasterio.open(file_path) as src:
            # CLCD 土地利用通常是单波段整型分类栅格
            band = src.read(1, masked=True)
            valid_values = band.compressed()

            return LandUseReadResult(
                file_path=str(file_path),
                year=year,
                province=province,
                status="ok",
                height=int(src.height),
                width=int(src.width),
                pixel_count=int(src.height * src.width),
                valid_pixel_count=int(valid_values.size),
                unique_class_count=int(np.unique(valid_values).size) if valid_values.size > 0 else 0,
                min_value=_safe_min(valid_values),
                max_value=_safe_max(valid_values),
                nodata=str(src.nodata) if src.nodata is not None else "",
                crs=str(src.crs) if src.crs is not None else "",
                transform=str(src.transform),
                error="",
            )
    except Exception as exc:  # pylint: disable=broad-except
        return LandUseReadResult(
            file_path=str(file_path),
            year=year,
            province=province,
            status="failed",
            height=0,
            width=0,
            pixel_count=0,
            valid_pixel_count=0,
            unique_class_count=0,
            min_value=float("nan"),
            max_value=float("nan"),
            nodata="",
            crs="",
            transform="",
            error=str(exc),
        )


def to_dataframe(results: Iterable[LandUseReadResult]) -> pd.DataFrame:
    df = pd.DataFrame([result.__dict__ for result in results])
    if df.empty:
        return df
    return df.sort_values(["year", "province", "file_path"], kind="mergesort").reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="并行读取 2018-2023 土地利用 tif 文件并输出汇总")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f"输入目录（默认: {DEFAULT_INPUT_DIR}）",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"输出汇总 CSV 路径（默认: {DEFAULT_OUTPUT_CSV}）",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="并行进程数（默认: CPU核数-1）",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=4,
        help="multiprocessing imap_unordered 的 chunksize（默认: 4）",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_csv = Path(args.output_csv).expanduser()
    processes = max(1, int(args.processes))
    chunksize = max(1, int(args.chunksize))

    if not input_dir.exists():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        return 1
    if not input_dir.is_dir():
        print(f"[ERROR] 输入路径不是目录: {input_dir}")
        return 1

    tif_files = find_tif_files(input_dir)
    if not tif_files:
        print(f"[WARN] 在目录中未找到 .tif/.tiff 文件: {input_dir}")
        return 2

    print("=" * 80)
    print(f"[INFO] 输入目录: {input_dir}")
    print(f"[INFO] 文件数量: {len(tif_files)}")
    print(f"[INFO] 并行进程: {processes}")
    print(f"[INFO] chunksize: {chunksize}")
    print("=" * 80)

    file_args = [str(path) for path in tif_files]
    results: list[LandUseReadResult] = []

    with mp.Pool(processes=processes) as pool:
        iterator = pool.imap_unordered(read_one_tif, file_args, chunksize=chunksize)
        for result in tqdm(iterator, total=len(file_args), desc="读取进度", unit="file"):
            results.append(result)

    df = to_dataframe(results)
    if df.empty:
        print("[ERROR] 无可输出结果。")
        return 3

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    ok_count = int((df["status"] == "ok").sum())
    failed_count = int((df["status"] == "failed").sum())
    years = sorted(df.loc[df["year"] != "", "year"].dropna().unique().tolist())
    provinces = sorted(df.loc[df["province"] != "", "province"].dropna().unique().tolist())

    print("=" * 80)
    print(f"[INFO] 输出文件: {output_csv}")
    print(f"[INFO] 成功: {ok_count} | 失败: {failed_count}")
    print(f"[INFO] 年份范围: {years}")
    print(f"[INFO] 省份数量: {len(provinces)}")
    if failed_count > 0:
        print("[INFO] 可在输出 CSV 的 error 列查看失败原因。")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    # Windows 多进程保护
    mp.freeze_support()
    raise SystemExit(main())
