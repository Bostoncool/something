from __future__ import annotations

import argparse
import multiprocessing as mp
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_INPUT_DIR = Path(r"F:\1.模型要用的\2018-2023[PM2.5-china]\Year")
DEFAULT_OUTPUT_CSV = Path(r"F:\1.模型要用的\pm25_yearly_summary_2018_2023.csv")
# 年度数据：Y1K 格式，文件名如 CHAP_PM2.5_Y1K_2018_V4.nc
FILENAME_PATTERN = re.compile(r"CHAP_PM2\.5_Y1K_(\d{4})(?:_V\d+)?\.nc$", re.IGNORECASE)
PM25_VAR_CANDIDATES = ("PM2.5", "pm25", "PM25")


@dataclass(frozen=True)
class ReadResult:
    file_path: str
    year: str
    status: str
    rows: int
    cols: int
    valid_count: int
    mean: float
    std: float
    min: float
    max: float
    p25: float
    p50: float
    p75: float
    error: str = ""


def find_nc_files(root_dir: Path) -> list[Path]:
    """递归扫描所有 .nc 文件，并按路径排序。"""
    return sorted(path for path in root_dir.rglob("*.nc") if path.is_file())


def parse_year_from_name(file_path: Path) -> str:
    """从文件名提取年份 YYYY，提取失败时返回空字符串。"""
    match = FILENAME_PATTERN.search(file_path.name)
    return match.group(1) if match else ""


def resolve_pm25_var_name(dataset: nc.Dataset) -> Optional[str]:
    """在候选变量名中选择 PM2.5 变量。"""
    for var_name in PM25_VAR_CANDIDATES:
        if var_name in dataset.variables:
            return var_name
    return None


def _safe_stat(func, values: np.ndarray) -> float:
    """空数组时返回 NaN，避免统计函数抛错。"""
    if values.size == 0:
        return float("nan")
    return float(func(values))


def process_one_file(file_path_str: str) -> ReadResult:
    """
    读取单个 PM2.5 nc 文件并返回年尺度统计结果。

    说明：
    - 为控制内存，本函数仅返回统计量，不返回整幅栅格。
    - 自动处理 _FillValue / missing_value / scale_factor / add_offset。
    """
    file_path = Path(file_path_str)
    year_str = parse_year_from_name(file_path)

    try:
        with nc.Dataset(file_path, mode="r") as ds:
            pm_var_name = resolve_pm25_var_name(ds)
            if pm_var_name is None:
                return ReadResult(
                    file_path=str(file_path),
                    year=year_str,
                    status="failed",
                    rows=0,
                    cols=0,
                    valid_count=0,
                    mean=float("nan"),
                    std=float("nan"),
                    min=float("nan"),
                    max=float("nan"),
                    p25=float("nan"),
                    p50=float("nan"),
                    p75=float("nan"),
                    error=f"未找到 PM2.5 变量，候选={PM25_VAR_CANDIDATES}",
                )

            var = ds.variables[pm_var_name]
            raw_data = var[:].astype(np.float32, copy=False)

            fill_value = getattr(var, "_FillValue", None)
            if fill_value is None:
                fill_value = getattr(var, "missing_value", None)

            scale_factor = float(getattr(var, "scale_factor", 1.0))
            add_offset = float(getattr(var, "add_offset", 0.0))

            # 先替换缺测，再进行缩放
            if fill_value is not None:
                raw_data = np.where(raw_data == fill_value, np.nan, raw_data)

            data = raw_data * scale_factor + add_offset
            flat = data.ravel()
            valid = flat[~np.isnan(flat)]

            rows = int(data.shape[-2]) if data.ndim >= 2 else int(data.shape[0])
            cols = int(data.shape[-1]) if data.ndim >= 2 else 1

            return ReadResult(
                file_path=str(file_path),
                year=year_str,
                status="ok",
                rows=rows,
                cols=cols,
                valid_count=int(valid.size),
                mean=_safe_stat(np.mean, valid),
                std=_safe_stat(np.std, valid),
                min=_safe_stat(np.min, valid),
                max=_safe_stat(np.max, valid),
                p25=_safe_stat(lambda x: np.percentile(x, 25), valid),
                p50=_safe_stat(lambda x: np.percentile(x, 50), valid),
                p75=_safe_stat(lambda x: np.percentile(x, 75), valid),
                error="",
            )
    except Exception as exc:  # pylint: disable=broad-except
        return ReadResult(
            file_path=str(file_path),
            year=year_str,
            status="failed",
            rows=0,
            cols=0,
            valid_count=0,
            mean=float("nan"),
            std=float("nan"),
            min=float("nan"),
            max=float("nan"),
            p25=float("nan"),
            p50=float("nan"),
            p75=float("nan"),
            error=str(exc),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="并行读取 2018-2023 年度 PM2.5 NetCDF 文件并输出统计汇总"
    )
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
        default=8,
        help="multiprocessing imap_unordered 的 chunksize（默认: 8）",
    )
    return parser


def to_dataframe(results: Iterable[ReadResult]) -> pd.DataFrame:
    df = pd.DataFrame([result.__dict__ for result in results])
    if df.empty:
        return df
    # 按年份和路径排序，便于后续追踪异常文件
    sort_cols = [col for col in ("year", "file_path") if col in df.columns]
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


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

    nc_files = find_nc_files(input_dir)
    if not nc_files:
        print(f"[WARN] 在目录中未找到 .nc 文件: {input_dir}")
        return 2

    print("=" * 80)
    print(f"[INFO] 输入目录: {input_dir}")
    print(f"[INFO] 文件数量: {len(nc_files)}")
    print(f"[INFO] 并行进程: {processes}")
    print(f"[INFO] chunksize: {chunksize}")
    print("=" * 80)

    file_args = [str(path) for path in nc_files]
    results: list[ReadResult] = []

    with mp.Pool(processes=processes) as pool:
        iterator = pool.imap_unordered(process_one_file, file_args, chunksize=chunksize)
        for result in tqdm(iterator, total=len(file_args), desc="读取进度", unit="file"):
            results.append(result)

    df = to_dataframe(results)
    if df.empty:
        print("[ERROR] 无可输出结果。")
        return 3

    ok_count = int((df["status"] == "ok").sum())
    failed_count = int((df["status"] == "failed").sum())

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"[INFO] 输出文件: {output_csv}")
    print(f"[INFO] 成功: {ok_count} | 失败: {failed_count}")
    if failed_count > 0:
        print("[INFO] 可在输出 CSV 的 error 列查看失败原因。")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    # Windows 多进程保护
    mp.freeze_support()
    raise SystemExit(main())
