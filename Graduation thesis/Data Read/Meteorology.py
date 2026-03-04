from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_INPUT_DIR = Path(r"F:\1.模型要用的\Year")
DEFAULT_OUTPUT_LONG_CSV = Path(r"F:\1.模型要用的\era5_yearly_summary_long_2018_2023.csv")
DEFAULT_OUTPUT_WIDE_CSV = Path(r"F:\1.模型要用的\era5_yearly_summary_wide_2018_2023.csv")

YEAR_START = 2018
YEAR_END = 2023

TIME_DIM_CANDIDATES = ("time", "valid_time", "forecast_time", "verification_time")
COORD_VAR_NAMES = {
    "latitude",
    "longitude",
    "lat",
    "lon",
    "time",
    "valid_time",
    "forecast_time",
    "verification_time",
    "number",
    "expver",
    "surface",
}
YEAR_PATTERN = re.compile(r"(20\d{2})")

# ERA5 变量单位转换：原始单位 -> 目标单位
# 参考：https://cds.climate.copernicus.eu/
UNIT_CONVERSIONS = {
    # 气温（K -> ℃）
    "2m_temperature": lambda x: x - 273.15,
    "t2m": lambda x: x - 273.15,
    # 海平面气压（Pa -> hPa）
    "mean_sea_level_pressure": lambda x: x / 100.0,
    "msl": lambda x: x / 100.0,
    # 露点温度（K -> ℃）
    "2m_dewpoint_temperature": lambda x: x - 273.15,
    "d2m": lambda x: x - 273.15,
    # u/v 风速分量已是 m/s，无需转换
    "10m_u_component_of_wind": None,
    "10m_v_component_of_wind": None,
    "u10": None,
    "v10": None,
    # 云量（0-1 比例 -> oktas 0-8）
    "total_cloud_cover": lambda x: x * 8.0,
    "tcc": lambda x: x * 8.0,
    # 降水量（m -> mm）
    "total_precipitation": lambda x: x * 1000.0,
    "tp": lambda x: x * 1000.0,
}


def _create_junction_for_path(root_dir: Path) -> Optional[Path]:
    """
    若 root_dir 含方括号 []，创建临时目录联结（mklink /J）以规避 netCDF4 底层 C 库的路径解析问题。
    联结不需要管理员权限。返回联结路径，失败则返回 None。
    """
    if sys.platform != "win32" or ("[" not in str(root_dir) and "]" not in str(root_dir)):
        return None
    try:
        import tempfile
        root_resolved = root_dir.resolve()
        link_base = Path(tempfile.gettempdir()) / "era5_meteorology_read"
        link_base.mkdir(exist_ok=True)
        link_path = link_base / "data"
        if link_path.exists():
            subprocess.run(
                ["cmd", "/c", "rmdir", str(link_path)],
                check=False,
                capture_output=True,
            )
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(root_resolved)],
            check=True,
            capture_output=True,
        )
        if link_path.exists():
            return link_path
    except (subprocess.CalledProcessError, OSError):  # pylint: disable=broad-except
        pass
    return None


def _path_for_netcdf(path: Path) -> str:
    """
    返回 netCDF4 可正确打开的路径字符串。
    Windows 下路径含方括号 [] 时，底层 C 库可能将其误解析为通配符导致失败。
    优先尝试短路径（8.3 格式）以规避该问题。
    注意：勿对 path 调用 resolve()，否则通过目录联结访问时会解析回含方括号的原始路径。
    """
    path_str = os.path.normpath(str(path))
    if sys.platform != "win32":
        return path_str

    if "[" not in path_str and "]" not in path_str:
        return path_str

    # 尝试短路径（8.3 格式）
    try:
        import ctypes
        buf_size = ctypes.windll.kernel32.GetShortPathNameW(path_str, None, 0)
        if buf_size > 0:
            buf = ctypes.create_unicode_buffer(buf_size)
            if ctypes.windll.kernel32.GetShortPathNameW(path_str, buf, buf_size):
                short_path = buf.value
                if short_path and short_path != path_str:
                    return short_path
    except Exception:  # pylint: disable=broad-except
        pass

    return path_str


@dataclass(frozen=True)
class YearlyRecord:
    file_path: str
    folder_name: str
    variable_name: str
    year: str
    mean: float
    std: float
    min: float
    max: float
    valid_count: int
    status: str
    error: str = ""


def find_nc_files(root_dir: Path) -> list[Path]:
    """递归扫描输入目录下所有 .nc 文件。"""
    return sorted(path for path in root_dir.rglob("*.nc") if path.is_file())


def sample_one_per_folder(nc_files: list[Path]) -> list[Path]:
    """
    每种气象类型（按父目录 folder_name 区分）只保留一个样本文件。
    用于测试模式，避免读取全部数据。
    """
    seen: dict[str, Path] = {}
    for path in nc_files:
        folder = path.parent.name
        if folder not in seen:
            seen[folder] = path
    return sorted(seen.values(), key=lambda p: (p.parent.name, str(p)))


def _safe_stat(func, values: np.ndarray, axis: int = 1) -> np.ndarray:
    """空数组时返回 NaN，避免统计函数报错。"""
    if values.size == 0:
        return np.full((values.shape[0],), np.nan, dtype=np.float64)
    return func(values, axis=axis)


def parse_year_from_filename(file_name: str) -> str:
    """从文件名提取年份 YYYY，提取失败时返回空字符串。"""
    match = YEAR_PATTERN.search(file_name)
    return match.group(1) if match else ""


def resolve_target_variable_name(dataset: nc.Dataset, folder_name: str) -> Optional[str]:
    """
    优先按目录名寻找同名变量；找不到则回退到第一个“非坐标且至少1维”的变量。
    例如目录名: 2m_dewpoint_temperature -> 变量可能是 d2m / 2m_dewpoint_temperature。
    """
    if folder_name in dataset.variables:
        return folder_name

    # 常见 ERA5 变量别名映射
    alias_map = {
        "2m_dewpoint_temperature": ("d2m",),
        "2m_temperature": ("t2m",),
        "10m_u_component_of_wind": ("u10",),
        "10m_v_component_of_wind": ("v10",),
        "mean_sea_level_pressure": ("msl",),
        "total_cloud_cover": ("tcc",),
        "total_precipitation": ("tp",),
    }
    for alias in alias_map.get(folder_name, ()):
        if alias in dataset.variables:
            return alias

    for name, var in dataset.variables.items():
        if name in COORD_VAR_NAMES:
            continue
        if getattr(var, "ndim", 0) >= 1:
            return name

    return None


def extract_year_strings(dataset: nc.Dataset, var: nc.Variable) -> Optional[list[str]]:
    """
    根据变量维度找到时间维并转为 YYYY 列表。
    找不到时间维时返回 None。
    """
    time_dim_name = next((d for d in var.dimensions if d in TIME_DIM_CANDIDATES), None)
    if time_dim_name is None:
        return None
    if time_dim_name not in dataset.variables:
        return None

    time_var = dataset.variables[time_dim_name]
    try:
        units = getattr(time_var, "units")
        calendar = getattr(time_var, "calendar", "standard")
        raw = time_var[:]
        dt_values = nc.num2date(raw, units=units, calendar=calendar)
        return [str(pd.Timestamp(item).year) for item in dt_values]
    except Exception:
        try:
            values = pd.to_datetime(time_var[:], errors="coerce")
            return [str(pd.Timestamp(v).year) for v in values if pd.notna(v)]
        except Exception:
            return None


def build_yearly_records(file_path_str: str) -> list[YearlyRecord]:
    """
    读取单个 .nc 文件，按“年”计算栅格统计量（mean/std/min/max/valid_count）。
    返回该文件对应的多条 YearlyRecord（若文件里包含多个时间步）。
    """
    file_path = Path(file_path_str)
    folder_name = file_path.parent.name
    year_candidate = parse_year_from_filename(file_path.name)

    if not file_path.exists():
        return [
            YearlyRecord(
                file_path=str(file_path),
                folder_name=folder_name,
                variable_name="",
                year=year_candidate,
                mean=float("nan"),
                std=float("nan"),
                min=float("nan"),
                max=float("nan"),
                valid_count=0,
                status="failed",
                error=f"文件不存在: {file_path}",
            )
        ]

    try:
        nc_path = _path_for_netcdf(file_path)
        with nc.Dataset(nc_path, mode="r") as ds:
            var_name = resolve_target_variable_name(ds, folder_name=folder_name)
            if var_name is None:
                return [
                    YearlyRecord(
                        file_path=str(file_path),
                        folder_name=folder_name,
                        variable_name="",
                        year=year_candidate,
                        mean=float("nan"),
                        std=float("nan"),
                        min=float("nan"),
                        max=float("nan"),
                        valid_count=0,
                        status="failed",
                        error="未识别到目标变量（变量列表为空或只有坐标变量）。",
                    )
                ]

            var = ds.variables[var_name]
            data = np.array(var[:], dtype=np.float32)

            fill_value = getattr(var, "_FillValue", None)
            if fill_value is None:
                fill_value = getattr(var, "missing_value", None)
            if fill_value is not None:
                data = np.where(data == fill_value, np.nan, data)

            scale_factor = float(getattr(var, "scale_factor", 1.0))
            add_offset = float(getattr(var, "add_offset", 0.0))
            data = data * scale_factor + add_offset

            if data.ndim == 0:
                data = data.reshape(1, 1)

            # 应用 ERA5 单位转换（K->℃, Pa->hPa, m->mm 等）
            convert_fn = UNIT_CONVERSIONS.get(folder_name)
            if convert_fn is not None:
                data = convert_fn(data)

            time_axis = next((i for i, d in enumerate(var.dimensions) if d in TIME_DIM_CANDIDATES), None)
            if time_axis is None:
                flat = data.reshape(1, -1)
                year_strings = [year_candidate]
            else:
                time_first = np.moveaxis(data, time_axis, 0)
                flat = time_first.reshape(time_first.shape[0], -1)
                year_strings = extract_year_strings(ds, var)
                if year_strings is None or len(year_strings) != flat.shape[0]:
                    # 时间解码失败时，尽量从文件名补一个年份，再兜底为空串
                    fallback_year = year_candidate
                    year_strings = [fallback_year for _ in range(flat.shape[0])]

            valid = np.where(np.isfinite(flat), flat, np.nan)
            valid_count = np.sum(np.isfinite(valid), axis=1).astype(int)

            mean_values = _safe_stat(np.nanmean, valid)
            std_values = _safe_stat(np.nanstd, valid)
            min_values = _safe_stat(np.nanmin, valid)
            max_values = _safe_stat(np.nanmax, valid)

            records: list[YearlyRecord] = []
            for idx in range(flat.shape[0]):
                records.append(
                    YearlyRecord(
                        file_path=str(file_path),
                        folder_name=folder_name,
                        variable_name=var_name,
                        year=year_strings[idx],
                        mean=float(mean_values[idx]) if np.isfinite(mean_values[idx]) else float("nan"),
                        std=float(std_values[idx]) if np.isfinite(std_values[idx]) else float("nan"),
                        min=float(min_values[idx]) if np.isfinite(min_values[idx]) else float("nan"),
                        max=float(max_values[idx]) if np.isfinite(max_values[idx]) else float("nan"),
                        valid_count=int(valid_count[idx]),
                        status="ok",
                        error="",
                    )
                )
            return records
    except Exception as exc:  # pylint: disable=broad-except
        return [
            YearlyRecord(
                file_path=str(file_path),
                folder_name=folder_name,
                variable_name="",
                year=year_candidate,
                mean=float("nan"),
                std=float("nan"),
                min=float("nan"),
                max=float("nan"),
                valid_count=0,
                status="failed",
                error=str(exc),
            )
        ]


def to_dataframe(records: Iterable[YearlyRecord]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(record) for record in records])
    if df.empty:
        return df
    df["_year_num"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.sort_values(["_year_num", "folder_name", "file_path"], kind="mergesort").reset_index(drop=True)
    df = df.drop(columns=["_year_num"])
    return df


def _add_wind_derived_columns(wide: pd.DataFrame) -> pd.DataFrame:
    """
    从 u10、v10 计算风速（m/s）和风向（°）。
    风速 = √(u² + v²)，风向 = 180 + atan2(u, v) × (180/π)
    """
    u_col = next((c for c in ("10m_u_component_of_wind", "u10") if c in wide.columns), None)
    v_col = next((c for c in ("10m_v_component_of_wind", "v10") if c in wide.columns), None)
    if u_col is None or v_col is None:
        return wide

    u = wide[u_col].astype(float)
    v = wide[v_col].astype(float)
    wide = wide.copy()
    wide["wind_speed_ms"] = np.sqrt(u**2 + v**2)
    wide["wind_direction_deg"] = 180.0 + np.degrees(np.arctan2(u, v))
    return wide


def build_wide_table(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    从长表生成宽表：
    - 行：year
    - 列：每个 folder_name 对应的年均值（mean）
    - 若存在 u10、v10，则额外计算 wind_speed_ms、wind_direction_deg
    """
    ok_df = df_long[df_long["status"] == "ok"].copy()
    if ok_df.empty:
        return pd.DataFrame()

    wide = ok_df.pivot_table(
        index="year",
        columns="folder_name",
        values="mean",
        aggfunc="mean",
    ).sort_index()
    wide.columns.name = None
    wide = wide.reset_index()
    wide = _add_wind_derived_columns(wide)
    return wide


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="并行读取 ERA5 气象 NetCDF 文件，按年输出统计（2018-2023）。"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f"输入根目录（默认: {DEFAULT_INPUT_DIR}）",
    )
    parser.add_argument(
        "--output-long-csv",
        type=str,
        default=str(DEFAULT_OUTPUT_LONG_CSV),
        help=f"长表输出路径（默认: {DEFAULT_OUTPUT_LONG_CSV}）",
    )
    parser.add_argument(
        "--output-wide-csv",
        type=str,
        default=str(DEFAULT_OUTPUT_WIDE_CSV),
        help=f"宽表输出路径（默认: {DEFAULT_OUTPUT_WIDE_CSV}）",
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
    parser.add_argument(
        "--year-start",
        type=int,
        default=YEAR_START,
        help=f"起始年份（默认: {YEAR_START}）",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=YEAR_END,
        help=f"结束年份（默认: {YEAR_END}）",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="测试模式：每种气象类型只读取一个样本文件，用于快速验证",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_long_csv = Path(args.output_long_csv).expanduser()
    output_wide_csv = Path(args.output_wide_csv).expanduser()
    processes = max(1, int(args.processes))
    chunksize = max(1, int(args.chunksize))

    if not input_dir.exists():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        return 1
    if not input_dir.is_dir():
        print(f"[ERROR] 输入路径不是目录: {input_dir}")
        return 1

    # Windows 下路径含方括号 [] 时，netCDF4 底层 C 库可能无法正确打开，通过目录联结规避
    actual_input_dir: Path = input_dir
    junction_path: Optional[Path] = None
    if sys.platform == "win32" and ("[" in str(input_dir) or "]" in str(input_dir)):
        junction_path = _create_junction_for_path(input_dir)
        if junction_path is not None:
            actual_input_dir = junction_path
            print("[INFO] 检测到路径含方括号，已创建临时目录联结以规避 netCDF4 兼容性问题")

    nc_files = find_nc_files(actual_input_dir)
    if not nc_files:
        print(f"[WARN] 在目录中未找到 .nc 文件: {input_dir}")
        return 2

    if args.test:
        nc_files = sample_one_per_folder(nc_files)
        processes = 1  # 测试模式用单进程，避免 Windows 多进程路径解析问题
        print("[INFO] 测试模式：每种气象类型仅读取一个样本文件，单进程运行")

    # 解析为绝对路径并过滤存在的文件
    # 若使用目录联结，勿 resolve，否则会解析回含方括号的原始路径导致 netCDF4 失败
    nc_files_resolved = [p.resolve() for p in nc_files] if junction_path is None else list(nc_files)
    missing = [p for p in nc_files_resolved if not p.exists()]
    nc_files = [p for p in nc_files_resolved if p.exists()]
    if missing:
        print(f"[WARN] 以下 {len(missing)} 个文件不存在，已跳过:")
        for p in missing[:5]:
            print(f"      - {p}")
        if len(missing) > 5:
            print(f"      ... 及其他 {len(missing) - 5} 个")
    if not nc_files:
        print("[ERROR] 无有效文件可处理（全部不存在或已跳过）。")
        return 4

    print("=" * 80)
    print(f"[INFO] 输入目录: {input_dir}")
    print(f"[INFO] 待处理文件数: {len(nc_files)}")
    print(f"[INFO] 并行进程: {processes}")
    print(f"[INFO] chunksize: {chunksize}")
    print("=" * 80)

    file_args = [str(path) for path in nc_files]
    all_records: list[YearlyRecord] = []

    try:
        with mp.Pool(processes=processes) as pool:
            iterator = pool.imap_unordered(build_yearly_records, file_args, chunksize=chunksize)
            for batch_records in tqdm(iterator, total=len(file_args), desc="读取进度", unit="file"):
                all_records.extend(batch_records)

        if not all_records:
            print("[ERROR] 未读取到任何记录，请检查文件格式或路径。")
            return 3

        df_long = to_dataframe(all_records)
        if df_long.empty:
            print("[ERROR] 无可输出结果。")
            return 3

        year_start = int(args.year_start)
        year_end = int(args.year_end)
        if year_start <= year_end:
            before_filter = len(df_long)
            year_num = pd.to_numeric(df_long["year"], errors="coerce")
            valid_years = year_num.notna()
            in_range = (year_num >= year_start) & (year_num <= year_end)
            df_filtered = df_long[valid_years & in_range].copy()
            if len(df_filtered) == 0 and before_filter > 0:
                n_valid = int(valid_years.sum())
                print(f"[WARN] 年份过滤 ({year_start} ~ {year_end}) 移除了全部 {before_filter} 条记录。")
                print(f"       有效年份: {n_valid} 条，无效: {before_filter - n_valid} 条。")
                if n_valid > 0:
                    valid_df = df_long.loc[valid_years].copy()
                    valid_year_num = pd.to_numeric(valid_df["year"], errors="coerce")
                    print(
                        f"       数据年份范围: {int(valid_year_num.min())} ~ {int(valid_year_num.max())}"
                    )
                print("       已跳过年份过滤，输出全部数据。")
            else:
                df_long = df_filtered

        ok_count = int((df_long["status"] == "ok").sum())
        failed_count = int((df_long["status"] == "failed").sum())

        output_long_csv.parent.mkdir(parents=True, exist_ok=True)
        df_long.to_csv(output_long_csv, index=False, encoding="utf-8-sig")

        df_wide = build_wide_table(df_long)
        if not df_wide.empty:
            output_wide_csv.parent.mkdir(parents=True, exist_ok=True)
            df_wide.to_csv(output_wide_csv, index=False, encoding="utf-8-sig")

        print("=" * 80)
        print(f"[INFO] 长表输出: {output_long_csv}")
        if not df_wide.empty:
            print(f"[INFO] 宽表输出: {output_wide_csv}")
        print(f"[INFO] 成功记录: {ok_count} | 失败记录: {failed_count}")
        if failed_count > 0:
            print("[INFO] 可在长表 CSV 的 error 列查看失败原因。")
        print("=" * 80)
        return 0
    finally:
        if junction_path is not None:
            try:
                subprocess.run(["cmd", "/c", "rmdir", str(junction_path)], check=False, capture_output=True)
            except Exception:  # pylint: disable=broad-except
                pass


if __name__ == "__main__":
    # Windows 多进程保护
    mp.freeze_support()
    raise SystemExit(main())
