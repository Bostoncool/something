from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import shap
import xarray as xr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 路径基准：脚本所在目录及其上一级工作区根目录（便于在 ML NEW / MachineLearning 等子目录下统一读写）
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent  # 工作区根目录，与 "ML NEW" 或 "MachineLearning" 平级
THESIS_DIR = WORKSPACE_ROOT
# 读取路径：均基于工作区根目录
DATA_ROOT = WORKSPACE_ROOT / "1.模型要用的"
DEFAULT_CORRELATION_DIR = WORKSPACE_ROOT / "Correlation"
DEFAULT_DATA_READ_DIR = WORKSPACE_ROOT / "Data Read"
DEFAULT_PM25_DAY_DIR = DATA_ROOT / "2018-2023[PM2.5-china-clusters]"
DEFAULT_ERA5_DAY_DIR = DATA_ROOT / "2018-2023[ERA5_PM2.5-clusters]"
DEFAULT_CITY_GEOJSON_DIR = DATA_ROOT / "地图数据"
DEFAULT_POLLUTANT_DAY_DIRS: dict[str, Path] = {
    "co": DATA_ROOT / "[2018--2023]CO-clusters",
    "no2": DATA_ROOT / "[2018-2023]NO2-clusters",
    "o3": DATA_ROOT / "[2018-2023]O3-clusters",
    "pm10": DATA_ROOT / "[2018-2023]PM10-clusters",
    "so2": DATA_ROOT / "[2018-2023]SO2-clusters",
}

POLLUTANT_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "co": ("co", "carbon_monoxide"),
    "no2": ("no2", "nitrogen_dioxide"),
    "o3": ("o3", "ozone"),
    "pm10": ("pm10", "pm_10"),
    "so2": ("so2", "sulfur_dioxide", "sulphur_dioxide"),
}

CITY_ALIASES = ["city", "cityname", "city_name", "name", "城市", "城市名", "市名"]
DATE_ALIASES = ["date", "datetime", "time", "day", "日期", "监测日期"]
PM25_ALIASES = ["pm25", "pm2.5", "pm_25", "pm2_5", "pm2d5", "浓度", "pm25浓度"]


@dataclass(frozen=True)
class RegionConfig:
    name: str
    cities: list[str]
    city_to_province: dict[str, str]
    pm25_csv_path: Path


def progress_iter(iterable: Any, *, desc: str, total: int | None = None, leave: bool = False) -> Any:
    return tqdm(iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True)


def choose_initial_worker_count(
    *,
    task_count: int,
    cpu_cores: int,
    requested_workers: int | None = None,
    label: str = "NC",
) -> int:
    """
    Pick a safer initial worker count for NetCDF reads.

    Strategy:
    - Never exceed task_count.
    - NC 读取以磁盘 I/O 为主，并发过高会争抢磁盘带宽，反而变慢；故上限偏保守。
    - For many-core machines, cap workers to reduce memory pressure and I/O contention.
    - Respect user request but apply safe upper bound.
    """
    cpu_cores = max(1, int(cpu_cores))
    task_count = max(1, int(task_count))
    requested = cpu_cores if requested_workers is None else max(1, int(requested_workers))

    if task_count < 50:
        safe_cap = min(cpu_cores, 16)
    elif task_count < 200:
        safe_cap = min(cpu_cores, 20)
    else:
        # 大量文件时限制为 16，避免 I/O 争用导致整体变慢；若磁盘很快可传 max_workers 提高
        safe_cap = min(cpu_cores, 16)
    safe_cap = max(1, safe_cap)

    chosen = max(1, min(requested, safe_cap, task_count))
    if chosen < requested:
        print(
            f"[INFO] {label} 并发自动限流: 请求 {requested} -> 使用 {chosen} "
            f"(tasks={task_count}, cpu={cpu_cores}, safe_cap={safe_cap})"
        )
    return chosen


def iter_bounded_executor_results(
    *,
    executor: ProcessPoolExecutor,
    tasks: list[Any],
    submit_task: Callable[[ProcessPoolExecutor, Any], Any],
    max_in_flight: int,
) -> Any:
    """
    Bounded future producer:
    keep only limited submitted tasks in memory and yield results on completion.
    """
    if max_in_flight <= 0:
        max_in_flight = 1
    pending: dict[Any, Any] = {}
    task_iter = iter(tasks)

    for _ in range(max_in_flight):
        try:
            task = next(task_iter)
        except StopIteration:
            break
        future = submit_task(executor, task)
        pending[future] = task

    while pending:
        done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
        for future in done:
            task = pending.pop(future, None)
            if task is None:
                continue
            yield task, future.result()
        for _ in range(len(done)):
            try:
                task = next(task_iter)
            except StopIteration:
                break
            future = submit_task(executor, task)
            pending[future] = task


def canonical_col_name(text: str) -> str:
    return re.sub(r"[\s\-\._:/\\\(\)\[\]\{\}]+", "", str(text).strip().lower())


def sanitize_factor_name(name: str) -> str:
    text = re.sub(r"[^\w]+", "_", str(name).strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def normalize_city_name(city_name: str) -> str:
    return str(city_name).strip().replace("市", "").replace("地区", "")


def detect_column(df: pd.DataFrame, aliases: list[str], contains_tokens: list[str] | None = None) -> str | None:
    alias_set = {canonical_col_name(alias) for alias in aliases}
    columns = [str(col).strip() for col in df.columns]

    for col in columns:
        if canonical_col_name(col) in alias_set:
            return col

    if contains_tokens:
        token_set = {canonical_col_name(token) for token in contains_tokens}
        for col in columns:
            canonical = canonical_col_name(col)
            if any(token in canonical for token in token_set):
                return col
    return None


def expand_input_paths(raw_inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in raw_inputs:
        for part in [piece.strip() for piece in re.split(r"[;,]", item) if piece.strip()]:
            path = Path(part).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Input path not found: {path}")
            if path.is_file():
                paths.append(path)
            else:
                for suffix in ("*.csv", "*.parquet", "*.xlsx", "*.xls"):
                    paths.extend(sorted(path.rglob(suffix)))
    unique_paths = sorted({path.resolve() for path in paths if path.is_file()})
    if not unique_paths:
        raise FileNotFoundError("No readable daily files found in input paths.")
    return unique_paths


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def read_daily_long_file(path: Path) -> pd.DataFrame:
    raw = read_table(path).copy()
    raw.columns = [str(col).strip() for col in raw.columns]

    city_col = detect_column(raw, CITY_ALIASES, contains_tokens=["city", "城市"])
    date_col = detect_column(raw, DATE_ALIASES, contains_tokens=["date", "日期", "time"])
    pm25_col = detect_column(raw, PM25_ALIASES, contains_tokens=["pm25", "pm2.5"])

    if city_col is None or date_col is None or pm25_col is None:
        raise ValueError(
            f"Cannot detect columns in {path.name}. "
            f"Detected city={city_col}, date={date_col}, pm25={pm25_col}. "
            "Expected long-table fields like city/date/pm25."
        )

    df = raw.rename(columns={city_col: "city", date_col: "date", pm25_col: "pm25"})[
        ["city", "date", "pm25"]
    ].copy()
    date_series = pd.to_datetime(df["date"], errors="coerce")
    if date_series.isna().all():
        date_series = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    if getattr(date_series.dt, "tz", None) is not None:
        date_series = date_series.dt.tz_localize(None)

    df["date"] = date_series.dt.normalize()
    df["city"] = df["city"].map(normalize_city_name)
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df = df.dropna(subset=["city", "date", "pm25"]).copy()
    df["city"] = df["city"].astype(str).str.strip()
    return df


def load_daily_long_dataset(raw_inputs: list[str]) -> pd.DataFrame:
    files = expand_input_paths(raw_inputs)
    records: list[pd.DataFrame] = []
    failed: list[str] = []

    for file_path in progress_iter(files, desc="读取 daily 输入文件", total=len(files)):
        try:
            records.append(read_daily_long_file(file_path))
        except Exception as exc:  # pylint: disable=broad-except
            failed.append(f"{file_path.name}: {exc}")

    if not records:
        sample_errors = "\n".join(failed[:5])
        raise ValueError(f"All daily input files failed.\n{sample_errors}")

    daily = pd.concat(records, ignore_index=True)
    daily = (
        daily.groupby(["city", "date"], as_index=False)["pm25"]
        .mean()
        .sort_values(["city", "date"], kind="mergesort")
        .reset_index(drop=True)
    )
    daily["year"] = daily["date"].dt.year.astype(int)
    if failed:
        print(f"[WARN] {len(failed)} files failed and were skipped.")
    print(f"[INFO] Daily long-table samples: {len(daily):,}")
    return daily


def load_module_from_path(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_region_configs(correlation_dir: Path, module_tag: str) -> tuple[Any, list[RegionConfig]]:
    bth_module = load_module_from_path(
        correlation_dir / "BTH-Geo-detector.py",
        f"bth_geo_detector_for_{module_tag}",
    )
    yrd_module = load_module_from_path(
        correlation_dir / "YRD-Geo-detector.py",
        f"yrd_geo_detector_for_{module_tag}",
    )
    prd_module = load_module_from_path(
        correlation_dir / "PRD-Geo-detector.py",
        f"prd_geo_detector_for_{module_tag}",
    )
    configs = [
        RegionConfig(
            name="BTH",
            cities=list(getattr(bth_module, "BTH_CITIES")),
            city_to_province=dict(getattr(bth_module, "CITY_TO_PROVINCE")),
            pm25_csv_path=correlation_dir / "inputs" / "bth_pm25_city_year.csv",
        ),
        RegionConfig(
            name="YRD",
            cities=list(getattr(yrd_module, "YRD_CITIES")),
            city_to_province=dict(getattr(yrd_module, "YRD_CITY_TO_PROVINCE")),
            pm25_csv_path=correlation_dir / "inputs" / "yrd_pm25_city_year.csv",
        ),
        RegionConfig(
            name="PRD",
            cities=list(getattr(prd_module, "PRD_CITIES")),
            city_to_province=dict(getattr(prd_module, "PRD_CITY_TO_PROVINCE")),
            pm25_csv_path=correlation_dir / "inputs" / "prd_pm25_city_year.csv",
        ),
    ]
    return bth_module, configs


def build_city_cluster_map(configs: list[RegionConfig]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for config in configs:
        for city in config.cities:
            rows.append({"city": normalize_city_name(city), "cluster": config.name})
    return pd.DataFrame(rows).drop_duplicates(subset=["city"], keep="first")


def parse_day_from_filename(file_name: str) -> pd.Timestamp | None:
    match = re.search(r"(20\d{2}[01]\d[0-3]\d)", file_name)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")


def parse_month_from_filename(file_name: str) -> pd.Timestamp | None:
    match = re.search(r"(20\d{2}[01]\d)(?!\d)", file_name)
    if not match:
        return None
    return pd.to_datetime(f"{match.group(1)}01", format="%Y%m%d", errors="coerce")


def parse_year_from_filename(file_name: str) -> int | None:
    match = re.search(r"(20\d{2})(?!\d)", file_name)
    if not match:
        return None
    return int(match.group(1))


def build_date_scale_pairs_from_filename(file_name: str) -> list[tuple[pd.Timestamp, float]]:
    day_ts = parse_day_from_filename(file_name)
    if pd.notna(day_ts):
        return [(pd.Timestamp(day_ts).normalize(), 1.0)]

    month_start = parse_month_from_filename(file_name)
    if pd.notna(month_start):
        month_start = pd.Timestamp(month_start).normalize()
        month_days = int(month_start.days_in_month)
        dates = pd.date_range(month_start, periods=month_days, freq="D")
        return [(pd.Timestamp(date), 1.0 / float(month_days)) for date in dates]

    year_val = parse_year_from_filename(file_name)
    if year_val is not None:
        start = pd.Timestamp(f"{year_val}-01-01")
        end = pd.Timestamp(f"{year_val}-12-31")
        dates = pd.date_range(start, end, freq="D")
        return [(pd.Timestamp(date), 1.0 / 365.0) for date in dates]
    return []


def infer_year_from_filename(file_name: str) -> int | None:
    day_ts = parse_day_from_filename(file_name)
    if pd.notna(day_ts):
        return int(pd.Timestamp(day_ts).year)
    month_ts = parse_month_from_filename(file_name)
    if pd.notna(month_ts):
        return int(pd.Timestamp(month_ts).year)
    return parse_year_from_filename(file_name)


def infer_era5_factor_name_from_path(
    nc_file: Path,
    factor_alias_map: dict[str, tuple[str, ...]],
) -> str:
    factor_names = set(factor_alias_map.keys())
    alias_to_factor = {
        str(alias).strip().lower(): factor
        for factor, aliases in factor_alias_map.items()
        for alias in aliases
    }
    path_nodes = [node.name.strip() for node in [nc_file.parent, *nc_file.parents]]
    stem_lower = nc_file.stem.strip().lower()

    for node in path_nodes:
        if not node:
            continue
        if node in factor_names:
            return node
        mapped = alias_to_factor.get(node.lower())
        if mapped is not None:
            return mapped

    for factor in sorted(factor_names):
        if factor.lower() in stem_lower:
            return factor
    for alias, factor in alias_to_factor.items():
        if alias and alias in stem_lower:
            return factor
    return nc_file.parent.name.strip()


def infer_time_dim_name(data_array: Any) -> str | None:
    for dim in getattr(data_array, "dims", []):
        if "time" in str(dim).lower():
            return str(dim)
    return None


def reduce_non_spatial_dims(data_array: Any, time_dim: str | None) -> Any:
    keep_dims = {"latitude", "longitude", "lat", "lon"}
    if time_dim is not None:
        keep_dims.add(time_dim)
    out = data_array
    for dim in list(getattr(out, "dims", [])):
        if dim not in keep_dims:
            out = out.mean(dim=dim, skipna=True)
    return out


def get_city_bounds(city_gdf_wgs84: Any) -> tuple[float, float, float, float]:
    lon_min, lat_min, lon_max, lat_max = city_gdf_wgs84.total_bounds
    return float(lon_min), float(lat_min), float(lon_max), float(lat_max)


def clip_dataarray_by_bounds(data_array: Any, bounds: tuple[float, float, float, float]) -> Any:
    lon_min, lat_min, lon_max, lat_max = bounds
    out = data_array
    lon_dim = None
    lat_dim = None
    for candidate in ("longitude", "lon"):
        if candidate in getattr(out, "dims", []):
            lon_dim = candidate
            break
    for candidate in ("latitude", "lat"):
        if candidate in getattr(out, "dims", []):
            lat_dim = candidate
            break
    if lon_dim is None or lat_dim is None:
        return out

    lon_values = np.asarray(out[lon_dim].values)
    lat_values = np.asarray(out[lat_dim].values)
    if lon_values.size == 0 or lat_values.size == 0:
        return out

    lon_ascending = bool(lon_values[0] <= lon_values[-1])
    lat_ascending = bool(lat_values[0] <= lat_values[-1])

    def _slice_with_lon_range(one_lon_min: float, one_lon_max: float) -> Any:
        lon_slice = slice(one_lon_min, one_lon_max) if lon_ascending else slice(one_lon_max, one_lon_min)
        lat_slice = slice(lat_min, lat_max) if lat_ascending else slice(lat_max, lat_min)
        return out.sel({lon_dim: lon_slice, lat_dim: lat_slice})

    clipped = _slice_with_lon_range(lon_min, lon_max)
    lon_clipped = np.asarray(clipped[lon_dim].values)
    lat_clipped = np.asarray(clipped[lat_dim].values)
    if lon_clipped.size > 0 and lat_clipped.size > 0:
        return clipped

    # Fallback for datasets still using [0, 360] longitude convention.
    lon_min_src = float(np.nanmin(lon_values))
    lon_max_src = float(np.nanmax(lon_values))
    if lon_min_src >= 0.0 and lon_max_src > 180.0:
        shifted_min = lon_min if lon_min >= 0.0 else lon_min + 360.0
        shifted_max = lon_max if lon_max >= 0.0 else lon_max + 360.0
        shifted_clipped = _slice_with_lon_range(shifted_min, shifted_max)
        shifted_lon = np.asarray(shifted_clipped[lon_dim].values)
        shifted_lat = np.asarray(shifted_clipped[lat_dim].values)
        if shifted_lon.size > 0 and shifted_lat.size > 0:
            return shifted_clipped

    return clipped


def _stable_hash(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _load_cached_dataframe(cache_path: Path, label: str) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    try:
        data = pd.read_parquet(cache_path)
        print(f"[INFO] 命中{label}缓存: {cache_path.name} | rows={len(data):,}")
        return data
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] 读取{label}缓存失败，回退实时读取: {cache_path.name} | {exc}")
        print(traceback.format_exc())
        return None


def _save_cached_dataframe(df: pd.DataFrame, cache_path: Path, label: str) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"[INFO] 已写入{label}缓存: {cache_path.name} | rows={len(df):,}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] 写入{label}缓存失败: {cache_path.name} | {exc}")
        print(traceback.format_exc())


_READ_RESULT = TypeVar("_READ_RESULT")


def normalize_opened_nc_dataset(ds: Any, *, decode_cf: bool = True) -> Any:
    """标准化 NC 数据集坐标与维度，增强跨来源兼容性。"""
    rename_map: dict[str, str] = {}
    for tkey in ("valid_time", "forecast_time", "verification_time", "time1", "time2"):
        if tkey in ds.coords and "time" not in ds.coords:
            rename_map[tkey] = "time"
            break
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)

    # Unify longitude convention to [-180, 180] when source uses [0, 360].
    for lon_coord in ("longitude", "lon"):
        if lon_coord not in ds.coords:
            continue
        try:
            lon_values = np.asarray(ds[lon_coord].values, dtype=float)
            if lon_values.ndim != 1 or lon_values.size == 0:
                continue
            lon_min = float(np.nanmin(lon_values))
            lon_max = float(np.nanmax(lon_values))
            if lon_min >= 0.0 and lon_max > 180.0:
                normalized_lon = ((lon_values + 180.0) % 360.0) - 180.0
                ds = ds.assign_coords({lon_coord: normalized_lon})
                ds = ds.sortby(lon_coord)
        except Exception:  # pylint: disable=broad-except
            continue

    if decode_cf:
        try:
            ds = xr.decode_cf(ds)
        except Exception:  # pylint: disable=broad-except
            pass

    drop_vars = [extra for extra in ("expver", "surface") if extra in ds.variables]
    if drop_vars:
        ds = ds.drop_vars(drop_vars)

    if "number" in ds.dims:
        ds = ds.mean(dim="number", skipna=True)

    if "time" in ds.coords:
        try:
            ds = ds.sortby("time")
        except Exception:  # pylint: disable=broad-except
            pass
    return ds


def read_nc_with_fallback(
    nc_file: Path,
    reader: Callable[[Any], _READ_RESULT],
    *,
    prefer_fast_decode: bool = False,
) -> _READ_RESULT:
    attempts: list[tuple[Path, str]] = [(nc_file, "h5netcdf"), (nc_file, "netcdf4")]
    temp_dir_obj = None
    errors: list[str] = []

    try:
        decode_plans = (
            [(False, False), (True, True)] if prefer_fast_decode else [(True, True)]
        )
        for candidate_path, engine in attempts:
            for decode_times, decode_cf in decode_plans:
                try:
                    with xr.open_dataset(candidate_path, engine=engine, decode_times=decode_times) as ds:
                        normalized_ds = normalize_opened_nc_dataset(ds, decode_cf=decode_cf)
                        return reader(normalized_ds)
                except Exception as exc:  # pylint: disable=broad-except
                    errors.append(
                        f"{engine}(decode_times={decode_times},decode_cf={decode_cf})@{candidate_path.name}: {exc}"
                    )

        if "[" in str(nc_file) or "]" in str(nc_file):
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="daily_nc_")
            temp_path = Path(temp_dir_obj.name) / nc_file.name
            shutil.copy2(nc_file, temp_path)
            for engine in ("h5netcdf", "netcdf4"):
                for decode_times, decode_cf in decode_plans:
                    try:
                        with xr.open_dataset(temp_path, engine=engine, decode_times=decode_times) as ds:
                            normalized_ds = normalize_opened_nc_dataset(ds, decode_cf=decode_cf)
                            return reader(normalized_ds)
                    except Exception as exc:  # pylint: disable=broad-except
                        errors.append(
                            f"{engine}(decode_times={decode_times},decode_cf={decode_cf})@temp:{exc}"
                        )
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()

    raise RuntimeError(f"Failed reading {nc_file.name}. " + " | ".join(errors[:3]))


def infer_year_from_nc_file(nc_file: Path) -> int | None:
    try:
        resolved = str(nc_file.resolve())
        stat = nc_file.stat()
        cache_key = (resolved, int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        cache_key = (str(nc_file), -1, -1)
    if cache_key in _PM25_FILE_YEAR_CACHE:
        return _PM25_FILE_YEAR_CACHE[cache_key]

    def _reader(ds: Any) -> int | None:
        if "time" not in ds.coords:
            return None
        time_values = pd.to_datetime(pd.Index(ds["time"].values), errors="coerce")
        valid_times = pd.Series(time_values).dropna()
        if valid_times.empty:
            return None
        return int(valid_times.dt.year.min())

    try:
        inferred = read_nc_with_fallback(nc_file, _reader)
        _PM25_FILE_YEAR_CACHE[cache_key] = inferred
        return inferred
    except Exception:  # pylint: disable=broad-except
        _PM25_FILE_YEAR_CACHE[cache_key] = None
        return None


def build_city_daily_rows_from_dataarray(
    *,
    base_module: Any,
    data_array: Any,
    file_name: str,
    city_gdf_wgs84: Any,
    value_col: str,
    spatial_extractor: Callable[[Any], Any],
    city_mean_extractor: Callable[[Any, Any, str], tuple[pd.DataFrame, dict[str, float]]] | None = None,
    perf_stats: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    time_dim = infer_time_dim_name(data_array)
    out = reduce_non_spatial_dims(data_array, time_dim=time_dim)

    if time_dim is not None and time_dim in getattr(out, "dims", []):
        time_values = pd.to_datetime(pd.Index(out[time_dim].values), errors="coerce")
        norm_days = pd.Series(time_values).dt.normalize()
        valid_mask = norm_days.notna()

        if valid_mask.any():
            day_to_indices: dict[pd.Timestamp, list[int]] = {}
            valid_positions = np.flatnonzero(valid_mask.to_numpy())
            valid_days = pd.DatetimeIndex(norm_days[valid_mask])
            for pos, day in zip(valid_positions, valid_days):
                day_to_indices.setdefault(pd.Timestamp(day), []).append(int(pos))
            for day in sorted(day_to_indices):
                idx = day_to_indices[day]
                day_slice = out.isel({time_dim: idx})
                day_field = day_slice.mean(dim=time_dim, skipna=True)
                field_2d = spatial_extractor(day_field)
                if city_mean_extractor is not None:
                    city_df, metrics = city_mean_extractor(field_2d, city_gdf_wgs84, value_col)
                    if perf_stats is not None and metrics:
                        for key, val in metrics.items():
                            perf_stats[key] = float(perf_stats.get(key, 0.0)) + float(val)
                else:
                    city_df = base_module._extract_city_mean_from_2d_field(field_2d, city_gdf_wgs84, value_col)
                city_df = city_df[["city", value_col]].copy()
                city_df["city"] = city_df["city"].map(normalize_city_name)
                city_df["date"] = day
                city_df[value_col] = pd.to_numeric(city_df[value_col], errors="coerce")
                rows.append(city_df[["city", "date", value_col]])

    if rows:
        return pd.concat(rows, ignore_index=True)

    date_scale_pairs = build_date_scale_pairs_from_filename(file_name)
    if not date_scale_pairs:
        return pd.DataFrame(columns=["city", "date", value_col])

    field_2d = spatial_extractor(out)
    if city_mean_extractor is not None:
        city_base, metrics = city_mean_extractor(field_2d, city_gdf_wgs84, value_col)
        if perf_stats is not None and metrics:
            for key, val in metrics.items():
                perf_stats[key] = float(perf_stats.get(key, 0.0)) + float(val)
    else:
        city_base = base_module._extract_city_mean_from_2d_field(field_2d, city_gdf_wgs84, value_col)
    city_base = city_base[["city", value_col]].copy()
    city_base["city"] = city_base["city"].map(normalize_city_name)
    city_base[value_col] = pd.to_numeric(city_base[value_col], errors="coerce")

    for one_date, scale in date_scale_pairs:
        tmp = city_base.copy()
        tmp["date"] = pd.Timestamp(one_date).normalize()
        tmp[value_col] = tmp[value_col] * float(scale)
        rows.append(tmp[["city", "date", value_col]])
    return pd.concat(rows, ignore_index=True)


_PM25_WORKER_CONTEXT: dict[str, Any] = {}
_ERA5_WORKER_CONTEXT: dict[str, Any] = {}
_YEAR_FACTOR_PANEL_CACHE: dict[str, pd.DataFrame] = {}
_PM25_FILE_YEAR_CACHE: dict[tuple[str, int, int], int | None] = {}
_PM25_GRID_CITY_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_ERA5_GRID_CITY_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _coord_signature(values: np.ndarray) -> tuple[Any, ...]:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return (0,)
    mid_idx = int(arr.size // 2)
    step = float(arr[1] - arr[0]) if arr.size > 1 else 0.0
    return (
        int(arr.size),
        round(float(arr[0]), 6),
        round(float(arr[mid_idx]), 6),
        round(float(arr[-1]), 6),
        round(float(np.nanmin(arr)), 6),
        round(float(np.nanmax(arr)), 6),
        round(step, 6),
    )


def _infer_region_from_path(path_like: str | Path) -> str | None:
    parts = [part.strip().upper() for part in Path(path_like).parts if str(part).strip()]
    for key in ("BTH", "YRD", "PRD"):
        if key in parts:
            return key
    return None


def _build_chunked_paths(file_paths: list[str], chunk_size: int) -> list[tuple[str, ...]]:
    size = max(1, int(chunk_size))
    return [tuple(file_paths[idx : idx + size]) for idx in range(0, len(file_paths), size)]


def _build_chunked_era5_tasks(
    tasks: list[tuple[str, str, str]],
    chunk_size: int,
) -> list[tuple[tuple[str, str, str], ...]]:
    """将 (path, folder_name, value_col) 任务列表按 chunk_size 打包，用于降低调度与 IPC 开销。"""
    size = max(1, int(chunk_size))
    return [tuple(tasks[idx : idx + size]) for idx in range(0, len(tasks), size)]


def _extract_city_mean_with_grid_cache(
    *,
    da2d: Any,
    city_gdf_wgs84: Any,
    value_col: str,
    base_module: Any,
    region_name: str | None = None,
    worker_context: dict[str, Any] | None = None,
    grid_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """网格→城市映射缓存聚合。worker_context/grid_cache 为 None 时使用 PM2.5 全局变量（供 ERA5 传入自有 context/cache）。"""
    metrics = {
        "spatial_aggregate_time": 0.0,
        "cache_build_hits": 0.0,
        "cache_build_misses": 0.0,
    }
    begin = time.perf_counter()
    ctx = worker_context if worker_context is not None else _PM25_WORKER_CONTEXT
    cache = grid_cache if grid_cache is not None else _PM25_GRID_CITY_CACHE
    regional_city_gdf = city_gdf_wgs84
    try:
        gpd_module = getattr(base_module, "gpd", None)
        if gpd_module is None:
            raise RuntimeError("geopandas is unavailable in base module")

        region_map = ctx.get("region_city_gdf")
        if isinstance(region_map, dict) and region_name in region_map:
            regional_city_gdf = region_map[region_name]

        lon = da2d["longitude"].to_numpy()
        lat = da2d["latitude"].to_numpy()
        values = da2d.to_numpy().astype(float, copy=False)

        lon_float = np.asarray(lon, dtype=float)
        if lon_float.ndim == 1 and lon_float.size > 0:
            lon_min_raw = float(np.nanmin(lon_float))
            lon_max_raw = float(np.nanmax(lon_float))
            if lon_min_raw >= 0.0 and lon_max_raw > 180.0:
                lon_wrapped = ((lon_float + 180.0) % 360.0) - 180.0
                lon_order = np.argsort(lon_wrapped)
                lon = lon_wrapped[lon_order]
                values = values[:, lon_order]

        lat_is_desc = bool(lat[0] > lat[-1])
        lon_min, lat_min, lon_max, lat_max = regional_city_gdf.total_bounds
        if lat_is_desc:
            lat_idx = np.where((lat <= lat_max) & (lat >= lat_min))[0]
        else:
            lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        if len(lat_idx) == 0 or len(lon_idx) == 0:
            raise ValueError("NC 与目标城市范围无重叠。")

        lat_sub = np.asarray(lat[lat_idx], dtype=float)
        lon_sub = np.asarray(lon[lon_idx], dtype=float)
        val_sub = values[np.ix_(lat_idx, lon_idx)]

        cache_key = (
            region_name or "ALL",
            _coord_signature(lat_sub),
            _coord_signature(lon_sub),
            round(float(lon_min), 6),
            round(float(lat_min), 6),
            round(float(lon_max), 6),
            round(float(lat_max), 6),
            int(len(regional_city_gdf)),
        )
        cache_item = cache.get(cache_key)

        if cache_item is None:
            metrics["cache_build_misses"] += 1.0
            lon_mesh = np.tile(lon_sub, len(lat_sub))
            lat_mesh = np.repeat(lat_sub, len(lon_sub))
            points = gpd_module.GeoDataFrame(
                {"flat_idx": np.arange(lon_mesh.size, dtype=np.int64)},
                geometry=gpd_module.points_from_xy(lon_mesh, lat_mesh),
                crs="EPSG:4326",
            )
            joined = gpd_module.sjoin(
                points,
                regional_city_gdf[["city", "city_norm", "geometry"]],
                how="inner",
                predicate="intersects",
            )
            if joined.empty:
                raise ValueError("NC 像元未匹配到任何目标城市边界。")
            joined = joined.loc[:, ["flat_idx", "city"]].drop_duplicates(subset=["flat_idx"], keep="first")
            city_names = sorted(joined["city"].dropna().astype(str).unique().tolist())
            if not city_names:
                raise ValueError("城市映射为空。")
            city_to_code = {name: idx for idx, name in enumerate(city_names)}
            cache_item = {
                "flat_idx": joined["flat_idx"].to_numpy(dtype=np.int64, copy=False),
                "city_code": joined["city"].map(city_to_code).to_numpy(dtype=np.int64, copy=False),
                "city_names": city_names,
            }
            cache[cache_key] = cache_item
        else:
            metrics["cache_build_hits"] += 1.0

        flat_values = np.asarray(val_sub, dtype=float).reshape(-1)
        map_flat_idx = np.asarray(cache_item["flat_idx"], dtype=np.int64)
        map_city_code = np.asarray(cache_item["city_code"], dtype=np.int64)
        mapped_values = flat_values[map_flat_idx]
        valid_mask = np.isfinite(mapped_values)
        if not valid_mask.any():
            return pd.DataFrame(columns=["city", value_col]), metrics
        city_names = list(cache_item["city_names"])
        city_count = len(city_names)
        sums = np.bincount(map_city_code[valid_mask], weights=mapped_values[valid_mask], minlength=city_count)
        counts = np.bincount(map_city_code[valid_mask], minlength=city_count)
        valid_city_mask = counts > 0
        means = np.divide(
            sums[valid_city_mask],
            counts[valid_city_mask],
            out=np.zeros(valid_city_mask.sum(), dtype=float),
            where=counts[valid_city_mask] > 0,
        )
        out_df = pd.DataFrame(
            {
                "city": [city_names[idx] for idx in np.flatnonzero(valid_city_mask)],
                value_col: means,
            }
        ).dropna(subset=[value_col])
        return out_df, metrics
    except Exception:  # pylint: disable=broad-except
        fallback_df = base_module._extract_city_mean_from_2d_field(da2d, regional_city_gdf, value_col)
        return fallback_df, metrics
    finally:
        metrics["spatial_aggregate_time"] += time.perf_counter() - begin


def _scan_pm25_nc_files(pm25_day_dir: Path) -> list[Path]:
    if not pm25_day_dir.exists():
        raise FileNotFoundError(f"PM2.5 day directory not found: {pm25_day_dir}")
    all_files = sorted(path for path in pm25_day_dir.rglob("*.nc") if path.is_file())
    if not all_files:
        raise FileNotFoundError(f"No .nc files found in PM2.5 day directory: {pm25_day_dir}")
    return all_files


def _resolve_pm25_file_year(nc_file: Path) -> int | None:
    """从文件名或（若失败）打开 NC 读 time 维解析年份。打开文件较慢，构建索引时宜用仅文件名版本。"""
    file_year = infer_year_from_filename(nc_file.name)
    if file_year is not None:
        return int(file_year)
    return infer_year_from_nc_file(nc_file)


def _resolve_pm25_file_year_from_name_only(nc_file: Path) -> int | None:
    """仅从文件名解析年份，不打开 NC 文件。用于索引构建等需要快速扫描的场景。"""
    file_year = infer_year_from_filename(nc_file.name)
    return int(file_year) if file_year is not None else None


def _filter_nc_files_by_year(nc_files: list[Path], allowed_years: set[int] | None) -> list[Path]:
    if not allowed_years:
        return nc_files
    selected_files: list[Path] = []
    for nc_file in nc_files:
        file_year = _resolve_pm25_file_year(nc_file)
        if file_year is None or file_year in allowed_years:
            selected_files.append(nc_file)
    return selected_files


def build_pm25_nc_file_index(
    pm25_day_dir: Path,
    *,
    resolve_year_without_open: bool = True,
) -> dict[str, Any]:
    """
    构建 PM2.5 NC 文件索引。
    默认 resolve_year_without_open=True：仅从文件名解析年份，不打开 NC 文件，避免构建索引时
    对大量「文件名无年份」的文件做慢速 open，从而显著提速；无法从文件名解析的将归入 unresolved。
    """
    all_files = _scan_pm25_nc_files(pm25_day_dir)

    year_to_files: dict[int, list[Path]] = {}
    unresolved_year_files: list[Path] = []
    resolve_fn = _resolve_pm25_file_year_from_name_only if resolve_year_without_open else _resolve_pm25_file_year
    for nc_file in all_files:
        file_year = resolve_fn(nc_file)
        if file_year is None:
            unresolved_year_files.append(nc_file)
            continue
        year_to_files.setdefault(int(file_year), []).append(nc_file)

    print(
        "[INFO] 已构建 PM2.5 NC 文件索引: "
        f"total={len(all_files)}, years={len(year_to_files)}, unresolved={len(unresolved_year_files)}"
    )
    return {
        "all_files": all_files,
        "year_to_files": year_to_files,
        "unresolved_year_files": unresolved_year_files,
    }


def select_pm25_nc_files_from_index(
    pm25_nc_index: dict[str, Any],
    allowed_years: set[int] | None = None,
) -> list[Path]:
    all_files = list(pm25_nc_index.get("all_files", []))
    if not allowed_years:
        return all_files

    year_to_files = pm25_nc_index.get("year_to_files", {})
    unresolved_year_files = list(pm25_nc_index.get("unresolved_year_files", []))
    selected: list[Path] = []
    for year in sorted(allowed_years):
        selected.extend(year_to_files.get(int(year), []))
    selected.extend(unresolved_year_files)
    dedup: dict[str, Path] = {}
    for one in selected:
        dedup[str(one.resolve())] = one
    return sorted(dedup.values())


def _init_pm25_worker(
    base_module_path: str,
    city_geojson_dir: str,
    city_list: list[str],
    city_to_cluster: dict[str, str] | None = None,
    pm25_precropped: bool = False,
) -> None:
    global _PM25_WORKER_CONTEXT, _PM25_GRID_CITY_CACHE  # pylint: disable=global-statement
    module = load_module_from_path(Path(base_module_path), f"pm25_reader_worker_{os.getpid()}")
    city_gdf_wgs84 = module._load_city_geojson(Path(city_geojson_dir), city_list)
    city_gdf_wgs84 = city_gdf_wgs84.copy()
    city_gdf_wgs84["city"] = city_gdf_wgs84["city"].map(normalize_city_name)
    city_gdf_wgs84["city_norm"] = city_gdf_wgs84["city"].map(normalize_city_name)
    region_city_gdf: dict[str, Any] = {}
    if city_to_cluster:
        city_to_cluster_norm = {
            normalize_city_name(city): str(cluster).upper()
            for city, cluster in city_to_cluster.items()
        }
        city_gdf_wgs84["cluster"] = city_gdf_wgs84["city_norm"].map(city_to_cluster_norm)
        for region_name in ("BTH", "YRD", "PRD"):
            one_gdf = city_gdf_wgs84.loc[city_gdf_wgs84["cluster"] == region_name].copy()
            if not one_gdf.empty:
                region_city_gdf[region_name] = one_gdf

    ctx: dict[str, Any] = {
        "base_module": module,
        "city_gdf_wgs84": city_gdf_wgs84,
        "pm25_precropped": bool(pm25_precropped),
        "region_city_gdf": region_city_gdf,
    }
    if not pm25_precropped:
        ctx["city_bounds"] = get_city_bounds(city_gdf_wgs84)
    _PM25_WORKER_CONTEXT = ctx
    _PM25_GRID_CITY_CACHE = {}


def _read_single_pm25_nc_worker(
    nc_file_str: str,
) -> tuple[str, pd.DataFrame | None, str | None, dict[str, float]]:
    nc_file = Path(nc_file_str)
    base_module = _PM25_WORKER_CONTEXT["base_module"]
    city_gdf_wgs84 = _PM25_WORKER_CONTEXT["city_gdf_wgs84"]
    pm25_precropped = _PM25_WORKER_CONTEXT.get("pm25_precropped", False)
    city_bounds = _PM25_WORKER_CONTEXT.get("city_bounds")
    region_name = _infer_region_from_path(nc_file_str)
    return _read_single_pm25_nc_core(
        nc_file,
        base_module,
        city_gdf_wgs84,
        city_bounds,
        skip_clip=pm25_precropped,
        region_name=region_name,
    )


def _read_pm25_nc_chunk_worker(file_chunk: tuple[str, ...]) -> list[tuple[str, pd.DataFrame | None, str | None, dict[str, float]]]:
    results: list[tuple[str, pd.DataFrame | None, str | None, dict[str, float]]] = []
    for file_path in file_chunk:
        results.append(_read_single_pm25_nc_worker(file_path))
    return results


def _read_single_pm25_nc_core(
    nc_file: Path,
    base_module: Any,
    city_gdf_wgs84: Any,
    city_bounds: tuple[float, float, float, float] | None,
    *,
    skip_clip: bool = False,
    region_name: str | None = None,
) -> tuple[str, pd.DataFrame | None, str | None, dict[str, float]]:
    file_name = nc_file.name
    perf_stats = {
        "open_dataset_time": 0.0,
        "spatial_aggregate_time": 0.0,
        "cache_build_hits": 0.0,
        "cache_build_misses": 0.0,
    }

    def reader(ds: Any) -> pd.DataFrame:
        standardized = base_module._standardize_nc_coords(ds)
        pm_var_name = base_module._choose_pm25_var(standardized)
        if skip_clip or city_bounds is None:
            pm_da = standardized[pm_var_name]
        else:
            pm_da = clip_dataarray_by_bounds(standardized[pm_var_name], city_bounds)
        return build_city_daily_rows_from_dataarray(
            base_module=base_module,
            data_array=pm_da,
            file_name=file_name,
            city_gdf_wgs84=city_gdf_wgs84,
            value_col="pm25",
            spatial_extractor=base_module._extract_2d_pm25,
            city_mean_extractor=lambda da2d, city_gdf, value_col: _extract_city_mean_with_grid_cache(
                da2d=da2d,
                city_gdf_wgs84=city_gdf,
                value_col=value_col,
                base_module=base_module,
                region_name=region_name,
            ),
            perf_stats=perf_stats,
        )

    try:
        read_begin = time.perf_counter()
        one_df = read_nc_with_fallback(nc_file, reader, prefer_fast_decode=True)
        total_read = time.perf_counter() - read_begin
        perf_stats["open_dataset_time"] += max(0.0, total_read - perf_stats["spatial_aggregate_time"])
        return file_name, one_df, None, perf_stats
    except Exception as exc:  # pylint: disable=broad-except
        return file_name, None, str(exc), perf_stats


def _read_single_pm25_nc_local(
    nc_file: Path,
    base_module: Any,
    city_gdf_wgs84: Any,
    city_bounds: tuple[float, float, float, float] | None,
    *,
    skip_clip: bool = False,
    region_name: str | None = None,
) -> tuple[str, pd.DataFrame | None, str | None, dict[str, float]]:
    return _read_single_pm25_nc_core(
        nc_file,
        base_module,
        city_gdf_wgs84,
        city_bounds,
        skip_clip=skip_clip,
        region_name=region_name,
    )


def _init_era5_worker(
    base_module_path: str,
    met_module_path: str,
    city_geojson_dir: str,
    city_list: list[str],
    city_to_cluster: dict[str, str] | None = None,
) -> None:
    global _ERA5_WORKER_CONTEXT, _ERA5_GRID_CITY_CACHE  # pylint: disable=global-statement
    base_module = load_module_from_path(Path(base_module_path), f"era5_base_worker_{os.getpid()}")
    met_module = load_module_from_path(Path(met_module_path), f"era5_met_worker_{os.getpid()}")
    city_gdf_wgs84 = base_module._load_city_geojson(Path(city_geojson_dir), city_list)
    city_gdf_wgs84 = city_gdf_wgs84.copy()
    city_gdf_wgs84["city"] = city_gdf_wgs84["city"].map(normalize_city_name)
    city_gdf_wgs84["city_norm"] = city_gdf_wgs84["city"].map(normalize_city_name)
    region_city_gdf: dict[str, Any] = {}
    if city_to_cluster:
        city_to_cluster_norm = {
            normalize_city_name(city): str(cluster).upper()
            for city, cluster in city_to_cluster.items()
        }
        city_gdf_wgs84["cluster"] = city_gdf_wgs84["city_norm"].map(city_to_cluster_norm)
        for region_name in ("BTH", "YRD", "PRD"):
            one_gdf = city_gdf_wgs84.loc[city_gdf_wgs84["cluster"] == region_name].copy()
            if not one_gdf.empty:
                region_city_gdf[region_name] = one_gdf
    _ERA5_GRID_CITY_CACHE = {}
    _ERA5_WORKER_CONTEXT = {
        "base_module": base_module,
        "city_gdf_wgs84": city_gdf_wgs84,
        "unit_conversions": dict(getattr(met_module, "UNIT_CONVERSIONS", {})),
        "region_city_gdf": region_city_gdf,
    }


def _read_single_era5_nc_worker(
    nc_file_str: str,
    folder_name: str,
    value_col: str,
    alias_map: dict[str, tuple[str, ...]],
) -> tuple[str, pd.DataFrame | None, str | None, dict[str, float]]:
    nc_file = Path(nc_file_str)
    base_module = _ERA5_WORKER_CONTEXT["base_module"]
    city_gdf_wgs84 = _ERA5_WORKER_CONTEXT["city_gdf_wgs84"]
    unit_conversions: dict[str, Any] = _ERA5_WORKER_CONTEXT["unit_conversions"]
    return _read_single_era5_nc_core(
        nc_file=nc_file,
        folder_name=folder_name,
        value_col=value_col,
        alias_map=alias_map,
        base_module=base_module,
        city_gdf_wgs84=city_gdf_wgs84,
        unit_conversions=unit_conversions,
    )


def _read_era5_nc_chunk_worker(
    task_chunk: tuple[tuple[str, str, str], ...],
    alias_map: dict[str, tuple[str, ...]],
) -> list[tuple[str, pd.DataFrame | None, str | None, dict[str, float]]]:
    results: list[tuple[str, pd.DataFrame | None, str | None, dict[str, float]]] = []
    for nc_file_str, folder_name, value_col in task_chunk:
        results.append(
            _read_single_era5_nc_worker(nc_file_str, folder_name, value_col, alias_map)
        )
    return results


def _read_single_era5_nc_core(
    *,
    nc_file: Path,
    folder_name: str,
    value_col: str,
    alias_map: dict[str, tuple[str, ...]],
    base_module: Any,
    city_gdf_wgs84: Any,
    unit_conversions: dict[str, Any],
) -> tuple[str, pd.DataFrame | None, str | None, dict[str, float]]:
    file_name = nc_file.name
    region_name = _infer_region_from_path(nc_file)
    perf_stats = {
        "open_dataset_time": 0.0,
        "spatial_aggregate_time": 0.0,
        "cache_build_hits": 0.0,
        "cache_build_misses": 0.0,
    }

    def reader(ds: Any) -> pd.DataFrame:
        standardized = base_module._standardize_nc_coords(ds)
        preferred = [folder_name, *alias_map.get(folder_name, ())]
        var_name = base_module._choose_spatial_var(standardized, preferred_names=preferred)
        data_array = standardized[var_name]

        convert_fn = unit_conversions.get(folder_name) or unit_conversions.get(var_name)
        if callable(convert_fn):
            data_array = convert_fn(data_array)

        return build_city_daily_rows_from_dataarray(
            base_module=base_module,
            data_array=data_array,
            file_name=file_name,
            city_gdf_wgs84=city_gdf_wgs84,
            value_col=value_col,
            spatial_extractor=base_module._extract_2d_spatial_field,
            city_mean_extractor=lambda da2d, city_gdf, vcol: _extract_city_mean_with_grid_cache(
                da2d=da2d,
                city_gdf_wgs84=city_gdf,
                value_col=vcol,
                base_module=base_module,
                region_name=region_name,
                worker_context=_ERA5_WORKER_CONTEXT,
                grid_cache=_ERA5_GRID_CITY_CACHE,
            ),
            perf_stats=perf_stats,
        )

    try:
        read_begin = time.perf_counter()
        one_df = read_nc_with_fallback(nc_file, reader, prefer_fast_decode=True)
        total_read = time.perf_counter() - read_begin
        perf_stats["open_dataset_time"] += max(0.0, total_read - perf_stats["spatial_aggregate_time"])
        return file_name, one_df, None, perf_stats
    except Exception as exc:  # pylint: disable=broad-except
        return file_name, None, str(exc), perf_stats


def _read_single_era5_nc_local(
    nc_file: Path,
    folder_name: str,
    value_col: str,
    alias_map: dict[str, tuple[str, ...]],
    base_module: Any,
    city_gdf_wgs84: Any,
    unit_conversions: dict[str, Any],
) -> tuple[str, pd.DataFrame | None, str | None, dict[str, float]]:
    return _read_single_era5_nc_core(
        nc_file=nc_file,
        folder_name=folder_name,
        value_col=value_col,
        alias_map=alias_map,
        base_module=base_module,
        city_gdf_wgs84=city_gdf_wgs84,
        unit_conversions=unit_conversions,
    )


def load_pm25_daily_from_nc(
    *,
    pm25_day_dir: Path,
    city_geojson_dir: Path,
    base_module: Any,
    city_list: list[str],
    allowed_years: set[int] | None = None,
    max_workers: int | None = None,
    pm25_nc_index: dict[str, Any] | None = None,
    city_to_cluster: dict[str, str] | None = None,
    pm25_precropped: bool = False,
    perf_stats: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    从 NC 文件读取 PM2.5 日值。pm25_precropped=True 时跳过空间裁剪（数据已按城市群裁剪好）。
    """
    if not pm25_day_dir.exists():
        raise FileNotFoundError(f"PM2.5 day directory not found: {pm25_day_dir}")
    if not city_geojson_dir.exists():
        raise FileNotFoundError(f"City geojson directory not found: {city_geojson_dir}")

    if pm25_nc_index is not None:
        nc_files = select_pm25_nc_files_from_index(pm25_nc_index, allowed_years)
    else:
        nc_files = _scan_pm25_nc_files(pm25_day_dir)
        nc_files = _filter_nc_files_by_year(nc_files, allowed_years)
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in PM2.5 day directory: {pm25_day_dir}")

    rows: list[pd.DataFrame] = []
    perf_totals = {
        "open_dataset_time": 0.0,
        "spatial_aggregate_time": 0.0,
        "cache_build_hits": 0.0,
        "cache_build_misses": 0.0,
        "processed_files": 0.0,
        "failed_files": 0.0,
    }
    read_begin = time.perf_counter()
    cpu_cores = os.cpu_count() or 1
    worker_count = choose_initial_worker_count(
        task_count=len(nc_files),
        cpu_cores=cpu_cores,
        requested_workers=max_workers,
        label="PM2.5 NC",
    )
    city_gdf_wgs84_main = base_module._load_city_geojson(city_geojson_dir, city_list)
    city_bounds = None if pm25_precropped else get_city_bounds(city_gdf_wgs84_main)
    print(f"[INFO] PM2.5 NC 初始并发进程数: {worker_count} (CPU cores={cpu_cores})")
    if pm25_precropped:
        print("[INFO] PM2.5 数据已按城市群预裁剪，跳过空间裁剪步骤")
    else:
        print(f"[INFO] PM2.5 空间裁剪范围: lon[{city_bounds[0]:.4f}, {city_bounds[2]:.4f}], lat[{city_bounds[1]:.4f}, {city_bounds[3]:.4f}]")
    if allowed_years:
        print(f"[INFO] PM2.5 年份过滤: {sorted(allowed_years)} | 命中文件数: {len(nc_files)}")

    base_module_path = str(Path(getattr(base_module, "__file__")).resolve())
    file_paths = [str(path.resolve()) for path in nc_files]
    current_workers = worker_count
    remaining_file_paths = list(file_paths)
    pbar = tqdm(total=len(nc_files), desc="读取 PM2.5 NC 文件", dynamic_ncols=True)
    while remaining_file_paths:
        chunk_size = max(8, min(32, len(remaining_file_paths) // max(1, current_workers * 4)))
        remaining_chunks = _build_chunked_paths(remaining_file_paths, chunk_size=chunk_size)
        processed_in_round: set[str] = set()
        max_in_flight = min(len(remaining_chunks), max(4, current_workers * 2))
        try:
            with ProcessPoolExecutor(
                max_workers=current_workers,
                initializer=_init_pm25_worker,
                initargs=(
                    base_module_path,
                    str(city_geojson_dir.resolve()),
                    city_list,
                    city_to_cluster,
                    pm25_precropped,
                ),
            ) as executor:
                for one_chunk, result_list in iter_bounded_executor_results(
                    executor=executor,
                    tasks=remaining_chunks,
                    submit_task=lambda one_executor, one_paths: one_executor.submit(_read_pm25_nc_chunk_worker, one_paths),
                    max_in_flight=max_in_flight,
                ):
                    for task_path in one_chunk:
                        processed_in_round.add(task_path)
                    for file_name, one_df, err_msg, one_perf in result_list:
                        pbar.update(1)
                        perf_totals["processed_files"] += 1.0
                        for key in ("open_dataset_time", "spatial_aggregate_time", "cache_build_hits", "cache_build_misses"):
                            perf_totals[key] += float(one_perf.get(key, 0.0))
                        if err_msg is not None:
                            perf_totals["failed_files"] += 1.0
                            print(f"[WARN] Skip PM2.5 file: {file_name} | {err_msg}")
                        elif one_df is not None and not one_df.empty:
                            rows.append(one_df)
            remaining_file_paths = []
        except (BrokenProcessPool, MemoryError, OSError, RuntimeError) as exc:
            remaining_file_paths = [path for path in remaining_file_paths if path not in processed_in_round]
            if current_workers <= 1:
                print(f"[WARN] 多进程读取失败，回退单进程读取。原因: {exc}")
                for idx, nc_file in enumerate(
                    progress_iter(
                        remaining_file_paths,
                        desc="读取 PM2.5 NC 文件[单进程]",
                        total=len(remaining_file_paths),
                    ),
                    start=1,
                ):
                    file_name, one_df, err_msg, one_perf = _read_single_pm25_nc_local(
                        Path(nc_file),
                        base_module,
                        city_gdf_wgs84_main,
                        city_bounds,
                        skip_clip=pm25_precropped,
                        region_name=_infer_region_from_path(nc_file),
                    )
                    pbar.update(1)
                    perf_totals["processed_files"] += 1.0
                    for key in ("open_dataset_time", "spatial_aggregate_time", "cache_build_hits", "cache_build_misses"):
                        perf_totals[key] += float(one_perf.get(key, 0.0))
                    if err_msg is not None:
                        perf_totals["failed_files"] += 1.0
                        print(f"[WARN] Skip PM2.5 file: {file_name} | {err_msg}")
                    elif one_df is not None and not one_df.empty:
                        rows.append(one_df)
                remaining_file_paths = []
                break
            next_workers = max(1, current_workers // 2)
            print(
                "[WARN] PM2.5 多进程池异常退出，"
                f"进程数将从 {current_workers} 降到 {next_workers} 后重试。"
                f"剩余文件数: {len(remaining_file_paths)}。原因: {exc}"
            )
            current_workers = next_workers
    pbar.close()

    if not rows:
        raise ValueError("No valid PM2.5 daily samples extracted from NetCDF files.")

    daily = pd.concat(rows, ignore_index=True)
    daily = (
        daily.groupby(["city", "date"], as_index=False)["pm25"]
        .mean()
        .sort_values(["city", "date"], kind="mergesort")
        .reset_index(drop=True)
    )
    daily["year"] = daily["date"].dt.year.astype(int)
    elapsed = time.perf_counter() - read_begin
    files_done = int(perf_totals["processed_files"])
    fps = float(files_done / elapsed) if elapsed > 0 else 0.0
    print(
        "[INFO] PM2.5 读取统计: "
        f"files={files_done}, failed={int(perf_totals['failed_files'])}, "
        f"elapsed={elapsed:.2f}s, files_per_second={fps:.2f}, "
        f"open_dataset_time={perf_totals['open_dataset_time']:.2f}s, "
        f"spatial_aggregate_time={perf_totals['spatial_aggregate_time']:.2f}s, "
        f"cache_hits={int(perf_totals['cache_build_hits'])}, "
        f"cache_misses={int(perf_totals['cache_build_misses'])}"
    )
    if perf_stats is not None:
        perf_stats.update(
            {
                "open_dataset_time": float(perf_totals["open_dataset_time"]),
                "spatial_aggregate_time": float(perf_totals["spatial_aggregate_time"]),
                "cache_build_hits": int(perf_totals["cache_build_hits"]),
                "cache_build_misses": int(perf_totals["cache_build_misses"]),
                "processed_files": int(perf_totals["processed_files"]),
                "failed_files": int(perf_totals["failed_files"]),
                "files_per_second": float(fps),
                "elapsed_seconds": float(elapsed),
            }
        )
    print(f"[INFO] PM2.5 daily samples from NC: {len(daily):,}")
    return daily


def load_era5_daily_features_from_nc(
    *,
    era5_day_dir: Path,
    city_geojson_dir: Path,
    base_module: Any,
    city_list: list[str],
    data_read_dir: Path,
    module_tag: str,
    allowed_years: set[int] | None = None,
    max_workers: int | None = None,
    city_to_cluster: dict[str, str] | None = None,
    perf_stats: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    从 NC 读取 ERA5 日尺度气象特征。已做与 PM2.5 同类的优化：网格→城市缓存、按区域城市子集、
    chunk 批量任务、快速解码与性能统计。
    """
    if not era5_day_dir.exists():
        raise FileNotFoundError(f"ERA5 day directory not found: {era5_day_dir}")
    if not city_geojson_dir.exists():
        raise FileNotFoundError(f"City geojson directory not found: {city_geojson_dir}")

    met_module = load_module_from_path(
        data_read_dir / "Meteorology.py",
        f"meteorology_reader_for_{module_tag}",
    )
    unit_conversions: dict[str, Any] = dict(getattr(met_module, "UNIT_CONVERSIONS", {}))
    alias_map = {
        "2m_dewpoint_temperature": ("d2m",),
        "2m_temperature": ("t2m",),
        "10m_u_component_of_wind": ("u10",),
        "10m_v_component_of_wind": ("v10",),
        "mean_sea_level_pressure": ("msl",),
        "total_cloud_cover": ("tcc",),
        "total_precipitation": ("tp",),
    }

    city_gdf_wgs84 = base_module._load_city_geojson(city_geojson_dir, city_list)
    nc_files = sorted(path for path in era5_day_dir.rglob("*.nc") if path.is_file())
    nc_files = _filter_nc_files_by_year(nc_files, allowed_years)
    if not nc_files:
        print(f"[WARN] No ERA5 .nc files found: {era5_day_dir}")
        return pd.DataFrame(columns=["city", "date"])
    if allowed_years:
        print(f"[INFO] ERA5 年份过滤: {sorted(allowed_years)} | 命中文件数: {len(nc_files)}")

    cpu_cores = os.cpu_count() or 1
    worker_count = choose_initial_worker_count(
        task_count=len(nc_files),
        cpu_cores=cpu_cores,
        requested_workers=max_workers,
        label="ERA5 NC",
    )
    print(f"[INFO] ERA5 NC 初始并发进程数: {worker_count} (CPU cores={cpu_cores})")

    met_module_path = str((data_read_dir / "Meteorology.py").resolve())
    base_module_path = str(Path(getattr(base_module, "__file__")).resolve())
    alias_map_tuple = {key: tuple(val) for key, val in alias_map.items()}

    rows: list[pd.DataFrame] = []
    tasks: list[tuple[str, str, str]] = []
    for nc_file in nc_files:
        folder_name = infer_era5_factor_name_from_path(nc_file, alias_map)
        value_col = f"met_{sanitize_factor_name(folder_name)}"
        tasks.append((str(nc_file.resolve()), folder_name, value_col))

    perf_totals = {
        "open_dataset_time": 0.0,
        "spatial_aggregate_time": 0.0,
        "cache_build_hits": 0.0,
        "cache_build_misses": 0.0,
        "processed_files": 0.0,
        "failed_files": 0.0,
    }
    read_begin = time.perf_counter()
    current_workers = worker_count
    remaining_tasks = list(tasks)
    pbar = tqdm(total=len(tasks), desc="读取 ERA5 NC 文件", dynamic_ncols=True)
    while remaining_tasks:
        chunk_size = max(8, min(32, len(remaining_tasks) // max(1, current_workers * 4)))
        remaining_chunks = _build_chunked_era5_tasks(remaining_tasks, chunk_size=chunk_size)
        processed_in_round: set[tuple[str, str, str]] = set()
        max_in_flight = min(len(remaining_chunks), max(4, current_workers * 2))
        try:
            with ProcessPoolExecutor(
                max_workers=current_workers,
                initializer=_init_era5_worker,
                initargs=(
                    base_module_path,
                    met_module_path,
                    str(city_geojson_dir.resolve()),
                    city_list,
                    city_to_cluster,
                ),
            ) as executor:
                for one_chunk, result_list in iter_bounded_executor_results(
                    executor=executor,
                    tasks=remaining_chunks,
                    submit_task=lambda one_executor, one_chunk: one_executor.submit(
                        _read_era5_nc_chunk_worker, one_chunk, alias_map_tuple
                    ),
                    max_in_flight=max_in_flight,
                ):
                    for task in one_chunk:
                        processed_in_round.add(task)
                    for file_name, one_df, err_msg, one_perf in result_list:
                        pbar.update(1)
                        perf_totals["processed_files"] += 1.0
                        for key in ("open_dataset_time", "spatial_aggregate_time", "cache_build_hits", "cache_build_misses"):
                            perf_totals[key] += float(one_perf.get(key, 0.0))
                        if err_msg is not None:
                            perf_totals["failed_files"] += 1.0
                            print(f"[WARN] Skip ERA5 file: {file_name} | {err_msg}")
                        elif one_df is not None and not one_df.empty:
                            rows.append(one_df)
            remaining_tasks = []
        except (BrokenProcessPool, MemoryError, OSError, RuntimeError) as exc:
            remaining_tasks = [task for task in remaining_tasks if task not in processed_in_round]
            if current_workers <= 1:
                print(f"[WARN] ERA5 多进程读取失败，回退单进程读取。原因: {exc}")
                for file_path, folder_name, value_col in progress_iter(
                    remaining_tasks,
                    desc="读取 ERA5 NC 文件[单进程]",
                    total=len(remaining_tasks),
                ):
                    file_name, one_df, err_msg, one_perf = _read_single_era5_nc_local(
                        Path(file_path),
                        folder_name,
                        value_col,
                        alias_map_tuple,
                        base_module,
                        city_gdf_wgs84,
                        unit_conversions,
                    )
                    pbar.update(1)
                    perf_totals["processed_files"] += 1.0
                    for key in ("open_dataset_time", "spatial_aggregate_time", "cache_build_hits", "cache_build_misses"):
                        perf_totals[key] += float(one_perf.get(key, 0.0))
                    if err_msg is not None:
                        perf_totals["failed_files"] += 1.0
                        print(f"[WARN] Skip ERA5 file: {file_name} | {err_msg}")
                    elif one_df is not None and not one_df.empty:
                        rows.append(one_df)
                remaining_tasks = []
                break
            next_workers = max(1, current_workers // 2)
            print(
                "[WARN] ERA5 多进程池异常退出，"
                f"进程数将从 {current_workers} 降到 {next_workers} 后重试。"
                f"剩余文件数: {len(remaining_tasks)}。原因: {exc}"
            )
            current_workers = next_workers
    pbar.close()

    elapsed = time.perf_counter() - read_begin
    files_done = int(perf_totals["processed_files"])
    fps = float(files_done / elapsed) if elapsed > 0 else 0.0
    print(
        "[INFO] ERA5 读取统计: "
        f"files={files_done}, failed={int(perf_totals['failed_files'])}, "
        f"elapsed={elapsed:.2f}s, files_per_second={fps:.2f}, "
        f"open_dataset_time={perf_totals['open_dataset_time']:.2f}s, "
        f"spatial_aggregate_time={perf_totals['spatial_aggregate_time']:.2f}s, "
        f"cache_hits={int(perf_totals['cache_build_hits'])}, "
        f"cache_misses={int(perf_totals['cache_build_misses'])}"
    )
    if perf_stats is not None:
        perf_stats.update(
            {
                "open_dataset_time": float(perf_totals["open_dataset_time"]),
                "spatial_aggregate_time": float(perf_totals["spatial_aggregate_time"]),
                "cache_build_hits": int(perf_totals["cache_build_hits"]),
                "cache_build_misses": int(perf_totals["cache_build_misses"]),
                "processed_files": int(perf_totals["processed_files"]),
                "failed_files": int(perf_totals["failed_files"]),
                "files_per_second": float(fps),
                "elapsed_seconds": float(elapsed),
            }
        )

    if not rows:
        print("[WARN] No valid daily ERA5 features extracted.")
        return pd.DataFrame(columns=["city", "date"])

    combined = pd.concat(rows, ignore_index=True, sort=False)
    value_cols = [col for col in combined.columns if col not in {"city", "date"}]
    for col in value_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    daily_features = (
        combined.groupby(["city", "date"], as_index=False)[value_cols]
        .mean()
        .sort_values(["city", "date"], kind="mergesort")
        .reset_index(drop=True)
    )
    print(f"[INFO] ERA5 daily feature rows: {len(daily_features):,} | feature count: {len(value_cols)}")
    return daily_features


def _normalize_pollutant_key(name: str) -> str:
    key = sanitize_factor_name(name)
    if key.startswith("pollutant_"):
        key = key[len("pollutant_") :]
    return key


def load_pollutant_daily_features_from_nc(
    *,
    pollutant_name: str,
    pollutant_day_dir: Path,
    city_geojson_dir: Path,
    base_module: Any,
    city_list: list[str],
    data_read_dir: Path,
    module_tag: str,
    allowed_years: set[int] | None = None,
    max_workers: int | None = None,
    city_to_cluster: dict[str, str] | None = None,
    perf_stats: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    通用污染物（CO/NO2/O3/PM10/SO2）NC 日特征读取。
    读取流程与 ERA5 保持一致，输出列形如 pollutant_{name}。
    """
    pollutant_key = _normalize_pollutant_key(pollutant_name)
    pollutant_label = pollutant_key.upper()
    if not pollutant_day_dir.exists():
        raise FileNotFoundError(f"{pollutant_label} day directory not found: {pollutant_day_dir}")
    if not city_geojson_dir.exists():
        raise FileNotFoundError(f"City geojson directory not found: {city_geojson_dir}")

    # 复用 ERA5 worker 初始化，污染物场默认不做单位转换。
    met_module_path = data_read_dir / "Meteorology.py"
    if not met_module_path.exists():
        raise FileNotFoundError(f"Missing Meteorology.py: {met_module_path}")
    alias_map: dict[str, tuple[str, ...]] = {
        pollutant_key: tuple(POLLUTANT_VAR_ALIASES.get(pollutant_key, (pollutant_key,))),
    }

    city_gdf_wgs84 = base_module._load_city_geojson(city_geojson_dir, city_list)
    nc_files = sorted(path for path in pollutant_day_dir.rglob("*.nc") if path.is_file())
    nc_files = _filter_nc_files_by_year(nc_files, allowed_years)
    if not nc_files:
        print(f"[WARN] No {pollutant_label} .nc files found: {pollutant_day_dir}")
        return pd.DataFrame(columns=["city", "date"])
    if allowed_years:
        print(f"[INFO] {pollutant_label} 年份过滤: {sorted(allowed_years)} | 命中文件数: {len(nc_files)}")

    cpu_cores = os.cpu_count() or 1
    worker_count = choose_initial_worker_count(
        task_count=len(nc_files),
        cpu_cores=cpu_cores,
        requested_workers=max_workers,
        label=f"{pollutant_label} NC",
    )
    print(f"[INFO] {pollutant_label} NC 初始并发进程数: {worker_count} (CPU cores={cpu_cores})")

    base_module_path = str(Path(getattr(base_module, "__file__")).resolve())
    alias_map_tuple = {key: tuple(val) for key, val in alias_map.items()}
    value_col = f"pollutant_{sanitize_factor_name(pollutant_key)}"

    rows: list[pd.DataFrame] = []
    tasks: list[tuple[str, str, str]] = []
    for nc_file in nc_files:
        tasks.append((str(nc_file.resolve()), pollutant_key, value_col))

    perf_totals = {
        "open_dataset_time": 0.0,
        "spatial_aggregate_time": 0.0,
        "cache_build_hits": 0.0,
        "cache_build_misses": 0.0,
        "processed_files": 0.0,
        "failed_files": 0.0,
    }
    read_begin = time.perf_counter()
    current_workers = worker_count
    remaining_tasks = list(tasks)
    pbar = tqdm(total=len(tasks), desc=f"读取 {pollutant_label} NC 文件", dynamic_ncols=True)
    while remaining_tasks:
        chunk_size = max(8, min(32, len(remaining_tasks) // max(1, current_workers * 4)))
        remaining_chunks = _build_chunked_era5_tasks(remaining_tasks, chunk_size=chunk_size)
        processed_in_round: set[tuple[str, str, str]] = set()
        max_in_flight = min(len(remaining_chunks), max(4, current_workers * 2))
        try:
            with ProcessPoolExecutor(
                max_workers=current_workers,
                initializer=_init_era5_worker,
                initargs=(
                    base_module_path,
                    str(met_module_path.resolve()),
                    str(city_geojson_dir.resolve()),
                    city_list,
                    city_to_cluster,
                ),
            ) as executor:
                for one_chunk, result_list in iter_bounded_executor_results(
                    executor=executor,
                    tasks=remaining_chunks,
                    submit_task=lambda one_executor, one_chunk: one_executor.submit(
                        _read_era5_nc_chunk_worker, one_chunk, alias_map_tuple
                    ),
                    max_in_flight=max_in_flight,
                ):
                    for task in one_chunk:
                        processed_in_round.add(task)
                    for file_name, one_df, err_msg, one_perf in result_list:
                        pbar.update(1)
                        perf_totals["processed_files"] += 1.0
                        for key in ("open_dataset_time", "spatial_aggregate_time", "cache_build_hits", "cache_build_misses"):
                            perf_totals[key] += float(one_perf.get(key, 0.0))
                        if err_msg is not None:
                            perf_totals["failed_files"] += 1.0
                            print(f"[WARN] Skip {pollutant_label} file: {file_name} | {err_msg}")
                        elif one_df is not None and not one_df.empty:
                            rows.append(one_df)
            remaining_tasks = []
        except (BrokenProcessPool, MemoryError, OSError, RuntimeError) as exc:
            remaining_tasks = [task for task in remaining_tasks if task not in processed_in_round]
            if current_workers <= 1:
                print(f"[WARN] {pollutant_label} 多进程读取失败，回退单进程读取。原因: {exc}")
                for file_path, folder_name, one_value_col in progress_iter(
                    remaining_tasks,
                    desc=f"读取 {pollutant_label} NC 文件[单进程]",
                    total=len(remaining_tasks),
                ):
                    file_name, one_df, err_msg, one_perf = _read_single_era5_nc_local(
                        Path(file_path),
                        folder_name,
                        one_value_col,
                        alias_map_tuple,
                        base_module,
                        city_gdf_wgs84,
                        {},
                    )
                    pbar.update(1)
                    perf_totals["processed_files"] += 1.0
                    for key in ("open_dataset_time", "spatial_aggregate_time", "cache_build_hits", "cache_build_misses"):
                        perf_totals[key] += float(one_perf.get(key, 0.0))
                    if err_msg is not None:
                        perf_totals["failed_files"] += 1.0
                        print(f"[WARN] Skip {pollutant_label} file: {file_name} | {err_msg}")
                    elif one_df is not None and not one_df.empty:
                        rows.append(one_df)
                remaining_tasks = []
                break
            next_workers = max(1, current_workers // 2)
            print(
                f"[WARN] {pollutant_label} 多进程池异常退出，"
                f"进程数将从 {current_workers} 降到 {next_workers} 后重试。"
                f"剩余文件数: {len(remaining_tasks)}。原因: {exc}"
            )
            current_workers = next_workers
    pbar.close()

    elapsed = time.perf_counter() - read_begin
    files_done = int(perf_totals["processed_files"])
    fps = float(files_done / elapsed) if elapsed > 0 else 0.0
    print(
        f"[INFO] {pollutant_label} 读取统计: "
        f"files={files_done}, failed={int(perf_totals['failed_files'])}, "
        f"elapsed={elapsed:.2f}s, files_per_second={fps:.2f}, "
        f"open_dataset_time={perf_totals['open_dataset_time']:.2f}s, "
        f"spatial_aggregate_time={perf_totals['spatial_aggregate_time']:.2f}s, "
        f"cache_hits={int(perf_totals['cache_build_hits'])}, "
        f"cache_misses={int(perf_totals['cache_build_misses'])}"
    )
    if perf_stats is not None:
        perf_stats.update(
            {
                "open_dataset_time": float(perf_totals["open_dataset_time"]),
                "spatial_aggregate_time": float(perf_totals["spatial_aggregate_time"]),
                "cache_build_hits": int(perf_totals["cache_build_hits"]),
                "cache_build_misses": int(perf_totals["cache_build_misses"]),
                "processed_files": int(perf_totals["processed_files"]),
                "failed_files": int(perf_totals["failed_files"]),
                "files_per_second": float(fps),
                "elapsed_seconds": float(elapsed),
            }
        )

    if not rows:
        print(f"[WARN] No valid daily {pollutant_label} features extracted.")
        return pd.DataFrame(columns=["city", "date"])

    combined = pd.concat(rows, ignore_index=True, sort=False)
    value_cols = [col for col in combined.columns if col not in {"city", "date"}]
    for col in value_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    daily_features = (
        combined.groupby(["city", "date"], as_index=False)[value_cols]
        .mean()
        .sort_values(["city", "date"], kind="mergesort")
        .reset_index(drop=True)
    )
    print(
        f"[INFO] {pollutant_label} daily feature rows: {len(daily_features):,} "
        f"| feature count: {len(value_cols)}"
    )
    return daily_features


def build_one_region_factor_panel(base_module: Any, config: RegionConfig, data_read_dir: Path) -> pd.DataFrame:
    pm25_csv = config.pm25_csv_path.expanduser().resolve()
    if not pm25_csv.exists():
        raise FileNotFoundError(f"Missing PM2.5 city-year file: {pm25_csv}")

    old_cities = list(getattr(base_module, "BTH_CITIES"))
    old_map = dict(getattr(base_module, "CITY_TO_PROVINCE"))
    try:
        base_module.BTH_CITIES = list(config.cities)
        base_module.CITY_TO_PROVINCE = dict(config.city_to_province)
        panel = base_module.build_panel_from_interfaces(
            data_read_dir=data_read_dir,
            pm25_city_year_csv=pm25_csv,
        )
    finally:
        base_module.BTH_CITIES = old_cities
        base_module.CITY_TO_PROVINCE = old_map

    panel = panel.copy()
    panel.columns = [str(col).strip() for col in panel.columns]
    if {"city", "year"} - set(panel.columns):
        raise ValueError(f"Region panel missing key columns for {config.name}.")

    panel["city"] = panel["city"].map(normalize_city_name)
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel = panel.dropna(subset=["city", "year"]).copy()
    panel["year"] = panel["year"].astype(int)
    panel["cluster"] = config.name

    if "pm25" in panel.columns:
        panel = panel.drop(columns=["pm25"])

    factor_candidates = [col for col in panel.columns if col not in {"city", "year", "cluster"}]
    for col in factor_candidates:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    numeric_cols = [col for col in factor_candidates if pd.api.types.is_numeric_dtype(panel[col]) and panel[col].notna().any()]
    if not numeric_cols:
        return panel[["city", "year", "cluster"]].drop_duplicates()

    return panel.groupby(["city", "year", "cluster"], as_index=False)[numeric_cols].mean().reset_index(drop=True)


def build_year_factor_panel(base_module: Any, configs: list[RegionConfig], data_read_dir: Path) -> pd.DataFrame:
    region_panels: list[pd.DataFrame] = []
    for config in configs:
        try:
            one_panel = build_one_region_factor_panel(base_module, config, data_read_dir)
            region_panels.append(one_panel)
            print(f"[INFO] Built {config.name} year-factor panel: {len(one_panel):,} rows")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Failed to build {config.name} factor panel: {exc}")

    if not region_panels:
        return pd.DataFrame(columns=["city", "year"])

    combined = pd.concat(region_panels, ignore_index=True, sort=False)
    combined.columns = [str(col).strip() for col in combined.columns]
    numeric_cols = [
        col
        for col in combined.columns
        if col not in {"city", "year", "cluster"} and pd.api.types.is_numeric_dtype(combined[col])
    ]
    if numeric_cols:
        return combined.groupby(["city", "year"], as_index=False)[numeric_cols].mean()
    return combined[["city", "year"]].drop_duplicates()


def assign_cluster_and_filter_cities(daily_df: pd.DataFrame, city_cluster_map: pd.DataFrame) -> pd.DataFrame:
    city_to_cluster = dict(zip(city_cluster_map["city"], city_cluster_map["cluster"]))
    out = daily_df.copy()
    out["cluster"] = out["city"].map(city_to_cluster)
    before = len(out)
    out = out.dropna(subset=["cluster"]).copy()
    removed = before - len(out)
    if removed > 0:
        print(f"[WARN] Dropped {removed:,} rows outside BTH/YRD/PRD city list.")
    print(f"[INFO] Kept samples in BTH/YRD/PRD: {len(out):,}")
    return out


def merge_year_factors_with_daily_scaling(daily_df: pd.DataFrame, factor_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if factor_df.empty:
        return daily_df.copy(), []

    factor_cols = [col for col in factor_df.columns if col not in {"city", "year"}]
    conflict_cols = [col for col in factor_cols if col in daily_df.columns]
    if conflict_cols:
        rename_map = {col: f"{col}_yearly" for col in conflict_cols}
        factor_df = factor_df.rename(columns=rename_map)
        factor_cols = [rename_map.get(col, col) for col in factor_cols]
        print(
            f"[INFO] Year-factor columns renamed to avoid conflicts: {len(conflict_cols)} columns "
            f"(example: {conflict_cols[0]} -> {rename_map[conflict_cols[0]]})"
        )

    merged = daily_df.merge(factor_df, on=["city", "year"], how="left")
    existing_factor_cols = [col for col in factor_cols if col in merged.columns]
    if existing_factor_cols:
        merged[existing_factor_cols] = merged[existing_factor_cols].apply(pd.to_numeric, errors="coerce")

    coverage = float(merged[existing_factor_cols].notna().any(axis=1).mean()) if existing_factor_cols else 0.0
    print(f"[INFO] Year-factor merge coverage: {coverage:.2%}")
    print("[INFO] Year-level factors keep yearly values without dividing by 365.")
    return merged, existing_factor_cols


def scale_monthly_columns_to_daily(df: pd.DataFrame, monthly_cols: list[str]) -> pd.DataFrame:
    if not monthly_cols:
        return df
    out = df.copy()
    days = out["date"].dt.days_in_month.astype(float)
    for col in monthly_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce") / days
    print(f"[INFO] Monthly factors converted to daily scale: {len(monthly_cols)} columns")
    return out


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["city", "date"], kind="mergesort").reset_index(drop=True).copy()
    out["month"] = out["date"].dt.month
    out["dayofyear"] = out["date"].dt.dayofyear
    out["dayofweek"] = out["date"].dt.dayofweek
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    grouped = out.groupby("city", observed=True)["pm25"]
    for lag in (1, 2, 3, 7, 14):
        out[f"lag_{lag}"] = grouped.shift(lag)
    out["roll_mean_3"] = grouped.transform(lambda series: series.shift(1).rolling(window=3, min_periods=1).mean())
    out["roll_mean_7"] = grouped.transform(lambda series: series.shift(1).rolling(window=7, min_periods=1).mean())
    out["roll_std_7"] = grouped.transform(lambda series: series.shift(1).rolling(window=7, min_periods=2).std())
    return out


def split_by_time(
    df: pd.DataFrame,
    train_end_year: int,
    valid_year: int,
    test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc[df["year"] <= train_end_year].copy()
    valid = df.loc[df["year"] == valid_year].copy()
    test = df.loc[df["year"] == test_year].copy()
    if train.empty or valid.empty or test.empty:
        raise ValueError(
            f"Invalid split: train={len(train)}, valid={len(valid)}, test={len(test)}. "
            "Please adjust split years or input data."
        )
    return train, valid, test


def build_model_matrices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    train = train_df.copy()
    valid = valid_df.copy()
    test = test_df.copy()

    _append_city_cluster_codes(train=train, valid=valid, test=test)

    non_feature_cols = {"date", "city", "cluster", "pm25", "city_code", "cluster_code"}
    candidate_cols = [col for col in train.columns if col not in non_feature_cols]
    feature_cols = [col for col in candidate_cols if pd.api.types.is_numeric_dtype(train[col])]
    feature_cols = [col for col in feature_cols if train[col].notna().any()]
    if not feature_cols:
        raise ValueError("No usable numerical features detected.")

    fill_values = train[feature_cols].median(numeric_only=True)
    x_train = train[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(fill_values)
    x_valid = valid[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(fill_values)
    x_test = test[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(fill_values)
    y_train = train["pm25"].astype(float)
    y_valid = valid["pm25"].astype(float)
    y_test = test["pm25"].astype(float)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, feature_cols


def _append_city_cluster_codes(*, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
    city_categories = sorted(train["city"].dropna().astype(str).unique().tolist())
    cluster_categories = sorted(train["cluster"].dropna().astype(str).unique().tolist())
    for frame in (train, valid, test):
        frame["city_code"] = pd.Categorical(frame["city"], categories=city_categories).codes
        frame["cluster_code"] = pd.Categorical(frame["cluster"], categories=cluster_categories).codes
        frame["city_code"] = frame["city_code"].replace(-1, np.nan)
        frame["cluster_code"] = frame["cluster_code"].replace(-1, np.nan)


def build_sequence_matrices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seq_len: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    list[str],
    list[str],
]:
    """
    Build (X_seq, y, meta) per split for sequence models. Each sample uses the
    last seq_len days (before target date) as input; target is PM2.5 on target date.
    Returns city_categories for consistent city encoding (e.g. ST-Transformer).
    """
    train = train_df.copy()
    valid = valid_df.copy()
    test = test_df.copy()

    city_categories = sorted(train["city"].dropna().astype(str).unique().tolist())
    _append_city_cluster_codes(train=train, valid=valid, test=test)

    non_feature_cols = {"date", "city", "cluster", "pm25", "city_code", "cluster_code"}
    candidate_cols = [col for col in train.columns if col not in non_feature_cols]
    feature_cols = [col for col in candidate_cols if pd.api.types.is_numeric_dtype(train[col])]
    feature_cols = [col for col in feature_cols if train[col].notna().any()]
    if not feature_cols:
        raise ValueError("No usable numerical features detected for sequence matrices.")

    fill_values = train[feature_cols].median(numeric_only=True)
    for frame in (train, valid, test):
        frame[feature_cols] = frame[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(fill_values)

    full_df = (
        pd.concat([train, valid, test], ignore_index=True)
        .sort_values(["city", "date"], kind="mergesort")
        .reset_index(drop=True)
    )
    city_histories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    by_city = full_df.groupby("city", observed=True, sort=False)
    for city, city_df in by_city:
        city_dates = city_df["date"].to_numpy(dtype="datetime64[ns]")
        city_features = city_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        city_histories[str(city)] = (city_dates, city_features)

    def _collect_sequences(
        target_df: pd.DataFrame, split_name: str
    ) -> tuple[list[np.ndarray], list[float], list[dict[str, Any]]]:
        x_list: list[np.ndarray] = []
        y_list: list[float] = []
        meta_list: list[dict[str, Any]] = []
        iterator = progress_iter(
            target_df.itertuples(index=False),
            desc=f"构建序列样本[{split_name}]",
            total=len(target_df),
        )
        for row in iterator:
            city = str(getattr(row, "city"))
            date = getattr(row, "date")
            history = city_histories.get(city)
            if history is None:
                continue
            city_dates, city_features = history
            target_date = np.datetime64(pd.Timestamp(date).to_datetime64())
            idx = int(np.searchsorted(city_dates, target_date, side="left"))
            if idx < seq_len:
                continue
            feat = city_features[idx - seq_len : idx]
            x_list.append(feat)
            y_list.append(float(getattr(row, "pm25")))
            meta_list.append(
                {
                    "date": getattr(row, "date"),
                    "city": getattr(row, "city"),
                    "cluster": getattr(row, "cluster"),
                    "pm25": getattr(row, "pm25"),
                }
            )
        return x_list, y_list, meta_list

    x_train_list, y_train_list, meta_train_list = _collect_sequences(train, "train")
    x_valid_list, y_valid_list, meta_valid_list = _collect_sequences(valid, "valid")
    x_test_list, y_test_list, meta_test_list = _collect_sequences(test, "test")

    def _to_arrays(
        x_list: list[np.ndarray],
        y_list: list[float],
        meta_list: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        if not x_list:
            n_f = len(feature_cols)
            return (
                np.zeros((0, seq_len, n_f), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                pd.DataFrame(columns=["date", "city", "cluster", "pm25"]),
            )
        X = np.stack(x_list, axis=0)
        y = np.array(y_list, dtype=np.float32)
        meta = pd.DataFrame(meta_list)
        return X, y, meta

    X_train, y_train, meta_train = _to_arrays(x_train_list, y_train_list, meta_train_list)
    X_valid, y_valid, meta_valid = _to_arrays(x_valid_list, y_valid_list, meta_valid_list)
    X_test, y_test, meta_test = _to_arrays(x_test_list, y_test_list, meta_test_list)

    return (
        X_train, y_train, meta_train,
        X_valid, y_valid, meta_valid,
        X_test, y_test, meta_test,
        feature_cols,
        city_categories,
    )


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def metrics_by_cluster(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for cluster, group in pred_df.groupby("cluster", observed=True):
        metrics = compute_metrics(group["y_true"], group["y_pred"].to_numpy())
        rows.append({"cluster": cluster, **metrics, "n_samples": int(len(group))})
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def export_feature_quality_report(
    *,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    year_factor_cols: list[str],
    met_cols: list[str],
) -> pd.DataFrame:
    """
    Export a unified feature quality report for split-wise diagnostics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    split_frames = {"train": train_df.copy(), "valid": valid_df.copy(), "test": test_df.copy()}
    non_feature_cols = {"date", "city", "cluster", "pm25", "city_code", "cluster_code"}
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    candidate_cols = [col for col in all_df.columns if col not in non_feature_cols]
    feature_cols = [col for col in candidate_cols if pd.api.types.is_numeric_dtype(all_df[col])]

    rows: list[dict[str, Any]] = []
    for split_name, frame in split_frames.items():
        if frame.empty:
            continue
        for feature in feature_cols:
            series = pd.to_numeric(frame[feature], errors="coerce")
            rows.append(
                {
                    "metric_type": "feature_stats",
                    "split": split_name,
                    "feature": feature,
                    "city": "",
                    "year": np.nan,
                    "missing_rate": float(series.isna().mean()),
                    "coverage_rate": float(series.notna().mean()),
                    "variance": float(series.var(ddof=0)) if series.notna().any() else np.nan,
                    "n_unique": int(series.nunique(dropna=True)),
                    "is_constant": bool(series.nunique(dropna=True) <= 1),
                    "has_yearly_pair": bool(f"{feature}_yearly" in year_factor_cols),
                    "is_era5_feature": bool(feature in met_cols),
                }
            )

    available_met_cols = [col for col in met_cols if col in all_df.columns]
    for split_name, frame in split_frames.items():
        if frame.empty:
            continue
        for met_col in available_met_cols:
            series = pd.to_numeric(frame[met_col], errors="coerce")
            rows.append(
                {
                    "metric_type": "era5_split_feature_coverage",
                    "split": split_name,
                    "feature": met_col,
                    "city": "",
                    "year": np.nan,
                    "missing_rate": float(series.isna().mean()),
                    "coverage_rate": float(series.notna().mean()),
                    "variance": np.nan,
                    "n_unique": np.nan,
                    "is_constant": np.nan,
                    "has_yearly_pair": bool(f"{met_col}_yearly" in year_factor_cols),
                    "is_era5_feature": True,
                }
            )

        key_cols = [col for col in ("city", "year") if col in frame.columns]
        if len(key_cols) == 2 and available_met_cols:
            coverage_by_city_year = (
                frame[key_cols + available_met_cols]
                .groupby(["city", "year"], observed=True)[available_met_cols]
                .agg(lambda values: float(pd.to_numeric(values, errors="coerce").notna().mean()))
                .reset_index()
            )
            coverage_long = coverage_by_city_year.melt(
                id_vars=["city", "year"],
                var_name="feature",
                value_name="coverage_rate",
            )
            for one in coverage_long.itertuples(index=False):
                rows.append(
                    {
                        "metric_type": "era5_city_year_coverage",
                        "split": split_name,
                        "feature": str(one.feature),
                        "city": str(one.city),
                        "year": int(one.year),
                        "missing_rate": float(1.0 - float(one.coverage_rate)),
                        "coverage_rate": float(one.coverage_rate),
                        "variance": np.nan,
                        "n_unique": np.nan,
                        "is_constant": np.nan,
                        "has_yearly_pair": bool(f"{one.feature}_yearly" in year_factor_cols),
                        "is_era5_feature": True,
                    }
                )

    overlap_features = sorted([met for met in available_met_cols if f"{met}_yearly" in year_factor_cols])
    for feature in overlap_features:
        rows.append(
            {
                "metric_type": "era5_yearly_overlap_summary",
                "split": "all",
                "feature": feature,
                "city": "",
                "year": np.nan,
                "missing_rate": np.nan,
                "coverage_rate": np.nan,
                "variance": np.nan,
                "n_unique": np.nan,
                "is_constant": np.nan,
                "has_yearly_pair": True,
                "is_era5_feature": True,
            }
        )

    report_df = pd.DataFrame(rows)
    report_path = output_dir / "feature_quality_report.csv"
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Feature quality report exported: {report_path}")
    return report_df


def prepare_training_table(
    *,
    module_tag: str,
    correlation_dir: Path,
    data_read_dir: Path,
    city_geojson_dir: Path,
    daily_input: list[str] | None,
    pm25_day_dir: Path,
    era5_day_dir: Path,
    include_era5_daily: bool,
    allowed_years: list[int] | None = None,
    pm25_workers: int | None = None,
    era5_workers: int | None = None,
    cache_dir: Path | None = None,
    enable_cache: bool = False,
    prebuilt_year_factor_df: pd.DataFrame | None = None,
    prebuilt_pm25_nc_index: dict[str, Any] | None = None,
    prepare_stats: dict[str, Any] | None = None,
    use_year_factors: bool = True,
    pm25_precropped: bool = False,
    pollutant_day_dirs: dict[str, Path] | None = None,
    include_pollutant_daily: bool = True,
    strict_pollutant_dirs: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    stage_begin = time.perf_counter()
    base_module, configs = load_region_configs(correlation_dir, module_tag=module_tag)
    city_cluster_map = build_city_cluster_map(configs)
    city_list = city_cluster_map["city"].dropna().astype(str).unique().tolist()
    year_filter_set = set(allowed_years) if allowed_years else None
    cache_root = cache_dir.expanduser().resolve() if cache_dir is not None else None
    cache_enabled = bool(enable_cache and cache_root is not None)
    cache_hits = {"pm25": False, "era5": False}
    pollutant_keys = sorted((pollutant_day_dirs or DEFAULT_POLLUTANT_DAY_DIRS).keys())
    pollutant_cache_hits: dict[str, bool] = {key: False for key in pollutant_keys}
    city_to_cluster = (
        city_cluster_map.dropna(subset=["city", "cluster"])
        .assign(city=lambda d: d["city"].astype(str), cluster=lambda d: d["cluster"].astype(str))
        .set_index("city")["cluster"]
        .to_dict()
    )

    def _build_cache_key(tag: str) -> str:
        pollutant_dir_payload = {
            str(_normalize_pollutant_key(key)): str(Path(path).expanduser().resolve())
            for key, path in (pollutant_day_dirs or DEFAULT_POLLUTANT_DAY_DIRS).items()
        }
        payload = {
            "tag": tag,
            "module_tag": module_tag,
            "include_era5": include_era5_daily,
            "include_pollutant_daily": include_pollutant_daily,
            "pm25_precropped": pm25_precropped,
            "pm25_day_dir": str(pm25_day_dir.resolve()),
            "era5_day_dir": str(era5_day_dir.resolve()),
            "pollutant_day_dirs": pollutant_dir_payload,
            "city_geojson_dir": str(city_geojson_dir.resolve()),
            "allowed_years": sorted(year_filter_set) if year_filter_set else [],
            "city_count": len(city_list),
        }
        return _stable_hash(payload)

    pm25_begin = time.perf_counter()
    pm25_perf_stats: dict[str, Any] = {
        "open_dataset_time": 0.0,
        "spatial_aggregate_time": 0.0,
        "cache_build_hits": 0,
        "cache_build_misses": 0,
        "processed_files": 0,
        "failed_files": 0,
        "files_per_second": 0.0,
        "elapsed_seconds": 0.0,
    }
    if daily_input:
        print("[WARN] daily_input 已停用，当前仅使用 PM2.5/ERA5/污染物 NC 数据源。")
    daily_df = None
    pm25_cache_path = None
    if cache_enabled:
        pm25_cache_path = cache_root / f"pm25_daily_{_build_cache_key('pm25')}.parquet"
        daily_df = _load_cached_dataframe(pm25_cache_path, "PM2.5")
        cache_hits["pm25"] = daily_df is not None
    if daily_df is None:
        daily_df = load_pm25_daily_from_nc(
            pm25_day_dir=pm25_day_dir,
            city_geojson_dir=city_geojson_dir,
            base_module=base_module,
            city_list=city_list,
            allowed_years=year_filter_set,
            max_workers=pm25_workers,
            pm25_nc_index=prebuilt_pm25_nc_index,
            city_to_cluster=city_to_cluster,
            pm25_precropped=pm25_precropped,
            perf_stats=pm25_perf_stats,
        )
        if cache_enabled and pm25_cache_path is not None:
            _save_cached_dataframe(daily_df, pm25_cache_path, "PM2.5")
    pm25_seconds = time.perf_counter() - pm25_begin

    daily_df = assign_cluster_and_filter_cities(daily_df, city_cluster_map)

    met_cols: list[str] = []
    era5_seconds = 0.0
    era5_perf_stats: dict[str, Any] = {
        "open_dataset_time": 0.0,
        "spatial_aggregate_time": 0.0,
        "cache_build_hits": 0,
        "cache_build_misses": 0,
        "processed_files": 0,
        "failed_files": 0,
        "files_per_second": 0.0,
        "elapsed_seconds": 0.0,
    }
    if include_era5_daily:
        era5_begin = time.perf_counter()
        era5_df = None
        era5_cache_path = None
        if cache_enabled:
            era5_cache_path = cache_root / f"era5_daily_{_build_cache_key('era5')}.parquet"
            era5_df = _load_cached_dataframe(era5_cache_path, "ERA5")
            cache_hits["era5"] = era5_df is not None
        if era5_df is None:
            era5_df = load_era5_daily_features_from_nc(
                era5_day_dir=era5_day_dir,
                city_geojson_dir=city_geojson_dir,
                base_module=base_module,
                city_list=city_list,
                data_read_dir=data_read_dir,
                module_tag=module_tag,
                allowed_years=year_filter_set,
                max_workers=era5_workers,
                city_to_cluster=city_to_cluster,
                perf_stats=era5_perf_stats,
            )
            if cache_enabled and era5_cache_path is not None and not era5_df.empty:
                _save_cached_dataframe(era5_df, era5_cache_path, "ERA5")
        if not era5_df.empty:
            met_cols = [col for col in era5_df.columns if col not in {"city", "date"}]
            daily_df = daily_df.merge(era5_df, on=["city", "date"], how="left")
            print(f"[INFO] Merged ERA5 daily features: {len(met_cols)} columns")
        era5_seconds = time.perf_counter() - era5_begin

    pollutant_cols: list[str] = []
    pollutant_seconds = 0.0
    pollutant_stats: dict[str, dict[str, Any]] = {}
    if include_pollutant_daily:
        pollutant_begin = time.perf_counter()
        raw_dirs = pollutant_day_dirs or DEFAULT_POLLUTANT_DAY_DIRS
        for raw_key, raw_dir in sorted(raw_dirs.items(), key=lambda kv: _normalize_pollutant_key(str(kv[0]))):
            pollutant_key = _normalize_pollutant_key(str(raw_key))
            pollutant_dir = Path(raw_dir).expanduser().resolve()
            if strict_pollutant_dirs and not pollutant_dir.exists():
                raise FileNotFoundError(f"Pollutant day directory not found: [{pollutant_key}] {pollutant_dir}")
            if not pollutant_dir.exists():
                print(f"[WARN] Pollutant day directory missing, skip [{pollutant_key}]: {pollutant_dir}")
                continue

            one_perf_stats: dict[str, Any] = {
                "open_dataset_time": 0.0,
                "spatial_aggregate_time": 0.0,
                "cache_build_hits": 0,
                "cache_build_misses": 0,
                "processed_files": 0,
                "failed_files": 0,
                "files_per_second": 0.0,
                "elapsed_seconds": 0.0,
            }
            pollutant_df = None
            pollutant_cache_path = None
            if cache_enabled:
                pollutant_cache_path = cache_root / f"pollutant_{pollutant_key}_daily_{_build_cache_key(f'pollutant_{pollutant_key}')}.parquet"
                pollutant_df = _load_cached_dataframe(pollutant_cache_path, pollutant_key.upper())
                pollutant_cache_hits[pollutant_key] = pollutant_df is not None
            if pollutant_df is None:
                pollutant_df = load_pollutant_daily_features_from_nc(
                    pollutant_name=pollutant_key,
                    pollutant_day_dir=pollutant_dir,
                    city_geojson_dir=city_geojson_dir,
                    base_module=base_module,
                    city_list=city_list,
                    data_read_dir=data_read_dir,
                    module_tag=module_tag,
                    allowed_years=year_filter_set,
                    max_workers=era5_workers,
                    city_to_cluster=city_to_cluster,
                    perf_stats=one_perf_stats,
                )
                if cache_enabled and pollutant_cache_path is not None and not pollutant_df.empty:
                    _save_cached_dataframe(pollutant_df, pollutant_cache_path, pollutant_key.upper())
            if not pollutant_df.empty:
                one_cols = [col for col in pollutant_df.columns if col not in {"city", "date"}]
                daily_df = daily_df.merge(pollutant_df, on=["city", "date"], how="left")
                pollutant_cols.extend(one_cols)
                print(f"[INFO] Merged {pollutant_key.upper()} daily features: {len(one_cols)} columns")
            pollutant_stats[pollutant_key] = one_perf_stats
        pollutant_seconds = time.perf_counter() - pollutant_begin

    if use_year_factors or prebuilt_year_factor_df is not None:
        print("[INFO] Year-factor 读取链路已停用，忽略 use_year_factors/prebuilt_year_factor_df 参数。")
    year_factor_seconds = 0.0
    year_factor_cols: list[str] = []
    merged_df = daily_df.copy()
    merged_df = scale_monthly_columns_to_daily(merged_df, monthly_cols=[])
    total_seconds = time.perf_counter() - stage_begin
    print(
        f"[INFO] Data prepare timing(s): pm25={pm25_seconds:.2f}, "
        f"era5={era5_seconds:.2f}, pollutant={pollutant_seconds:.2f}, "
        f"year_factor={year_factor_seconds:.2f}, total={total_seconds:.2f}"
    )
    all_daily_feature_cols = list(dict.fromkeys([*met_cols, *pollutant_cols]))
    if prepare_stats is not None:
        prepare_stats.update(
            {
                "cache_hit_pm25": cache_hits["pm25"],
                "cache_hit_era5": cache_hits["era5"],
                "cache_hit_pollutants": dict(pollutant_cache_hits),
                "pm25_seconds": float(pm25_seconds),
                "pm25_open_dataset_seconds": float(pm25_perf_stats.get("open_dataset_time", 0.0)),
                "pm25_spatial_aggregate_seconds": float(pm25_perf_stats.get("spatial_aggregate_time", 0.0)),
                "pm25_grid_cache_hits": int(pm25_perf_stats.get("cache_build_hits", 0)),
                "pm25_grid_cache_misses": int(pm25_perf_stats.get("cache_build_misses", 0)),
                "pm25_processed_files": int(pm25_perf_stats.get("processed_files", 0)),
                "pm25_failed_files": int(pm25_perf_stats.get("failed_files", 0)),
                "pm25_files_per_second": float(pm25_perf_stats.get("files_per_second", 0.0)),
                "era5_seconds": float(era5_seconds),
                "era5_open_dataset_seconds": float(era5_perf_stats.get("open_dataset_time", 0.0)),
                "era5_spatial_aggregate_seconds": float(era5_perf_stats.get("spatial_aggregate_time", 0.0)),
                "era5_grid_cache_hits": int(era5_perf_stats.get("cache_build_hits", 0)),
                "era5_grid_cache_misses": int(era5_perf_stats.get("cache_build_misses", 0)),
                "era5_processed_files": int(era5_perf_stats.get("processed_files", 0)),
                "era5_failed_files": int(era5_perf_stats.get("failed_files", 0)),
                "era5_files_per_second": float(era5_perf_stats.get("files_per_second", 0.0)),
                "pollutant_seconds": float(pollutant_seconds),
                "pollutant_feature_cols": list(pollutant_cols),
                "pollutant_perf_stats": pollutant_stats,
                "year_factor_seconds": float(year_factor_seconds),
                "data_prepare_seconds": float(total_seconds),
            }
        )
    return merged_df, year_factor_cols, all_daily_feature_cols


def build_prediction_frames(
    *,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_train: np.ndarray,
    pred_valid: np.ndarray,
    pred_test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _pack(frame: pd.DataFrame, pred: np.ndarray, split: str) -> pd.DataFrame:
        out = frame[["date", "city", "cluster", "pm25"]].copy()
        out = out.rename(columns={"pm25": "y_true"})
        out["y_pred"] = pred
        out["split"] = split
        return out

    train_pred_df = _pack(train_df, pred_train, "train")
    valid_pred_df = _pack(valid_df, pred_valid, "valid")
    test_pred_df = _pack(test_df, pred_test, "test")
    all_pred_df = pd.concat([train_pred_df, valid_pred_df, test_pred_df], ignore_index=True)
    return all_pred_df, test_pred_df


def evaluate_generalization(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics_df.copy()
    metrics = metrics.set_index("split")

    train_rmse = float(metrics.loc["train", "rmse"]) if "train" in metrics.index else np.nan
    valid_rmse = float(metrics.loc["valid", "rmse"]) if "valid" in metrics.index else np.nan
    test_rmse = float(metrics.loc["test", "rmse"]) if "test" in metrics.index else np.nan

    train_mae = float(metrics.loc["train", "mae"]) if "train" in metrics.index else np.nan
    valid_mae = float(metrics.loc["valid", "mae"]) if "valid" in metrics.index else np.nan
    test_mae = float(metrics.loc["test", "mae"]) if "test" in metrics.index else np.nan

    train_r2 = float(metrics.loc["train", "r2"]) if "train" in metrics.index else np.nan
    valid_r2 = float(metrics.loc["valid", "r2"]) if "valid" in metrics.index else np.nan
    test_r2 = float(metrics.loc["test", "r2"]) if "test" in metrics.index else np.nan

    rmse_valid_train_ratio = valid_rmse / train_rmse if np.isfinite(train_rmse) and train_rmse > 0 else np.nan
    rmse_test_valid_ratio = test_rmse / valid_rmse if np.isfinite(valid_rmse) and valid_rmse > 0 else np.nan
    mae_valid_train_ratio = valid_mae / train_mae if np.isfinite(train_mae) and train_mae > 0 else np.nan
    mae_test_valid_ratio = test_mae / valid_mae if np.isfinite(valid_mae) and valid_mae > 0 else np.nan
    r2_drop_train_valid = train_r2 - valid_r2 if np.isfinite(train_r2) and np.isfinite(valid_r2) else np.nan
    r2_drop_valid_test = valid_r2 - test_r2 if np.isfinite(valid_r2) and np.isfinite(test_r2) else np.nan

    score = 0
    if np.isfinite(rmse_valid_train_ratio) and rmse_valid_train_ratio <= 1.2:
        score += 1
    if np.isfinite(rmse_test_valid_ratio) and rmse_test_valid_ratio <= 1.2:
        score += 1
    if np.isfinite(r2_drop_train_valid) and r2_drop_train_valid <= 0.1:
        score += 1
    if np.isfinite(r2_drop_valid_test) and r2_drop_valid_test <= 0.1:
        score += 1

    if score >= 4:
        level = "strong"
    elif score >= 2:
        level = "moderate"
    else:
        level = "weak"

    return pd.DataFrame(
        [
            {
                "generalization_level": level,
                "score_0_to_4": score,
                "rmse_valid_train_ratio": rmse_valid_train_ratio,
                "rmse_test_valid_ratio": rmse_test_valid_ratio,
                "mae_valid_train_ratio": mae_valid_train_ratio,
                "mae_test_valid_ratio": mae_test_valid_ratio,
                "r2_drop_train_valid": r2_drop_train_valid,
                "r2_drop_valid_test": r2_drop_valid_test,
            }
        ]
    )


def export_generalization_artifacts(metrics_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_data = metrics_df.copy()
    plot_data.to_csv(output_dir / "generalization_plot_data.csv", index=False, encoding="utf-8-sig")

    assessment = evaluate_generalization(metrics_df)
    assessment.to_csv(output_dir / "generalization_assessment.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ordered = plot_data.set_index("split").reindex(["train", "valid", "test"]).reset_index()
    axes[0].bar(ordered["split"], ordered["r2"], color="#4C72B0")
    axes[0].set_title("R2 Across Splits")
    axes[0].set_ylim(min(0, float(np.nanmin(ordered["r2"])) - 0.05), 1.0)
    axes[1].bar(ordered["split"], ordered["rmse"], color="#55A868")
    axes[1].set_title("RMSE Across Splits")
    axes[2].bar(ordered["split"], ordered["mae"], color="#C44E52")
    axes[2].set_title("MAE Across Splits")
    for axis in axes:
        axis.set_xlabel("split")
    plt.tight_layout()
    plt.savefig(output_dir / "generalization_metrics_bar.png", dpi=300)
    plt.close()

    return assessment


def export_regression_artifacts(
    *,
    all_pred_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_sorted = all_pred_df.sort_values(["split", "date", "city"], kind="mergesort").reset_index(drop=True)
    all_sorted.to_csv(output_dir / "regression_all_splits_data.csv", index=False, encoding="utf-8-sig")

    test_df = all_sorted.loc[all_sorted["split"] == "test"].copy()
    test_df.to_csv(output_dir / "regression_test_data.csv", index=False, encoding="utf-8-sig")

    if not test_df.empty:
        fig = plt.figure(figsize=(6.5, 6))
        plt.scatter(test_df["y_true"], test_df["y_pred"], alpha=0.5, s=18, color="#4C72B0", label="test points")
        y_min = float(min(test_df["y_true"].min(), test_df["y_pred"].min()))
        y_max = float(max(test_df["y_true"].max(), test_df["y_pred"].max()))
        grid = np.linspace(y_min, y_max, 100)
        plt.plot(grid, grid, "--", color="#555555", linewidth=1.2, label="y = x")
        if len(test_df) >= 2:
            slope, intercept = np.polyfit(test_df["y_true"], test_df["y_pred"], deg=1)
            plt.plot(grid, slope * grid + intercept, color="#DD8452", linewidth=1.4, label="fit line")
        plt.xlabel("Actual PM2.5")
        plt.ylabel("Predicted PM2.5")
        plt.title(f"{model_name} Test Prediction vs Actual")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "regression_pred_vs_actual_test.png", dpi=300)
        plt.close()

    fig = plt.figure(figsize=(7, 6))
    palette = {"train": "#4C72B0", "valid": "#55A868", "test": "#C44E52"}
    for split_name, one in all_sorted.groupby("split", observed=True):
        plt.scatter(
            one["y_true"],
            one["y_pred"],
            alpha=0.35,
            s=12,
            color=palette.get(split_name, "#888888"),
            label=split_name,
        )
    y_min = float(min(all_sorted["y_true"].min(), all_sorted["y_pred"].min()))
    y_max = float(max(all_sorted["y_true"].max(), all_sorted["y_pred"].max()))
    grid = np.linspace(y_min, y_max, 100)
    plt.plot(grid, grid, "--", color="#444444", linewidth=1.1, label="y = x")
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title(f"{model_name} Prediction vs Actual (All Splits)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "regression_pred_vs_actual_all_splits.png", dpi=300)
    plt.close()


def _compute_shap_values(model: Any, x_sample: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "parameters"):
        try:
            import torch  # 惰性导入，避免 SVR 等非 PyTorch 脚本加载 CUDA 库
            params = list(model.parameters())
            if not params:
                raise ValueError("PyTorch model has no parameters")
            device = next(model.parameters()).device
            model.eval()
            x_tensor = torch.from_numpy(x_sample.values.astype(np.float32)).to(device)
            n_background = min(100, len(x_tensor))
            background = x_tensor[:n_background]
            explainer = shap.DeepExplainer(model, background)
            shap_tensor = explainer.shap_values(x_tensor)
            if isinstance(shap_tensor, list):
                shap_tensor = shap_tensor[0]
            shap_values = np.asarray(
                shap_tensor.cpu().numpy() if hasattr(shap_tensor, "cpu") else shap_tensor,
                dtype=np.float64,
            )
            if shap_values.ndim > 2:
                shap_values = np.squeeze(shap_values)
            return shap_values
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"SHAP DeepExplainer failed for PyTorch model: {exc}") from exc

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_sample)
    except Exception:  # pylint: disable=broad-except
        explainer = shap.Explainer(model, x_sample)
        explanation = explainer(x_sample)
        shap_values = getattr(explanation, "values", explanation)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim > 2:
        shap_values = shap_values[0]
    return shap_values


def export_shap_artifacts(
    *,
    model: Any,
    x_for_shap: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    shap_max_samples: int,
    shap_max_display: int,
    random_state: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    if x_for_shap.empty:
        raise ValueError("No samples available for SHAP analysis.")

    sample_size = min(len(x_for_shap), max(1, int(shap_max_samples)))
    if len(x_for_shap) > sample_size:
        x_sample = x_for_shap.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    else:
        x_sample = x_for_shap.reset_index(drop=True).copy()

    shap_values = _compute_shap_values(model, x_sample)
    if shap_values.shape[1] != x_sample.shape[1]:
        raise ValueError(
            f"SHAP values shape mismatch: shap={shap_values.shape}, features={x_sample.shape}"
        )

    feature_cols = list(x_sample.columns)
    sample_df = x_sample.copy()
    sample_df.insert(0, "row_id", np.arange(len(sample_df), dtype=int))
    sample_df.to_csv(output_dir / "shap_sample_features.csv", index=False, encoding="utf-8-sig")

    shap_wide = pd.DataFrame(shap_values, columns=feature_cols)
    shap_wide.insert(0, "row_id", np.arange(len(shap_wide), dtype=int))
    shap_wide.to_csv(output_dir / "shap_values_wide.csv", index=False, encoding="utf-8-sig")

    shap_long = shap_wide.melt(id_vars="row_id", var_name="feature", value_name="shap_value")
    feat_long = sample_df.melt(id_vars="row_id", var_name="feature", value_name="feature_value")
    beeswarm_data = shap_long.merge(feat_long, on=["row_id", "feature"], how="left")
    beeswarm_data["abs_shap"] = beeswarm_data["shap_value"].abs()
    beeswarm_data.to_csv(output_dir / "shap_beeswarm_data_long.csv", index=False, encoding="utf-8-sig")

    shap_importance = (
        beeswarm_data.groupby("feature", as_index=False)["abs_shap"]
        .mean()
        .rename(columns={"abs_shap": "mean_abs_shap"})
        .sort_values("mean_abs_shap", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    shap_importance.to_csv(output_dir / "shap_importance_bar_data.csv", index=False, encoding="utf-8-sig")

    try:
        shap.summary_plot(
            shap_values,
            x_sample,
            max_display=int(max(5, shap_max_display)),
            show=False,
        )
        plt.title(f"{model_name} SHAP Beeswarm")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_beeswarm.png", dpi=300)
        plt.close()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to render SHAP beeswarm: {exc}")
        plt.close()

    top_imp = shap_importance.head(int(max(5, shap_max_display)))
    if not top_imp.empty:
        fig = plt.figure(figsize=(8, max(4, 0.35 * len(top_imp) + 1.5)))
        sns.barplot(data=top_imp, x="mean_abs_shap", y="feature", color="#4C72B0")
        plt.gca().invert_yaxis()
        plt.xlabel("Mean |SHAP value|")
        plt.ylabel("Feature")
        plt.title(f"{model_name} SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_importance_bar.png", dpi=300)
        plt.close(fig)

    return shap_importance
