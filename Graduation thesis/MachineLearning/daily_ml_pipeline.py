from __future__ import annotations

import importlib.util
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
DEFAULT_CORRELATION_DIR = THESIS_DIR / "Correlation"
DEFAULT_DATA_READ_DIR = THESIS_DIR / "Data Read"
DEFAULT_PM25_DAY_DIR = Path(r"F:\1.模型要用的\2018-2023[PM2.5-china]\Day")
DEFAULT_ERA5_DAY_DIR = Path(r"F:\1.模型要用的\2018-2023[ERA5_PM2.5]")
DEFAULT_CITY_GEOJSON_DIR = Path(r"F:\1.模型要用的\地图数据")

CITY_ALIASES = ["city", "cityname", "city_name", "name", "城市", "城市名", "市名"]
DATE_ALIASES = ["date", "datetime", "time", "day", "日期", "监测日期"]
PM25_ALIASES = ["pm25", "pm2.5", "pm_25", "pm2_5", "pm2d5", "浓度", "pm25浓度"]


@dataclass(frozen=True)
class RegionConfig:
    name: str
    cities: list[str]
    city_to_province: dict[str, str]
    pm25_csv_path: Path


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

    for file_path in files:
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


def read_nc_with_fallback(nc_file: Path, reader: Callable[[Any], pd.DataFrame]) -> pd.DataFrame:
    if xr is None:
        raise ImportError("xarray is required to read NetCDF files. Please install xarray.")

    attempts: list[tuple[Path, str]] = [(nc_file, "h5netcdf"), (nc_file, "netcdf4")]
    temp_dir_obj = None
    errors: list[str] = []

    try:
        for candidate_path, engine in attempts:
            try:
                with xr.open_dataset(candidate_path, engine=engine, decode_times=True) as ds:
                    return reader(ds)
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(f"{engine}@{candidate_path.name}: {exc}")

        if "[" in str(nc_file) or "]" in str(nc_file):
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="daily_nc_")
            temp_path = Path(temp_dir_obj.name) / nc_file.name
            shutil.copy2(nc_file, temp_path)
            for engine in ("h5netcdf", "netcdf4"):
                try:
                    with xr.open_dataset(temp_path, engine=engine, decode_times=True) as ds:
                        return reader(ds)
                except Exception as exc:  # pylint: disable=broad-except
                    errors.append(f"{engine}@temp:{exc}")
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()

    raise RuntimeError(f"Failed reading {nc_file.name}. " + " | ".join(errors[:3]))


def build_city_daily_rows_from_dataarray(
    *,
    base_module: Any,
    data_array: Any,
    file_name: str,
    city_gdf_wgs84: Any,
    value_col: str,
    spatial_extractor: Callable[[Any], Any],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    time_dim = infer_time_dim_name(data_array)
    out = reduce_non_spatial_dims(data_array, time_dim=time_dim)

    if time_dim is not None and time_dim in getattr(out, "dims", []):
        time_values = pd.to_datetime(pd.Index(out[time_dim].values), errors="coerce")
        norm_days = pd.Series(time_values).dt.normalize()
        valid_mask = norm_days.notna()

        if valid_mask.any():
            unique_days = sorted(pd.DatetimeIndex(norm_days[valid_mask].unique()))
            for day in unique_days:
                idx = np.where(norm_days == day)[0]
                day_slice = out.isel({time_dim: idx})
                day_field = day_slice.mean(dim=time_dim, skipna=True)
                field_2d = spatial_extractor(day_field)
                city_df = base_module._extract_city_mean_from_2d_field(field_2d, city_gdf_wgs84, value_col)
                city_df = city_df[["city", value_col]].copy()
                city_df["city"] = city_df["city"].map(normalize_city_name)
                city_df["date"] = pd.Timestamp(day).normalize()
                city_df[value_col] = pd.to_numeric(city_df[value_col], errors="coerce")
                rows.append(city_df[["city", "date", value_col]])

    if rows:
        return pd.concat(rows, ignore_index=True)

    date_scale_pairs = build_date_scale_pairs_from_filename(file_name)
    if not date_scale_pairs:
        return pd.DataFrame(columns=["city", "date", value_col])

    field_2d = spatial_extractor(out)
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


def load_pm25_daily_from_nc(
    *,
    pm25_day_dir: Path,
    city_geojson_dir: Path,
    base_module: Any,
    city_list: list[str],
) -> pd.DataFrame:
    if not pm25_day_dir.exists():
        raise FileNotFoundError(f"PM2.5 day directory not found: {pm25_day_dir}")
    if not city_geojson_dir.exists():
        raise FileNotFoundError(f"City geojson directory not found: {city_geojson_dir}")

    city_gdf_wgs84 = base_module._load_city_geojson(city_geojson_dir, city_list)
    nc_files = sorted(path for path in pm25_day_dir.rglob("*.nc") if path.is_file())
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in PM2.5 day directory: {pm25_day_dir}")

    rows: list[pd.DataFrame] = []
    for idx, nc_file in enumerate(nc_files, start=1):
        def reader(ds: Any) -> pd.DataFrame:
            standardized = base_module._standardize_nc_coords(ds)
            pm_var_name = base_module._choose_pm25_var(standardized)
            return build_city_daily_rows_from_dataarray(
                base_module=base_module,
                data_array=standardized[pm_var_name],
                file_name=nc_file.name,
                city_gdf_wgs84=city_gdf_wgs84,
                value_col="pm25",
                spatial_extractor=base_module._extract_2d_pm25,
            )

        try:
            one_df = read_nc_with_fallback(nc_file, reader)
            if not one_df.empty:
                rows.append(one_df)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Skip PM2.5 file: {nc_file.name} | {exc}")
        if idx % 200 == 0:
            print(f"[INFO] PM2.5 files processed: {idx}/{len(nc_files)}")

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
) -> pd.DataFrame:
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
    if not nc_files:
        print(f"[WARN] No ERA5 .nc files found: {era5_day_dir}")
        return pd.DataFrame(columns=["city", "date"])

    rows: list[pd.DataFrame] = []
    for idx, nc_file in enumerate(nc_files, start=1):
        folder_name = nc_file.parent.name.strip()
        value_col = f"met_{sanitize_factor_name(folder_name)}"

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
                file_name=nc_file.name,
                city_gdf_wgs84=city_gdf_wgs84,
                value_col=value_col,
                spatial_extractor=base_module._extract_2d_spatial_field,
            )

        try:
            one_df = read_nc_with_fallback(nc_file, reader)
            if not one_df.empty:
                rows.append(one_df)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Skip ERA5 file: {nc_file.name} | {exc}")
        if idx % 200 == 0:
            print(f"[INFO] ERA5 files processed: {idx}/{len(nc_files)}")

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
    merged = daily_df.merge(factor_df, on=["city", "year"], how="left")
    for col in factor_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce") / 365.0

    coverage = float(merged[factor_cols].notna().any(axis=1).mean()) if factor_cols else 0.0
    print(f"[INFO] Year-factor merge coverage: {coverage:.2%}")
    print("[INFO] Year-level factors have been converted to daily scale by dividing by 365.")
    return merged, factor_cols


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

    city_categories = sorted(train["city"].dropna().astype(str).unique().tolist())
    cluster_categories = sorted(train["cluster"].dropna().astype(str).unique().tolist())
    for frame in (train, valid, test):
        frame["city_code"] = pd.Categorical(frame["city"], categories=city_categories).codes
        frame["cluster_code"] = pd.Categorical(frame["cluster"], categories=cluster_categories).codes
        frame["city_code"] = frame["city_code"].replace(-1, np.nan)
        frame["cluster_code"] = frame["cluster_code"].replace(-1, np.nan)

    non_feature_cols = {"date", "city", "cluster", "pm25"}
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
    cluster_categories = sorted(train["cluster"].dropna().astype(str).unique().tolist())
    for frame in (train, valid, test):
        frame["city_code"] = pd.Categorical(frame["city"], categories=city_categories).codes
        frame["cluster_code"] = pd.Categorical(frame["cluster"], categories=cluster_categories).codes
        frame["city_code"] = frame["city_code"].replace(-1, np.nan)
        frame["cluster_code"] = frame["cluster_code"].replace(-1, np.nan)

    non_feature_cols = {"date", "city", "cluster", "pm25", "city_code", "cluster_code"}
    candidate_cols = [col for col in train.columns if col not in non_feature_cols]
    feature_cols = [col for col in candidate_cols if pd.api.types.is_numeric_dtype(train[col])]
    feature_cols = [col for col in feature_cols if train[col].notna().any()]
    if not feature_cols:
        raise ValueError("No usable numerical features detected for sequence matrices.")

    fill_values = train[feature_cols].median(numeric_only=True)
    for frame in (train, valid, test):
        for col in feature_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(fill_values[col])

    full_df = (
        pd.concat([train, valid, test], ignore_index=True)
        .sort_values(["city", "date"], kind="mergesort")
        .reset_index(drop=True)
    )

    def _collect_sequences(target_df: pd.DataFrame) -> tuple[list[np.ndarray], list[float], list[dict[str, Any]]]:
        x_list: list[np.ndarray] = []
        y_list: list[float] = []
        meta_list: list[dict[str, Any]] = []
        by_city = full_df.groupby("city", observed=True)
        for _, row in target_df.iterrows():
            city, date = row["city"], row["date"]
            if city not in by_city.groups:
                continue
            city_df = by_city.get_group(city)
            past = city_df.loc[city_df["date"] < date].tail(seq_len)
            if len(past) < seq_len:
                continue
            feat = past[feature_cols].values.astype(np.float32)
            x_list.append(feat)
            y_list.append(float(row["pm25"]))
            meta_list.append({
                "date": row["date"],
                "city": row["city"],
                "cluster": row["cluster"],
                "pm25": row["pm25"],
            })
        return x_list, y_list, meta_list

    x_train_list, y_train_list, meta_train_list = _collect_sequences(train)
    x_valid_list, y_valid_list, meta_valid_list = _collect_sequences(valid)
    x_test_list, y_test_list, meta_test_list = _collect_sequences(test)

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
) -> tuple[pd.DataFrame, list[str], list[str]]:
    base_module, configs = load_region_configs(correlation_dir, module_tag=module_tag)
    city_cluster_map = build_city_cluster_map(configs)
    city_list = city_cluster_map["city"].dropna().astype(str).unique().tolist()

    if daily_input:
        daily_df = load_daily_long_dataset(daily_input)
    else:
        daily_df = load_pm25_daily_from_nc(
            pm25_day_dir=pm25_day_dir,
            city_geojson_dir=city_geojson_dir,
            base_module=base_module,
            city_list=city_list,
        )

    daily_df = assign_cluster_and_filter_cities(daily_df, city_cluster_map)

    met_cols: list[str] = []
    if include_era5_daily:
        era5_df = load_era5_daily_features_from_nc(
            era5_day_dir=era5_day_dir,
            city_geojson_dir=city_geojson_dir,
            base_module=base_module,
            city_list=city_list,
            data_read_dir=data_read_dir,
            module_tag=module_tag,
        )
        if not era5_df.empty:
            met_cols = [col for col in era5_df.columns if col not in {"city", "date"}]
            daily_df = daily_df.merge(era5_df, on=["city", "date"], how="left")
            print(f"[INFO] Merged ERA5 daily features: {len(met_cols)} columns")

    year_factor_df = build_year_factor_panel(base_module, configs, data_read_dir)
    merged_df, year_factor_cols = merge_year_factors_with_daily_scaling(daily_df, year_factor_df)
    merged_df = scale_monthly_columns_to_daily(merged_df, monthly_cols=[])
    return merged_df, year_factor_cols, met_cols


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

    if plt is not None:
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

    if plt is None:
        return

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
    if shap is None:
        raise ImportError("shap is not installed. Please run: pip install shap")

    # PyTorch model: use DeepExplainer (model has .parameters(), expects tensor on same device)
    if torch is not None and hasattr(model, "parameters"):
        try:
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

    # Tree / sklearn-style model
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

    if plt is not None:
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
            if plt is not None:
                plt.close()

        top_imp = shap_importance.head(int(max(5, shap_max_display)))
        if not top_imp.empty:
            fig = plt.figure(figsize=(8, max(4, 0.35 * len(top_imp) + 1.5)))
            if sns is not None:
                sns.barplot(data=top_imp, x="mean_abs_shap", y="feature", color="#4C72B0")
            else:
                plt.barh(top_imp["feature"], top_imp["mean_abs_shap"], color="#4C72B0")
            plt.gca().invert_yaxis()
            plt.xlabel("Mean |SHAP value|")
            plt.ylabel("Feature")
            plt.title(f"{model_name} SHAP Feature Importance")
            plt.tight_layout()
            plt.savefig(output_dir / "shap_importance_bar.png", dpi=300)
            plt.close(fig)

    return shap_importance
