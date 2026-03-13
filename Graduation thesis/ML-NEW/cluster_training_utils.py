from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd


def is_memory_pressure_error(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    memory_tokens = [
        "out of memory",
        "cannot allocate memory",
        "unable to allocate",
        "std::bad_alloc",
        "cuda out of memory",
    ]
    return any(token in text for token in memory_tokens)


def merge_prepare_stats(total: dict[str, Any], one: dict[str, Any]) -> None:
    total["cache_hit_pm25"] = bool(total.get("cache_hit_pm25", False) or one.get("cache_hit_pm25", False))
    total["cache_hit_era5"] = bool(total.get("cache_hit_era5", False) or one.get("cache_hit_era5", False))
    for key in (
        "pm25_seconds",
        "era5_seconds",
        "year_factor_seconds",
        "data_prepare_seconds",
        "pm25_open_dataset_seconds",
        "pm25_spatial_aggregate_seconds",
        "pm25_files_per_second",
    ):
        total[key] = float(total.get(key, 0.0)) + float(one.get(key, 0.0))
    for key in ("pm25_grid_cache_hits", "pm25_grid_cache_misses", "pm25_processed_files", "pm25_failed_files"):
        total[key] = int(total.get(key, 0)) + int(one.get(key, 0))


def discover_years_from_pm25_index(pm25_nc_index: dict[str, Any] | None) -> list[int]:
    if not pm25_nc_index:
        return []
    year_to_files = pm25_nc_index.get("year_to_files", {})
    years: list[int] = []
    for year in year_to_files.keys():
        try:
            years.append(int(year))
        except (TypeError, ValueError):
            continue
    return sorted(set(years))


def prepare_training_table_with_fallback(
    *,
    module_tag: str,
    correlation_dir: Path,
    data_read_dir: Path,
    city_geojson_dir: Path,
    daily_input: list[str] | None,
    pm25_day_dir: Path,
    era5_day_dir: Path,
    include_era5_daily: bool,
    pm25_workers: int,
    era5_workers: int,
    cache_dir: Path,
    cache_enabled: bool,
    train_end_year: int,
    valid_year: int,
    test_year: int,
    prepare_fn: Callable[..., tuple[pd.DataFrame, list[str], list[str]]],
    prebuilt_pm25_nc_index: dict[str, Any] | None = None,
    use_year_factors: bool = True,
    pm25_precropped: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any], str, list[int]]:
    years = discover_years_from_pm25_index(prebuilt_pm25_nc_index)
    needed_years = sorted(
        {
            year
            for year in years
            if year <= int(train_end_year) or year == int(valid_year) or year == int(test_year)
        }
    )
    full_allowed_years = None if daily_input else (needed_years or None)
    prepare_stats_main: dict[str, Any] = {}
    try:
        base_df, year_factor_cols, met_cols = prepare_fn(
            module_tag=module_tag,
            correlation_dir=correlation_dir,
            data_read_dir=data_read_dir,
            city_geojson_dir=city_geojson_dir,
            daily_input=daily_input,
            pm25_day_dir=pm25_day_dir,
            era5_day_dir=era5_day_dir,
            include_era5_daily=include_era5_daily,
            allowed_years=full_allowed_years,
            pm25_workers=pm25_workers,
            era5_workers=era5_workers,
            cache_dir=cache_dir,
            enable_cache=cache_enabled,
            prebuilt_pm25_nc_index=prebuilt_pm25_nc_index,
            prepare_stats=prepare_stats_main,
            use_year_factors=use_year_factors,
            pm25_precropped=pm25_precropped,
        )
        return base_df, year_factor_cols, met_cols, prepare_stats_main, "full", []
    except Exception as exc:
        if not is_memory_pressure_error(exc):
            raise
        if daily_input:
            raise RuntimeError("Daily input mode does not support yearly fallback.") from exc
        if not needed_years:
            raise RuntimeError("Yearly fallback cannot infer available years from PM2.5 index.") from exc

        all_frames: list[pd.DataFrame] = []
        year_factor_col_set: set[str] = set()
        met_col_set: set[str] = set()
        prepare_stats_main = {}
        for year in needed_years:
            one_stats: dict[str, Any] = {}
            one_df, one_year_factor_cols, one_met_cols = prepare_fn(
                module_tag=module_tag,
                correlation_dir=correlation_dir,
                data_read_dir=data_read_dir,
                city_geojson_dir=city_geojson_dir,
                daily_input=daily_input,
                pm25_day_dir=pm25_day_dir,
                era5_day_dir=era5_day_dir,
                include_era5_daily=include_era5_daily,
                allowed_years=[year],
                pm25_workers=pm25_workers,
                era5_workers=era5_workers,
                cache_dir=cache_dir,
                enable_cache=cache_enabled,
                prebuilt_pm25_nc_index=prebuilt_pm25_nc_index,
                prepare_stats=one_stats,
                use_year_factors=use_year_factors,
            )
            all_frames.append(one_df)
            year_factor_col_set.update(one_year_factor_cols)
            met_col_set.update(one_met_cols)
            merge_prepare_stats(prepare_stats_main, one_stats)

        base_df = pd.concat(all_frames, ignore_index=True).sort_values(["date", "city"], kind="mergesort")
        return base_df, sorted(year_factor_col_set), sorted(met_col_set), prepare_stats_main, "yearly_fallback", needed_years
