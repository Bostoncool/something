import glob
import hashlib
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

# 忽略警告，使用更明确的方式
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (16, 8)


class DataCache:
    """Simple cache based on file modification time."""

    def __init__(self, cache_dir: str = "cache_meteorology_nc") -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, filepath: str) -> str:
        stat = os.stat(filepath)
        payload = f"{filepath}_{stat.st_mtime}".encode("utf-8")
        return hashlib.md5(payload).hexdigest()

    def load(self, filepath: str) -> Optional[Dict[str, float]]:
        cache_file = os.path.join(self.cache_dir, f"{self._cache_key(filepath)}.pkl")
        if not os.path.exists(cache_file):
            return None
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            return None

    def save(self, filepath: str, data: Dict[str, float]) -> None:
        cache_file = os.path.join(self.cache_dir, f"{self._cache_key(filepath)}.pkl")
        pd.to_pickle(data, cache_file)


# 独立函数用于多进程处理
def _process_single_file_worker(
    filepath: str,
    cache_dir: str,
    beijing_lats: np.ndarray,
    beijing_lons: np.ndarray,
    era5_vars: List[str],
) -> Optional[Dict[str, float]]:
    """独立的工作函数，用于多进程处理单个文件。"""
    cache = DataCache(cache_dir)
    cached = cache.load(filepath)
    if cached:
        return cached

    try:
        with xr.open_dataset(
            filepath, engine="netcdf4", decode_times=True, cache=False
        ) as dataset:
            # 重命名坐标
            rename_map: Dict[str, str] = {}
            for time_key in (
                "valid_time",
                "forecast_time",
                "verification_time",
                "time1",
                "time2",
            ):
                if time_key in dataset.coords and "time" not in dataset.coords:
                    rename_map[time_key] = "time"
            if "lat" in dataset.coords and "latitude" not in dataset.coords:
                rename_map["lat"] = "latitude"
            if "lon" in dataset.coords and "longitude" not in dataset.coords:
                rename_map["lon"] = "longitude"
            if rename_map:
                dataset = dataset.rename(rename_map)

            try:
                dataset = xr.decode_cf(dataset)
            except Exception:
                pass

            for extra in ("expver", "surface"):
                if extra in dataset.variables:
                    dataset = dataset.drop_vars(extra)

            if "number" in dataset.dims:
                dataset = dataset.mean(dim="number", skipna=True)

            available_vars = [var for var in era5_vars if var in dataset.data_vars]
            if not available_vars:
                print(f"[WARN] {os.path.basename(filepath)} missing target variables, skipping")
                return None

            # 选择北京区域
            if "latitude" in dataset.coords and "longitude" in dataset.coords:
                latitude_values = dataset["latitude"].values
                if latitude_values[0] > latitude_values[-1]:
                    lat_slice = slice(beijing_lats.max(), beijing_lats.min())
                else:
                    lat_slice = slice(beijing_lats.min(), beijing_lats.max())

                dataset = dataset.sel(
                    latitude=lat_slice,
                    longitude=slice(beijing_lons.min(), beijing_lons.max()),
                )
                if "latitude" in dataset.dims and "longitude" in dataset.dims:
                    dataset = dataset.mean(dim=["latitude", "longitude"], skipna=True)

            if "time" not in dataset.coords:
                print(f"[WARN] {os.path.basename(filepath)} missing time dimension, skipping")
                return None

            dataset = dataset.sortby("time")
            dataset = dataset.resample(time="1D").mean(keep_attrs=False)
            dataset = dataset.dropna("time", how="all")
            if dataset.sizes.get("time", 0) == 0:
                print(f"[WARN] {os.path.basename(filepath)} no valid time steps, skipping")
                return None

            dataset.load()

            stats: Dict[str, float] = {}
            time_index = pd.to_datetime(dataset["time"].values)
            if len(time_index) > 0:
                first_time = time_index[0]
                stats["year"] = int(first_time.year)
                stats["month"] = int(first_time.month)
                stats["days"] = int(len(time_index))
            else:
                stats["days"] = int(dataset.sizes.get("time", 0))

            if "year" not in stats or "month" not in stats:
                match = re.search(r"(\d{4})(\d{2})", os.path.basename(filepath))
                if match:
                    stats["year"] = int(match.group(1))
                    stats["month"] = int(match.group(2))

            year = stats.get("year")
            month = stats.get("month")
            if year is not None and month is not None:
                stats["year_month"] = f"{int(year):04d}{int(month):02d}"
            else:
                stats["year_month"] = os.path.basename(filepath)

            for var in available_vars:
                try:
                    values = np.asarray(dataset[var].values, dtype=np.float32)
                    values = values[np.isfinite(values)]
                    if values.size == 0:
                        continue

                    if var in {"t2m", "d2m", "mn2t"} and np.nanmax(values) > 100:
                        values = values - 273.15

                    stats[f"{var}_mean"] = float(np.nanmean(values))
                    stats[f"{var}_std"] = float(np.nanstd(values))
                    stats[f"{var}_min"] = float(np.nanmin(values))
                    stats[f"{var}_max"] = float(np.nanmax(values))
                    stats[f"{var}_median"] = float(np.nanmedian(values))
                except Exception as err:
                    print(f"[ERROR] {os.path.basename(filepath)} variable {var} processing failed: {err}")
                    continue

            stats["source_file"] = os.path.basename(filepath)
            cache.save(filepath, stats)
            print(
                f"  [+] {os.path.basename(filepath)} -> variables: {len(available_vars)}, "
                f"time steps: {stats.get('days', 0)}"
            )
            return stats
    except Exception as exc:
        print(
            f"[ERROR] Failed to open {os.path.basename(filepath)}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


class MeteorologicalAnalyzerNC:
    """Beijing meteorological data analyzer based on ERA5 NetCDF."""

    def __init__(self, data_dir: str = ".", max_workers: Optional[int] = None) -> None:
        self.data_dir = data_dir
        self.max_workers = (
            max_workers
            if max_workers
            else min(max(4, (os.cpu_count() or 4) - 1), 12)
        )
        self.cache = DataCache()

        self.meteo_data = pd.DataFrame()

        self.beijing_lats = np.arange(39.0, 41.25, 0.25)
        self.beijing_lons = np.arange(115.0, 117.25, 0.25)

        self.era5_vars = [
            "d2m",
            "t2m",
            "u10",
            "v10",
            "u100",
            "v100",
            "blh",
            "sp",
            "tcwv",
            "tp",
            "avg_tprate",
            "tisr",
            "str",
            "cvh",
            "cvl",
            "mn2t",
            "sd",
            "lsm",
        ]

        self.param_names_en = {
            "t2m": "2m Temperature",
            "d2m": "2m Dewpoint Temperature",
            "blh": "Boundary Layer Height",
            "cvh": "High Vegetation Cover",
            "avg_tprate": "Mean Precipitation Rate",
            "u10": "10m U Wind Component",
            "v10": "10m V Wind Component",
            "u100": "100m U Wind Component",
            "v100": "100m V Wind Component",
            "lsm": "Land Sea Mask",
            "cvl": "Low Vegetation Cover",
            "mn2t": "Minimum 2m Temperature",
            "sp": "Surface Pressure",
            "sd": "Snow Depth",
            "str": "Surface Net Thermal Radiation",
            "tisr": "TOA Incident Solar Radiation",
            "tcwv": "Total Column Water Vapour",
            "tp": "Total Precipitation",
        }

    # -------------------------------------------------------------------------
    # Data Discovery
    # -------------------------------------------------------------------------
    def find_meteorological_files(self) -> List[str]:
        pattern = os.path.join(self.data_dir, "**", "*.nc")
        files = glob.glob(pattern, recursive=True)
        files.sort()

        print(f"Searching directory: {self.data_dir}")
        print(f"Found {len(files)} NetCDF files")
        if files:
            print(f"Example file: {os.path.basename(files[0])}")
        return files

    # -------------------------------------------------------------------------
    # NC File Processing
    # -------------------------------------------------------------------------
    def _rename_common_coords(self, dataset: xr.Dataset) -> xr.Dataset:
        rename_map: Dict[str, str] = {}
        for time_key in (
            "valid_time",
            "forecast_time",
            "verification_time",
            "time1",
            "time2",
        ):
            if time_key in dataset.coords and "time" not in dataset.coords:
                rename_map[time_key] = "time"
        if "lat" in dataset.coords and "latitude" not in dataset.coords:
            rename_map["lat"] = "latitude"
        if "lon" in dataset.coords and "longitude" not in dataset.coords:
            rename_map["lon"] = "longitude"
        return dataset.rename(rename_map) if rename_map else dataset

    def _select_beijing_region(self, dataset: xr.Dataset) -> xr.Dataset:
        if "latitude" not in dataset.coords or "longitude" not in dataset.coords:
            return dataset

        latitude_values = dataset["latitude"].values
        if latitude_values[0] > latitude_values[-1]:
            lat_slice = slice(self.beijing_lats.max(), self.beijing_lats.min())
        else:
            lat_slice = slice(self.beijing_lats.min(), self.beijing_lats.max())

        dataset = dataset.sel(
            latitude=lat_slice,
            longitude=slice(self.beijing_lons.min(), self.beijing_lons.max()),
        )
        if "latitude" in dataset.dims and "longitude" in dataset.dims:
            dataset = dataset.mean(dim=["latitude", "longitude"], skipna=True)
        return dataset

    # -------------------------------------------------------------------------
    # Main Data Loading Workflow
    # -------------------------------------------------------------------------
    def load_all_data(self) -> None:
        print("\nStarting to load meteorological data (NC) in parallel...")
        print(f"Number of processes: {self.max_workers}")

        files = self.find_meteorological_files()
        if not files:
            print("Error: No NetCDF files found")
            self.meteo_data = pd.DataFrame()
            return

        all_stats: List[Dict[str, float]] = []
        total = len(files)
        completed = 0

        # 准备参数用于多进程
        cache_dir = self.cache.cache_dir
        beijing_lats = self.beijing_lats
        beijing_lons = self.beijing_lons
        era5_vars = self.era5_vars

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_file_worker,
                    path,
                    cache_dir,
                    beijing_lats,
                    beijing_lons,
                    era5_vars,
                ): path
                for path in files
            }
            for future in as_completed(futures):
                filepath = futures[future]
                completed += 1
                try:
                    result = future.result()
                    if result:
                        all_stats.append(result)
                except Exception as exc:
                    print(f"[ERROR] Failed to process {os.path.basename(filepath)}: {exc}")

                if completed % 20 == 0 or completed == total:
                    print(f"Progress: [{completed}/{total}] Processed {os.path.basename(filepath)}")

        if not all_stats:
            print("Error: Data loading failed, no valid statistics obtained")
            self.meteo_data = pd.DataFrame()
            return

        aggregated: Dict[Tuple[int, int], Dict[str, float]] = {}
        leftovers: List[Dict[str, float]] = []

        for item in all_stats:
            year = item.get("year")
            month = item.get("month")
            if year is None or month is None or pd.isna(year) or pd.isna(month):
                leftovers.append(item)
                continue

            year_int = int(year)
            month_int = int(month)
            key = (year_int, month_int)

            bucket = aggregated.setdefault(
                key,
                {
                    "year": year_int,
                    "month": month_int,
                    "year_month": item.get(
                        "year_month", f"{year_int:04d}{month_int:02d}"
                    ),
                    "days": int(item.get("days", 0)),
                    "source_files": [],
                },
            )

            bucket["days"] = max(bucket.get("days", 0), int(item.get("days", 0)))

            source_file = item.get("source_file")
            if source_file:
                sources = bucket.setdefault("source_files", [])
                sources.append(source_file)

            for key_name, value in item.items():
                if key_name in {
                    "source_file",
                    "year",
                    "month",
                    "year_month",
                    "days",
                }:
                    continue
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue
                if key_name not in bucket or pd.isna(bucket[key_name]):
                    bucket[key_name] = value

        if aggregated:
            aggregated_list: List[Dict[str, float]] = []
            for key in sorted(aggregated.keys()):
                row = aggregated[key]
                if "source_files" in row:
                    unique_sources = []
                    for src in row["source_files"]:
                        if src not in unique_sources:
                            unique_sources.append(src)
                    row["source_files"] = "|".join(unique_sources)
                aggregated_list.append(row)
            aggregated_list.extend(leftovers)
            self.meteo_data = pd.DataFrame(aggregated_list)
        else:
            self.meteo_data = pd.DataFrame(all_stats)
        self._finalize_dataset()

    def _finalize_dataset(self) -> None:
        numeric_cols = self.meteo_data.select_dtypes(include=[np.number]).columns
        self.meteo_data[numeric_cols] = self.meteo_data[numeric_cols].replace(
            [np.inf, -np.inf], np.nan
        )
        self.meteo_data = self.meteo_data.sort_values(["year", "month"]).reset_index(
            drop=True
        )

        if {"year", "month"}.issubset(self.meteo_data.columns):
            # 使用字典方式创建日期，更简洁且避免字符串拼接
            self.meteo_data["date"] = pd.to_datetime(
                {
                    "year": self.meteo_data["year"].astype(int),
                    "month": self.meteo_data["month"].astype(int),
                    "day": 1,
                }
            )

        if {"u10_mean", "v10_mean"}.issubset(self.meteo_data.columns):
            self.meteo_data["wind_speed_10m"] = np.sqrt(
                self.meteo_data["u10_mean"] ** 2 + self.meteo_data["v10_mean"] ** 2
            )
        if {"u100_mean", "v100_mean"}.issubset(self.meteo_data.columns):
            self.meteo_data["wind_speed_100m"] = np.sqrt(
                self.meteo_data["u100_mean"] ** 2 + self.meteo_data["v100_mean"] ** 2
            )

        print("\nMeteorological data loading completed!")
        print(f"Total records: {len(self.meteo_data)}")
        if "date" in self.meteo_data.columns:
            print(
                f"Time range: {self.meteo_data['date'].min()} -> "
                f"{self.meteo_data['date'].max()}"
            )

    # -------------------------------------------------------------------------
    # Visualization and Statistics
    # -------------------------------------------------------------------------
    def plot_temperature_timeseries(self, save_path: str = "temperature_timeseries_nc.tif") -> None:
        if self.meteo_data.empty:
            print("Error: No data available for temperature time series plotting")
            return

        print("\nStarting to plot temperature time series...")

        temp_params = ["t2m", "d2m", "mn2t"]
        available_temps = [p for p in temp_params if f"{p}_mean" in self.meteo_data.columns]

        if not available_temps:
            print("Warning: No available temperature variables")
            return

        fig, ax = plt.subplots(figsize=(18, 6))
        colors = ["#e74c3c", "#3498db", "#2ecc71"]

        for idx, param in enumerate(available_temps):
            mean_col = f"{param}_mean"
            min_col = f"{param}_min"
            max_col = f"{param}_max"

            ax.plot(
                self.meteo_data["date"],
                self.meteo_data[mean_col],
                linewidth=2.5,
                color=colors[idx],
                alpha=0.9,
                label=f"{self.param_names_en.get(param, param)} (Mean)",
                marker="o",
                markersize=4,
            )

            if min_col in self.meteo_data.columns and max_col in self.meteo_data.columns:
                ax.fill_between(
                    self.meteo_data["date"],
                    self.meteo_data[min_col],
                    self.meteo_data[max_col],
                    alpha=0.2,
                    color=colors[idx],
                )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="0°C")
        ax.set_xlabel("Time", fontsize=14, fontweight="bold")
        ax.set_ylabel("Temperature (°C)", fontsize=14, fontweight="bold")
        ax.set_title("Beijing Temperature Time Series (NC)", fontsize=18, fontweight="bold", pad=30)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=11, framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
        plt.xticks(rotation=45, ha="right")

        if "t2m_mean" in self.meteo_data.columns:
            stats_text = (
                "2m Temperature Statistics:\n"
                f"Sample: {len(self.meteo_data)} months\n"
                f"Mean: {self.meteo_data['t2m_mean'].mean():.2f} °C\n"
                f"Median: {self.meteo_data['t2m_mean'].median():.2f} °C\n"
                f"Max: {self.meteo_data['t2m_mean'].max():.2f} °C\n"
                f"Min: {self.meteo_data['t2m_mean'].min():.2f} °C"
            )
            ax.text(
                0.02,
                0.02,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", format="tif")
        print(f"Temperature time series plot saved to: {save_path}")
        plt.close()  # 关闭图形，确保图片单独输出

    def plot_wind_speed_timeseries(self, save_path: str = "wind_speed_timeseries_nc.tif") -> None:
        if self.meteo_data.empty:
            print("Error: No data available for wind speed time series")
            return

        print("\nStarting to plot wind speed time series...")

        if "wind_speed_10m" not in self.meteo_data.columns and {
            "u10_mean",
            "v10_mean",
        }.issubset(self.meteo_data.columns):
            self.meteo_data["wind_speed_10m"] = np.sqrt(
                self.meteo_data["u10_mean"] ** 2 + self.meteo_data["v10_mean"] ** 2
            )
        if "wind_speed_100m" not in self.meteo_data.columns and {
            "u100_mean",
            "v100_mean",
        }.issubset(self.meteo_data.columns):
            self.meteo_data["wind_speed_100m"] = np.sqrt(
                self.meteo_data["u100_mean"] ** 2 + self.meteo_data["v100_mean"] ** 2
            )

        available = [
            col for col in ["wind_speed_10m", "wind_speed_100m"] if col in self.meteo_data.columns
        ]
        if not available:
            print("Warning: No available wind speed variables")
            return

        fig, ax = plt.subplots(figsize=(18, 6))

        if "wind_speed_10m" in available:
            ax.plot(
                self.meteo_data["date"],
                self.meteo_data["wind_speed_10m"],
                linewidth=2.5,
                color="#3498db",
                alpha=0.9,
                label="10m Wind Speed",
                marker="o",
                markersize=4,
            )
        if "wind_speed_100m" in available:
            ax.plot(
                self.meteo_data["date"],
                self.meteo_data["wind_speed_100m"],
                linewidth=2.5,
                color="#e74c3c",
                alpha=0.9,
                label="100m Wind Speed",
                marker="s",
                markersize=4,
            )

        ax.set_xlabel("Time", fontsize=14, fontweight="bold")
        ax.set_ylabel("Wind Speed (m/s)", fontsize=14, fontweight="bold")
        ax.set_title("Beijing Wind Speed Time Series (NC)", fontsize=18, fontweight="bold", pad=30)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=11)
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", format="tif")
        print(f"Wind speed time series plot saved to: {save_path}")
        plt.close()

    def plot_wind_speed_distribution(self, save_path: str = "wind_speed_distribution_nc.tif") -> None:
        if self.meteo_data.empty:
            print("Error: No data available for wind speed distribution")
            return

        print("\nStarting to plot wind speed distribution...")

        if "wind_speed_10m" not in self.meteo_data.columns and {
            "u10_mean",
            "v10_mean",
        }.issubset(self.meteo_data.columns):
            self.meteo_data["wind_speed_10m"] = np.sqrt(
                self.meteo_data["u10_mean"] ** 2 + self.meteo_data["v10_mean"] ** 2
            )
        if "wind_speed_100m" not in self.meteo_data.columns and {
            "u100_mean",
            "v100_mean",
        }.issubset(self.meteo_data.columns):
            self.meteo_data["wind_speed_100m"] = np.sqrt(
                self.meteo_data["u100_mean"] ** 2 + self.meteo_data["v100_mean"] ** 2
            )

        available = [
            col for col in ["wind_speed_10m", "wind_speed_100m"] if col in self.meteo_data.columns
        ]
        if not available:
            print("Warning: No available wind speed variables")
            return

        fig, ax = plt.subplots(figsize=(16, 8))

        wind_data = []
        labels = []
        if "wind_speed_10m" in available:
            wind_data.append(self.meteo_data["wind_speed_10m"].dropna())
            labels.append("10m Wind Speed")
        if "wind_speed_100m" in available:
            wind_data.append(self.meteo_data["wind_speed_100m"].dropna())
            labels.append("100m Wind Speed")

        ax.hist(
            wind_data,
            bins=30,
            label=labels,
            color=["#3498db", "#e74c3c"][: len(wind_data)],
            alpha=0.7,
            edgecolor="white",
            linewidth=1,
        )
        ax.set_xlabel("Wind Speed (m/s)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=14, fontweight="bold")
        ax.set_title("Wind Speed Distribution (NC)", fontsize=18, fontweight="bold", pad=30)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=11)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", format="tif")
        print(f"Wind speed distribution plot saved to: {save_path}")
        plt.close()

    def plot_precipitation_timeseries(self, save_path: str = "precipitation_timeseries_nc.tif") -> None:
        if self.meteo_data.empty:
            print("Error: No data available for precipitation time series")
            return

        print("\nStarting to plot precipitation time series...")

        if "tp_mean" not in self.meteo_data.columns:
            print("Warning: No available total precipitation variable")
            return

        fig, ax = plt.subplots(figsize=(18, 6))

        ax.bar(
            self.meteo_data["date"],
            self.meteo_data["tp_mean"],
            width=20,
            color="#3498db",
            alpha=0.7,
            label="Total Precipitation",
        )
        ax.set_xlabel("Time", fontsize=14, fontweight="bold")
        ax.set_ylabel("Precipitation (mm)", fontsize=14, fontweight="bold")
        ax.set_title("Beijing Precipitation Time Series (NC)", fontsize=18, fontweight="bold", pad=30)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.legend(loc="upper right", fontsize=11)
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", format="tif")
        print(f"Precipitation time series plot saved to: {save_path}")
        plt.close()

    def plot_precipitation_rate_timeseries(self, save_path: str = "precipitation_rate_timeseries_nc.tif") -> None:
        if self.meteo_data.empty:
            print("Error: No data available for precipitation rate time series")
            return

        print("\nStarting to plot precipitation rate time series...")

        if "avg_tprate_mean" not in self.meteo_data.columns:
            print("Warning: No available precipitation rate variable")
            return

        fig, ax = plt.subplots(figsize=(18, 6))

        ax.plot(
            self.meteo_data["date"],
            self.meteo_data["avg_tprate_mean"],
            linewidth=2.5,
            color="#2ecc71",
            alpha=0.9,
            label="Mean Precipitation Rate",
            marker="o",
            markersize=4,
        )
        ax.set_xlabel("Time", fontsize=14, fontweight="bold")
        ax.set_ylabel("Precipitation Rate (mm/h)", fontsize=14, fontweight="bold")
        ax.set_title(
            "Beijing Mean Precipitation Rate Time Series (NC)", fontsize=18, fontweight="bold", pad=30
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=11)
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", format="tif")
        print(f"Precipitation rate time series plot saved to: {save_path}")
        plt.close()

    def _get_season(self, month: int) -> str:
        """辅助函数：根据月份返回季节"""
        if month in (12, 1, 2):
            return "Winter"
        if month in (3, 4, 5):
            return "Spring"
        if month in (6, 7, 8):
            return "Summer"
        return "Autumn"

    def plot_temperature_seasonal(self, save_path: str = "temperature_seasonal_nc.tif") -> None:
        if self.meteo_data.empty:
            print("Error: No data available for temperature seasonal analysis")
            return

        print("\nStarting to plot temperature seasonal distribution...")

        if "t2m_mean" not in self.meteo_data.columns:
            print("Warning: No available temperature variable")
            return

        if "season" not in self.meteo_data.columns:
            self.meteo_data["season"] = self.meteo_data["month"].astype(int).apply(self._get_season)
        season_order = ["Spring", "Summer", "Autumn", "Winter"]

        fig, ax = plt.subplots(figsize=(12, 12))

        try:
            plot_data = self.meteo_data[["season", "t2m_mean"]].dropna()
            if not plot_data.empty:
                sns.boxplot(
                    data=plot_data,
                    x="season",
                    y="t2m_mean",
                    order=season_order,
                    palette="Set2",
                    ax=ax,
                )
                ax.set_xlabel("Season", fontsize=14, fontweight="bold")
                ax.set_ylabel("Temperature (°C)", fontsize=14, fontweight="bold")
                ax.set_title(
                    "2m Temperature Seasonal Distribution (NC)",
                    fontsize=18,
                    fontweight="bold",
                    pad=30,
                )
                ax.grid(True, alpha=0.3, linestyle="--", axis="y")
            else:
                ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
        except Exception as err:
            print(f"Temperature seasonal distribution plotting failed: {err}")
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", format="tif")
        print(f"Temperature seasonal distribution plot saved to: {save_path}")
        plt.close()

    def plot_wind_pressure_seasonal(self, save_path: str = "wind_pressure_seasonal_nc.tif") -> None:
        if self.meteo_data.empty:
            print("Error: No data available for wind/pressure seasonal analysis")
            return

        print("\nStarting to plot wind/pressure seasonal distribution...")

        if "season" not in self.meteo_data.columns:
            self.meteo_data["season"] = self.meteo_data["month"].astype(int).apply(self._get_season)
        season_order = ["Spring", "Summer", "Autumn", "Winter"]

        target_var = (
            "wind_speed_10m"
            if "wind_speed_10m" in self.meteo_data.columns
            else "sp_mean"
            if "sp_mean" in self.meteo_data.columns
            else None
        )

        if not target_var:
            print("Warning: No available wind speed or surface pressure variable")
            return

        fig, ax = plt.subplots(figsize=(12, 12))

        try:
            plot_data = self.meteo_data[["season", target_var]].dropna()
            if not plot_data.empty:
                palette = "Set3" if target_var == "wind_speed_10m" else "coolwarm"
                ylabel = "Wind Speed (m/s)" if target_var == "wind_speed_10m" else "Surface Pressure (Pa)"
                title = (
                    "10m Wind Speed Seasonal Distribution (NC)"
                    if target_var == "wind_speed_10m"
                    else "Surface Pressure Seasonal Distribution (NC)"
                )
                sns.boxplot(
                    data=plot_data,
                    x="season",
                    y=target_var,
                    order=season_order,
                    palette=palette,
                    ax=ax,
                )
                ax.set_xlabel("Season", fontsize=14, fontweight="bold")
                ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
                ax.set_title(title, fontsize=18, fontweight="bold", pad=30)
                ax.grid(True, alpha=0.3, linestyle="--", axis="y")
            else:
                ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
        except Exception as err:
            print(f"Wind/pressure seasonal distribution plotting failed: {err}")
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", format="tif")
        print(f"Wind/pressure seasonal distribution plot saved to: {save_path}")
        plt.close()

    def plot_precipitation_seasonal(self, save_path: str = "precipitation_seasonal_nc.tif") -> None:
        if self.meteo_data.empty:
            print("Error: No data available for precipitation seasonal analysis")
            return

        print("\nStarting to plot precipitation seasonal distribution...")

        if "tp_mean" not in self.meteo_data.columns:
            print("Warning: No available precipitation variable")
            return

        if "season" not in self.meteo_data.columns:
            self.meteo_data["season"] = self.meteo_data["month"].astype(int).apply(self._get_season)
        season_order = ["Spring", "Summer", "Autumn", "Winter"]

        fig, ax = plt.subplots(figsize=(12, 12))

        try:
            plot_data = self.meteo_data[["season", "tp_mean"]].dropna()
            if not plot_data.empty:
                sns.boxplot(
                    data=plot_data,
                    x="season",
                    y="tp_mean",
                    order=season_order,
                    palette="Blues",
                    ax=ax,
                )
                ax.set_xlabel("Season", fontsize=14, fontweight="bold")
                ax.set_ylabel("Precipitation (mm)", fontsize=14, fontweight="bold")
                ax.set_title(
                    "Precipitation Seasonal Distribution (NC)",
                    fontsize=18,
                    fontweight="bold",
                    pad=30,
                )
                ax.grid(True, alpha=0.3, linestyle="--", axis="y")
            else:
                ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
        except Exception as err:
            print(f"Precipitation seasonal distribution plotting failed: {err}")
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", format="tif")
        print(f"Precipitation seasonal distribution plot saved to: {save_path}")
        plt.close()

    def plot_blh_seasonal(self, save_path: str = "blh_seasonal_nc.tif") -> None:
        if self.meteo_data.empty:
            print("Error: No data available for boundary layer height seasonal analysis")
            return

        print("\nStarting to plot boundary layer height seasonal distribution...")

        if "blh_mean" not in self.meteo_data.columns:
            print("Warning: No available boundary layer height variable")
            return

        if "season" not in self.meteo_data.columns:
            self.meteo_data["season"] = self.meteo_data["month"].astype(int).apply(self._get_season)
        season_order = ["Spring", "Summer", "Autumn", "Winter"]

        fig, ax = plt.subplots(figsize=(12, 12))

        try:
            plot_data = self.meteo_data[["season", "blh_mean"]].dropna()
            if not plot_data.empty:
                sns.boxplot(
                    data=plot_data,
                    x="season",
                    y="blh_mean",
                    order=season_order,
                    palette="Greens",
                    ax=ax,
                )
                ax.set_xlabel("Season", fontsize=14, fontweight="bold")
                ax.set_ylabel("Boundary Layer Height (m)", fontsize=14, fontweight="bold")
                ax.set_title(
                    "Boundary Layer Height Seasonal Distribution (NC)",
                    fontsize=18,
                    fontweight="bold",
                    pad=30,
                )
                ax.grid(True, alpha=0.3, linestyle="--", axis="y")
            else:
                ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")
        except Exception as err:
            print(f"Boundary layer height seasonal distribution plotting failed: {err}")
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center", va="center")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", format="tif")
        print(f"Boundary layer height seasonal distribution plot saved to: {save_path}")
        plt.close()

    def save_data_summary(self, save_path: str = "meteorological_summary_nc.csv") -> None:
        if self.meteo_data.empty:
            print("Error: No data to export")
            return

        print("\nStarting to save meteorological statistics summary...")

        self.meteo_data.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"Complete statistics saved to: {save_path}")

        numeric_cols = [
            col
            for col in self.meteo_data.select_dtypes(include=[np.number]).columns
            if col not in {"year", "month"}
        ]
        summary = self.meteo_data[numeric_cols].describe().T
        summary["parameter"] = summary.index
        summary = summary[
            ["parameter", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        ]
        summary_path = save_path.replace(".csv", "_statistics.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"Statistics summary saved to: {summary_path}")

    def clean_cache(self) -> None:
        """清理缓存目录中的所有缓存文件"""
        cache_dir = self.cache.cache_dir
        if not os.path.exists(cache_dir):
            print(f"\nCache directory does not exist: {cache_dir}")
            return

        try:
            cache_files = glob.glob(os.path.join(cache_dir, "*.pkl"))
            if not cache_files:
                print(f"\nNo cache files found in {cache_dir}")
                return

            removed_count = 0
            for cache_file in cache_files:
                try:
                    os.remove(cache_file)
                    removed_count += 1
                except Exception as err:
                    print(f"[WARN] Failed to remove {os.path.basename(cache_file)}: {err}")

            print(f"\nCache cleanup completed: {removed_count} cache files removed from {cache_dir}")
            
            # 尝试删除空目录（如果目录为空）
            try:
                if not os.listdir(cache_dir):
                    os.rmdir(cache_dir)
                    print(f"Empty cache directory removed: {cache_dir}")
            except Exception:
                pass  # 如果目录不为空或删除失败，忽略错误
        except Exception as err:
            print(f"[ERROR] Cache cleanup failed: {err}")

    # -------------------------------------------------------------------------
    # Main Workflow
    # -------------------------------------------------------------------------
    def run_analysis(self) -> None:
        print("=" * 60)
        print("Beijing Meteorological Data Analysis (NC)")
        print("=" * 60)

        self.load_all_data()
        if self.meteo_data.empty:
            print("Error: Data loading failed, please check data path and file format")
            return

        self.plot_temperature_timeseries()
        self.plot_wind_speed_timeseries()
        self.plot_wind_speed_distribution()
        self.plot_precipitation_timeseries()
        self.plot_precipitation_rate_timeseries()
        self.plot_temperature_seasonal()
        self.plot_wind_pressure_seasonal()
        self.plot_precipitation_seasonal()
        self.plot_blh_seasonal()
        self.save_data_summary()

        print("\nAnalysis completed!")
        
        # 清理缓存
        self.clean_cache()


def main() -> None:
    data_dir = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-NC"
    if not os.path.exists(data_dir):
        fallback_paths = [
            "/root/autodl-tmp/ERA5-Beijing-NC",
            "/root/autodl-tmp/ERA5-Data",
            "/root/autodl-tmp/Short-essay-Beijing/Statistics/Meteorological/NC",
        ]
        for path in fallback_paths:
            if os.path.exists(path):
                data_dir = path
                print(f"Using fallback path: {data_dir}")
                break
        else:
            print("Error: ERA5 NetCDF data directory not found, please adjust data_dir in main()")
            return

    print(f"Data directory: {data_dir}")
    print("To change the data path, please modify the data_dir variable in the main() function\n")

    analyzer = MeteorologicalAnalyzerNC(data_dir=data_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()


