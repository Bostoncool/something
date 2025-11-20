import os
import glob
import hashlib
import multiprocessing
import pickle
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Matplotlib configuration
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 12
mpl.rcParams["figure.figsize"] = (10, 6)


def process_single_nc_file(
    filepath: str,
    beijing_lats: np.ndarray,
    beijing_lons: np.ndarray,
    era5_vars: List[str],
) -> Optional[Dict[str, Any]]:
    """Process a single ERA5 NetCDF file and extract statistical features (独立函数用于多进程)."""
    # 创建临时缓存实例用于多进程
    cache = DataCache()
    cached_data = None
    try:
        cached_data = cache.get_cached_data(filepath)
    except Exception:
        cached_data = None

    if cached_data:
        return cached_data

    try:
        with xr.open_dataset(
            filepath, engine="netcdf4", decode_times=True
        ) as dataset:
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

            drop_vars: List[str] = []
            for extra_coord in ("expver", "surface"):
                if extra_coord in dataset:
                    drop_vars.append(extra_coord)
            if drop_vars:
                dataset = dataset.drop_vars(drop_vars)

            if "number" in dataset.dims:
                dataset = dataset.mean(dim="number", skipna=True)

            available_vars = [
                var for var in era5_vars if var in dataset.data_vars
            ]
            if not available_vars:
                print(
                    f"[WARN] {os.path.basename(filepath)} does not contain target variables, skipped"
                )
                return None

            if "latitude" in dataset.coords and "longitude" in dataset.coords:
                lat_values = dataset["latitude"]
                if lat_values[0] > lat_values[-1]:
                    lat_slice = slice(
                        beijing_lats.max(), beijing_lats.min()
                    )
                else:
                    lat_slice = slice(
                        beijing_lats.min(), beijing_lats.max()
                    )
                dataset = dataset.sel(
                    latitude=lat_slice,
                    longitude=slice(
                        beijing_lons.min(), beijing_lons.max()
                    ),
                )
                if "latitude" in dataset.dims and "longitude" in dataset.dims:
                    dataset = dataset.mean(
                        dim=["latitude", "longitude"], skipna=True
                    )

            if "time" not in dataset.coords:
                print(f"[WARN] {os.path.basename(filepath)} missing time coordinate, skipped")
                return None

            dataset = dataset.sortby("time")
            dataset = dataset.resample(time="1D").mean()
            dataset = dataset.dropna("time", how="all")

            if dataset.sizes.get("time", 0) == 0:
                print(
                    f"[WARN] {os.path.basename(filepath)} no valid time after resampling, skipped"
                )
                return None

            stats: Dict[str, Any] = {"source_file": os.path.basename(filepath)}

            try:
                time_index = pd.to_datetime(dataset["time"].values)
                if len(time_index) > 0:
                    first_time = time_index[0]
                    stats["year"] = int(first_time.year)
                    stats["month"] = int(first_time.month)
                    stats["days"] = int(len(time_index))
            except Exception:
                stats["year"] = np.nan
                stats["month"] = np.nan
                stats["days"] = dataset.sizes.get("time", np.nan)

            if pd.isna(stats.get("year")) or pd.isna(stats.get("month")):
                filename = os.path.basename(filepath)
                match = re.search(r"(\d{4})(\d{2})", filename)
                if match:
                    stats["year"] = int(match.group(1))
                    stats["month"] = int(match.group(2))

            day_count = stats.get("days", 0)
            if pd.isna(day_count):
                day_count = 0

            for var in available_vars:
                try:
                    values = dataset[var].values
                    values = values[np.isfinite(values)]
                    if values.size == 0:
                        stats[f"{var}_mean"] = np.nan
                        stats[f"{var}_std"] = np.nan
                        stats[f"{var}_min"] = np.nan
                        stats[f"{var}_max"] = np.nan
                        continue
                    if var in {"t2m", "mn2t", "d2m"} and np.nanmax(values) > 100:
                        values = values - 273.15
                    stats[f"{var}_mean"] = float(np.nanmean(values))
                    stats[f"{var}_std"] = float(np.nanstd(values))
                    stats[f"{var}_min"] = float(np.nanmin(values))
                    stats[f"{var}_max"] = float(np.nanmax(values))
                except Exception as err:
                    print(
                        f"Error extracting variable {var} from {os.path.basename(filepath)}: {err}"
                    )
                    stats[f"{var}_mean"] = np.nan
                    stats[f"{var}_std"] = np.nan
                    stats[f"{var}_min"] = np.nan
                    stats[f"{var}_max"] = np.nan

            cache.save_cached_data(filepath, stats)
            print(
                f"  [+] {os.path.basename(filepath)} -> Variables: "
                f"{len(available_vars)}, Days: {int(day_count)}"
            )
            return stats
    except Exception as exc:
        print(
            f"[ERROR] Failed to process {os.path.basename(filepath)}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


class DataCache:
    """Cache processing results to avoid redundant calculations."""

    def __init__(self, cache_dir: str = "cache") -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_key(self, filepath: str) -> str:
        file_stat = os.stat(filepath)
        return hashlib.md5(
            f"{filepath}_{file_stat.st_mtime}".encode()
        ).hexdigest()

    def get_cached_data(self, filepath: str) -> Optional[Dict[str, Any]]:
        cache_key = self.get_cache_key(filepath)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as handle:
                    return pickle.load(handle)
            except Exception:
                return None
        return None

    def save_cached_data(self, filepath: str, data: Dict[str, Any]) -> None:
        cache_key = self.get_cache_key(filepath)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        with open(cache_file, "wb") as handle:
            pickle.dump(data, handle)

    def clear_cache(self) -> None:
        if not os.path.exists(self.cache_dir):
            return
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cache directory cleared: {self.cache_dir}")


class BeijingPCAAnalyzerNC:
    """PCA analyzer for Beijing multi-meteorological factor pollution variation (NC data)."""

    def __init__(
        self,
        meteo_data_dir: str = ".",
        pollution_data_dir: str = ".",
        extra_pollution_data_dir: str = ".",
    ) -> None:
        self.meteo_data_dir = meteo_data_dir
        self.pollution_data_dir = pollution_data_dir
        self.extra_pollution_data_dir = extra_pollution_data_dir

        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None

        self.meteorological_data: List[Dict[str, Any]] = []
        self.pollution_data: List[Dict[str, Any]] = []
        self.extra_pollution_data: List[Dict[str, Any]] = []

        self.cache = DataCache()
        
        # 文件名到路径的字典映射，用于O(1)查找
        self.pollution_file_map: Dict[str, str] = {}
        self.extra_pollution_file_map: Dict[str, str] = {}

        self.beijing_lats = np.arange(39.0, 41.25, 0.25)
        self.beijing_lons = np.arange(115.0, 117.25, 0.25)

        self.meteo_columns = {
            "t2m": "2m_temperature",
            "d2m": "2m_dewpoint_temperature",
            "blh": "boundary_layer_height",
            "cvh": "high_vegetation_cover",
            "avg_tprate": "mean_total_precipitation_rate",
            "u10": "10m_u_component_of_wind",
            "v10": "10m_v_component_of_wind",
            "u100": "100m_u_component_of_wind",
            "v100": "100m_v_component_of_wind",
            "lsm": "land_sea_mask",
            "cvl": "low_vegetation_cover",
            "mn2t": "minimum_2m_temperature_since_previous_post_processing",
            "sp": "surface_pressure",
            "sd": "snow_depth",
            "str": "surface_net_thermal_radiation",
            "tisr": "toa_incident_solar_radiation",
            "tcwv": "total_column_water_vapour",
            "tp": "total_precipitation",
        }

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


    def load_meteo_data_parallel(self) -> None:
        """Load meteorological data in parallel (NC format)."""
        print("Starting parallel loading of meteorological data (NC format)...")
        start_time = time.time()

        if not os.path.exists(self.meteo_data_dir):
            print(f"Warning: Meteorological data directory does not exist: {self.meteo_data_dir}")
            return

        all_nc_files = glob.glob(
            os.path.join(self.meteo_data_dir, "**", "*.nc"), recursive=True
        )
        print(f"Found {len(all_nc_files)} NetCDF files")

        if not all_nc_files:
            print("No NC files found, please check directory path configuration")
            return

        print(f"Sample files: {[os.path.basename(f) for f in all_nc_files[:5]]}")

        max_workers = min(max(4, multiprocessing.cpu_count() - 1), 12)
        print(f"Using {max_workers} processes to process NC files")

        total_files = len(all_nc_files)
        processed_files = 0
        successful_files = 0

        self.meteorological_data = []
        aggregated_stats: Dict[Tuple[int, int], Dict[str, Any]] = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    process_single_nc_file,
                    filepath,
                    self.beijing_lats,
                    self.beijing_lons,
                    self.era5_vars,
                ): filepath
                for filepath in all_nc_files
            }
            for future in as_completed(future_to_file):
                processed_files += 1
                result = future.result()

                if result:
                    year = result.get("year")
                    month = result.get("month")
                    if pd.isna(year) or pd.isna(month) or year is None or month is None:
                        continue
                    year = int(year)
                    month = int(month)
                    key = (year, month)

                    if key not in aggregated_stats:
                        aggregated_stats[key] = {"year": year, "month": month}

                    result_days = result.get("days", 0)
                    if pd.isna(result_days):
                        result_days = 0
                    aggregated_stats[key]["days"] = int(
                        max(aggregated_stats[key].get("days", 0), int(result_days))
                    )

                    for key_name, value in result.items():
                        if key_name in {"source_file", "year", "month", "days"}:
                            continue
                        aggregated_stats[key][key_name] = value
                    successful_files += 1

                if processed_files % 50 == 0 or processed_files == total_files:
                    percentage = processed_files / total_files * 100
                    print(f"  Progress: {processed_files}/{total_files} ({percentage:.1f}%)")

        aggregated_list = [
            aggregated_stats[key] for key in sorted(aggregated_stats.keys())
        ]
        self.meteorological_data.extend(aggregated_list)

        elapsed = time.time() - start_time
        print(f"Meteorological data loading completed, elapsed time: {elapsed:.2f} seconds")
        print(f"Successfully processed {successful_files}/{total_files} NC files")
        print(f"Aggregated monthly data count: {len(self.meteorological_data)}")

    def _build_pollution_file_map(self) -> None:
        """构建污染数据文件名到路径的字典映射，用于O(1)查找."""
        print("Building pollution file path mapping...")
        self.pollution_file_map.clear()
        
        def pollution_file_filter(filename: str) -> bool:
            return filename.startswith("beijing_all_") and filename.endswith(".csv")
        
        search_pattern = os.path.join(self.pollution_data_dir, "**", "*.csv")
        for filepath in glob.glob(search_pattern, recursive=True):
            filename = os.path.basename(filepath)
            if pollution_file_filter(filename):
                self.pollution_file_map[filename] = filepath
        
        print(f"Built mapping for {len(self.pollution_file_map)} pollution data files")

    def load_pollution_data(self) -> None:
        """Load PM2.5, PM10, AQI pollution data (CSV format)."""
        print("Starting to load pollution data...")
        start_time = time.time()

        # 如果映射未构建，先构建
        if not self.pollution_file_map:
            self._build_pollution_file_map()

        print(f"Found {len(self.pollution_file_map)} pollution data files")

        for filename, filepath in self.pollution_file_map.items():
            try:
                cached_data = self.cache.get_cached_data(filepath)
                if cached_data:
                    self.pollution_data.append(cached_data)
                    continue

                df = pd.read_csv(filepath)
                if not df.empty and "type" in df.columns:
                    pm25_data = df[df["type"] == "PM2.5"].iloc[:, 3:].values
                    pm10_data = df[df["type"] == "PM10"].iloc[:, 3:].values
                    aqi_data = df[df["type"] == "AQI"].iloc[:, 3:].values

                    if len(pm25_data) > 0:
                        daily_pm25 = (
                            np.nanmean(pm25_data, axis=0)
                            if pm25_data.ndim > 1
                            else pm25_data
                        )
                        daily_pm10 = (
                            np.nanmean(pm10_data, axis=0)
                            if pm10_data.ndim > 1
                            else pm10_data
                        )
                        daily_aqi = (
                            np.nanmean(aqi_data, axis=0)
                            if aqi_data.ndim > 1
                            else aqi_data
                        )

                        pollution_stats = {
                            "pm25_mean": np.nanmean(daily_pm25),
                            "pm10_mean": np.nanmean(daily_pm10),
                            "aqi_mean": np.nanmean(daily_aqi),
                            "pm25_max": np.nanmax(daily_pm25),
                            "pm10_max": np.nanmax(daily_pm10),
                            "aqi_max": np.nanmax(daily_aqi),
                        }

                        self.cache.save_cached_data(filepath, pollution_stats)
                        self.pollution_data.append(pollution_stats)
                        print(f"Loaded pollution data: {os.path.basename(filepath)}")
            except Exception as exc:
                print(f"Error loading file {filepath}: {exc}")

        elapsed = time.time() - start_time
        print(f"Pollution data loading completed, elapsed time: {elapsed:.2f} seconds")
        print(f"Successfully loaded data from {len(self.pollution_data)} files")

    def _build_extra_pollution_file_map(self) -> None:
        """构建额外污染数据文件名到路径的字典映射，用于O(1)查找."""
        print("Building extra pollution file path mapping...")
        self.extra_pollution_file_map.clear()
        
        if not os.path.exists(self.extra_pollution_data_dir):
            print(
                f"Warning: Additional pollution data directory does not exist: {self.extra_pollution_data_dir}, skipped"
            )
            return

        def extra_pollution_file_filter(filename: str) -> bool:
            return filename.startswith("beijing_extra_") and filename.endswith(".csv")

        search_pattern = os.path.join(self.extra_pollution_data_dir, "**", "*.csv")
        for filepath in glob.glob(search_pattern, recursive=True):
            filename = os.path.basename(filepath)
            if extra_pollution_file_filter(filename):
                self.extra_pollution_file_map[filename] = filepath
        
        print(f"Built mapping for {len(self.extra_pollution_file_map)} additional pollution data files")

    def load_extra_pollution_data(self) -> None:
        """Load additional pollution data (SO2, CO, O3, NO2) (CSV format)."""
        print("Starting to load additional pollution data (SO2, CO, O3, NO2)...")
        start_time = time.time()

        # 如果映射未构建，先构建
        if not self.extra_pollution_file_map:
            self._build_extra_pollution_file_map()

        if not self.extra_pollution_file_map:
            print("No additional pollution data files found")
            return

        print(f"Found {len(self.extra_pollution_file_map)} additional pollution data files")

        for filename, filepath in self.extra_pollution_file_map.items():
            try:
                cached_data = self.cache.get_cached_data(filepath)
                if cached_data:
                    self.extra_pollution_data.append(cached_data)
                    continue

                df = pd.read_csv(filepath)
                if not df.empty and "type" in df.columns:
                    so2_data = df[df["type"] == "SO2"].iloc[:, 3:].values
                    co_data = df[df["type"] == "CO"].iloc[:, 3:].values
                    o3_data = df[df["type"] == "O3"].iloc[:, 3:].values
                    no2_data = df[df["type"] == "NO2"].iloc[:, 3:].values

                    if (
                        len(so2_data) > 0
                        or len(co_data) > 0
                        or len(o3_data) > 0
                        or len(no2_data) > 0
                    ):
                        daily_so2 = (
                            np.nanmean(so2_data, axis=0)
                            if so2_data.ndim > 1 and len(so2_data) > 0
                            else (
                                so2_data
                                if len(so2_data) > 0
                                else np.array([np.nan])
                            )
                        )
                        daily_co = (
                            np.nanmean(co_data, axis=0)
                            if co_data.ndim > 1 and len(co_data) > 0
                            else (
                                co_data if len(co_data) > 0 else np.array([np.nan])
                            )
                        )
                        daily_o3 = (
                            np.nanmean(o3_data, axis=0)
                            if o3_data.ndim > 1 and len(o3_data) > 0
                            else (
                                o3_data if len(o3_data) > 0 else np.array([np.nan])
                            )
                        )
                        daily_no2 = (
                            np.nanmean(no2_data, axis=0)
                            if no2_data.ndim > 1 and len(no2_data) > 0
                            else (
                                no2_data
                                if len(no2_data) > 0
                                else np.array([np.nan])
                            )
                        )

                        extra_stats = {
                            "so2_mean": np.nanmean(daily_so2),
                            "co_mean": np.nanmean(daily_co),
                            "o3_mean": np.nanmean(daily_o3),
                            "no2_mean": np.nanmean(daily_no2),
                            "so2_max": np.nanmax(daily_so2),
                            "co_max": np.nanmax(daily_co),
                            "o3_max": np.nanmax(daily_o3),
                            "no2_max": np.nanmax(daily_no2),
                        }

                        self.cache.save_cached_data(filepath, extra_stats)
                        self.extra_pollution_data.append(extra_stats)
                        print(f"Loaded additional pollution data: {os.path.basename(filepath)}")
            except Exception as exc:
                print(f"Error loading file {filepath}: {exc}")

        elapsed = time.time() - start_time
        print(f"Additional pollution data loading completed, elapsed time: {elapsed:.2f} seconds")
        print(f"Successfully loaded data from {len(self.extra_pollution_data)} files")

    def load_data(self) -> None:
        """Load all meteorological and pollution data."""
        print("Starting to load all data...")
        # 先构建文件映射，避免重复遍历
        self._build_pollution_file_map()
        self._build_extra_pollution_file_map()
        # 然后加载数据
        self.load_meteo_data_parallel()
        self.load_pollution_data()
        self.load_extra_pollution_data()
        print("Data loading completed!")

    def prepare_combined_data(self) -> pd.DataFrame:
        """Combine meteorological and pollution data."""
        print("Preparing to combine data...")

        if not self.meteorological_data or not self.pollution_data:
            print("Error: Insufficient data, cannot proceed with analysis")
            print(f"Meteorological data count: {len(self.meteorological_data)}")
            print(f"Pollution data count: {len(self.pollution_data)}")
            print(f"Additional pollution data count: {len(self.extra_pollution_data)}")
            return pd.DataFrame()

        meteo_df = pd.DataFrame(self.meteorological_data)
        pollution_df = pd.DataFrame(self.pollution_data)
        extra_pollution_df = (
            pd.DataFrame(self.extra_pollution_data)
            if self.extra_pollution_data
            else pd.DataFrame()
        )

        data_lengths = [len(meteo_df), len(pollution_df)]
        if not extra_pollution_df.empty:
            data_lengths.append(len(extra_pollution_df))
        min_len = min(data_lengths)

        meteo_df = meteo_df.head(min_len)
        pollution_df = pollution_df.head(min_len)
        if not extra_pollution_df.empty:
            extra_pollution_df = extra_pollution_df.head(min_len)

        frames = [meteo_df, pollution_df]
        if not extra_pollution_df.empty:
            frames.append(extra_pollution_df)

        combined_data = pd.concat(frames, axis=1)
        combined_data = combined_data.dropna(how="all")
        combined_data = combined_data.ffill().bfill()

        numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if combined_data[column].isna().any():
                mean_val = combined_data[column].mean()
                if not pd.isna(mean_val):
                    combined_data.loc[:, column] = combined_data[column].fillna(mean_val)

        print(f"Final data shape: {combined_data.shape}")
        print(f"Number of columns after merging: {len(combined_data.columns)}")
        print(f"Total null values after merging: {combined_data.isna().sum().sum()}")

        meteo_features = [
            col for col in combined_data.columns if any(x in col for x in self.meteo_columns.keys())
        ]
        pollution_features = [
            col
            for col in combined_data.columns
            if any(x in col.lower() for x in ["pm25", "pm10", "aqi", "so2", "co", "o3", "no2"])
        ]
        wind_features = [
            col for col in combined_data.columns if any(wind in col for wind in ["u10", "v10", "u100", "v100"])
        ]

        print(f"Meteorological parameter count: {len(meteo_features)}")
        if meteo_features:
            preview = meteo_features[:10]
            print(f"Meteorological parameter examples: {preview}{' ...' if len(meteo_features) > 10 else ''}")
        print(f"Wind component parameter count: {len(wind_features)}")
        print(f"Pollution parameter count: {len(pollution_features)}")

        return combined_data

    def perform_pca_analysis(
        self, data: pd.DataFrame, n_components: int = 2
    ) -> Tuple[Optional[np.ndarray], List[str], Optional[np.ndarray]]:
        """Perform PCA analysis."""
        print("Performing PCA analysis...")

        if data.empty:
            print("Error: No data available for PCA")
            return None, [], None

        numeric_columns = data.select_dtypes(include=[np.number]).columns
        feature_columns = [
            column for column in numeric_columns if column not in {"year", "month"}
        ]

        X = data[feature_columns].values
        X_scaled = self.scaler.fit_transform(X)

        self.pca = PCA(n_components=min(n_components, len(feature_columns)))
        X_pca = self.pca.fit_transform(X_scaled)

        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        print(f"Variance contribution ratio of the first {len(explained_variance_ratio)} principal components:")
        for idx, (var_ratio, cum_var_ratio) in enumerate(
            zip(explained_variance_ratio, cumulative_variance_ratio), start=1
        ):
            print(
                f"PC{idx}: {var_ratio:.4f} ({var_ratio * 100:.2f}%), "
                f"Cumulative: {cum_var_ratio:.4f} ({cum_var_ratio * 100:.2f}%)"
            )

        print("\nPrincipal component contribution analysis:")
        for idx in range(len(explained_variance_ratio)):
            loadings = self.pca.components_[idx]
            feature_loadings = sorted(
                zip(feature_columns, loadings), key=lambda item: abs(item[1]), reverse=True
            )
            print(f"\nTop 5 features contributing to PC{idx + 1}:")
            for feature, loading in feature_loadings[:5]:
                print(f"  {feature}: {loading:.4f}")

        return X_pca, feature_columns, explained_variance_ratio

    def analyze_correlations(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Analyze Pearson correlations between meteorological and pollution indicators."""
        print("Analyzing correlations...")

        if data.empty:
            print("Error: No data available for correlation analysis")
            return None

        numeric_columns = data.select_dtypes(include=[np.number]).columns
        feature_columns = [
            column for column in numeric_columns if column not in {"year", "month"}
        ]

        correlation_matrix = data[feature_columns].corr()

        pollution_features = [
            col
            for col in feature_columns
            if any(x in col.lower() for x in ["pm25", "pm10", "aqi", "so2", "co", "o3", "no2"])
        ]
        meteo_features = [
            col for col in feature_columns if any(x in col for x in self.meteo_columns.keys())
        ]

        if pollution_features and meteo_features:
            print(
                f"Detected {len(pollution_features)} pollution indicators and "
                f"{len(meteo_features)} meteorological factors"
            )
            for pollution_feature in pollution_features:
                correlations = correlation_matrix[pollution_feature][meteo_features].abs()
                correlations = correlations.dropna()
                top_correlations = correlations.nlargest(5)
                print(f"\nMeteorological factors most correlated with {pollution_feature}:")
                for meteo_feature, corr in top_correlations.items():
                    print(f"  {meteo_feature}: {corr:.3f}")

        return correlation_matrix

    def plot_correlation_heatmap(
        self,
        correlation_matrix: Optional[pd.DataFrame],
        save_path: str = "beijing_correlation_heatmap_nc.png",
    ) -> None:
        """Plot correlation heatmap."""
        if correlation_matrix is None:
            print("Error: No correlation data available for plotting")
            return

        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(20, 16))

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=False,
            cmap="RdYlBu_r",
            center=0,
            square=True,
            cbar_kws={"shrink": 0.8, "aspect": 50, "label": "Correlation Coefficient"},
            linewidths=0.2,
            linecolor="white",
            ax=ax,
        )

        plt.title(
            "Beijing Meteorological Factors and Pollution Indicators Correlation Heatmap (NC Data)",
            fontsize=22,
            fontweight="bold",
            pad=50,
            color="#2E3440",
        )
        plt.tight_layout()
        plt.savefig(
            save_path,
            dpi=1200,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)

        print(f"Correlation heatmap saved to: {save_path}")

    def plot_pca_results(
        self,
        X_pca: Optional[np.ndarray],
        feature_names: List[str],
        explained_variance_ratio: Optional[np.ndarray],
        save_path: str = "beijing_pca_results_nc.png",
    ) -> None:
        """Plot PCA results."""
        if X_pca is None or explained_variance_ratio is None:
            print("Error: No PCA results available for plotting")
            return

        plt.style.use("default")
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

        axes[0, 0].scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            alpha=0.7,
            c=colors[0],
            s=60,
            edgecolors="white",
            linewidth=0.5,
        )
        axes[0, 0].set_xlabel("First Principal Component (PC1)", fontsize=14, fontweight="bold")
        axes[0, 0].set_ylabel("Second Principal Component (PC2)", fontsize=14, fontweight="bold")
        axes[0, 0].set_title(
            "PCA Principal Component Scatter Plot (NC Data)", fontsize=16, fontweight="bold", pad=20
        )
        axes[0, 0].grid(True, alpha=0.3, linestyle="--")
        axes[0, 0].spines["top"].set_visible(False)
        axes[0, 0].spines["right"].set_visible(False)

        bars = axes[0, 1].bar(
            range(1, len(explained_variance_ratio) + 1),
            explained_variance_ratio,
            color=colors[1],
            alpha=0.8,
            edgecolor="white",
            linewidth=1,
        )
        axes[0, 1].set_xlabel("Principal Component", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel("Variance Contribution Ratio", fontsize=12, fontweight="bold")
        axes[0, 1].set_title(
            "Variance Contribution Ratio of Each Principal Component (NC Data)", fontsize=14, fontweight="bold", pad=15
        )
        axes[0, 1].grid(True, alpha=0.3, linestyle="--", axis="y")
        axes[0, 1].spines["top"].set_visible(False)
        axes[0, 1].spines["right"].set_visible(False)
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        axes[1, 0].plot(
            range(1, len(cumulative_variance_ratio) + 1),
            cumulative_variance_ratio,
            color=colors[2],
            linewidth=3,
            marker="o",
            markersize=10,
            markerfacecolor="white",
            markeredgecolor=colors[2],
            markeredgewidth=2,
        )
        axes[1, 0].set_xlabel("Number of Principal Components", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Cumulative Variance Contribution Ratio", fontsize=12, fontweight="bold")
        axes[1, 0].set_title(
            "Cumulative Variance Contribution Ratio (NC Data)", fontsize=14, fontweight="bold", pad=15
        )
        axes[1, 0].grid(True, alpha=0.3, linestyle="--")
        axes[1, 0].spines["top"].set_visible(False)
        axes[1, 0].spines["right"].set_visible(False)
        for idx, (x_coord, y_coord) in enumerate(
            zip(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)
        ):
            axes[1, 0].text(
                x_coord,
                y_coord + 0.02,
                f"{y_coord:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        if feature_names and self.pca is not None:
            loadings = self.pca.components_
            feature_importance = np.abs(loadings[0])
            sorted_idx = np.argsort(feature_importance)[::-1]
            top_n = min(15, len(sorted_idx))
            top_features = [feature_names[i] for i in sorted_idx[:top_n]]
            top_importance = feature_importance[sorted_idx[:top_n]]

            bars = axes[1, 1].barh(
                range(len(top_features)),
                top_importance,
                color=colors[3],
                alpha=0.8,
                edgecolor="white",
                linewidth=1,
            )
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features, fontsize=10)
            axes[1, 1].set_xlabel("Feature Importance (Absolute Value)", fontsize=12, fontweight="bold")
            axes[1, 1].set_title(
                "First Principal Component Feature Importance (NC Data)", fontsize=14, fontweight="bold", pad=15
            )
            axes[1, 1].grid(True, alpha=0.3, linestyle="--", axis="x")
            axes[1, 1].spines["top"].set_visible(False)
            axes[1, 1].spines["right"].set_visible(False)
            for idx, bar in enumerate(bars):
                width = bar.get_width()
                axes[1, 1].text(
                    width + 0.01,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{width:.3f}",
                    ha="left",
                    va="center",
                    fontweight="bold",
                    fontsize=9,
                )

        plt.tight_layout(pad=3.0)
        plt.savefig(
            save_path,
            dpi=1200,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)

        print(f"PCA results plot saved to: {save_path}")

    def generate_analysis_report(
        self,
        data: pd.DataFrame,
        correlation_matrix: Optional[pd.DataFrame],
        X_pca: Optional[np.ndarray],
        feature_names: List[str],
        explained_variance_ratio: Optional[np.ndarray],
    ) -> None:
        """Generate analysis report."""
        print("\n" + "=" * 80)
        print("Beijing Multi-Meteorological Factor Pollution Variation PCA Analysis Report (NC Data)")
        print("=" * 80)

        if data.empty:
            print("Error: No data available for report generation")
            return

        print("\n1. Data Overview:")
        print(f"   - Data shape: {data.shape}")
        print(f"   - Number of features: {len(feature_names)}")
        print(f"   - Number of samples: {len(data)}")

        meteo_features = [
            col for col in feature_names if any(x in col for x in self.meteo_columns.keys())
        ]
        pollution_features = [
            col
            for col in feature_names
            if any(x in col.lower() for x in ["pm25", "pm10", "aqi", "so2", "co", "o3", "no2"])
        ]

        print("\n2. Feature Classification:")
        print(f"   - Number of meteorological factors: {len(meteo_features)}")
        print(f"   - Number of pollution indicators: {len(pollution_features)}")

        print("\n3. Meteorological Factor Examples:")
        for feature in meteo_features[:10]:
            print(f"   - {feature}")
        if len(meteo_features) > 10:
            print(f"   ... and {len(meteo_features) - 10} more meteorological factors")

        print("\n4. Pollution Indicators:")
        for feature in pollution_features:
            print(f"   - {feature}")

        if correlation_matrix is not None:
            print("\n5. Pearson Correlation Analysis:")
            correlations = correlation_matrix.values
            correlations = correlations[~np.isnan(correlations)]
            correlations = correlations[np.abs(correlations) > 0]
            print(f"   - Total number of correlations: {len(correlations)}")
            print(f"   - Mean correlation coefficient: {np.mean(correlations):.4f}")
            print(f"   - Standard deviation: {np.std(correlations):.4f}")
            if pollution_features and meteo_features:
                for pollution_feature in pollution_features:
                    corr_values = (
                        correlation_matrix[pollution_feature][meteo_features]
                        .abs()
                        .dropna()
                    )
                    top_correlations = corr_values.nlargest(3)
                    print(f"   Meteorological factors most correlated with {pollution_feature}:")
                    for meteo_feature, corr in top_correlations.items():
                        print(f"     - {meteo_feature}: {corr:.3f}")

        if X_pca is not None and explained_variance_ratio is not None:
            print("\n6. PCA Analysis Results:")
            for idx, var_ratio in enumerate(explained_variance_ratio, start=1):
                print(
                    f"   - PC{idx} variance contribution ratio: {var_ratio:.4f} ({var_ratio * 100:.2f}%)"
                )
            print(
                f"   - Cumulative variance contribution ratio: {np.sum(explained_variance_ratio):.4f} "
                f"({np.sum(explained_variance_ratio) * 100:.2f}%)"
            )
            if len(explained_variance_ratio) >= 2 and self.pca is not None:
                print("\n7. Physical Meaning of Principal Components (First 3 Principal Components):")
                for idx in range(min(3, len(explained_variance_ratio))):
                    loadings = self.pca.components_[idx]
                    feature_loadings = sorted(
                        zip(feature_names, loadings),
                        key=lambda item: abs(item[1]),
                        reverse=True,
                    )
                    print(f"   Top 5 features contributing to PC{idx + 1}:")
                    for feature, loading in feature_loadings[:5]:
                        print(f"     - {feature}: {loading:.4f}")

        print("\n8. Key Findings:")
        print("   - Multiple meteorological factors jointly affect pollution level changes, PCA helps reveal underlying patterns")
        print("   - Temperature, humidity, wind speed, and boundary layer height significantly affect pollutant dispersion")
        print("   - Precipitation and radiation flux may be related to pollution reduction, requiring further interpretation with principal components")
        print("=" * 80)

    def run_analysis(self) -> None:
        """Run complete analysis pipeline."""
        print("Beijing Multi-Meteorological Factor Pollution Variation PCA Analysis (NC Data)")
        print("=" * 60)

        try:
            self.load_data()
            combined_data = self.prepare_combined_data()
            if combined_data.empty:
                print("Error: Unable to prepare data, please check source files")
                return

            X_pca, feature_names, explained_variance_ratio = self.perform_pca_analysis(
                combined_data
            )
            correlation_matrix = self.analyze_correlations(combined_data)
            self.plot_correlation_heatmap(correlation_matrix)
            self.plot_pca_results(
                X_pca, feature_names, explained_variance_ratio
            )
            self.generate_analysis_report(
                combined_data,
                correlation_matrix,
                X_pca,
                feature_names,
                explained_variance_ratio,
            )
        finally:
            print("\nClearing cache directory...")
            self.cache.clear_cache()
            print("Analysis pipeline completed.")


def main() -> None:
    """Main function, configure data paths and execute analysis."""
    meteo_data_dir = "/root/autodl-tmp/ERA5-Beijing-NC"
    pollution_data_dir = "/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)"
    extra_pollution_data_dir = "/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)"

    print("Data directory confirmation:")
    print(f"Meteorological data directory (NC): {meteo_data_dir}")
    print(f"Pollution data directory (CSV): {pollution_data_dir}")
    print(f"Additional pollution data directory (CSV): {extra_pollution_data_dir}")
    print("If paths are incorrect, please modify the configuration in main().")

    analyzer = BeijingPCAAnalyzerNC(
        meteo_data_dir, pollution_data_dir, extra_pollution_data_dir
    )
    analyzer.run_analysis()


if __name__ == "__main__":
    main()

