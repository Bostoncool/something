import os
import glob
import hashlib
import multiprocessing
import pickle
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from netCDF4 import Dataset  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Matplotlib configuration
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 12
mpl.rcParams["figure.figsize"] = (10, 6)


class DataCache:
    """缓存处理结果，避免重复计算。"""

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
        print(f"缓存目录已清空: {self.cache_dir}")


class BeijingPCAAnalyzerNC:
    """北京多气象因子污染变化 PCA 分析器（NC 数据）。"""

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

    def process_single_nc_file(self, filepath: str) -> Optional[Dict[str, Any]]:
        """处理单个 ERA5 NetCDF 文件并提取统计特征。"""
        cached_data = None
        try:
            cached_data = self.cache.get_cached_data(filepath)
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
                    var for var in self.era5_vars if var in dataset.data_vars
                ]
                if not available_vars:
                    print(
                        f"[WARN] {os.path.basename(filepath)} 不含目标变量，已跳过"
                    )
                    return None

                if "latitude" in dataset.coords and "longitude" in dataset.coords:
                    lat_values = dataset["latitude"]
                    if lat_values[0] > lat_values[-1]:
                        lat_slice = slice(
                            self.beijing_lats.max(), self.beijing_lats.min()
                        )
                    else:
                        lat_slice = slice(
                            self.beijing_lats.min(), self.beijing_lats.max()
                        )
                    dataset = dataset.sel(
                        latitude=lat_slice,
                        longitude=slice(
                            self.beijing_lons.min(), self.beijing_lons.max()
                        ),
                    )
                    if "latitude" in dataset.dims and "longitude" in dataset.dims:
                        dataset = dataset.mean(
                            dim=["latitude", "longitude"], skipna=True
                        )

                if "time" not in dataset.coords:
                    print(f"[WARN] {os.path.basename(filepath)} 缺少时间坐标，已跳过")
                    return None

                dataset = dataset.sortby("time")
                dataset = dataset.resample(time="1D").mean(keep_attrs=False)
                dataset = dataset.dropna("time", how="all")

                if dataset.sizes.get("time", 0) == 0:
                    print(
                        f"[WARN] {os.path.basename(filepath)} 重采样后无有效时间，已跳过"
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
                            f"提取 {os.path.basename(filepath)} 中的变量 {var} 时出错: {err}"
                        )
                        stats[f"{var}_mean"] = np.nan
                        stats[f"{var}_std"] = np.nan
                        stats[f"{var}_min"] = np.nan
                        stats[f"{var}_max"] = np.nan

                self.cache.save_cached_data(filepath, stats)
                print(
                    f"  [+] {os.path.basename(filepath)} -> 变量: "
                    f"{len(available_vars)}, 天数: {int(day_count)}"
                )
                return stats
        except Exception as exc:
            print(
                f"[ERROR] 处理 {os.path.basename(filepath)} 失败: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

    def load_meteo_data_parallel(self) -> None:
        """并行加载气象数据 (NC)。"""
        print("开始并行加载气象数据（NC 格式）...")
        start_time = time.time()

        if not os.path.exists(self.meteo_data_dir):
            print(f"警告: 气象数据目录不存在: {self.meteo_data_dir}")
            return

        all_nc_files = glob.glob(
            os.path.join(self.meteo_data_dir, "**", "*.nc"), recursive=True
        )
        print(f"找到 {len(all_nc_files)} 个 NetCDF 文件")

        if not all_nc_files:
            print("未找到任何 NC 文件，请检查目录路径配置")
            return

        print(f"示例文件: {[os.path.basename(f) for f in all_nc_files[:5]]}")

        max_workers = min(max(4, multiprocessing.cpu_count() - 1), 12)
        print(f"使用 {max_workers} 个线程处理 NC 文件")

        total_files = len(all_nc_files)
        processed_files = 0
        successful_files = 0

        self.meteorological_data = []
        aggregated_stats: Dict[Tuple[int, int], Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single_nc_file, filepath): filepath
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
                    print(f"  进度: {processed_files}/{total_files} ({percentage:.1f}%)")

        aggregated_list = [
            aggregated_stats[key] for key in sorted(aggregated_stats.keys())
        ]
        self.meteorological_data.extend(aggregated_list)

        elapsed = time.time() - start_time
        print(f"气象数据加载完成，耗时: {elapsed:.2f} 秒")
        print(f"成功处理 {successful_files}/{total_files} 个 NC 文件")
        print(f"聚合后月份数据量: {len(self.meteorological_data)}")

    def load_pollution_data(self) -> None:
        """加载 PM2.5、PM10、AQI 污染数据 (CSV)。"""
        print("开始加载污染数据...")
        start_time = time.time()

        def pollution_file_filter(filename: str) -> bool:
            return filename.startswith("beijing_all_") and filename.endswith(".csv")

        all_pollution_files: List[str] = []
        search_pattern = os.path.join(self.pollution_data_dir, "**", "*.csv")
        for filepath in glob.glob(search_pattern, recursive=True):
            if pollution_file_filter(os.path.basename(filepath)):
                all_pollution_files.append(filepath)

        print(f"找到 {len(all_pollution_files)} 个污染数据文件")

        for filepath in all_pollution_files:
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
                        print(f"已加载污染数据: {os.path.basename(filepath)}")
            except Exception as exc:
                print(f"加载文件 {filepath} 出错: {exc}")

        elapsed = time.time() - start_time
        print(f"污染数据加载完成，耗时: {elapsed:.2f} 秒")
        print(f"成功加载 {len(self.pollution_data)} 个文件的数据")

    def load_extra_pollution_data(self) -> None:
        """加载 SO2、CO、O3、NO2 等额外污染数据 (CSV)。"""
        print("开始加载额外污染数据 (SO2, CO, O3, NO2)...")
        start_time = time.time()

        if not os.path.exists(self.extra_pollution_data_dir):
            print(
                f"警告: 额外污染数据目录不存在: {self.extra_pollution_data_dir}，已跳过"
            )
            return

        def extra_pollution_file_filter(filename: str) -> bool:
            return filename.startswith("beijing_extra_") and filename.endswith(".csv")

        all_extra_files: List[str] = []
        search_pattern = os.path.join(self.extra_pollution_data_dir, "**", "*.csv")
        for filepath in glob.glob(search_pattern, recursive=True):
            if extra_pollution_file_filter(os.path.basename(filepath)):
                all_extra_files.append(filepath)

        print(f"找到 {len(all_extra_files)} 个额外污染数据文件")
        if not all_extra_files:
            print("未找到额外污染数据文件")
            return

        for filepath in all_extra_files:
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
                        print(f"已加载额外污染数据: {os.path.basename(filepath)}")
            except Exception as exc:
                print(f"加载文件 {filepath} 出错: {exc}")

        elapsed = time.time() - start_time
        print(f"额外污染数据加载完成，耗时: {elapsed:.2f} 秒")
        print(f"成功加载 {len(self.extra_pollution_data)} 个文件的数据")

    def load_data(self) -> None:
        """整体加载气象与污染数据。"""
        print("开始加载全部数据...")
        self.load_meteo_data_parallel()
        self.load_pollution_data()
        self.load_extra_pollution_data()
        print("数据加载完成!")

    def prepare_combined_data(self) -> pd.DataFrame:
        """合并气象与污染数据。"""
        print("准备合并数据...")

        if not self.meteorological_data or not self.pollution_data:
            print("错误: 数据不足，无法开展分析")
            print(f"气象数据数量: {len(self.meteorological_data)}")
            print(f"污染数据数量: {len(self.pollution_data)}")
            print(f"额外污染数据数量: {len(self.extra_pollution_data)}")
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

        print(f"最终数据形状: {combined_data.shape}")
        print(f"合并后的列名数量: {len(combined_data.columns)}")
        print(f"合并后空值总数: {combined_data.isna().sum().sum()}")

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

        print(f"气象参数数量: {len(meteo_features)}")
        if meteo_features:
            preview = meteo_features[:10]
            print(f"气象参数示例: {preview}{' ...' if len(meteo_features) > 10 else ''}")
        print(f"风分量参数数量: {len(wind_features)}")
        print(f"污染参数数量: {len(pollution_features)}")

        return combined_data

    def perform_pca_analysis(
        self, data: pd.DataFrame, n_components: int = 2
    ) -> Tuple[Optional[np.ndarray], List[str], Optional[np.ndarray]]:
        """执行 PCA 分析。"""
        print("执行 PCA 分析...")

        if data.empty:
            print("错误: 无可用于 PCA 的数据")
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

        print(f"前 {len(explained_variance_ratio)} 个主成分的方差贡献率:")
        for idx, (var_ratio, cum_var_ratio) in enumerate(
            zip(explained_variance_ratio, cumulative_variance_ratio), start=1
        ):
            print(
                f"PC{idx}: {var_ratio:.4f} ({var_ratio * 100:.2f}%), "
                f"累计: {cum_var_ratio:.4f} ({cum_var_ratio * 100:.2f}%)"
            )

        print("\n主成分贡献分析:")
        for idx in range(len(explained_variance_ratio)):
            loadings = self.pca.components_[idx]
            feature_loadings = sorted(
                zip(feature_columns, loadings), key=lambda item: abs(item[1]), reverse=True
            )
            print(f"\nPC{idx + 1} 贡献度前 5 的特征:")
            for feature, loading in feature_loadings[:5]:
                print(f"  {feature}: {loading:.4f}")

        return X_pca, feature_columns, explained_variance_ratio

    def analyze_correlations(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """分析气象与污染指标的 Pearson 相关性。"""
        print("分析相关性...")

        if data.empty:
            print("错误: 无可用于相关性分析的数据")
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
                f"检测到 {len(pollution_features)} 个污染指标与 "
                f"{len(meteo_features)} 个气象因子"
            )
            for pollution_feature in pollution_features:
                correlations = correlation_matrix[pollution_feature][meteo_features].abs()
                correlations = correlations.dropna()
                top_correlations = correlations.nlargest(5)
                print(f"\n与 {pollution_feature} 最相关的气象因子:")
                for meteo_feature, corr in top_correlations.items():
                    print(f"  {meteo_feature}: {corr:.3f}")

        return correlation_matrix

    def plot_correlation_heatmap(
        self,
        correlation_matrix: Optional[pd.DataFrame],
        save_path: str = "beijing_correlation_heatmap_nc.png",
    ) -> None:
        """绘制相关性热力图。"""
        if correlation_matrix is None:
            print("错误: 无相关性数据可用于绘图")
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
            cbar_kws={"shrink": 0.8, "aspect": 50, "label": "相关系数"},
            linewidths=0.2,
            linecolor="white",
            ax=ax,
        )

        plt.title(
            "北京气象因子与污染指标相关性热力图 (NC 数据)",
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

        print(f"相关性热力图已保存至: {save_path}")

    def plot_pca_results(
        self,
        X_pca: Optional[np.ndarray],
        feature_names: List[str],
        explained_variance_ratio: Optional[np.ndarray],
        save_path: str = "beijing_pca_results_nc.png",
    ) -> None:
        """绘制 PCA 结果图。"""
        if X_pca is None or explained_variance_ratio is None:
            print("错误: 无 PCA 结果可用于绘制")
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
        axes[0, 0].set_xlabel("第一主成分 (PC1)", fontsize=14, fontweight="bold")
        axes[0, 0].set_ylabel("第二主成分 (PC2)", fontsize=14, fontweight="bold")
        axes[0, 0].set_title(
            "PCA 主成分散点图 (NC 数据)", fontsize=16, fontweight="bold", pad=20
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
        axes[0, 1].set_xlabel("主成分", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel("方差贡献率", fontsize=12, fontweight="bold")
        axes[0, 1].set_title(
            "各主成分方差贡献率 (NC 数据)", fontsize=14, fontweight="bold", pad=15
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
        axes[1, 0].set_xlabel("主成分数量", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("累计方差贡献率", fontsize=12, fontweight="bold")
        axes[1, 0].set_title(
            "累计方差贡献率 (NC 数据)", fontsize=14, fontweight="bold", pad=15
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
            axes[1, 1].set_xlabel("特征重要性 (绝对值)", fontsize=12, fontweight="bold")
            axes[1, 1].set_title(
                "第一主成分特征重要性 (NC 数据)", fontsize=14, fontweight="bold", pad=15
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

        print(f"PCA 结果图已保存至: {save_path}")

    def generate_analysis_report(
        self,
        data: pd.DataFrame,
        correlation_matrix: Optional[pd.DataFrame],
        X_pca: Optional[np.ndarray],
        feature_names: List[str],
        explained_variance_ratio: Optional[np.ndarray],
    ) -> None:
        """生成分析报告。"""
        print("\n" + "=" * 80)
        print("北京多气象因子污染变化 PCA 分析报告 (NC 数据)")
        print("=" * 80)

        if data.empty:
            print("错误: 无数据用于生成报告")
            return

        print("\n1. 数据概览:")
        print(f"   - 数据形状: {data.shape}")
        print(f"   - 特征数量: {len(feature_names)}")
        print(f"   - 样本数量: {len(data)}")

        meteo_features = [
            col for col in feature_names if any(x in col for x in self.meteo_columns.keys())
        ]
        pollution_features = [
            col
            for col in feature_names
            if any(x in col.lower() for x in ["pm25", "pm10", "aqi", "so2", "co", "o3", "no2"])
        ]

        print("\n2. 特征分类:")
        print(f"   - 气象因子数量: {len(meteo_features)}")
        print(f"   - 污染指标数量: {len(pollution_features)}")

        print("\n3. 气象因子示例:")
        for feature in meteo_features[:10]:
            print(f"   - {feature}")
        if len(meteo_features) > 10:
            print(f"   ... 以及其他 {len(meteo_features) - 10} 个气象因子")

        print("\n4. 污染指标:")
        for feature in pollution_features:
            print(f"   - {feature}")

        if correlation_matrix is not None:
            print("\n5. Pearson 相关性分析:")
            correlations = correlation_matrix.values
            correlations = correlations[~np.isnan(correlations)]
            correlations = correlations[np.abs(correlations) > 0]
            print(f"   - 相关性总数: {len(correlations)}")
            print(f"   - 平均相关系数: {np.mean(correlations):.4f}")
            print(f"   - 标准差: {np.std(correlations):.4f}")
            if pollution_features and meteo_features:
                for pollution_feature in pollution_features:
                    corr_values = (
                        correlation_matrix[pollution_feature][meteo_features]
                        .abs()
                        .dropna()
                    )
                    top_correlations = corr_values.nlargest(3)
                    print(f"   {pollution_feature} 相关性最强的气象因子:")
                    for meteo_feature, corr in top_correlations.items():
                        print(f"     - {meteo_feature}: {corr:.3f}")

        if X_pca is not None and explained_variance_ratio is not None:
            print("\n6. PCA 分析结果:")
            for idx, var_ratio in enumerate(explained_variance_ratio, start=1):
                print(
                    f"   - PC{idx} 方差贡献率: {var_ratio:.4f} ({var_ratio * 100:.2f}%)"
                )
            print(
                f"   - 累计方差贡献率: {np.sum(explained_variance_ratio):.4f} "
                f"({np.sum(explained_variance_ratio) * 100:.2f}%)"
            )
            if len(explained_variance_ratio) >= 2 and self.pca is not None:
                print("\n7. 主成分物理含义 (前 3 个主成分):")
                for idx in range(min(3, len(explained_variance_ratio))):
                    loadings = self.pca.components_[idx]
                    feature_loadings = sorted(
                        zip(feature_names, loadings),
                        key=lambda item: abs(item[1]),
                        reverse=True,
                    )
                    print(f"   PC{idx + 1} 贡献度前 5 的特征:")
                    for feature, loading in feature_loadings[:5]:
                        print(f"     - {feature}: {loading:.4f}")

        print("\n8. 主要发现:")
        print("   - 多个气象因子共同作用于污染水平变化，PCA 有助于揭示潜在模式")
        print("   - 温度、湿度、风速与边界层高度对污染物扩散影响显著")
        print("   - 降水与辐射通量可能与污染削减相关，需要结合主成分进一步解释")
        print("=" * 80)

    def run_analysis(self) -> None:
        """运行完整分析流程。"""
        print("北京多气象因子污染变化 PCA 分析 (NC 数据)")
        print("=" * 60)

        try:
            self.load_data()
            combined_data = self.prepare_combined_data()
            if combined_data.empty:
                print("错误: 无法准备数据，请检查源文件")
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
            print("\n清理缓存目录...")
            self.cache.clear_cache()
            print("分析流程结束。")


def main() -> None:
    """主函数，配置数据路径并执行分析。"""
    meteo_data_dir = r"E:\DATA Science\ERA5-Beijing-NC"
    pollution_data_dir = r"E:\DATA Science\Benchmark\all(AQI+PM2.5+PM10)"
    extra_pollution_data_dir = r"E:\DATA Science\Benchmark\extra(SO2+NO2+CO+O3)"

    print("数据目录确认:")
    print(f"气象数据目录 (NC): {meteo_data_dir}")
    print(f"污染数据目录 (CSV): {pollution_data_dir}")
    print(f"额外污染数据目录 (CSV): {extra_pollution_data_dir}")
    print("如路径有误，请修改 main() 中的配置。")

    analyzer = BeijingPCAAnalyzerNC(
        meteo_data_dir, pollution_data_dir, extra_pollution_data_dir
    )
    analyzer.run_analysis()


if __name__ == "__main__":
    main()

