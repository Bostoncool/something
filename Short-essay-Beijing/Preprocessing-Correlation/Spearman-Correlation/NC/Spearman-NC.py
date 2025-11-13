import glob
import hashlib
import multiprocessing
import os
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
import torch
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset  # noqa: F401

warnings.filterwarnings("ignore")

mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 12
mpl.rcParams["figure.figsize"] = (12, 8)


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


class BeijingSpearmanAnalyzerNC:
    """北京多气象因子污染变化 Spearman 相关分析器（NC 数据）。"""

    def __init__(
        self,
        meteo_data_dir: str = ".",
        pollution_data_dir: str = ".",
        extra_pollution_data_dir: str = ".",
    ) -> None:
        self.meteo_data_dir = meteo_data_dir
        self.pollution_data_dir = pollution_data_dir
        self.extra_pollution_data_dir = extra_pollution_data_dir

        self.cache = DataCache()

        self.meteorological_data: List[Dict[str, Any]] = []
        self.pollution_data: List[Dict[str, Any]] = []
        self.extra_pollution_data: List[Dict[str, Any]] = []

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

    # ----------------------------------------------------------------------------------
    # 数据加载
    # ----------------------------------------------------------------------------------
    def process_single_nc_file(self, filepath: str) -> Optional[Dict[str, Any]]:
        """处理单个 ERA5 NetCDF 文件并提取统计特征。"""
        try:
            cached_data = self.cache.get_cached_data(filepath)
        except Exception:
            cached_data = None

        if cached_data:
            return cached_data

        try:
            with xr.open_dataset(
                filepath, engine="netcdf4", decode_times=True, cache=False
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
                    print(f"[WARN] {os.path.basename(filepath)} 不含目标变量，已跳过")
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

                dataset.load()

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
                    f"{len(available_vars)}, 天数: {int(stats.get('days', 0))}"
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
                    if (
                        pd.isna(year)
                        or pd.isna(month)
                        or year is None
                        or month is None
                    ):
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
                            else (co_data if len(co_data) > 0 else np.array([np.nan]))
                        )
                        daily_o3 = (
                            np.nanmean(o3_data, axis=0)
                            if o3_data.ndim > 1 and len(o3_data) > 0
                            else (o3_data if len(o3_data) > 0 else np.array([np.nan]))
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

    # ----------------------------------------------------------------------------------
    # 数据预处理与组合
    # ----------------------------------------------------------------------------------
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
        combined_data = combined_data.replace([np.inf, -np.inf], np.nan)

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
            col
            for col in combined_data.columns
            if any(wind in col for wind in ["u10", "v10", "u100", "v100"])
        ]

        print(f"气象参数数量: {len(meteo_features)}")
        if meteo_features:
            preview = meteo_features[:10]
            print(f"气象参数示例: {preview}{' ...' if len(meteo_features) > 10 else ''}")
        print(f"风分量参数数量: {len(wind_features)}")
        print(f"污染参数数量: {len(pollution_features)}")

        return combined_data

    # ----------------------------------------------------------------------------------
    # 相关性分析
    # ----------------------------------------------------------------------------------
    def calculate_spearman_correlation_torch(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """利用 PyTorch 计算 Spearman 相关矩阵。"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        feature_columns = [
            column for column in numeric_columns if column not in {"year", "month"}
        ]

        values = (
            data[feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=np.float32)
        )
        tensor = torch.tensor(values, dtype=torch.float32)
        ranks = torch.full_like(tensor, float("nan"))

        for idx in range(tensor.shape[1]):
            column = tensor[:, idx]
            valid_mask = torch.isfinite(column)
            if valid_mask.sum() == 0:
                continue
            valid_values = column[valid_mask]
            sorted_indices = torch.argsort(valid_values)
            rank_values = torch.zeros_like(valid_values)
            rank_values[sorted_indices] = torch.arange(
                1, len(valid_values) + 1, dtype=torch.float32
            )

            unique_values, inverse_indices, counts = torch.unique(
                valid_values, return_inverse=True, return_counts=True
            )
            for unique_idx, count in enumerate(counts):
                if count <= 1:
                    continue
                tie_mask = inverse_indices == unique_idx
                tie_rank = rank_values[tie_mask].mean()
                rank_values[tie_mask] = tie_rank

            column_ranks = torch.full_like(column, float("nan"))
            column_ranks[valid_mask] = rank_values
            ranks[:, idx] = column_ranks

        n_features = ranks.shape[1]
        correlation_matrix = torch.full(
            (n_features, n_features), float("nan"), dtype=torch.float32
        )

        for i in range(n_features):
            for j in range(i, n_features):
                valid_mask = torch.isfinite(ranks[:, i]) & torch.isfinite(ranks[:, j])
                n_valid = int(valid_mask.sum().item())
                if n_valid <= 1:
                    continue
                rank_i = ranks[valid_mask, i]
                rank_j = ranks[valid_mask, j]
                diff = rank_i - rank_j
                sum_d2 = torch.sum(diff * diff)
                denominator = n_valid * (n_valid**2 - 1)
                if denominator == 0:
                    continue
                spearman = 1 - (6.0 * sum_d2) / denominator
                correlation_matrix[i, j] = spearman
                correlation_matrix[j, i] = spearman

        diag_indices = torch.arange(n_features)
        correlation_matrix[diag_indices, diag_indices] = 1.0

        return pd.DataFrame(
            correlation_matrix.numpy(),
            columns=feature_columns,
            index=feature_columns,
        )

    def analyze_correlations(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """分析气象与污染指标的 Spearman 相关性。"""
        print("分析相关性...")

        if data.empty:
            print("错误: 无可用于相关性分析的数据")
            return None

        correlation_matrix = self.calculate_spearman_correlation_torch(data)

        pollution_features = [
            col
            for col in correlation_matrix.columns
            if any(x in col.lower() for x in ["pm25", "pm10", "aqi", "so2", "co", "o3", "no2"])
        ]
        meteo_features = [
            col
            for col in correlation_matrix.columns
            if any(x in col for x in self.meteo_columns.keys())
        ]

        if pollution_features and meteo_features:
            print(
                f"检测到 {len(pollution_features)} 个污染指标与 "
                f"{len(meteo_features)} 个气象因子"
            )
            for pollution_feature in pollution_features:
                correlations = (
                    correlation_matrix[pollution_feature][meteo_features]
                    .abs()
                    .dropna()
                )
                top_correlations = correlations.nlargest(5)
                print(f"\n与 {pollution_feature} 最相关的气象因子:")
                for meteo_feature, corr in top_correlations.items():
                    print(f"  {meteo_feature}: {corr:.3f}")

        return correlation_matrix

    # ----------------------------------------------------------------------------------
    # 可视化
    # ----------------------------------------------------------------------------------
    def plot_correlation_heatmap(
        self,
        correlation_matrix: Optional[pd.DataFrame],
        save_path: str = "beijing_spearman_correlation_heatmap_nc.png",
    ) -> None:
        """绘制 Spearman 相关性热力图。"""
        if correlation_matrix is None or correlation_matrix.empty:
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
            cbar_kws={"shrink": 0.8, "aspect": 50, "label": "Spearman 相关系数"},
            linewidths=0.2,
            linecolor="white",
            ax=ax,
        )

        plt.title(
            "北京气象因子与污染指标 Spearman 相关性热力图 (NC 数据)",
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

    def gradient_image(
        self,
        ax: plt.Axes,
        extent: Tuple[float, float, float, float],
        cmap: LinearSegmentedColormap,
        cmap_range: Tuple[float, float] = (0, 0.5),
        direction: int = 0,
    ) -> None:
        """创建渐变背景。"""
        phi = direction * np.pi / 2
        vec = np.array([np.cos(phi), np.sin(phi)])
        grid = np.array([[vec @ [0, 0], vec @ [1, 0]], [vec @ [0, 1], vec @ [1, 1]]])
        a, b = cmap_range
        grid = a + (b - a) / grid.max() * grid
        ax.imshow(
            grid,
            extent=extent,
            interpolation="bicubic",
            vmin=0,
            vmax=1,
            aspect="auto",
            cmap=cmap,
        )

    def plot_top_correlations(
        self,
        correlation_matrix: Optional[pd.DataFrame],
        save_path: str = "beijing_spearman_top_correlations_nc.png",
    ) -> None:
        """绘制最强相关性的对比图。"""
        if correlation_matrix is None or correlation_matrix.empty:
            print("错误: 无相关性数据可用于绘图")
            return

        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        correlations: List[Dict[str, Any]] = []
        for i in range(len(upper_triangle.columns)):
            for j in range(i + 1, len(upper_triangle.columns)):
                value = upper_triangle.iloc[i, j]
                if not pd.isna(value):
                    correlations.append(
                        {
                            "feature1": upper_triangle.columns[i],
                            "feature2": upper_triangle.columns[j],
                            "correlation": value,
                        }
                    )

        if not correlations:
            print("警告: 未找到有效的相关性结果")
            return

        correlations.sort(key=lambda item: abs(item["correlation"]), reverse=True)
        top_correlations = correlations[:20]

        fig, ax = plt.subplots(figsize=(20, 14))

        features = []
        for corr_item in top_correlations:
            feat1 = (
                str(corr_item["feature1"])
                .replace("_mean", "")
                .replace("_std", "")
                .replace("_min", "")
                .replace("_max", "")
            )
            feat2 = (
                str(corr_item["feature2"])
                .replace("_mean", "")
                .replace("_std", "")
                .replace("_min", "")
                .replace("_max", "")
            )
            features.append(f"{feat1}\nvs\n{feat2}")

        values = [corr_item["correlation"] for corr_item in top_correlations]
        y_pos = np.arange(len(features))

        for y, value in zip(y_pos, values):
            left = 0 if value >= 0 else value
            right = value if value >= 0 else 0
            colors = (
                [(114 / 255, 188 / 255, 213 / 255), (1, 1, 1)]
                if value >= 0
                else [(255 / 255, 99 / 255, 71 / 255), (1, 1, 1)]
            )
            cmap = LinearSegmentedColormap.from_list("corr_cmap", colors, N=256)
            bar_height = 0.6
            self.gradient_image(
                ax,
                extent=(left, right, y - bar_height / 2, y + bar_height / 2),
                cmap=cmap,
                cmap_range=(0, 0.8),
            )

        for y, value in zip(y_pos, values):
            if value >= 0:
                ax.text(
                    value + 0.01,
                    y,
                    f"{value:.3f}",
                    ha="left",
                    va="center",
                    fontweight="bold",
                    fontsize=11,
                )
            else:
                ax.text(
                    value - 0.01,
                    y,
                    f"{value:.3f}",
                    ha="right",
                    va="center",
                    fontweight="bold",
                    fontsize=11,
                    color="red",
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel("Spearman 相关系数", fontsize=14, fontweight="bold")
        ax.set_title(
            "北京气象因子与污染指标最强 Spearman 相关性 Top 20 (NC 数据)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        ax.grid(True, alpha=0.3, axis="x")

        max_abs_value = max(abs(v) for v in values)
        ax.set_xlim(-max_abs_value * 1.1, max_abs_value * 1.1)

        plt.tight_layout()
        plt.savefig(
            save_path,
            dpi=1200,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)

        print(f"Top 20 相关性图已保存至: {save_path}")

    # ----------------------------------------------------------------------------------
    # 报告输出
    # ----------------------------------------------------------------------------------
    def generate_analysis_report(
        self,
        data: pd.DataFrame,
        correlation_matrix: Optional[pd.DataFrame],
    ) -> None:
        """生成分析报告。"""
        print("\n" + "=" * 80)
        print("北京多气象因子污染变化 Spearman 相关分析报告 (NC 数据)")
        print("=" * 80)

        if data.empty:
            print("错误: 无数据用于生成报告")
            return

        numeric_columns = data.select_dtypes(include=[np.number]).columns
        feature_columns = [
            column for column in numeric_columns if column not in {"year", "month"}
        ]

        print("\n1. 数据概览:")
        print(f"   - 数据形状: {data.shape}")
        print(f"   - 特征数量: {len(feature_columns)}")
        print(f"   - 样本数量: {len(data)}")

        meteo_features = [
            col for col in feature_columns if any(x in col for x in self.meteo_columns.keys())
        ]
        pollution_features = [
            col
            for col in feature_columns
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

        if correlation_matrix is not None and not correlation_matrix.empty:
            print("\n5. Spearman 相关性分析:")
            correlations = correlation_matrix.values
            correlations = correlations[~np.isnan(correlations)]
            correlations = correlations[np.abs(correlations) > 0]
            if correlations.size > 0:
                print(f"   - 相关性总数: {len(correlations)}")
                print(f"   - 平均相关系数: {np.mean(correlations):.4f}")
                print(f"   - 标准差: {np.std(correlations):.4f}")
            else:
                print("   - 暂无显著相关性")
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

        print("\n6. 主要发现:")
        print("   - 多个气象因子与污染指标存在非线性相关，需要结合季节性进一步解释")
        print("   - 温度与湿度对污染累积具有显著作用，风场与边界层高度影响污染扩散")
        print("   - 降水与辐射通量可能与污染削减相关，可结合时序分析进一步验证")
        print("=" * 80)

    # ----------------------------------------------------------------------------------
    # 主流程
    # ----------------------------------------------------------------------------------
    def run_analysis(self) -> None:
        """运行完整分析流程。"""
        print("北京多气象因子污染变化 Spearman 相关分析 (NC 数据)")
        print("=" * 60)

        try:
            self.load_data()
            combined_data = self.prepare_combined_data()
            if combined_data.empty:
                print("错误: 无法准备数据，请检查源文件")
                return

            correlation_matrix = self.analyze_correlations(combined_data)
            self.plot_correlation_heatmap(correlation_matrix)
            self.plot_top_correlations(correlation_matrix)
            self.generate_analysis_report(combined_data, correlation_matrix)
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

    analyzer = BeijingSpearmanAnalyzerNC(
        meteo_data_dir, pollution_data_dir, extra_pollution_data_dir
    )
    analyzer.run_analysis()


if __name__ == "__main__":
    main()

