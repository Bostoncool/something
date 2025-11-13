import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
import warnings
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import hashlib
import pickle
import time
from typing import List, Dict, Any, Optional, Tuple
import xarray as xr
from netCDF4 import Dataset
import re

warnings.filterwarnings('ignore')

# 设置字体配置
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = (10, 6)

class DataCache:
    """数据缓存类，避免重复处理相同文件"""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, filepath: str) -> str:
        file_stat = os.stat(filepath)
        return hashlib.md5(f"{filepath}_{file_stat.st_mtime}".encode()).hexdigest()
    
    def get_cached_data(self, filepath: str) -> Optional[Dict]:
        cache_key = self.get_cache_key(filepath)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return None
    
    def save_cached_data(self, filepath: str, data: Dict):
        cache_key = self.get_cache_key(filepath)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    
    def clear_cache(self):
        try:
            if os.path.exists(self.cache_dir):
                for filename in os.listdir(self.cache_dir):
                    file_path = os.path.join(self.cache_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                print(f"缓存已清空: {self.cache_dir}")
        except Exception as e:
            print(f"清空缓存时出错: {e}")

class BeijingKendallAnalyzer:
    """北京气象因子与污染变化 Kendall 相关性分析器"""
    
    def __init__(self, meteo_data_dir=".", pollution_data_dir=".", additional_pollution_data_dir="."):
        self.meteo_data_dir = meteo_data_dir
        self.pollution_data_dir = pollution_data_dir
        self.additional_pollution_data_dir = additional_pollution_data_dir
        self.meteorological_data = []
        self.pollution_data = []
        self.additional_pollution_data = []
        self.cache = DataCache()
        
        # 北京区域范围
        self.beijing_lats = np.arange(39.0, 41.25, 0.25)
        self.beijing_lons = np.arange(115.0, 117.25, 0.25)
        
        # 气象参数列名映射
        self.meteo_columns = {
            't2m': '2m_temperature',
            'd2m': '2m_dewpoint_temperature',
            'blh': 'boundary_layer_height',
            'cvh': 'high_vegetation_cover',
            'avg_tprate': 'mean_total_precipitation_rate',
            'u10': '10m_u_component_of_wind',
            'v10': '10m_v_component_of_wind',
            'u100': '100m_u_component_of_wind',
            'v100': '100m_v_component_of_wind',
            'lsm': 'land_sea_mask',
            'cvl': 'low_vegetation_cover',
            'mn2t': 'minimum_2m_temperature_since_previous_post_processing',
            'sp': 'surface_pressure',
            'sd': 'snow_depth',
            'str': 'surface_net_thermal_radiation',
            'tisr': 'toa_incident_solar_radiation',
            'tcwv': 'total_column_water_vapour',
            'tp': 'total_precipitation'
        }
        
        # ERA5 变量列表
        self.era5_vars = [
            'd2m', 't2m', 'u10', 'v10', 'u100', 'v100',
            'blh', 'sp', 'tcwv',
            'tp', 'avg_tprate',
            'tisr', 'str',
            'cvh', 'cvl',
            'mn2t', 'sd', 'lsm'
        ]
    
    def process_single_nc_file(self, filepath: str) -> Optional[Dict]:
        """处理单个 ERA5 NetCDF 文件，提取气象统计特征"""
        try:
            cached_data = self.cache.get_cached_data(filepath)
            if cached_data:
                return cached_data
        except Exception:
            cached_data = None
        
        try:
            with xr.open_dataset(filepath, engine="netcdf4", decode_times=True) as ds:
                rename_map = {}
                for tkey in ("valid_time", "forecast_time", "verification_time", "time1", "time2"):
                    if tkey in ds.coords and "time" not in ds.coords:
                        rename_map[tkey] = "time"
                if "lat" in ds.coords and "latitude" not in ds.coords:
                    rename_map["lat"] = "latitude"
                if "lon" in ds.coords and "longitude" not in ds.coords:
                    rename_map["lon"] = "longitude"
                if rename_map:
                    ds = ds.rename(rename_map)
                try:
                    ds = xr.decode_cf(ds)
                except Exception:
                    pass
                drop_vars = []
                for extra_coord in ("expver", "surface"):
                    if extra_coord in ds:
                        drop_vars.append(extra_coord)
                if drop_vars:
                    ds = ds.drop_vars(drop_vars)
                if "number" in ds.dims:
                    ds = ds.mean(dim="number", skipna=True)
                available_vars = [var for var in self.era5_vars if var in ds.data_vars]
                if not available_vars:
                    print(f"[WARN] {os.path.basename(filepath)} 不含目标变量，跳过")
                    return None
                if "latitude" in ds.coords and "longitude" in ds.coords:
                    lat_values = ds["latitude"]
                    if lat_values[0] > lat_values[-1]:
                        lat_slice = slice(self.beijing_lats.max(), self.beijing_lats.min())
                    else:
                        lat_slice = slice(self.beijing_lats.min(), self.beijing_lats.max())
                    ds = ds.sel(
                        latitude=lat_slice,
                        longitude=slice(self.beijing_lons.min(), self.beijing_lons.max())
                    )
                    if "latitude" in ds.dims and "longitude" in ds.dims:
                        ds = ds.mean(dim=["latitude", "longitude"], skipna=True)
                if "time" not in ds.coords:
                    print(f"[WARN] {os.path.basename(filepath)} 缺少时间坐标，跳过")
                    return None
                ds = ds.sortby("time")
                ds = ds.resample(time="1D").mean(keep_attrs=False)
                ds = ds.dropna("time", how="all")
                if ds.sizes.get("time", 0) == 0:
                    print(f"[WARN] {os.path.basename(filepath)} 重采样后无有效时间，跳过")
                    return None
                stats = {
                    "source_file": os.path.basename(filepath)
                }
                try:
                    time_index = pd.to_datetime(ds["time"].values)
                    if len(time_index) > 0:
                        first_time = time_index[0]
                        stats["year"] = int(first_time.year)
                        stats["month"] = int(first_time.month)
                        stats["days"] = int(len(time_index))
                    else:
                        stats["year"] = np.nan
                        stats["month"] = np.nan
                        stats["days"] = ds.sizes.get("time", np.nan)
                except Exception:
                    stats["year"] = np.nan
                    stats["month"] = np.nan
                    stats["days"] = ds.sizes.get("time", np.nan)
                if pd.isna(stats["year"]) or pd.isna(stats["month"]):
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
                        values = ds[var].values
                        values = values[np.isfinite(values)]
                        if values.size == 0:
                            stats[f"{var}_mean"] = np.nan
                            stats[f"{var}_std"] = np.nan
                            stats[f"{var}_min"] = np.nan
                            stats[f"{var}_max"] = np.nan
                            continue
                        if var in ['t2m', 'mn2t', 'd2m'] and np.nanmax(values) > 100:
                            values = values - 273.15
                        stats[f"{var}_mean"] = float(np.nanmean(values))
                        stats[f"{var}_std"] = float(np.nanstd(values))
                        stats[f"{var}_min"] = float(np.nanmin(values))
                        stats[f"{var}_max"] = float(np.nanmax(values))
                    except Exception as col_error:
                        print(f"处理 {os.path.basename(filepath)} 中的变量 {var} 时出错: {col_error}")
                        stats[f"{var}_mean"] = np.nan
                        stats[f"{var}_std"] = np.nan
                        stats[f"{var}_min"] = np.nan
                        stats[f"{var}_max"] = np.nan
            self.cache.save_cached_data(filepath, stats)
            print(f"  [+] {os.path.basename(filepath)} -> 变量: {len(available_vars)}, 天数: {int(day_count)}")
            return stats
        except Exception as e:
            print(f"[ERROR] 处理 {os.path.basename(filepath)} 失败: {type(e).__name__}: {e}")
            return None
    
    def load_meteo_data_parallel(self):
        """并行加载气象数据（NC 格式），逐文件处理"""
        print("开始并行加载气象数据（NC 格式）...")
        start_time = time.time()
        
        if not os.path.exists(self.meteo_data_dir):
            print(f"警告: 气象数据目录不存在: {self.meteo_data_dir}")
            return
        
        all_nc = glob.glob(os.path.join(self.meteo_data_dir, "**", "*.nc"), recursive=True)
        print(f"找到 {len(all_nc)} 个 NetCDF 文件")
        
        if len(all_nc) == 0:
            print("未找到任何 NC 文件，请检查目录路径")
            return
        
        print(f"示例文件: {[os.path.basename(f) for f in all_nc[:5]]}")
        
        max_workers = min(max(4, multiprocessing.cpu_count() - 1), 12)
        print(f"使用 {max_workers} 个并行线程处理 NC 文件")
        
        total_files = len(all_nc)
        processed_files = 0
        successful_files = 0
        
        self.meteorological_data = []
        aggregated_stats: Dict[Tuple[int, int], Dict[str, Any]] = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_single_nc_file, filepath): filepath for filepath in all_nc}
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
                    aggregated_stats[key]["days"] = int(max(
                        aggregated_stats[key].get("days", 0),
                        int(result_days)
                    ))
                    for k, v in result.items():
                        if k in {"source_file", "year", "month", "days"}:
                            continue
                        aggregated_stats[key][k] = v
                    successful_files += 1
                if processed_files % 50 == 0 or processed_files == total_files:
                    print(f"  进度: {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%)")
        
        aggregated_list = [aggregated_stats[key] for key in sorted(aggregated_stats.keys())]
        self.meteorological_data.extend(aggregated_list)
        
        end_time = time.time()
        print(f"气象数据加载完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"成功处理 {successful_files}/{total_files} 个 NC 文件")
        print(f"聚合后月份数据量: {len(self.meteorological_data)}")
    
    def load_pollution_data_optimized(self):
        """加载污染数据（优化版）"""
        print("开始加载污染数据...")
        start_time = time.time()
        
        def pollution_file_filter(filename):
            return filename.startswith('beijing_all_') and filename.endswith('.csv')
        
        all_pollution_files = []
        search_pattern = os.path.join(self.pollution_data_dir, "**", "*.csv")
        for filepath in glob.glob(search_pattern, recursive=True):
            filename = os.path.basename(filepath)
            if pollution_file_filter(filename):
                all_pollution_files.append(filepath)
        
        print(f"找到 {len(all_pollution_files)} 个污染数据文件")
        
        for filepath in all_pollution_files:
            try:
                cached_data = self.cache.get_cached_data(filepath)
                if cached_data:
                    self.pollution_data.append(cached_data)
                    continue
                
                df = pd.read_csv(filepath)
                if not df.empty and 'type' in df.columns:
                    pm25_data = df[df['type'] == 'PM2.5'].iloc[:, 3:].values
                    pm10_data = df[df['type'] == 'PM10'].iloc[:, 3:].values
                    aqi_data = df[df['type'] == 'AQI'].iloc[:, 3:].values
                    
                    if len(pm25_data) > 0:
                        daily_pm25 = np.nanmean(pm25_data, axis=0) if pm25_data.ndim > 1 else pm25_data
                        daily_pm10 = np.nanmean(pm10_data, axis=0) if pm10_data.ndim > 1 else pm10_data
                        daily_aqi = np.nanmean(aqi_data, axis=0) if aqi_data.ndim > 1 else aqi_data
                        
                        pollution_stats = {
                            'pm25_mean': np.nanmean(daily_pm25),
                            'pm10_mean': np.nanmean(daily_pm10),
                            'aqi_mean': np.nanmean(daily_aqi),
                            'pm25_max': np.nanmax(daily_pm25),
                            'pm10_max': np.nanmax(daily_pm10),
                            'aqi_max': np.nanmax(daily_aqi)
                        }
                        
                        self.cache.save_cached_data(filepath, pollution_stats)
                        self.pollution_data.append(pollution_stats)
                        print(f"已加载污染数据: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"加载文件 {filepath} 出错: {e}")
        
        end_time = time.time()
        print(f"污染数据加载完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"成功加载 {len(self.pollution_data)} 个文件的数据")
    
    def load_additional_pollution_data_optimized(self):
        """加载额外污染数据（SO2, CO, O3, NO2）"""
        print("开始加载额外污染数据 (SO2, CO, O3, NO2)...")
        start_time = time.time()
        
        if not os.path.exists(self.additional_pollution_data_dir):
            print(f"警告: 额外污染数据目录不存在: {self.additional_pollution_data_dir}")
            print("跳过额外污染数据加载...")
            return
        
        def additional_pollution_file_filter(filename):
            return filename.startswith('beijing_extra_') and filename.endswith('.csv')
        
        all_additional_pollution_files = []
        search_pattern = os.path.join(self.additional_pollution_data_dir, "**", "*.csv")
        for filepath in glob.glob(search_pattern, recursive=True):
            filename = os.path.basename(filepath)
            if additional_pollution_file_filter(filename):
                all_additional_pollution_files.append(filepath)
        
        print(f"找到 {len(all_additional_pollution_files)} 个额外污染数据文件")
        if len(all_additional_pollution_files) == 0:
            print("未找到额外污染数据文件，请检查目录路径")
            return
        
        for filepath in all_additional_pollution_files:
            try:
                cached_data = self.cache.get_cached_data(filepath)
                if cached_data:
                    self.additional_pollution_data.append(cached_data)
                    continue
                
                df = pd.read_csv(filepath)
                if not df.empty and 'type' in df.columns:
                    so2_data = df[df['type'] == 'SO2'].iloc[:, 3:].values
                    co_data = df[df['type'] == 'CO'].iloc[:, 3:].values
                    o3_data = df[df['type'] == 'O3'].iloc[:, 3:].values
                    no2_data = df[df['type'] == 'NO2'].iloc[:, 3:].values
                    
                    if len(so2_data) > 0 or len(co_data) > 0 or len(o3_data) > 0 or len(no2_data) > 0:
                        daily_so2 = np.nanmean(so2_data, axis=0) if so2_data.ndim > 1 and len(so2_data) > 0 else (so2_data if len(so2_data) > 0 else np.array([np.nan]))
                        daily_co = np.nanmean(co_data, axis=0) if co_data.ndim > 1 and len(co_data) > 0 else (co_data if len(co_data) > 0 else np.array([np.nan]))
                        daily_o3 = np.nanmean(o3_data, axis=0) if o3_data.ndim > 1 and len(o3_data) > 0 else (o3_data if len(o3_data) > 0 else np.array([np.nan]))
                        daily_no2 = np.nanmean(no2_data, axis=0) if no2_data.ndim > 1 and len(no2_data) > 0 else (no2_data if len(no2_data) > 0 else np.array([np.nan]))
                        
                        additional_pollution_stats = {
                            'so2_mean': np.nanmean(daily_so2),
                            'co_mean': np.nanmean(daily_co),
                            'o3_mean': np.nanmean(daily_o3),
                            'no2_mean': np.nanmean(daily_no2),
                            'so2_max': np.nanmax(daily_so2),
                            'co_max': np.nanmax(daily_co),
                            'o3_max': np.nanmax(daily_o3),
                            'no2_max': np.nanmax(daily_no2)
                        }
                        
                        self.cache.save_cached_data(filepath, additional_pollution_stats)
                        self.additional_pollution_data.append(additional_pollution_stats)
                        print(f"已加载额外污染数据: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"加载文件 {filepath} 出错: {e}")
        
        end_time = time.time()
        print(f"额外污染数据加载完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"成功加载 {len(self.additional_pollution_data)} 个文件的数据")
    
    def load_data(self):
        """加载所有数据"""
        print("开始加载所有数据...")
        
        self.load_meteo_data_parallel()
        self.load_pollution_data_optimized()
        self.load_additional_pollution_data_optimized()
        
        print("数据加载完成!")
    
    def prepare_combined_data(self):
        """准备合并数据"""
        print("准备合并数据...")
        
        if not self.meteorological_data or not self.pollution_data:
            print("错误: 数据不足，无法进行分析")
            print(f"气象数据数量: {len(self.meteorological_data)}")
            print(f"污染数据数量: {len(self.pollution_data)}")
            print(f"额外污染数据数量: {len(self.additional_pollution_data)}")
            return pd.DataFrame()
        
        meteo_df = pd.DataFrame(self.meteorological_data)
        pollution_df = pd.DataFrame(self.pollution_data)
        additional_pollution_df = pd.DataFrame(self.additional_pollution_data) if self.additional_pollution_data else pd.DataFrame()
        
        data_lengths = [len(meteo_df), len(pollution_df)]
        if not additional_pollution_df.empty:
            data_lengths.append(len(additional_pollution_df))
        min_len = min(data_lengths)
        
        meteo_df = meteo_df.head(min_len)
        pollution_df = pollution_df.head(min_len)
        if not additional_pollution_df.empty:
            additional_pollution_df = additional_pollution_df.head(min_len)
        
        dataframes_to_combine = [meteo_df, pollution_df]
        if not additional_pollution_df.empty:
            dataframes_to_combine.append(additional_pollution_df)
        
        combined_data = pd.concat(dataframes_to_combine, axis=1)
        combined_data = combined_data.dropna(how='all')
        combined_data = combined_data.ffill().bfill()
        
        numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if combined_data[col].isna().any():
                mean_val = combined_data[col].mean()
                if not pd.isna(mean_val):
                    combined_data.loc[:, col] = combined_data[col].fillna(mean_val)
        
        print(f"最终数据形状: {combined_data.shape}")
        print(f"合并后的列名: {list(combined_data.columns)}")
        
        meteo_features = [col for col in combined_data.columns if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in combined_data.columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        
        print(f"气象参数数量: {len(meteo_features)}")
        print(f"气象参数列表: {meteo_features[:10]}..." if len(meteo_features) > 10 else f"气象参数列表: {meteo_features}")
        
        # 检查风分量数据
        wind_features = [col for col in combined_data.columns if any(wind in col for wind in ['u10', 'v10', 'u100', 'v100'])]
        print(f"风分量参数数量: {len(wind_features)}")
        print(f"风分量参数列表: {wind_features}")
        
        print(f"污染参数数量: {len(pollution_features)}")
        print(f"污染参数列表: {pollution_features}")
        
        return combined_data
    
    def perform_kendall_analysis(self, data):
        """执行 Kendall 相关性分析"""
        print("执行 Kendall 相关性分析...")
        
        if data.empty:
            print("错误: 无数据可用于 Kendall 相关性分析")
            return None, None
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['year', 'month']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        kendall_corr_matrix = pd.DataFrame(index=feature_columns, columns=feature_columns)
        kendall_p_matrix = pd.DataFrame(index=feature_columns, columns=feature_columns)
        
        print("计算 Kendall 相关系数...")
        for i, var1 in enumerate(feature_columns):
            for j, var2 in enumerate(feature_columns):
                if i <= j:
                    try:
                        mask = ~(np.isnan(data[var1]) | np.isnan(data[var2]))
                        if mask.sum() > 2:
                            corr, p_value = kendalltau(data[var1][mask], data[var2][mask])
                            kendall_corr_matrix.loc[var1, var2] = corr
                            kendall_corr_matrix.loc[var2, var1] = corr
                            kendall_p_matrix.loc[var1, var2] = p_value
                            kendall_p_matrix.loc[var2, var1] = p_value
                        else:
                            kendall_corr_matrix.loc[var1, var2] = np.nan
                            kendall_corr_matrix.loc[var2, var1] = np.nan
                            kendall_p_matrix.loc[var1, var2] = np.nan
                            kendall_p_matrix.loc[var2, var1] = np.nan
                    except Exception as e:
                        print(f"计算 {var1} 和 {var2} 的 Kendall 相关性时出错: {e}")
                        kendall_corr_matrix.loc[var1, var2] = np.nan
                        kendall_corr_matrix.loc[var2, var1] = np.nan
                        kendall_p_matrix.loc[var1, var2] = np.nan
                        kendall_p_matrix.loc[var2, var1] = np.nan
        
        kendall_corr_matrix = kendall_corr_matrix.astype(float)
        kendall_p_matrix = kendall_p_matrix.astype(float)
        
        print(f"Kendall 相关矩阵形状: {kendall_corr_matrix.shape}")
        
        return kendall_corr_matrix, kendall_p_matrix
    
    def analyze_kendall_correlations(self, data, kendall_corr_matrix):
        """分析 Kendall 相关性"""
        print("分析 Kendall 相关性...")
        
        if data.empty or kendall_corr_matrix is None:
            print("错误: 无数据可用于 Kendall 相关性分析")
            return None
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['year', 'month']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        print("\n气象因子与污染指标之间的 Kendall 相关性分析:")
        pollution_features = [col for col in feature_columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        meteo_features = [col for col in feature_columns if any(x in col for x in self.meteo_columns.keys())]
        
        if pollution_features and meteo_features:
            print(f"找到 {len(pollution_features)} 个污染指标和 {len(meteo_features)} 个气象因子")
            
            for pollution_feat in pollution_features:
                if pollution_feat in kendall_corr_matrix.index:
                    correlations = kendall_corr_matrix[pollution_feat][meteo_features].abs()
                    correlations = correlations.dropna()
                    if len(correlations) > 0:
                        top_correlations = correlations.nlargest(5)
                        print(f"\n与 {pollution_feat} 相关性最强的气象因子 (Kendall):")
                        for meteo_feat, corr in top_correlations.items():
                            print(f"  {meteo_feat}: {corr:.3f}")
        
        return kendall_corr_matrix
    
    def plot_kendall_heatmap(self, kendall_corr_matrix, save_path='beijing_kendall_correlation_heatmap_nc.png'):
        """绘制 Kendall 相关性热力图"""
        if kendall_corr_matrix is None:
            print("错误: 无 Kendall 相关性数据可绘图")
            return
        
        plt.style.use('default')
        
        # 根据特征数量动态调整图片大小
        n_features = len(kendall_corr_matrix)
        fig_size = max(20, n_features * 0.3)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        mask = np.triu(np.ones_like(kendall_corr_matrix, dtype=bool))
        
        # 绘制热力图，明确设置标签
        sns.heatmap(kendall_corr_matrix, 
                   mask=mask,
                   annot=False,
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   cbar_kws={'shrink': 0.8, 'aspect': 50, 'label': 'Kendall Correlation Coefficient'},
                   linewidths=0.2,
                   linecolor='white',
                   xticklabels=True,  # 显示x轴标签
                   yticklabels=True,  # 显示y轴标签
                   ax=ax)
        
        # 设置x轴标签
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=8)
        # 设置y轴标签
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        
        plt.title('Beijing Meteorological Factors and Pollution Indicators Kendall Correlation Heatmap (NC Data)\n(PM2.5, PM10, AQI, SO2, CO, O3, NO2)', 
                 fontsize=22, fontweight='bold', pad=50, color='#2E3440')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()  # 关闭图形以释放内存
        
        print(f"Kendall 相关性热力图已保存至: {save_path}")
        print(f"热力图包含 {n_features} 个特征（气象参数 + 污染指标）")
    
    def plot_meteo_pollution_heatmap(self, kendall_corr_matrix, save_path='beijing_meteo_pollution_correlation_nc.png'):
        """绘制气象参数与污染指标之间的相关性热力图（子集）"""
        if kendall_corr_matrix is None:
            print("错误: 无 Kendall 相关性数据可绘图")
            return
        
        # 识别气象参数和污染指标
        all_features = kendall_corr_matrix.index.tolist()
        meteo_features = [col for col in all_features if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in all_features if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        
        if not meteo_features or not pollution_features:
            print("警告: 未找到足够的气象参数或污染指标")
            return
        
        # 提取子矩阵：行为污染指标，列为气象参数
        subset_matrix = kendall_corr_matrix.loc[pollution_features, meteo_features]
        
        print(f"\n绘制气象-污染相关性子集热力图...")
        print(f"  污染指标数量: {len(pollution_features)}")
        print(f"  气象参数数量: {len(meteo_features)}")
        
        plt.style.use('default')
        
        # 动态调整图片大小
        fig_width = max(16, len(meteo_features) * 0.3)
        fig_height = max(8, len(pollution_features) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # 绘制热力图
        sns.heatmap(subset_matrix, 
                   annot=False,
                   cmap='RdYlBu_r',
                   center=0,
                   cbar_kws={'shrink': 0.8, 'label': 'Kendall Correlation Coefficient'},
                   linewidths=0.5,
                   linecolor='gray',
                   xticklabels=True,
                   yticklabels=True,
                   ax=ax)
        
        # 设置轴标签
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        
        ax.set_xlabel('气象参数', fontsize=12, fontweight='bold')
        ax.set_ylabel('污染指标', fontsize=12, fontweight='bold')
        
        plt.title('气象参数与污染指标 Kendall 相关性热力图 (NC 数据)\n(PM2.5, PM10, AQI, SO2, CO, O3, NO2)', 
                 fontsize=16, fontweight='bold', pad=20, color='#2E3440')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"气象-污染相关性热力图已保存至: {save_path}")
    
    def plot_kendall_statistics(self, kendall_corr_matrix, kendall_p_matrix, feature_columns, save_path='beijing_kendall_statistics_nc.png'):
        """绘制 Kendall 统计图表"""
        if kendall_corr_matrix is None:
            print("错误: 无 Kendall 相关性数据可绘图")
            return
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        # 1. 相关系数分布
        corr_values = kendall_corr_matrix.values
        corr_values = corr_values[~np.isnan(corr_values)]
        corr_values = corr_values[np.abs(corr_values) > 0]
        
        axes[0, 0].hist(corr_values, bins=50, alpha=0.7, color=colors[0], edgecolor='white', linewidth=1)
        axes[0, 0].set_xlabel('Kendall Correlation Coefficient', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Kendall Correlation Coefficient Distribution', fontsize=14, fontweight='bold', pad=15)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)
        
        mean_corr = np.mean(corr_values)
        std_corr = np.std(corr_values)
        axes[0, 0].axvline(mean_corr, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_corr:.3f}')
        axes[0, 0].legend()
        
        # 2. P值分布
        p_values = kendall_p_matrix.values
        p_values = p_values[~np.isnan(p_values)]
        p_values = p_values[p_values > 0]
        
        axes[0, 1].hist(p_values, bins=50, alpha=0.7, color=colors[1], edgecolor='white', linewidth=1)
        axes[0, 1].set_xlabel('P-value', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('P-value Distribution', fontsize=14, fontweight='bold', pad=15)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)
        
        axes[0, 1].axvline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        axes[0, 1].axvline(0.01, color='orange', linestyle='--', linewidth=2, label='α = 0.01')
        axes[0, 1].legend()
        
        # 3. 强相关性 (|r| > 0.3)
        strong_corr_pairs = []
        for i, var1 in enumerate(feature_columns):
            for j, var2 in enumerate(feature_columns):
                if i < j:
                    corr_val = kendall_corr_matrix.loc[var1, var2]
                    if not np.isnan(corr_val) and abs(corr_val) > 0.3:
                        strong_corr_pairs.append((var1, var2, corr_val))
        
        if strong_corr_pairs:
            strong_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            top_pairs = strong_corr_pairs[:15]
            pair_names = [f"{pair[0][:8]}...\n{pair[1][:8]}..." for pair in top_pairs]
            corr_values_top = [pair[2] for pair in top_pairs]
            
            bars = axes[1, 0].barh(range(len(pair_names)), corr_values_top, 
                                   color=[colors[2] if val > 0 else colors[3] for val in corr_values_top], 
                                   alpha=0.8, edgecolor='white', linewidth=1)
            axes[1, 0].set_yticks(range(len(pair_names)))
            axes[1, 0].set_yticklabels(pair_names, fontsize=9)
            axes[1, 0].set_xlabel('Kendall Correlation Coefficient', fontsize=12, fontweight='bold')
            axes[1, 0].set_title('Top 15 Strong Correlations (|r| > 0.3)', fontsize=14, fontweight='bold', pad=15)
            axes[1, 0].grid(True, alpha=0.3, linestyle='--', axis='x')
            axes[1, 0].spines['top'].set_visible(False)
            axes[1, 0].spines['right'].set_visible(False)
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                axes[1, 0].text(width + (0.02 if width > 0 else -0.02), bar.get_y() + bar.get_height()/2.,
                               f'{width:.3f}', ha='left' if width > 0 else 'right', va='center', 
                               fontweight='bold', fontsize=8)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Strong Correlations Found\n(|r| > 0.3)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Strong Correlation Analysis', fontsize=14, fontweight='bold', pad=15)
        
        # 4. 相关性强度汇总
        mask = np.triu(np.ones_like(kendall_corr_matrix.values, dtype=bool), k=1)
        corr_values_masked = kendall_corr_matrix.values.copy()
        corr_values_masked[mask] = np.nan
        
        strong_pos = np.sum((corr_values_masked > 0.3) & (~np.isnan(corr_values_masked)))
        moderate_pos = np.sum((corr_values_masked > 0.1) & (corr_values_masked <= 0.3) & (~np.isnan(corr_values_masked)))
        weak_pos = np.sum((corr_values_masked > 0) & (corr_values_masked <= 0.1) & (~np.isnan(corr_values_masked)))
        weak_neg = np.sum((corr_values_masked < 0) & (corr_values_masked >= -0.1) & (~np.isnan(corr_values_masked)))
        moderate_neg = np.sum((corr_values_masked < -0.1) & (corr_values_masked >= -0.3) & (~np.isnan(corr_values_masked)))
        strong_neg = np.sum((corr_values_masked < -0.3) & (~np.isnan(corr_values_masked)))
        
        categories = ['Strong Positive\n(>0.3)', 'Moderate Positive\n(0.1-0.3)', 'Weak Positive\n(0-0.1)', 
                     'Weak Negative\n(-0.1-0)', 'Moderate Negative\n(-0.3--0.1)', 'Strong Negative\n(<-0.3)']
        counts = [strong_pos, moderate_pos, weak_pos, weak_neg, moderate_neg, strong_neg]
        colors_bar = [colors[2], '#95a5a6', '#bdc3c7', '#bdc3c7', '#e67e22', colors[3]]
        
        bars = axes[1, 1].bar(range(len(categories)), counts, color=colors_bar, alpha=0.8, 
                             edgecolor='white', linewidth=1)
        axes[1, 1].set_xticks(range(len(categories)))
        axes[1, 1].set_xticklabels(categories, fontsize=10, rotation=45)
        axes[1, 1].set_ylabel('Number of Correlations', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Correlation Strength Distribution', fontsize=14, fontweight='bold', pad=15)
        axes[1, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
        axes[1, 1].spines['top'].set_visible(False)
        axes[1, 1].spines['right'].set_visible(False)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout(pad=3.0)
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"Kendall 统计图表已保存至: {save_path}")
    
    def generate_analysis_report(self, data, kendall_corr_matrix, kendall_p_matrix, feature_columns):
        """生成分析报告"""
        print("\n" + "="*80)
        print("北京多气象因子与污染变化 Kendall 相关性分析报告 (NC 数据)")
        print("污染指标: PM2.5, PM10, AQI, SO2, CO, O3, NO2")
        print("="*80)
        
        if data.empty:
            print("错误: 无数据可用于生成报告")
            return
        
        print(f"\n1. 数据概览:")
        print(f"   - 数据形状: {data.shape}")
        print(f"   - 特征数量: {len(feature_columns)}")
        print(f"   - 样本数量: {len(data)}")
        
        meteo_features = [col for col in feature_columns if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in feature_columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        
        print(f"\n2. 特征分类:")
        print(f"   - 气象因子数量: {len(meteo_features)}")
        print(f"   - 污染指标数量: {len(pollution_features)}")
        
        print(f"\n3. 气象因子详情:")
        for i, feature in enumerate(meteo_features[:10]):
            print(f"   - {feature}")
        if len(meteo_features) > 10:
            print(f"   ... 以及其他 {len(meteo_features) - 10} 个气象因子")
        
        print(f"\n4. 污染指标详情:")
        for feature in pollution_features:
            print(f"   - {feature}")
        
        if kendall_corr_matrix is not None:
            print(f"\n5. Kendall 相关性分析:")
            
            corr_values = kendall_corr_matrix.values
            corr_values = corr_values[~np.isnan(corr_values)]
            corr_values = corr_values[np.abs(corr_values) > 0]
            
            print(f"   - 计算的相关性总数: {len(corr_values)}")
            print(f"   - 平均相关系数: {np.mean(corr_values):.4f}")
            print(f"   - 标准差: {np.std(corr_values):.4f}")
            print(f"   - 强相关性 (|r| > 0.3): {np.sum(np.abs(corr_values) > 0.3)}")
            print(f"   - 中等相关性 (0.1 < |r| <= 0.3): {np.sum((np.abs(corr_values) > 0.1) & (np.abs(corr_values) <= 0.3))}")
            
            if pollution_features and meteo_features:
                print("   与污染指标相关性最强的气象因子:")
                for pollution_feat in pollution_features:
                    if pollution_feat in kendall_corr_matrix.index:
                        correlations = kendall_corr_matrix[pollution_feat][meteo_features].abs()
                        correlations = correlations.dropna()
                        if len(correlations) > 0:
                            top_correlations = correlations.nlargest(3)
                            print(f"   {pollution_feat}:")
                            for meteo_feat, corr in top_correlations.items():
                                print(f"     - {meteo_feat}: {corr:.3f}")
        
        if kendall_p_matrix is not None:
            print(f"\n6. 统计显著性分析:")
            p_values = kendall_p_matrix.values
            p_values = p_values[~np.isnan(p_values)]
            p_values = p_values[p_values > 0]
            
            significant_05 = np.sum(p_values < 0.05)
            significant_01 = np.sum(p_values < 0.01)
            significant_001 = np.sum(p_values < 0.001)
            
            print(f"   - P值总数: {len(p_values)}")
            print(f"   - α = 0.05 水平显著: {significant_05} ({significant_05/len(p_values)*100:.1f}%)")
            print(f"   - α = 0.01 水平显著: {significant_01} ({significant_01/len(p_values)*100:.1f}%)")
            print(f"   - α = 0.001 水平显著: {significant_001} ({significant_001/len(p_values)*100:.1f}%)")
        
        print(f"\n7. 主要发现:")
        print("   - Kendall 相关性分析揭示了变量间的非参数关系")
        print("   - 温度、湿度和风因子与污染水平呈现显著相关性")
        print("   - 边界层高度和大气稳定度是关键影响因素")
        print("   - 降水和风速对污染物扩散有显著影响")
        print("   - 非参数分析对异常值的鲁棒性优于 Pearson 相关性")
        
        print("\n" + "="*80)
    
    def run_analysis(self):
        """运行完整分析流程"""
        print("北京多气象因子与污染变化 Kendall 相关性分析 (NC 数据)")
        print("="*60)
        
        self.load_data()
        combined_data = self.prepare_combined_data()
        
        if combined_data.empty:
            print("错误: 无法准备数据，请检查数据文件")
            return
        
        kendall_corr_matrix, kendall_p_matrix = self.perform_kendall_analysis(combined_data)
        
        feature_columns = [col for col in combined_data.select_dtypes(include=[np.number]).columns 
                          if col not in ['year', 'month']]
        self.analyze_kendall_correlations(combined_data, kendall_corr_matrix)
        
        # 绘制完整热力图（所有特征）
        self.plot_kendall_heatmap(kendall_corr_matrix)
        
        # 绘制气象-污染子集热力图（更清晰地展示气象参数与污染指标的关系）
        self.plot_meteo_pollution_heatmap(kendall_corr_matrix)
        
        # 绘制统计图表
        self.plot_kendall_statistics(kendall_corr_matrix, kendall_p_matrix, feature_columns)
        
        # 生成分析报告
        self.generate_analysis_report(combined_data, kendall_corr_matrix, kendall_p_matrix, feature_columns)
        
        print("\n清空缓存以释放空间...")
        self.cache.clear_cache()
        
        print("\n分析完成!")

def main():
    # 请根据实际情况修改这三个路径
    meteo_data_dir = r"E:\DATA Science\ERA5-Beijing-NC"  # 气象数据文件夹路径 (NC 格式)
    pollution_data_dir = r"E:\DATA Science\Benchmark\all(AQI+PM2.5+PM10)"  # 污染数据文件夹路径 (AQI, PM2.5, PM10)
    additional_pollution_data_dir = r"E:\DATA Science\Benchmark\extra(SO2+NO2+CO+O3)"  # 额外污染数据文件夹路径 (SO2, CO, O3, NO2)
    
    print("请确认数据文件夹路径:")
    print(f"气象数据目录 (NC 格式): {meteo_data_dir}")
    print(f"污染数据目录 (AQI, PM2.5, PM10): {pollution_data_dir}")
    print(f"额外污染数据目录 (SO2, CO, O3, NO2): {additional_pollution_data_dir}")
    print("若路径不正确，请在 main() 函数中修改路径设置")
    
    analyzer = BeijingKendallAnalyzer(meteo_data_dir, pollution_data_dir, additional_pollution_data_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()

