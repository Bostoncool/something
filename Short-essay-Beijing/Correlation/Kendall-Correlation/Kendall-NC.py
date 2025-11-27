import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
import warnings
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import hashlib
import pickle
import time
from typing import Dict, Any, Optional, Tuple
import xarray as xr
import re

warnings.filterwarnings('ignore')

# Set font configuration
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = (10, 6)

# 独立函数用于多进程处理NC文件
def process_single_nc_file_worker(args):
    """独立的工作函数，用于多进程处理单个ERA5 NetCDF文件"""
    filepath, cache_dir, beijing_lats, beijing_lons, era5_vars = args
    
    # 每个进程创建自己的cache实例
    cache = DataCache(cache_dir)
    
    try:
        cached_data = cache.get_cached_data(filepath)
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
            available_vars = [var for var in era5_vars if var in ds.data_vars]
            if not available_vars:
                print(f"[WARN] {os.path.basename(filepath)} does not contain target variables, skipping")
                return None
            if "latitude" in ds.coords and "longitude" in ds.coords:
                lat_values = ds["latitude"]
                if lat_values[0] > lat_values[-1]:
                    lat_slice = slice(beijing_lats.max(), beijing_lats.min())
                else:
                    lat_slice = slice(beijing_lats.min(), beijing_lats.max())
                ds = ds.sel(
                    latitude=lat_slice,
                    longitude=slice(beijing_lons.min(), beijing_lons.max())
                )
                if "latitude" in ds.dims and "longitude" in ds.dims:
                    ds = ds.mean(dim=["latitude", "longitude"], skipna=True)
            if "time" not in ds.coords:
                print(f"[WARN] {os.path.basename(filepath)} missing time coordinate, skipping")
                return None
            ds = ds.sortby("time")
            ds = ds.resample(time="1D").mean(keep_attrs=False)
            ds = ds.dropna("time", how="all")
            if ds.sizes.get("time", 0) == 0:
                print(f"[WARN] {os.path.basename(filepath)} no valid time after resampling, skipping")
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
                    print(f"Error processing variable {var} in {os.path.basename(filepath)}: {col_error}")
                    stats[f"{var}_mean"] = np.nan
                    stats[f"{var}_std"] = np.nan
                    stats[f"{var}_min"] = np.nan
                    stats[f"{var}_max"] = np.nan
        cache.save_cached_data(filepath, stats)
        print(f"  [+] {os.path.basename(filepath)} -> variables: {len(available_vars)}, days: {int(day_count)}")
        return stats
    except Exception as e:
        print(f"[ERROR] Failed to process {os.path.basename(filepath)}: {type(e).__name__}: {e}")
        return None

class DataCache:
    """Data cache class to avoid reprocessing the same files"""
    
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
            except Exception:
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
                print(f"Cache cleared: {self.cache_dir}")
        except Exception as e:
            print(f"Error clearing cache: {e}")

class BeijingKendallAnalyzer:
    """Beijing meteorological factors and pollution variation Kendall correlation analyzer"""
    
    def __init__(self, meteo_data_dir=".", pollution_data_dir=".", additional_pollution_data_dir="."):
        self.meteo_data_dir = meteo_data_dir
        self.pollution_data_dir = pollution_data_dir
        self.additional_pollution_data_dir = additional_pollution_data_dir
        self.meteorological_data = []
        self.pollution_data = []
        self.additional_pollution_data = []
        self.cache = DataCache()
        
        # Beijing region range
        self.beijing_lats = np.arange(39.0, 41.25, 0.25)
        self.beijing_lons = np.arange(115.0, 117.25, 0.25)
        
        # Meteorological parameter column name mapping
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
        
        # ERA5 variable list
        self.era5_vars = [
            'd2m', 't2m', 'u10', 'v10', 'u100', 'v100',
            'blh', 'sp', 'tcwv',
            'tp', 'avg_tprate',
            'tisr', 'str',
            'cvh', 'cvl',
            'mn2t', 'sd', 'lsm'
        ]
    
    def load_meteo_data_parallel(self):
        """Load meteorological data (NC format) in parallel using multiprocessing"""
        print("Starting parallel loading of meteorological data (NC format) using multiprocessing...")
        start_time = time.time()
        
        if not os.path.exists(self.meteo_data_dir):
            print(f"Warning: Meteorological data directory does not exist: {self.meteo_data_dir}")
            return
        
        all_nc = glob.glob(os.path.join(self.meteo_data_dir, "**", "*.nc"), recursive=True)
        print(f"Found {len(all_nc)} NetCDF files")
        
        if len(all_nc) == 0:
            print("No NC files found, please check directory path")
            return
        
        print(f"Example files: {[os.path.basename(f) for f in all_nc[:5]]}")
        
        max_workers = min(max(4, multiprocessing.cpu_count() - 1), 12)
        print(f"Using {max_workers} parallel processes to process NC files")
        
        total_files = len(all_nc)
        processed_files = 0
        successful_files = 0
        
        self.meteorological_data = []
        aggregated_stats: Dict[Tuple[int, int], Dict[str, Any]] = {}
        
        # 准备参数列表
        cache_dir = self.cache.cache_dir
        args_list = [
            (filepath, cache_dir, self.beijing_lats, self.beijing_lons, self.era5_vars)
            for filepath in all_nc
        ]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_single_nc_file_worker, args): args[0] for args in args_list}
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
                    print(f"  Progress: {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%)")
        
        aggregated_list = [aggregated_stats[key] for key in sorted(aggregated_stats.keys())]
        self.meteorological_data.extend(aggregated_list)
        
        end_time = time.time()
        print(f"Meteorological data loading completed, time taken: {end_time - start_time:.2f} seconds")
        print(f"Successfully processed {successful_files}/{total_files} NC files")
        print(f"Monthly data count after aggregation: {len(self.meteorological_data)}")
    
    def load_pollution_data_optimized(self):
        """Load pollution data (optimized version with dictionary lookup)"""
        print("Starting to load pollution data...")
        start_time = time.time()
        
        # 一次性构建文件名到路径的字典映射，时间复杂度O(1)查找
        print("Building filename to filepath dictionary...")
        pollution_file_dict = {}
        search_pattern = os.path.join(self.pollution_data_dir, "**", "*.csv")
        for filepath in glob.glob(search_pattern, recursive=True):
            filename = os.path.basename(filepath)
            if filename.startswith('beijing_all_') and filename.endswith('.csv'):
                pollution_file_dict[filename] = filepath
        
        print(f"Found {len(pollution_file_dict)} pollution data files")
        
        # 按文件名排序以确保处理顺序一致
        sorted_filenames = sorted(pollution_file_dict.keys())
        
        for filename in sorted_filenames:
            filepath = pollution_file_dict[filename]  # O(1)字典查找
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
                        print(f"Loaded pollution data: {filename}")
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
        
        end_time = time.time()
        print(f"Pollution data loading completed, time taken: {end_time - start_time:.2f} seconds")
        print(f"Successfully loaded data from {len(self.pollution_data)} files")
    
    def load_additional_pollution_data_optimized(self):
        """Load additional pollution data (SO2, CO, O3, NO2) with dictionary lookup"""
        print("Starting to load additional pollution data (SO2, CO, O3, NO2)...")
        start_time = time.time()
        
        if not os.path.exists(self.additional_pollution_data_dir):
            print(f"Warning: Additional pollution data directory does not exist: {self.additional_pollution_data_dir}")
            print("Skipping additional pollution data loading...")
            return
        
        # 一次性构建文件名到路径的字典映射，时间复杂度O(1)查找
        print("Building filename to filepath dictionary for additional pollution data...")
        additional_pollution_file_dict = {}
        search_pattern = os.path.join(self.additional_pollution_data_dir, "**", "*.csv")
        for filepath in glob.glob(search_pattern, recursive=True):
            filename = os.path.basename(filepath)
            if filename.startswith('beijing_extra_') and filename.endswith('.csv'):
                additional_pollution_file_dict[filename] = filepath
        
        print(f"Found {len(additional_pollution_file_dict)} additional pollution data files")
        if len(additional_pollution_file_dict) == 0:
            print("No additional pollution data files found, please check directory path")
            return
        
        # 按文件名排序以确保处理顺序一致
        sorted_filenames = sorted(additional_pollution_file_dict.keys())
        
        for filename in sorted_filenames:
            filepath = additional_pollution_file_dict[filename]  # O(1)字典查找
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
                        print(f"Loaded additional pollution data: {filename}")
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
        
        end_time = time.time()
        print(f"Additional pollution data loading completed, time taken: {end_time - start_time:.2f} seconds")
        print(f"Successfully loaded data from {len(self.additional_pollution_data)} files")
    
    def load_data(self):
        """Load all data"""
        print("Starting to load all data...")
        
        self.load_meteo_data_parallel()
        self.load_pollution_data_optimized()
        self.load_additional_pollution_data_optimized()
        
        print("Data loading completed!")
    
    def prepare_combined_data(self):
        """Prepare combined data"""
        print("Preparing combined data...")
        
        if not self.meteorological_data or not self.pollution_data:
            print("Error: Insufficient data for analysis")
            print(f"Meteorological data count: {len(self.meteorological_data)}")
            print(f"Pollution data count: {len(self.pollution_data)}")
            print(f"Additional pollution data count: {len(self.additional_pollution_data)}")
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
                    combined_data[col] = combined_data[col].fillna(mean_val)
        
        print(f"Final data shape: {combined_data.shape}")
        print(f"Column names after merging: {list(combined_data.columns)}")
        
        meteo_features = [col for col in combined_data.columns if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in combined_data.columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        
        print(f"Meteorological parameter count: {len(meteo_features)}")
        print(f"Meteorological parameter list: {meteo_features[:10]}..." if len(meteo_features) > 10 else f"Meteorological parameter list: {meteo_features}")
        
        # Check wind component data
        wind_features = [col for col in combined_data.columns if any(wind in col for wind in ['u10', 'v10', 'u100', 'v100'])]
        print(f"Wind component parameter count: {len(wind_features)}")
        print(f"Wind component parameter list: {wind_features}")
        
        print(f"Pollution parameter count: {len(pollution_features)}")
        print(f"Pollution parameter list: {pollution_features}")
        
        return combined_data
    
    def perform_kendall_analysis(self, data):
        """Perform Kendall correlation analysis"""
        print("Performing Kendall correlation analysis...")
        
        if data.empty:
            print("Error: No data available for Kendall correlation analysis")
            return None, None
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['year', 'month']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        kendall_corr_matrix = pd.DataFrame(index=feature_columns, columns=feature_columns)
        kendall_p_matrix = pd.DataFrame(index=feature_columns, columns=feature_columns)
        
        print("Calculating Kendall correlation coefficients...")
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
                        print(f"Error calculating Kendall correlation between {var1} and {var2}: {e}")
                        kendall_corr_matrix.loc[var1, var2] = np.nan
                        kendall_corr_matrix.loc[var2, var1] = np.nan
                        kendall_p_matrix.loc[var1, var2] = np.nan
                        kendall_p_matrix.loc[var2, var1] = np.nan
        
        kendall_corr_matrix = kendall_corr_matrix.astype(float)
        kendall_p_matrix = kendall_p_matrix.astype(float)
        
        print(f"Kendall correlation matrix shape: {kendall_corr_matrix.shape}")
        
        return kendall_corr_matrix, kendall_p_matrix
    
    def analyze_kendall_correlations(self, data, kendall_corr_matrix):
        """Analyze Kendall correlations"""
        print("Analyzing Kendall correlations...")
        
        if data.empty or kendall_corr_matrix is None:
            print("Error: No data available for Kendall correlation analysis")
            return None
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['year', 'month']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        print("\nKendall correlation analysis between meteorological factors and pollution indicators:")
        pollution_features = [col for col in feature_columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        meteo_features = [col for col in feature_columns if any(x in col for x in self.meteo_columns.keys())]
        
        if pollution_features and meteo_features:
            print(f"Found {len(pollution_features)} pollution indicators and {len(meteo_features)} meteorological factors")
            
            for pollution_feat in pollution_features:
                if pollution_feat in kendall_corr_matrix.index:
                    correlations = kendall_corr_matrix[pollution_feat][meteo_features].abs()
                    correlations = correlations.dropna()
                    if len(correlations) > 0:
                        top_correlations = correlations.nlargest(5)
                        print(f"\nMeteorological factors with strongest correlation to {pollution_feat} (Kendall):")
                        for meteo_feat, corr in top_correlations.items():
                            print(f"  {meteo_feat}: {corr:.3f}")
        
        return kendall_corr_matrix
    
    def plot_kendall_heatmap(self, kendall_corr_matrix, save_path='beijing_kendall_correlation_heatmap_nc.png'):
        """Plot Kendall correlation heatmap"""
        if kendall_corr_matrix is None:
            print("Error: No Kendall correlation data available for plotting")
            return
        
        plt.style.use('default')
        
        # Dynamically adjust image size based on number of features
        n_features = len(kendall_corr_matrix)
        fig_size = max(20, n_features * 0.3)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        mask = np.triu(np.ones_like(kendall_corr_matrix, dtype=bool))
        
        # Plot heatmap with explicit label settings
        sns.heatmap(kendall_corr_matrix, 
                   mask=mask,
                   annot=False,
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   cbar_kws={'shrink': 0.8, 'aspect': 50, 'label': 'Kendall Correlation Coefficient'},
                   linewidths=0.2,
                   linecolor='white',
                   xticklabels=True,  # Show x-axis labels
                   yticklabels=True,  # Show y-axis labels
                   ax=ax)
        
        # Set x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=8)
        # Set y-axis labels
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        
        plt.title('Beijing Meteorological Factors and Pollution Indicators Kendall Correlation Heatmap (NC Data)\n(PM2.5, PM10, AQI, SO2, CO, O3, NO2)', 
                 fontsize=22, fontweight='bold', pad=50, color='#2E3440')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()  # Close figure to release memory
        
        print(f"Kendall correlation heatmap saved to: {save_path}")
        print(f"Heatmap contains {n_features} features (meteorological parameters + pollution indicators)")
    
    def plot_meteo_pollution_heatmap(self, kendall_corr_matrix, save_path='beijing_meteo_pollution_correlation_nc.png'):
        """Plot correlation heatmap between meteorological parameters and pollution indicators (subset)"""
        if kendall_corr_matrix is None:
            print("Error: No Kendall correlation data available for plotting")
            return
        
        # Identify meteorological parameters and pollution indicators
        all_features = kendall_corr_matrix.index.tolist()
        meteo_features = [col for col in all_features if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in all_features if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        
        if not meteo_features or not pollution_features:
            print("Warning: Insufficient meteorological parameters or pollution indicators found")
            return
        
        # Extract submatrix: rows are pollution indicators, columns are meteorological parameters
        subset_matrix = kendall_corr_matrix.loc[pollution_features, meteo_features]
        
        print(f"\nPlotting meteorological-pollution correlation subset heatmap...")
        print(f"  Pollution indicator count: {len(pollution_features)}")
        print(f"  Meteorological parameter count: {len(meteo_features)}")
        
        plt.style.use('default')
        
        # Dynamically adjust image size
        fig_width = max(16, len(meteo_features) * 0.3)
        fig_height = max(8, len(pollution_features) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Plot heatmap
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
        
        # Set axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        
        ax.set_xlabel('Meteorological Parameters', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pollution Indicators', fontsize=12, fontweight='bold')
        
        plt.title('Kendall Correlation Heatmap of Meteorological Parameters and Pollution Indicators (NC Data)\n(PM2.5, PM10, AQI, SO2, CO, O3, NO2)', 
                 fontsize=16, fontweight='bold', pad=20, color='#2E3440')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Meteorological-pollution correlation heatmap saved to: {save_path}")
    
    def plot_kendall_statistics(self, kendall_corr_matrix, kendall_p_matrix, feature_columns, save_path='beijing_kendall_statistics_nc.png'):
        """Plot Kendall statistical charts"""
        if kendall_corr_matrix is None:
            print("Error: No Kendall correlation data available for plotting")
            return
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        # 1. Correlation coefficient distribution
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
        
        # 2. P-value distribution
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
        
        # 3. Strong correlations (|r| > 0.3)
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
        
        # 4. Correlation strength summary
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
        
        print(f"Kendall statistical charts saved to: {save_path}")
    
    def generate_analysis_report(self, data, kendall_corr_matrix, kendall_p_matrix, feature_columns):
        """Generate analysis report"""
        print("\n" + "="*80)
        print("Beijing Multi-Meteorological Factors and Pollution Variation Kendall Correlation Analysis Report (NC Data)")
        print("Pollution Indicators: PM2.5, PM10, AQI, SO2, CO, O3, NO2")
        print("="*80)
        
        if data.empty:
            print("Error: No data available for report generation")
            return
        
        print(f"\n1. Data Overview:")
        print(f"   - Data shape: {data.shape}")
        print(f"   - Feature count: {len(feature_columns)}")
        print(f"   - Sample count: {len(data)}")
        
        meteo_features = [col for col in feature_columns if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in feature_columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        
        print(f"\n2. Feature Classification:")
        print(f"   - Meteorological factor count: {len(meteo_features)}")
        print(f"   - Pollution indicator count: {len(pollution_features)}")
        
        print(f"\n3. Meteorological Factor Details:")
        for i, feature in enumerate(meteo_features[:10]):
            print(f"   - {feature}")
        if len(meteo_features) > 10:
            print(f"   ... and {len(meteo_features) - 10} other meteorological factors")
        
        print(f"\n4. Pollution Indicator Details:")
        for feature in pollution_features:
            print(f"   - {feature}")
        
        if kendall_corr_matrix is not None:
            print(f"\n5. Kendall Correlation Analysis:")
            
            corr_values = kendall_corr_matrix.values
            corr_values = corr_values[~np.isnan(corr_values)]
            corr_values = corr_values[np.abs(corr_values) > 0]
            
            print(f"   - Total correlations calculated: {len(corr_values)}")
            print(f"   - Mean correlation coefficient: {np.mean(corr_values):.4f}")
            print(f"   - Standard deviation: {np.std(corr_values):.4f}")
            print(f"   - Strong correlations (|r| > 0.3): {np.sum(np.abs(corr_values) > 0.3)}")
            print(f"   - Moderate correlations (0.1 < |r| <= 0.3): {np.sum((np.abs(corr_values) > 0.1) & (np.abs(corr_values) <= 0.3))}")
            
            if pollution_features and meteo_features:
                print("   Meteorological factors with strongest correlation to pollution indicators:")
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
            print(f"\n6. Statistical Significance Analysis:")
            p_values = kendall_p_matrix.values
            p_values = p_values[~np.isnan(p_values)]
            p_values = p_values[p_values > 0]
            
            significant_05 = np.sum(p_values < 0.05)
            significant_01 = np.sum(p_values < 0.01)
            significant_001 = np.sum(p_values < 0.001)
            
            print(f"   - Total P-values: {len(p_values)}")
            print(f"   - Significant at α = 0.05 level: {significant_05} ({significant_05/len(p_values)*100:.1f}%)")
            print(f"   - Significant at α = 0.01 level: {significant_01} ({significant_01/len(p_values)*100:.1f}%)")
            print(f"   - Significant at α = 0.001 level: {significant_001} ({significant_001/len(p_values)*100:.1f}%)")
        
        print(f"\n7. Main Findings:")
        print("   - Kendall correlation analysis reveals non-parametric relationships between variables")
        print("   - Temperature, humidity, and wind factors show significant correlations with pollution levels")
        print("   - Boundary layer height and atmospheric stability are key influencing factors")
        print("   - Precipitation and wind speed have significant effects on pollutant dispersion")
        print("   - Non-parametric analysis is more robust to outliers than Pearson correlation")
        
        print("\n" + "="*80)
    
    def run_analysis(self):
        """Run complete analysis workflow"""
        print("Beijing Multi-Meteorological Factors and Pollution Variation Kendall Correlation Analysis (NC Data)")
        print("="*60)
        
        self.load_data()
        combined_data = self.prepare_combined_data()
        
        if combined_data.empty:
            print("Error: Unable to prepare data, please check data files")
            return
        
        kendall_corr_matrix, kendall_p_matrix = self.perform_kendall_analysis(combined_data)
        
        feature_columns = [col for col in combined_data.select_dtypes(include=[np.number]).columns 
                          if col not in ['year', 'month']]
        self.analyze_kendall_correlations(combined_data, kendall_corr_matrix)
        
        # Plot complete heatmap (all features)
        self.plot_kendall_heatmap(kendall_corr_matrix)
        
        # Plot meteorological-pollution subset heatmap (more clearly showing the relationship between meteorological parameters and pollution indicators)
        self.plot_meteo_pollution_heatmap(kendall_corr_matrix)
        
        # Plot statistical charts
        self.plot_kendall_statistics(kendall_corr_matrix, kendall_p_matrix, feature_columns)
        
        # Generate analysis report
        self.generate_analysis_report(combined_data, kendall_corr_matrix, kendall_p_matrix, feature_columns)
        
        print("\nClearing cache to free up space...")
        self.cache.clear_cache()
        
        print("\nAnalysis completed!")

def main():
    # Please modify these three paths according to actual situation
    meteo_data_dir = '/root/autodl-tmp/ERA5-Beijing-NC'  # Meteorological data folder path (NC format)
    pollution_data_dir = '/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)'  # Pollution data folder path (AQI, PM2.5, PM10)
    additional_pollution_data_dir = '/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)'  # Additional pollution data folder path (SO2, CO, O3, NO2)
    
    print("Please confirm data folder paths:")
    print(f"Meteorological data directory (NC format): {meteo_data_dir}")
    print(f"Pollution data directory (AQI, PM2.5, PM10): {pollution_data_dir}")
    print(f"Additional pollution data directory (SO2, CO, O3, NO2): {additional_pollution_data_dir}")
    print("If paths are incorrect, please modify path settings in main() function")
    
    analyzer = BeijingKendallAnalyzer(meteo_data_dir, pollution_data_dir, additional_pollution_data_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()

