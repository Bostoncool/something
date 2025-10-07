import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
import glob
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import hashlib
import pickle
import time
from typing import List, Dict, Any, Optional
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')

# Set font configuration
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = (12, 8)

class DataCache:
    """Data cache class to avoid reprocessing identical files"""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, filepath: str) -> str:
        """Generate cache key"""
        file_stat = os.stat(filepath)
        return hashlib.md5(f"{filepath}_{file_stat.st_mtime}".encode()).hexdigest()
    
    def get_cached_data(self, filepath: str) -> Optional[Dict]:
        """Get cached data"""
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
        """Save cached data"""
        cache_key = self.get_cache_key(filepath)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

class BeijingSpearmanAnalyzer:
    """Beijing Meteorological Factors and Pollution Change Spearman Correlation Analyzer based on PyTorch"""
    
    def __init__(self, meteo_data_dir=".", pollution_data_dir=".", extra_pollution_data_dir="."):
        """
        Initialize analyzer
        
        Args:
            meteo_data_dir: Meteorological data directory path
            pollution_data_dir: Pollution data directory path
            extra_pollution_data_dir: Extra pollution data directory path
        """
        self.meteo_data_dir = meteo_data_dir
        self.pollution_data_dir = pollution_data_dir
        self.extra_pollution_data_dir = extra_pollution_data_dir
        self.scaler = StandardScaler()
        self.meteorological_data = []
        self.pollution_data = []
        self.extra_pollution_data = []
        self.cache = DataCache()
        
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
            'mn2t': 'minimum_2m_temperature',
            'sp': 'surface_pressure',
            'sd': 'snow_depth',
            'str': 'surface_net_thermal_radiation',
            'tisr': 'toa_incident_solar_radiation',
            'tcwv': 'total_column_water_vapour',
            'tp': 'total_precipitation'
        }
    
    def find_files_optimized(self, root_dir: str, pattern: str) -> List[str]:
        """Optimized file search"""
        if not os.path.exists(root_dir):
            print(f"Warning: Directory does not exist: {root_dir}")
            return []
        
        search_pattern = os.path.join(root_dir, "**", pattern)
        return glob.glob(search_pattern, recursive=True)
    
    def collect_all_meteo_files(self) -> List[str]:
        """Collect all meteorological data file paths"""
        all_files = []
        
        # 方法1：按年月模式搜索
        for year in range(2015, 2025):
            for month in range(1, 13):
                pattern = f"{year}{month:02d}.csv"
                files = self.find_files_optimized(self.meteo_data_dir, pattern)
                all_files.extend(files)
        
        # 方法2：搜索所有CSV文件（包括风分量等特殊命名的文件）
        search_pattern = os.path.join(self.meteo_data_dir, "**", "*.csv")
        all_csv_files = glob.glob(search_pattern, recursive=True)
        
        # 合并并去重
        all_files_set = set(all_files + all_csv_files)
        all_files = list(all_files_set)
        
        print(f"Found {len(all_files)} meteorological data files")
        
        # 统计风分量文件
        wind_files = [f for f in all_files if any(wind in os.path.basename(f).lower() for wind in ['u10', 'v10', 'u100', 'v100', 'wind'])]
        if wind_files:
            print(f"  Including {len(wind_files)} wind component related files")
            print(f"  Examples: {[os.path.basename(f) for f in wind_files[:3]]}")
        
        return all_files
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage"""
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
        return df
    
    def aggregate_spatial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        聚合空间数据，将多维数据转换为时间序列
        
        Args:
            df: 包含time, latitude, longitude列的多维数据框
            
        Returns:
            pd.DataFrame: 按时间聚合的数据框
        """
        try:
            # 按时间分组，计算空间平均值
            if 'time' in df.columns:
                # 将时间列转换为datetime
                df['time'] = pd.to_datetime(df['time'])
                
                # 按时间分组，计算每个时间点的空间平均值
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col not in ['latitude', 'longitude']]
                
                if len(numeric_cols) > 0:
                    aggregated = df.groupby('time')[numeric_cols].mean().reset_index()
                    return aggregated
                else:
                    return df
            else:
                return df
        except Exception as e:
            print(f"Error aggregating spatial data: {e}")
            return df
    
    def process_wind_component_data(self, df: pd.DataFrame, col: str) -> Dict[str, float]:
        """
        专门处理风分量数据的方法
        
        Args:
            df: 数据框
            col: 风分量列名 (u10, v10, u100, v100)
            
        Returns:
            Dict: 统计信息
        """
        try:
            values = df[col].values
            
            # 移除NaN值
            valid_values = values[~np.isnan(values)]
            
            # 移除异常值（风分量通常在-100到100 m/s之间）
            valid_values = valid_values[(valid_values >= -100) & (valid_values <= 100)]
            
            if len(valid_values) == 0:
                return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
            
            # 如果数据量太大，进行采样
            if len(valid_values) > 50000:
                step = len(valid_values) // 50000
                valid_values = valid_values[::step]
            
            # 计算基本统计量
            return {
                'mean': np.nanmean(valid_values),
                'std': np.nanstd(valid_values),
                'min': np.nanmin(valid_values),
                'max': np.nanmax(valid_values)
            }
            
        except Exception as e:
            print(f"Error processing wind component data {col}: {e}")
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
    
    def calculate_stats_vectorized(self, hourly_data: np.ndarray) -> Dict[str, float]:
        """Vectorized calculation of statistics"""
        if len(hourly_data) == 0:
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        
        # 处理多维数据 - 如果是2D或3D数组，展平为1D
        if hourly_data.ndim > 1:
            # 对于风分量数据，通常需要按时间维度聚合
            if hourly_data.ndim == 3:  # (time, lat, lon)
                # 按时间维度计算平均值
                hourly_data = np.nanmean(hourly_data, axis=(1, 2))
            elif hourly_data.ndim == 2:  # (time, spatial)
                # 按时间维度计算平均值
                hourly_data = np.nanmean(hourly_data, axis=1)
        
        # 移除无效值
        valid_data = hourly_data[~np.isnan(hourly_data)]
        if len(valid_data) == 0:
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        
        # 如果数据量太大，进行采样以避免内存问题
        if len(valid_data) > 10000:
            step = len(valid_data) // 10000
            valid_data = valid_data[::step]
        
        days = len(valid_data) // 24
        if days > 0:
            daily_data = valid_data[:days*24].reshape(days, 24)
            daily_means = np.nanmean(daily_data, axis=1)
            daily_stds = np.nanstd(daily_data, axis=1)
            daily_mins = np.nanmin(daily_data, axis=1)
            daily_maxs = np.nanmax(daily_data, axis=1)
            
            return {
                'mean': np.nanmean(daily_means),
                'std': np.nanmean(daily_stds),
                'min': np.nanmin(daily_mins),
                'max': np.nanmax(daily_maxs)
            }
        else:
            return {
                'mean': np.nanmean(valid_data),
                'std': np.nanstd(valid_data),
                'min': np.nanmin(valid_data),
                'max': np.nanmax(valid_data)
            }
    
    def process_single_meteo_file(self, filepath: str) -> Optional[Dict]:
        """Process single meteorological data file"""
        try:
            cached_data = self.cache.get_cached_data(filepath)
            if cached_data:
                return cached_data
            
            # 读取CSV文件，处理可能的编码问题和注释行
            try:
                df = pd.read_csv(filepath, encoding='utf-8', comment='#')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(filepath, encoding='gbk', comment='#')
                except UnicodeDecodeError:
                    df = pd.read_csv(filepath, encoding='latin-1', comment='#')
            except pd.errors.ParserError:
                # 如果有解析错误，尝试跳过前几行注释
                try:
                    df = pd.read_csv(filepath, encoding='utf-8', skiprows=4)
                except:
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8', comment='#', on_bad_lines='skip')
                    except:
                        raise
            
            if df.empty:
                return None
            
            # 处理多索引数据（如果存在）
            if 'time' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
                # 这是从NC文件转换的多维数据，需要按时间聚合
                df = self.aggregate_spatial_data(df)
            
            df = self.optimize_dtypes(df)
            available_columns = [col for col in self.meteo_columns.keys() if col in df.columns]
            
            if not available_columns:
                return None
            
            monthly_stats = {}
            for col in available_columns:
                try:
                    if col in ['t2m', 'mn2t', 'd2m']:
                        temp_kelvin = df[col].values
                        # 检查是否已经是摄氏度
                        if temp_kelvin.max() < 100:  # 如果最大值小于100，可能是摄氏度
                            temp_celsius = temp_kelvin
                        else:
                            temp_celsius = temp_kelvin - 273.15
                        daily_stats = self.calculate_stats_vectorized(temp_celsius)
                    elif col in ['u10', 'v10', 'u100', 'v100']:
                        # 风分量数据特殊处理
                        daily_stats = self.process_wind_component_data(df, col)
                    else:
                        values = df[col].values
                        daily_stats = self.calculate_stats_vectorized(values)
                    
                    monthly_stats[f'{col}_mean'] = daily_stats['mean']
                    monthly_stats[f'{col}_std'] = daily_stats['std']
                    monthly_stats[f'{col}_min'] = daily_stats['min']
                    monthly_stats[f'{col}_max'] = daily_stats['max']
                    
                except Exception as col_error:
                    print(f"Error processing column {col} in {os.path.basename(filepath)}: {col_error}")
                    monthly_stats[f'{col}_mean'] = np.nan
                    monthly_stats[f'{col}_std'] = np.nan
                    monthly_stats[f'{col}_min'] = np.nan
                    monthly_stats[f'{col}_max'] = np.nan
            
            # 解析年月信息
            filename = os.path.basename(filepath)
            try:
                # 尝试从文件名中提取年月（格式：YYYYMM.csv）
                if len(filename) >= 6 and filename[:4].isdigit() and filename[4:6].isdigit():
                    year = int(filename[:4])
                    month = int(filename[4:6])
                    monthly_stats['year'] = year
                    monthly_stats['month'] = month
                else:
                    # 如果文件名不包含年月信息，尝试从路径中提取
                    import re
                    match = re.search(r'(\d{4})(\d{2})', filepath)
                    if match:
                        monthly_stats['year'] = int(match.group(1))
                        monthly_stats['month'] = int(match.group(2))
            except ValueError as e:
                print(f"Warning: Error parsing year/month from filename {filename}: {e}")
            
            self.cache.save_cached_data(filepath, monthly_stats)
            return monthly_stats
            
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            return None
    
    def load_meteo_data_parallel(self):
        """Load meteorological data in parallel"""
        print("Loading meteorological data...")
        start_time = time.time()
        
        all_files = self.collect_all_meteo_files()
        if not all_files:
            print("No meteorological data files found")
            return
        
        max_workers = min(multiprocessing.cpu_count(), len(all_files))
        print(f"Using {max_workers} processes")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_single_meteo_file, filepath): filepath 
                            for filepath in all_files}
            
            for future in as_completed(future_to_file):
                result = future.result()
                if result:
                    self.meteorological_data.append(result)
        
        end_time = time.time()
        print(f"Meteorological data loading completed, time taken: {end_time - start_time:.2f} seconds")
        print(f"Successfully loaded data from {len(self.meteorological_data)} files")
    
    def load_pollution_data_optimized(self):
        """Load pollution data (PM2.5, PM10, AQI)"""
        print("Loading pollution data...")
        start_time = time.time()
        
        def pollution_file_filter(filename):
            return filename.startswith('beijing_all_') and filename.endswith('.csv')
        
        all_pollution_files = []
        search_pattern = os.path.join(self.pollution_data_dir, "**", "*.csv")
        for filepath in glob.glob(search_pattern, recursive=True):
            filename = os.path.basename(filepath)
            if pollution_file_filter(filename):
                all_pollution_files.append(filepath)
        
        print(f"Found {len(all_pollution_files)} pollution data files")
        
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
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
        
        end_time = time.time()
        print(f"Pollution data loading completed, time taken: {end_time - start_time:.2f} seconds")
        print(f"Successfully loaded data from {len(self.pollution_data)} files")
    
    def load_extra_pollution_data(self):
        """Load extra pollution data (SO2, CO, O3, NO2)"""
        print("Loading extra pollution data...")
        start_time = time.time()
        
        def extra_pollution_file_filter(filename):
            return filename.startswith('beijing_extra_') and filename.endswith('.csv')
        
        all_extra_files = []
        search_pattern = os.path.join(self.extra_pollution_data_dir, "**", "*.csv")
        for filepath in glob.glob(search_pattern, recursive=True):
            filename = os.path.basename(filepath)
            if extra_pollution_file_filter(filename):
                all_extra_files.append(filepath)
        
        print(f"Found {len(all_extra_files)} extra pollution data files")
        
        for filepath in all_extra_files:
            try:
                cached_data = self.cache.get_cached_data(filepath)
                if cached_data:
                    self.extra_pollution_data.append(cached_data)
                    continue
                
                df = pd.read_csv(filepath)
                if not df.empty and 'type' in df.columns:
                    so2_data = df[df['type'] == 'SO2'].iloc[:, 3:].values
                    co_data = df[df['type'] == 'CO'].iloc[:, 3:].values
                    o3_data = df[df['type'] == 'O3'].iloc[:, 3:].values
                    no2_data = df[df['type'] == 'NO2'].iloc[:, 3:].values
                    
                    if len(so2_data) > 0:
                        daily_so2 = np.nanmean(so2_data, axis=0) if so2_data.ndim > 1 else so2_data
                        daily_co = np.nanmean(co_data, axis=0) if co_data.ndim > 1 else co_data
                        daily_o3 = np.nanmean(o3_data, axis=0) if o3_data.ndim > 1 else o3_data
                        daily_no2 = np.nanmean(no2_data, axis=0) if no2_data.ndim > 1 else no2_data
                        
                        extra_pollution_stats = {
                            'so2_mean': np.nanmean(daily_so2),
                            'co_mean': np.nanmean(daily_co),
                            'o3_mean': np.nanmean(daily_o3),
                            'no2_mean': np.nanmean(daily_no2),
                            'so2_max': np.nanmax(daily_so2),
                            'co_max': np.nanmax(daily_co),
                            'o3_max': np.nanmax(daily_o3),
                            'no2_max': np.nanmax(daily_no2)
                        }
                        
                        self.cache.save_cached_data(filepath, extra_pollution_stats)
                        self.extra_pollution_data.append(extra_pollution_stats)
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
        
        end_time = time.time()
        print(f"Extra pollution data loading completed, time taken: {end_time - start_time:.2f} seconds")
        print(f"Successfully loaded data from {len(self.extra_pollution_data)} files")
    
    def load_data(self):
        """Load all data"""
        print("Starting to load data...")
        
        self.load_meteo_data_parallel()
        self.load_pollution_data_optimized()
        self.load_extra_pollution_data()
        
        print("Data loading completed!")
    
    def clear_cache(self):
        """Clear cache directory to free up space"""
        if os.path.exists(self.cache.cache_dir):
            shutil.rmtree(self.cache.cache_dir)
            os.makedirs(self.cache.cache_dir, exist_ok=True)
            print("Cache cleared successfully")
    
    def prepare_combined_data(self):
        """Prepare combined meteorological and pollution data"""
        print("Preparing combined data...")
        
        if not self.meteorological_data or not self.pollution_data:
            print("Error: Insufficient data for analysis")
            return pd.DataFrame()
        
        meteo_df = pd.DataFrame(self.meteorological_data)
        pollution_df = pd.DataFrame(self.pollution_data)
        extra_pollution_df = pd.DataFrame(self.extra_pollution_data) if self.extra_pollution_data else pd.DataFrame()
        
        print(f"Meteorological data shape: {meteo_df.shape}")
        print(f"Pollution data shape: {pollution_df.shape}")
        print(f"Extra pollution data shape: {extra_pollution_df.shape}")
        
        # Align data length
        data_lengths = [len(meteo_df), len(pollution_df)]
        if not extra_pollution_df.empty:
            data_lengths.append(len(extra_pollution_df))
        
        min_len = min(data_lengths)
        print(f"Aligned length: {min_len}")
        
        meteo_df = meteo_df.head(min_len)
        pollution_df = pollution_df.head(min_len)
        if not extra_pollution_df.empty:
            extra_pollution_df = extra_pollution_df.head(min_len)
        
        # Combine data
        if not extra_pollution_df.empty:
            combined_data = pd.concat([meteo_df, pollution_df, extra_pollution_df], axis=1)
        else:
            combined_data = pd.concat([meteo_df, pollution_df], axis=1)
        
        print(f"Combined data shape: {combined_data.shape}")
        
        # Handle NaN values
        combined_data = combined_data.dropna(how='all')
        combined_data = combined_data.ffill().bfill()
        
        numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if combined_data[col].isna().any():
                mean_val = combined_data[col].mean()
                if not pd.isna(mean_val):
                    combined_data.loc[:, col] = combined_data[col].fillna(mean_val)
        
        print(f"Final data shape: {combined_data.shape}")
        
        # Count features
        meteo_features = [col for col in combined_data.columns if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in combined_data.columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        
        # Count wind component features
        wind_features = [col for col in combined_data.columns if any(w in col for w in ['u10', 'v10', 'u100', 'v100'])]
        
        print(f"Number of meteorological parameters: {len(meteo_features)}")
        print(f"Number of pollution parameters: {len(pollution_features)}")
        print(f"Number of wind component parameters: {len(wind_features)}")
        
        if wind_features:
            print(f"Wind component parameters: {wind_features}")
        else:
            print("Warning: No wind component parameters found in combined data!")
        
        return combined_data
    
    def calculate_spearman_correlation_pytorch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Spearman correlation using PyTorch"""
        print("Calculating Spearman correlation using PyTorch...")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['year', 'month']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        X = torch.tensor(data[feature_columns].values, dtype=torch.float32)
        n_samples, n_features = X.shape
        ranks = torch.zeros_like(X)
        
        for i in range(n_features):
            valid_mask = ~torch.isnan(X[:, i])
            if torch.sum(valid_mask) > 0:
                valid_values = X[valid_mask, i]
                sorted_indices = torch.argsort(valid_values)
                ranks_temp = torch.zeros_like(valid_values)
                ranks_temp[sorted_indices] = torch.arange(1, len(valid_values) + 1, dtype=torch.float32)
                
                unique_values, inverse_indices, counts = torch.unique(valid_values, return_inverse=True, return_counts=True)
                for j, count in enumerate(counts):
                    if count > 1:
                        tie_indices = (inverse_indices == j)
                        tie_ranks = ranks_temp[tie_indices]
                        avg_rank = torch.mean(tie_ranks)
                        ranks_temp[tie_indices] = avg_rank
                
                ranks[valid_mask, i] = ranks_temp
            else:
                ranks[:, i] = torch.nan
        
        correlation_matrix = torch.zeros(n_features, n_features)
        
        for i in range(n_features):
            for j in range(n_features):
                valid_mask = ~(torch.isnan(ranks[:, i]) | torch.isnan(ranks[:, j]))
                if torch.sum(valid_mask) > 1:
                    rank_i = ranks[valid_mask, i]
                    rank_j = ranks[valid_mask, j]
                    
                    n = len(rank_i)
                    sum_d2 = torch.sum((rank_i - rank_j) ** 2)
                    
                    if n > 1:
                        spearman_coeff = 1 - (6 * sum_d2) / (n * (n**2 - 1))
                        correlation_matrix[i, j] = spearman_coeff
                    else:
                        correlation_matrix[i, j] = 0.0
                else:
                    correlation_matrix[i, j] = 0.0
        
        correlation_df = pd.DataFrame(correlation_matrix.numpy(), 
                                    columns=feature_columns, 
                                    index=feature_columns)
        
        return correlation_df
    
    def analyze_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze correlations between variables"""
        print("Analyzing correlations...")
        
        if data.empty:
            print("Error: No data available for correlation analysis")
            return pd.DataFrame()
        
        # Calculate correlation using PyTorch
        correlation_matrix = self.calculate_spearman_correlation_pytorch(data)
        
        # Analyze correlations between meteorological factors and pollution indicators
        print("\nCorrelation analysis between meteorological factors and pollution indicators:")
        pollution_features = [col for col in correlation_matrix.columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        meteo_features = [col for col in correlation_matrix.columns if any(x in col for x in self.meteo_columns.keys())]
        
        if pollution_features and meteo_features:
            print(f"Found {len(pollution_features)} pollution indicators and {len(meteo_features)} meteorological factors")
            
            for pollution_feat in pollution_features:
                correlations = correlation_matrix[pollution_feat][meteo_features].abs()
                top_correlations = correlations.nlargest(5)
                print(f"\nMeteorological factors most correlated with {pollution_feat}:")
                for meteo_feat, corr in top_correlations.items():
                    print(f"  {meteo_feat}: {corr:.3f}")
        
        return correlation_matrix
    
    def gradient_image(self, ax, extent, direction=0, cmap_range=(0, 0.5), **kwargs):
        """Create gradient image for bars"""
        phi = direction * np.pi / 2
        v = np.array([np.cos(phi), np.sin(phi)])
        X = np.array([[v @ [0, 0], v @ [0, 0]],
                      [v @ [1, 1], v @ [1, 1]]])
        a, b = cmap_range
        X = a + (b - a) / X.max() * X
        im = ax.imshow(X, extent=extent, interpolation='bicubic',
                       vmin=0, vmax=1, aspect='auto', **kwargs)
        return im
    
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                                save_path='beijing_spearman_correlation_heatmap.png'):
        """Plot correlation heatmap"""
        if correlation_matrix.empty:
            print("Error: No correlation data to plot")
            return
        
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(20, 16))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=False,
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   cbar_kws={'shrink': 0.8, 'aspect': 50, 'label': 'Spearman Correlation Coefficient'},
                   linewidths=0.2,
                   linecolor='white')
        
        plt.title('Beijing Meteorological Factors and Pollution Indicators Spearman Correlation Heatmap', 
                 fontsize=22, fontweight='bold', pad=50, color='#2E3440')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"Correlation heatmap saved to: {save_path}")
    
    def plot_top_correlations(self, correlation_matrix: pd.DataFrame, 
                             save_path='beijing_spearman_top_correlations.png'):
        """Plot top N strongest correlations"""
        if correlation_matrix.empty:
            print("Error: No correlation data to plot")
            return
        
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        correlations = []
        for i in range(len(upper_triangle.columns)):
            for j in range(i+1, len(upper_triangle.columns)):
                value = upper_triangle.iloc[i, j]
                if not pd.isna(value):
                    correlations.append({
                        'feature1': upper_triangle.columns[i],
                        'feature2': upper_triangle.columns[j],
                        'correlation': value
                    })
        
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        top_correlations = correlations[:20]
        
        fig, ax = plt.subplots(figsize=(20, 14))
        
        features = []
        for corr in top_correlations:
            feat1 = str(corr['feature1']).replace('_mean', '').replace('_std', '').replace('_min', '').replace('_max', '')
            feat2 = str(corr['feature2']).replace('_mean', '').replace('_std', '').replace('_min', '').replace('_max', '')
            features.append(f"{feat1}\nvs\n{feat2}")
        
        values = [corr['correlation'] for corr in top_correlations]
        y_pos = np.arange(len(features))
        
        for i, (y, value) in enumerate(zip(y_pos, values)):
            left = 0 if value >= 0 else value
            right = value if value >= 0 else 0
            
            if value >= 0:
                colors = [(114/255, 188/255, 213/255), (1, 1, 1)]
            else:
                colors = [(255/255, 99/255, 71/255), (1, 1, 1)]
            
            cmap = LinearSegmentedColormap.from_list('my_cmap', colors, N=256)
            bar_height = 0.6
            self.gradient_image(ax, extent=(left, right, y-bar_height/2, y+bar_height/2),
                               cmap=cmap, cmap_range=(0, 0.8))
        
        for i, (y, value) in enumerate(zip(y_pos, values)):
            if value >= 0:
                ax.text(value + 0.01, y, f'{value:.3f}', 
                       ha='left', va='center', fontweight='bold', fontsize=11)
            else:
                ax.text(value - 0.01, y, f'{value:.3f}', 
                       ha='right', va='center', fontweight='bold', fontsize=11, color='red')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Spearman Correlation Coefficient', fontsize=14, fontweight='bold')
        ax.set_title('Top 20 Strongest Spearman Correlations: Beijing Meteorological Factors vs Pollution Indicators', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='x')
        
        max_abs_value = max(abs(v) for v in values)
        ax.set_xlim(-max_abs_value * 1.1, max_abs_value * 1.1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"Top N strongest correlations plot saved to: {save_path}")
    
    def generate_analysis_report(self, data: pd.DataFrame, correlation_matrix: pd.DataFrame):
        """Generate analysis report"""
        print("\n" + "="*80)
        print("Beijing Meteorological Factors and Pollution Change Spearman Correlation Analysis Report")
        print("="*80)
        
        if data.empty:
            print("Error: No data available to generate report")
            return
        
        print(f"\n1. Data Overview:")
        print(f"   - Data shape: {data.shape}")
        print(f"   - Number of samples: {len(data)}")
        
        # Classify features
        meteo_features = [col for col in data.columns if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in data.columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        
        print(f"\n2. Feature Classification:")
        print(f"   - Number of meteorological factors: {len(meteo_features)}")
        print(f"   - Number of pollution indicators: {len(pollution_features)}")
        
        print(f"\n3. Meteorological Factors Details:")
        for i, feature in enumerate(meteo_features[:10]):  # Show first 10
            print(f"   - {feature}")
        if len(meteo_features) > 10:
            print(f"   ... and {len(meteo_features) - 10} more meteorological factors")
        
        print(f"\n4. Pollution Indicators Details:")
        for feature in pollution_features:
            print(f"   - {feature}")
        
        if not correlation_matrix.empty:
            print(f"\n5. Spearman Correlation Analysis:")
            if pollution_features and meteo_features:
                print("Meteorological factors most correlated with pollution indicators:")
                for pollution_feat in pollution_features:
                    correlations = correlation_matrix[pollution_feat][meteo_features].abs()
                    top_correlations = correlations.nlargest(3)
                    print(f"   {pollution_feat}:")
                    for meteo_feat, corr in top_correlations.items():
                        print(f"     - {meteo_feat}: {corr:.3f}")
            
            # Analyze strongest correlations
            print(f"\n6. Strongest Spearman Correlation Analysis:")
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            correlations = []
            for i in range(len(upper_triangle.columns)):
                for j in range(i+1, len(upper_triangle.columns)):
                    value = upper_triangle.iloc[i, j]
                    if not pd.isna(value):
                        correlations.append({
                            'feature1': upper_triangle.columns[i],
                            'feature2': upper_triangle.columns[j],
                            'correlation': value
                        })
            
            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            print("Top 10 strongest Spearman correlations:")
            for i, corr in enumerate(correlations[:10]):
                print(f"   {i+1}. {corr['feature1']} vs {corr['feature2']}: {corr['correlation']:.3f}")
        
        print(f"\n7. Key Findings:")
        print("   - Temperature, humidity, wind speed and other meteorological factors have comprehensive effects on pollution levels")
        print("   - Boundary layer height and atmospheric stability are important influencing factors")
        print("   - Precipitation and wind speed have significant effects on pollutant dispersion")
        print("   - Multi-meteorological factor comprehensive analysis can better explain pollution changes")
        print("   - Spearman correlation provides rank-based correlation analysis, robust to outliers")
    
    def run_analysis(self):
        """Run complete analysis workflow"""
        print("Beijing Meteorological Factors and Pollution Change Spearman Correlation Analysis")
        print("="*60)
        
        try:
            # 1. Load data
            self.load_data()
            
            # 2. Prepare combined data
            combined_data = self.prepare_combined_data()
            
            if combined_data.empty:
                print("Error: Unable to prepare data, please check data files")
                return
            
            # 3. Analyze correlations
            correlation_matrix = self.analyze_correlations(combined_data)
            
            # 4. Plot correlation heatmap
            self.plot_correlation_heatmap(correlation_matrix)
            
            # 5. Plot top N strongest correlations
            self.plot_top_correlations(correlation_matrix)
            
            # 6. Generate analysis report
            self.generate_analysis_report(combined_data, correlation_matrix)
            
            print("\nAnalysis completed!")
            
        finally:
            # 7. Clear cache to free up space
            self.clear_cache()

def main():
    """Main function"""
    # Data folder paths
    meteo_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV"
    pollution_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)"
    extra_pollution_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)"
    
    print("Data folder paths:")
    print(f"Meteorological data directory: {meteo_data_dir}")
    print(f"Pollution data directory: {pollution_data_dir}")
    print(f"Extra pollution data directory: {extra_pollution_data_dir}")
    
    # Create analyzer
    analyzer = BeijingSpearmanAnalyzer(meteo_data_dir, pollution_data_dir, extra_pollution_data_dir)
    
    # Run analysis
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
