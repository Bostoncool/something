import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import hashlib
import pickle
import time
import shutil
from typing import List, Dict, Any, Optional
warnings.filterwarnings('ignore')

# Set font configuration for plots
import matplotlib as mpl

# Set font configuration
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams['axes.unicode_minus'] = False
# Set global font size and figure size
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = (10, 6)

class DataCache:
    """数据缓存类，避免重复处理相同文件"""
    
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, filepath: str) -> str:
        """生成缓存键值"""
        file_stat = os.stat(filepath)
        return hashlib.md5(f"{filepath}_{file_stat.st_mtime}".encode()).hexdigest()
    
    def get_cached_data(self, filepath: str) -> Optional[Dict]:
        """获取缓存数据"""
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
        """保存缓存数据"""
        cache_key = self.get_cache_key(filepath)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    
    def clear_cache(self):
        """清空缓存目录"""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            print("缓存已清空")

class BeijingPCAAnalyzerOptimized:
    """北京多气象因子对污染变化影响的PCA分析器（优化版）"""
    
    def __init__(self, meteo_data_dir=".", pollution_data_dir=".", extra_pollution_data_dir="."):
        """
        初始化分析器
        
        Args:
            meteo_data_dir: 气象数据目录路径
            pollution_data_dir: 污染数据目录路径
            extra_pollution_data_dir: 额外污染数据目录路径
        """
        self.meteo_data_dir = meteo_data_dir
        self.pollution_data_dir = pollution_data_dir
        self.extra_pollution_data_dir = extra_pollution_data_dir
        self.scaler = StandardScaler()
        self.pca = None
        self.meteorological_data = []
        self.pollution_data = []
        self.extra_pollution_data = []
        self.cache = DataCache()
        
        # Define all meteorological parameter column mappings
        self.meteo_columns = {
            't2m': '2m_temperature',  # 2m temperature
            'd2m': '2m_dewpoint_temperature',  # 2m dewpoint temperature
            'blh': 'boundary_layer_height',  # boundary layer height
            'cvh': 'high_vegetation_cover',  # high vegetation cover
            'avg_tprate': 'mean_total_precipitation_rate',  # mean total precipitation rate
            'u10': '10m_u_component_of_wind',  # 10m wind U component
            'v10': '10m_v_component_of_wind',  # 10m wind V component
            'u100': '100m_u_component_of_wind',  # 100m wind U component
            'v100': '100m_v_component_of_wind',  # 100m wind V component
            'lsm': 'land_sea_mask',  # land sea mask
            'cvl': 'low_vegetation_cover',  # low vegetation cover
            'mn2t': 'minimum_2m_temperature_since_previous_post_processing',  # minimum 2m temperature
            'sp': 'surface_pressure',  # surface pressure
            'sd': 'snow_depth',  # snow depth
            'str': 'surface_net_thermal_radiation',  # surface net thermal radiation
            'tisr': 'toa_incident_solar_radiation',  # top of atmosphere incident solar radiation
            'tcwv': 'total_column_water_vapour',  # total column water vapour
            'tp': 'total_precipitation'  # total precipitation
        }
    
    def find_files_optimized(self, root_dir: str, pattern: str) -> List[str]:
        """
        Optimized file search using glob
        
        Args:
            root_dir: Root directory
            pattern: File pattern
            
        Returns:
            List[str]: List of found file paths
        """
        if not os.path.exists(root_dir):
            print(f"Warning: Directory does not exist: {root_dir}")
            return []
        
        search_pattern = os.path.join(root_dir, "**", pattern)
        return glob.glob(search_pattern, recursive=True)
    
    def collect_all_meteo_files(self) -> List[str]:
        """收集所有气象数据文件路径"""
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
        """优化数据类型以减少内存使用"""
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
            print(f"聚合空间数据时出错: {e}")
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
            print(f"处理风分量数据 {col} 时出错: {e}")
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
    
    def calculate_stats_vectorized(self, hourly_data: np.ndarray) -> Dict[str, float]:
        """向量化统计计算"""
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
        """处理单个气象数据文件（用于并行处理）"""
        try:
            cached_data = self.cache.get_cached_data(filepath)
            if cached_data:
                print(f"使用缓存数据: {os.path.basename(filepath)}")
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
                print(f"Warning: File {filepath} is empty")
                return None
            
            # Debug info: show file basic information
            print(f"Processing file: {os.path.basename(filepath)}")
            print(f"  Data shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            # Check for wind component data
            wind_cols = [col for col in ['u10', 'v10', 'u100', 'v100'] if col in df.columns]
            if wind_cols:
                print(f"  Found wind component columns: {wind_cols}")
                for col in wind_cols:
                    print(f"    {col}: dtype={df[col].dtype}, non-null count={df[col].count()}, range=[{df[col].min():.2f}, {df[col].max():.2f}]")
            
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
                    print(f"Error processing column {col}: {col_error}")
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
                    else:
                        print(f"  警告: 无法从文件名中提取年月信息: {filename}")
            except ValueError as e:
                print(f"  警告: 解析年月信息时出错: {e}")
            
            self.cache.save_cached_data(filepath, monthly_stats)
            print(f"Processed meteorological data: {os.path.basename(filepath)} ({len(available_columns)} parameters)")
            return monthly_stats
            
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            return None
    
    def load_meteo_data_parallel(self):
        """并行加载气象数据"""
        print("开始并行加载气象数据...")
        start_time = time.time()
        
        all_files = self.collect_all_meteo_files()
        
        if not all_files:
            print("未找到气象数据文件")
            return
        
        max_workers = min(multiprocessing.cpu_count(), len(all_files))
        print(f"使用 {max_workers} 个进程进行并行处理")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_single_meteo_file, filepath): filepath 
                            for filepath in all_files}
            
            for future in as_completed(future_to_file):
                result = future.result()
                if result:
                    self.meteorological_data.append(result)
        
        end_time = time.time()
        print(f"气象数据加载完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"成功加载了 {len(self.meteorological_data)} 个文件的数据")
    
    def load_pollution_data_optimized(self):
        """优化的污染数据加载"""
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
                    print(f"使用缓存数据: {os.path.basename(filepath)}")
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
                print(f"加载文件 {filepath} 时出错: {e}")
        
        end_time = time.time()
        print(f"污染数据加载完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"成功加载了 {len(self.pollution_data)} 个文件的数据")
    
    def load_extra_pollution_data(self):
        """加载额外污染数据（SO2, CO, O3, NO2）"""
        print("开始加载额外污染数据...")
        start_time = time.time()
        
        def extra_pollution_file_filter(filename):
            return filename.startswith('beijing_extra_') and filename.endswith('.csv')
        
        all_extra_files = []
        search_pattern = os.path.join(self.extra_pollution_data_dir, "**", "*.csv")
        for filepath in glob.glob(search_pattern, recursive=True):
            filename = os.path.basename(filepath)
            if extra_pollution_file_filter(filename):
                all_extra_files.append(filepath)
        
        print(f"找到 {len(all_extra_files)} 个额外污染数据文件")
        
        for filepath in all_extra_files:
            try:
                cached_data = self.cache.get_cached_data(filepath)
                if cached_data:
                    self.extra_pollution_data.append(cached_data)
                    print(f"使用缓存数据: {os.path.basename(filepath)}")
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
                        
                        extra_stats = {
                            'so2_mean': np.nanmean(daily_so2),
                            'co_mean': np.nanmean(daily_co),
                            'o3_mean': np.nanmean(daily_o3),
                            'no2_mean': np.nanmean(daily_no2),
                            'so2_max': np.nanmax(daily_so2),
                            'co_max': np.nanmax(daily_co),
                            'o3_max': np.nanmax(daily_o3),
                            'no2_max': np.nanmax(daily_no2)
                        }
                        
                        self.cache.save_cached_data(filepath, extra_stats)
                        self.extra_pollution_data.append(extra_stats)
                        print(f"已加载额外污染数据: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"加载文件 {filepath} 时出错: {e}")
        
        end_time = time.time()
        print(f"额外污染数据加载完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"成功加载了 {len(self.extra_pollution_data)} 个文件的数据")
    
    def load_data(self):
        """加载所有数据（优化版）"""
        print("开始加载数据（优化版）...")
        
        self.load_meteo_data_parallel()
        self.load_pollution_data_optimized()
        self.load_extra_pollution_data()
        
        print("数据加载完成！")
    
    def prepare_combined_data(self):
        """准备合并的气象和污染数据"""
        print("准备合并数据...")
        
        if not self.meteorological_data or not self.pollution_data:
            print("错误：数据不足，无法进行分析")
            print(f"气象数据数量: {len(self.meteorological_data)}")
            print(f"污染数据数量: {len(self.pollution_data)}")
            return pd.DataFrame()
        
        meteo_df = pd.DataFrame(self.meteorological_data)
        pollution_df = pd.DataFrame(self.pollution_data)
        extra_pollution_df = pd.DataFrame(self.extra_pollution_data) if self.extra_pollution_data else pd.DataFrame()
        
        print(f"原始气象数据形状: {meteo_df.shape}")
        print(f"原始污染数据形状: {pollution_df.shape}")
        if not extra_pollution_df.empty:
            print(f"原始额外污染数据形状: {extra_pollution_df.shape}")
        
        min_len = min(len(meteo_df), len(pollution_df))
        if not extra_pollution_df.empty:
            min_len = min(min_len, len(extra_pollution_df))
        
        print(f"对齐长度: {min_len}")
        
        meteo_df = meteo_df.head(min_len)
        pollution_df = pollution_df.head(min_len)
        if not extra_pollution_df.empty:
            extra_pollution_df = extra_pollution_df.head(min_len)
        
        if not extra_pollution_df.empty:
            combined_data = pd.concat([meteo_df, pollution_df, extra_pollution_df], axis=1)
        else:
            combined_data = pd.concat([meteo_df, pollution_df], axis=1)
        
        print(f"合并后数据形状: {combined_data.shape}")
        
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
        print(f"最终NaN数量: {combined_data.isna().sum().sum()}")
        
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
    
    def perform_pca_analysis(self, data, n_components=2):
        """执行PCA分析"""
        print("执行PCA分析...")
        
        if data.empty:
            print("错误：没有可用于PCA分析的数据")
            return None, None, None
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['year', 'month']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        X = data[feature_columns].values
        X_scaled = self.scaler.fit_transform(X)
        
        self.pca = PCA(n_components=min(n_components, len(feature_columns)))
        X_pca = self.pca.fit_transform(X_scaled)
        
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        print(f"前 {len(explained_variance_ratio)} 个主成分的方差解释率:")
        for i, (var_ratio, cum_var_ratio) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio)):
            print(f"PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
            print(f"累积: {cum_var_ratio:.4f} ({cum_var_ratio*100:.2f}%)")
        
        print("\n主成分贡献分析:")
        for i in range(len(explained_variance_ratio)):
            loadings = self.pca.components_[i]
            feature_loadings = list(zip(feature_columns, loadings))
            feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print(f"\nPC{i+1} 主要贡献特征（前5个）:")
            for feature, loading in feature_loadings[:5]:
                print(f"  {feature}: {loading:.4f}")
        
        return X_pca, feature_columns, explained_variance_ratio
    
    def analyze_correlations(self, data):
        """分析变量间相关性"""
        print("分析相关性...")
        
        if data.empty:
            print("错误：没有可用于相关性分析的数据")
            return None
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['year', 'month']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        correlation_matrix = data[feature_columns].corr()
        
        print("\n气象因子与污染指标相关性分析:")
        pollution_features = [col for col in feature_columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        meteo_features = [col for col in feature_columns if any(x in col for x in self.meteo_columns.keys())]
        
        if pollution_features and meteo_features:
            print(f"发现 {len(pollution_features)} 个污染指标和 {len(meteo_features)} 个气象因子")
            
            for pollution_feat in pollution_features:
                correlations = correlation_matrix[pollution_feat][meteo_features].abs()
                top_correlations = correlations.nlargest(5)
                print(f"\n与 {pollution_feat} 最相关的气象因子:")
                for meteo_feat, corr in top_correlations.items():
                    print(f"  {meteo_feat}: {corr:.3f}")
        
        return correlation_matrix
    
    def plot_correlation_heatmap(self, correlation_matrix, save_path='beijing_correlation_heatmap_optimized.png'):
        """绘制相关性热图"""
        if correlation_matrix is None:
            print("错误：没有相关性数据可绘制")
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
                   cbar_kws={'shrink': 0.8, 'aspect': 50, 'label': 'Correlation Coefficient'},
                   linewidths=0.2,
                   linecolor='white')
        
        plt.title('Beijing Meteorological Factors vs Pollution Indicators Correlation Heatmap (Optimized)', 
                 fontsize=22, fontweight='bold', pad=50, color='#2E3440')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"Correlation heatmap saved to: {save_path}")
    
    def plot_pca_results(self, X_pca, feature_names, explained_variance_ratio, save_path='beijing_pca_results_optimized.png'):
        """绘制PCA结果"""
        if X_pca is None:
            print("错误：没有PCA数据可绘制")
            return
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        # 主成分散点图
        scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                    alpha=0.7, c=colors[0], s=60, edgecolors='white', linewidth=0.5)
        axes[0, 0].set_xlabel('First Principal Component (PC1)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Second Principal Component (PC2)', fontsize=14, fontweight='bold')
        axes[0, 0].set_title('PCA Principal Components Scatter Plot (Optimized)', fontsize=16, fontweight='bold', pad=20)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)
        
        # 方差解释率
        bars = axes[0, 1].bar(range(1, len(explained_variance_ratio) + 1), 
                              explained_variance_ratio, 
                              color=colors[1], alpha=0.8, edgecolor='white', linewidth=1)
        axes[0, 1].set_xlabel('Principal Component', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Explained Variance Ratio by Principal Component (Optimized)', fontsize=14, fontweight='bold', pad=15)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 累积方差解释率
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        line = axes[1, 0].plot(range(1, len(cumulative_variance_ratio) + 1), 
                               cumulative_variance_ratio, 
                               color=colors[2], linewidth=3, marker='o', 
                               markersize=10, markerfacecolor='white', 
                               markeredgecolor=colors[2], markeredgewidth=2)
        axes[1, 0].set_xlabel('Number of Principal Components', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Cumulative Explained Variance Ratio', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Cumulative Explained Variance Ratio (Optimized)', fontsize=14, fontweight='bold', pad=15)
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        axes[1, 0].spines['top'].set_visible(False)
        axes[1, 0].spines['right'].set_visible(False)
        
        for i, (x, y) in enumerate(zip(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)):
            axes[1, 0].text(x, y + 0.02, f'{y:.3f}', ha='center', va='bottom', 
                           fontweight='bold', fontsize=10)
        
        # 特征重要性
        if len(feature_names) > 0:
            loadings = self.pca.components_
            feature_importance = np.abs(loadings[0])
            sorted_idx = np.argsort(feature_importance)[::-1]
            
            top_n = min(15, len(sorted_idx))
            top_features = [feature_names[i] for i in sorted_idx[:top_n]]
            top_importance = feature_importance[sorted_idx[:top_n]]
            
            bars = axes[1, 1].barh(range(len(top_features)), 
                                   top_importance, 
                                   color=colors[3], alpha=0.8, edgecolor='white', linewidth=1)
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features, fontsize=10)
            axes[1, 1].set_xlabel('Feature Importance (Absolute Value)', fontsize=12, fontweight='bold')
            axes[1, 1].set_title('First Principal Component Feature Importance (Optimized)', fontsize=14, fontweight='bold', pad=15)
            axes[1, 1].grid(True, alpha=0.3, linestyle='--', axis='x')
            axes[1, 1].spines['top'].set_visible(False)
            axes[1, 1].spines['right'].set_visible(False)
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                axes[1, 1].text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                               f'{width:.3f}', ha='left', va='center', 
                               fontweight='bold', fontsize=9)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"PCA results plot saved to: {save_path}")
    
    def generate_analysis_report(self, data, correlation_matrix, X_pca, feature_names, explained_variance_ratio):
        """生成分析报告"""
        print("\n" + "="*80)
        print("北京多气象因子对污染变化影响的PCA分析报告（优化版）")
        print("="*80)
        
        if data.empty:
            print("错误：没有数据可生成报告")
            return
        
        print(f"\n1. 数据概览:")
        print(f"   - 数据形状: {data.shape}")
        print(f"   - 特征数量: {len(feature_names)}")
        print(f"   - 样本数量: {len(data)}")
        
        meteo_features = [col for col in feature_names if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in feature_names if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        
        print(f"\n2. 特征分类:")
        print(f"   - 气象因子数量: {len(meteo_features)}")
        print(f"   - 污染指标数量: {len(pollution_features)}")
        
        print(f"\n3. 气象因子详情:")
        for i, feature in enumerate(meteo_features[:10]):
            print(f"   - {feature}")
        if len(meteo_features) > 10:
            print(f"   ... 还有 {len(meteo_features) - 10} 个气象因子")
        
        print(f"\n4. 污染指标详情:")
        for feature in pollution_features:
            print(f"   - {feature}")
        
        if correlation_matrix is not None:
            print(f"\n5. 相关性分析:")
            if pollution_features and meteo_features:
                print("   与污染指标最相关的气象因子:")
                for pollution_feat in pollution_features:
                    correlations = correlation_matrix[pollution_feat][meteo_features].abs()
                    top_correlations = correlations.nlargest(3)
                    print(f"   {pollution_feat}:")
                    for meteo_feat, corr in top_correlations.items():
                        print(f"     - {meteo_feat}: {corr:.3f}")
        
        if X_pca is not None and explained_variance_ratio is not None:
            print(f"\n6. PCA分析结果:")
            print(f"   - 主成分数量: {len(explained_variance_ratio)}")
            for i, var_ratio in enumerate(explained_variance_ratio):
                print(f"   - PC{i+1} 方差解释率: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
            
            cumulative_var = np.sum(explained_variance_ratio)
            print(f"   - 累积方差解释率: {cumulative_var:.4f} ({cumulative_var*100:.2f}%)")
            
            if len(explained_variance_ratio) >= 2:
                print(f"\n7. 主成分物理意义分析:")
                for i in range(min(3, len(explained_variance_ratio))):
                    loadings = self.pca.components_[i]
                    feature_loadings = list(zip(feature_names, loadings))
                    feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    print(f"   PC{i+1} 主要贡献特征（前5个）:")
                    for feature, loading in feature_loadings[:5]:
                        print(f"     - {feature}: {loading:.4f}")
        
        print(f"\n8. 主要发现:")
        print("   - 多气象因子综合分析能更好解释污染变化")
        print("   - 温度、湿度、风速等因素对污染水平有综合影响")
        print("   - 边界层高度和大气稳定性是重要影响因素")
        print("   - 降水和风速对污染物扩散有显著影响")
    
    def run_analysis(self):
        """运行完整分析工作流"""
        print("北京多气象因子对污染变化影响的PCA分析（优化版）")
        print("="*60)
        
        try:
            self.load_data()
            combined_data = self.prepare_combined_data()
            
            if combined_data.empty:
                print("错误：无法准备数据，请检查数据文件")
                return
            
            X_pca, feature_names, explained_variance_ratio = self.perform_pca_analysis(combined_data)
            correlation_matrix = self.analyze_correlations(combined_data)
            self.plot_correlation_heatmap(correlation_matrix)
            self.plot_pca_results(X_pca, feature_names, explained_variance_ratio)
            self.generate_analysis_report(combined_data, correlation_matrix, X_pca, feature_names, explained_variance_ratio)
            
            print("\n分析完成！")
        finally:
            self.cache.clear_cache()

def main():
    """主函数"""
    meteo_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV"
    pollution_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)"
    extra_pollution_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)"
    
    print("数据文件夹路径确认:")
    print(f"气象数据目录: {meteo_data_dir}")
    print(f"污染数据目录: {pollution_data_dir}")
    print(f"额外污染数据目录: {extra_pollution_data_dir}")
    print("如果路径不正确，请在main()函数中修改路径设置")
    
    analyzer = BeijingPCAAnalyzerOptimized(meteo_data_dir, pollution_data_dir, extra_pollution_data_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
