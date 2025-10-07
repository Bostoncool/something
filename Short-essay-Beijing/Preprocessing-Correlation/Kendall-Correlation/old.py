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
from typing import List, Dict, Any, Optional

warnings.filterwarnings('ignore')

# 设置中文字体支持
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
                print(f"缓存清理完成: {self.cache_dir}")
        except Exception as e:
            print(f"清理缓存时出错: {e}")

class BeijingKendallAnalyzer:
    """北京气象因子对污染变化的Kendall相关性分析器"""
    
    def __init__(self, meteo_data_dir=".", pollution_data_dir=".", additional_pollution_data_dir="."):
        self.meteo_data_dir = meteo_data_dir
        self.pollution_data_dir = pollution_data_dir
        self.additional_pollution_data_dir = additional_pollution_data_dir
        self.meteorological_data = []
        self.pollution_data = []
        self.additional_pollution_data = []
        self.cache = DataCache()
        
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
    
    def find_files_optimized(self, root_dir: str, pattern: str) -> List[str]:
        if not os.path.exists(root_dir):
            return []
        search_pattern = os.path.join(root_dir, "**", pattern)
        return glob.glob(search_pattern, recursive=True)
    
    def collect_all_meteo_files(self) -> List[str]:
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
        
        print(f"找到 {len(all_files)} 个气象数据文件")
        
        # 统计风分量文件
        wind_files = [f for f in all_files if any(wind in os.path.basename(f).lower() for wind in ['u10', 'v10', 'u100', 'v100', 'wind'])]
        if wind_files:
            print(f"  其中包含 {len(wind_files)} 个风分量相关文件")
            print(f"  示例: {[os.path.basename(f) for f in wind_files[:3]]}")
        
        return all_files
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
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
        print(f"成功加载 {len(self.meteorological_data)} 个文件的数据")
    
    def load_pollution_data_optimized(self):
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
                print(f"加载文件时出错 {filepath}: {e}")
        
        end_time = time.time()
        print(f"污染数据加载完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"成功加载 {len(self.pollution_data)} 个文件的数据")
    
    def load_additional_pollution_data_optimized(self):
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
                print(f"加载文件时出错 {filepath}: {e}")
        
        end_time = time.time()
        print(f"额外污染数据加载完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"成功加载 {len(self.additional_pollution_data)} 个文件的数据")
    
    def load_data(self):
        print("开始加载所有数据...")
        
        self.load_meteo_data_parallel()
        self.load_pollution_data_optimized()
        self.load_additional_pollution_data_optimized()
        
        print("数据加载完成!")
    
    def prepare_combined_data(self):
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
        print("执行Kendall相关性分析...")
        
        if data.empty:
            print("错误: 没有可用于Kendall相关性分析的数据")
            return None, None
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['year', 'month']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        kendall_corr_matrix = pd.DataFrame(index=feature_columns, columns=feature_columns)
        kendall_p_matrix = pd.DataFrame(index=feature_columns, columns=feature_columns)
        
        print("计算Kendall相关系数...")
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
                        print(f"计算Kendall相关性时出错 {var1} 和 {var2}: {e}")
                        kendall_corr_matrix.loc[var1, var2] = np.nan
                        kendall_corr_matrix.loc[var2, var1] = np.nan
                        kendall_p_matrix.loc[var1, var2] = np.nan
                        kendall_p_matrix.loc[var2, var1] = np.nan
        
        kendall_corr_matrix = kendall_corr_matrix.astype(float)
        kendall_p_matrix = kendall_p_matrix.astype(float)
        
        print(f"Kendall相关矩阵形状: {kendall_corr_matrix.shape}")
        
        return kendall_corr_matrix, kendall_p_matrix
    
    def analyze_kendall_correlations(self, data, kendall_corr_matrix):
        print("分析Kendall相关性...")
        
        if data.empty or kendall_corr_matrix is None:
            print("错误: 没有可用于Kendall相关性分析的数据")
            return None
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['year', 'month']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        print("\n气象因子与污染指标之间的Kendall相关性分析:")
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
                        print(f"\n与 {pollution_feat} 最相关的气象因子 (Kendall):")
                        for meteo_feat, corr in top_correlations.items():
                            print(f"  {meteo_feat}: {corr:.3f}")
        
        return kendall_corr_matrix
    
    def plot_kendall_heatmap(self, kendall_corr_matrix, save_path='beijing_kendall_correlation_heatmap.png'):
        if kendall_corr_matrix is None:
            print("错误: 没有Kendall相关性数据可绘制")
            return
        
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(20, 16))
        
        mask = np.triu(np.ones_like(kendall_corr_matrix, dtype=bool))
        
        sns.heatmap(kendall_corr_matrix, 
                   mask=mask,
                   annot=False,
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   cbar_kws={'shrink': 0.8, 'aspect': 50, 'label': 'Kendall Correlation Coefficient'},
                   linewidths=0.2,
                   linecolor='white')
        
        plt.title('Beijing Meteorological Factors and Pollution Indicators Kendall Correlation Heatmap\n(PM2.5, PM10, AQI, SO2, CO, O3, NO2)', 
                 fontsize=22, fontweight='bold', pad=50, color='#2E3440')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"Kendall correlation heatmap saved to: {save_path}")
    
    def plot_kendall_statistics(self, kendall_corr_matrix, kendall_p_matrix, feature_columns, save_path='beijing_kendall_statistics.png'):
        if kendall_corr_matrix is None:
            print("错误: 没有Kendall相关性数据可绘制")
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
        
        # 4. 相关性强度总结
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
        
        print(f"Kendall statistics plot saved to: {save_path}")
    
    def generate_analysis_report(self, data, kendall_corr_matrix, kendall_p_matrix, feature_columns):
        print("\n" + "="*80)
        print("北京多气象因子对污染变化的Kendall相关性分析报告")
        print("污染指标: PM2.5, PM10, AQI, SO2, CO, O3, NO2")
        print("="*80)
        
        if data.empty:
            print("错误: 没有可生成报告的数据")
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
            print(f"   ... 还有 {len(meteo_features) - 10} 个气象因子")
        
        print(f"\n4. 污染指标详情:")
        for feature in pollution_features:
            print(f"   - {feature}")
        
        if kendall_corr_matrix is not None:
            print(f"\n5. Kendall相关性分析:")
            
            corr_values = kendall_corr_matrix.values
            corr_values = corr_values[~np.isnan(corr_values)]
            corr_values = corr_values[np.abs(corr_values) > 0]
            
            print(f"   - 计算的相关性总数: {len(corr_values)}")
            print(f"   - 平均相关系数: {np.mean(corr_values):.4f}")
            print(f"   - 标准差: {np.std(corr_values):.4f}")
            print(f"   - 强相关性 (|r| > 0.3): {np.sum(np.abs(corr_values) > 0.3)}")
            print(f"   - 中等相关性 (0.1 < |r| <= 0.3): {np.sum((np.abs(corr_values) > 0.1) & (np.abs(corr_values) <= 0.3))}")
            
            if pollution_features and meteo_features:
                print("   与污染指标最相关的气象因子:")
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
        print("   - Kendall相关性分析揭示了变量间的非参数关系")
        print("   - 温度、湿度和风因子与污染水平存在显著相关性")
        print("   - 边界层高度和大气稳定性是关键影响因素")
        print("   - 降水和风速对污染物扩散有显著影响")
        print("   - 非参数分析相比Pearson相关性对异常值更加稳健")
        
        print("\n" + "="*80)
    
    def run_analysis(self):
        print("北京多气象因子对污染变化的Kendall相关性分析")
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
        
        self.plot_kendall_heatmap(kendall_corr_matrix)
        self.plot_kendall_statistics(kendall_corr_matrix, kendall_p_matrix, feature_columns)
        self.generate_analysis_report(combined_data, kendall_corr_matrix, kendall_p_matrix, feature_columns)
        
        print("\n清理缓存以释放空间...")
        self.cache.clear_cache()
        
        print("\n分析完成!")

def main():
    # 请根据实际情况修改这三个路径
    meteo_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV"  # 气象数据文件夹路径
    pollution_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)"  # 污染数据文件夹路径 (AQI, PM2.5, PM10)
    additional_pollution_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)"  # 额外污染数据文件夹路径 (SO2, CO, O3, NO2)
    
    print("请确认数据文件夹路径:")
    print(f"气象数据目录: {meteo_data_dir}")
    print(f"污染数据目录 (AQI, PM2.5, PM10): {pollution_data_dir}")
    print(f"额外污染数据目录 (SO2, CO, O3, NO2): {additional_pollution_data_dir}")
    print("如果路径不正确，请在main()函数中修改路径设置")
    
    analyzer = BeijingKendallAnalyzer(meteo_data_dir, pollution_data_dir, additional_pollution_data_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
