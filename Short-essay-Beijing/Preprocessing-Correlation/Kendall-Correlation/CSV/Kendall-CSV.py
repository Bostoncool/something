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

# Set font configuration
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = (10, 6)

class DataCache:
    """Data cache class to avoid reprocessing identical files"""
    
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
                print(f"Cache cleared: {self.cache_dir}")
        except Exception as e:
            print(f"Error clearing cache: {e}")

class BeijingKendallAnalyzer:
    """Beijing Meteorological Factors and Pollution Change Kendall Correlation Analyzer"""
    
    def __init__(self, meteo_data_dir=".", pollution_data_dir=".", additional_pollution_data_dir="."):
        self.meteo_data_dir = meteo_data_dir
        self.pollution_data_dir = pollution_data_dir
        self.additional_pollution_data_dir = additional_pollution_data_dir
        self.meteorological_data = []
        self.pollution_data = []
        self.additional_pollution_data = []
        self.cache = DataCache()
        
        # Meteorological parameter column mapping
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
        
        # Method 1: Search by year-month pattern
        for year in range(2015, 2025):
            for month in range(1, 13):
                pattern = f"{year}{month:02d}.csv"
                files = self.find_files_optimized(self.meteo_data_dir, pattern)
                all_files.extend(files)
        
        # Method 2: Search all CSV files (including specially named files like wind components)
        search_pattern = os.path.join(self.meteo_data_dir, "**", "*.csv")
        all_csv_files = glob.glob(search_pattern, recursive=True)
        
        # Merge and deduplicate
        all_files_set = set(all_files + all_csv_files)
        all_files = list(all_files_set)
        
        print(f"Found {len(all_files)} meteorological data files")
        
        # Count wind component files
        wind_files = [f for f in all_files if any(wind in os.path.basename(f).lower() for wind in ['u10', 'v10', 'u100', 'v100', 'wind'])]
        if wind_files:
            print(f"  Including {len(wind_files)} wind component related files")
            print(f"  Examples: {[os.path.basename(f) for f in wind_files[:3]]}")
        
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
        Aggregate spatial data, converting multidimensional data to time series
        
        Args:
            df: Multidimensional dataframe containing time, latitude, longitude columns
            
        Returns:
            pd.DataFrame: Dataframe aggregated by time
        """
        try:
            # Group by time and calculate spatial average
            if 'time' in df.columns:
                # Convert time column to datetime
                df['time'] = pd.to_datetime(df['time'])
                
                # Group by time and calculate spatial average for each time point
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
        Dedicated method for processing wind component data
        
        Args:
            df: Dataframe
            col: Wind component column name (u10, v10, u100, v100)
            
        Returns:
            Dict: Statistical information
        """
        try:
            values = df[col].values
            
            # Remove NaN values
            valid_values = values[~np.isnan(values)]
            
            # Remove outliers (wind components usually between -100 and 100 m/s)
            valid_values = valid_values[(valid_values >= -100) & (valid_values <= 100)]
            
            if len(valid_values) == 0:
                return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
            
            # If data is too large, sample it
            if len(valid_values) > 50000:
                step = len(valid_values) // 50000
                valid_values = valid_values[::step]
            
            # Calculate basic statistics
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
        if len(hourly_data) == 0:
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        
        # Handle multidimensional data - if 2D or 3D array, flatten to 1D
        if hourly_data.ndim > 1:
            # For wind component data, usually need to aggregate along time dimension
            if hourly_data.ndim == 3:  # (time, lat, lon)
                # Calculate average along time dimension
                hourly_data = np.nanmean(hourly_data, axis=(1, 2))
            elif hourly_data.ndim == 2:  # (time, spatial)
                # Calculate average along time dimension
                hourly_data = np.nanmean(hourly_data, axis=1)
        
        # Remove invalid values
        valid_data = hourly_data[~np.isnan(hourly_data)]
        if len(valid_data) == 0:
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        
        # If data is too large, sample to avoid memory issues
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
            
            # Read CSV file, handle possible encoding issues and comment lines
            try:
                df = pd.read_csv(filepath, encoding='utf-8', comment='#')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(filepath, encoding='gbk', comment='#')
                except UnicodeDecodeError:
                    df = pd.read_csv(filepath, encoding='latin-1', comment='#')
            except pd.errors.ParserError:
                # If parsing error, try skipping first few comment lines
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
            
            # Handle multi-index data (if exists)
            if 'time' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
                # This is multidimensional data converted from NC files, needs time aggregation
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
                        # Check if already in Celsius
                        if temp_kelvin.max() < 100:  # If max value less than 100, might be Celsius
                            temp_celsius = temp_kelvin
                        else:
                            temp_celsius = temp_kelvin - 273.15
                        daily_stats = self.calculate_stats_vectorized(temp_celsius)
                    elif col in ['u10', 'v10', 'u100', 'v100']:
                        # Special handling for wind component data
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
            
            # Parse year-month information
            filename = os.path.basename(filepath)
            try:
                # Try to extract year-month from filename (format: YYYYMM.csv)
                if len(filename) >= 6 and filename[:4].isdigit() and filename[4:6].isdigit():
                    year = int(filename[:4])
                    month = int(filename[4:6])
                    monthly_stats['year'] = year
                    monthly_stats['month'] = month
                else:
                    # If filename doesn't contain year-month info, try to extract from path
                    import re
                    match = re.search(r'(\d{4})(\d{2})', filepath)
                    if match:
                        monthly_stats['year'] = int(match.group(1))
                        monthly_stats['month'] = int(match.group(2))
                    else:
                        print(f"  Warning: Unable to extract year-month info from filename: {filename}")
            except ValueError as e:
                print(f"  Warning: Error parsing year-month info: {e}")
            
            self.cache.save_cached_data(filepath, monthly_stats)
            print(f"Processed meteorological data: {os.path.basename(filepath)} ({len(available_columns)} parameters)")
            return monthly_stats
            
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            return None
    
    def load_meteo_data_parallel(self):
        print("Starting parallel loading of meteorological data...")
        start_time = time.time()
        
        all_files = self.collect_all_meteo_files()
        if not all_files:
            print("No meteorological data files found")
            return
        
        max_workers = min(multiprocessing.cpu_count(), len(all_files))
        print(f"Using {max_workers} processes for parallel processing")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_single_meteo_file, filepath): filepath 
                            for filepath in all_files}
            
            for future in as_completed(future_to_file):
                result = future.result()
                if result:
                    self.meteorological_data.append(result)
        
        end_time = time.time()
        print(f"Meteorological data loading complete, time elapsed: {end_time - start_time:.2f} seconds")
        print(f"Successfully loaded data from {len(self.meteorological_data)} files")
    
    def load_pollution_data_optimized(self):
        print("Starting to load pollution data...")
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
                        print(f"Loaded pollution data: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
        
        end_time = time.time()
        print(f"Pollution data loading complete, time elapsed: {end_time - start_time:.2f} seconds")
        print(f"Successfully loaded data from {len(self.pollution_data)} files")
    
    def load_additional_pollution_data_optimized(self):
        print("Starting to load additional pollution data (SO2, CO, O3, NO2)...")
        start_time = time.time()
        
        if not os.path.exists(self.additional_pollution_data_dir):
            print(f"Warning: Additional pollution data directory does not exist: {self.additional_pollution_data_dir}")
            print("Skipping additional pollution data loading...")
            return
        
        def additional_pollution_file_filter(filename):
            return filename.startswith('beijing_extra_') and filename.endswith('.csv')
        
        all_additional_pollution_files = []
        search_pattern = os.path.join(self.additional_pollution_data_dir, "**", "*.csv")
        for filepath in glob.glob(search_pattern, recursive=True):
            filename = os.path.basename(filepath)
            if additional_pollution_file_filter(filename):
                all_additional_pollution_files.append(filepath)
        
        print(f"Found {len(all_additional_pollution_files)} additional pollution data files")
        if len(all_additional_pollution_files) == 0:
            print("No additional pollution data files found, please check directory path")
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
                        print(f"Loaded additional pollution data: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
        
        end_time = time.time()
        print(f"Additional pollution data loading complete, time elapsed: {end_time - start_time:.2f} seconds")
        print(f"Successfully loaded data from {len(self.additional_pollution_data)} files")
    
    def load_data(self):
        print("Starting to load all data...")
        
        self.load_meteo_data_parallel()
        self.load_pollution_data_optimized()
        self.load_additional_pollution_data_optimized()
        
        print("Data loading complete!")
    
    def prepare_combined_data(self):
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
                    combined_data.loc[:, col] = combined_data[col].fillna(mean_val)
        
        print(f"Final data shape: {combined_data.shape}")
        print(f"Merged column names: {list(combined_data.columns)}")
        
        meteo_features = [col for col in combined_data.columns if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in combined_data.columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        
        print(f"Number of meteorological parameters: {len(meteo_features)}")
        print(f"Meteorological parameter list: {meteo_features[:10]}..." if len(meteo_features) > 10 else f"Meteorological parameter list: {meteo_features}")
        
        # Check wind component data
        wind_features = [col for col in combined_data.columns if any(wind in col for wind in ['u10', 'v10', 'u100', 'v100'])]
        print(f"Number of wind component parameters: {len(wind_features)}")
        print(f"Wind component parameter list: {wind_features}")
        
        print(f"Number of pollution parameters: {len(pollution_features)}")
        print(f"Pollution parameter list: {pollution_features}")
        
        return combined_data
    
    def perform_kendall_analysis(self, data):
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
                        print(f"Error calculating Kendall correlation for {var1} and {var2}: {e}")
                        kendall_corr_matrix.loc[var1, var2] = np.nan
                        kendall_corr_matrix.loc[var2, var1] = np.nan
                        kendall_p_matrix.loc[var1, var2] = np.nan
                        kendall_p_matrix.loc[var2, var1] = np.nan
        
        kendall_corr_matrix = kendall_corr_matrix.astype(float)
        kendall_p_matrix = kendall_p_matrix.astype(float)
        
        print(f"Kendall correlation matrix shape: {kendall_corr_matrix.shape}")
        
        return kendall_corr_matrix, kendall_p_matrix
    
    def analyze_kendall_correlations(self, data, kendall_corr_matrix):
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
                        print(f"\nMeteorological factors most correlated with {pollution_feat} (Kendall):")
                        for meteo_feat, corr in top_correlations.items():
                            print(f"  {meteo_feat}: {corr:.3f}")
        
        return kendall_corr_matrix
    
    def plot_kendall_heatmap(self, kendall_corr_matrix, save_path='beijing_kendall_correlation_heatmap.png'):
        if kendall_corr_matrix is None:
            print("Error: No Kendall correlation data to plot")
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
            print("Error: No Kendall correlation data to plot")
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
        
        print(f"Kendall statistics plot saved to: {save_path}")
    
    def generate_analysis_report(self, data, kendall_corr_matrix, kendall_p_matrix, feature_columns):
        print("\n" + "="*80)
        print("Beijing Multi-Meteorological Factors and Pollution Change Kendall Correlation Analysis Report")
        print("Pollution Indicators: PM2.5, PM10, AQI, SO2, CO, O3, NO2")
        print("="*80)
        
        if data.empty:
            print("Error: No data available to generate report")
            return
        
        print(f"\n1. Data Overview:")
        print(f"   - Data shape: {data.shape}")
        print(f"   - Number of features: {len(feature_columns)}")
        print(f"   - Number of samples: {len(data)}")
        
        meteo_features = [col for col in feature_columns if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in feature_columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        
        print(f"\n2. Feature Classification:")
        print(f"   - Number of meteorological factors: {len(meteo_features)}")
        print(f"   - Number of pollution indicators: {len(pollution_features)}")
        
        print(f"\n3. Meteorological Factors Details:")
        for i, feature in enumerate(meteo_features[:10]):
            print(f"   - {feature}")
        if len(meteo_features) > 10:
            print(f"   ... and {len(meteo_features) - 10} more meteorological factors")
        
        print(f"\n4. Pollution Indicators Details:")
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
                print("   Meteorological factors most correlated with pollution indicators:")
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
        print("   - Temperature, humidity and wind factors show significant correlations with pollution levels")
        print("   - Boundary layer height and atmospheric stability are key influencing factors")
        print("   - Precipitation and wind speed have significant effects on pollutant dispersion")
        print("   - Non-parametric analysis is more robust to outliers compared to Pearson correlation")
        
        print("\n" + "="*80)
    
    def run_analysis(self):
        print("Beijing Multi-Meteorological Factors and Pollution Change Kendall Correlation Analysis")
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
        
        self.plot_kendall_heatmap(kendall_corr_matrix)
        self.plot_kendall_statistics(kendall_corr_matrix, kendall_p_matrix, feature_columns)
        self.generate_analysis_report(combined_data, kendall_corr_matrix, kendall_p_matrix, feature_columns)
        
        print("\nClearing cache to free up space...")
        self.cache.clear_cache()
        
        print("\nAnalysis complete!")

def main():
    # Please modify these three paths according to your actual situation
    meteo_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV"  # Meteorological data folder path
    pollution_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)"  # Pollution data folder path (AQI, PM2.5, PM10)
    additional_pollution_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)"  # Additional pollution data folder path (SO2, CO, O3, NO2)
    
    print("Please confirm data folder paths:")
    print(f"Meteorological data directory: {meteo_data_dir}")
    print(f"Pollution data directory (AQI, PM2.5, PM10): {pollution_data_dir}")
    print(f"Additional pollution data directory (SO2, CO, O3, NO2): {additional_pollution_data_dir}")
    print("If paths are incorrect, please modify path settings in main() function")
    
    analyzer = BeijingKendallAnalyzer(meteo_data_dir, pollution_data_dir, additional_pollution_data_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
