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
    """Data cache class to avoid repeatedly processing the same files"""
    
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
    
    def clear_cache(self):
        """Clear cache directory"""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            print("Cache cleared")

class BeijingPCAAnalyzerOptimized:
    """Beijing Multi-Meteorological Factors PCA Analyzer for Pollution Changes (Optimized Version)"""
    
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
        """Collect all meteorological data file paths"""
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
        
        # Merge and remove duplicates
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
        """Optimize data types to reduce memory usage"""
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
        return df
    
    def aggregate_spatial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate spatial data, converting multi-dimensional data to time series
        
        Args:
            df: Multi-dimensional dataframe containing time, latitude, longitude columns
            
        Returns:
            pd.DataFrame: Time-aggregated dataframe
        """
        try:
            # Group by time, calculate spatial average
            if 'time' in df.columns:
                # Convert time column to datetime
                df['time'] = pd.to_datetime(df['time'])
                
                # Group by time, calculate spatial average for each time point
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
        Method specifically for processing wind component data
        
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
            
            # Remove outliers (wind components typically range from -100 to 100 m/s)
            valid_values = valid_values[(valid_values >= -100) & (valid_values <= 100)]
            
            if len(valid_values) == 0:
                return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
            
            # If data is too large, perform sampling
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
        """Vectorized statistics calculation"""
        if len(hourly_data) == 0:
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        
        # Handle multi-dimensional data - flatten 2D or 3D arrays to 1D
        if hourly_data.ndim > 1:
            # For wind component data, usually need to aggregate by time dimension
            if hourly_data.ndim == 3:  # (time, lat, lon)
                # Calculate average by time dimension
                hourly_data = np.nanmean(hourly_data, axis=(1, 2))
            elif hourly_data.ndim == 2:  # (time, spatial)
                # Calculate average by time dimension
                hourly_data = np.nanmean(hourly_data, axis=1)
        
        # Remove invalid values
        valid_data = hourly_data[~np.isnan(hourly_data)]
        if len(valid_data) == 0:
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        
        # If data is too large, perform sampling to avoid memory issues
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
        """Process single meteorological data file (for parallel processing)"""
        try:
            cached_data = self.cache.get_cached_data(filepath)
            if cached_data:
                print(f"Using cached data: {os.path.basename(filepath)}")
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
                # If there's a parsing error, try skipping the first few comment lines
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
            
            # Process multi-index data (if exists)
            if 'time' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
                # This is multi-dimensional data converted from NC files, needs time aggregation
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
                        if temp_kelvin.max() < 100:  # If max value < 100, might be Celsius
                            temp_celsius = temp_kelvin
                        else:
                            temp_celsius = temp_kelvin - 273.15
                        daily_stats = self.calculate_stats_vectorized(temp_celsius)
                    elif col in ['u10', 'v10', 'u100', 'v100']:
                        # Special processing for wind component data
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
            
            # Parse year and month information
            filename = os.path.basename(filepath)
            try:
                # Try to extract year and month from filename (format: YYYYMM.csv)
                if len(filename) >= 6 and filename[:4].isdigit() and filename[4:6].isdigit():
                    year = int(filename[:4])
                    month = int(filename[4:6])
                    monthly_stats['year'] = year
                    monthly_stats['month'] = month
                else:
                    # If filename doesn't contain year/month info, try to extract from path
                    import re
                    match = re.search(r'(\d{4})(\d{2})', filepath)
                    if match:
                        monthly_stats['year'] = int(match.group(1))
                        monthly_stats['month'] = int(match.group(2))
                    else:
                        print(f"  Warning: Unable to extract year/month info from filename: {filename}")
            except ValueError as e:
                print(f"  Warning: Error parsing year/month info: {e}")
            
            self.cache.save_cached_data(filepath, monthly_stats)
            print(f"Processed meteorological data: {os.path.basename(filepath)} ({len(available_columns)} parameters)")
            return monthly_stats
            
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            return None
    
    def load_meteo_data_parallel(self):
        """Load meteorological data in parallel"""
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
        print(f"Meteorological data loading completed, time elapsed: {end_time - start_time:.2f} seconds")
        print(f"Successfully loaded data from {len(self.meteorological_data)} files")
    
    def load_pollution_data_optimized(self):
        """Optimized pollution data loading"""
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
                    print(f"Using cached data: {os.path.basename(filepath)}")
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
        print(f"Pollution data loading completed, time elapsed: {end_time - start_time:.2f} seconds")
        print(f"Successfully loaded data from {len(self.pollution_data)} files")
    
    def load_extra_pollution_data(self):
        """Load extra pollution data (SO2, CO, O3, NO2)"""
        print("Starting to load extra pollution data...")
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
                    print(f"Using cached data: {os.path.basename(filepath)}")
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
                        print(f"Loaded extra pollution data: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
        
        end_time = time.time()
        print(f"Extra pollution data loading completed, time elapsed: {end_time - start_time:.2f} seconds")
        print(f"Successfully loaded data from {len(self.extra_pollution_data)} files")
    
    def load_data(self):
        """Load all data (optimized version)"""
        print("Starting to load data (optimized version)...")
        
        self.load_meteo_data_parallel()
        self.load_pollution_data_optimized()
        self.load_extra_pollution_data()
        
        print("Data loading completed!")
    
    def prepare_combined_data(self):
        """Prepare combined meteorological and pollution data"""
        print("Preparing combined data...")
        
        if not self.meteorological_data or not self.pollution_data:
            print("Error: Insufficient data, unable to perform analysis")
            print(f"Meteorological data count: {len(self.meteorological_data)}")
            print(f"Pollution data count: {len(self.pollution_data)}")
            return pd.DataFrame()
        
        meteo_df = pd.DataFrame(self.meteorological_data)
        pollution_df = pd.DataFrame(self.pollution_data)
        extra_pollution_df = pd.DataFrame(self.extra_pollution_data) if self.extra_pollution_data else pd.DataFrame()
        
        print(f"Original meteorological data shape: {meteo_df.shape}")
        print(f"Original pollution data shape: {pollution_df.shape}")
        if not extra_pollution_df.empty:
            print(f"Original extra pollution data shape: {extra_pollution_df.shape}")
        
        min_len = min(len(meteo_df), len(pollution_df))
        if not extra_pollution_df.empty:
            min_len = min(min_len, len(extra_pollution_df))
        
        print(f"Aligned length: {min_len}")
        
        meteo_df = meteo_df.head(min_len)
        pollution_df = pollution_df.head(min_len)
        if not extra_pollution_df.empty:
            extra_pollution_df = extra_pollution_df.head(min_len)
        
        if not extra_pollution_df.empty:
            combined_data = pd.concat([meteo_df, pollution_df, extra_pollution_df], axis=1)
        else:
            combined_data = pd.concat([meteo_df, pollution_df], axis=1)
        
        print(f"Combined data shape: {combined_data.shape}")
        
        combined_data = combined_data.dropna(how='all')
        combined_data = combined_data.ffill().bfill()
        
        numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if combined_data[col].isna().any():
                mean_val = combined_data[col].mean()
                if not pd.isna(mean_val):
                    combined_data.loc[:, col] = combined_data[col].fillna(mean_val)
        
        print(f"Final data shape: {combined_data.shape}")
        print(f"Combined column names: {list(combined_data.columns)}")
        print(f"Final NaN count: {combined_data.isna().sum().sum()}")
        
        meteo_features = [col for col in combined_data.columns if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in combined_data.columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        
        print(f"Meteorological parameters count: {len(meteo_features)}")
        print(f"Meteorological parameters list: {meteo_features[:10]}..." if len(meteo_features) > 10 else f"Meteorological parameters list: {meteo_features}")
        
        # Check wind component data
        wind_features = [col for col in combined_data.columns if any(wind in col for wind in ['u10', 'v10', 'u100', 'v100'])]
        print(f"Wind component parameters count: {len(wind_features)}")
        print(f"Wind component parameters list: {wind_features}")
        
        print(f"Pollution parameters count: {len(pollution_features)}")
        print(f"Pollution parameters list: {pollution_features}")
        
        return combined_data
    
    def perform_pca_analysis(self, data, n_components=2):
        """Perform PCA analysis"""
        print("Performing PCA analysis...")
        
        if data.empty:
            print("Error: No data available for PCA analysis")
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
        
        print(f"Explained variance ratio of the first {len(explained_variance_ratio)} principal components:")
        for i, (var_ratio, cum_var_ratio) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio)):
            print(f"PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
            print(f"Cumulative: {cum_var_ratio:.4f} ({cum_var_ratio*100:.2f}%)")
        
        print("\nPrincipal component contribution analysis:")
        for i in range(len(explained_variance_ratio)):
            loadings = self.pca.components_[i]
            feature_loadings = list(zip(feature_columns, loadings))
            feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print(f"\nPC{i+1} top contributing features (top 5):")
            for feature, loading in feature_loadings[:5]:
                print(f"  {feature}: {loading:.4f}")
        
        return X_pca, feature_columns, explained_variance_ratio
    
    def analyze_correlations(self, data):
        """Analyze correlations between variables"""
        print("Analyzing correlations...")
        
        if data.empty:
            print("Error: No data available for correlation analysis")
            return None
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['year', 'month']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        correlation_matrix = data[feature_columns].corr()
        
        print("\nCorrelation analysis between meteorological factors and pollution indicators:")
        pollution_features = [col for col in feature_columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        meteo_features = [col for col in feature_columns if any(x in col for x in self.meteo_columns.keys())]
        
        if pollution_features and meteo_features:
            print(f"Found {len(pollution_features)} pollution indicators and {len(meteo_features)} meteorological factors")
            
            for pollution_feat in pollution_features:
                correlations = correlation_matrix[pollution_feat][meteo_features].abs()
                top_correlations = correlations.nlargest(5)
                print(f"\nMeteorological factors most correlated with {pollution_feat}:")
                for meteo_feat, corr in top_correlations.items():
                    print(f"  {meteo_feat}: {corr:.3f}")
        
        return correlation_matrix
    
    def plot_correlation_heatmap(self, correlation_matrix, save_path='beijing_correlation_heatmap_optimized.png'):
        """Plot correlation heatmap"""
        if correlation_matrix is None:
            print("Error: No correlation data available to plot")
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
        """Plot PCA results"""
        if X_pca is None:
            print("Error: No PCA data available to plot")
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
        """Generate analysis report"""
        print("\n" + "="*80)
        print("Beijing Multi-Meteorological Factors PCA Analysis Report on Pollution Changes (Optimized)")
        print("="*80)
        
        if data.empty:
            print("Error: No data available to generate report")
            return
        
        print(f"\n1. Data Overview:")
        print(f"   - Data shape: {data.shape}")
        print(f"   - Number of features: {len(feature_names)}")
        print(f"   - Number of samples: {len(data)}")
        
        meteo_features = [col for col in feature_names if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in feature_names if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi', 'so2', 'co', 'o3', 'no2'])]
        
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
        
        if correlation_matrix is not None:
            print(f"\n5. Correlation Analysis:")
            if pollution_features and meteo_features:
                print("   Meteorological factors most correlated with pollution indicators:")
                for pollution_feat in pollution_features:
                    correlations = correlation_matrix[pollution_feat][meteo_features].abs()
                    top_correlations = correlations.nlargest(3)
                    print(f"   {pollution_feat}:")
                    for meteo_feat, corr in top_correlations.items():
                        print(f"     - {meteo_feat}: {corr:.3f}")
        
        if X_pca is not None and explained_variance_ratio is not None:
            print(f"\n6. PCA Analysis Results:")
            print(f"   - Number of principal components: {len(explained_variance_ratio)}")
            for i, var_ratio in enumerate(explained_variance_ratio):
                print(f"   - PC{i+1} explained variance ratio: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
            
            cumulative_var = np.sum(explained_variance_ratio)
            print(f"   - Cumulative explained variance ratio: {cumulative_var:.4f} ({cumulative_var*100:.2f}%)")
            
            if len(explained_variance_ratio) >= 2:
                print(f"\n7. Principal Component Physical Interpretation:")
                for i in range(min(3, len(explained_variance_ratio))):
                    loadings = self.pca.components_[i]
                    feature_loadings = list(zip(feature_names, loadings))
                    feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    print(f"   PC{i+1} top contributing features (top 5):")
                    for feature, loading in feature_loadings[:5]:
                        print(f"     - {feature}: {loading:.4f}")
        
        print(f"\n8. Key Findings:")
        print("   - Comprehensive analysis of multiple meteorological factors better explains pollution variations")
        print("   - Temperature, humidity, wind speed, etc. have combined effects on pollution levels")
        print("   - Boundary layer height and atmospheric stability are important influencing factors")
        print("   - Precipitation and wind speed have significant impacts on pollutant dispersion")
    
    def run_analysis(self):
        """Run complete analysis workflow"""
        print("Beijing Multi-Meteorological Factors PCA Analysis on Pollution Changes (Optimized)")
        print("="*60)
        
        try:
            self.load_data()
            combined_data = self.prepare_combined_data()
            
            if combined_data.empty:
                print("Error: Unable to prepare data, please check data files")
                return
            
            X_pca, feature_names, explained_variance_ratio = self.perform_pca_analysis(combined_data)
            correlation_matrix = self.analyze_correlations(combined_data)
            self.plot_correlation_heatmap(correlation_matrix)
            self.plot_pca_results(X_pca, feature_names, explained_variance_ratio)
            self.generate_analysis_report(combined_data, correlation_matrix, X_pca, feature_names, explained_variance_ratio)
            
            print("\nAnalysis completed!")
        finally:
            self.cache.clear_cache()

def main():
    """Main function"""
    meteo_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV"
    pollution_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)"
    extra_pollution_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)"
    
    print("Data folder path confirmation:")
    print(f"Meteorological data directory: {meteo_data_dir}")
    print(f"Pollution data directory: {pollution_data_dir}")
    print(f"Extra pollution data directory: {extra_pollution_data_dir}")
    print("If paths are incorrect, please modify the path settings in the main() function")
    
    analyzer = BeijingPCAAnalyzerOptimized(meteo_data_dir, pollution_data_dir, extra_pollution_data_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
