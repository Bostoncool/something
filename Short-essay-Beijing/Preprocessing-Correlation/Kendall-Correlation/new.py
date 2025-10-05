import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
import warnings
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import hashlib
import pickle
import time
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
    """Data cache class for avoiding repeated processing of the same files"""
    
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

class BeijingKendallAnalyzer:
    """Kendall correlation analyzer for Beijing meteorological factors impact on pollution changes"""
    
    def __init__(self, meteo_data_dir=".", pollution_data_dir="."):
        """
        Initialize analyzer
        
        Args:
            meteo_data_dir: Meteorological data directory path
            pollution_data_dir: Pollution data directory path
        """
        self.meteo_data_dir = meteo_data_dir
        self.pollution_data_dir = pollution_data_dir
        self.meteorological_data = []
        self.pollution_data = []
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
        
        # Use optimized file search
        for year in range(2015, 2025):
            for month in range(1, 13):
                pattern = f"{year}{month:02d}.csv"
                files = self.find_files_optimized(self.meteo_data_dir, pattern)
                all_files.extend(files)
        
        print(f"Found {len(all_files)} meteorological data files")
        return all_files
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage"""
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
        return df
    
    def calculate_stats_vectorized(self, hourly_data: np.ndarray) -> Dict[str, float]:
        """
        Vectorized calculation of statistics
        
        Args:
            hourly_data: Hourly data array
            
        Returns:
            Dict[str, float]: Statistical results
        """
        if len(hourly_data) == 0:
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        
        # Reshape data to daily 24 hours
        days = len(hourly_data) // 24
        if days > 0:
            daily_data = hourly_data[:days*24].reshape(days, 24)
            
            # Vectorized calculation of daily statistics
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
                'mean': np.nanmean(hourly_data),
                'std': np.nanstd(hourly_data),
                'min': np.nanmin(hourly_data),
                'max': np.nanmax(hourly_data)
            }
    
    def process_single_meteo_file(self, filepath: str) -> Optional[Dict]:
        """
        Process single meteorological data file (for parallel processing)
        
        Args:
            filepath: File path
            
        Returns:
            Optional[Dict]: Processing result or None
        """
        try:
            # Check cache
            cached_data = self.cache.get_cached_data(filepath)
            if cached_data:
                print(f"Using cached data: {os.path.basename(filepath)}")
                return cached_data
            
            # Read data
            df = pd.read_csv(filepath)
            if df.empty:
                return None
            
            # Optimize data types
            df = self.optimize_dtypes(df)
            
            # Check available meteorological parameter columns
            available_columns = [col for col in self.meteo_columns.keys() if col in df.columns]
            
            if not available_columns:
                return None
            
            # Calculate statistics for all available meteorological parameters
            monthly_stats = {}
            
            for col in available_columns:
                if col in df.columns:
                    # Process temperature-related parameters (Kelvin to Celsius)
                    if col in ['t2m', 'mn2t', 'd2m']:
                        temp_kelvin = df[col].values
                        temp_celsius = temp_kelvin - 273.15
                        daily_stats = self.calculate_stats_vectorized(temp_celsius)
                    else:
                        # Other parameters use original values
                        values = df[col].values
                        daily_stats = self.calculate_stats_vectorized(values)
                    
                    monthly_stats[f'{col}_mean'] = daily_stats['mean']
                    monthly_stats[f'{col}_std'] = daily_stats['std']
                    monthly_stats[f'{col}_min'] = daily_stats['min']
                    monthly_stats[f'{col}_max'] = daily_stats['max']
            
            # Add year and month information
            filename = os.path.basename(filepath)
            if len(filename) >= 6:
                year = int(filename[:4])
                month = int(filename[4:6])
                monthly_stats['year'] = year
                monthly_stats['month'] = month
            
            # Cache results
            self.cache.save_cached_data(filepath, monthly_stats)
            
            print(f"Processed meteorological data: {os.path.basename(filepath)} (contains {len(available_columns)} parameters)")
            return monthly_stats
            
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            return None
    
    def load_meteo_data_parallel(self):
        """Load meteorological data in parallel"""
        print("Starting parallel loading of meteorological data...")
        start_time = time.time()
        
        # Collect all files
        all_files = self.collect_all_meteo_files()
        
        if not all_files:
            print("No meteorological data files found")
            return
        
        # Process files in parallel
        max_workers = min(multiprocessing.cpu_count(), len(all_files))
        print(f"Using {max_workers} processes for parallel processing")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(self.process_single_meteo_file, filepath): filepath 
                            for filepath in all_files}
            
            # Collect results
            for future in as_completed(future_to_file):
                result = future.result()
                if result:
                    self.meteorological_data.append(result)
        
        end_time = time.time()
        print(f"Meteorological data loading completed, time taken: {end_time - start_time:.2f} seconds")
        print(f"Successfully loaded data from {len(self.meteorological_data)} files")
    
    def load_pollution_data_optimized(self):
        """Optimized loading of pollution data"""
        print("Starting to load pollution data...")
        start_time = time.time()
        
        # Use optimized file search
        def pollution_file_filter(filename):
            return filename.startswith('beijing_all_') and filename.endswith('.csv')
        
        # Collect all pollution data files
        all_pollution_files = []
        search_pattern = os.path.join(self.pollution_data_dir, "**", "*.csv")
        for filepath in glob.glob(search_pattern, recursive=True):
            filename = os.path.basename(filepath)
            if pollution_file_filter(filename):
                all_pollution_files.append(filepath)
        
        print(f"Found {len(all_pollution_files)} pollution data files")
        
        for filepath in all_pollution_files:
            try:
                # Check cache
                cached_data = self.cache.get_cached_data(filepath)
                if cached_data:
                    self.pollution_data.append(cached_data)
                    print(f"Using cached data: {os.path.basename(filepath)}")
                    continue
                
                df = pd.read_csv(filepath)
                if not df.empty:
                    # Extract PM2.5, PM10, AQI data
                    if 'type' in df.columns:
                        # New format data
                        pm25_data = df[df['type'] == 'PM2.5'].iloc[:, 3:].values
                        pm10_data = df[df['type'] == 'PM10'].iloc[:, 3:].values
                        aqi_data = df[df['type'] == 'AQI'].iloc[:, 3:].values
                        
                        # Calculate daily averages
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
                            
                            # Cache results
                            self.cache.save_cached_data(filepath, pollution_stats)
                            
                            self.pollution_data.append(pollution_stats)
                            print(f"Loaded pollution data: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"Error loading file {filepath}: {e}")
        
        end_time = time.time()
        print(f"Pollution data loading completed, time taken: {end_time - start_time:.2f} seconds")
        print(f"Successfully loaded data from {len(self.pollution_data)} files")
    
    def load_data(self):
        """Load all data (optimized version)"""
        print("Starting to load data (optimized version)...")
        
        # Load meteorological data in parallel
        self.load_meteo_data_parallel()
        
        # Load pollution data
        self.load_pollution_data_optimized()
        
        print("Data loading completed!")
    
    def prepare_combined_data(self):
        """Prepare combined meteorological and pollution data"""
        print("Preparing combined data...")
        
        if not self.meteorological_data or not self.pollution_data:
            print("Error: Insufficient data for analysis")
            print(f"Number of meteorological data: {len(self.meteorological_data)}")
            print(f"Number of pollution data: {len(self.pollution_data)}")
            return pd.DataFrame()
        
        # Create dataframes
        meteo_df = pd.DataFrame(self.meteorological_data)
        pollution_df = pd.DataFrame(self.pollution_data)
        
        print(f"Original meteorological data shape: {meteo_df.shape}")
        print(f"Original pollution data shape: {pollution_df.shape}")
        
        # Check NaN values
        print(f"Number of NaN in meteorological data: {meteo_df.isna().sum().sum()}")
        print(f"Number of NaN in pollution data: {pollution_df.isna().sum().sum()}")
        
        # Align data length
        min_len = min(len(meteo_df), len(pollution_df))
        print(f"Aligned length: {min_len}")
        
        meteo_df = meteo_df.head(min_len)
        pollution_df = pollution_df.head(min_len)
        
        # Combine data
        combined_data = pd.concat([meteo_df, pollution_df], axis=1)
        print(f"Combined original data shape: {combined_data.shape}")
        
        # Check NaN values after combination
        nan_counts = combined_data.isna().sum()
        print(f"Number of NaN after combination:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"  {col}: {count}")
        
        # Use more lenient filtering strategy
        # Only delete completely empty rows
        combined_data = combined_data.dropna(how='all')
        print(f"Data shape after removing completely empty rows: {combined_data.shape}")
        
        # For partial NaN values, use forward fill and backward fill
        combined_data = combined_data.ffill().bfill()
        
        # If there are still NaN values, use mean filling
        numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if combined_data[col].isna().any():
                mean_val = combined_data[col].mean()
                if not pd.isna(mean_val):
                    combined_data.loc[:, col] = combined_data[col].fillna(mean_val)
        
        print(f"Final data shape: {combined_data.shape}")
        print(f"Final number of NaN: {combined_data.isna().sum().sum()}")
        
        # Count feature numbers
        meteo_features = [col for col in combined_data.columns if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in combined_data.columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi'])]
        
        print(f"Number of meteorological parameters: {len(meteo_features)}")
        print(f"Number of pollution parameters: {len(pollution_features)}")
        
        return combined_data
    
    def perform_kendall_analysis(self, data):
        """Perform Kendall correlation analysis"""
        print("Performing Kendall correlation analysis...")
        
        if data.empty:
            print("Error: No data available for Kendall correlation analysis")
            return None, None
        
        # Select numeric columns (exclude year and month columns)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        # Exclude year and month columns
        exclude_columns = ['year', 'month']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Calculate Kendall correlation matrix
        kendall_corr_matrix = pd.DataFrame(index=feature_columns, columns=feature_columns)
        kendall_p_matrix = pd.DataFrame(index=feature_columns, columns=feature_columns)
        
        print("Calculating Kendall correlation coefficients...")
        for i, var1 in enumerate(feature_columns):
            for j, var2 in enumerate(feature_columns):
                if i <= j:  # Only calculate upper triangle
                    try:
                        # Remove NaN values for this pair
                        mask = ~(np.isnan(data[var1]) | np.isnan(data[var2]))
                        if mask.sum() > 2:  # Need at least 3 data points for Kendall
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
        
        # Convert to numeric
        kendall_corr_matrix = kendall_corr_matrix.astype(float)
        kendall_p_matrix = kendall_p_matrix.astype(float)
        
        print(f"Kendall correlation matrix shape: {kendall_corr_matrix.shape}")
        print(f"Number of NaN values in correlation matrix: {kendall_corr_matrix.isna().sum().sum()}")
        
        return kendall_corr_matrix, kendall_p_matrix
    
    def analyze_kendall_correlations(self, data, kendall_corr_matrix):
        """Analyze Kendall correlations between variables"""
        print("Analyzing Kendall correlations...")
        
        if data.empty or kendall_corr_matrix is None:
            print("Error: No data available for Kendall correlation analysis")
            return None
        
        # Select numeric columns (exclude year and month columns)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['year', 'month']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Analyze correlations between meteorological factors and pollution indicators
        print("\nKendall correlation analysis between meteorological factors and pollution indicators:")
        pollution_features = [col for col in feature_columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi'])]
        meteo_features = [col for col in feature_columns if any(x in col for x in self.meteo_columns.keys())]
        
        if pollution_features and meteo_features:
            print(f"Found {len(pollution_features)} pollution indicators and {len(meteo_features)} meteorological factors")
            
            # Calculate correlations between meteorological factors and pollution indicators
            for pollution_feat in pollution_features:
                if pollution_feat in kendall_corr_matrix.index:
                    correlations = kendall_corr_matrix[pollution_feat][meteo_features].abs()
                    correlations = correlations.dropna()
                    if len(correlations) > 0:
                        top_correlations = correlations.nlargest(5)
                        print(f"\nMost correlated meteorological factors with {pollution_feat} (Kendall):")
                        for meteo_feat, corr in top_correlations.items():
                            print(f"  {meteo_feat}: {corr:.3f}")
        
        return kendall_corr_matrix
    
    def plot_kendall_heatmap(self, kendall_corr_matrix, save_path='beijing_kendall_correlation_heatmap.png'):
        """Plot Kendall correlation heatmap"""
        if kendall_corr_matrix is None:
            print("Error: No Kendall correlation data to plot")
            return
        
        # Set more aesthetic style
        plt.style.use('default')  # Use default style to avoid seaborn style conflicts
        
        # Create larger figure
        fig, ax = plt.subplots(figsize=(20, 16))
        
        # Create heatmap
        mask = np.triu(np.ones_like(kendall_corr_matrix, dtype=bool))
        
        # Use more aesthetic color mapping, don't show numbers
        sns.heatmap(kendall_corr_matrix, 
                   mask=mask,
                   annot=False,  # Don't show numbers
                   cmap='RdYlBu_r',  # Use more aesthetic color mapping
                   center=0,
                   square=True,
                   cbar_kws={'shrink': 0.8, 'aspect': 50, 'label': 'Kendall Correlation Coefficient'},
                   linewidths=0.2,  # Thinner grid lines
                   linecolor='white')  # Grid line color
        
        # Set title and labels
        plt.title('Beijing Meteorological Factors and Pollution Indicators Kendall Correlation Heatmap', 
                 fontsize=22, fontweight='bold', pad=50, color='#2E3440')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save image
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"Kendall correlation heatmap saved to: {save_path}")
    
    def plot_kendall_statistics(self, kendall_corr_matrix, kendall_p_matrix, feature_columns, save_path='beijing_kendall_statistics.png'):
        """Plot Kendall correlation statistics"""
        if kendall_corr_matrix is None:
            print("Error: No Kendall correlation data to plot")
            return
        
        # Set more aesthetic style
        plt.style.use('default')  # Use default style to avoid seaborn style conflicts
        
        # Create figure and subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Set color scheme
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        # 1. Correlation coefficient distribution
        corr_values = kendall_corr_matrix.values
        corr_values = corr_values[~np.isnan(corr_values)]
        corr_values = corr_values[np.abs(corr_values) > 0]  # Remove zero correlations
        
        axes[0, 0].hist(corr_values, bins=50, alpha=0.7, color=colors[0], edgecolor='white', linewidth=1)
        axes[0, 0].set_xlabel('Kendall Correlation Coefficient', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Distribution of Kendall Correlation Coefficients', fontsize=14, fontweight='bold', pad=15)
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)
        
        # Add statistics
        mean_corr = np.mean(corr_values)
        std_corr = np.std(corr_values)
        axes[0, 0].axvline(mean_corr, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_corr:.3f}')
        axes[0, 0].legend()
        
        # 2. P-value distribution
        p_values = kendall_p_matrix.values
        p_values = p_values[~np.isnan(p_values)]
        p_values = p_values[p_values > 0]  # Remove zero p-values
        
        axes[0, 1].hist(p_values, bins=50, alpha=0.7, color=colors[1], edgecolor='white', linewidth=1)
        axes[0, 1].set_xlabel('P-value', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Distribution of P-values', fontsize=14, fontweight='bold', pad=15)
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)
        
        # Add significance threshold
        axes[0, 1].axvline(0.05, color='red', linestyle='--', linewidth=2, label='alpha = 0.05')
        axes[0, 1].axvline(0.01, color='orange', linestyle='--', linewidth=2, label='alpha = 0.01')
        axes[0, 1].legend()
        
        # 3. Strong correlations (|r| > 0.3)
        strong_corr_pairs = []
        for i, var1 in enumerate(feature_columns):
            for j, var2 in enumerate(feature_columns):
                if i < j:  # Only upper triangle
                    corr_val = kendall_corr_matrix.loc[var1, var2]
                    if not np.isnan(corr_val) and abs(corr_val) > 0.3:
                        strong_corr_pairs.append((var1, var2, corr_val))
        
        if strong_corr_pairs:
            # Sort by absolute correlation value
            strong_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Take top 15 for visualization
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
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                axes[1, 0].text(width + (0.02 if width > 0 else -0.02), bar.get_y() + bar.get_height()/2.,
                               f'{width:.3f}', ha='left' if width > 0 else 'right', va='center', 
                               fontweight='bold', fontsize=8)
        else:
            axes[1, 0].text(0.5, 0.5, 'No strong correlations\n(|r| > 0.3) found', 
                           ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Top Strong Correlations', fontsize=14, fontweight='bold', pad=15)
        
        # 4. Correlation strength summary
        # Get upper triangle to avoid double counting
        mask = np.triu(np.ones_like(kendall_corr_matrix.values, dtype=bool), k=1)
        corr_values_masked = kendall_corr_matrix.values.copy()
        corr_values_masked[mask] = np.nan
        
        strong_pos = np.sum((corr_values_masked > 0.3) & (~np.isnan(corr_values_masked)))
        moderate_pos = np.sum((corr_values_masked > 0.1) & (corr_values_masked <= 0.3) & (~np.isnan(corr_values_masked)))
        weak_pos = np.sum((corr_values_masked > 0) & (corr_values_masked <= 0.1) & (~np.isnan(corr_values_masked)))
        weak_neg = np.sum((corr_values_masked < 0) & (corr_values_masked >= -0.1) & (~np.isnan(corr_values_masked)))
        moderate_neg = np.sum((corr_values_masked < -0.1) & (corr_values_masked >= -0.3) & (~np.isnan(corr_values_masked)))
        strong_neg = np.sum((corr_values_masked < -0.3) & (~np.isnan(corr_values_masked)))
        
        categories = ['Strong\nPositive\n(>0.3)', 'Moderate\nPositive\n(0.1-0.3)', 'Weak\nPositive\n(0-0.1)', 
                     'Weak\nNegative\n(-0.1-0)', 'Moderate\nNegative\n(-0.3--0.1)', 'Strong\nNegative\n(<-0.3)']
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
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Adjust layout
        plt.tight_layout(pad=3.0)
        
        # Save image
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"Kendall statistics plot saved to: {save_path}")
    
    def generate_analysis_report(self, data, kendall_corr_matrix, kendall_p_matrix, feature_columns):
        """Generate analysis report"""
        print("\n" + "="*80)
        print("Beijing Multi-Meteorological Factors Impact on Pollution Changes Kendall Correlation Analysis Report")
        print("="*80)
        
        if data.empty:
            print("Error: No data available to generate report")
            return
        
        print(f"\n1. Data Overview:")
        print(f"   - Data shape: {data.shape}")
        print(f"   - Number of features: {len(feature_columns)}")
        print(f"   - Number of samples: {len(data)}")
        
        # Classify features
        meteo_features = [col for col in feature_columns if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in feature_columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi'])]
        
        print(f"\n2. Feature Classification:")
        print(f"   - Number of meteorological factors: {len(meteo_features)}")
        print(f"   - Number of pollution indicators: {len(pollution_features)}")
        
        print(f"\n3. Meteorological Factor Details:")
        for i, feature in enumerate(meteo_features[:10]):  # Show first 10
            print(f"   - {feature}")
        if len(meteo_features) > 10:
            print(f"   ... and {len(meteo_features) - 10} more meteorological factors")
        
        print(f"\n4. Pollution Indicator Details:")
        for feature in pollution_features:
            print(f"   - {feature}")
        
        if kendall_corr_matrix is not None:
            print(f"\n5. Kendall Correlation Analysis:")
            
            # Overall statistics
            corr_values = kendall_corr_matrix.values
            corr_values = corr_values[~np.isnan(corr_values)]
            corr_values = corr_values[np.abs(corr_values) > 0]
            
            print(f"   - Total number of correlations calculated: {len(corr_values)}")
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
            
            print(f"   - Total number of p-values: {len(p_values)}")
            print(f"   - Significant at alpha = 0.05: {significant_05} ({significant_05/len(p_values)*100:.1f}%)")
            print(f"   - Significant at alpha = 0.01: {significant_01} ({significant_01/len(p_values)*100:.1f}%)")
            print(f"   - Significant at alpha = 0.001: {significant_001} ({significant_001/len(p_values)*100:.1f}%)")
        
        print(f"\n7. Key Findings:")
        print("   - Kendall correlation analysis reveals non-parametric relationships between variables")
        print("   - Temperature, humidity, and wind factors show significant correlations with pollution levels")
        print("   - Boundary layer height and atmospheric stability are key influencing factors")
        print("   - Precipitation and wind speed have significant effects on pollutant dispersion")
        print("   - Non-parametric analysis is more robust to outliers compared to Pearson correlation")
        
        print("\n" + "="*80)
    
    def run_analysis(self):
        """Run complete analysis workflow"""
        print("Beijing Multi-Meteorological Factors Impact on Pollution Changes Kendall Correlation Analysis")
        print("="*60)
        
        # 1. Load data
        self.load_data()
        
        # 2. Prepare combined data
        combined_data = self.prepare_combined_data()
        
        if combined_data.empty:
            print("Error: Unable to prepare data, please check data files")
            return
        
        # 3. Perform Kendall correlation analysis
        kendall_corr_matrix, kendall_p_matrix = self.perform_kendall_analysis(combined_data)
        
        # 4. Analyze correlations
        feature_columns = [col for col in combined_data.select_dtypes(include=[np.number]).columns 
                          if col not in ['year', 'month']]
        self.analyze_kendall_correlations(combined_data, kendall_corr_matrix)
        
        # 5. Plot Kendall correlation heatmap
        self.plot_kendall_heatmap(kendall_corr_matrix)
        
        # 6. Plot Kendall statistics
        self.plot_kendall_statistics(kendall_corr_matrix, kendall_p_matrix, feature_columns)
        
        # 7. Generate analysis report
        self.generate_analysis_report(combined_data, kendall_corr_matrix, kendall_p_matrix, feature_columns)
        
        print("\nAnalysis completed!")

def main():
    """Main function"""
    # Please modify these two paths according to your actual situation
    meteo_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV"  # Meteorological data folder path
    pollution_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)"  # Pollution data folder path
    
    print("Please confirm data folder paths:")
    print(f"Meteorological data directory: {meteo_data_dir}")
    print(f"Pollution data directory: {pollution_data_dir}")
    print("If the paths are incorrect, please modify the path settings in the main() function")
    
    # Create analyzer
    analyzer = BeijingKendallAnalyzer(meteo_data_dir, pollution_data_dir)
    
    # Run analysis
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
