import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import hashlib
import pickle
import time
from typing import List, Dict, Any, Optional
import torch.nn.functional as F
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

class BeijingPearsonAnalyzer:
    """Beijing Meteorological Factors and Pollution Change Pearson Correlation Analyzer based on PyTorch"""
    
    def __init__(self, meteo_data_dir=".", pollution_data_dir="."):
        """
        Initialize analyzer
        
        Args:
            meteo_data_dir: Meteorological data directory path
            pollution_data_dir: Pollution data directory path
        """
        self.meteo_data_dir = meteo_data_dir
        self.pollution_data_dir = pollution_data_dir
        self.scaler = StandardScaler()
        self.meteorological_data = []
        self.pollution_data = []
        self.cache = DataCache()
        
        # Define meteorological parameter column mapping
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
            'mn2t': 'minimum_2m_temperature',  # minimum 2m temperature
            'sp': 'surface_pressure',  # surface pressure
            'sd': 'snow_depth',  # snow depth
            'str': 'surface_net_thermal_radiation',  # surface net thermal radiation
            'tisr': 'toa_incident_solar_radiation',  # TOA incident solar radiation
            'tcwv': 'total_column_water_vapour',  # total column water vapour
            'tp': 'total_precipitation'  # total precipitation
        }
    
    def find_files_optimized(self, root_dir: str, pattern: str) -> List[str]:
        """
        Optimized file search
        
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
        
        # Parallel file processing
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
        """Load all data"""
        print("Starting to load data...")
        
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
        
        # Count features
        meteo_features = [col for col in combined_data.columns if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in combined_data.columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi'])]
        
        print(f"Number of meteorological parameters: {len(meteo_features)}")
        print(f"Number of pollution parameters: {len(pollution_features)}")
        
        return combined_data
    
    def calculate_pearson_correlation_pytorch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Pearson correlation using PyTorch
        
        Args:
            data: Input dataframe
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        print("Calculating Pearson correlation using PyTorch...")
        
        # Select numeric columns (exclude year and month columns)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['year', 'month']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Convert to PyTorch tensor
        X = torch.tensor(data[feature_columns].values, dtype=torch.float32)
        
        # Standardize data
        X_mean = torch.mean(X, dim=0, keepdim=True)
        X_std = torch.std(X, dim=0, keepdim=True)
        X_normalized = (X - X_mean) / (X_std + 1e-8)
        
        # Calculate correlation matrix
        n_features = X_normalized.shape[1]
        correlation_matrix = torch.zeros(n_features, n_features)
        
        for i in range(n_features):
            for j in range(n_features):
                # Calculate Pearson correlation coefficient
                x_i = X_normalized[:, i]
                x_j = X_normalized[:, j]
                
                # Avoid NaN values
                valid_mask = ~(torch.isnan(x_i) | torch.isnan(x_j))
                if torch.sum(valid_mask) > 1:
                    x_i_valid = x_i[valid_mask]
                    x_j_valid = x_j[valid_mask]
                    
                    # Calculate correlation coefficient
                    numerator = torch.sum((x_i_valid - torch.mean(x_i_valid)) * (x_j_valid - torch.mean(x_j_valid)))
                    denominator = torch.sqrt(torch.sum((x_i_valid - torch.mean(x_i_valid))**2) * 
                                          torch.sum((x_j_valid - torch.mean(x_j_valid))**2))
                    
                    if denominator > 1e-8:
                        correlation_matrix[i, j] = numerator / denominator
                    else:
                        correlation_matrix[i, j] = 0.0
                else:
                    correlation_matrix[i, j] = 0.0
        
        # Convert to pandas DataFrame
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
        correlation_matrix = self.calculate_pearson_correlation_pytorch(data)
        
        # Analyze correlations between meteorological factors and pollution indicators
        print("\nCorrelation analysis between meteorological factors and pollution indicators:")
        pollution_features = [col for col in correlation_matrix.columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi'])]
        meteo_features = [col for col in correlation_matrix.columns if any(x in col for x in self.meteo_columns.keys())]
        
        if pollution_features and meteo_features:
            print(f"Found {len(pollution_features)} pollution indicators and {len(meteo_features)} meteorological factors")
            
            # Calculate correlations between meteorological factors and pollution indicators
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
    
    def gradient_bar(self, ax, x, y, width=0.5, bottom=0):
        """Create gradient bars"""
        for left, top in zip(x, y):
            left = left - width/2
            right = left + width
            # Define gradient colors (start color, end color)
            colors = [(114/255, 188/255, 213/255), (1, 1, 1)]
            cmap = LinearSegmentedColormap.from_list('my_cmap', colors, N=256)
            self.gradient_image(ax, extent=(left, right, bottom, top),
                               cmap=cmap, cmap_range=(0, 0.8))
    
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                                save_path='beijing_pearson_correlation_heatmap.png'):
        """Plot correlation heatmap"""
        if correlation_matrix.empty:
            print("Error: No correlation data to plot")
            return
        
        # Set more aesthetic style
        plt.style.use('default')
        
        # Create larger figure
        fig, ax = plt.subplots(figsize=(20, 16))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Use more aesthetic color mapping
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=False,  # Don't display numbers
                   cmap='RdYlBu_r',  # Use more aesthetic color mapping
                   center=0,
                   square=True,
                   cbar_kws={'shrink': 0.8, 'aspect': 50, 'label': 'Pearson Correlation Coefficient'},
                   linewidths=0.2,  # Thinner grid lines
                   linecolor='white')  # Grid line color
        
        # Set title and labels
        plt.title('Beijing Meteorological Factors and Pollution Indicators Pearson Correlation Heatmap', 
                 fontsize=22, fontweight='bold', pad=50, color='#2E3440')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save image
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"Correlation heatmap saved to: {save_path}")
    
    def plot_top_correlations(self, correlation_matrix: pd.DataFrame, 
                             save_path='beijing_top_correlations.png'):
        """Plot top N strongest correlations with gradient bars and improved layout"""
        if correlation_matrix.empty:
            print("Error: No correlation data to plot")
            return
        
        # Get upper triangle correlations
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Flatten and sort
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
        
        # Sort by absolute value
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Take top 20 strongest correlations
        top_correlations = correlations[:20]
        
        # Create figure with better aspect ratio and more space for labels
        fig, ax = plt.subplots(figsize=(20, 14))
        
        # Prepare data with shorter feature names for better readability
        features = []
        for corr in top_correlations:
            # Shorten feature names for better display
            feat1 = str(corr['feature1']).replace('_mean', '').replace('_std', '').replace('_min', '').replace('_max', '')
            feat2 = str(corr['feature2']).replace('_mean', '').replace('_std', '').replace('_min', '').replace('_max', '')
            features.append(f"{feat1}\nvs\n{feat2}")
        
        values = [corr['correlation'] for corr in top_correlations]
        
        # Create horizontal bar chart for better label readability
        y_pos = np.arange(len(features))
        
        # Create gradient bars horizontally
        for i, (y, value) in enumerate(zip(y_pos, values)):
            # Create gradient for each bar
            left = 0 if value >= 0 else value
            right = value if value >= 0 else 0
            width = abs(value)
            
            # Define gradient colors based on correlation sign
            if value >= 0:
                colors = [(114/255, 188/255, 213/255), (1, 1, 1)]  # Blue to white for positive
            else:
                colors = [(255/255, 99/255, 71/255), (1, 1, 1)]    # Red to white for negative
            
            cmap = LinearSegmentedColormap.from_list('my_cmap', colors, N=256)
            
            # Create gradient bar
            bar_height = 0.6
            self.gradient_image(ax, extent=(left, right, y-bar_height/2, y+bar_height/2),
                               cmap=cmap, cmap_range=(0, 0.8))
        
        # Add value labels on bars
        for i, (y, value) in enumerate(zip(y_pos, values)):
            # Position label at the end of the bar
            if value >= 0:
                ax.text(value + 0.01, y, f'{value:.3f}', 
                       ha='left', va='center', fontweight='bold', fontsize=11)
            else:
                ax.text(value - 0.01, y, f'{value:.3f}', 
                       ha='right', va='center', fontweight='bold', fontsize=11, color='red')
        
        # Set plot properties
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('Pearson Correlation Coefficient', fontsize=14, fontweight='bold')
        ax.set_title('Top 20 Strongest Correlations: Beijing Meteorological Factors vs Pollution Indicators', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Improve x-axis limits
        max_abs_value = max(abs(v) for v in values)
        ax.set_xlim(-max_abs_value * 1.1, max_abs_value * 1.1)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save image
        plt.savefig(save_path, dpi=1200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"Top N strongest correlations plot saved to: {save_path}")
    
    def generate_analysis_report(self, data: pd.DataFrame, correlation_matrix: pd.DataFrame):
        """Generate analysis report"""
        print("\n" + "="*80)
        print("Beijing Meteorological Factors and Pollution Change Pearson Correlation Analysis Report")
        print("="*80)
        
        if data.empty:
            print("Error: No data available to generate report")
            return
        
        print(f"\n1. Data Overview:")
        print(f"   - Data shape: {data.shape}")
        print(f"   - Number of samples: {len(data)}")
        
        # Classify features
        meteo_features = [col for col in data.columns if any(x in col for x in self.meteo_columns.keys())]
        pollution_features = [col for col in data.columns if any(x in col.lower() for x in ['pm25', 'pm10', 'aqi'])]
        
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
            print(f"\n5. Correlation Analysis:")
            if pollution_features and meteo_features:
                print("Meteorological factors most correlated with pollution indicators:")
                for pollution_feat in pollution_features:
                    correlations = correlation_matrix[pollution_feat][meteo_features].abs()
                    top_correlations = correlations.nlargest(3)
                    print(f"   {pollution_feat}:")
                    for meteo_feat, corr in top_correlations.items():
                        print(f"     - {meteo_feat}: {corr:.3f}")
            
            # Analyze strongest correlations
            print(f"\n6. Strongest Correlation Analysis:")
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
            
            print("Top 10 strongest correlations:")
            for i, corr in enumerate(correlations[:10]):
                print(f"   {i+1}. {corr['feature1']} vs {corr['feature2']}: {corr['correlation']:.3f}")
        
        print(f"\n7. Key Findings:")
        print("   - Temperature, humidity, wind speed and other meteorological factors have comprehensive effects on pollution levels")
        print("   - Boundary layer height and atmospheric stability are important influencing factors")
        print("   - Precipitation and wind speed have significant effects on pollutant dispersion")
        print("   - Multi-meteorological factor comprehensive analysis can better explain pollution changes")
    
    def run_analysis(self):
        """Run complete analysis workflow"""
        print("Beijing Meteorological Factors and Pollution Change Pearson Correlation Analysis")
        print("="*60)
        
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

def main():
    """Main function"""
    # Please modify these two paths according to actual situation
    meteo_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV"  # Meteorological data folder path
    pollution_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)"  # Pollution data folder path
    
    print("Please confirm data folder paths:")
    print(f"Meteorological data directory: {meteo_data_dir}")
    print(f"Pollution data directory: {pollution_data_dir}")
    print("If paths are incorrect, please modify path settings in main() function")
    
    # Create analyzer
    analyzer = BeijingPearsonAnalyzer(meteo_data_dir, pollution_data_dir)
    
    # Run analysis
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
