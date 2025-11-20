import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

warnings.filterwarnings('ignore')

# Font configuration
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]  # English font
plt.rcParams['axes.unicode_minus'] = False  # Correctly display minus sign
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (16, 8)

class MeteorologicalAnalyzer:
    """Beijing Meteorological Data Analyzer"""
    
    def __init__(self, data_dir=".", max_workers=4):
        self.data_dir = data_dir
        self.meteo_data = pd.DataFrame()
        self.max_workers = max_workers  # Number of threads for parallel processing
        self.print_lock = Lock()  # Thread-safe print lock
        
        # Chinese parameter name mapping
        self.param_names_cn = {
            't2m': '2m Temperature',
            'd2m': '2m Dewpoint Temperature',
            'blh': 'Boundary Layer Height',
            'cvh': 'High Vegetation Cover',
            'avg_tprate': 'Mean Precipitation Rate',
            'u10': '10m U Wind Component',
            'v10': '10m V Wind Component',
            'u100': '100m U Wind Component',
            'v100': '100m V Wind Component',
            'lsm': 'Land Sea Mask',
            'cvl': 'Low Vegetation Cover',
            'mn2t': 'Minimum 2m Temperature',
            'sp': 'Surface Pressure',
            'sd': 'Snow Depth',
            'str': 'Surface Net Thermal Radiation',
            'tisr': 'TOA Incident Solar Radiation',
            'tcwv': 'Total Column Water Vapour',
            'tp': 'Total Precipitation'
        }
        
        # English parameter name mapping
        self.param_names_en = {
            't2m': '2m Temperature',
            'd2m': '2m Dewpoint Temperature',
            'blh': 'Boundary Layer Height',
            'cvh': 'High Vegetation Cover',
            'avg_tprate': 'Mean Precipitation Rate',
            'u10': '10m U Wind Component',
            'v10': '10m V Wind Component',
            'u100': '100m U Wind Component',
            'v100': '100m V Wind Component',
            'lsm': 'Land Sea Mask',
            'cvl': 'Low Vegetation Cover',
            'mn2t': 'Minimum 2m Temperature',
            'sp': 'Surface Pressure',
            'sd': 'Snow Depth',
            'str': 'Surface Net Thermal Radiation',
            'tisr': 'TOA Incident Solar Radiation',
            'tcwv': 'Total Column Water Vapour',
            'tp': 'Total Precipitation'
        }
        
        # Parameter units
        self.param_units = {
            't2m': '°C',
            'd2m': '°C',
            'blh': 'm',
            'cvh': '',
            'avg_tprate': 'mm/h',
            'u10': 'm/s',
            'v10': 'm/s',
            'u100': 'm/s',
            'v100': 'm/s',
            'lsm': '',
            'cvl': '',
            'mn2t': '°C',
            'sp': 'Pa',
            'sd': 'm',
            'str': 'J/m²',
            'tisr': 'J/m²',
            'tcwv': 'kg/m²',
            'tp': 'mm'
        }
        
    def find_meteorological_files(self):
        """Recursively search for all meteorological data CSV files"""
        print(f"Searching folder: {self.data_dir}")
        
        # Recursively search for all CSV files
        search_pattern = os.path.join(self.data_dir, "**", "*.csv")
        all_files = glob.glob(search_pattern, recursive=True)
        
        # Filter meteorological data files that match naming convention (YYYYMM.csv format)
        meteo_files = []
        for filepath in all_files:
            filename = os.path.basename(filepath)
            # Check filename format: YYYYMM.csv (6 digits)
            name_without_ext = filename.replace('.csv', '')
            if name_without_ext.isdigit() and len(name_without_ext) == 6:
                meteo_files.append(filepath)
        
        # Sort by filename
        meteo_files.sort()
        
        print(f"Found {len(meteo_files)} meteorological data files")
        if meteo_files:
            print(f"First file: {os.path.basename(meteo_files[0])}")
            print(f"Last file: {os.path.basename(meteo_files[-1])}")
        
        return meteo_files
    
    def aggregate_spatial_data(self, df):
        """
        Aggregate spatial data, convert multi-dimensional data to time series
        
        Args:
            df: Multi-dimensional dataframe containing time, latitude, longitude columns
            
        Returns:
            pd.DataFrame: Time-aggregated dataframe
        """
        try:
            # Group by time and calculate spatial average
            if 'time' in df.columns:
                # Convert time column to datetime type
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
    
    def process_single_file(self, filepath):
        """Process a single meteorological data file"""
        try:
            # Extract year-month information from filename
            filename = os.path.basename(filepath)
            year_month_str = filename.replace('.csv', '')
            
            # Read CSV file
            try:
                df = pd.read_csv(filepath, encoding='utf-8', comment='#')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(filepath, encoding='gbk', comment='#')
                except UnicodeDecodeError:
                    df = pd.read_csv(filepath, encoding='latin-1', comment='#')
            except pd.errors.ParserError:
                try:
                    df = pd.read_csv(filepath, encoding='utf-8', skiprows=4)
                except:
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8', comment='#', on_bad_lines='skip')
                    except:
                        raise
            
            if df.empty:
                with self.print_lock:
                    print(f"Warning: File is empty - {filename}")
                return None
            
            # Process multi-dimensional data (if exists)
            if 'time' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
                df = self.aggregate_spatial_data(df)
            
            # Print debug information (thread-safe)
            with self.print_lock:
                print(f"  Processing file: {filename}")
                print(f"  Data shape: {df.shape}")
                print(f"  Columns: {list(df.columns)[:5]}..." if len(df.columns) > 5 else f"  Columns: {list(df.columns)}")
            
            # Identify available meteorological parameter columns
            available_params = []
            for param in self.param_names_cn.keys():
                if param in df.columns:
                    available_params.append(param)
            
            if not available_params:
                with self.print_lock:
                    print(f"  Warning: No known meteorological parameter columns found")
                return None
            
            with self.print_lock:
                print(f"  Found meteorological parameters: {available_params}")
            
            # Process each meteorological parameter
            result = {
                'year_month': year_month_str,
                'year': int(year_month_str[:4]),
                'month': int(year_month_str[4:6]),
            }
            
            for param in available_params:
                try:
                    values = df[param].values
                    
                    # Process temperature data (Kelvin to Celsius)
                    if param in ['t2m', 'd2m', 'mn2t']:
                        # Check if already in Celsius
                        if values.max() > 100:  # If max value > 100, might be Kelvin
                            values = values - 273.15
                    
                    # Remove NaN values
                    valid_values = values[~np.isnan(values)]
                    
                    if len(valid_values) == 0:
                        continue
                    
                    # Calculate statistics
                    result[f'{param}_mean'] = np.nanmean(valid_values)
                    result[f'{param}_std'] = np.nanstd(valid_values)
                    result[f'{param}_min'] = np.nanmin(valid_values)
                    result[f'{param}_max'] = np.nanmax(valid_values)
                    result[f'{param}_median'] = np.nanmedian(valid_values)
                    
                except Exception as e:
                    with self.print_lock:
                        print(f"  Error processing parameter {param}: {e}")
                    continue
            
            with self.print_lock:
                print(f"  Successfully processed {len(available_params)} meteorological parameters")
            
            return result
            
        except Exception as e:
            with self.print_lock:
                print(f"Error: Failed to process file {filepath}: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def load_all_data(self):
        """Load all meteorological data using multi-threading"""
        print("\nStarting to load meteorological data using multi-threading...")
        print(f"Using {self.max_workers} worker threads")
        
        all_files = self.find_meteorological_files()
        
        if not all_files:
            print("Error: No meteorological data files found")
            return
        
        all_data = []
        completed_count = 0
        total_files = len(all_files)
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {executor.submit(self.process_single_file, filepath): filepath 
                            for filepath in all_files}
            
            # Process tasks in completion order
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                completed_count += 1
                
                try:
                    file_data = future.result()
                    if file_data:
                        all_data.append(file_data)
                    
                    with self.print_lock:
                        print(f"Progress: [{completed_count}/{total_files}] Completed {os.path.basename(filepath)}")
                        
                except Exception as e:
                    with self.print_lock:
                        print(f"Error processing file {os.path.basename(filepath)}: {e}")
        
        if not all_data:
            print("Error: Failed to load any data")
            self.meteo_data = pd.DataFrame()
            return
        
        # Convert to DataFrame
        self.meteo_data = pd.DataFrame(all_data)
        
        # Fix data aggregation issue: group by year_month and merge data
        print("\nFixing data aggregation issue...")
        self._fix_data_aggregation()
        
        # Sort by time
        self.meteo_data = self.meteo_data.sort_values(['year', 'month']).reset_index(drop=True)
        
        # Create datetime column
        self.meteo_data['date'] = pd.to_datetime(
            self.meteo_data['year'].astype(str) + '-' + 
            self.meteo_data['month'].astype(str).str.zfill(2) + '-01'
        )
        
        print(f"\nData loading completed!")
        print(f"Total {len(self.meteo_data)} monthly records")
        print(f"Time range: {self.meteo_data['date'].min()} to {self.meteo_data['date'].max()}")
        
        # Display available meteorological parameters
        param_cols = [col for col in self.meteo_data.columns if col.endswith('_mean')]
        available_params = [col.replace('_mean', '') for col in param_cols]
        print(f"Available meteorological parameters: {', '.join([self.param_names_cn.get(p, p) for p in available_params])}")
    
    def _fix_data_aggregation(self):
        """Fix data aggregation issue - merge data scattered across multiple rows into single rows"""
        print("Fixing data aggregation issue...")
        
        # Get all parameter columns (excluding metadata columns)
        param_columns = [col for col in self.meteo_data.columns 
                        if col not in ['year_month', 'year', 'month', 'date', 'wind_speed_10m', 'wind_speed_100m', 'season']]
        
        # Group by year_month and aggregate data
        aggregated_data = []
        
        for year_month in self.meteo_data['year_month'].unique():
            year_month_data = self.meteo_data[self.meteo_data['year_month'] == year_month]
            
            # Create a row containing all parameters
            row = {
                'year_month': year_month,
                'year': year_month_data['year'].iloc[0],
                'month': year_month_data['month'].iloc[0]
            }
            
            # For each parameter, find the first non-null value
            for col in param_columns:
                non_null_values = year_month_data[col].dropna()
                if len(non_null_values) > 0:
                    row[col] = non_null_values.iloc[0]
                else:
                    row[col] = np.nan
            
            aggregated_data.append(row)
        
        # Update dataframe
        self.meteo_data = pd.DataFrame(aggregated_data)
        print(f"Data aggregation completed, aggregated from {len(self.meteo_data) * len(param_columns)} rows to {len(self.meteo_data)} rows")
    
    def plot_temperature_timeseries(self, save_path='temperature_timeseries.png'):
        """Plot temperature time series chart"""
        if self.meteo_data.empty:
            print("Error: No data available for plotting")
            return
        
        print("\nStarting to plot temperature time series...")
        
        # Check available temperature parameters
        temp_params = ['t2m', 'd2m', 'mn2t']
        available_temps = [p for p in temp_params if f'{p}_mean' in self.meteo_data.columns]
        
        if not available_temps:
            print("Warning: No temperature data available")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(20, 8))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        for i, param in enumerate(available_temps):
            mean_col = f'{param}_mean'
            min_col = f'{param}_min'
            max_col = f'{param}_max'
            
            # Plot mean value line
            ax.plot(self.meteo_data['date'], self.meteo_data[mean_col],
                    linewidth=2.5, color=colors[i], alpha=0.9, 
                    label=f'{self.param_names_en[param]} (Mean)', marker='o', markersize=4)
            
            # If min and max values exist, plot range fill
            if min_col in self.meteo_data.columns and max_col in self.meteo_data.columns:
                ax.fill_between(self.meteo_data['date'], 
                               self.meteo_data[min_col], 
                               self.meteo_data[max_col],
                               alpha=0.2, color=colors[i])
        
        # Add zero degree reference line
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='0°C Reference Line')
        
        # Set labels and title
        ax.set_xlabel('Time', fontsize=14, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=14, fontweight='bold')
        ax.set_title('Beijing Temperature Time Series', 
                     fontsize=18, fontweight='bold', pad=30)
        
        # Set grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add legend - move to upper right to avoid overlap with title
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9, 
                 bbox_to_anchor=(0.98, 0.98))
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add statistics text box - move to lower left to avoid overlap
        if 't2m_mean' in self.meteo_data.columns:
            stats_text = f"2m Temperature Statistics:\n"
            stats_text += f"Sample: {len(self.meteo_data)} months\n"
            stats_text += f"Mean: {self.meteo_data['t2m_mean'].mean():.2f} °C\n"
            stats_text += f"Median: {self.meteo_data['t2m_mean'].median():.2f} °C\n"
            stats_text += f"Max: {self.meteo_data['t2m_mean'].max():.2f} °C\n"
            stats_text += f"Min: {self.meteo_data['t2m_mean'].min():.2f} °C"
            
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Temperature time series chart saved to: {save_path}")
        
        plt.show()
    
    def plot_wind_speed_analysis(self, save_path='wind_speed_analysis.png'):
        """Plot wind speed analysis chart"""
        if self.meteo_data.empty:
            print("Error: No data available for plotting")
            return
        
        print("\nStarting to plot wind speed analysis chart...")
        
        # Check available wind speed parameters
        wind_params = ['u10', 'v10', 'u100', 'v100']
        available_winds = [p for p in wind_params if f'{p}_mean' in self.meteo_data.columns]
        
        if not available_winds:
            print("Warning: No wind speed data available")
            return
        
        # Calculate wind speed magnitude (if U and V components exist)
        if 'u10_mean' in self.meteo_data.columns and 'v10_mean' in self.meteo_data.columns:
            self.meteo_data['wind_speed_10m'] = np.sqrt(
                self.meteo_data['u10_mean']**2 + self.meteo_data['v10_mean']**2
            )
        
        if 'u100_mean' in self.meteo_data.columns and 'v100_mean' in self.meteo_data.columns:
            self.meteo_data['wind_speed_100m'] = np.sqrt(
                self.meteo_data['u100_mean']**2 + self.meteo_data['v100_mean']**2
            )
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(20, 12))
        
        # Subplot 1: Wind speed time series
        if 'wind_speed_10m' in self.meteo_data.columns:
            axes[0].plot(self.meteo_data['date'], self.meteo_data['wind_speed_10m'],
                        linewidth=2.5, color='#3498db', alpha=0.9, 
                        label='10m Wind Speed', marker='o', markersize=4)
        
        if 'wind_speed_100m' in self.meteo_data.columns:
            axes[0].plot(self.meteo_data['date'], self.meteo_data['wind_speed_100m'],
                        linewidth=2.5, color='#e74c3c', alpha=0.9, 
                        label='100m Wind Speed', marker='s', markersize=4)
        
        axes[0].set_xlabel('Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Wind Speed (m/s)', fontsize=14, fontweight='bold')
        axes[0].set_title('Beijing Wind Speed Time Series', 
                         fontsize=16, fontweight='bold', pad=15)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(loc='upper right', fontsize=11)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Subplot 2: Wind speed distribution histogram
        wind_data = []
        wind_labels = []
        
        if 'wind_speed_10m' in self.meteo_data.columns:
            wind_data.append(self.meteo_data['wind_speed_10m'].dropna())
            wind_labels.append('10m Wind Speed')
        
        if 'wind_speed_100m' in self.meteo_data.columns:
            wind_data.append(self.meteo_data['wind_speed_100m'].dropna())
            wind_labels.append('100m Wind Speed')
        
        if wind_data:
            axes[1].hist(wind_data, bins=30, label=wind_labels, 
                        color=['#3498db', '#e74c3c'][:len(wind_data)], 
                        alpha=0.7, edgecolor='white', linewidth=1)
            
            axes[1].set_xlabel('Wind Speed (m/s)', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Frequency', fontsize=14, fontweight='bold')
            axes[1].set_title('Wind Speed Distribution', 
                             fontsize=16, fontweight='bold', pad=15)
            axes[1].grid(True, alpha=0.3, linestyle='--')
            axes[1].legend(loc='upper right', fontsize=11)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Wind speed analysis chart saved to: {save_path}")
        
        plt.show()
    
    def plot_precipitation_analysis(self, save_path='precipitation_analysis.png'):
        """Plot precipitation analysis chart"""
        if self.meteo_data.empty:
            print("Error: No data available for plotting")
            return
        
        print("\nStarting to plot precipitation analysis chart...")
        
        # Check available precipitation parameters
        precip_params = ['tp', 'avg_tprate']
        available_precips = [p for p in precip_params if f'{p}_mean' in self.meteo_data.columns]
        
        if not available_precips:
            print("Warning: No precipitation data available")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(20, 12))
        
        # Subplot 1: Precipitation time series
        if 'tp_mean' in self.meteo_data.columns:
            axes[0].bar(self.meteo_data['date'], self.meteo_data['tp_mean'],
                       width=20, color='#3498db', alpha=0.7, label='Total Precipitation')
            
            axes[0].set_xlabel('Time', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Precipitation (mm)', fontsize=14, fontweight='bold')
            axes[0].set_title('Beijing Precipitation Time Series', 
                             fontsize=16, fontweight='bold', pad=15)
            axes[0].grid(True, alpha=0.3, linestyle='--', axis='y')
            axes[0].legend(loc='upper right', fontsize=11)
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Subplot 2: Precipitation rate time series
        if 'avg_tprate_mean' in self.meteo_data.columns:
            axes[1].plot(self.meteo_data['date'], self.meteo_data['avg_tprate_mean'],
                        linewidth=2.5, color='#2ecc71', alpha=0.9, 
                        label='Mean Precipitation Rate', marker='o', markersize=4)
            
            axes[1].set_xlabel('Time', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Precipitation Rate (mm/h)', fontsize=14, fontweight='bold')
            axes[1].set_title('Beijing Mean Precipitation Rate Time Series', 
                             fontsize=16, fontweight='bold', pad=15)
            axes[1].grid(True, alpha=0.3, linestyle='--')
            axes[1].legend(loc='upper right', fontsize=11)
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Precipitation analysis chart saved to: {save_path}")
        
        plt.show()
    
    def plot_seasonal_analysis(self, save_path='seasonal_analysis.png'):
        """Plot seasonal analysis chart"""
        if self.meteo_data.empty:
            print("Error: No data available for plotting")
            return
        
        print("\nStarting to plot seasonal analysis chart...")
        
        # Add season column
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'
        
        self.meteo_data['season'] = self.meteo_data['month'].apply(get_season)
        
        # Create figure (2x2 subplots) with larger height to accommodate spacing
        fig, axes = plt.subplots(2, 2, figsize=(20, 18))
        
        season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
        
        # Subplot 1: Temperature seasonal distribution box plot
        if 't2m_mean' in self.meteo_data.columns:
            try:
                # Remove NaN values
                plot_data = self.meteo_data[['season', 't2m_mean']].dropna()
                if not plot_data.empty:
                    sns.boxplot(data=plot_data, x='season', y='t2m_mean', 
                               order=season_order, palette='Set2', ax=axes[0, 0])
                    axes[0, 0].set_xlabel('Season', fontsize=12, fontweight='bold')
                    axes[0, 0].set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
                    axes[0, 0].set_title('2m Temperature Seasonal Distribution', 
                                        fontsize=14, fontweight='bold', pad=20)
                    axes[0, 0].grid(True, alpha=0.3, linestyle='--', axis='y')
            except Exception as e:
                print(f"Error plotting temperature box plot: {e}")
                axes[0, 0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[0, 0].transAxes)
        
        # Subplot 2: Wind speed seasonal distribution box plot
        if 'wind_speed_10m' in self.meteo_data.columns:
            try:
                # Remove NaN values
                plot_data = self.meteo_data[['season', 'wind_speed_10m']].dropna()
                if not plot_data.empty:
                    sns.boxplot(data=plot_data, x='season', y='wind_speed_10m', 
                               order=season_order, palette='Set3', ax=axes[0, 1])
                    axes[0, 1].set_xlabel('Season', fontsize=12, fontweight='bold')
                    axes[0, 1].set_ylabel('Wind Speed (m/s)', fontsize=12, fontweight='bold')
                    axes[0, 1].set_title('10m Wind Speed Seasonal Distribution', 
                                        fontsize=14, fontweight='bold', pad=20)
                    axes[0, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
            except Exception as e:
                print(f"Error plotting wind speed box plot: {e}")
                axes[0, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[0, 1].transAxes)
        else:
            # If wind_speed_10m is not available, try using pressure data
            if 'sp_mean' in self.meteo_data.columns:
                try:
                    plot_data = self.meteo_data[['season', 'sp_mean']].dropna()
                    if not plot_data.empty:
                        sns.boxplot(data=plot_data, x='season', y='sp_mean', 
                                   order=season_order, palette='Set3', ax=axes[0, 1])
                        axes[0, 1].set_xlabel('Season', fontsize=12, fontweight='bold')
                        axes[0, 1].set_ylabel('Surface Pressure (Pa)', fontsize=12, fontweight='bold')
                        axes[0, 1].set_title('Surface Pressure Seasonal Distribution', 
                                            fontsize=14, fontweight='bold', pad=20)
                        axes[0, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
                except Exception as e:
                    print(f"Error plotting pressure box plot: {e}")
                    axes[0, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Subplot 3: Precipitation seasonal distribution box plot
        if 'tp_mean' in self.meteo_data.columns:
            try:
                plot_data = self.meteo_data[['season', 'tp_mean']].dropna()
                if not plot_data.empty:
                    sns.boxplot(data=plot_data, x='season', y='tp_mean', 
                               order=season_order, palette='Blues', ax=axes[1, 0])
                    axes[1, 0].set_xlabel('Season', fontsize=12, fontweight='bold')
                    axes[1, 0].set_ylabel('Precipitation (mm)', fontsize=12, fontweight='bold')
                    axes[1, 0].set_title('Precipitation Seasonal Distribution', 
                                        fontsize=14, fontweight='bold', pad=20)
                    axes[1, 0].grid(True, alpha=0.3, linestyle='--', axis='y')
            except Exception as e:
                print(f"Error plotting precipitation box plot: {e}")
                axes[1, 0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Subplot 4: Boundary layer height seasonal distribution box plot
        if 'blh_mean' in self.meteo_data.columns:
            try:
                plot_data = self.meteo_data[['season', 'blh_mean']].dropna()
                if not plot_data.empty:
                    sns.boxplot(data=plot_data, x='season', y='blh_mean', 
                               order=season_order, palette='Greens', ax=axes[1, 1])
                    axes[1, 1].set_xlabel('Season', fontsize=12, fontweight='bold')
                    axes[1, 1].set_ylabel('Boundary Layer Height (m)', fontsize=12, fontweight='bold')
                    axes[1, 1].set_title('Boundary Layer Height Seasonal Distribution', 
                                        fontsize=14, fontweight='bold', pad=20)
                    axes[1, 1].grid(True, alpha=0.3, linestyle='--', axis='y')
            except Exception as e:
                print(f"Error plotting boundary layer height box plot: {e}")
                axes[1, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        # Adjust layout with increased vertical spacing
        plt.tight_layout(h_pad=4.0, w_pad=2.0)
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Seasonal analysis chart saved to: {save_path}")
        
        plt.show()
    
    def save_data_summary(self, save_path='meteorological_summary.csv'):
        """Save data summary"""
        if self.meteo_data.empty:
            print("Error: No data available to save")
            return
        
        print("\nSaving data summary...")
        
        # Save complete data
        self.meteo_data.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"Data summary saved to: {save_path}")
        
        # Generate statistical summary
        summary_stats_path = save_path.replace('.csv', '_statistics.csv')
        
        # Get all numeric columns
        numeric_cols = self.meteo_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['year', 'month']]
        
        # Calculate statistical information
        summary_stats = self.meteo_data[numeric_cols].describe().T
        summary_stats['parameter'] = summary_stats.index
        summary_stats = summary_stats[['parameter', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        
        summary_stats.to_csv(summary_stats_path, index=False, encoding='utf-8-sig')
        print(f"Statistical summary saved to: {summary_stats_path}")
    
    def run_analysis(self):
        """Run complete analysis"""
        print("="*60)
        print("Beijing Meteorological Data Analysis")
        print("="*60)
        
        # Load data
        self.load_all_data()
        
        if self.meteo_data.empty:
            print("Error: Unable to load data, please check data file path")
            return
        
        # Plot temperature time series
        self.plot_temperature_timeseries()
        
        # Plot wind speed analysis
        self.plot_wind_speed_analysis()
        
        # Plot precipitation analysis
        self.plot_precipitation_analysis()
        
        # Plot seasonal analysis
        self.plot_seasonal_analysis()
        
        # Save data summary
        self.save_data_summary()
        
        print("\nAnalysis completed!")


def main():
    # Please modify data path according to actual situation
    # Example path (based on project structure, meteorological data is usually in these paths)
    data_dir = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV"
    
    # If the above path does not exist, try other possible paths
    if not os.path.exists(data_dir):
        # Try to find other possible paths
        possible_paths = [
            r"C:\Users\IU\Desktop\something\Short-Essay-Beijing\Format-Transfor",
            r"C:\Users\IU\Desktop\something\Graduation thesis",
            r"C:\Users\IU\Desktop\ERA5-Data",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_dir = path
                print(f"Using data path: {data_dir}")
                break
        else:
            print("Error: Meteorological data folder not found, please manually set data_dir variable")
            print("Please modify data_dir in main() function to your data folder path")
            return
    
    print(f"Data folder path: {data_dir}")
    print("If the path is incorrect, please modify the data_dir variable in the main() function\n")
    
    # Create analyzer and run
    # max_workers: Number of threads for parallel file processing (default: 4)
    # If you have more CPU cores, you can increase this value to speed up processing; if memory is limited, you can reduce this value
    analyzer = MeteorologicalAnalyzer(data_dir, max_workers=8)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()

