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
plt.rcParams['axes.unicode_minus'] = False  # Display minus sign properly
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (16, 8)

class PM25Analyzer:
    """Beijing PM2.5 Concentration Analyzer"""
    
    def __init__(self, data_dir=".", max_workers=4):
        self.data_dir = data_dir
        self.pm25_data = []
        self.max_workers = max_workers  # Number of threads for parallel processing
        self.print_lock = Lock()  # Lock for thread-safe printing
        
    def find_pollution_files(self):
        """Recursively search for all beijing_all_YYYYMMDD.csv files"""
        print(f"Searching folder: {self.data_dir}")
        
        # Recursively search for all CSV files matching the naming convention
        search_pattern = os.path.join(self.data_dir, "**", "beijing_all_*.csv")
        all_files = glob.glob(search_pattern, recursive=True)
        
        # Filter files matching YYYYMMDD format
        pollution_files = []
        for filepath in all_files:
            filename = os.path.basename(filepath)
            # Check file name format: beijing_all_YYYYMMDD.csv
            if filename.startswith('beijing_all_') and filename.endswith('.csv'):
                date_part = filename.replace('beijing_all_', '').replace('.csv', '')
                # Verify if it's an 8-digit number (YYYYMMDD)
                if date_part.isdigit() and len(date_part) == 8:
                    pollution_files.append(filepath)
        
        # Sort by filename
        pollution_files.sort()
        
        print(f"Found {len(pollution_files)} pollution data files")
        if pollution_files:
            print(f"First file: {os.path.basename(pollution_files[0])}")
            print(f"Last file: {os.path.basename(pollution_files[-1])}")
        
        return pollution_files
    
    def process_single_file(self, filepath):
        """Process a single pollution data file"""
        try:
            # Extract date from filename
            filename = os.path.basename(filepath)
            date_str = filename.replace('beijing_all_', '').replace('.csv', '')
            
            # Read CSV file
            df = pd.read_csv(filepath, encoding='utf-8')
            
            if df.empty:
                print(f"Warning: File is empty - {filename}")
                return None
            
            # Check required columns
            if 'type' not in df.columns:
                print(f"Warning: File missing 'type' column - {filename}")
                return None
            
            if 'hour' not in df.columns:
                print(f"Warning: File missing 'hour' column - {filename}")
                return None
            
            # Extract PM2.5 data rows
            pm25_rows = df[df['type'] == 'PM2.5'].copy()
            
            if pm25_rows.empty:
                print(f"Warning: No PM2.5 data in file - {filename}")
                return None
            
            # Identify station columns (exclude non-station columns)
            non_station_cols = ["date", "hour", "type"]
            station_cols = [col for col in pm25_rows.columns if col not in non_station_cols]
            
            # Print debug information (thread-safe)
            with self.print_lock:
                print(f"  File: {filename}")
                print(f"  Total columns: {len(df.columns)}, Stations: {len(station_cols)}, PM2.5 data rows: {len(pm25_rows)}")
            
            if not station_cols:
                with self.print_lock:
                    print(f"  Warning: No station columns found")
                return None
            
            # Create timestamp for each hour
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            
            results = []
            
            # Iterate through each row (each row represents one hour)
            for idx, row in pm25_rows.iterrows():
                try:
                    hour = int(row['hour'])
                    
                    # Validate hour range
                    if hour < 0 or hour > 23:
                        with self.print_lock:
                            print(f"  Warning: Invalid hour value {hour}, skipping")
                        continue
                    
                    # Get PM2.5 values from all stations and calculate average
                    station_values = row[station_cols].values
                    
                    # Convert to numeric type and calculate average
                    station_values = pd.to_numeric(station_values, errors='coerce')
                    pm25_avg = np.nanmean(station_values)
                    
                    # If valid value exists, add to results
                    if not np.isnan(pm25_avg):
                        timestamp = date_obj.replace(hour=hour)
                        results.append({
                            'datetime': timestamp,
                            'pm25': pm25_avg,
                            'date': date_str,
                            'hour': hour
                        })
                
                except (ValueError, KeyError) as e:
                    with self.print_lock:
                        print(f"  Warning: Error processing row: {e}")
                    continue
            
            if results:
                with self.print_lock:
                    print(f"  Successfully processed {len(results)} hours of data")
            
            return results
            
        except Exception as e:
            with self.print_lock:
                print(f"Error: Error processing file {filepath}: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def load_all_data(self):
        """Load all pollution data using multithreading"""
        print("\nStarting to load pollution data with multithreading...")
        print(f"Using {self.max_workers} worker threads")
        
        all_files = self.find_pollution_files()
        
        if not all_files:
            print("Error: No pollution data files found")
            return
        
        all_data = []
        completed_count = 0
        total_files = len(all_files)
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {executor.submit(self.process_single_file, filepath): filepath 
                            for filepath in all_files}
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                completed_count += 1
                
                try:
                    file_data = future.result()
                    if file_data:
                        all_data.extend(file_data)
                    
                    with self.print_lock:
                        print(f"Progress: [{completed_count}/{total_files}] Completed {os.path.basename(filepath)}")
                        
                except Exception as e:
                    with self.print_lock:
                        print(f"Error processing file {os.path.basename(filepath)}: {e}")
        
        if not all_data:
            print("Error: Failed to load any data")
            # Initialize as empty DataFrame even if no data
            self.pm25_data = pd.DataFrame(columns=['datetime', 'pm25', 'date', 'hour'])
            return
        
        # Convert to DataFrame
        self.pm25_data = pd.DataFrame(all_data)
        
        # Sort by time
        self.pm25_data = self.pm25_data.sort_values('datetime').reset_index(drop=True)
        
        print(f"\nData loading completed!")
        print(f"Total {len(self.pm25_data)} records")
        print(f"Time range: {self.pm25_data['datetime'].min()} to {self.pm25_data['datetime'].max()}")
        print(f"PM2.5 concentration range: {self.pm25_data['pm25'].min():.2f} - {self.pm25_data['pm25'].max():.2f} μg/m³")
        print(f"PM2.5 average concentration: {self.pm25_data['pm25'].mean():.2f} μg/m³")
    
    def plot_timeseries(self, save_path='pm25_timeseries.png'):
        """Plot PM2.5 time series"""
        if self.pm25_data.empty:
            print("Error: No data available for plotting")
            return
        
        print("\nStarting to plot time series...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(20, 8))
        
        # Plot line chart
        ax.plot(self.pm25_data['datetime'], self.pm25_data['pm25'], 
                linewidth=0.8, color='#3498db', alpha=0.7, label='PM2.5 Concentration')
        
        # Add moving average line (24 hours)
        if len(self.pm25_data) > 24:
            rolling_mean = self.pm25_data['pm25'].rolling(window=24, center=True).mean()
            ax.plot(self.pm25_data['datetime'], rolling_mean, 
                    linewidth=2, color='#e74c3c', alpha=0.9, label='24-hour Moving Average')
        
        # Add air quality level reference lines
        ax.axhline(y=35, color='#2ecc71', linestyle='--', linewidth=1, alpha=0.6, label='Good/Moderate Boundary (35 μg/m³)')
        ax.axhline(y=75, color='#f39c12', linestyle='--', linewidth=1, alpha=0.6, label='Light Pollution (75 μg/m³)')
        ax.axhline(y=115, color='#e67e22', linestyle='--', linewidth=1, alpha=0.6, label='Moderate Pollution (115 μg/m³)')
        ax.axhline(y=150, color='#c0392b', linestyle='--', linewidth=1, alpha=0.6, label='Heavy Pollution (150 μg/m³)')
        ax.axhline(y=250, color='#8e44ad', linestyle='--', linewidth=1, alpha=0.6, label='Severe Pollution (250 μg/m³)')
        
        # Set labels and title
        ax.set_xlabel('Time', fontsize=14, fontweight='bold')
        ax.set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=14, fontweight='bold')
        ax.set_title('Beijing PM2.5 Concentration Time Series\n(Average of All Stations)', 
                     fontsize=18, fontweight='bold', pad=20)
        
        # Set grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add statistics text box
        stats_text = f"Statistics:\n"
        stats_text += f"Sample size: {len(self.pm25_data)} hours\n"
        stats_text += f"Mean: {self.pm25_data['pm25'].mean():.2f} μg/m³\n"
        stats_text += f"Median: {self.pm25_data['pm25'].median():.2f} μg/m³\n"
        stats_text += f"Max: {self.pm25_data['pm25'].max():.2f} μg/m³\n"
        stats_text += f"Min: {self.pm25_data['pm25'].min():.2f} μg/m³"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Time series plot saved to: {save_path}")
        
        plt.show()
    
    def plot_pm25_histogram(self, save_path='pm25_histogram.png'):
        """Plot PM2.5 concentration distribution histogram"""
        if self.pm25_data.empty:
            print("Error: No data available for plotting")
            return
        
        print("\nStarting to plot PM2.5 concentration distribution histogram...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot histogram
        n, bins, patches = ax.hist(self.pm25_data['pm25'], bins=50, 
                                   color='#3498db', alpha=0.7, edgecolor='white', linewidth=1)
        
        # Add statistical lines
        ax.axvline(self.pm25_data['pm25'].mean(), color='red', 
                   linestyle='--', linewidth=3, label=f"Mean: {self.pm25_data['pm25'].mean():.2f} μg/m³")
        ax.axvline(self.pm25_data['pm25'].median(), color='orange', 
                   linestyle='--', linewidth=3, label=f"Median: {self.pm25_data['pm25'].median():.2f} μg/m³")
        
        # Add air quality level reference lines
        ax.axvline(35, color='#2ecc71', linestyle=':', linewidth=2, alpha=0.8, label='Good/Moderate Boundary (35 μg/m³)')
        ax.axvline(75, color='#f39c12', linestyle=':', linewidth=2, alpha=0.8, label='Light Pollution (75 μg/m³)')
        ax.axvline(115, color='#e67e22', linestyle=':', linewidth=2, alpha=0.8, label='Moderate Pollution (115 μg/m³)')
        ax.axvline(150, color='#c0392b', linestyle=':', linewidth=2, alpha=0.8, label='Heavy Pollution (150 μg/m³)')
        ax.axvline(250, color='#8e44ad', linestyle=':', linewidth=2, alpha=0.8, label='Severe Pollution (250 μg/m³)')
        
        # Set labels and title
        ax.set_xlabel('PM2.5 Concentration (μg/m³)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title('Beijing PM2.5 Concentration Distribution Histogram\n(Average of All Stations)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Set grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Create custom legend with statistical lines and air quality reference lines
        legend_elements = []
        legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=3, 
                                         label=f"Mean: {self.pm25_data['pm25'].mean():.2f} μg/m³"))
        legend_elements.append(plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=3, 
                                         label=f"Median: {self.pm25_data['pm25'].median():.2f} μg/m³"))
        legend_elements.append(plt.Line2D([0], [0], color='#2ecc71', linestyle=':', linewidth=2, 
                                         label='Good/Moderate Boundary (35 μg/m³)'))
        legend_elements.append(plt.Line2D([0], [0], color='#f39c12', linestyle=':', linewidth=2, 
                                         label='Light Pollution (75 μg/m³)'))
        legend_elements.append(plt.Line2D([0], [0], color='#e67e22', linestyle=':', linewidth=2, 
                                         label='Moderate Pollution (115 μg/m³)'))
        legend_elements.append(plt.Line2D([0], [0], color='#c0392b', linestyle=':', linewidth=2, 
                                         label='Heavy Pollution (150 μg/m³)'))
        legend_elements.append(plt.Line2D([0], [0], color='#8e44ad', linestyle=':', linewidth=2, 
                                         label='Severe Pollution (250 μg/m³)'))
        
        # Create main legend (statistical lines and reference lines)
        legend1 = ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        ax.add_artist(legend1)
        
        # Add statistics text box below the legend
        stats_text = f"Statistics:\n"
        stats_text += f"Sample size: {len(self.pm25_data)} hours\n"
        stats_text += f"Mean: {self.pm25_data['pm25'].mean():.2f} μg/m³\n"
        stats_text += f"Median: {self.pm25_data['pm25'].median():.2f} μg/m³\n"
        stats_text += f"Std Dev: {self.pm25_data['pm25'].std():.2f} μg/m³\n"
        stats_text += f"Max: {self.pm25_data['pm25'].max():.2f} μg/m³\n"
        stats_text += f"Min: {self.pm25_data['pm25'].min():.2f} μg/m³"
        
        # Place statistics text box below the legend in top right, right aligned
        ax.text(0.98, 0.45, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"PM2.5 concentration distribution histogram saved to: {save_path}")
        
        plt.show()
    
    def plot_aqi_distribution(self, save_path='aqi_distribution.png'):
        """Plot air quality level distribution"""
        if self.pm25_data.empty:
            print("Error: No data available for plotting")
            return
        
        print("\nStarting to plot air quality level distribution...")
        
        # Define air quality level function
        def get_aqi_level(pm25):
            if pm25 <= 35:
                return 'Good'
            elif pm25 <= 75:
                return 'Moderate'
            elif pm25 <= 115:
                return 'Light Pollution'
            elif pm25 <= 150:
                return 'Moderate Pollution'
            elif pm25 <= 250:
                return 'Heavy Pollution'
            else:
                return 'Severe Pollution'
        
        # Calculate air quality levels
        self.pm25_data['aqi_level'] = self.pm25_data['pm25'].apply(get_aqi_level)
        level_counts = self.pm25_data['aqi_level'].value_counts()
        
        # Ensure ordering by air quality sequence
        level_order = ['Good', 'Moderate', 'Light Pollution', 'Moderate Pollution', 'Heavy Pollution', 'Severe Pollution']
        level_counts = level_counts.reindex([l for l in level_order if l in level_counts.index])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Define colors
        level_colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#c0392b', '#8e44ad']
        
        # Plot pie chart
        wedges, texts, autotexts = ax.pie(level_counts.values, 
                                          labels=level_counts.index, 
                                          autopct='%1.1f%%',
                                          colors=level_colors[:len(level_counts)], 
                                          startangle=90,
                                          radius=1.0,  # Adjust pie radius, default 1.0, can be adjusted between 0.8-1.5
                                          pctdistance=0.7,  # Distance of percentage text from center, default 0.6, can be adjusted between 0.4-0.85
                                          labeldistance=1.2,  # Distance of label text from center, default 1.1, can be adjusted between 1.0-1.3
                                          textprops={'fontsize': 14, 'fontweight': 'bold'},
                                          explode=[0.05] * len(level_counts))  # Separate each segment
        
        # Beautify percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        # Set title
        ax.set_title('Beijing Air Quality Level Distribution\n(Based on PM2.5 Concentration)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        legend_labels = []
        for level, count in level_counts.items():
            percentage = (count / len(self.pm25_data)) * 100
            legend_labels.append(f"{level}: {count} hours ({percentage:.1f}%)")
        
        ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=11, title="Air Quality Level Statistics", title_fontsize=12)
        
        # Ensure pie chart is circular
        ax.axis('equal')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Air quality level distribution plot saved to: {save_path}")
        
        plt.show()
    
    def save_data_summary(self, save_path='pm25_summary.csv'):
        """Save data summary"""
        if self.pm25_data.empty:
            print("Error: No data available to save")
            return
        
        print("\nSaving data summary...")
        
        # Calculate daily statistics
        self.pm25_data['date_obj'] = pd.to_datetime(self.pm25_data['date'], format='%Y%m%d')
        daily_stats = self.pm25_data.groupby('date').agg({
            'pm25': ['mean', 'median', 'min', 'max', 'std', 'count']
        }).reset_index()
        
        daily_stats.columns = ['Date', 'Mean', 'Median', 'Min', 'Max', 'Std Dev', 'Sample Count']
        daily_stats.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"Data summary saved to: {save_path}")
    
    def run_analysis(self):
        """Run complete analysis"""
        print("="*60)
        print("Beijing PM2.5 Concentration Analysis")
        print("="*60)
        
        # Load data
        self.load_all_data()
        
        if self.pm25_data.empty:
            print("Error: Unable to load data, please check data file path")
            return
        
        # Plot time series
        self.plot_timeseries()
        
        # Plot PM2.5 concentration distribution histogram
        self.plot_pm25_histogram()
        
        # Plot air quality level distribution
        self.plot_aqi_distribution()
        
        # Save data summary
        self.save_data_summary()
        
        print("\nAnalysis completed!")


def main():
    # Please modify the data path according to your actual situation
    # Example path (based on search results, pollution data is typically in these paths)
    data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)"
    
    # If the above path does not exist, try other common paths
    if not os.path.exists(data_dir):
        # Try to find other possible paths
        possible_paths = [
            r"C:\Users\IU\Desktop\Beijing-AQI-Date",
            r"C:\Users\IU\Desktop\Beijing-AQI-Benchmark",
            r"C:\Users\IU\Desktop\Datebase Origin",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_dir = path
                print(f"Using data path: {data_dir}")
                break
        else:
            print("Error: Data folder not found, please manually set the data_dir variable")
            print("Please modify data_dir in the main() function to your data folder path")
            return
    
    print(f"Data folder path: {data_dir}")
    print("If the path is incorrect, please modify the data_dir variable in the main() function\n")
    
    # Create analyzer and run
    # max_workers: Number of threads for parallel file processing (default: 4)
    # Increase for faster processing with more CPU cores, decrease if memory is limited
    analyzer = PM25Analyzer(data_dir, max_workers=16)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()