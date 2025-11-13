"""
Beijing PM2.5 Concentration Prediction - 2D CNN Model
Using 2D Convolutional Neural Network for time series prediction

Features:
- 2D CNN architecture, treating time window and features as 2D images
- Uses 30 days of historical data to predict current day PM2.5 concentration
- Supports hyperparameter optimization (Bayesian optimization)
- Gradient×Input method for feature importance analysis
- Complete model evaluation and visualization

Data Sources:
- Pollution data: Benchmark dataset (PM2.5, PM10, SO2, NO2, CO, O3)
- Meteorological data: ERA5 reanalysis data
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import pickle
from pathlib import Path
import glob
import multiprocessing

warnings.filterwarnings('ignore')

# PyTorch related
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Get CPU core count
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

# Try to import tqdm progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm not installed, progress display will use simplified version.")
    print("      Use 'pip install tqdm' to get better progress bar display.")

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Bayesian optimization (optional)
try:
    from bayes_opt import BayesianOptimization
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    print("Note: bayesian-optimization not installed, will use grid search.")
    print("      Use 'pip install bayesian-optimization' to enable Bayesian optimization.")
    BAYESIAN_OPT_AVAILABLE = False

# Set English fonts
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# Set random seed
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - 2D CNN Model")
print("=" * 80)

# ============================== Part 1: Configuration and Path Setup ==============================
print("\nConfiguring parameters...")

# Data paths
pollution_all_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)'
era5_path = r'C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV'

# Output path
output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)

# Model save path
model_dir = Path('./models')
model_dir.mkdir(exist_ok=True)

# Date range
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

# Beijing geographic range
beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

# Pollutant list
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# ERA5 variables
era5_vars = [
    'd2m', 't2m', 'u10', 'v10', 'u100', 'v100',  # Temperature, wind speed
    'blh', 'sp', 'tcwv',  # Boundary layer height, pressure, water vapor
    'tp', 'avg_tprate',  # Precipitation
    'tisr', 'str',  # Radiation
    'cvh', 'cvl',  # Cloud cover
    'mn2t', 'sd', 'lsm'  # Others
]

# CNN specific parameters
WINDOW_SIZE = 30  # Use past 30 days data
BATCH_SIZE = 32
DEVICE = torch.device('cpu')  # CPU version

print(f"Data time range: {start_date.date()} to {end_date.date()}")
print(f"Target variable: PM2.5 concentration")
print(f"Time window size: {WINDOW_SIZE} days")
print(f"Batch size: {BATCH_SIZE}")
print(f"Device: {DEVICE}")
print(f"Output directory: {output_dir}")
print(f"Model save directory: {model_dir}")
print(f"CPU cores: {CPU_COUNT}, parallel workers: {MAX_WORKERS}")

# ============================== Part 2: Data Loading Functions ==============================
def daterange(start, end):
    """Generate date sequence"""
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def find_file(base_path, date_str, prefix):
    """Find file for specified date"""
    filename = f"{prefix}_{date_str}.csv"
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def read_pollution_day(date):
    """Read single day pollution data"""
    date_str = date.strftime('%Y%m%d')
    all_file = find_file(pollution_all_path, date_str, 'beijing_all')
    extra_file = find_file(pollution_extra_path, date_str, 'beijing_extra')
    
    if not all_file or not extra_file:
        return None
    
    try:
        df_all = pd.read_csv(all_file, encoding='utf-8', on_bad_lines='skip')
        df_extra = pd.read_csv(extra_file, encoding='utf-8', on_bad_lines='skip')
        
        # Filter out 24-hour average and AQI
        df_all = df_all[~df_all['type'].str.contains('_24h|AQI', na=False)]
        df_extra = df_extra[~df_extra['type'].str.contains('_24h', na=False)]
        
        # Merge
        df_poll = pd.concat([df_all, df_extra], ignore_index=True)
        
        # Convert to long format
        df_poll = df_poll.melt(id_vars=['date', 'hour', 'type'], 
                                var_name='station', value_name='value')
        df_poll['value'] = pd.to_numeric(df_poll['value'], errors='coerce')
        
        # Remove negative values and outliers
        df_poll = df_poll[df_poll['value'] >= 0]
        
        # Aggregate by date and type (average all stations)
        df_daily = df_poll.groupby(['date', 'type'])['value'].mean().reset_index()
        
        # Convert to wide format
        df_daily = df_daily.pivot(index='date', columns='type', values='value')
        
        # Convert index to datetime format
        df_daily.index = pd.to_datetime(df_daily.index, format='%Y%m%d', errors='coerce')
        
        # Keep only required pollutants
        df_daily = df_daily[[col for col in pollutants if col in df_daily.columns]]
        
        return df_daily
    except Exception as e:
        return None

def read_all_pollution():
    """Read all pollution data in parallel"""
    print("\nLoading pollution data...")
    print(f"Using {MAX_WORKERS} parallel workers")
    dates = list(daterange(start_date, end_date))
    pollution_dfs = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, date): date for date in dates}
        
        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Loading pollution data", unit="day"):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
        else:
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
                if i % 500 == 0 or i == len(futures):
                    print(f"  Processed {i}/{len(futures)} days ({i/len(futures)*100:.1f}%)")
    
    if pollution_dfs:
        print(f"  Successfully read {len(pollution_dfs)}/{len(dates)} days of data")
        print("  Merging data...")
        df_poll_all = pd.concat(pollution_dfs)
        df_poll_all.ffill(inplace=True)
        df_poll_all.fillna(df_poll_all.mean(), inplace=True)
        print(f"Pollution data loading complete, shape: {df_poll_all.shape}")
        return df_poll_all
    return pd.DataFrame()

def read_era5_month(year, month):
    """Read single month ERA5 data"""
    month_str = f"{year}{month:02d}"
    all_files = glob.glob(os.path.join(era5_path, "**", f"*{month_str}*.csv"), recursive=True)
    
    if not all_files:
        return None
    
    monthly_data = None
    loaded_vars = []
    
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', 
                            low_memory=False, comment='#')
            
            if df.empty or 'time' not in df.columns:
                continue
            
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.dropna(subset=['time'])
            
            if len(df) == 0:
                continue
            
            if 'latitude' in df.columns and 'longitude' in df.columns:
                df = df[(df['latitude'] >= beijing_lats.min()) & 
                       (df['latitude'] <= beijing_lats.max()) &
                       (df['longitude'] >= beijing_lons.min()) & 
                       (df['longitude'] <= beijing_lons.max())]
                
                if len(df) == 0:
                    continue
            
            if 'expver' in df.columns:
                if '0001' in df['expver'].values:
                    df = df[df['expver'] == '0001']
                else:
                    first_expver = df['expver'].iloc[0]
                    df = df[df['expver'] == first_expver]
            
            df['date'] = df['time'].dt.date
            avail_vars = [v for v in era5_vars if v in df.columns]
            
            if not avail_vars:
                continue
            
            for col in avail_vars:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df_daily = df.groupby('date')[avail_vars].mean().reset_index()
            df_daily.set_index('date', inplace=True)
            df_daily.index = pd.to_datetime(df_daily.index)
            
            if monthly_data is None:
                monthly_data = df_daily
            else:
                monthly_data = monthly_data.join(df_daily, how='outer')
            
            loaded_vars.extend(avail_vars)
            
        except Exception as e:
            continue
    
    if monthly_data is not None and not monthly_data.empty:
        print(f"  Successfully read: {year}-{month:02d}, days: {len(monthly_data)}, variables: {len(loaded_vars)}")
        return monthly_data
    else:
        return None

def read_all_era5():
    """Read all ERA5 data in parallel"""
    print("\nLoading meteorological data...")
    print(f"Using {MAX_WORKERS} parallel workers")
    print(f"Meteorological data directory: {era5_path}")
    
    if os.path.exists(era5_path):
        all_csv = glob.glob(os.path.join(era5_path, "**", "*.csv"), recursive=True)
        print(f"Found {len(all_csv)} CSV files")
        if all_csv:
            print(f"Sample files: {[os.path.basename(f) for f in all_csv[:5]]}")
    
    era5_dfs = []
    years = range(2015, 2025)
    months = range(1, 13)
    
    month_tasks = [(year, month) for year in years for month in months 
                   if not (year == 2024 and month > 12)]
    total_months = len(month_tasks)
    print(f"Attempting to load {total_months} months of data...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_era5_month, year, month): (year, month) 
                  for year, month in month_tasks}
        
        successful_reads = 0
        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Loading meteorological data", unit="month"):
                result = future.result()
                if result is not None and not result.empty:
                    era5_dfs.append(result)
                    successful_reads += 1
        else:
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None and not result.empty:
                    era5_dfs.append(result)
                    successful_reads += 1
                if i % 20 == 0 or i == len(futures):
                    print(f"  Progress: {i}/{len(futures)} months (success: {successful_reads}, {i/len(futures)*100:.1f}%)")
        
        print(f"  Total successfully read: {successful_reads}/{len(futures)} months")
    
    if era5_dfs:
        print("\nMerging meteorological data...")
        df_era5_all = pd.concat(era5_dfs, axis=0)
        print("  Removing duplicates...")
        df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep='first')]
        print("  Sorting...")
        df_era5_all.sort_index(inplace=True)
        
        print(f"Merged shape: {df_era5_all.shape}")
        print(f"Time range: {df_era5_all.index.min()} to {df_era5_all.index.max()}")
        
        print("  Handling missing values...")
        initial_na = df_era5_all.isna().sum().sum()
        df_era5_all.ffill(inplace=True)
        df_era5_all.bfill(inplace=True)
        df_era5_all.fillna(df_era5_all.mean(), inplace=True)
        final_na = df_era5_all.isna().sum().sum()
        
        print(f"Missing value handling: {initial_na} -> {final_na}")
        print(f"Meteorological data loading complete, shape: {df_era5_all.shape}")
        
        return df_era5_all
    else:
        print("\n❌ Error: Failed to load any meteorological data files!")
        return pd.DataFrame()

# ============================== Part 3: Feature Engineering ==============================
def create_features(df):
    """Create additional features"""
    df_copy = df.copy()
    
    # 1. Wind speed features
    if 'u10' in df_copy and 'v10' in df_copy:
        df_copy['wind_speed_10m'] = np.sqrt(df_copy['u10']**2 + df_copy['v10']**2)
        df_copy['wind_dir_10m'] = np.arctan2(df_copy['v10'], df_copy['u10']) * 180 / np.pi
        df_copy['wind_dir_10m'] = (df_copy['wind_dir_10m'] + 360) % 360
    
    if 'u100' in df_copy and 'v100' in df_copy:
        df_copy['wind_speed_100m'] = np.sqrt(df_copy['u100']**2 + df_copy['v100']**2)
        df_copy['wind_dir_100m'] = np.arctan2(df_copy['v100'], df_copy['u100']) * 180 / np.pi
        df_copy['wind_dir_100m'] = (df_copy['wind_dir_100m'] + 360) % 360
    
    # 2. Time features
    df_copy['year'] = df_copy.index.year
    df_copy['month'] = df_copy.index.month
    df_copy['day'] = df_copy.index.day
    df_copy['day_of_year'] = df_copy.index.dayofyear
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['week_of_year'] = df_copy.index.isocalendar().week
    
    # Season feature
    df_copy['season'] = df_copy['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )
    
    # Heating season indicator
    df_copy['is_heating_season'] = ((df_copy['month'] >= 11) | (df_copy['month'] <= 3)).astype(int)
    
    # 3. Temperature related features
    if 't2m' in df_copy and 'd2m' in df_copy:
        df_copy['temp_dewpoint_diff'] = df_copy['t2m'] - df_copy['d2m']
    
    # 4. Lag features
    if 'PM2.5' in df_copy:
        df_copy['PM2.5_lag1'] = df_copy['PM2.5'].shift(1)
        df_copy['PM2.5_lag3'] = df_copy['PM2.5'].shift(3)
        df_copy['PM2.5_lag7'] = df_copy['PM2.5'].shift(7)
        
        df_copy['PM2.5_ma3'] = df_copy['PM2.5'].rolling(window=3, min_periods=1).mean()
        df_copy['PM2.5_ma7'] = df_copy['PM2.5'].rolling(window=7, min_periods=1).mean()
        df_copy['PM2.5_ma30'] = df_copy['PM2.5'].rolling(window=30, min_periods=1).mean()
    
    # 5. Relative humidity estimation
    if 't2m' in df_copy and 'd2m' in df_copy:
        df_copy['relative_humidity'] = 100 * np.exp((17.625 * (df_copy['d2m'] - 273.15)) / 
                                                      (243.04 + (df_copy['d2m'] - 273.15))) / \
                                        np.exp((17.625 * (df_copy['t2m'] - 273.15)) / 
                                               (243.04 + (df_copy['t2m'] - 273.15)))
        df_copy['relative_humidity'] = df_copy['relative_humidity'].clip(0, 100)
    
    # 6. Wind direction category
    if 'wind_dir_10m' in df_copy:
        df_copy['wind_dir_category'] = pd.cut(df_copy['wind_dir_10m'], 
                                                bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                                                labels=[0, 1, 2, 3, 4, 5, 6, 7],
                                                include_lowest=True).astype(int)
    
    return df_copy

# ============================== Part 4: Data Loading and Preprocessing ==============================
print("\n" + "=" * 80)
print("Step 1: Data Loading and Preprocessing")
print("=" * 80)

df_pollution = read_all_pollution()
df_era5 = read_all_era5()

# Check data loading
print("\nData loading check:")
print(f"  Pollution data shape: {df_pollution.shape}")
print(f"  Meteorological data shape: {df_era5.shape}")

if df_pollution.empty:
    print("\n⚠️ Warning: Pollution data is empty! Please check data path and files.")
    import sys
    sys.exit(1)

if df_era5.empty:
    print("\n⚠️ Warning: Meteorological data is empty! Please check data path and files.")
    import sys
    sys.exit(1)

# Ensure index is datetime type
df_pollution.index = pd.to_datetime(df_pollution.index)
df_era5.index = pd.to_datetime(df_era5.index)

print(f"  Pollution data time range: {df_pollution.index.min()} to {df_pollution.index.max()}")
print(f"  Meteorological data time range: {df_era5.index.min()} to {df_era5.index.max()}")

# Merge data
print("\nMerging data...")
df_combined = df_pollution.join(df_era5, how='inner')

if df_combined.empty:
    print("\n❌ Error: Data is empty after merging!")
    import sys
    sys.exit(1)

# Create features
print("\nCreating features...")
df_combined = create_features(df_combined)

# Clean data
print("\nCleaning data...")
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
initial_rows = len(df_combined)
df_combined.dropna(inplace=True)
final_rows = len(df_combined)
print(f"Removed {initial_rows - final_rows} rows containing missing values")

print(f"\nMerged data shape: {df_combined.shape}")
print(f"Time range: {df_combined.index.min().date()} to {df_combined.index.max().date()}")
print(f"Number of samples: {len(df_combined)}")
print(f"Number of features: {df_combined.shape[1]}")

print(f"\nFeature list (first 20):")
for i, col in enumerate(df_combined.columns[:20], 1):
    print(f"  {i}. {col}")
if len(df_combined.columns) > 20:
    print(f"  ... and {len(df_combined.columns) - 20} more features")

# ============================== Part 5: CNN Data Preparation ==============================
print("\n" + "=" * 80)
print("Step 2: CNN Data Preparation (Sliding Window)")
print("=" * 80)

# Define target variable
target = 'PM2.5'

# Excluded columns
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']

# Select numeric features
numeric_features = [col for col in df_combined.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_cols]

print(f"\nNumber of selected features: {len(numeric_features)}")
print(f"Target variable: {target}")

# Prepare data
X_raw = df_combined[numeric_features].values
y_raw = df_combined[target].values

print(f"\nRaw data shape:")
print(f"  X: {X_raw.shape}")
print(f"  y: {y_raw.shape}")

# Standardize features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

print(f"\nStandardized data shape:")
print(f"  X: {X_scaled.shape}")
print(f"  y: {y_scaled.shape}")

# Create sliding window dataset
def create_sliding_windows(X, y, window_size):
    """
    Create sliding window dataset
    
    Args:
        X: Feature data [samples, features]
        y: Target data [samples]
        window_size: Window size
    
    Returns:
        X_windows: [num_windows, window_size, features]
        y_windows: [num_windows]
    """
    num_samples = len(X)
    num_windows = num_samples - window_size + 1
    num_features = X.shape[1]
    
    X_windows = np.zeros((num_windows, window_size, num_features))
    y_windows = np.zeros(num_windows)
    
    for i in range(num_windows):
        X_windows[i] = X[i:i+window_size]
        y_windows[i] = y[i+window_size-1]  # Predict PM2.5 for last day of window
    
    return X_windows, y_windows

print(f"\nCreating {WINDOW_SIZE} day sliding windows...")
X_windows, y_windows = create_sliding_windows(X_scaled, y_scaled, WINDOW_SIZE)

print(f"Sliding window data shape:")
print(f"  X_windows: {X_windows.shape}  # [num_samples, time_steps, num_features]")
print(f"  y_windows: {y_windows.shape}")

# Save feature names and date index (for subsequent analysis)
feature_names = numeric_features
date_index = df_combined.index[WINDOW_SIZE-1:]

print(f"\nPM2.5 Statistics:")
print(f"  Mean: {y_raw.mean():.2f} μg/m³")
print(f"  Std Dev: {y_raw.std():.2f} μg/m³")
print(f"  Min: {y_raw.min():.2f} μg/m³")
print(f"  Max: {y_raw.max():.2f} μg/m³")
print(f"  Median: {np.median(y_raw):.2f} μg/m³")

# ============================== Part 6: PyTorch Dataset and DataLoader ==============================
print("\n" + "=" * 80)
print("Step 3: Creating PyTorch Dataset")
print("=" * 80)

class TimeSeriesDataset(Dataset):
    """Time series dataset"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Split by time order: 70% training, 15% validation, 15% test
n_samples = len(X_windows)
train_size = int(n_samples * 0.70)
val_size = int(n_samples * 0.15)

X_train = X_windows[:train_size]
X_val = X_windows[train_size:train_size + val_size]
X_test = X_windows[train_size + val_size:]

y_train = y_windows[:train_size]
y_val = y_windows[train_size:train_size + val_size]
y_test = y_windows[train_size + val_size:]

print(f"\nTraining set: {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
print(f"  Time range: {date_index[0].date()} to {date_index[train_size-1].date()}")

print(f"\nValidation set: {len(X_val)} samples ({len(X_val)/n_samples*100:.1f}%)")
print(f"  Time range: {date_index[train_size].date()} to {date_index[train_size+val_size-1].date()}")

print(f"\nTest set: {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")
print(f"  Time range: {date_index[train_size+val_size].date()} to {date_index[-1].date()}")

# Create datasets and data loaders
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\nData loaders created:")
print(f"  Training batches: {len(train_loader)}")
print(f"  Validation batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ============================== Part 7: 2D CNN Model Definition ==============================
print("\n" + "=" * 80)
print("Step 4: Defining 2D CNN Model")
print("=" * 80)

class PM25CNN2D(nn.Module):
    """
    2D CNN model for PM2.5 prediction
    Input: [batch, 1, window_size, num_features] 
    Output: [batch] (single PM2.5 value)
    """
    def __init__(self, window_size, num_features, num_conv_layers=3, 
                 base_filters=32, kernel_size=3, dropout_rate=0.3):
        super(PM25CNN2D, self).__init__()
        
        self.window_size = window_size
        self.num_features = num_features
        
        # Convolutional layers
        conv_layers = []
        in_channels = 1
        
        for i in range(num_conv_layers):
            out_channels = base_filters * (2 ** i)
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate feature map size after convolution
        self.feature_size = self._get_conv_output_size()
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(64, 1)
        )
        
        # Weight initialization
        self._initialize_weights()
    
    def _get_conv_output_size(self):
        """Calculate feature size after convolutional layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.window_size, self.num_features)
            dummy_output = self.conv_layers(dummy_input)
            return int(np.prod(dummy_output.shape[1:]))
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [batch, window_size, num_features]
        # Add channel dimension
        x = x.unsqueeze(1)  # [batch, 1, window_size, num_features]
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x.squeeze()

num_features = X_train.shape[2]
print(f"\nModel input dimensions:")
print(f"  Window size: {WINDOW_SIZE}")
print(f"  Number of features: {num_features}")

# ============================== Part 8: Training and Evaluation Functions ==============================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(X_batch)
    
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            total_loss += loss.item() * len(X_batch)
            predictions.extend(y_pred.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    return avg_loss, predictions, actuals

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, patience=20, verbose=True):
    """Train model (with early stopping)"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"Training complete! Best model at epoch {best_epoch}, validation loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses, best_epoch

def evaluate_model(y_true, y_pred, dataset_name):
    """Evaluate model performance"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Avoid division by zero error
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    return {
        'Dataset': dataset_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

# ============================== Part 9: Basic Model Training ==============================
print("\n" + "=" * 80)
print("Step 5: Training Basic CNN Model")
print("=" * 80)

# Basic model parameters
basic_params = {
    'num_conv_layers': 3,
    'base_filters': 32,
    'kernel_size': 3,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'num_epochs': 200,
    'patience': 20
}

print("\nBasic model parameters:")
for key, value in basic_params.items():
    print(f"  {key}: {value}")

# Create basic model
model_basic = PM25CNN2D(
    window_size=WINDOW_SIZE,
    num_features=num_features,
    num_conv_layers=basic_params['num_conv_layers'],
    base_filters=basic_params['base_filters'],
    kernel_size=basic_params['kernel_size'],
    dropout_rate=basic_params['dropout_rate']
).to(DEVICE)

# Count model parameters
total_params = sum(p.numel() for p in model_basic.parameters())
trainable_params = sum(p.numel() for p in model_basic.parameters() if p.requires_grad)
print(f"\nModel parameter statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer_basic = optim.Adam(model_basic.parameters(), lr=basic_params['learning_rate'])

# Train basic model
train_losses_basic, val_losses_basic, best_epoch_basic = train_model(
    model_basic, train_loader, val_loader, criterion, optimizer_basic,
    num_epochs=basic_params['num_epochs'],
    device=DEVICE,
    patience=basic_params['patience'],
    verbose=True
)

print(f"\n✓ Basic model training complete")
print(f"  Best epoch: {best_epoch_basic}")
print(f"  Final training loss: {train_losses_basic[best_epoch_basic-1]:.4f}")
print(f"  Final validation loss: {val_losses_basic[best_epoch_basic-1]:.4f}")

# Evaluate basic model
print("\nEvaluating basic model...")

_, y_train_pred_basic_scaled, y_train_actual_scaled = validate(model_basic, train_loader, criterion, DEVICE)
_, y_val_pred_basic_scaled, y_val_actual_scaled = validate(model_basic, val_loader, criterion, DEVICE)
_, y_test_pred_basic_scaled, y_test_actual_scaled = validate(model_basic, test_loader, criterion, DEVICE)

# Denormalize
y_train_pred_basic = scaler_y.inverse_transform(y_train_pred_basic_scaled.reshape(-1, 1)).flatten()
y_train_actual_basic = scaler_y.inverse_transform(y_train_actual_scaled.reshape(-1, 1)).flatten()

y_val_pred_basic = scaler_y.inverse_transform(y_val_pred_basic_scaled.reshape(-1, 1)).flatten()
y_val_actual_basic = scaler_y.inverse_transform(y_val_actual_scaled.reshape(-1, 1)).flatten()

y_test_pred_basic = scaler_y.inverse_transform(y_test_pred_basic_scaled.reshape(-1, 1)).flatten()
y_test_actual_basic = scaler_y.inverse_transform(y_test_actual_scaled.reshape(-1, 1)).flatten()

# Calculate performance metrics
results_basic = []
results_basic.append(evaluate_model(y_train_actual_basic, y_train_pred_basic, 'Train'))
results_basic.append(evaluate_model(y_val_actual_basic, y_val_pred_basic, 'Validation'))
results_basic.append(evaluate_model(y_test_actual_basic, y_test_pred_basic, 'Test'))

results_basic_df = pd.DataFrame(results_basic)
print("\nBasic model performance:")
print(results_basic_df.to_string(index=False))

# ============================== Part 10: Hyperparameter Optimization ==============================
print("\n" + "=" * 80)
print("Step 6: Hyperparameter Optimization")
print("=" * 80)

if BAYESIAN_OPT_AVAILABLE:
    print("\nUsing Bayesian optimization for hyperparameter search...")
    
    def cnn_evaluate(num_conv_layers, base_filters, kernel_size, 
                     learning_rate, dropout_rate):
        """Objective function for Bayesian optimization"""
        # Parameter conversion
        num_conv_layers = int(num_conv_layers)
        base_filters = int(base_filters)
        kernel_size = int(kernel_size)
        
        # Create model
        model_temp = PM25CNN2D(
            window_size=WINDOW_SIZE,
            num_features=num_features,
            num_conv_layers=num_conv_layers,
            base_filters=base_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate
        ).to(DEVICE)
        
        # Train
        optimizer_temp = optim.Adam(model_temp.parameters(), lr=learning_rate)
        _, _, _ = train_model(
            model_temp, train_loader, val_loader, criterion, optimizer_temp,
            num_epochs=100, device=DEVICE, patience=15, verbose=False
        )
        
        # Evaluate
        val_loss, _, _ = validate(model_temp, val_loader, criterion, DEVICE)
        
        # Return negative loss (Bayesian optimization maximizes)
        return -val_loss
    
    # Parameter search space
    pbounds = {
        'num_conv_layers': (2, 4),
        'base_filters': (16, 64),
        'kernel_size': (3, 5),
        'learning_rate': (0.0001, 0.01),
        'dropout_rate': (0.2, 0.5)
    }
    
    optimizer_bo = BayesianOptimization(
        f=cnn_evaluate,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    optimizer_bo.maximize(init_points=5, n_iter=10)
    
    # Get best parameters
    best_params = optimizer_bo.max['params']
    best_params['num_conv_layers'] = int(best_params['num_conv_layers'])
    best_params['base_filters'] = int(best_params['base_filters'])
    best_params['kernel_size'] = int(best_params['kernel_size'])
    
    print(f"\nBest parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  Best validation loss: {-optimizer_bo.max['target']:.4f}")
    
else:
    # Grid search (simplified version)
    print("\nUsing grid search for hyperparameter optimization...")
    
    param_grid = {
        'num_conv_layers': [2, 3],
        'base_filters': [32, 64],
        'kernel_size': [3, 5],
        'learning_rate': [0.001, 0.005],
        'dropout_rate': [0.3, 0.4]
    }
    
    from itertools import product
    total_combinations = int(np.prod([len(v) for v in param_grid.values()]))
    print(f"Total {total_combinations} parameter combinations")
    
    best_val_loss_grid = float('inf')
    best_params = {}
    
    for i, combo in enumerate(product(*param_grid.values()), 1):
        print(f"\nTesting combination {i}/{total_combinations}...")
        params_test = dict(zip(param_grid.keys(), combo))
        
        model_temp = PM25CNN2D(
            window_size=WINDOW_SIZE,
            num_features=num_features,
            num_conv_layers=params_test['num_conv_layers'],
            base_filters=params_test['base_filters'],
            kernel_size=params_test['kernel_size'],
            dropout_rate=params_test['dropout_rate']
        ).to(DEVICE)
        
        optimizer_temp = optim.Adam(model_temp.parameters(), lr=params_test['learning_rate'])
        _, _, _ = train_model(
            model_temp, train_loader, val_loader, criterion, optimizer_temp,
            num_epochs=100, device=DEVICE, patience=15, verbose=False
        )
        
        val_loss, _, _ = validate(model_temp, val_loader, criterion, DEVICE)
        print(f"  Validation loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss_grid:
            best_val_loss_grid = val_loss
            best_params = params_test.copy()
    
    print(f"\nBest parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  Best validation loss: {best_val_loss_grid:.4f}")

# ============================== Part 11: Training Optimized Model ==============================
print("\n" + "=" * 80)
print("Step 7: Training Optimized Model with Best Parameters")
print("=" * 80)

print("\nOptimized model parameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# Create optimized model
model_optimized = PM25CNN2D(
    window_size=WINDOW_SIZE,
    num_features=num_features,
    num_conv_layers=best_params['num_conv_layers'],
    base_filters=best_params['base_filters'],
    kernel_size=best_params['kernel_size'],
    dropout_rate=best_params['dropout_rate']
).to(DEVICE)

optimizer_opt = optim.Adam(model_optimized.parameters(), lr=best_params['learning_rate'])

# Train optimized model
train_losses_opt, val_losses_opt, best_epoch_opt = train_model(
    model_optimized, train_loader, val_loader, criterion, optimizer_opt,
    num_epochs=300, device=DEVICE, patience=30, verbose=True
)

print(f"\n✓ Optimized model training complete")
print(f"  Best epoch: {best_epoch_opt}")
print(f"  Final training loss: {train_losses_opt[best_epoch_opt-1]:.4f}")
print(f"  Final validation loss: {val_losses_opt[best_epoch_opt-1]:.4f}")

# Evaluate optimized model
print("\nEvaluating optimized model...")

_, y_train_pred_opt_scaled, _ = validate(model_optimized, train_loader, criterion, DEVICE)
_, y_val_pred_opt_scaled, _ = validate(model_optimized, val_loader, criterion, DEVICE)
_, y_test_pred_opt_scaled, _ = validate(model_optimized, test_loader, criterion, DEVICE)

# Denormalize
y_train_pred_opt = scaler_y.inverse_transform(y_train_pred_opt_scaled.reshape(-1, 1)).flatten()
y_val_pred_opt = scaler_y.inverse_transform(y_val_pred_opt_scaled.reshape(-1, 1)).flatten()
y_test_pred_opt = scaler_y.inverse_transform(y_test_pred_opt_scaled.reshape(-1, 1)).flatten()

# Calculate performance metrics
results_opt = []
results_opt.append(evaluate_model(y_train_actual_basic, y_train_pred_opt, 'Train'))
results_opt.append(evaluate_model(y_val_actual_basic, y_val_pred_opt, 'Validation'))
results_opt.append(evaluate_model(y_test_actual_basic, y_test_pred_opt, 'Test'))

results_opt_df = pd.DataFrame(results_opt)
print("\nOptimized model performance:")
print(results_opt_df.to_string(index=False))

# ============================== Part 12: Model Comparison ==============================
print("\n" + "=" * 80)
print("Step 8: Model Performance Comparison")
print("=" * 80)

# Merge results
results_basic_df['Model'] = 'CNN_Basic'
results_opt_df['Model'] = 'CNN_Optimized'
all_results = pd.concat([results_basic_df, results_opt_df])

# Rearrange column order
all_results = all_results[['Model', 'Dataset', 'R²', 'RMSE', 'MAE', 'MAPE']]

print("\nAll models performance comparison:")
print(all_results.to_string(index=False))

# Test set performance comparison
test_results = all_results[all_results['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\nTest set performance ranking:")
print(test_results.to_string(index=False))

# Performance improvement
basic_test_r2 = results_basic_df[results_basic_df['Dataset'] == 'Test']['R²'].values[0]
opt_test_r2 = results_opt_df[results_opt_df['Dataset'] == 'Test']['R²'].values[0]
basic_test_rmse = results_basic_df[results_basic_df['Dataset'] == 'Test']['RMSE'].values[0]
opt_test_rmse = results_opt_df[results_opt_df['Dataset'] == 'Test']['RMSE'].values[0]

if basic_test_r2 != 0:
    r2_improvement = (opt_test_r2 - basic_test_r2) / abs(basic_test_r2) * 100
else:
    r2_improvement = 0

if basic_test_rmse != 0:
    rmse_improvement = (basic_test_rmse - opt_test_rmse) / basic_test_rmse * 100
else:
    rmse_improvement = 0

print(f"\nOptimization effect:")
print(f"  R² improvement: {r2_improvement:.2f}%")
print(f"  RMSE reduction: {rmse_improvement:.2f}%")

# ============================== Part 13: Feature Importance Analysis (Gradient×Input) ==============================
print("\n" + "=" * 80)
print("Step 9: Feature Importance Analysis (Gradient×Input Method)")
print("=" * 80)

def compute_gradient_importance(model, X_samples, device, num_samples=500):
    """
    Compute feature importance using Gradient×Input method
    
    Args:
        model: Trained model
        X_samples: Sample data [num_samples, window_size, num_features]
        device: Computing device
        num_samples: Number of samples to use
    
    Returns:
        feature_importance: [num_features] importance score for each feature
    """
    model.eval()
    
    # Randomly select samples
    if len(X_samples) > num_samples:
        indices = np.random.choice(len(X_samples), num_samples, replace=False)
        X_samples = X_samples[indices]
    
    X_tensor = torch.FloatTensor(X_samples).to(device)
    X_tensor.requires_grad = True
    
    # Forward pass
    outputs = model(X_tensor)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=outputs.sum(),
        inputs=X_tensor,
        create_graph=False
    )[0]
    
    # Calculate importance: |gradient × input|
    importance = (gradients * X_tensor).abs()
    
    # Average over time and sample dimensions to get importance for each feature
    importance = importance.mean(dim=[0, 1])  # [num_features]
    
    return importance.detach().cpu().numpy()

print("\nCalculating feature importance...")
feature_importance_scores = compute_gradient_importance(
    model_optimized, X_train, DEVICE, num_samples=500
)

# Normalize importance scores
feature_importance_scores_norm = (feature_importance_scores / feature_importance_scores.sum()) * 100

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance_scores,
    'Importance_Norm': feature_importance_scores_norm
})

# Sort
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print(f"\nTop 20 important features:")
print(feature_importance.head(20)[['Feature', 'Importance_Norm']].to_string(index=False))

# ============================== Part 14: Visualization ==============================
print("\n" + "=" * 80)
print("Step 10: Generating Visualization Charts")
print("=" * 80)

# 14.1 Training process curves
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Basic model
axes[0].plot(train_losses_basic, label='Training Loss', linewidth=2)
axes[0].plot(val_losses_basic, label='Validation Loss', linewidth=2)
axes[0].axvline(x=best_epoch_basic-1, color='r', linestyle='--', 
                label=f'Best epoch({best_epoch_basic})', linewidth=1.5)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('CNN Basic Model - Training Process', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Optimized model
axes[1].plot(train_losses_opt, label='Training Loss', linewidth=2)
axes[1].plot(val_losses_opt, label='Validation Loss', linewidth=2)
axes[1].axvline(x=best_epoch_opt-1, color='r', linestyle='--',
                label=f'Best epoch({best_epoch_opt})', linewidth=1.5)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss (MSE)', fontsize=12)
axes[1].set_title('CNN Optimized Model - Training Process', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
print("Saved: training_curves.png")
plt.close()

# 14.2 Prediction vs actual scatter plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

models_data = [
    ('Basic', y_train_pred_basic, y_train_actual_basic, 'Train'),
    ('Basic', y_val_pred_basic, y_val_actual_basic, 'Val'),
    ('Basic', y_test_pred_basic, y_test_actual_basic, 'Test'),
    ('Optimized', y_train_pred_opt, y_train_actual_basic, 'Train'),
    ('Optimized', y_val_pred_opt, y_val_actual_basic, 'Val'),
    ('Optimized', y_test_pred_opt, y_test_actual_basic, 'Test')
]

for idx, (model_name, y_pred, y_true, dataset) in enumerate(models_data):
    row = idx // 3
    col = idx % 3
    
    ax = axes[row, col]
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    
    # Ideal prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    ax.set_xlabel('Actual PM2.5 Concentration (μg/m³)', fontsize=11)
    ax.set_ylabel('Predicted PM2.5 Concentration (μg/m³)', fontsize=11)
    ax.set_title(f'CNN_{model_name} - {dataset}\nR²={r2:.4f}, RMSE={rmse:.2f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
print("Saved: prediction_scatter.png")
plt.close()

# 14.3 Time series prediction comparison
fig, axes = plt.subplots(2, 1, figsize=(18, 10))

# Test set index
test_date_index = date_index[train_size+val_size:]

# Plot last 300 points
plot_range = min(300, len(y_test_actual_basic))
plot_idx = range(len(y_test_actual_basic) - plot_range, len(y_test_actual_basic))
time_idx = test_date_index[plot_idx]

axes[0].plot(time_idx, y_test_actual_basic[plot_idx], 'k-', label='Actual', 
             linewidth=2, alpha=0.8)
axes[0].plot(time_idx, y_test_pred_basic[plot_idx], 'b--', label='Basic Model Prediction', 
             linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
axes[0].set_title('CNN Basic Model - Time Series Prediction Comparison (Last 300 Days of Test Set)', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

axes[1].plot(time_idx, y_test_actual_basic[plot_idx], 'k-', label='Actual', 
             linewidth=2, alpha=0.8)
axes[1].plot(time_idx, y_test_pred_opt[plot_idx], 'g--', label='Optimized Model Prediction', 
             linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
axes[1].set_title('CNN Optimized Model - Time Series Prediction Comparison (Last 300 Days of Test Set)', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(output_dir / 'timeseries_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: timeseries_comparison.png")
plt.close()

# 14.4 Residual analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (model_name, y_pred, y_true, dataset) in enumerate(models_data):
    row = idx // 3
    col = idx % 3
    
    ax = axes[row, col]
    
    residuals = y_true - y_pred
    
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Value (μg/m³)', fontsize=11)
    ax.set_ylabel('Residual (μg/m³)', fontsize=11)
    ax.set_title(f'CNN_{model_name} - {dataset}\nMean Residual={residuals.mean():.2f}, Std={residuals.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: residuals_analysis.png")
plt.close()

# 14.5 Feature importance plot
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

top_n = 20
top_features = feature_importance.head(top_n)

ax.barh(range(top_n), top_features['Importance_Norm'], color='steelblue')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['Feature'], fontsize=10)
ax.set_xlabel('Importance (%)', fontsize=12)
ax.set_title(f'Top {top_n} Important Features (Gradient×Input Method)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: feature_importance.png")
plt.close()

# 14.6 Model performance comparison bar charts
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

test_results_plot = all_results[all_results['Dataset'] == 'Test']
models = test_results_plot['Model'].tolist()
x_pos = np.arange(len(models))
colors = ['blue', 'green']

metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
for i, metric in enumerate(metrics):
    axes[i].bar(x_pos, test_results_plot[metric], color=colors, alpha=0.7, 
                edgecolor='black', linewidth=1.5)
    axes[i].set_xticks(x_pos)
    axes[i].set_xticklabels(['Basic', 'Optimized'], fontsize=11)
    axes[i].set_ylabel(metric, fontsize=12)
    
    if metric == 'R²':
        axes[i].set_title(f'{metric} Comparison\n(Higher is Better)', fontsize=12, fontweight='bold')
    else:
        axes[i].set_title(f'{metric} Comparison\n(Lower is Better)', fontsize=12, fontweight='bold')
    
    axes[i].grid(True, alpha=0.3, axis='y')
    
    # Display values
    for j, v in enumerate(test_results_plot[metric]):
        if metric == 'MAPE':
            axes[i].text(j, v, f'{v:.1f}%', ha='center', va='bottom', 
                         fontsize=10, fontweight='bold')
        else:
            axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom', 
                         fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison.png")
plt.close()

# 14.7 Error distribution histograms
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

errors_basic = y_test_actual_basic - y_test_pred_basic
errors_opt = y_test_actual_basic - y_test_pred_opt

axes[0].hist(errors_basic, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
axes[0].set_xlabel('Prediction Error (μg/m³)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title(f'Basic Model - Prediction Error Distribution\nMean={errors_basic.mean():.2f}, Std={errors_basic.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].hist(errors_opt, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
axes[1].set_xlabel('Prediction Error (μg/m³)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title(f'Optimized Model - Prediction Error Distribution\nMean={errors_opt.mean():.2f}, Std={errors_opt.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: error_distribution.png")
plt.close()

# ============================== Part 15: Save Results ==============================
print("\n" + "=" * 80)
print("Step 11: Saving Results")
print("=" * 80)

# Save model performance
all_results.to_csv(output_dir / 'model_performance.csv', index=False, encoding='utf-8-sig')
print("Saved: model_performance.csv")

# Save feature importance
feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False, encoding='utf-8-sig')
print("Saved: feature_importance.csv")

# Save best parameters
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv(output_dir / 'best_parameters.csv', index=False, encoding='utf-8-sig')
print("Saved: best_parameters.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'Date': test_date_index,
    'Actual': y_test_actual_basic,
    'Prediction_Basic': y_test_pred_basic,
    'Prediction_Optimized': y_test_pred_opt,
    'Error_Basic': y_test_actual_basic - y_test_pred_basic,
    'Error_Optimized': y_test_actual_basic - y_test_pred_opt
})
predictions_df.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
print("Saved: predictions.csv")

# Save model (PyTorch format)
torch.save({
    'model_state_dict': model_optimized.state_dict(),
    'optimizer_state_dict': optimizer_opt.state_dict(),
    'best_epoch': best_epoch_opt,
    'train_losses': train_losses_opt,
    'val_losses': val_losses_opt,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'feature_names': feature_names,
    'hyperparameters': best_params
}, model_dir / 'cnn_optimized.pth')
print("Saved: cnn_optimized.pth")

# Save model architecture information
model_info = {
    'window_size': WINDOW_SIZE,
    'num_features': num_features,
    'total_params': total_params,
    'trainable_params': trainable_params,
    'best_params': best_params
}

with open(model_dir / 'model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("Saved: model_info.pkl")

# ============================== Part 16: Summary Report ==============================
print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)

print("\nGenerated files:")
print("\nCSV files:")
print("  - model_performance.csv       Model performance comparison")
print("  - feature_importance.csv      Feature importance")
print("  - best_parameters.csv         Best parameters")
print("  - predictions.csv             Prediction results")

print("\nChart files:")
print("  - training_curves.png         Training process curves")
print("  - prediction_scatter.png      Prediction vs actual scatter plots")
print("  - timeseries_comparison.png   Time series comparison")
print("  - residuals_analysis.png      Residual analysis")
print("  - feature_importance.png      Feature importance plot")
print("  - model_comparison.png        Model performance comparison")
print("  - error_distribution.png      Error distribution")

print("\nModel files:")
print("  - cnn_optimized.pth           CNN model (PyTorch format)")
print("  - model_info.pkl              Model information")

# Best model information
best_model = test_results.iloc[0]
print(f"\nBest model: {best_model['Model']}")
print(f"  R² Score: {best_model['R²']:.4f}")
print(f"  RMSE: {best_model['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model['MAPE']:.2f}%")

print("\nTop 5 most important features:")
for i, (idx, row) in enumerate(feature_importance.head(5).iterrows(), 1):
    print(f"  {i}. {row['Feature']}: {row['Importance_Norm']:.2f}%")

print(f"\nModel architecture:")
print(f"  Time window: {WINDOW_SIZE} days")
print(f"  Number of features: {num_features}")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

print("\n" + "=" * 80)
print("CNN PM2.5 Concentration Prediction Complete!")
print("=" * 80)
