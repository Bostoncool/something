import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import glob
import multiprocessing
from multiprocessing import Pool
import time

import xarray as xr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from itertools import product

warnings.filterwarnings('ignore')

CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

if hasattr(multiprocessing, 'set_start_method'):
    try:
        if os.name != 'nt':
            multiprocessing.set_start_method('fork', force=True)
        else:
            multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - 2D CNN Model")
print("=" * 80)

pollution_all_path = '/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)'
pollution_extra_path = '/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)'
era5_path = '/root/autodl-tmp/ERA5-Beijing-NC'

output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Output helpers (UTF-8-SIG) - naming style aligns with RF-CSV.py
# ---------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, path: Path, index: bool = False):
    """统一 CSV 输出（UTF-8-SIG），便于后续脚本读取与绘图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding='utf-8-sig')
    print(f"✓ Saved CSV: {path}")

start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

WINDOW_SIZE = 30

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = DEVICE.type == 'cuda'
NON_BLOCKING = PIN_MEMORY

DATALOADER_WORKERS = min(16, MAX_WORKERS) if DEVICE.type == 'cuda' else 0
PERSISTENT_WORKERS = DATALOADER_WORKERS > 0
PREFETCH_FACTOR = 4 if DATALOADER_WORKERS > 0 else 2

USE_AMP = True if DEVICE.type == 'cuda' else False

def get_optimal_batch_size(model_class, window_size, num_features, device, 
                           min_batch=64, max_batch=512, step=32):
    if device.type != 'cuda':
        return 32
    
    torch.cuda.empty_cache()
    test_model = model_class(
        window_size=window_size,
        num_features=num_features,
        num_conv_layers=3,
        base_filters=32,
        kernel_size=3,
        dropout_rate=0.3
    ).to(device)
    test_model.train()
    
    test_X = torch.randn(1, window_size, num_features, dtype=torch.float32).to(device)
    test_y = torch.randn(1, dtype=torch.float32).to(device)
    
    optimal_batch = min_batch
    current_batch = min_batch
    test_optimizer = optim.Adam(test_model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    
    while current_batch <= max_batch:
        try:
            torch.cuda.empty_cache()
            test_model.zero_grad()
            batch_X = test_X.repeat(current_batch, 1, 1)
            batch_y = test_y.repeat(current_batch)
            
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    y_pred = test_model(batch_X)
                    loss = nn.MSELoss()(y_pred, batch_y)
                scaler.scale(loss).backward()
                scaler.step(test_optimizer)
                scaler.update()
            else:
                y_pred = test_model(batch_X)
                loss = nn.MSELoss()(y_pred, batch_y)
                loss.backward()
                test_optimizer.step()
            
            optimal_batch = current_batch
            current_batch += step
            del batch_X, batch_y, y_pred, loss
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                test_model.zero_grad()
                if scaler:
                    scaler.update()
                break
            else:
                test_model.zero_grad()
                if scaler:
                    scaler.update()
                raise e
    
    optimal_batch = int(optimal_batch * 0.9)
    if optimal_batch < min_batch:
        optimal_batch = min_batch
    
    del test_model, test_X, test_y, test_optimizer
    if scaler:
        del scaler
    torch.cuda.empty_cache()
    
    return optimal_batch

BATCH_SIZE = 128

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
def daterange(start, end):
    """Generate date sequence"""
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def build_file_index(base_path, prefix):
    file_index = {}
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename.startswith(prefix) and filename.endswith('.csv'):
                try:
                    date_str = filename.replace(f"{prefix}_", "").replace(".csv", "")
                    if len(date_str) == 8 and date_str.isdigit():
                        file_path = os.path.join(root, filename)
                        file_index[date_str] = file_path
                except Exception:
                    continue
    return file_index

def read_pollution_day(args):
    date, file_index_all, file_index_extra, pollution_all_path, pollution_extra_path, pollutants = args
    date_str = date.strftime('%Y%m%d')
    
    all_file = file_index_all.get(date_str)
    extra_file = file_index_extra.get(date_str)
    
    if not all_file or not extra_file:
        return None
    if not os.path.exists(all_file) or not os.path.exists(extra_file):
        return None
    
    try:
        df_all = pd.read_csv(all_file, encoding='utf-8', on_bad_lines='skip')
        df_extra = pd.read_csv(extra_file, encoding='utf-8', on_bad_lines='skip')
        
        df_all = df_all[~df_all['type'].str.contains('_24h|AQI', na=False)]
        df_extra = df_extra[~df_extra['type'].str.contains('_24h', na=False)]
        
        df_poll = pd.concat([df_all, df_extra], ignore_index=True)
        df_poll = df_poll.melt(id_vars=['date', 'hour', 'type'], 
                                var_name='station', value_name='value')
        df_poll['value'] = pd.to_numeric(df_poll['value'], errors='coerce')
        df_poll = df_poll[df_poll['value'] >= 0]
        
        df_daily = df_poll.groupby(['date', 'type'])['value'].mean().reset_index()
        df_daily = df_daily.pivot(index='date', columns='type', values='value')
        df_daily.index = pd.to_datetime(df_daily.index, format='%Y%m%d', errors='coerce')
        df_daily = df_daily[[col for col in pollutants if col in df_daily.columns]]
        
        return df_daily
    except Exception:
        return None

def read_all_pollution():
    print("\nLoading pollution data...")
    file_index_all = build_file_index(pollution_all_path, 'beijing_all')
    file_index_extra = build_file_index(pollution_extra_path, 'beijing_extra')
    
    dates = list(daterange(start_date, end_date))
    args_list = [
        (date, file_index_all, file_index_extra, pollution_all_path, pollution_extra_path, pollutants)
        for date in dates
    ]
    
    pollution_dfs = []
    with Pool(processes=MAX_WORKERS) as pool:
        if TQDM_AVAILABLE:
            results = list(tqdm(
                pool.imap(read_pollution_day, args_list),
                total=len(args_list),
                desc="Loading pollution data",
                unit="day"
            ))
        else:
            results = pool.map(read_pollution_day, args_list)
        
        for result in results:
            if result is not None:
                pollution_dfs.append(result)
    
    if pollution_dfs:
        df_poll_all = pd.concat(pollution_dfs)
        df_poll_all.ffill(inplace=True)
        df_poll_all.fillna(df_poll_all.mean(), inplace=True)
        print(f"Pollution data loaded: {df_poll_all.shape}")
        return df_poll_all
    return pd.DataFrame()

def read_single_era5_file(args):
    file_path, beijing_lat_min, beijing_lat_max, beijing_lon_min, beijing_lon_max = args
    
    try:
        with xr.open_dataset(file_path, engine="netcdf4", decode_times=True) as ds:
            rename_map = {}
            for tkey in ("valid_time", "forecast_time", "verification_time", "time1", "time2"):
                if tkey in ds.coords and "time" not in ds.coords:
                    rename_map[tkey] = "time"
            if "lat" in ds.coords and "latitude" not in ds.coords:
                rename_map["lat"] = "latitude"
            if "lon" in ds.coords and "longitude" not in ds.coords:
                rename_map["lon"] = "longitude"
            if rename_map:
                ds = ds.rename(rename_map)
            
            try:
                ds = xr.decode_cf(ds)
            except Exception:
                pass
            
            drop_vars = []
            for coord in ("expver", "surface"):
                if coord in ds:
                    drop_vars.append(coord)
            if drop_vars:
                ds = ds.drop_vars(drop_vars)
            
            if "number" in ds.dims:
                ds = ds.mean(dim="number", skipna=True)
            
            if "time" not in ds.coords:
                return None
            
            data_vars = [v for v in ds.data_vars if v not in drop_vars]
            if not data_vars:
                return None
            
            ds = ds.sortby('time')
            
            if 'latitude' in ds.coords and 'longitude' in ds.coords:
                lat_values = ds['latitude']
                if len(lat_values) > 0:
                    if lat_values[0] > lat_values[-1]:
                        lat_slice = slice(beijing_lat_max, beijing_lat_min)
                    else:
                        lat_slice = slice(beijing_lat_min, beijing_lat_max)
                    ds = ds.sel(
                        latitude=lat_slice,
                        longitude=slice(beijing_lon_min, beijing_lon_max)
                    )
                    if 'latitude' in ds.dims and 'longitude' in ds.dims:
                        ds = ds.mean(dim=['latitude', 'longitude'], skipna=True)
            
            ds_daily = ds.resample(time='1D').mean(keep_attrs=False)
            ds_daily = ds_daily.dropna('time', how='all')
            
            if ds_daily.sizes.get('time', 0) == 0:
                return None
            
            ds_daily = ds_daily.load()
            
            result = {}
            for var in data_vars:
                if var in ds_daily.data_vars:
                    var_ds = ds_daily[[var]]
                    result[var] = var_ds
            
            return result if result else None
            
    except Exception:
        return None

def read_all_era5():
    """
    Read all ERA5 data from NetCDF files recursively.
    Each file may contain only one variable, so we need to:
    1. Recursively find all .nc files
    2. Read each file and extract variables
    3. Group variables by name and align by time
    4. Merge all variables together
    """
    print("\nLoading meteorological data...")
    print(f"Using {MAX_WORKERS} parallel workers")
    print(f"Meteorological data directory: {era5_path}")
    print(f"Directory exists: {os.path.exists(era5_path)}")
    
    if not os.path.exists(era5_path):
        print(f"Error: Directory {era5_path} does not exist!")
        return pd.DataFrame()
    
    # Recursively find all NetCDF files
    print("\nSearching for NetCDF files...")
    all_nc_files = glob.glob(os.path.join(era5_path, "**", "*.nc"), recursive=True)
    print(f"Found {len(all_nc_files)} NetCDF files")
    
    if not all_nc_files:
        print("Error: No NetCDF files found!")
        return pd.DataFrame()
    
    if len(all_nc_files) > 0:
        print(f"Sample files:")
        for f in all_nc_files[:5]:
            print(f"  - {os.path.basename(f)}")
        if len(all_nc_files) > 5:
            print(f"  ... and {len(all_nc_files) - 5} more files")
    
    # Read all files in parallel using multiprocessing
    print(f"\nReading {len(all_nc_files)} files in parallel...")
    variable_datasets = {}  # {variable_name: [list of datasets]}
    successful_files = 0
    failed_files = 0
    
    # Prepare arguments for multiprocessing
    beijing_lat_min = float(beijing_lats.min())
    beijing_lat_max = float(beijing_lats.max())
    beijing_lon_min = float(beijing_lons.min())
    beijing_lon_max = float(beijing_lons.max())
    
    args_list = [
        (file_path, beijing_lat_min, beijing_lat_max, beijing_lon_min, beijing_lon_max)
        for file_path in all_nc_files
    ]
    
    # Use multiprocessing Pool
    with Pool(processes=MAX_WORKERS) as pool:
        if TQDM_AVAILABLE:
            results = list(tqdm(
                pool.imap(read_single_era5_file, args_list),
                total=len(args_list),
                desc="Reading NetCDF files",
                unit="file"
            ))
        else:
            results = pool.map(read_single_era5_file, args_list)
        
        # Process results
        for i, result in enumerate(results, 1):
            try:
                if result is not None:
                    # Group datasets by variable name
                    for var_name, var_ds in result.items():
                        if var_name not in variable_datasets:
                            variable_datasets[var_name] = []
                        variable_datasets[var_name].append(var_ds)
                    successful_files += 1
                else:
                    failed_files += 1
            except Exception as exc:
                failed_files += 1
            
            if not TQDM_AVAILABLE and (i % 200 == 0 or i == len(results)):
                print(f"  Progress: {i}/{len(results)} files (success: {successful_files}, failed: {failed_files}, {i/len(results)*100:.1f}%)")
    
    print(f"\nFile reading complete:")
    print(f"  Successfully read: {successful_files} files")
    print(f"  Failed: {failed_files} files")
    print(f"  Variables found: {len(variable_datasets)}")
    
    if not variable_datasets:
        print("\nError: No variables were extracted from files!")
        return pd.DataFrame()
    
    # Print variable statistics
    print(f"\nVariable statistics:")
    for var_name, ds_list in sorted(variable_datasets.items()):
        total_days = sum(ds.sizes.get('time', 0) for ds in ds_list)
        print(f"  {var_name}: {len(ds_list)} files, ~{total_days} time points")
    
    # Merge datasets for each variable
    print(f"\nMerging datasets by variable...")
    merged_variables = {}
    
    for var_name, ds_list in variable_datasets.items():
        if not ds_list:
            continue
        
        try:
            # Merge all datasets for this variable
            merged_ds = xr.merge(ds_list, compat='override', join='outer')
            
            # Convert to DataFrame
            df_var = merged_ds.to_dataframe()
            
            # Handle time index
            if 'time' in df_var.index.names:
                df_var.index = pd.to_datetime(df_var.index.get_level_values('time'))
            elif isinstance(df_var.index, pd.DatetimeIndex):
                df_var.index = pd.to_datetime(df_var.index)
            
            # Remove duplicates and sort
            df_var = df_var[~df_var.index.duplicated(keep='first')]
            df_var.sort_index(inplace=True)
            
            # Keep only the variable column (remove coordinate columns)
            var_cols = [col for col in df_var.columns if col == var_name]
            if var_cols:
                merged_variables[var_name] = df_var[var_cols]
                print(f"  [+] {var_name}: {len(df_var)} time points")
            else:
                print(f"  [-] {var_name}: No valid columns found")
        except Exception as exc:
            print(f"  [ERROR] Failed to merge {var_name}: {type(exc).__name__}: {exc}")
            continue
    
    if not merged_variables:
        print("\nError: No variables were successfully merged!")
        return pd.DataFrame()
    
    # Merge all variables together
    print(f"\nMerging all variables...")
    df_list = list(merged_variables.values())
    df_era5_all = pd.concat(df_list, axis=1, join='outer')
    
    # Remove duplicates and sort
    print("  Removing duplicates...")
    df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep='first')]
    print("  Sorting by time...")
    df_era5_all.sort_index(inplace=True)
    
    print(f"\nMerged shape: {df_era5_all.shape}")
    print(f"Time range: {df_era5_all.index.min()} to {df_era5_all.index.max()}")
    print(f"Variables: {list(df_era5_all.columns)}")
    
    # Filter by date range
    print(f"\nFiltering by date range ({start_date.date()} to {end_date.date()})...")
    initial_rows = len(df_era5_all)
    df_era5_all = df_era5_all.loc[
        (df_era5_all.index >= start_date) & (df_era5_all.index <= end_date)
    ]
    final_rows = len(df_era5_all)
    print(f"  Rows: {initial_rows} -> {final_rows}")
    
    # Handle missing values
    print("\nHandling missing values...")
    initial_na = df_era5_all.isna().sum().sum()
    df_era5_all.ffill(inplace=True)
    df_era5_all.bfill(inplace=True)
    df_era5_all.fillna(df_era5_all.mean(), inplace=True)
    final_na = df_era5_all.isna().sum().sum()
    
    print(f"Missing value handling: {initial_na} -> {final_na}")
    print(f"Meteorological data loading complete, shape: {df_era5_all.shape}")
    
    return df_era5_all

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

df_era5 = read_all_era5()
df_pollution = read_all_pollution()


# Check data loading
print("\nData loading check:")
print(f"  Pollution data shape: {df_pollution.shape}")
print(f"  Meteorological data shape: {df_era5.shape}")

if df_pollution.empty:
    print("\nWarning: Pollution data is empty! Please check data path and files.")
    import sys
    sys.exit(1)

if df_era5.empty:
    print("\nWarning: Meteorological data is empty! Please check data path and files.")
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
    print("\nError: Data is empty after merging!")
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

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=DATALOADER_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else None,
    drop_last=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=DATALOADER_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else None,
    drop_last=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=DATALOADER_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else None,
    drop_last=False
)

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
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """
    Train one epoch with mixed precision support
    
    Args:
        model: Model to train
        dataloader: Data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Computing device
        scaler: GradScaler for mixed precision (optional)
    """
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device, non_blocking=NON_BLOCKING)
        y_batch = y_batch.to(device, non_blocking=NON_BLOCKING)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * len(X_batch)
    
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device, use_amp=False):
    """
    Validate model with mixed precision support
    
    Args:
        model: Model to validate
        dataloader: Data loader
        criterion: Loss function
        device: Computing device
        use_amp: Whether to use mixed precision
    """
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device, non_blocking=NON_BLOCKING)
            y_batch = y_batch.to(device, non_blocking=NON_BLOCKING)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
            else:
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
                num_epochs, device, patience=20, verbose=True, use_amp=False):
    """
    Train model with early stopping and mixed precision support
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs
        device: Computing device
        patience: Early stopping patience
        verbose: Whether to print progress
        use_amp: Whether to use mixed precision training
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    print(f"\nTraining {num_epochs} epochs...")
    if use_amp:
        print("  Mixed precision training (AMP): Enabled")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, _, _ = validate(model, val_loader, criterion, device, use_amp)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
        
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
    
    total_time = time.time() - start_time
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"Training complete! Best model at epoch {best_epoch}, validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
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

# Optimize batch size for GPU
if DEVICE.type == 'cuda':
    print("\n" + "=" * 80)
    print("Optimizing Batch Size for GPU Memory")
    print("=" * 80)
    optimal_batch_size = get_optimal_batch_size(
        PM25CNN2D, WINDOW_SIZE, num_features, DEVICE
    )
    
    if optimal_batch_size != BATCH_SIZE:
        print(f"\nUpdating batch size: {BATCH_SIZE} -> {optimal_batch_size}")
        BATCH_SIZE = optimal_batch_size
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=DATALOADER_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else None,
            drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=DATALOADER_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else None,
            drop_last=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=DATALOADER_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else None,
            drop_last=False
        )
        print(f"DataLoader updated, new batch size: {BATCH_SIZE}")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

USE_COMPILE = False
if USE_COMPILE and hasattr(torch, 'compile'):
    print("\nEnabling PyTorch compilation optimization (torch.compile)...")
    model_basic = torch.compile(model_basic, mode='reduce-overhead')
    print("Model compilation complete")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer_basic = optim.Adam(model_basic.parameters(), lr=basic_params['learning_rate'])

# Train basic model
train_losses_basic, val_losses_basic, best_epoch_basic = train_model(
    model_basic, train_loader, val_loader, criterion, optimizer_basic,
    num_epochs=basic_params['num_epochs'],
    device=DEVICE,
    patience=basic_params['patience'],
    verbose=True,
    use_amp=USE_AMP
)

print(f"\n✓ Basic model training complete")
print(f"  Best epoch: {best_epoch_basic}")
print(f"  Final training loss: {train_losses_basic[best_epoch_basic-1]:.4f}")
print(f"  Final validation loss: {val_losses_basic[best_epoch_basic-1]:.4f}")

# Evaluate basic model
print("\nEvaluating basic model...")

_, y_train_pred_basic_scaled, y_train_actual_scaled = validate(model_basic, train_loader, criterion, DEVICE, USE_AMP)
_, y_val_pred_basic_scaled, y_val_actual_scaled = validate(model_basic, val_loader, criterion, DEVICE, USE_AMP)
_, y_test_pred_basic_scaled, y_test_actual_scaled = validate(model_basic, test_loader, criterion, DEVICE, USE_AMP)

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

# ============================== Part 10: Hyperparameter Optimization (Grid Search) ==============================
print("\n" + "=" * 80)
print("Step 6: Hyperparameter Optimization (Grid Search)")
print("=" * 80)

# Define grid search parameter space
param_grid = {
    'num_conv_layers': [2, 3, 4],
    'base_filters': [32, 64],
    'kernel_size': [3, 5],
    'learning_rate': [0.0005, 0.001, 0.005],
    'dropout_rate': [0.2, 0.3, 0.4]
}

total_combinations = int(np.prod([len(v) for v in param_grid.values()]))
print(f"\nGrid search configuration:")
print(f"  Total parameter combinations: {total_combinations}")
print(f"\nParameter search space:")
for key, values in param_grid.items():
    print(f"  {key}: {values}")

grid_search_results = []
best_val_loss_grid = float('inf')
best_params = {}

if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()

grid_search_start_time = time.time()

print(f"\nStarting grid search...")
print("=" * 80)

if TQDM_AVAILABLE:
    param_combinations = list(product(*param_grid.values()))
    iterator = tqdm(enumerate(param_combinations, 1), total=total_combinations, 
                    desc="Grid search progress", unit="combination")
else:
    param_combinations = list(product(*param_grid.values()))
    iterator = enumerate(param_combinations, 1)

for i, combo in iterator:
    params_test = dict(zip(param_grid.keys(), combo))
    
    if not TQDM_AVAILABLE:
        print(f"\n[{i}/{total_combinations}] Testing parameter combination:")
        for key, value in params_test.items():
            print(f"  {key}: {value}")
    
    try:
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        
        model_temp = PM25CNN2D(
            window_size=WINDOW_SIZE,
            num_features=num_features,
            num_conv_layers=params_test['num_conv_layers'],
            base_filters=params_test['base_filters'],
            kernel_size=params_test['kernel_size'],
            dropout_rate=params_test['dropout_rate']
        ).to(DEVICE)
        
        optimizer_temp = optim.Adam(model_temp.parameters(), lr=params_test['learning_rate'])
        
        train_start_time = time.time()
        _, _, best_epoch_temp = train_model(
            model_temp, train_loader, val_loader, criterion, optimizer_temp,
            num_epochs=100, device=DEVICE, patience=15, verbose=False, use_amp=USE_AMP
        )
        train_time = time.time() - train_start_time
        
        val_loss, _, _ = validate(model_temp, val_loader, criterion, DEVICE, USE_AMP)
        
        result_entry = params_test.copy()
        result_entry['val_loss'] = val_loss
        result_entry['best_epoch'] = best_epoch_temp
        result_entry['train_time'] = train_time
        grid_search_results.append(result_entry)
        
        if val_loss < best_val_loss_grid:
            best_val_loss_grid = val_loss
            best_params = params_test.copy()
            if not TQDM_AVAILABLE:
                print(f"  New best validation loss: {val_loss:.4f} (Epoch {best_epoch_temp}, Time: {train_time:.1f}s)")
        else:
            if not TQDM_AVAILABLE:
                print(f"  Validation loss: {val_loss:.4f} (Epoch {best_epoch_temp}, Time: {train_time:.1f}s)")
        
        del model_temp, optimizer_temp
        
    except Exception as e:
        print(f"\n  Parameter combination {i} failed: {type(e).__name__}: {e}")
        result_entry = params_test.copy()
        result_entry['val_loss'] = float('inf')
        result_entry['best_epoch'] = 0
        result_entry['train_time'] = 0
        result_entry['error'] = str(e)
        grid_search_results.append(result_entry)
        continue

grid_search_total_time = time.time() - grid_search_start_time

grid_search_df = pd.DataFrame(grid_search_results)
grid_search_df = grid_search_df.sort_values('val_loss', ascending=True)
save_csv(grid_search_df, output_dir / 'grid_search_results__cnn.csv', index=False)
print(f"\nGrid search results saved: grid_search_results__cnn.csv")

print("\n" + "=" * 80)
print("Grid search complete!")
print("=" * 80)
print(f"\nBest parameter combination:")
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"  Best validation loss: {best_val_loss_grid:.4f}")
print(f"  Total search time: {grid_search_total_time:.2f}s ({grid_search_total_time/60:.2f} minutes)")
print(f"  Average time per combination: {grid_search_total_time/total_combinations:.2f}s")

print(f"\nTop 5 best parameter combinations:")
top5_results = grid_search_df.head(5)
for idx, (_, row) in enumerate(top5_results.iterrows(), 1):
    print(f"\n  Rank {idx}:")
    print(f"    Validation loss: {row['val_loss']:.4f}")
    print(f"    Best epoch: {row['best_epoch']}")
    print(f"    Training time: {row['train_time']:.1f}s")
    for key in param_grid.keys():
        print(f"    {key}: {row[key]}")

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
    num_epochs=300, device=DEVICE, patience=30, verbose=True, use_amp=USE_AMP
)

print(f"\n✓ Optimized model training complete")
print(f"  Best epoch: {best_epoch_opt}")
print(f"  Final training loss: {train_losses_opt[best_epoch_opt-1]:.4f}")
print(f"  Final validation loss: {val_losses_opt[best_epoch_opt-1]:.4f}")

# Evaluate optimized model
print("\nEvaluating optimized model...")

_, y_train_pred_opt_scaled, _ = validate(model_optimized, train_loader, criterion, DEVICE, USE_AMP)
_, y_val_pred_opt_scaled, _ = validate(model_optimized, val_loader, criterion, DEVICE, USE_AMP)
_, y_test_pred_opt_scaled, _ = validate(model_optimized, test_loader, criterion, DEVICE, USE_AMP)

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

# ============================== Part 14: CSV Outputs for Plotting (No Image Generation) ==============================
print("\n" + "=" * 80)
print("Step 10: Export Plot CSVs (No Image Generation)")
print("=" * 80)

# ---------------------------------------------------------------------------
# 10.1 Training curves (per model)
# ---------------------------------------------------------------------------
train_curves_basic_df = pd.DataFrame({
    'epoch': np.arange(1, len(train_losses_basic) + 1),
    'train_loss': train_losses_basic,
    'val_loss': val_losses_basic,
})
train_curves_basic_df['is_best_epoch'] = (train_curves_basic_df['epoch'] == int(best_epoch_basic)).astype(int)
save_csv(train_curves_basic_df, output_dir / 'plot_training_curves__cnn_basic.csv', index=False)

train_curves_opt_df = pd.DataFrame({
    'epoch': np.arange(1, len(train_losses_opt) + 1),
    'train_loss': train_losses_opt,
    'val_loss': val_losses_opt,
})
train_curves_opt_df['is_best_epoch'] = (train_curves_opt_df['epoch'] == int(best_epoch_opt)).astype(int)
save_csv(train_curves_opt_df, output_dir / 'plot_training_curves__cnn_optimized.csv', index=False)

# ---------------------------------------------------------------------------
# 10.2 Metrics (consistent naming style with RF-CSV)
# ---------------------------------------------------------------------------
metrics_all = all_results.copy()
metrics_all['Model'] = metrics_all['Model'].astype(str).str.lower()
metrics_all['Dataset'] = metrics_all['Dataset'].astype(str).str.lower()
save_csv(metrics_all, output_dir / 'metrics__all_models__train_val_test.csv', index=False)

for model_name in metrics_all['Model'].unique():
    model_part = metrics_all[metrics_all['Model'] == model_name]
    save_csv(model_part, output_dir / f'metrics__{model_name}__train_val_test.csv', index=False)
    for ds_name in model_part['Dataset'].unique():
        ds_part = model_part[model_part['Dataset'] == ds_name]
        if not ds_part.empty:
            save_csv(ds_part, output_dir / f'metrics__{model_name}__{ds_name}.csv', index=False)

test_ranking = metrics_all[metrics_all['Dataset'] == 'test'].sort_values('R²', ascending=False)
save_csv(test_ranking, output_dir / 'plot_metrics_ranking__test_only.csv', index=False)

# ---------------------------------------------------------------------------
# 10.3 Hyperparameters (per model)
# ---------------------------------------------------------------------------
save_csv(pd.DataFrame([basic_params]), output_dir / 'cnn_parameters__cnn_basic.csv', index=False)
save_csv(pd.DataFrame([best_params]), output_dir / 'cnn_parameters__cnn_optimized.csv', index=False)

# ---------------------------------------------------------------------------
# 10.4 Feature importance (optimized CNN + TopN)
# ---------------------------------------------------------------------------
feature_importance_out = feature_importance.copy().reset_index(drop=True)
save_csv(feature_importance_out, output_dir / 'plot_feature_importance__cnn_optimized.csv', index=False)
top_n = min(20, len(feature_importance_out))
save_csv(feature_importance_out.head(top_n), output_dir / f'plot_feature_importance_top{top_n}__cnn_optimized.csv', index=False)

# ---------------------------------------------------------------------------
# 10.5 Scatter / residuals (per model × per dataset)
# ---------------------------------------------------------------------------
train_date_index = pd.to_datetime(date_index[:train_size])
val_date_index = pd.to_datetime(date_index[train_size:train_size + val_size])
test_date_index = pd.to_datetime(date_index[train_size + val_size:])

models_data = [
    ('cnn_basic', y_train_pred_basic, y_train_actual_basic, train_date_index, 'train'),
    ('cnn_basic', y_val_pred_basic, y_val_actual_basic, val_date_index, 'val'),
    ('cnn_basic', y_test_pred_basic, y_test_actual_basic, test_date_index, 'test'),
    ('cnn_optimized', y_train_pred_opt, y_train_actual_basic, train_date_index, 'train'),
    ('cnn_optimized', y_val_pred_opt, y_val_actual_basic, val_date_index, 'val'),
    ('cnn_optimized', y_test_pred_opt, y_test_actual_basic, test_date_index, 'test'),
]

for model_name, y_pred, y_true, dt_idx, dataset_name in models_data:
    scatter_df = pd.DataFrame({
        'Date': pd.to_datetime(dt_idx),
        'Actual_PM25': np.asarray(y_true),
        'Predicted_PM25': np.asarray(y_pred),
    })
    save_csv(scatter_df, output_dir / f'plot_scatter__{model_name}__{dataset_name}.csv', index=False)

    residuals = np.asarray(y_true) - np.asarray(y_pred)
    residuals_df = pd.DataFrame({
        'Date': pd.to_datetime(dt_idx),
        'Actual_PM25': np.asarray(y_true),
        'Predicted_PM25': np.asarray(y_pred),
        'Residual': residuals,
    })
    save_csv(residuals_df, output_dir / f'plot_residuals__{model_name}__{dataset_name}.csv', index=False)

# ---------------------------------------------------------------------------
# 10.6 Time series compare (test set, last N points, sampled)
#      - Use integer x_axis to avoid date gaps causing “knotting” (same idea as RF-CSV)
# ---------------------------------------------------------------------------
plot_range = min(300, len(y_test_actual_basic))

plot_df = pd.DataFrame({
    'time': test_date_index,
    'y_true': np.asarray(y_test_actual_basic),
    'y_pred_cnn_basic': np.asarray(y_test_pred_basic),
    'y_pred_cnn_optimized': np.asarray(y_test_pred_opt),
}).sort_values('time').reset_index(drop=True)

plot_df_subset = plot_df.iloc[-plot_range:].copy()
step = 4
plot_df_sampled = plot_df_subset.iloc[::step].copy().reset_index(drop=True)
plot_df_sampled['x_axis'] = np.arange(len(plot_df_sampled))

ts_common = plot_df_sampled[['x_axis', 'time', 'y_true']].copy()
save_csv(ts_common, output_dir / f'plot_ts_last{plot_range}_sampled__actual.csv', index=False)

ts_basic = ts_common.assign(y_pred=plot_df_sampled['y_pred_cnn_basic'].values)
save_csv(ts_basic, output_dir / f'plot_ts_last{plot_range}_sampled__cnn_basic.csv', index=False)

ts_opt = ts_common.assign(y_pred=plot_df_sampled['y_pred_cnn_optimized'].values)
save_csv(ts_opt, output_dir / f'plot_ts_last{plot_range}_sampled__cnn_optimized.csv', index=False)

# ---------------------------------------------------------------------------
# 10.7 Error distribution (per model, test set)
# ---------------------------------------------------------------------------
errors_basic = np.asarray(y_test_actual_basic) - np.asarray(y_test_pred_basic)
errors_opt = np.asarray(y_test_actual_basic) - np.asarray(y_test_pred_opt)

error_basic_df = pd.DataFrame({
    'Date': test_date_index,
    'Actual_PM25': np.asarray(y_test_actual_basic),
    'Predicted_PM25': np.asarray(y_test_pred_basic),
    'Error': errors_basic,
})
save_csv(error_basic_df, output_dir / 'plot_error_distribution__cnn_basic__test.csv', index=False)

error_opt_df = pd.DataFrame({
    'Date': test_date_index,
    'Actual_PM25': np.asarray(y_test_actual_basic),
    'Predicted_PM25': np.asarray(y_test_pred_opt),
    'Error': errors_opt,
})
save_csv(error_opt_df, output_dir / 'plot_error_distribution__cnn_optimized__test.csv', index=False)

# ============================== Part 15: Save Results ==============================
print("\n" + "=" * 80)
print("Step 11: Saving Results")
print("=" * 80)

# Save predictions (split + combined, UTF-8-SIG)
pred_test_basic_df = pd.DataFrame({
    'Date': test_date_index,
    'Actual_PM25': np.asarray(y_test_actual_basic),
    'Predicted_PM25': np.asarray(y_test_pred_basic),
    'Error': errors_basic,
})
save_csv(pred_test_basic_df, output_dir / 'cnn_predictions__cnn_basic__test.csv', index=False)

pred_test_opt_df = pd.DataFrame({
    'Date': test_date_index,
    'Actual_PM25': np.asarray(y_test_actual_basic),
    'Predicted_PM25': np.asarray(y_test_pred_opt),
    'Error': errors_opt,
})
save_csv(pred_test_opt_df, output_dir / 'cnn_predictions__cnn_optimized__test.csv', index=False)

predictions_all_models_df = pd.DataFrame({
    'Date': test_date_index,
    'Actual_PM25': np.asarray(y_test_actual_basic),
    'Predicted_CNN_Basic': np.asarray(y_test_pred_basic),
    'Predicted_CNN_Optimized': np.asarray(y_test_pred_opt),
    'Error_CNN_Basic': errors_basic,
    'Error_CNN_Optimized': errors_opt,
})
save_csv(predictions_all_models_df, output_dir / 'cnn_predictions_all_models__test.csv', index=False)

# ============================== Part 16: Summary Report ==============================
print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)

print("\nGenerated files:")
print("\nCSV files (for external plotting scripts):")
print("  - grid_search_results__cnn.csv")
print("  - metrics__all_models__train_val_test.csv")
print("  - metrics__cnn_basic__train_val_test.csv / metrics__cnn_optimized__train_val_test.csv")
print("  - metrics__cnn_basic__train.csv / val.csv / test.csv (and optimized)")
print("  - plot_metrics_ranking__test_only.csv")
print("  - cnn_parameters__cnn_basic.csv / cnn_parameters__cnn_optimized.csv")
print("  - plot_training_curves__cnn_basic.csv / plot_training_curves__cnn_optimized.csv")
print("  - plot_scatter__cnn_basic__train.csv / val.csv / test.csv (and optimized)")
print("  - plot_residuals__cnn_basic__train.csv / val.csv / test.csv (and optimized)")
print("  - plot_ts_last300_sampled__actual.csv / __cnn_basic.csv / __cnn_optimized.csv")
print("  - plot_error_distribution__cnn_basic__test.csv / __cnn_optimized__test.csv")
print("  - plot_feature_importance__cnn_optimized.csv / plot_feature_importance_top20__cnn_optimized.csv")
print("  - cnn_predictions__cnn_basic__test.csv / cnn_predictions__cnn_optimized__test.csv / cnn_predictions_all_models__test.csv")

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
