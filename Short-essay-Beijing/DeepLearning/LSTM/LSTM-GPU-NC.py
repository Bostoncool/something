import os
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from pathlib import Path
import glob
import multiprocessing
from itertools import product
import time
import xarray as xr
from netCDF4 import Dataset as NetCDFDataset

warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training

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

# Machine learning libraries
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Set English font for matplotlib
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus signs are displayed correctly
plt.rcParams['figure.dpi'] = 100

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # GPU performance optimization settings
    torch.backends.cudnn.benchmark = True  # Automatically find optimal algorithms
    torch.backends.cudnn.deterministic = False  # Allow non-determinism for speed improvement
    # Enable TensorFloat-32 for faster matrix operations (Ampere+ GPUs)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - LSTM + Attention Model (GPU Accelerated Version)")
print("=" * 80)
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU model: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Current GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
else:
    print("Warning: GPU not detected, will use CPU (slower speed)")

# ============================== Part 1: Configuration and Path Setup ==============================
print("\nConfiguring parameters...")

# Data paths
pollution_all_path = r'/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)'
pollution_extra_path = r'/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)'
era5_path = r'/root/autodl-tmp/ERA5-Beijing-NC'

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
    'd2m', 't2m', 'u10', 'v10', 'u100', 'v100',
    'blh', 'sp', 'tcwv',
    'tp', 'avg_tprate',
    'tisr', 'str',
    'cvh', 'cvl',
    'mn2t', 'sd', 'lsm'
]

# LSTM specific configuration - GPU optimized version
SEQUENCE_LENGTHS = [7, 14, 30]
BATCH_SIZE = 256
EPOCHS = 100
EARLY_STOP_PATIENCE = 20

# GPU optimization configuration
USE_AMP = True
NUM_WORKERS = 8
PIN_MEMORY = True
PREFETCH_FACTOR = 4
USE_COMPILE = True

print(f"Data time range: {start_date.date()} to {end_date.date()}")
print(f"Target variable: PM2.5 concentration")
print(f"Sequence lengths: {SEQUENCE_LENGTHS} days")
print(f"Batch size: {BATCH_SIZE} (GPU optimized, increased for better utilization)")
print(f"Mixed precision training: {'Enabled' if USE_AMP else 'Disabled'}")
print(f"DataLoader worker threads: {NUM_WORKERS} (optimized)")
print(f"Prefetch factor: {PREFETCH_FACTOR} (reduces GPU idle time)")
print(f"Torch compile: {'Enabled' if USE_COMPILE and hasattr(torch, 'compile') else 'Disabled'}")
print(f"Output directory: {output_dir}")
print(f"Model save directory: {model_dir}")
print(f"CPU cores: {CPU_COUNT}, Parallel worker threads: {MAX_WORKERS}")

# ============================== Part 2: GPU Performance Monitoring Tools ==============================
def print_gpu_memory_usage(stage=""):
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
        if stage:
            print(f"  [{stage}] GPU memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={max_allocated:.2f}GB")
        else:
            print(f"  GPU memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={max_allocated:.2f}GB")

def get_gpu_utilization():
    """Get GPU utilization percentage using nvidia-smi"""
    if not torch.cuda.is_available():
        return None
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split('\n')[0])
    except Exception:
        pass
    return None

def print_gpu_status(stage=""):
    """Print comprehensive GPU status"""
    if torch.cuda.is_available():
        print_gpu_memory_usage(stage)
        gpu_util = get_gpu_utilization()
        if gpu_util is not None:
            print(f"  GPU utilization: {gpu_util}%")

def clear_gpu_memory():
    """Clear GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ============================== Part 3: Data Loading Functions (Reused) ==============================
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

def read_pollution_day(date, pollution_all_path_local=None, pollution_extra_path_local=None, pollutants_list=None, file_map_all=None, file_map_extra=None):
    """Read single day pollution data"""
    if pollution_all_path_local is None:
        pollution_all_path_local = pollution_all_path
    if pollution_extra_path_local is None:
        pollution_extra_path_local = pollution_extra_path
    if pollutants_list is None:
        pollutants_list = pollutants
    
    date_str = date.strftime('%Y%m%d')
    
    if file_map_all is not None and file_map_extra is not None:
        filename_all = f"beijing_all_{date_str}.csv"
        filename_extra = f"beijing_extra_{date_str}.csv"
        all_file = file_map_all.get(filename_all)
        extra_file = file_map_extra.get(filename_extra)
    else:
        all_file = find_file(pollution_all_path_local, date_str, 'beijing_all')
        extra_file = find_file(pollution_extra_path_local, date_str, 'beijing_extra')
    
    if not all_file or not extra_file:
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
        df_daily = df_daily[[col for col in pollutants_list if col in df_daily.columns]]
        
        return df_daily
    except Exception as e:
        return None

def build_file_map(base_path, prefix):
    """Build file path mapping to avoid repeated directory traversal"""
    file_map = {}
    filename_pattern = f"{prefix}_"
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename.startswith(filename_pattern) and filename.endswith('.csv'):
                file_map[filename] = os.path.join(root, filename)
    return file_map

def read_all_pollution():
    """Read all pollution data using multiprocessing"""
    print("\nLoading pollution data...")
    
    file_map_all = build_file_map(pollution_all_path, 'beijing_all')
    file_map_extra = build_file_map(pollution_extra_path, 'beijing_extra')
    print(f"Found {len(file_map_all)} 'all' files and {len(file_map_extra)} 'extra' files")
    
    dates = list(daterange(start_date, end_date))
    pollution_dfs = []
    
    pollution_all_path_local = pollution_all_path
    pollution_extra_path_local = pollution_extra_path
    pollutants_list = list(pollutants)
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, date, pollution_all_path_local, 
                                   pollution_extra_path_local, pollutants_list, 
                                   file_map_all, file_map_extra): date for date in dates}
        
        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Loading pollution data", unit="days"):
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

def read_single_nc_file(file_path, era5_vars_list=None, beijing_lats_array=None, beijing_lons_array=None):
    """Read single NetCDF file and extract variable data"""
    if era5_vars_list is None:
        era5_vars_list = era5_vars
    if beijing_lats_array is None:
        beijing_lats_array = beijing_lats
    if beijing_lons_array is None:
        beijing_lons_array = beijing_lons
    
    try:
        with NetCDFDataset(file_path, mode='r') as nc_file:
            file_vars = list(nc_file.variables.keys())
            available_vars = [v for v in era5_vars_list if v in file_vars]
            
            if not available_vars:
                coord_vars = ['time', 'latitude', 'longitude', 'lat', 'lon', 
                             'expver', 'surface', 'number', 'valid_time', 
                             'forecast_time', 'verification_time', 'time1', 'time2']
                data_vars = [v for v in file_vars if v not in coord_vars]
                if data_vars:
                    available_vars = data_vars[:1]
                else:
                    return None
        
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
            
            var_to_read = available_vars[0] if available_vars else None
            if var_to_read and var_to_read not in ds.data_vars:
                return None
            
            if var_to_read:
                ds_subset = ds[[var_to_read]]
            else:
                data_vars_list = list(ds.data_vars)
                if not data_vars_list:
                    return None
                var_to_read = data_vars_list[0]
                ds_subset = ds[[var_to_read]]
            
            ds_subset = ds_subset.sortby('time')
            
            if 'latitude' in ds_subset.coords and 'longitude' in ds_subset.coords:
                lat_values = ds_subset['latitude']
                if len(lat_values) > 0:
                    if lat_values[0] > lat_values[-1]:
                        lat_slice = slice(beijing_lats_array.max(), beijing_lats_array.min())
                    else:
                        lat_slice = slice(beijing_lats_array.min(), beijing_lats_array.max())
                    ds_subset = ds_subset.sel(
                        latitude=lat_slice,
                        longitude=slice(beijing_lons_array.min(), beijing_lons_array.max())
                    )
                    if 'latitude' in ds_subset.dims and 'longitude' in ds_subset.dims:
                        ds_subset = ds_subset.mean(dim=['latitude', 'longitude'], skipna=True)
            
            ds_daily = ds_subset.resample(time='1D').mean(keep_attrs=False)
            ds_daily = ds_daily.dropna('time', how='all')
            
            if ds_daily.sizes.get('time', 0) == 0:
                return None
            
            df = ds_daily.to_dataframe()
            df.index = pd.to_datetime(df.index)
            
            return (var_to_read, df)
            
    except Exception as exc:
        return None

def read_all_era5():
    """Read all ERA5 NetCDF files recursively, group by variable, then merge by time"""
    print("\nLoading meteorological data...")
    
    if not os.path.exists(era5_path):
        print(f"Error: Directory {era5_path} does not exist!")
        return pd.DataFrame()
    
    all_nc_files = glob.glob(os.path.join(era5_path, "**", "*.nc"), recursive=True)
    print(f"Found {len(all_nc_files)} NetCDF files")
    
    if len(all_nc_files) == 0:
        print("Error: No NetCDF files found!")
        return pd.DataFrame()
    
    print(f"Reading {len(all_nc_files)} files in parallel...")
    file_results = {}
    
    era5_vars_list = list(era5_vars)
    beijing_lats_array = np.array(beijing_lats)
    beijing_lons_array = np.array(beijing_lons)
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_single_nc_file, file_path, era5_vars_list, beijing_lats_array, beijing_lons_array): file_path 
                  for file_path in all_nc_files}
        
        successful_reads = 0
        failed_reads = 0
        
        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Reading NetCDF files", unit="files"):
                result = future.result()
                if result is not None:
                    var_name, df = result
                    if var_name not in file_results:
                        file_results[var_name] = []
                    file_results[var_name].append(df)
                    successful_reads += 1
                else:
                    failed_reads += 1
        else:
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    var_name, df = result
                    if var_name not in file_results:
                        file_results[var_name] = []
                    file_results[var_name].append(df)
                    successful_reads += 1
                else:
                    failed_reads += 1
                
                if i % 100 == 0 or i == len(futures):
                    print(f"  Progress: {i}/{len(futures)} files (success: {successful_reads}, failed: {failed_reads}, {i/len(futures)*100:.1f}%)")
    
    print(f"Reading complete: {successful_reads} successful, {failed_reads} failed")
    print(f"Found {len(file_results)} unique variables")
    
    if not file_results:
        print("Error: No data was successfully read!")
        return pd.DataFrame()
    
    print("\nMerging data by variable...")
    variable_dfs = {}
    
    for var_name, df_list in file_results.items():
        print(f"Processing variable '{var_name}': {len(df_list)} files")
        
        if len(df_list) == 1:
            var_df = df_list[0].copy()
        else:
            var_df = pd.concat(df_list, axis=0)
        
        var_df = var_df[~var_df.index.duplicated(keep='first')]
        var_df.sort_index(inplace=True)
        
        if len(var_df.columns) == 1:
            var_df.columns = [var_name]
        elif var_name not in var_df.columns:
            first_col = var_df.columns[0]
            var_df = var_df[[first_col]].copy()
            var_df.columns = [var_name]
        
        variable_dfs[var_name] = var_df
        print(f"  Time range: {var_df.index.min()} to {var_df.index.max()}, {len(var_df)} days")
    
    print("\nMerging all variables by time...")
    if len(variable_dfs) == 1:
        df_era5_all = list(variable_dfs.values())[0]
    else:
        var_names_list = list(variable_dfs.keys())
        df_era5_all = variable_dfs[var_names_list[0]].copy()
        
        for var_name in var_names_list[1:]:
            var_df = variable_dfs[var_name]
            df_era5_all = df_era5_all.join(var_df, how='outer', rsuffix='_dup')
            if f'{var_name}_dup' in df_era5_all.columns:
                df_era5_all[f'{var_name}'] = df_era5_all[f'{var_name}'].fillna(df_era5_all[f'{var_name}_dup'])
                df_era5_all.drop(columns=[f'{var_name}_dup'], inplace=True)
    
    print(f"Merged shape: {df_era5_all.shape}")
    print(f"Time range: {df_era5_all.index.min()} to {df_era5_all.index.max()}")
    
    df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep='first')]
    df_era5_all.sort_index(inplace=True)
    
    initial_na = df_era5_all.isna().sum().sum()
    df_era5_all.ffill(inplace=True)
    df_era5_all.bfill(inplace=True)
    df_era5_all.fillna(df_era5_all.mean(), inplace=True)
    final_na = df_era5_all.isna().sum().sum()
    
    print(f"Missing values: {initial_na} -> {final_na}")
    print(f"Meteorological data loading complete, shape: {df_era5_all.shape}")
    
    return df_era5_all

# ============================== Part 4: Feature Engineering (Reused) ==============================
def create_features(df):
    """Create additional features"""
    df_copy = df.copy()
    
    # Wind speed features
    if 'u10' in df_copy and 'v10' in df_copy:
        df_copy['wind_speed_10m'] = np.sqrt(df_copy['u10']**2 + df_copy['v10']**2)
        df_copy['wind_dir_10m'] = np.arctan2(df_copy['v10'], df_copy['u10']) * 180 / np.pi
        df_copy['wind_dir_10m'] = (df_copy['wind_dir_10m'] + 360) % 360
    
    if 'u100' in df_copy and 'v100' in df_copy:
        df_copy['wind_speed_100m'] = np.sqrt(df_copy['u100']**2 + df_copy['v100']**2)
        df_copy['wind_dir_100m'] = np.arctan2(df_copy['v100'], df_copy['u100']) * 180 / np.pi
        df_copy['wind_dir_100m'] = (df_copy['wind_dir_100m'] + 360) % 360
    
    # Time features
    df_copy['year'] = df_copy.index.year
    df_copy['month'] = df_copy.index.month
    df_copy['day'] = df_copy.index.day
    df_copy['day_of_year'] = df_copy.index.dayofyear
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['week_of_year'] = df_copy.index.isocalendar().week
    
    # Season features
    df_copy['season'] = df_copy['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )
    
    # Heating season
    df_copy['is_heating_season'] = ((df_copy['month'] >= 11) | (df_copy['month'] <= 3)).astype(int)
    
    # Temperature related
    if 't2m' in df_copy and 'd2m' in df_copy:
        df_copy['temp_dewpoint_diff'] = df_copy['t2m'] - df_copy['d2m']
    
    # Lag features
    if 'PM2.5' in df_copy:
        df_copy['PM2.5_lag1'] = df_copy['PM2.5'].shift(1)
        df_copy['PM2.5_lag3'] = df_copy['PM2.5'].shift(3)
        df_copy['PM2.5_lag7'] = df_copy['PM2.5'].shift(7)
        df_copy['PM2.5_ma3'] = df_copy['PM2.5'].rolling(window=3, min_periods=1).mean()
        df_copy['PM2.5_ma7'] = df_copy['PM2.5'].rolling(window=7, min_periods=1).mean()
        df_copy['PM2.5_ma30'] = df_copy['PM2.5'].rolling(window=30, min_periods=1).mean()
    
    # Relative humidity
    if 't2m' in df_copy and 'd2m' in df_copy:
        df_copy['relative_humidity'] = 100 * np.exp((17.625 * (df_copy['d2m'] - 273.15)) / 
                                                      (243.04 + (df_copy['d2m'] - 273.15))) / \
                                        np.exp((17.625 * (df_copy['t2m'] - 273.15)) / 
                                               (243.04 + (df_copy['t2m'] - 273.15)))
        df_copy['relative_humidity'] = df_copy['relative_humidity'].clip(0, 100)
    
    # Wind direction classification
    if 'wind_dir_10m' in df_copy:
        df_copy['wind_dir_category'] = pd.cut(df_copy['wind_dir_10m'], 
                                                bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                                                labels=[0, 1, 2, 3, 4, 5, 6, 7],
                                                include_lowest=True).astype(int)
    
    return df_copy

# ============================== Part 5: Sequence Data Preparation (GPU Optimized) ==============================
def create_sequences(X, y, lookback):
    """
    Convert time series data to LSTM input format
    
    Parameters:
        X: Feature data (DataFrame)
        y: Target data (Series)
        lookback: Sequence length (lookback days)
    
    Returns:
        X_seq: 3D array [samples, lookback, features]
        y_seq: 1D array [samples]
        indices: Corresponding date indices
    """
    X_seq, y_seq, indices = [], [], []
    
    for i in range(lookback, len(X)):
        X_seq.append(X.iloc[i-lookback:i].values)
        y_seq.append(y.iloc[i])
        indices.append(X.index[i])
    
    return np.array(X_seq), np.array(y_seq), indices

def prepare_data_for_lstm(X, y, lookback, train_ratio=0.7, val_ratio=0.15):
    """
    Prepare dataset for LSTM (GPU optimized version)
    
    Returns:
        Dictionary containing DataLoaders and indices for training, validation, and test sets
    """
    # Create sequences
    X_seq, y_seq, indices = create_sequences(X, y, lookback)
    
    # Split by time order
    n_samples = len(X_seq)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    X_train = X_seq[:train_size]
    X_val = X_seq[train_size:train_size + val_size]
    X_test = X_seq[train_size + val_size:]
    
    y_train = y_seq[:train_size]
    y_val = y_seq[train_size:train_size + val_size]
    y_test = y_seq[train_size + val_size:]
    
    idx_train = indices[:train_size]
    idx_val = indices[train_size:train_size + val_size]
    idx_test = indices[train_size + val_size:]
    
    # Standardization
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Reshape to 2D for standardization
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = scaler_X.fit_transform(X_train_2d).reshape(X_train.shape)
    
    X_val_2d = X_val.reshape(-1, X_val.shape[-1])
    X_val_scaled = scaler_X.transform(X_val_2d).reshape(X_val.shape)
    
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler_X.transform(X_test_2d).reshape(X_test.shape)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled)
    
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_idx': idx_train,
        'val_idx': idx_val,
        'test_idx': idx_test,
        'scaler_y': scaler_y,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

# ============================== Part 6: LSTM + Attention Model Definition ==============================
class Attention(nn.Module):
    """Attention mechanism layer"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output: [batch, seq_len, hidden_size]
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights: [batch, seq_len, 1]
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_output, dim=1)
        # context: [batch, hidden_size]
        
        return context, attention_weights

class LSTMAttentionModel(nn.Module):
    """LSTM + Attention Model"""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMAttentionModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for analysis
        self.last_attention_weights = None
        
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        lstm_out, _ = self.lstm(x)
        # lstm_out: [batch, seq_len, hidden_size]
        
        context, attention_weights = self.attention(lstm_out)
        # context: [batch, hidden_size]
        
        # Save attention weights
        self.last_attention_weights = attention_weights.detach()
        
        out = self.dropout(context)
        out = self.fc(out)
        # out: [batch, 1]
        
        return out.squeeze()

# ============================== Part 7: Training and Evaluation Functions (Mixed Precision Optimization) ==============================
def train_model(model, train_loader, val_loader, epochs, learning_rate, patience=20, verbose=True):
    """
    Train LSTM model (GPU accelerated + Mixed precision training)
    
    Returns:
        Training history (loss curves) and best model state
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # GradScaler for mixed precision training
    scaler = GradScaler(enabled=USE_AMP)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Performance statistics
    epoch_times = []
    samples_per_sec = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_samples = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision training
            with autocast(enabled=USE_AMP):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent gradient explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * X_batch.size(0)
            train_samples += X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                
                with autocast(enabled=USE_AMP):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Performance statistics
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        samples_per_sec.append(train_samples / epoch_time)
        
        if verbose and (epoch + 1) % 10 == 0:
            avg_speed = np.mean(samples_per_sec[-10:])
            gpu_util = get_gpu_utilization()
            gpu_info = f", GPU Util: {gpu_util}%" if gpu_util is not None else ""
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"Speed: {avg_speed:.0f} samples/s{gpu_info}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at Epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Clear GPU cache
    clear_gpu_memory()
    
    avg_epoch_time = np.mean(epoch_times)
    avg_samples_per_sec = np.mean(samples_per_sec)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': len(train_losses) - patience_counter,
        'best_val_loss': best_val_loss,
        'avg_epoch_time': avg_epoch_time,
        'avg_samples_per_sec': avg_samples_per_sec
    }

def evaluate_model(model, data_loader, scaler_y, y_true):
    """Evaluate model performance"""
    model.eval()
    # GPU优化：保持在GPU上累积，最后一次性转换到CPU，减少同步次数
    predictions = []
    
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            
            with autocast(enabled=USE_AMP):
                outputs = model(X_batch)
            
            # GPU优化：保持在GPU上，避免频繁的CPU-GPU同步
            predictions.append(outputs)
    
    # GPU优化：一次性转换所有结果到CPU，减少同步开销
    predictions = torch.cat(predictions).cpu().numpy()
    
    # Inverse standardization
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    r2 = r2_score(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    mae = mean_absolute_error(y_true, predictions)
    mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100
    
    return {
        'predictions': predictions,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def extract_attention_importance(model, data_loader, feature_names):
    """
    Extract attention weights as feature importance
    
    Returns:
        Feature importance DataFrame
    """
    model.eval()
    all_attention_weights = []
    
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            
            with autocast(enabled=USE_AMP):
                _ = model(X_batch)
            
            # GPU优化：保持在GPU上累积，最后一次性转换
            # Get attention weights [batch, seq_len, 1]
            all_attention_weights.append(model.last_attention_weights)
    
    # GPU优化：一次性转换所有attention weights到CPU
    all_attention_weights = torch.cat(all_attention_weights, dim=0).cpu().numpy()
    # Shape: [total_samples, seq_len, 1]
    
    # Average over time steps to get feature importance for each sample
    # Here we calculate the average attention weights across all samples
    avg_attention = all_attention_weights.mean(axis=0).squeeze()  # [seq_len]
    
    # Since LSTM looks at the entire sequence, we assign equal attention weights to each feature
    # A more accurate method would use feature-level attention, but this is a simplified approach
    feature_importance = np.ones(len(feature_names)) * avg_attention.mean()
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance / feature_importance.sum() * 100  # Normalize to percentage
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    return importance_df

# ============================== Part 8: Data Loading and Preprocessing ==============================
print("\n" + "=" * 80)
print("Step 1: Data Loading and Preprocessing")
print("=" * 80)

df_era5 = read_all_era5()
df_pollution = read_all_pollution()


print("\nData loading check:")
print(f"  Pollution data shape: {df_pollution.shape}")
print(f"  Meteorological data shape: {df_era5.shape}")

if df_pollution.empty or df_era5.empty:
    print("\nError: Data loading failed!")
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
print(f"Sample count: {len(df_combined)}")

# ============================== Part 9: Feature Selection ==============================
print("\n" + "=" * 80)
print("Step 2: Feature Selection")
print("=" * 80)

target = 'PM2.5'
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']

numeric_features = [col for col in df_combined.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_cols]

print(f"\nNumber of selected features: {len(numeric_features)}")
print(f"Target variable: {target}")

X = df_combined[numeric_features].copy()
y = df_combined[target].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

print(f"\nPM2.5 statistics:")
print(f"  Mean: {y.mean():.2f} μg/m³")
print(f"  Std Dev: {y.std():.2f} μg/m³")
print(f"  Range: [{y.min():.2f}, {y.max():.2f}] μg/m³")

# ============================== Part 10: Prepare Data for Multiple Sequence Lengths ==============================
print("\n" + "=" * 80)
print("Step 3: Prepare Data for Multiple Sequence Lengths")
print("=" * 80)

datasets = {}
for lookback in SEQUENCE_LENGTHS:
    print(f"\nPreparing data for sequence length={lookback} days...")
    data_dict = prepare_data_for_lstm(X, y, lookback)
    datasets[lookback] = data_dict
    
    print(f"  Training set: {len(data_dict['train_idx'])} samples")
    print(f"  Validation set: {len(data_dict['val_idx'])} samples")
    print(f"  Test set: {len(data_dict['test_idx'])} samples")

print_gpu_memory_usage("Data preparation complete")

# ============================== Part 11: Train Basic Models ==============================
print("\n" + "=" * 80)
print("Step 4: Train Basic LSTM Models")
print("=" * 80)

# Basic parameters
# GPU优化：增大模型规模以提升计算密度和GPU利用率
basic_params = {
    'hidden_size': 128,  # GPU优化：从64增加到128以提升计算密度
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001
}

print(f"\nBasic model parameters: {basic_params}")

basic_models = {}
basic_histories = {}
basic_results = []

for lookback in SEQUENCE_LENGTHS:
    print(f"\n{'='*60}")
    print(f"Training basic model for sequence length={lookback} days")
    print(f"{'='*60}")
    
    data = datasets[lookback]
    input_size = X.shape[1]
    
    model = LSTMAttentionModel(
        input_size=input_size,
        hidden_size=basic_params['hidden_size'],
        num_layers=basic_params['num_layers'],
        dropout=basic_params['dropout']
    ).to(device)
    
    # GPU优化：使用torch.compile加速模型（PyTorch 2.0+）
    if USE_COMPILE and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='max-autotune')
            print("  Model compiled with torch.compile for maximum performance")
        except Exception as e:
            print(f"  Warning: torch.compile failed ({e}), continuing without compilation")
    
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
    print_gpu_memory_usage("Model loading complete")
    
    train_start = time.time()
    history = train_model(
        model=model,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        epochs=EPOCHS,
        learning_rate=basic_params['learning_rate'],
        patience=EARLY_STOP_PATIENCE,
        verbose=True
    )
    train_time = time.time() - train_start
    
    print(f"  Training time: {train_time/60:.2f} minutes")
    print(f"  Average per epoch: {history['avg_epoch_time']:.2f} seconds")
    print(f"  Training speed: {history['avg_samples_per_sec']:.0f} samples/s")
    
    basic_models[lookback] = model
    basic_histories[lookback] = history
    
    # Evaluation
    print(f"\nEvaluating basic model for sequence length={lookback} days:")
    
    train_eval = evaluate_model(model, data['train_loader'], data['scaler_y'], data['y_train'])
    val_eval = evaluate_model(model, data['val_loader'], data['scaler_y'], data['y_val'])
    test_eval = evaluate_model(model, data['test_loader'], data['scaler_y'], data['y_test'])
    
    basic_results.append({
        'Model': f'LSTM_Basic_Seq{lookback}',
        'Sequence_Length': lookback,
        'Dataset': 'Train',
        'R²': train_eval['R²'],
        'RMSE': train_eval['RMSE'],
        'MAE': train_eval['MAE'],
        'MAPE': train_eval['MAPE']
    })
    
    basic_results.append({
        'Model': f'LSTM_Basic_Seq{lookback}',
        'Sequence_Length': lookback,
        'Dataset': 'Validation',
        'R²': val_eval['R²'],
        'RMSE': val_eval['RMSE'],
        'MAE': val_eval['MAE'],
        'MAPE': val_eval['MAPE']
    })
    
    basic_results.append({
        'Model': f'LSTM_Basic_Seq{lookback}',
        'Sequence_Length': lookback,
        'Dataset': 'Test',
        'R²': test_eval['R²'],
        'RMSE': test_eval['RMSE'],
        'MAE': test_eval['MAE'],
        'MAPE': test_eval['MAPE']
    })
    
    print(f"  Training set - R²: {train_eval['R²']:.4f}, RMSE: {train_eval['RMSE']:.2f}")
    print(f"  Validation set - R²: {val_eval['R²']:.4f}, RMSE: {val_eval['RMSE']:.2f}")
    print(f"  Test set - R²: {test_eval['R²']:.4f}, RMSE: {test_eval['RMSE']:.2f}")
    print_gpu_memory_usage(f"Seq{lookback} training complete")

basic_results_df = pd.DataFrame(basic_results)
print("\nBasic model performance summary:")
print(basic_results_df.to_string(index=False))

# ============================== Part 12: Grid Search Hyperparameter Optimization (Accelerated Version) ==============================
print("\n" + "=" * 80)
print("Step 5: Grid Search Hyperparameter Optimization (GPU Accelerated Version)")
print("=" * 80)

# GPU optimization: Reduce search space but increase model size for better GPU utilization
param_grid = {
    'hidden_size': [128, 256],  # GPU优化：增大hidden_size以提升GPU利用率（从64,128改为128,256）
    'num_layers': [2, 3],  # Reduced from 3 to 2
    'dropout': [0.2],  # Fixed to most common value
    'learning_rate': [0.001, 0.005]  # Reduced from 3 to 2
}

total_combinations = int(np.prod([len(v) for v in param_grid.values()]))
print(f"Parameter grid (GPU optimized): {param_grid}")
print(f"Total {total_combinations} parameter combinations (originally 81, optimized to {total_combinations})")

best_params_per_seq = {}
optimized_models = {}
optimized_histories = {}
optimized_results = []

for lookback in SEQUENCE_LENGTHS:
    print(f"\n{'='*60}")
    print(f"Optimizing model for sequence length={lookback} days")
    print(f"{'='*60}")
    
    data = datasets[lookback]
    input_size = X.shape[1]
    
    best_val_rmse = float('inf')
    best_params = None
    best_model_state = None
    
    param_combos = list(product(*param_grid.values()))
    
    grid_search_start = time.time()
    
    if TQDM_AVAILABLE:
        iterator = tqdm(param_combos, desc=f"Grid search(Seq{lookback})", unit="combos")
    else:
        iterator = param_combos
        print(f"Starting grid search...")
    
    for i, (hidden_size, num_layers, dropout, lr) in enumerate(iterator):
        model = LSTMAttentionModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        
        # Training (using fewer epochs and patience to speed up search)
        history = train_model(
            model=model,
            train_loader=data['train_loader'],
            val_loader=data['val_loader'],
            epochs=50,  # Reduced epochs
            learning_rate=lr,
            patience=10,  # Reduced patience
            verbose=False
        )
        
        val_rmse = np.sqrt(history['best_val_loss'])
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_params = {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'learning_rate': lr
            }
            best_model_state = model.state_dict().copy()
        
        # Clear GPU memory
        del model
        clear_gpu_memory()
        
        if not TQDM_AVAILABLE and (i + 1) % 5 == 0:
            print(f"  Tested {i+1}/{total_combinations} combinations, current best RMSE: {best_val_rmse:.4f}")
    
    grid_search_time = time.time() - grid_search_start
    print(f"\nGrid search time: {grid_search_time/60:.2f} minutes")
    
    best_params_per_seq[lookback] = best_params
    
    print(f"\nBest parameters for sequence length={lookback} days:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  Best validation RMSE: {best_val_rmse:.4f}")
    
    # Retrain complete model using best parameters
    print(f"\nRetraining with best parameters...")
    model_opt = LSTMAttentionModel(
        input_size=input_size,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(device)
    
    # GPU优化：使用torch.compile加速模型（PyTorch 2.0+）
    if USE_COMPILE and hasattr(torch, 'compile'):
        try:
            model_opt = torch.compile(model_opt, mode='max-autotune')
            print("  Model compiled with torch.compile for maximum performance")
        except Exception as e:
            print(f"  Warning: torch.compile failed ({e}), continuing without compilation")
    
    history_opt = train_model(
        model=model_opt,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        epochs=EPOCHS,
        learning_rate=best_params['learning_rate'],
        patience=EARLY_STOP_PATIENCE,
        verbose=True
    )
    
    optimized_models[lookback] = model_opt
    optimized_histories[lookback] = history_opt
    
    # Evaluate optimized model
    print(f"\nEvaluating optimized model for sequence length={lookback} days:")
    
    train_eval = evaluate_model(model_opt, data['train_loader'], data['scaler_y'], data['y_train'])
    val_eval = evaluate_model(model_opt, data['val_loader'], data['scaler_y'], data['y_val'])
    test_eval = evaluate_model(model_opt, data['test_loader'], data['scaler_y'], data['y_test'])
    
    optimized_results.append({
        'Model': f'LSTM_Optimized_Seq{lookback}',
        'Sequence_Length': lookback,
        'Dataset': 'Train',
        'R²': train_eval['R²'],
        'RMSE': train_eval['RMSE'],
        'MAE': train_eval['MAE'],
        'MAPE': train_eval['MAPE']
    })
    
    optimized_results.append({
        'Model': f'LSTM_Optimized_Seq{lookback}',
        'Sequence_Length': lookback,
        'Dataset': 'Validation',
        'R²': val_eval['R²'],
        'RMSE': val_eval['RMSE'],
        'MAE': val_eval['MAE'],
        'MAPE': val_eval['MAPE']
    })
    
    optimized_results.append({
        'Model': f'LSTM_Optimized_Seq{lookback}',
        'Sequence_Length': lookback,
        'Dataset': 'Test',
        'R²': test_eval['R²'],
        'RMSE': test_eval['RMSE'],
        'MAE': test_eval['MAE'],
        'MAPE': test_eval['MAPE']
    })
    
    print(f"  Training set - R²: {train_eval['R²']:.4f}, RMSE: {train_eval['RMSE']:.2f}")
    print(f"  Validation set - R²: {val_eval['R²']:.4f}, RMSE: {val_eval['RMSE']:.2f}")
    print(f"  Test set - R²: {test_eval['R²']:.4f}, RMSE: {test_eval['RMSE']:.2f}")
    print_gpu_memory_usage(f"Seq{lookback} optimization complete")

optimized_results_df = pd.DataFrame(optimized_results)
print("\nOptimized model performance summary:")
print(optimized_results_df.to_string(index=False))

# ============================== Part 13: Model Comparison ==============================
print("\n" + "=" * 80)
print("Step 6: Model Performance Comparison")
print("=" * 80)

all_results = pd.concat([basic_results_df, optimized_results_df])
print("\nAll model performance comparison:")
print(all_results.to_string(index=False))

# Test set performance ranking
test_results = all_results[all_results['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\nTest set performance ranking:")
print(test_results.to_string(index=False))

# Find best model
best_model_info = test_results.iloc[0]
print(f"\nBest model: {best_model_info['Model']}")
print(f"  Sequence length: {best_model_info['Sequence_Length']} days")
print(f"  R² Score: {best_model_info['R²']:.4f}")
print(f"  RMSE: {best_model_info['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model_info['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model_info['MAPE']:.2f}%")

# ============================== Part 14: Attention Feature Importance Analysis ==============================
print("\n" + "=" * 80)
print("Step 7: Attention Feature Importance Analysis")
print("=" * 80)

# Extract feature importance using best model
best_seq_length = int(best_model_info['Sequence_Length'])
best_model = optimized_models[best_seq_length]
best_data = datasets[best_seq_length]

feature_importance = extract_attention_importance(
    model=best_model,
    data_loader=best_data['test_loader'],
    feature_names=numeric_features
)

print(f"\nTop 20 important features (based on Attention weights):")
print(feature_importance.head(20).to_string(index=False))

# ============================== Part 15: Visualization ==============================
print("\n" + "=" * 80)
print("Step 8: Generate Visualization Charts")
print("=" * 80)

# 15.1 Training curves
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

for idx, lookback in enumerate(SEQUENCE_LENGTHS):
    # Basic model
    row = 0
    col = idx
    ax = axes[row, col]
    
    history = basic_histories[lookback]
    ax.plot(history['train_losses'], label='Training', linewidth=2)
    ax.plot(history['val_losses'], label='Validation', linewidth=2)
    ax.axvline(x=history['best_epoch'], color='r', linestyle='--', 
               label=f'Best Epoch({history["best_epoch"]})', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (MSE)', fontsize=11)
    ax.set_title(f'Basic Model - Seq{lookback} days', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Optimized model
    row = 1
    ax = axes[row, col]
    
    history = optimized_histories[lookback]
    ax.plot(history['train_losses'], label='Training', linewidth=2)
    ax.plot(history['val_losses'], label='Validation', linewidth=2)
    ax.axvline(x=history['best_epoch'], color='r', linestyle='--',
               label=f'Best Epoch({history["best_epoch"]})', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (MSE)', fontsize=11)
    ax.set_title(f'Optimized Model - Seq{lookback} days\n(GPU Accelerated)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
print("Saved: training_curves.png")
plt.close()

# 15.2 Prediction vs Actual scatter plots
n_models = len(SEQUENCE_LENGTHS) * 2  # Basic + Optimized
n_datasets = 3  # train, val, test
fig, axes = plt.subplots(n_models, n_datasets, figsize=(18, n_models * 5))

plot_row = 0
for lookback in SEQUENCE_LENGTHS:
    data = datasets[lookback]
    
    # Basic model
    model = basic_models[lookback]
    for col_idx, (loader_name, y_true) in enumerate([
        ('train_loader', data['y_train']),
        ('val_loader', data['y_val']),
        ('test_loader', data['y_test'])
    ]):
        eval_result = evaluate_model(model, data[loader_name], data['scaler_y'], y_true)
        y_pred = eval_result['predictions']
        
        ax = axes[plot_row, col_idx]
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal prediction line')
        
        dataset_name = ['Train', 'Val', 'Test'][col_idx]
        ax.set_xlabel('Actual PM2.5 Concentration (μg/m³)', fontsize=10)
        ax.set_ylabel('Predicted PM2.5 Concentration (μg/m³)', fontsize=10)
        ax.set_title(f'Basic_Seq{lookback} - {dataset_name}\nR²={eval_result["R²"]:.4f}, RMSE={eval_result["RMSE"]:.2f}', 
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plot_row += 1
    
    # Optimized model
    model = optimized_models[lookback]
    for col_idx, (loader_name, y_true) in enumerate([
        ('train_loader', data['y_train']),
        ('val_loader', data['y_val']),
        ('test_loader', data['y_test'])
    ]):
        eval_result = evaluate_model(model, data[loader_name], data['scaler_y'], y_true)
        y_pred = eval_result['predictions']
        
        ax = axes[plot_row, col_idx]
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal prediction line')
        
        dataset_name = ['Train', 'Val', 'Test'][col_idx]
        ax.set_xlabel('Actual PM2.5 Concentration (μg/m³)', fontsize=10)
        ax.set_ylabel('Predicted PM2.5 Concentration (μg/m³)', fontsize=10)
        ax.set_title(f'Optimized_Seq{lookback} - {dataset_name}\nR²={eval_result["R²"]:.4f}, RMSE={eval_result["RMSE"]:.2f}', 
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plot_row += 1

plt.tight_layout()
plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
print("Saved: prediction_scatter.png")
plt.close()

# 15.3 Time series prediction comparison (best model)
best_seq_length = int(best_model_info['Sequence_Length'])
best_data = datasets[best_seq_length]

fig, axes = plt.subplots(3, 1, figsize=(18, 15))

for idx, lookback in enumerate(SEQUENCE_LENGTHS):
    data = datasets[lookback]
    model = optimized_models[lookback]
    
    eval_result = evaluate_model(model, data['test_loader'], data['scaler_y'], data['y_test'])
    y_pred = eval_result['predictions']
    y_true = data['y_test']
    
    plot_range = min(300, len(y_true))
    plot_idx = range(len(y_true) - plot_range, len(y_true))
    time_idx = pd.DatetimeIndex(data['test_idx'])[plot_idx]
    
    ax = axes[idx]
    ax.plot(time_idx, y_true[plot_idx], 'k-', label='Actual', linewidth=2, alpha=0.8)
    ax.plot(time_idx, y_pred[plot_idx], 'r--', label='LSTM Prediction', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
    ax.set_title(f'LSTM Seq{lookback} days - Time Series Prediction Comparison (Last {plot_range} days of test set)\nR²={eval_result["R²"]:.4f}, RMSE={eval_result["RMSE"]:.2f}', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(output_dir / 'timeseries_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: timeseries_comparison.png")
plt.close()

# 15.4 Residual analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Basic model residuals
for idx, lookback in enumerate(SEQUENCE_LENGTHS):
    data = datasets[lookback]
    model = basic_models[lookback]
    eval_result = evaluate_model(model, data['test_loader'], data['scaler_y'], data['y_test'])
    residuals = data['y_test'] - eval_result['predictions']
    
    ax = axes[0, idx]
    ax.scatter(eval_result['predictions'], residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted value (μg/m³)', fontsize=11)
    ax.set_ylabel('Residuals (μg/m³)', fontsize=11)
    ax.set_title(f'Basic_Seq{lookback} - Test\nMean residuals={residuals.mean():.2f}, Std dev={residuals.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Optimized model residuals
for idx, lookback in enumerate(SEQUENCE_LENGTHS):
    data = datasets[lookback]
    model = optimized_models[lookback]
    eval_result = evaluate_model(model, data['test_loader'], data['scaler_y'], data['y_test'])
    residuals = data['y_test'] - eval_result['predictions']
    
    ax = axes[1, idx]
    ax.scatter(eval_result['predictions'], residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted value (μg/m³)', fontsize=11)
    ax.set_ylabel('Residuals (μg/m³)', fontsize=11)
    ax.set_title(f'Optimized_Seq{lookback} - Test\nMean residuals={residuals.mean():.2f}, Std dev={residuals.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: residuals_analysis.png")
plt.close()

# 15.5 Feature importance plot
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

top_n = 20
top_features = feature_importance.head(top_n)

ax.barh(range(top_n), top_features['Importance'], color='steelblue')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['Feature'], fontsize=10)
ax.set_xlabel('Importance (%)', fontsize=12)
ax.set_title(f'Top {top_n} Important Features (Based on Attention Weights)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: feature_importance.png")
plt.close()

# 15.6 Model performance comparison bar charts
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

test_results_sorted = test_results.copy()
models_list = test_results_sorted['Model'].tolist()
x_pos = np.arange(len(models_list))

metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
for i, metric in enumerate(metrics):
    axes[i].bar(x_pos, test_results_sorted[metric], alpha=0.7, 
                edgecolor='black', linewidth=1.5)
    axes[i].set_xticks(x_pos)
    axes[i].set_xticklabels([m.replace('LSTM_', '').replace('_Seq', '\nSeq') for m in models_list], 
                            fontsize=9, rotation=0)
    axes[i].set_ylabel(metric, fontsize=12)
    
    if metric == 'R²':
        axes[i].set_title(f'{metric} Comparison\n(Higher is better)', fontsize=12, fontweight='bold')
    else:
        axes[i].set_title(f'{metric} Comparison\n(Lower is better)', fontsize=12, fontweight='bold')
    
    axes[i].grid(True, alpha=0.3, axis='y')
    
    # Display values
    for j, v in enumerate(test_results_sorted[metric]):
        if metric == 'MAPE':
            axes[i].text(j, v, f'{v:.1f}%', ha='center', va='bottom', 
                         fontsize=8, fontweight='bold')
        else:
            axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom', 
                         fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison.png")
plt.close()

# 15.7 Error distribution histograms
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, lookback in enumerate(SEQUENCE_LENGTHS):
    data = datasets[lookback]
    
    # Basic model
    model = basic_models[lookback]
    eval_result = evaluate_model(model, data['test_loader'], data['scaler_y'], data['y_test'])
    errors = data['y_test'] - eval_result['predictions']
    
    ax = axes[0, idx]
    ax.hist(errors, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero error')
    ax.set_xlabel('Prediction error (μg/m³)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Basic_Seq{lookback} - Error Distribution\nMean={errors.mean():.2f}, Std dev={errors.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Optimized model
    model = optimized_models[lookback]
    eval_result = evaluate_model(model, data['test_loader'], data['scaler_y'], data['y_test'])
    errors = data['y_test'] - eval_result['predictions']
    
    ax = axes[1, idx]
    ax.hist(errors, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero error')
    ax.set_xlabel('Prediction error (μg/m³)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Optimized_Seq{lookback} - Error Distribution\nMean={errors.mean():.2f}, Std dev={errors.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: error_distribution.png")
plt.close()

# ============================== Part 16: Save Results ==============================
print("\n" + "=" * 80)
print("Step 9: Save Results")
print("=" * 80)

# Save model performance
all_results.to_csv(output_dir / 'model_performance.csv', index=False, encoding='utf-8-sig')
print("Saved: model_performance.csv")

# Save feature importance
feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False, encoding='utf-8-sig')
print("Saved: feature_importance.csv")

# Save best parameters
best_params_list = []
for lookback, params in best_params_per_seq.items():
    params_copy = params.copy()
    params_copy['sequence_length'] = lookback
    best_params_list.append(params_copy)

best_params_df = pd.DataFrame(best_params_list)
best_params_df.to_csv(output_dir / 'best_parameters.csv', index=False, encoding='utf-8-sig')
print("Saved: best_parameters.csv")

# Save prediction results (best model)
best_seq_length = int(best_model_info['Sequence_Length'])
best_data = datasets[best_seq_length]
best_model = optimized_models[best_seq_length]

test_eval = evaluate_model(best_model, best_data['test_loader'], best_data['scaler_y'], best_data['y_test'])

predictions_df = pd.DataFrame({
    'Date': pd.DatetimeIndex(best_data['test_idx']),
    'Actual': best_data['y_test'],
    'Prediction': test_eval['predictions'],
    'Error': best_data['y_test'] - test_eval['predictions']
})
predictions_df.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
print("Saved: predictions.csv")

# Save models
for lookback in SEQUENCE_LENGTHS:
    model_path = model_dir / f'lstm_gpu_optimized_seq{lookback}.pth'
    torch.save(optimized_models[lookback].state_dict(), model_path)
    print(f"Saved: lstm_gpu_optimized_seq{lookback}.pth")

# ============================== Part 17: Summary Report ==============================
print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)

print("\nGenerated files:")
print("\nCSV Files:")
print("  - model_performance.csv       Model performance comparison")
print("  - feature_importance.csv      Feature importance (Attention weights)")
print("  - best_parameters.csv         Best parameters for each sequence length")
print("  - predictions.csv             Prediction results (best model)")

print("\nChart Files:")
print("  - training_curves.png         Training process curves")
print("  - prediction_scatter.png      Prediction vs Actual scatter plots")
print("  - timeseries_comparison.png   Time series comparison")
print("  - residuals_analysis.png      Residual analysis")
print("  - feature_importance.png      Attention feature importance plot")
print("  - model_comparison.png        Model performance comparison")
print("  - error_distribution.png      Error distribution")

print("\nModel Files:")
for lookback in SEQUENCE_LENGTHS:
    print(f"  - lstm_gpu_optimized_seq{lookback}.pth      LSTM model (Seq{lookback} days, GPU optimized)")

print(f"\nBest model: {best_model_info['Model']}")
print(f"  Sequence length: {best_model_info['Sequence_Length']} days")
print(f"  R² Score: {best_model_info['R²']:.4f}")
print(f"  RMSE: {best_model_info['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model_info['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model_info['MAPE']:.2f}%")

print("\nTop 5 important features (based on Attention weights):")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.2f}%")

print("\nSequence length comparison:")
for lookback in SEQUENCE_LENGTHS:
    seq_results = test_results[test_results['Sequence_Length'] == lookback].iloc[0]
    print(f"  Seq{lookback} days: R²={seq_results['R²']:.4f}, RMSE={seq_results['RMSE']:.2f}")

# GPU performance summary
if torch.cuda.is_available():
    print("\nGPU Performance Summary:")
    print(f"  Batch size: {BATCH_SIZE} (optimized, originally 32)")
    print(f"  Mixed precision training: Enabled")
    print(f"  Grid search space: {total_combinations} combinations (optimized, originally 81)")
    print(f"  Final GPU memory usage:")
    print_gpu_memory_usage()

print("\n" + "=" * 80)
print("LSTM + Attention PM2.5 Concentration Prediction Complete! (GPU Accelerated Version)")
print("=" * 80)

