import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import pickle
from pathlib import Path
import glob
import multiprocessing
from multiprocessing import Pool, Manager
import calendar

import xarray as xr
from netCDF4 import Dataset as NetCDFDataset

warnings.filterwarnings('ignore')

# PyTorch related
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Get CPU core count
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

# Set multiprocessing start method (for compatibility)
# On Linux, 'fork' is default and works well
# On Windows, 'spawn' is required
if hasattr(multiprocessing, 'set_start_method'):
    try:
        # Try to use fork on Linux (faster), fallback to spawn if needed
        if os.name != 'nt':  # Not Windows
            multiprocessing.set_start_method('fork', force=True)
        else:  # Windows
            multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set, ignore
        pass

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

# Grid search for hyperparameter optimization
from itertools import product

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
pollution_all_path = '/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)'
pollution_extra_path = '/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)'
era5_path = '/root/autodl-tmp/ERA5-Beijing-NC'

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

# GPU优化配置 - RTX 5090 (32GB)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = DEVICE.type == 'cuda'
NON_BLOCKING = PIN_MEMORY

# 数据加载优化
DATALOADER_WORKERS = min(16, MAX_WORKERS) if DEVICE.type == 'cuda' else 0  # 增加workers
PERSISTENT_WORKERS = DATALOADER_WORKERS > 0  # 保持workers存活，减少重启开销
PREFETCH_FACTOR = 4 if DATALOADER_WORKERS > 0 else 2  # 预取更多批次

# 混合精度训练（AMP）- 显著提升训练速度并减少显存占用
USE_AMP = True if DEVICE.type == 'cuda' else False

# 动态batch size - 根据GPU显存自动调整
def get_optimal_batch_size(model_class, window_size, num_features, device, 
                           min_batch=64, max_batch=512, step=32):
    """
    自动确定最优batch size，充分利用GPU显存
    """
    if device.type != 'cuda':
        return 32
    
    print(f"\n正在测试最优batch size (范围: {min_batch}-{max_batch})...")
    torch.cuda.empty_cache()
    
    # 创建测试模型
    test_model = model_class(
        window_size=window_size,
        num_features=num_features,
        num_conv_layers=3,
        base_filters=32,
        kernel_size=3,
        dropout_rate=0.3
    ).to(device)
    test_model.train()
    
    # 创建测试数据
    test_X = torch.randn(1, window_size, num_features, dtype=torch.float32).to(device)
    test_y = torch.randn(1, dtype=torch.float32).to(device)
    
    optimal_batch = min_batch
    current_batch = min_batch
    
    # 创建临时optimizer用于测试（需要用于scaler.step）
    test_optimizer = optim.Adam(test_model.parameters(), lr=0.001)
    
    # 使用混合精度测试
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    
    while current_batch <= max_batch:
        try:
            torch.cuda.empty_cache()
            test_model.zero_grad()
            
            # 测试当前batch size
            batch_X = test_X.repeat(current_batch, 1, 1)
            batch_y = test_y.repeat(current_batch)
            
            # 前向传播和反向传播
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    y_pred = test_model(batch_X)
                    loss = nn.MSELoss()(y_pred, batch_y)
                scaler.scale(loss).backward()
                scaler.step(test_optimizer)  # 必须先调用step
                scaler.update()  # 然后才能update
            else:
                y_pred = test_model(batch_X)
                loss = nn.MSELoss()(y_pred, batch_y)
                loss.backward()
                test_optimizer.step()
            
            # 如果成功，更新最优batch
            optimal_batch = current_batch
            memory_used = torch.cuda.memory_allocated(device) / 1024**3
            print(f"  Batch size {current_batch:3d}: ✓ 通过 (显存: {memory_used:.2f} GB)")
            
            # 增加batch size
            current_batch += step
            
            # 清理
            del batch_X, batch_y, y_pred, loss
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Batch size {current_batch:3d}: ✗ 显存不足")
                torch.cuda.empty_cache()
                # 清理当前失败的batch
                test_model.zero_grad()
                if scaler:
                    scaler.update()  # 确保scaler状态正确
                break
            else:
                # 清理并重新抛出异常
                test_model.zero_grad()
                if scaler:
                    scaler.update()
                raise e
    
    # 使用90%的最大可用batch作为安全值
    optimal_batch = int(optimal_batch * 0.9)
    if optimal_batch < min_batch:
        optimal_batch = min_batch
    
    # 清理资源
    del test_model, test_X, test_y, test_optimizer
    if scaler:
        del scaler
    torch.cuda.empty_cache()
    
    print(f"✓ 最优batch size: {optimal_batch}")
    return optimal_batch

# 初始batch size，将在模型创建后自动调整
BATCH_SIZE = 128  # 初始值，会根据GPU显存自动优化

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # 允许非确定性操作以提升性能
    # 启用TensorFloat-32 (TF32) 加速（RTX 30系列及以上）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"Data time range: {start_date.date()} to {end_date.date()}")
print(f"Target variable: PM2.5 concentration")
print(f"Time window size: {WINDOW_SIZE} days")
print(f"Initial batch size: {BATCH_SIZE} (将自动优化)")
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability(0)
    print(f"CUDA capability: {capability[0]}.{capability[1]}")
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU memory: {total_memory:.2f} GB")
print(f"Output directory: {output_dir}")
print(f"Model save directory: {model_dir}")
print(f"CPU cores: {CPU_COUNT}, parallel workers: {MAX_WORKERS}")
print(f"Dataloader workers: {DATALOADER_WORKERS}, pin_memory: {PIN_MEMORY}")
print(f"Persistent workers: {PERSISTENT_WORKERS}, prefetch_factor: {PREFETCH_FACTOR}")
print(f"Mixed precision training (AMP): {'Enabled' if USE_AMP else 'Disabled'}")
if DEVICE.type == 'cuda':
    print(f"TF32 acceleration: Enabled")

# ============================== Part 2: Data Loading Functions ==============================
def daterange(start, end):
    """Generate date sequence"""
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def build_file_index(base_path, prefix):
    """
    Build file index for pollution data to avoid repeated directory traversal.
    Returns a dictionary mapping date_str to file_path.
    """
    file_index = {}
    print(f"  Building index for {prefix} files in {base_path}...")
    count = 0
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename.startswith(prefix) and filename.endswith('.csv'):
                # Extract date from filename: prefix_YYYYMMDD.csv
                try:
                    date_str = filename.replace(f"{prefix}_", "").replace(".csv", "")
                    if len(date_str) == 8 and date_str.isdigit():
                        file_path = os.path.join(root, filename)
                        file_index[date_str] = file_path
                        count += 1
                except Exception:
                    continue
    print(f"  Found {count} files for {prefix}")
    return file_index

def read_pollution_day(args):
    """
    Read single day pollution data (multiprocessing compatible).
    Args: (date, file_index_all, file_index_extra, pollution_all_path, pollution_extra_path, pollutants)
    """
    date, file_index_all, file_index_extra, pollution_all_path, pollution_extra_path, pollutants = args
    
    date_str = date.strftime('%Y%m%d')
    
    # Use index to find files directly
    all_file = file_index_all.get(date_str)
    extra_file = file_index_extra.get(date_str)
    
    if not all_file or not extra_file:
        return None
    
    # Verify files exist
    if not os.path.exists(all_file) or not os.path.exists(extra_file):
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
    """Read all pollution data using multiprocessing"""
    print("\nLoading pollution data...")
    print(f"Using {MAX_WORKERS} parallel workers")
    
    # Build file indices first (only once)
    print("Building file indices...")
    file_index_all = build_file_index(pollution_all_path, 'beijing_all')
    file_index_extra = build_file_index(pollution_extra_path, 'beijing_extra')
    
    dates = list(daterange(start_date, end_date))
    
    # Prepare arguments for multiprocessing
    args_list = [
        (date, file_index_all, file_index_extra, pollution_all_path, pollution_extra_path, pollutants)
        for date in dates
    ]
    
    pollution_dfs = []
    
    # Use multiprocessing Pool
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
            for i, result in enumerate(results, 1):
                if i % 500 == 0 or i == len(results):
                    print(f"  Processed {i}/{len(results)} days ({i/len(results)*100:.1f}%)")
        
        # Collect valid results
        for result in results:
            if result is not None:
                pollution_dfs.append(result)
    
    if pollution_dfs:
        print(f"  Successfully read {len(pollution_dfs)}/{len(dates)} days of data")
        print("  Merging data...")
        df_poll_all = pd.concat(pollution_dfs)
        df_poll_all.ffill(inplace=True)
        df_poll_all.fillna(df_poll_all.mean(), inplace=True)
        print(f"Pollution data loading complete, shape: {df_poll_all.shape}")
        return df_poll_all
    return pd.DataFrame()

def read_single_era5_file(args):
    """
    Read a single ERA5 NetCDF file and extract all data variables (multiprocessing compatible).
    Each file may contain only one variable.
    
    Args:
        args: tuple of (file_path, beijing_lat_min, beijing_lat_max, beijing_lon_min, beijing_lon_max)
    
    Returns:
        dict: {variable_name: xr.Dataset} or None if failed
    """
    file_path, beijing_lat_min, beijing_lat_max, beijing_lon_min, beijing_lon_max = args
    
    try:
        # Use context manager to ensure file is properly closed
        with xr.open_dataset(file_path, engine="netcdf4", decode_times=True) as ds:
            # Rename coordinates for consistency
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
            
            # Decode CF conventions
            try:
                ds = xr.decode_cf(ds)
            except Exception:
                pass
            
            # Drop coordinate variables that are not needed
            drop_vars = []
            for coord in ("expver", "surface"):
                if coord in ds:
                    drop_vars.append(coord)
            if drop_vars:
                ds = ds.drop_vars(drop_vars)
            
            # Handle ensemble dimension
            if "number" in ds.dims:
                ds = ds.mean(dim="number", skipna=True)
            
            # Check if time coordinate exists
            if "time" not in ds.coords:
                return None
            
            # Get all data variables (exclude coordinates)
            data_vars = [v for v in ds.data_vars if v not in drop_vars]
            if not data_vars:
                return None
            
            # Sort by time
            ds = ds.sortby('time')
            
            # Spatial subsetting for Beijing region
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
                    # Average over spatial dimensions
                    if 'latitude' in ds.dims and 'longitude' in ds.dims:
                        ds = ds.mean(dim=['latitude', 'longitude'], skipna=True)
            
            # Resample to daily
            ds_daily = ds.resample(time='1D').mean(keep_attrs=False)
            ds_daily = ds_daily.dropna('time', how='all')
            
            if ds_daily.sizes.get('time', 0) == 0:
                return None
            
            # Load data into memory
            ds_daily = ds_daily.load()
            
            # Return dictionary with variable name as key
            result = {}
            for var in data_vars:
                if var in ds_daily.data_vars:
                    # Create a dataset with just this variable
                    var_ds = ds_daily[[var]]
                    result[var] = var_ds
            
            return result if result else None
            
    except Exception as exc:
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

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=DATALOADER_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else None,
    drop_last=False  # 保留最后一个不完整的batch
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
    
    # Create GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    print(f"\n开始训练 {num_epochs} 个epoch...")
    if use_amp:
        print("  混合精度训练 (AMP): 已启用")
    
    import time
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
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n早停触发于 epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"训练完成! 最佳模型在 epoch {best_epoch}, 验证损失: {best_val_loss:.4f}")
    print(f"总训练时间: {total_time:.2f}s ({total_time/60:.2f} 分钟)")
    
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

# 自动优化batch size（仅在GPU上）
if DEVICE.type == 'cuda':
    print("\n" + "=" * 80)
    print("自动优化Batch Size以充分利用GPU显存")
    print("=" * 80)
    optimal_batch_size = get_optimal_batch_size(
        PM25CNN2D, WINDOW_SIZE, num_features, DEVICE
    )
    
    if optimal_batch_size != BATCH_SIZE:
        print(f"\n更新batch size: {BATCH_SIZE} -> {optimal_batch_size}")
        BATCH_SIZE = optimal_batch_size
        
        # 重新创建DataLoader with optimized batch size
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
        print(f"DataLoader已更新，新的batch size: {BATCH_SIZE}")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  验证批次数: {len(val_loader)}")
        print(f"  测试批次数: {len(test_loader)}")

# PyTorch 2.0+ 编译优化（可选，进一步提升性能）
USE_COMPILE = False  # 设置为True以启用torch.compile（需要PyTorch 2.0+）
if USE_COMPILE and hasattr(torch, 'compile'):
    print("\n启用PyTorch编译优化 (torch.compile)...")
    model_basic = torch.compile(model_basic, mode='reduce-overhead')
    print("✓ 模型编译完成")

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

# 定义网格搜索参数空间
param_grid = {
    'num_conv_layers': [2, 3, 4],
    'base_filters': [32, 64],
    'kernel_size': [3, 5],
    'learning_rate': [0.0005, 0.001, 0.005],
    'dropout_rate': [0.2, 0.3, 0.4]
}

# 计算总组合数
total_combinations = int(np.prod([len(v) for v in param_grid.values()]))
print(f"\n网格搜索配置:")
print(f"  总参数组合数: {total_combinations}")
print(f"\n参数搜索空间:")
for key, values in param_grid.items():
    print(f"  {key}: {values}")

# 存储所有尝试的结果
grid_search_results = []
best_val_loss_grid = float('inf')
best_params = {}

# 清理GPU缓存
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()

import time
grid_search_start_time = time.time()

print(f"\n开始网格搜索...")
print("=" * 80)

# 使用tqdm显示进度（如果可用）
if TQDM_AVAILABLE:
    param_combinations = list(product(*param_grid.values()))
    iterator = tqdm(enumerate(param_combinations, 1), total=total_combinations, 
                    desc="网格搜索进度", unit="组合")
else:
    param_combinations = list(product(*param_grid.values()))
    iterator = enumerate(param_combinations, 1)

for i, combo in iterator:
    params_test = dict(zip(param_grid.keys(), combo))
    
    if not TQDM_AVAILABLE:
        print(f"\n[{i}/{total_combinations}] 测试参数组合:")
        for key, value in params_test.items():
            print(f"  {key}: {value}")
    
    try:
        # 清理GPU缓存
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 创建模型
        model_temp = PM25CNN2D(
            window_size=WINDOW_SIZE,
            num_features=num_features,
            num_conv_layers=params_test['num_conv_layers'],
            base_filters=params_test['base_filters'],
            kernel_size=params_test['kernel_size'],
            dropout_rate=params_test['dropout_rate']
        ).to(DEVICE)
        
        # 创建优化器
        optimizer_temp = optim.Adam(model_temp.parameters(), lr=params_test['learning_rate'])
        
        # 训练模型（使用较少的epochs进行快速搜索）
        train_start_time = time.time()
        _, _, best_epoch_temp = train_model(
            model_temp, train_loader, val_loader, criterion, optimizer_temp,
            num_epochs=100, device=DEVICE, patience=15, verbose=False, use_amp=USE_AMP
        )
        train_time = time.time() - train_start_time
        
        # 评估模型
        val_loss, _, _ = validate(model_temp, val_loader, criterion, DEVICE, USE_AMP)
        
        # 记录结果
        result_entry = params_test.copy()
        result_entry['val_loss'] = val_loss
        result_entry['best_epoch'] = best_epoch_temp
        result_entry['train_time'] = train_time
        grid_search_results.append(result_entry)
        
        # 更新最佳参数
        if val_loss < best_val_loss_grid:
            best_val_loss_grid = val_loss
            best_params = params_test.copy()
            if not TQDM_AVAILABLE:
                print(f"  ✓ 新的最佳验证损失: {val_loss:.4f} (Epoch {best_epoch_temp}, 训练时间: {train_time:.1f}s)")
        else:
            if not TQDM_AVAILABLE:
                print(f"  验证损失: {val_loss:.4f} (Epoch {best_epoch_temp}, 训练时间: {train_time:.1f}s)")
        
        # 清理模型
        del model_temp, optimizer_temp
        
    except Exception as e:
        print(f"\n  ✗ 参数组合 {i} 训练失败: {type(e).__name__}: {e}")
        # 记录失败的结果
        result_entry = params_test.copy()
        result_entry['val_loss'] = float('inf')
        result_entry['best_epoch'] = 0
        result_entry['train_time'] = 0
        result_entry['error'] = str(e)
        grid_search_results.append(result_entry)
        continue

grid_search_total_time = time.time() - grid_search_start_time

# 保存网格搜索结果
grid_search_df = pd.DataFrame(grid_search_results)
grid_search_df = grid_search_df.sort_values('val_loss', ascending=True)
grid_search_df.to_csv(output_dir / 'grid_search_results.csv', index=False, encoding='utf-8-sig')
print(f"\n✓ 网格搜索结果已保存: grid_search_results.csv")

# 显示最佳参数
print("\n" + "=" * 80)
print("网格搜索完成!")
print("=" * 80)
print(f"\n最佳参数组合:")
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"  最佳验证损失: {best_val_loss_grid:.4f}")
print(f"  总搜索时间: {grid_search_total_time:.2f}s ({grid_search_total_time/60:.2f} 分钟)")
print(f"  平均每个组合训练时间: {grid_search_total_time/total_combinations:.2f}s")

# 显示Top 5最佳结果
print(f"\nTop 5 最佳参数组合:")
top5_results = grid_search_df.head(5)
for idx, (_, row) in enumerate(top5_results.iterrows(), 1):
    print(f"\n  排名 {idx}:")
    print(f"    验证损失: {row['val_loss']:.4f}")
    print(f"    最佳Epoch: {row['best_epoch']}")
    print(f"    训练时间: {row['train_time']:.1f}s")
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
print("  - grid_search_results.csv     Grid search results (all combinations)")

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
