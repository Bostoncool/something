import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import glob
import multiprocessing
from pathlib import Path
import calendar
import xarray as xr
from netCDF4 import Dataset
warnings.filterwarnings('ignore')

# Get CPU core count
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)  # Reserve 1 core for system

# Try to import tqdm progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm is not installed, progress display will use simplified version.")
    print("      You can use 'pip install tqdm' to install for better progress bar display.")

# Machine learning libraries
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# GAM library
try:
    from pygam import LinearGAM, s, f
    GAM_AVAILABLE = True
except ImportError:
    print("Warning: pygam is not installed, GAM model will be skipped. Please use 'pip install pygam' to install.")
    GAM_AVAILABLE = False

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# Set random seed
np.random.seed(42)

print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - Linear Baseline Model")
print("=" * 80)

# ============================== Part1: Define paths and parameters ==============================
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

# Define date range
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

# Beijing geographic range
beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

# Define pollutants
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# Define ERA5 variables (select important variables based on correlation analysis results)
# Complete variable list, will be filtered later
era5_vars = [
    'd2m', 't2m', 'u10', 'v10', 'u100', 'v100',  # Temperature, Wind speed
    'blh', 'sp', 'tcwv',  # Boundary layer height, Pressure, Water vapor
    'tp', 'avg_tprate',  # Precipitation
    'tisr', 'str',  # Radiation
    'cvh', 'cvl',  # Cloud cover
    'mn2t', 'sd', 'lsm'  # Other
]

print(f"Data time range: {start_date.date()} to {end_date.date()}")
print(f"Target variable: PM2.5 concentration")
print(f"Number of meteorological features: {len(era5_vars)}")
print(f"Output directory: {output_dir}")
print(f"Model save directory: {model_dir}")
print(f"CPU cores: {CPU_COUNT}, Parallel worker threads: {MAX_WORKERS}")

# ============================== Part2: Data Loading Functions ==============================
def daterange(start, end):
    """Generate date sequence"""
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def build_file_path_dict(base_path, prefix):
    """构建文件路径字典，将日期字符串映射到文件路径，时间复杂度O(1)"""
    file_dict = {}
    filename_pattern = f"{prefix}_"
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename.startswith(filename_pattern) and filename.endswith('.csv'):
                # 提取日期字符串 (格式: prefix_YYYYMMDD.csv)
                date_str = filename[len(filename_pattern):-4]  # 去掉前缀和后缀
                if len(date_str) == 8 and date_str.isdigit():  # 确保是8位数字日期
                    file_dict[date_str] = os.path.join(root, filename)
    return file_dict

def read_pollution_day(args):
    """Read single day pollution data
    Args: tuple of (date, all_file_dict, extra_file_dict, pollutants)
    """
    date, all_file_dict, extra_file_dict, pollutants = args
    date_str = date.strftime('%Y%m%d')
    all_file = all_file_dict.get(date_str)
    extra_file = extra_file_dict.get(date_str)
    
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
        
        # Aggregate by date and type (average across all stations)
        df_daily = df_poll.groupby(['date', 'type'])['value'].mean().reset_index()
        
        # Convert to wide format
        df_daily = df_daily.pivot(index='date', columns='type', values='value')
        
        # Convert index to datetime format
        df_daily.index = pd.to_datetime(df_daily.index, format='%Y%m%d', errors='coerce')
        
        # Keep only needed pollutants
        df_daily = df_daily[[col for col in pollutants if col in df_daily.columns]]
        
        return df_daily
    except Exception as e:
        return None

def read_all_pollution():
    """Read all pollution data in parallel using multiprocessing"""
    print("\nLoading pollution data...")
    print(f"Building file path dictionary (O(1) lookup)...")
    
    # 预先构建文件路径字典，避免每次遍历目录
    all_file_dict = build_file_path_dict(pollution_all_path, 'beijing_all')
    extra_file_dict = build_file_path_dict(pollution_extra_path, 'beijing_extra')
    
    print(f"  Found {len(all_file_dict)} files in all directory")
    print(f"  Found {len(extra_file_dict)} files in extra directory")
    print(f"Using {MAX_WORKERS} parallel worker processes")
    
    dates = list(daterange(start_date, end_date))
    pollution_dfs = []
    
    # 准备参数列表
    task_args = [(date, all_file_dict, extra_file_dict, pollutants) for date in dates]
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, args): args[0] for args in task_args}
        
        if TQDM_AVAILABLE:
            # Using tqdm progress bar
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Loading pollution data", unit="days"):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
        else:
            # Simplified progress display
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
                if i % 500 == 0 or i == len(futures):
                    print(f"  Processed {i}/{len(futures)} days ({i/len(futures)*100:.1f}%)")
    
    if pollution_dfs:
        print(f"  Successfully loaded {len(pollution_dfs)}/{len(dates)} days of data")
        print("  Merging data...")
        df_poll_all = pd.concat(pollution_dfs)
        # Forward fill then mean fill
        df_poll_all.ffill(inplace=True)
        df_poll_all.fillna(df_poll_all.mean(), inplace=True)
        print(f"Pollution data loading complete, shape: {df_poll_all.shape}")
        return df_poll_all
    return pd.DataFrame()

def read_era5_month(args):
    """读取指定年月的 ERA5 NetCDF 数据，并聚合为日尺度
    Args: tuple of (year, month, era5_path, era5_vars, beijing_lats, beijing_lons)
    """
    year, month, era5_path, era5_vars, beijing_lats, beijing_lons = args
    month_str = f"{year}{month:02d}"
    all_files = glob.glob(os.path.join(era5_path, "**", f"*{month_str}*.nc"), recursive=True)
    fallback_used = False
    if not all_files:
        all_files = glob.glob(os.path.join(era5_path, "**", "*.nc"), recursive=True)
        fallback_used = True
        if not all_files:
            return None

    monthly_datasets = []
    start_day = 1
    end_day = calendar.monthrange(year, month)[1]
    month_start = pd.to_datetime(f"{year}-{month:02d}-{start_day:02d}")
    month_end = pd.to_datetime(f"{year}-{month:02d}-{end_day:02d}") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    for file_path in all_files:
        try:
            with Dataset(file_path, mode='r') as nc_file:
                available_vars = [v for v in era5_vars if v in nc_file.variables]
            if not available_vars:
                print(f"[WARN] {os.path.basename(file_path)} 不含目标变量，跳过")
                continue

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

                drop_vars = [coord for coord in ("expver", "surface") if coord in ds]
                if drop_vars:
                    ds = ds.drop_vars(drop_vars)

                if "number" in ds.dims:
                    ds = ds.mean(dim="number", skipna=True)

                ds_subset = ds[available_vars]
                if "time" not in ds_subset.coords:
                    print(f"[WARN] {os.path.basename(file_path)} 缺少时间坐标，跳过")
                    continue

                ds_subset = ds_subset.sortby("time")
                if fallback_used:
                    try:
                        ds_subset = ds_subset.sel(time=slice(month_start, month_end))
                    except Exception as exc:
                        print(f"[WARN] {os.path.basename(file_path)} 时间筛选失败：{exc}")
                        continue
                    if ds_subset.sizes.get("time", 0) == 0:
                        continue

                if "latitude" in ds_subset.coords and "longitude" in ds_subset.coords:
                    lat_values = ds_subset["latitude"]
                    if lat_values[0] > lat_values[-1]:
                        lat_slice = slice(beijing_lats.max(), beijing_lats.min())
                    else:
                        lat_slice = slice(beijing_lats.min(), beijing_lats.max())
                    ds_subset = ds_subset.sel(latitude=lat_slice, longitude=slice(beijing_lons.min(), beijing_lons.max()))
                    if "latitude" in ds_subset.dims and "longitude" in ds_subset.dims:
                        ds_subset = ds_subset.mean(dim=["latitude", "longitude"], skipna=True)

                ds_daily = ds_subset.resample(time="1D").mean(keep_attrs=False)
                ds_daily = ds_daily.dropna("time", how="all")
                if ds_daily.sizes.get("time", 0) == 0:
                    continue

                monthly_datasets.append(ds_daily.load())
                print(f"  [+] {os.path.basename(file_path)} -> {year}-{month:02d}, 天数 {ds_daily.sizes.get('time', 0)}, 变量 {len(ds_daily.data_vars)}")
        except Exception as exc:
            print(f"[ERROR] 读取 {os.path.basename(file_path)} 失败：{type(exc).__name__}: {exc}")
            continue

    if not monthly_datasets:
        return None

    merged_ds = xr.merge(monthly_datasets, compat="override", join="outer")
    df_month = merged_ds.to_dataframe()
    df_month.index = pd.to_datetime(df_month.index)
    df_month = df_month.groupby(df_month.index).mean()
    if df_month.empty:
        return None
    print(f"  Successfully loaded {year}-{month:02d}, 天数 {len(df_month)}, 变量 {len(df_month.columns)}")
    return df_month

def read_all_era5():
    """Read all ERA5 data in parallel using multiprocessing"""
    print("\nLoading meteorological data...")
    print(f"Using {MAX_WORKERS} parallel worker processes")
    print(f"Meteorological data directory: {era5_path}")
    print(f"Check if directory exists: {os.path.exists(era5_path)}")
    
    if os.path.exists(era5_path):
        all_nc = glob.glob(os.path.join(era5_path, "**", "*.nc"), recursive=True)
        print(f"Found {len(all_nc)} NetCDF files")
        if all_nc:
            print(f"Sample files: {[os.path.basename(f) for f in all_nc[:5]]}")
    
    era5_dfs = []
    years = range(2015, 2025)
    months = range(1, 13)
    
    # Prepare all tasks with necessary parameters
    month_tasks = [(year, month) for year in years for month in months 
                   if not (year == 2024 and month > 12)]
    total_months = len(month_tasks)
    print(f"Attempting to load {total_months} months of data...")
    
    # Prepare arguments for each task
    task_args = [(year, month, era5_path, era5_vars, beijing_lats, beijing_lons) 
                 for year, month in month_tasks]
    
    # Using parallel processes to load ERA5 data (multiprocessing for thread-safe netcdf4)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_era5_month, args): args[:2] 
                  for args in task_args}
        
        successful_reads = 0
        if TQDM_AVAILABLE:
            # Using tqdm progress bar
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Loading meteorological data", unit="month"):
                result = future.result()
                if result is not None and not result.empty:
                    era5_dfs.append(result)
                    successful_reads += 1
        else:
            # Simplified progress display
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None and not result.empty:
                    era5_dfs.append(result)
                    successful_reads += 1
                if i % 20 == 0 or i == len(futures):
                    print(f"  Progress: {i}/{len(futures)} months (Success: {successful_reads}, {i/len(futures)*100:.1f}%)")
        
        print(f"  Total successfully loaded: {successful_reads}/{len(futures)} months")
    
    if era5_dfs:
        print("\nMerging meteorological data...")
        df_era5_all = pd.concat(era5_dfs, axis=0)
        
        # Deduplicate (may have duplicate dates)
        print("  Deduplicating...")
        df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep='first')]
        
        # Sort
        print("  Sorting...")
        df_era5_all.sort_index(inplace=True)
        
        print(f"Shape after merging: {df_era5_all.shape}")
        print(f"Time range: {df_era5_all.index.min()} to {df_era5_all.index.max()}")
        print(f"Available variables: {list(df_era5_all.columns[:10])}..." if len(df_era5_all.columns) > 10 else f"Available variables: {list(df_era5_all.columns)}")
        
        # Fill missing values
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
        print("Possible reasons:")
        print("1. File naming format mismatch (Expected format: *YYYYMM*.nc)")
        print("2. Incorrect file content format (Missing time coordinate)")
        print("3. Incorrect file path")
        return pd.DataFrame()

# ============================== Part3: Data Loading and Preprocessing ==============================
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("Step 1: Data Loading and Preprocessing")
    print("=" * 80)

    df_pollution = read_all_pollution()
    df_era5 = read_all_era5()

    # Check data loading status
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
        print("   Possible reason: Pollution data and meteorological data date indices have no intersection.")
        print(f"   Pollution data has {len(df_pollution)} rows")
        print(f"   Meteorological data has {len(df_era5)} rows")
        print(f"   After merging has {len(df_combined)} rows")
        import sys
        sys.exit(1)

    # Calculate wind speed magnitude (from wind components)
    if 'u10' in df_combined and 'v10' in df_combined:
        df_combined['wind_speed_10m'] = np.sqrt(df_combined['u10']**2 + df_combined['v10']**2)
        df_combined['wind_dir_10m'] = np.arctan2(df_combined['v10'], df_combined['u10']) * 180 / np.pi
    
    if 'u100' in df_combined and 'v100' in df_combined:
        df_combined['wind_speed_100m'] = np.sqrt(df_combined['u100']**2 + df_combined['v100']**2)

    # Add time features
    df_combined['month'] = df_combined.index.month
    df_combined['season'] = df_combined['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )
    df_combined['day_of_year'] = df_combined.index.dayofyear

    # Clean data
    df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_combined.dropna(inplace=True)

    print(f"\nData shape after merging: {df_combined.shape}")
    print(f"Time range: {df_combined.index.min().date()} to {df_combined.index.max().date()}")
    print(f"Number of samples: {len(df_combined)}")

    # ============================== Part4: Feature Selection ==============================
    print("\n" + "=" * 80)
    print("Step 2: Feature Selection - Based on Correlation Analysis Results")
    print("=" * 80)

    # Define target variable
    target = 'PM2.5'

    # Select features based on correlation analysis results
    # From Pearson and Kendall correlation analysis, features with strong correlation to PM2.5:
    # d2m (dew point temp), t2m (temperature), u10/v10 (wind speed), tcwv (water vapor), tp (precipitation), blh (boundary layer height), sp (pressure), str (radiation)
    selected_features = [
    'd2m',  # Dew point temperature - strongest correlation
    't2m',  # 2-meter temperature
    'wind_speed_10m',  # 10-meter wind speed (calculated)
    'tcwv',  # Total column water vapor
    'tp',  # Total precipitation
    'blh',  # Boundary layer height
    'sp',  # Surface pressure
    'str',  # Surface thermal radiation
    'tisr',  # TOA incident solar radiation
    'month',  # Month (time feature)
    'season',  # Season (time feature)
    ]

    # Check if features exist
    available_features = [f for f in selected_features if f in df_combined.columns]
    missing_features = [f for f in selected_features if f not in df_combined.columns]

    if missing_features:
        print(f"\nWarning: The following features do not exist and will be skipped: {missing_features}")

    print(f"\nSelected features ({len(available_features)} total):")
    for i, feat in enumerate(available_features, 1):
        print(f"  {i}. {feat}")

    # Prepare modeling data
    X = df_combined[available_features].copy()
    y = df_combined[target].copy()

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")

    # ============================== Data Validation ==============================
    # Check if data is empty
    if len(X) == 0 or len(y) == 0:
        print("\n" + "=" * 80)
        print("❌ Error: No available data!")
        print("=" * 80)
        print("\nPossible reasons:")
        print("1. Incorrect data path, unable to find data files")
        print("2. Pollution data or meteorological data failed to load")
        print("3. Indices have no intersection after data merging (check if date ranges match)")
        print("4. All rows deleted during data cleaning process")
        print("\nPlease check:")
        print(f"- Pollution data path: {pollution_all_path}")
        print(f"- Meteorological data path: {era5_path}")
        print(f"- Date range: {start_date.date()} to {end_date.date()}")
        print(f"\nPollution data shape: {df_pollution.shape}")
        print(f"Meteorological data shape: {df_era5.shape}")
        print(f"Data shape after merging: {df_combined.shape}")
        import sys
        sys.exit(1)

    print(f"Target variable statistics:")
    print(f"  Mean: {y.mean():.2f} μg/m³")
    print(f"  Std Dev: {y.std():.2f} μg/m³")
    print(f"  Min: {y.min():.2f} μg/m³")
    print(f"  Max: {y.max():.2f} μg/m³")
    print(f"  Median: {y.median():.2f} μg/m³")

    # ============================== Part5: Dataset Split ==============================
    print("\n" + "=" * 80)
    print("Step 3: Dataset Split")
    print("=" * 80)

    # Split by time order, first 80% as training set, last 20% as test set
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"\nTraining set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Time range: {X_train.index.min().date()} to {X_train.index.max().date()}")
    print(f"  PM2.5: {y_train.mean():.2f} ± {y_train.std():.2f} μg/m³")

    print(f"\nTest set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Time range: {X_test.index.min().date()} to {X_test.index.max().date()}")
    print(f"  PM2.5: {y_test.mean():.2f} ± {y_test.std():.2f} μg/m³")

    # ============================== Part6: Model Training - Multiple Linear Regression (MLR) ==============================
    print("\n" + "=" * 80)
    print("Model 1: Multiple Linear Regression (MLR)")
    print("=" * 80)

    # Create pipeline: standardization + linear regression
    mlr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
    ])

    # Train model
    print("\nTraining MLR model...")
    mlr_pipeline.fit(X_train, y_train)

    # Predict
    y_train_pred_mlr = mlr_pipeline.predict(X_train)
    y_test_pred_mlr = mlr_pipeline.predict(X_test)

    # Evaluate
    train_r2_mlr = r2_score(y_train, y_train_pred_mlr)
    train_rmse_mlr = np.sqrt(mean_squared_error(y_train, y_train_pred_mlr))
    train_mae_mlr = mean_absolute_error(y_train, y_train_pred_mlr)

    test_r2_mlr = r2_score(y_test, y_test_pred_mlr)
    test_rmse_mlr = np.sqrt(mean_squared_error(y_test, y_test_pred_mlr))
    test_mae_mlr = mean_absolute_error(y_test, y_test_pred_mlr)

    print("\nMLR Model Performance:")
    print(f"  Training set:")
    print(f"    R² Score: {train_r2_mlr:.4f}")
    print(f"    RMSE: {train_rmse_mlr:.4f}")
    print(f"    MAE: {train_mae_mlr:.4f}")
    print(f"  Test set:")
    print(f"    R² Score: {test_r2_mlr:.4f}")
    print(f"    RMSE: {test_rmse_mlr:.4f}")
    print(f"    MAE: {test_mae_mlr:.4f}")

    # Get feature coefficients
    mlr_coef = mlr_pipeline.named_steps['regressor'].coef_
    mlr_intercept = mlr_pipeline.named_steps['regressor'].intercept_

    print(f"\nModel Parameters:")
    print(f"  Intercept: {mlr_intercept:.4f}")
    print(f"  Feature coefficients:")
    for feat, coef in zip(available_features, mlr_coef):
        print(f"    {feat}: {coef:.4f}")

    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores_mlr = cross_val_score(mlr_pipeline, X_train, y_train, 
                                  cv=5, scoring='r2')
    print(f"  Cross-validation R² Scores: {cv_scores_mlr}")
    print(f"  Average R²: {cv_scores_mlr.mean():.4f} ± {cv_scores_mlr.std():.4f}")

    # ============================== Part7: Model Training - Ridge Regression ==============================
    print("\n" + "=" * 80)
    print("Model 2: Ridge Regression (L2 Regularization)")
    print("=" * 80)

    # Create Ridge regression pipeline
    ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=1.0))
    ])

    # Train
    print("\nTraining Ridge model...")
    ridge_pipeline.fit(X_train, y_train)

    # Predict
    y_train_pred_ridge = ridge_pipeline.predict(X_train)
    y_test_pred_ridge = ridge_pipeline.predict(X_test)

    # Evaluate
    train_r2_ridge = r2_score(y_train, y_train_pred_ridge)
    train_rmse_ridge = np.sqrt(mean_squared_error(y_train, y_train_pred_ridge))
    train_mae_ridge = mean_absolute_error(y_train, y_train_pred_ridge)

    test_r2_ridge = r2_score(y_test, y_test_pred_ridge)
    test_rmse_ridge = np.sqrt(mean_squared_error(y_test, y_test_pred_ridge))
    test_mae_ridge = mean_absolute_error(y_test, y_test_pred_ridge)

    print("\nRidge Model Performance:")
    print(f"  Training set:")
    print(f"    R² Score: {train_r2_ridge:.4f}")
    print(f"    RMSE: {train_rmse_ridge:.4f}")
    print(f"    MAE: {train_mae_ridge:.4f}")
    print(f"  Test set:")
    print(f"    R² Score: {test_r2_ridge:.4f}")
    print(f"    RMSE: {test_rmse_ridge:.4f}")
    print(f"    MAE: {test_mae_ridge:.4f}")

    # ============================== Part8: Model Training - GAM ==============================
    if GAM_AVAILABLE:
        print("\n" + "=" * 80)
        print("Model 3: Generalized Additive Model (GAM)")
        print("=" * 80)
        
        # Standardize data (GAM also needs this)
        scaler_gam = StandardScaler()
        X_train_scaled = scaler_gam.fit_transform(X_train)
        X_test_scaled = scaler_gam.transform(X_test)
        
        # Create GAM model
        # Add spline terms for each feature
        print("\nTraining GAM model...")
        gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + f(9) + f(10))
        
        try:
            gam.gridsearch(X_train_scaled, y_train.values)
            
            # Predict
            y_train_pred_gam = gam.predict(X_train_scaled)
            y_test_pred_gam = gam.predict(X_test_scaled)
            
            # Evaluate
            train_r2_gam = r2_score(y_train, y_train_pred_gam)
            train_rmse_gam = np.sqrt(mean_squared_error(y_train, y_train_pred_gam))
            train_mae_gam = mean_absolute_error(y_train, y_train_pred_gam)
            
            test_r2_gam = r2_score(y_test, y_test_pred_gam)
            test_rmse_gam = np.sqrt(mean_squared_error(y_test, y_test_pred_gam))
            test_mae_gam = mean_absolute_error(y_test, y_test_pred_gam)
            
            print("\nGAM Model Performance:")
            print(f"  Training set:")
            print(f"    R² Score: {train_r2_gam:.4f}")
            print(f"    RMSE: {train_rmse_gam:.4f}")
            print(f"    MAE: {train_mae_gam:.4f}")
            print(f"  Test set:")
            print(f"    R² Score: {test_r2_gam:.4f}")
            print(f"    RMSE: {test_rmse_gam:.4f}")
            print(f"    MAE: {test_mae_gam:.4f}")
            
            print(f"\nOptimal parameters: λ = {gam.lam}")
            
        except Exception as e:
            print(f"\nGAM model training error: {e}")
            GAM_AVAILABLE = False

    # ============================== Part9: Model Comparison ==============================
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)

    results = {
    'Model': ['MLR', 'Ridge'],
    'Train R²': [train_r2_mlr, train_r2_ridge],
    'Train RMSE': [train_rmse_mlr, train_rmse_ridge],
    'Train MAE': [train_mae_mlr, train_mae_ridge],
    'Test R²': [test_r2_mlr, test_r2_ridge],
    'Test RMSE': [test_rmse_mlr, test_rmse_ridge],
    'Test MAE': [test_mae_mlr, test_mae_ridge],
    }

    if GAM_AVAILABLE:
        results['Model'].append('GAM')
        results['Train R²'].append(train_r2_gam)
        results['Train RMSE'].append(train_rmse_gam)
        results['Train MAE'].append(train_mae_gam)
        results['Test R²'].append(test_r2_gam)
        results['Test RMSE'].append(test_rmse_gam)
        results['Test MAE'].append(test_mae_gam)

    results_df = pd.DataFrame(results)
    print("\nModel Performance Comparison:")
    print(results_df.to_string(index=False))

    # ============================== Part10: Visualization ==============================
    print("\n" + "=" * 80)
    print("Generating Visualization Results")
    print("=" * 80)

    # 1. Prediction vs Actual Values Scatter Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # MLR
    axes[0].scatter(y_test, y_test_pred_mlr, alpha=0.5, s=20)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual PM2.5 Concentration (μg/m³)', fontsize=12)
    axes[0].set_ylabel('Predicted PM2.5 Concentration (μg/m³)', fontsize=12)
    axes[0].set_title(f'MLR Model\nR²={test_r2_mlr:.4f}, RMSE={test_rmse_mlr:.2f}', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Ridge
    axes[1].scatter(y_test, y_test_pred_ridge, alpha=0.5, s=20, color='green')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual PM2.5 Concentration (μg/m³)', fontsize=12)
    axes[1].set_ylabel('Predicted PM2.5 Concentration (μg/m³)', fontsize=12)
    axes[1].set_title(f'Ridge Model\nR²={test_r2_ridge:.4f}, RMSE={test_rmse_ridge:.2f}', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # GAM
    if GAM_AVAILABLE:
        axes[2].scatter(y_test, y_test_pred_gam, alpha=0.5, s=20, color='orange')
        axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                     'r--', lw=2, label='Perfect Prediction')
        axes[2].set_xlabel('Actual PM2.5 Concentration (μg/m³)', fontsize=12)
        axes[2].set_ylabel('Predicted PM2.5 Concentration (μg/m³)', fontsize=12)
        axes[2].set_title(f'GAM Model\nR²={test_r2_gam:.4f}, RMSE={test_rmse_gam:.2f}', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'GAM Model Not Trained', ha='center', va='center', fontsize=14)
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
    print("\nSaved: prediction_scatter.png")
    plt.close()

    # 2. Time Series Prediction Comparison
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot only part of test set (last 200 points)
    plot_range = min(200, len(y_test))
    plot_indices = range(len(y_test) - plot_range, len(y_test))
    time_index = y_test.index[plot_indices]

    ax.plot(time_index, y_test.iloc[plot_indices], 'k-', label='Actual Values', linewidth=2, alpha=0.7)
    ax.plot(time_index, y_test_pred_mlr[plot_indices], 'b--', label='MLR Prediction', linewidth=1.5, alpha=0.7)
    ax.plot(time_index, y_test_pred_ridge[plot_indices], 'g-.', label='Ridge Prediction', linewidth=1.5, alpha=0.7)
    if GAM_AVAILABLE:
        ax.plot(time_index, y_test_pred_gam[plot_indices], 'r:', label='GAM Prediction', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
    ax.set_title('PM2.5 Concentration Prediction Time Series Comparison (Last 200 Days of Test Set)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_timeseries.png', dpi=300, bbox_inches='tight')
    print("Saved: prediction_timeseries.png")
    plt.close()

    # 3. Residual Analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # MLR Residuals
    residuals_mlr = y_test - y_test_pred_mlr
    axes[0].scatter(y_test_pred_mlr, residuals_mlr, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('MLR Model Residual Plot', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Ridge Residuals
    residuals_ridge = y_test - y_test_pred_ridge
    axes[1].scatter(y_test_pred_ridge, residuals_ridge, alpha=0.5, s=20, color='green')
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Values', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title('Ridge Model Residual Plot', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # GAM Residuals
    if GAM_AVAILABLE:
        residuals_gam = y_test - y_test_pred_gam
        axes[2].scatter(y_test_pred_gam, residuals_gam, alpha=0.5, s=20, color='orange')
        axes[2].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[2].set_xlabel('Predicted Values', fontsize=12)
        axes[2].set_ylabel('Residuals', fontsize=12)
        axes[2].set_title('GAM Model Residual Plot', fontsize=12)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: residuals_analysis.png")
    plt.close()

    # 4. Feature Importance (Based on MLR Coefficients)
    fig, ax = plt.subplots(figsize=(10, 6))

    feature_importance = pd.DataFrame({
    'Feature': available_features,
    'Coefficient': np.abs(mlr_coef)
    }).sort_values('Coefficient', ascending=True)

    ax.barh(feature_importance['Feature'], feature_importance['Coefficient'], color='steelblue')
    ax.set_xlabel('Absolute Coefficient Value', fontsize=12)
    ax.set_title('MLR Model Feature Importance (Absolute Coefficients)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: feature_importance.png")
    plt.close()

    # 5. Model Performance Comparison Bar Charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    models = results_df['Model'].tolist()
    x_pos = np.arange(len(models))

    # R² Score
    axes[0].bar(x_pos, results_df['Test R²'], color=['blue', 'green', 'orange'][:len(models)])
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('Test Set R² Score Comparison', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')

    # RMSE
    axes[1].bar(x_pos, results_df['Test RMSE'], color=['blue', 'green', 'orange'][:len(models)])
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(models)
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('Test Set RMSE Comparison', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')

    # MAE
    axes[2].bar(x_pos, results_df['Test MAE'], color=['blue', 'green', 'orange'][:len(models)])
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(models)
    axes[2].set_ylabel('MAE', fontsize=12)
    axes[2].set_title('Test Set MAE Comparison', fontsize=12)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: model_comparison.png")
    plt.close()

    # ============================== Part11: Save Results ==============================
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    # Save model performance
    results_df.to_csv(output_dir / 'model_performance.csv', index=False, encoding='utf-8-sig')
    print("\nSaved: model_performance.csv")

    # Save prediction results
    predictions_df = pd.DataFrame({
    'Date': y_test.index,
    'Actual': y_test.values,
    'MLR_Prediction': y_test_pred_mlr,
    'Ridge_Prediction': y_test_pred_ridge,
    })

    if GAM_AVAILABLE:
        predictions_df['GAM_Prediction'] = y_test_pred_gam

    predictions_df.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
    print("Saved: predictions.csv")

    # Save feature importance
    feature_importance_df = pd.DataFrame({
    'Feature': available_features,
    'MLR_Coefficient': mlr_coef,
    'Ridge_Coefficient': ridge_pipeline.named_steps['regressor'].coef_
    })
    feature_importance_df.to_csv(output_dir / 'feature_importance.csv', index=False, encoding='utf-8-sig')
    print("Saved: feature_importance.csv")

    # Save models (using pickle)
    import pickle
    with open(model_dir / 'mlr_model.pkl', 'wb') as f:
        pickle.dump(mlr_pipeline, f)
    print("Saved: mlr_model.pkl")

    with open(model_dir / 'ridge_model.pkl', 'wb') as f:
        pickle.dump(ridge_pipeline, f)
    print("Saved: ridge_model.pkl")

    if GAM_AVAILABLE:
        with open(model_dir / 'gam_model.pkl', 'wb') as f:
            pickle.dump((gam, scaler_gam), f)
        print("Saved: gam_model.pkl")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

    print("\nGenerated files:")
    print("\nCSV Files:")
    print("  - model_performance.csv       Model performance comparison")
    print("  - predictions.csv             Prediction results")
    print("  - feature_importance.csv      Feature importance")

    print("\nChart Files:")
    print("  - prediction_scatter.png      Prediction vs actual scatter plot")
    print("  - prediction_timeseries.png   Time series prediction comparison")
    print("  - residuals_analysis.png      Residual analysis plot")
    print("  - feature_importance.png      Feature importance plot")
    print("  - model_comparison.png        Model performance comparison plot")

    print("\nModel Files:")
    print("  - mlr_model.pkl               Multiple Linear Regression model")
    print("  - ridge_model.pkl             Ridge Regression model")
    if GAM_AVAILABLE:
        print("  - gam_model.pkl               Generalized Additive Model")

    # Best model information
    best_model_idx = results_df['Test R²'].idxmax()
    best_model = results_df.loc[best_model_idx]
    print(f"\nBest model: {best_model['Model']}")
    print(f"  R² Score: {best_model['Test R²']:.4f}")
    print(f"  RMSE: {best_model['Test RMSE']:.2f} μg/m³")
    print(f"  MAE: {best_model['Test MAE']:.2f} μg/m³")

    print("\nTop 5 Important Features:")
    for i, row in feature_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Coefficient']:.4f}")

    print("\n" + "=" * 80)
    print("MLR+GAM Baseline Model Training Complete!")
    print("=" * 80)

