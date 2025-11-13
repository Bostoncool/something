"""
Beijing PM2.5 Concentration Prediction - LightGBM Model
Using LightGBM gradient boosting decision tree for time series prediction

Features:
- Efficient gradient boosting algorithm
- Supports categorical features
- Built-in feature importance
- Early stopping mechanism to prevent overfitting
- Bayesian optimization hyperparameters

Data sources:
- Pollution data: Benchmark dataset (PM2.5, PM10, SO2, NO2, CO, O3)
- Meteorological data: ERA5 reanalysis data
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings
import pickle
from pathlib import Path
import glob
import multiprocessing

warnings.filterwarnings('ignore')

# Get CPU core count
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)  # Reserve 1 core for system

# Try importing tqdm progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm is not installed, progress display will use simplified version.")
    print("      You can use 'pip install tqdm' to install for better progress bar display.")

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# LightGBM
import lightgbm as lgb

# Bayesian optimization (optional)
try:
    from bayes_opt import BayesianOptimization
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    print("Note: bayesian-optimization is not installed, grid search will be used.")
    print("      You can use 'pip install bayesian-optimization' to install and enable Bayesian optimization.")
    BAYESIAN_OPT_AVAILABLE = False

# Set font for plotting
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# Set random seed
np.random.seed(42)

print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - LightGBM Model")
print("=" * 80)

# ============================== Part 1: Configuration and Path Settings ==============================
print("\nConfiguring parameters...")

# Data paths
pollution_all_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)'
era5_path = r'C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV'

# Output paths
output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)

# Model save paths
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
    'mn2t', 'sd', 'lsm'  # Other
]

print(f"Data time range: {start_date.date()} to {end_date.date()}")
print(f"Target variable: PM2.5 concentration")
print(f"Output directory: {output_dir}")
print(f"Model save directory: {model_dir}")
print(f"CPU core count: {CPU_COUNT}, parallel worker threads: {MAX_WORKERS}")

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
        
        # Aggregate by date and type (average across all stations)
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
    print(f"Using {MAX_WORKERS} parallel worker threads")
    dates = list(daterange(start_date, end_date))
    pollution_dfs = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, date): date for date in dates}
        
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
        print(f"  Successfully read {len(pollution_dfs)}/{len(dates)} days of data")
        print("  Merging data...")
        df_poll_all = pd.concat(pollution_dfs)
        # Forward fill then mean fill
        df_poll_all.ffill(inplace=True)
        df_poll_all.fillna(df_poll_all.mean(), inplace=True)
        print(f"Pollution data loading complete, shape: {df_poll_all.shape}")
        return df_poll_all
    return pd.DataFrame()

def read_era5_month(year, month):
    """Read single month ERA5 data - Handle structure with variables in separate folders"""
    month_str = f"{year}{month:02d}"
    
    # Find all files containing data for this month (from different variable folders)
    all_files = glob.glob(os.path.join(era5_path, "**", f"*{month_str}*.csv"), recursive=True)
    
    if not all_files:
        # print(f"  Warning: Meteorological data files for {year}-{month} not found")
        return None
    
    # Store data for all variables
    monthly_data = None
    loaded_vars = []
    
    for file_path in all_files:
        try:
            # Read single variable file
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', 
                            low_memory=False, comment='#')
            
            if df.empty or 'time' not in df.columns:
                continue
            
            # Process time
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.dropna(subset=['time'])
            
            if len(df) == 0:
                continue
            
            # Filter Beijing region
            if 'latitude' in df.columns and 'longitude' in df.columns:
                df = df[(df['latitude'] >= beijing_lats.min()) & 
                       (df['latitude'] <= beijing_lats.max()) &
                       (df['longitude'] >= beijing_lons.min()) & 
                       (df['longitude'] <= beijing_lons.max())]
                
                if len(df) == 0:
                    continue
            
            # Process expver
            if 'expver' in df.columns:
                if '0001' in df['expver'].values:
                    df = df[df['expver'] == '0001']
                else:
                    first_expver = df['expver'].iloc[0]
                    df = df[df['expver'] == first_expver]
            
            # Extract date
            df['date'] = df['time'].dt.date
            
            # Find which variables this file contains
            avail_vars = [v for v in era5_vars if v in df.columns]
            
            if not avail_vars:
                continue
            
            # Convert to numeric type
            for col in avail_vars:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Aggregate by date (spatial and temporal average)
            df_daily = df.groupby('date')[avail_vars].mean().reset_index()
            df_daily.set_index('date', inplace=True)
            df_daily.index = pd.to_datetime(df_daily.index)
            
            # Merge to monthly_data
            if monthly_data is None:
                monthly_data = df_daily
            else:
                # Using join to merge, keep all dates
                monthly_data = monthly_data.join(df_daily, how='outer')
            
            loaded_vars.extend(avail_vars)
            
        except Exception as e:
            # print(f"  Error: Failed to process file {os.path.basename(file_path)} - {e}")
            continue
    
    if monthly_data is not None and not monthly_data.empty:
        print(f"  Successfully read: {year}-{month:02d}, days: {len(monthly_data)}, variables: {len(loaded_vars)}")
        return monthly_data
    else:
        # print(f"  Warning: {year} Year {month} Month - no data loaded successfully")
        return None

def read_all_era5():
    """Read all ERA5 data in parallel"""
    print("\nLoading meteorological data...")
    print(f"Using {MAX_WORKERS} parallel worker threads")
    print(f"Meteorological data directory: {era5_path}")
    print(f"Checking if directory exists: {os.path.exists(era5_path)}")
    
    # First check what files are in the directory
    if os.path.exists(era5_path):
        all_csv = glob.glob(os.path.join(era5_path, "**", "*.csv"), recursive=True)
        print(f"Found {len(all_csv)} CSV files")
        if all_csv:
            print(f"Example files: {[os.path.basename(f) for f in all_csv[:5]]}")
    
    era5_dfs = []
    years = range(2015, 2025)
    months = range(1, 13)
    
    # Prepare all tasks
    month_tasks = [(year, month) for year in years for month in months 
                   if not (year == 2024 and month > 12)]
    total_months = len(month_tasks)
    print(f"Attempting to load {total_months} months of data...")
    
    # Using parallel threads to load ERA5 data
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_era5_month, year, month): (year, month) 
                  for year, month in month_tasks}
        
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
        
        print(f"  Total successfully read: {successful_reads}/{len(futures)} months")
    
    if era5_dfs:
        print("\nMerging meteorological data...")
        df_era5_all = pd.concat(era5_dfs, axis=0)
        
        # Deduplicate (possible duplicate dates)
        print("  Deduplicating...")
        df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep='first')]
        
        # Sort
        print("  Sorting...")
        df_era5_all.sort_index(inplace=True)
        
        print(f"Shape after merge: {df_era5_all.shape}")
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
        print("\n❌ Error: No meteorological data files loaded successfully!")
        print("Possible reasons:")
        print("1. File naming format does not match (Expected format: *YYYYMM*.csv)")
        print("2. File content format is incorrect (Missing time column)")
        print("3. File path is incorrect")
        return pd.DataFrame()

# ============================== Part 3: Feature Engineering ==============================
def create_features(df):
    """Create additional features"""
    df_copy = df.copy()
    
    # 1. Wind speed features
    if 'u10' in df_copy and 'v10' in df_copy:
        df_copy['wind_speed_10m'] = np.sqrt(df_copy['u10']**2 + df_copy['v10']**2)
        df_copy['wind_dir_10m'] = np.arctan2(df_copy['v10'], df_copy['u10']) * 180 / np.pi
        df_copy['wind_dir_10m'] = (df_copy['wind_dir_10m'] + 360) % 360  # Convert to 0-360 degrees
    
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
    
    # Season features
    df_copy['season'] = df_copy['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )
    
    # Is heating season (Beijing: Nov 15 - Mar 15)
    df_copy['is_heating_season'] = ((df_copy['month'] >= 11) | (df_copy['month'] <= 3)).astype(int)
    
    # 3. Temperature related features
    if 't2m' in df_copy and 'd2m' in df_copy:
        # Temperature-dewpoint difference (reflects relative humidity)
        df_copy['temp_dewpoint_diff'] = df_copy['t2m'] - df_copy['d2m']
    
    # 4. Lag features (previous 1, 3, 7 days PM2.5)
    if 'PM2.5' in df_copy:
        df_copy['PM2.5_lag1'] = df_copy['PM2.5'].shift(1)
        df_copy['PM2.5_lag3'] = df_copy['PM2.5'].shift(3)
        df_copy['PM2.5_lag7'] = df_copy['PM2.5'].shift(7)
        
        # Rolling average features
        df_copy['PM2.5_ma3'] = df_copy['PM2.5'].rolling(window=3, min_periods=1).mean()
        df_copy['PM2.5_ma7'] = df_copy['PM2.5'].rolling(window=7, min_periods=1).mean()
        df_copy['PM2.5_ma30'] = df_copy['PM2.5'].rolling(window=30, min_periods=1).mean()
    
    # 5. Relative humidity estimation (simplified formula)
    if 't2m' in df_copy and 'd2m' in df_copy:
        # Magnus formula approximation
        df_copy['relative_humidity'] = 100 * np.exp((17.625 * (df_copy['d2m'] - 273.15)) / 
                                                      (243.04 + (df_copy['d2m'] - 273.15))) / \
                                        np.exp((17.625 * (df_copy['t2m'] - 273.15)) / 
                                               (243.04 + (df_copy['t2m'] - 273.15)))
        df_copy['relative_humidity'] = df_copy['relative_humidity'].clip(0, 100)
    
    # 6. Wind direction category (by 8 directions)
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
    print("   Possible reason: No overlapping date indices between pollution and meteorological data.")
    print(f"   Pollution data has {len(df_pollution)} rows")
    print(f"   Meteorological data has {len(df_era5)} rows")
    print(f"   After merge: {len(df_combined)} rows")
    import sys
    sys.exit(1)

# Create features
print("\nCreating features...")
df_combined = create_features(df_combined)

# Clean data
print("\nCleaning data...")
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows containing NaN (mainly first few rows due to lag features)
initial_rows = len(df_combined)
df_combined.dropna(inplace=True)
final_rows = len(df_combined)
print(f"Removed {initial_rows - final_rows} rows containing missing values")

print(f"\nData shape after merge: {df_combined.shape}")
print(f"Time range: {df_combined.index.min().date()} to {df_combined.index.max().date()}")
print(f"Number of samples: {len(df_combined)}")
print(f"Number of features: {df_combined.shape[1]}")

# Display feature list
print(f"\nFeature list (top 20):")
for i, col in enumerate(df_combined.columns[:20], 1):
    print(f"  {i}. {col}")
if len(df_combined.columns) > 20:
    print(f"  ... and {len(df_combined.columns) - 20} more features")

# ============================== Part 5: Feature Selection and Data Preparation ==============================
print("\n" + "=" * 80)
print("Step 2: Feature Selection and Data Preparation")
print("=" * 80)

# Define target variable
target = 'PM2.5'

# Exclude columns (target variable, other pollutants, year, etc.)
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']

# Select numeric features
numeric_features = [col for col in df_combined.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_cols]

print(f"\nNumber of selected features: {len(numeric_features)}")
print(f"Target variable: {target}")

# Prepare modeling data
X = df_combined[numeric_features].copy()
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
    print("1. Data path is incorrect, unable to find data files")
    print("2. Pollution or meteorological data loading failed")
    print("3. No overlapping indices after data merge (check if date ranges match)")
    print("4. All rows deleted during data cleaning process")
    print("\nPlease check:")
    print(f"- Pollution data path: {pollution_all_path}")
    print(f"- Meteorological data path: {era5_path}")
    print(f"- Date range: {start_date.date()} to {end_date.date()}")
    print(f"\nPollution data shape: {df_pollution.shape}")
    print(f"Meteorological data shape: {df_era5.shape}")
    print(f"Data shape after merge: {df_combined.shape}")
    import sys
    sys.exit(1)

print(f"\nPM2.5 Statistics:")
print(f"  Mean: {y.mean():.2f} μg/m³")
print(f"  Std Dev: {y.std():.2f} μg/m³")
print(f"  Min: {y.min():.2f} μg/m³")
print(f"  Max: {y.max():.2f} μg/m³")
print(f"  Median: {y.median():.2f} μg/m³")

# ============================== Part 6: Dataset Split ==============================
print("\n" + "=" * 80)
print("Step 3: Dataset Split")
print("=" * 80)

# Split by temporal order: Train 70%, Validation 15%, Test 15%
n_samples = len(X)
train_size = int(n_samples * 0.70)
val_size = int(n_samples * 0.15)

X_train = X.iloc[:train_size]
X_val = X.iloc[train_size:train_size + val_size]
X_test = X.iloc[train_size + val_size:]

y_train = y.iloc[:train_size]
y_val = y.iloc[train_size:train_size + val_size]
y_test = y.iloc[train_size + val_size:]

print(f"\nTraining set: {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
print(f"  Time range: {X_train.index.min().date()} to {X_train.index.max().date()}")
print(f"  PM2.5: {y_train.mean():.2f} ± {y_train.std():.2f} μg/m³")

print(f"\nValidation set: {len(X_val)} samples ({len(X_val)/n_samples*100:.1f}%)")
print(f"  Time range: {X_val.index.min().date()} to {X_val.index.max().date()}")
print(f"  PM2.5: {y_val.mean():.2f} ± {y_val.std():.2f} μg/m³")

print(f"\nTest set: {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")
print(f"  Time range: {X_test.index.min().date()} to {X_test.index.max().date()}")
print(f"  PM2.5: {y_test.mean():.2f} ± {y_test.std():.2f} μg/m³")

# Create LightGBM datasets
lgb_train = lgb.Dataset(X_train, y_train, feature_name=list(X_train.columns))
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, feature_name=list(X_val.columns))

# ============================== Part 7: LightGBM Basic Model ==============================
print("\n" + "=" * 80)
print("Step 4: Training LightGBM Basic Model")
print("=" * 80)

# Basic parameters
params_basic = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'num_threads': MAX_WORKERS,  # Using multi-threading for acceleration
    'verbose': -1,
    'seed': 42
}

print("\nBasic model parameters:")
for key, value in params_basic.items():
    print(f"  {key}: {value}")

print("\nStarting training of basic model...")
evals_result_basic = {}
model_basic = lgb.train(
    params_basic,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100),
        lgb.record_evaluation(evals_result_basic)
    ]
)

print(f"\n✓ Basic model training complete")
print(f"  Best iteration: {model_basic.best_iteration}")
print(f"  Training set RMSE: {evals_result_basic['train']['rmse'][model_basic.best_iteration-1]:.4f}")
print(f"  Validation set RMSE: {evals_result_basic['valid']['rmse'][model_basic.best_iteration-1]:.4f}")

# Predict
y_train_pred_basic = model_basic.predict(X_train, num_iteration=model_basic.best_iteration)
y_val_pred_basic = model_basic.predict(X_val, num_iteration=model_basic.best_iteration)
y_test_pred_basic = model_basic.predict(X_test, num_iteration=model_basic.best_iteration)

# Evaluate
def evaluate_model(y_true, y_pred, dataset_name):
    """Evaluate model performance"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'Dataset': dataset_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

results_basic = []
results_basic.append(evaluate_model(y_train, y_train_pred_basic, 'Train'))
results_basic.append(evaluate_model(y_val, y_val_pred_basic, 'Validation'))
results_basic.append(evaluate_model(y_test, y_test_pred_basic, 'Test'))

results_basic_df = pd.DataFrame(results_basic)
print("\nBasic model performance:")
print(results_basic_df.to_string(index=False))

# ============================== Part 8: Hyperparameter Optimization ==============================
print("\n" + "=" * 80)
print("Step 5: Hyperparameter Optimization")
print("=" * 80)

if BAYESIAN_OPT_AVAILABLE:
    print("\nUsing Bayesian optimization for hyperparameter search...")
    
    def lgb_evaluate(num_leaves, max_depth, learning_rate, feature_fraction, 
                     bagging_fraction, min_child_samples):
        """Bayesian optimization objective function"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': int(num_leaves),
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': 5,
            'min_child_samples': int(min_child_samples),
            'feature_pre_filter': False,  # Allow dynamic adjustment of min_child_samples
            'num_threads': MAX_WORKERS,  # Using multi-threading for acceleration
            'verbose': -1,
            'seed': 42
        }
        
        # Recreate Dataset inside function to use new parameters each time
        lgb_train_temp = lgb.Dataset(X_train, y_train, feature_name=list(X_train.columns), 
                                      params={'feature_pre_filter': False})
        lgb_val_temp = lgb.Dataset(X_val, y_val, reference=lgb_train_temp, 
                                    feature_name=list(X_val.columns))
        
        model = lgb.train(
            params,
            lgb_train_temp,
            num_boost_round=500,
            valid_sets=[lgb_val_temp],
            callbacks=[lgb.early_stopping(stopping_rounds=30)]
        )
        
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        # Return negative RMSE (because Bayesian optimization maximizes)
        return -rmse
    
    # Define parameter search space
    pbounds = {
        'num_leaves': (20, 100),
        'max_depth': (3, 12),
        'learning_rate': (0.01, 0.1),
        'feature_fraction': (0.5, 1.0),
        'bagging_fraction': (0.5, 1.0),
        'min_child_samples': (10, 50)
    }
    
    optimizer = BayesianOptimization(
        f=lgb_evaluate,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    optimizer.maximize(init_points=5, n_iter=15)
    
    # Get best parameters
    best_params = optimizer.max['params']
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_samples'] = int(best_params['min_child_samples'])
    
    print(f"\nBest parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  Best validation RMSE: {-optimizer.max['target']:.4f}")
    
else:
    # Grid search (parallel version)
    print("\nUsing Grid search for hyperparameter optimization...")
    print(f"Using {min(MAX_WORKERS, 4)} parallel worker threads")
    
    param_grid = {
        'num_leaves': [31, 50, 70],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.03, 0.05, 0.07],
        'feature_fraction': [0.7, 0.8, 0.9],
    }
    
    from itertools import product
    total_combinations = int(np.prod([len(v) for v in param_grid.values()]))
    print(f"Total {total_combinations} parameter combinations")
    
    # Define single grid search function
    def evaluate_params(combo):
        """Evaluate single parameter combination"""
        params_test = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': combo[0],
            'max_depth': combo[1],
            'learning_rate': combo[2],
            'feature_fraction': combo[3],
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'feature_pre_filter': False,
            'num_threads': MAX_WORKERS,  # Using multi-threading for acceleration
            'verbose': -1,
            'seed': 42
        }
        
        # Create temporary Dataset
        lgb_train_temp = lgb.Dataset(X_train, y_train, feature_name=list(X_train.columns),
                                      params={'feature_pre_filter': False})
        lgb_val_temp = lgb.Dataset(X_val, y_val, reference=lgb_train_temp,
                                    feature_name=list(X_val.columns))
        
        model_temp = lgb.train(
            params_test,
            lgb_train_temp,
            num_boost_round=500,
            valid_sets=[lgb_val_temp],
            callbacks=[lgb.early_stopping(stopping_rounds=30)]
        )
        
        y_pred_temp = model_temp.predict(X_val, num_iteration=model_temp.best_iteration)
        rmse_temp = np.sqrt(mean_squared_error(y_val, y_pred_temp))
        
        return combo, rmse_temp
    
    # Prepare all parameter combinations
    all_combos = list(product(*param_grid.values()))
    
    # Execute grid search in parallel
    best_rmse = float('inf')
    best_params = {}
    
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 4)) as executor:
        futures = {executor.submit(evaluate_params, combo): combo for combo in all_combos}
        
        if TQDM_AVAILABLE:
            # Using tqdm progress bar
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Grid search", unit=" combination"):
                combo, rmse_temp = future.result()
                if rmse_temp < best_rmse:
                    best_rmse = rmse_temp
                    best_params = {
                        'num_leaves': combo[0],
                        'max_depth': combo[1],
                        'learning_rate': combo[2],
                        'feature_fraction': combo[3]
                    }
        else:
            # Simplified progress display
            for i, future in enumerate(as_completed(futures), 1):
                combo, rmse_temp = future.result()
                if rmse_temp < best_rmse:
                    best_rmse = rmse_temp
                    best_params = {
                        'num_leaves': combo[0],
                        'max_depth': combo[1],
                        'learning_rate': combo[2],
                        'feature_fraction': combo[3]
                    }
                if i % 10 == 0 or i == len(futures):
                    print(f"  Tested {i}/{total_combinations} combinations, current best RMSE: {best_rmse:.4f}")
    
    print(f"\nBest parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  Best validation RMSE: {best_rmse:.4f}")

# ============================== Part 9: Train Optimized Model ==============================
print("\n" + "=" * 80)
print("Step 6: Training Optimized Model with Best Parameters")
print("=" * 80)

params_optimized = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': best_params.get('num_leaves', 50),
    'max_depth': best_params.get('max_depth', 7),
    'learning_rate': best_params.get('learning_rate', 0.05),
    'feature_fraction': best_params.get('feature_fraction', 0.8),
    'bagging_fraction': best_params.get('bagging_fraction', 0.8),
    'bagging_freq': 5,
    'min_child_samples': best_params.get('min_child_samples', 20),
    'num_threads': MAX_WORKERS,  # Using multi-threading for acceleration
    'verbose': -1,
    'seed': 42
}

print("\nOptimized model parameters:")
for key, value in params_optimized.items():
    print(f"  {key}: {value}")

print("\nStarting training of optimized model...")
evals_result_opt = {}
model_optimized = lgb.train(
    params_optimized,
    lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100),
        lgb.record_evaluation(evals_result_opt)
    ]
)

print(f"\n✓ Optimized model training complete")
print(f"  Best iteration: {model_optimized.best_iteration}")
print(f"  Training set RMSE: {evals_result_opt['train']['rmse'][model_optimized.best_iteration-1]:.4f}")
print(f"  Validation set RMSE: {evals_result_opt['valid']['rmse'][model_optimized.best_iteration-1]:.4f}")

# Predict
y_train_pred_opt = model_optimized.predict(X_train, num_iteration=model_optimized.best_iteration)
y_val_pred_opt = model_optimized.predict(X_val, num_iteration=model_optimized.best_iteration)
y_test_pred_opt = model_optimized.predict(X_test, num_iteration=model_optimized.best_iteration)

# Evaluate
results_opt = []
results_opt.append(evaluate_model(y_train, y_train_pred_opt, 'Train'))
results_opt.append(evaluate_model(y_val, y_val_pred_opt, 'Validation'))
results_opt.append(evaluate_model(y_test, y_test_pred_opt, 'Test'))

results_opt_df = pd.DataFrame(results_opt)
print("\nOptimized model performance:")
print(results_opt_df.to_string(index=False))

# ============================== Part 10: Model Comparison ==============================
print("\n" + "=" * 80)
print("Step 7: Model Performance Comparison")
print("=" * 80)

# Merge results
results_basic_df['Model'] = 'LightGBM_Basic'
results_opt_df['Model'] = 'LightGBM_Optimized'
all_results = pd.concat([results_basic_df, results_opt_df])

# Reorder columns
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

r2_improvement = (opt_test_r2 - basic_test_r2) / basic_test_r2 * 100
rmse_improvement = (basic_test_rmse - opt_test_rmse) / basic_test_rmse * 100

print(f"\nOptimization Effect:")
print(f"  R² improvement: {r2_improvement:.2f}%")
print(f"  RMSE reduction: {rmse_improvement:.2f}%")

# ============================== Part 11: Feature Importance Analysis ==============================
print("\n" + "=" * 80)
print("Step 8: Feature Importance Analysis")
print("=" * 80)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': model_optimized.feature_name(),
    'Importance_Split': model_optimized.feature_importance(importance_type='split'),
    'Importance_Gain': model_optimized.feature_importance(importance_type='gain')
})

# Normalize importance
feature_importance['Importance_Split_Norm'] = (feature_importance['Importance_Split'] / 
                                                feature_importance['Importance_Split'].sum() * 100)
feature_importance['Importance_Gain_Norm'] = (feature_importance['Importance_Gain'] / 
                                               feature_importance['Importance_Gain'].sum() * 100)

# Sort by gain importance
feature_importance = feature_importance.sort_values('Importance_Gain', ascending=False)

print(f"\nTop 20 important features (by Gain):")
print(feature_importance.head(20)[['Feature', 'Importance_Gain_Norm']].to_string(index=False))

# ============================== Part 12: Visualization ==============================
print("\n" + "=" * 80)
print("Step 9: Generate Visualization Charts")
print("=" * 80)

# 12.1 Training curves
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Basic model
axes[0].plot(evals_result_basic['train']['rmse'], label='Training set', linewidth=2)
axes[0].plot(evals_result_basic['valid']['rmse'], label='Validation set', linewidth=2)
axes[0].axvline(x=model_basic.best_iteration, color='r', linestyle='--', 
                label=f'Best iteration({model_basic.best_iteration})', linewidth=1.5)
axes[0].set_xlabel('Iterations', fontsize=12)
axes[0].set_ylabel('RMSE', fontsize=12)
axes[0].set_title('LightGBM Basic Model - Training Process', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Optimized model
axes[1].plot(evals_result_opt['train']['rmse'], label='Training set', linewidth=2)
axes[1].plot(evals_result_opt['valid']['rmse'], label='Validation set', linewidth=2)
axes[1].axvline(x=model_optimized.best_iteration, color='r', linestyle='--',
                label=f'Best iteration({model_optimized.best_iteration})', linewidth=1.5)
axes[1].set_xlabel('Iterations', fontsize=12)
axes[1].set_ylabel('RMSE', fontsize=12)
axes[1].set_title('LightGBM Optimized Model - Training Process', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
print("Saved: training_curves.png")
plt.close()

# 12.2 Predicted vs Actual Values Scatter Plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

models_data = [
    ('Basic', y_train_pred_basic, y_train, 'Train'),
    ('Basic', y_val_pred_basic, y_val, 'Val'),
    ('Basic', y_test_pred_basic, y_test, 'Test'),
    ('Optimized', y_train_pred_opt, y_train, 'Train'),
    ('Optimized', y_val_pred_opt, y_val, 'Val'),
    ('Optimized', y_test_pred_opt, y_test, 'Test')
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
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal line')
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    ax.set_xlabel('Actual PM2.5 Concentration (μg/m³)', fontsize=11)
    ax.set_ylabel('Predicted PM2.5 Concentration (μg/m³)', fontsize=11)
    ax.set_title(f'LightGBM_{model_name} - {dataset}\nR²={r2:.4f}, RMSE={rmse:.2f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
print("Saved: prediction_scatter.png")
plt.close()

# 12.3 Time Series Prediction Comparison
fig, axes = plt.subplots(2, 1, figsize=(18, 10))

# Test set - Basic model
plot_range = min(300, len(y_test))
plot_idx = range(len(y_test) - plot_range, len(y_test))
time_idx = y_test.index[plot_idx]

axes[0].plot(time_idx, y_test.iloc[plot_idx], 'k-', label='Actual values', 
             linewidth=2, alpha=0.8)
axes[0].plot(time_idx, y_test_pred_basic[plot_idx], 'b--', label='Basic model prediction', 
             linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
axes[0].set_title('LightGBM Basic Model - Time Series Prediction Comparison (Last 300 days of Test set)', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

# Test set - Optimized model
axes[1].plot(time_idx, y_test.iloc[plot_idx], 'k-', label='Actual values', 
             linewidth=2, alpha=0.8)
axes[1].plot(time_idx, y_test_pred_opt[plot_idx], 'g--', label='Optimized model prediction', 
             linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
axes[1].set_title('LightGBM Optimized Model - Time Series Prediction Comparison (Last 300 days of Test set)', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(output_dir / 'timeseries_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: timeseries_comparison.png")
plt.close()

# 12.4 Residual analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (model_name, y_pred, y_true, dataset) in enumerate(models_data):
    row = idx // 3
    col = idx % 3
    
    ax = axes[row, col]
    
    residuals = y_true - y_pred
    
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted values (μg/m³)', fontsize=11)
    ax.set_ylabel('Residuals (μg/m³)', fontsize=11)
    ax.set_title(f'LightGBM_{model_name} - {dataset}\nResidual Mean={residuals.mean():.2f}, Std Dev={residuals.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: residuals_analysis.png")
plt.close()

# 12.5 Feature Importance Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

top_n = 20
top_features_gain = feature_importance.head(top_n)
top_features_split = feature_importance.sort_values('Importance_Split', ascending=False).head(top_n)

# Sort by Gain
axes[0].barh(range(top_n), top_features_gain['Importance_Gain_Norm'], color='steelblue')
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(top_features_gain['Feature'], fontsize=10)
axes[0].set_xlabel('Importance (%)', fontsize=12)
axes[0].set_title(f'Top {top_n} Important Features (by Gain)', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')
axes[0].invert_yaxis()

# Sort by Split
axes[1].barh(range(top_n), top_features_split['Importance_Split_Norm'], color='coral')
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(top_features_split['Feature'], fontsize=10)
axes[1].set_xlabel('Importance (%)', fontsize=12)
axes[1].set_title(f'Top {top_n} Important Features (by Split)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: feature_importance.png")
plt.close()

# 12.6 Model Performance Comparison Bar Chart
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

test_results = all_results[all_results['Dataset'] == 'Test']
models = test_results['Model'].tolist()
x_pos = np.arange(len(models))
colors = ['blue', 'green']

metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
for i, metric in enumerate(metrics):
    axes[i].bar(x_pos, test_results[metric], color=colors, alpha=0.7, 
                edgecolor='black', linewidth=1.5)
    axes[i].set_xticks(x_pos)
    axes[i].set_xticklabels(['Basic', 'Optimized'], fontsize=11)
    axes[i].set_ylabel(metric, fontsize=12)
    
    if metric == 'R²':
        axes[i].set_title(f'{metric} Comparison\n(Higher is better)', fontsize=12, fontweight='bold')
    else:
        axes[i].set_title(f'{metric} Comparison\n(Lower is better)', fontsize=12, fontweight='bold')
    
    axes[i].grid(True, alpha=0.3, axis='y')
    
    # Display values
    for j, v in enumerate(test_results[metric]):
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

# 12.7 Error Distribution Histogram
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

errors_basic = y_test - y_test_pred_basic
errors_opt = y_test - y_test_pred_opt

axes[0].hist(errors_basic, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero error')
axes[0].set_xlabel('Prediction Error (μg/m³)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title(f'Basic Model - Prediction Error Distribution\nMean={errors_basic.mean():.2f}, Std Dev={errors_basic.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].hist(errors_opt, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero error')
axes[1].set_xlabel('Prediction Error (μg/m³)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title(f'Optimized Model - Prediction Error Distribution\nMean={errors_opt.mean():.2f}, Std Dev={errors_opt.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: error_distribution.png")
plt.close()

# ============================== Part 13: Save Results ==============================
print("\n" + "=" * 80)
print("Step 10: Save Results")
print("=" * 80)

# Save model performance
all_results.to_csv(output_dir / 'model_performance.csv', index=False, encoding='utf-8-sig')
print("Saved: model_performance.csv")

# Save feature importance
feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False, encoding='utf-8-sig')
print("Saved: feature_importance.csv")

# Save best parameters
best_params_df = pd.DataFrame([params_optimized])
best_params_df.to_csv(output_dir / 'best_parameters.csv', index=False, encoding='utf-8-sig')
print("Saved: best_parameters.csv")

# Save prediction results
predictions_df = pd.DataFrame({
    'Date': y_test.index,
    'Actual': y_test.values,
    'Prediction_Basic': y_test_pred_basic,
    'Prediction_Optimized': y_test_pred_opt,
    'Error_Basic': y_test.values - y_test_pred_basic,
    'Error_Optimized': y_test.values - y_test_pred_opt
})
predictions_df.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
print("Saved: predictions.csv")

# Save model
model_optimized.save_model(str(model_dir / 'lightgbm_optimized.txt'))
print("Saved: lightgbm_optimized.txt")

# Using pickle to save model (optional)
with open(model_dir / 'lightgbm_optimized.pkl', 'wb') as f:
    pickle.dump(model_optimized, f)
print("Saved: lightgbm_optimized.pkl")

# ============================== Part 14: Summary Report ==============================
print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)

print("\nGenerated files:")
print("\nCSV Files:")
print("  - model_performance.csv       Model performance comparison")
print("  - feature_importance.csv      Feature importance")
print("  - best_parameters.csv         Best parameters")
print("  - predictions.csv             Prediction results")

print("\nChart Files:")
print("  - training_curves.png         Training curves")
print("  - prediction_scatter.png      Predicted vs Actual scatter plot")
print("  - timeseries_comparison.png   Time series comparison")
print("  - residuals_analysis.png      Residual analysis")
print("  - feature_importance.png      Feature importance plot")
print("  - model_comparison.png        Model performance comparison")
print("  - error_distribution.png      Error distribution")

print("\nModel Files:")
print("  - lightgbm_optimized.txt      LightGBM model (text format)")
print("  - lightgbm_optimized.pkl      LightGBM model (pickle format)")

# Best model information
best_model = test_results.iloc[0]
print(f"\nBest model: {best_model['Model']}")
print(f"  R² Score: {best_model['R²']:.4f}")
print(f"  RMSE: {best_model['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model['MAPE']:.2f}%")

print("\nTop 5 Most Important Features:")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance_Gain_Norm']:.2f}%")

print("\n" + "=" * 80)
print("LightGBM PM2.5 Concentration Prediction Complete!")
print("=" * 80)

