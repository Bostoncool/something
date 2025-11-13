"""
Beijing PM2.5 Concentration Prediction - Random Forest Model
Using meteorological variables with strong correlation (temperature, humidity, wind speed) to build Random Forest model to predict PM2.5 concentration

Main features:
- Temperature-related: t2m (2-meter temperature), d2m (dew point temperature)
- Humidity-related: tcwv (total column water vapor)
- Wind speed-related: wind_speed_10m, wind_speed_100m (calculated from u/v components)
- Other important meteorological factors: blh (boundary layer height), tp (precipitation), sp (pressure), str (radiation)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import glob
import multiprocessing
import warnings
import gc

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

# Set font
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# Set random seed
np.random.seed(42)

print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - Random Forest Model")
print("=" * 80)

# ============================== Data Path Configuration ==============================
pollution_all_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)'
era5_path = r'C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV'

# Output path
output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)

# Model save path
model_dir = Path('./models')
model_dir.mkdir(exist_ok=True)

# Time range: 2015-01-01 to 2024-12-31
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

# Beijing region range
beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

# Pollutant list
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# ERA5 variables
era5_vars = [
    'd2m', 't2m', 'u10', 'v10', 'u100', 'v100',  # Temperature, Wind speed
    'blh', 'sp', 'tcwv',  # Boundary layer height, Pressure, Water vapor
    'tp', 'avg_tprate',  # Precipitation
    'tisr', 'str',  # Radiation
    'cvh', 'cvl',  # Cloud cover
    'mn2t', 'sd', 'lsm'  # Other
]

print(f"\nConfiguration Parameters:")
print(f"Data time range: {start_date.date()} to {end_date.date()}")
print(f"Target variable: PM2.5 concentration")
print(f"Output directory: {output_dir}")
print(f"Model save directory: {model_dir}")
print(f"CPU cores: {CPU_COUNT}, Parallel worker threads: {MAX_WORKERS}")

# ============================== Helper Functions ==============================
def daterange(start, end):
    """Generate date range"""
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def find_file(base_path, date_str, prefix):
    """Find file for specified date"""
    filename = f"{prefix}_{date_str}.csv"
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

# ============================== Data Loading Functions ==============================
def read_pollution_day(date):
    """Read pollution data for a single day"""
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
        
        # Merge data
        df_poll = pd.concat([df_all, df_extra], ignore_index=True)
        
        # Convert to long format
        df_poll = df_poll.melt(id_vars=['date', 'hour', 'type'], 
                                var_name='station', value_name='value')
        df_poll['value'] = pd.to_numeric(df_poll['value'], errors='coerce')
        
        # Remove negative values and outliers
        df_poll = df_poll[df_poll['value'] >= 0]
        
        # Aggregate by date and type (average of all stations)
        df_daily = df_poll.groupby(['date', 'type'])['value'].mean().reset_index()
        
        # Convert to wide format
        df_daily = df_daily.pivot(index='date', columns='type', values='value')
        
        # Convert index to datetime format (key: ensure index is datetime type)
        df_daily.index = pd.to_datetime(df_daily.index, format='%Y%m%d', errors='coerce')
        
        # Keep only needed pollutants
        df_daily = df_daily[[col for col in pollutants if col in df_daily.columns]]
        
        return df_daily
    except Exception as e:
        # Silent error, don't print (avoid interfering with progress display)
        return None

def read_all_pollution():
    """Read all pollution data in parallel"""
    print("\n" + "=" * 80)
    print("Step 1: Load Pollution Data")
    print("=" * 80)
    print(f"\nUsing {MAX_WORKERS} parallel worker threads")
    
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
        print(f"\n  Successfully read {len(pollution_dfs)}/{len(dates)} days data")
        print("  Merging data...")
        df_poll_all = pd.concat(pollution_dfs)
        # Forward fill then mean fill
        df_poll_all.ffill(inplace=True)
        df_poll_all.fillna(df_poll_all.mean(), inplace=True)
        print(f"Pollution data loaded, shape: {df_poll_all.shape}")
        return df_poll_all
    return pd.DataFrame()

def read_era5_month(year, month):
    """Read single month ERA5 data - handles folder structure separated by variables"""
    month_str = f"{year}{month:02d}"
    
    # Find all files containing this month's data (from different variable folders)
    all_files = glob.glob(os.path.join(era5_path, "**", f"*{month_str}*.csv"), recursive=True)
    
    if not all_files:
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
            
            # Process time (don't convert to numeric)
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
                # Using join merge, keep all dates
                monthly_data = monthly_data.join(df_daily, how='outer')
            
            loaded_vars.extend(avail_vars)
            
        except Exception as e:
            continue
    
    if monthly_data is not None and not monthly_data.empty:
        return monthly_data
    else:
        return None

def read_all_era5():
    """Read all ERA5 data in parallel"""
    print("\n" + "=" * 80)
    print("Step 2: Load Meteorological Data")
    print("=" * 80)
    print(f"\nUsing {MAX_WORKERS} parallel worker threads")
    print(f"Meteorological data directory: {era5_path}")
    
    era5_dfs = []
    years = range(2015, 2025)
    months = range(1, 13)
    
    # Prepare all tasks
    month_tasks = [(year, month) for year in years for month in months 
                   if not (year == 2024 and month > 12)]
    total_months = len(month_tasks)
    print(f"Attempting to load {total_months} months of data...")
    
    # Using multiple parallel threads to load ERA5 data
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
        
        print(f"\n  Total successfully read: {successful_reads}/{len(futures)} months")
    
    if era5_dfs:
        print("\nMerging meteorological data...")
        df_era5_all = pd.concat(era5_dfs, axis=0)
        
        # Deduplicate (may have duplicate dates)
        print("  Deduplicating...")
        df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep='first')]
        
        # Sort
        print("  Sorting...")
        df_era5_all.sort_index(inplace=True)
        
        print(f"Shape after merge: {df_era5_all.shape}")
        print(f"Time range: {df_era5_all.index.min()} to {df_era5_all.index.max()}")
        
        # Fill missing values
        print("  Handling missing values...")
        initial_na = df_era5_all.isna().sum().sum()
        df_era5_all.ffill(inplace=True)
        df_era5_all.bfill(inplace=True)
        df_era5_all.fillna(df_era5_all.mean(), inplace=True)
        final_na = df_era5_all.isna().sum().sum()
        
        print(f"Missing value handling: {initial_na} -> {final_na}")
        print(f"Meteorological data loaded, shape: {df_era5_all.shape}")
        
        return df_era5_all
    else:
        print("\n❌ Error: No meteorological data files successfully loaded!")
        return pd.DataFrame()

# ============================== Load and Merge Data ==============================
df_pollution = read_all_pollution()
df_era5 = read_all_era5()

# Check data loading status
print("\n" + "=" * 80)
print("Step 3: Data Merge and Validation")
print("=" * 80)

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

print(f"\n  Pollution data time range: {df_pollution.index.min()} to {df_pollution.index.max()}")
print(f"  Meteorological data time range: {df_era5.index.min()} to {df_era5.index.max()}")

# Merge data
print("\nMerging data...")
df_combined = df_pollution.join(df_era5, how='inner')

if df_combined.empty:
    print("\n❌ Error: Data is empty after merge!")
    print("   Possible reason: Pollution data and meteorological data date indices have no intersection.")
    print(f"   Pollution data has {len(df_pollution)} rows")
    print(f"   Meteorological data has {len(df_era5)} rows")
    print(f"   After merge has {len(df_combined)} rows")
    import sys
    sys.exit(1)

print(f"Data shape after merge: {df_combined.shape}")
print(f"Time range: {df_combined.index.min().date()} to {df_combined.index.max().date()}")

# Create features
print("\nCreating features...")

# 1. Wind speed features
if 'u10' in df_combined.columns and 'v10' in df_combined.columns:
    df_combined['wind_speed_10m'] = np.sqrt(df_combined['u10']**2 + df_combined['v10']**2)
    df_combined['wind_dir_10m'] = np.arctan2(df_combined['v10'], df_combined['u10']) * 180 / np.pi
    df_combined['wind_dir_10m'] = (df_combined['wind_dir_10m'] + 360) % 360  # Convert to 0-360 degrees

if 'u100' in df_combined.columns and 'v100' in df_combined.columns:
    df_combined['wind_speed_100m'] = np.sqrt(df_combined['u100']**2 + df_combined['v100']**2)
    df_combined['wind_dir_100m'] = np.arctan2(df_combined['v100'], df_combined['u100']) * 180 / np.pi
    df_combined['wind_dir_100m'] = (df_combined['wind_dir_100m'] + 360) % 360

# 2. Time features
df_combined['year'] = df_combined.index.year
df_combined['month'] = df_combined.index.month
df_combined['day'] = df_combined.index.day
df_combined['day_of_year'] = df_combined.index.dayofyear
df_combined['day_of_week'] = df_combined.index.dayofweek
df_combined['week_of_year'] = df_combined.index.isocalendar().week

# Season features
df_combined['season'] = df_combined['month'].apply(
    lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
)

# Heating season (Beijing Nov 15 - Mar 15)
df_combined['is_heating_season'] = ((df_combined['month'] >= 11) | (df_combined['month'] <= 3)).astype(int)

# 3. Temperature-related features
if 't2m' in df_combined.columns and 'd2m' in df_combined.columns:
    # Temperature-dewpoint difference (reflects relative humidity)
    df_combined['temp_dewpoint_diff'] = df_combined['t2m'] - df_combined['d2m']

# 4. Relative humidity estimation (simplified formula)
if 't2m' in df_combined.columns and 'd2m' in df_combined.columns:
    # Magnus formula approximation
    df_combined['relative_humidity'] = 100 * np.exp((17.625 * (df_combined['d2m'] - 273.15)) / 
                                                      (243.04 + (df_combined['d2m'] - 273.15))) / \
                                        np.exp((17.625 * (df_combined['t2m'] - 273.15)) / 
                                               (243.04 + (df_combined['t2m'] - 273.15)))
    df_combined['relative_humidity'] = df_combined['relative_humidity'].clip(0, 100)

# Clean data
print("\nCleaning data...")
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows containing NaN
initial_rows = len(df_combined)
df_combined.dropna(inplace=True)
final_rows = len(df_combined)
print(f"Removed {initial_rows - final_rows} rows containing missing values")

print(f"\nFinal data shape: {df_combined.shape}")
print(f"Number of samples: {len(df_combined)}")

# PM2.5 statistics
if 'PM2.5' in df_combined.columns:
    print(f"\nPM2.5 Statistics:")
    print(f"  Mean: {df_combined['PM2.5'].mean():.2f} μg/m³")
    print(f"  Std Dev: {df_combined['PM2.5'].std():.2f} μg/m³")
    print(f"  Min: {df_combined['PM2.5'].min():.2f} μg/m³")
    print(f"  Max: {df_combined['PM2.5'].max():.2f} μg/m³")
    print(f"  Median: {df_combined['PM2.5'].median():.2f} μg/m³")

gc.collect()

# ============================== Feature Selection ==============================
print("\n" + "=" * 80)
print("Step 4: Feature Selection and Data Preparation")
print("=" * 80)

# Define target variable
target = 'PM2.5'

# Exclude columns (target variable, other pollutants, year, etc.)
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']

# Select features with strong correlation
selected_features = []

# Temperature features
if 't2m' in df_combined.columns:
    selected_features.append('t2m')
if 'd2m' in df_combined.columns:
    selected_features.append('d2m')
if 'temp_dewpoint_diff' in df_combined.columns:
    selected_features.append('temp_dewpoint_diff')

# Humidity features
if 'tcwv' in df_combined.columns:
    selected_features.append('tcwv')
if 'relative_humidity' in df_combined.columns:
    selected_features.append('relative_humidity')

# Wind speed features
if 'wind_speed_10m' in df_combined.columns:
    selected_features.append('wind_speed_10m')
if 'wind_speed_100m' in df_combined.columns:
    selected_features.append('wind_speed_100m')

# Other important meteorological factors
for feature in ['blh', 'tp', 'sp', 'str', 'tisr', 'avg_tprate']:
    if feature in df_combined.columns:
        selected_features.append(feature)

# Time features
for feature in ['month', 'season', 'day_of_year', 'day_of_week', 'is_heating_season']:
    if feature in df_combined.columns:
        selected_features.append(feature)

print(f"\nNumber of selected features: {len(selected_features)}")
print(f"Target variable: {target}")

# Display feature list
print(f"\nFeature list:")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i}. {feat}")

# Prepare modeling data
X = df_combined[selected_features].copy()
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
    print("1. Data path is incorrect, cannot find data files")
    print("2. Pollution data or meteorological data failed to load")
    print("3. After data merge, indices have no intersection (check if date ranges match)")
    print("4. All rows deleted during data cleaning process")
    import sys
    sys.exit(1)

print(f"\nPM2.5 Statistics:")
print(f"  Mean: {y.mean():.2f} μg/m³")
print(f"  Std Dev: {y.std():.2f} μg/m³")
print(f"  Min: {y.min():.2f} μg/m³")
print(f"  Max: {y.max():.2f} μg/m³")
print(f"  Median: {y.median():.2f} μg/m³")

# ============================== Dataset Split ==============================
print("\n" + "=" * 80)
print("Step 5: Dataset Split")
print("=" * 80)

# Split by time order: Training set 70%, Validation set 15%, Test set 15%
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

# ============================== Model Training ==============================
print("\n" + "=" * 80)
print("Step 6: Random Forest Model Training")
print("=" * 80)

# 6.1 Basic Random Forest model
print("\n6.1 Training basic Random Forest model...")
rf_basic = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("  Starting training...")
rf_basic.fit(X_train, y_train)
print("  ✓ Basic model training complete")

# Predict
y_train_pred_basic = rf_basic.predict(X_train)
y_val_pred_basic = rf_basic.predict(X_val)
y_test_pred_basic = rf_basic.predict(X_test)

# 6.2 Grid search optimized Random Forest
print("\n6.2 Grid search optimized Random Forest...")

# Using smaller search space for quick testing
param_grid_small = {
    'n_estimators': [100, 200],
    'max_depth': [20, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

print(f"  Parameter grid: {param_grid_small}")
print(f"  Total {int(np.prod([len(v) for v in param_grid_small.values()]))} parameter combinations")

rf_grid = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf_grid, 
    param_grid_small, 
    cv=3, 
    scoring='r2',
    verbose=1,
    n_jobs=-1
)

print("  Starting grid search...")
grid_search.fit(X_train, y_train)
rf_optimized = grid_search.best_estimator_

print(f"\n  ✓ Grid search complete")
print(f"  Best parameters: {grid_search.best_params_}")
print(f"  Best cross-validation R²: {grid_search.best_score_:.4f}")

# Predict
y_train_pred_opt = rf_optimized.predict(X_train)
y_val_pred_opt = rf_optimized.predict(X_val)
y_test_pred_opt = rf_optimized.predict(X_test)

# ============================== Model Evaluation ==============================
print("\n" + "=" * 80)
print("Step 7: Model Evaluation")
print("=" * 80)

def evaluate_model(y_true, y_pred, model_name, dataset):
    """Evaluate model performance"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'Model': model_name,
        'Dataset': dataset,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

# Evaluate all models
results = []
results.append(evaluate_model(y_train, y_train_pred_basic, 'RF_Basic', 'Train'))
results.append(evaluate_model(y_val, y_val_pred_basic, 'RF_Basic', 'Validation'))
results.append(evaluate_model(y_test, y_test_pred_basic, 'RF_Basic', 'Test'))
results.append(evaluate_model(y_train, y_train_pred_opt, 'RF_Optimized', 'Train'))
results.append(evaluate_model(y_val, y_val_pred_opt, 'RF_Optimized', 'Validation'))
results.append(evaluate_model(y_test, y_test_pred_opt, 'RF_Optimized', 'Test'))

results_df = pd.DataFrame(results)
print("\nModel performance comparison:")
print(results_df.to_string(index=False))

# Test set performance ranking
test_results = results_df[results_df['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\nTest set performance ranking:")
print(test_results.to_string(index=False))

# Performance improvement
basic_test_r2 = results_df[(results_df['Model'] == 'RF_Basic') & (results_df['Dataset'] == 'Test')]['R²'].values[0]
opt_test_r2 = results_df[(results_df['Model'] == 'RF_Optimized') & (results_df['Dataset'] == 'Test')]['R²'].values[0]
basic_test_rmse = results_df[(results_df['Model'] == 'RF_Basic') & (results_df['Dataset'] == 'Test')]['RMSE'].values[0]
opt_test_rmse = results_df[(results_df['Model'] == 'RF_Optimized') & (results_df['Dataset'] == 'Test')]['RMSE'].values[0]

if opt_test_r2 > basic_test_r2:
    r2_improvement = (opt_test_r2 - basic_test_r2) / abs(basic_test_r2) * 100
    print(f"\nOptimization Effect:")
    print(f"  R² improvement: {r2_improvement:.2f}%")
    
if opt_test_rmse < basic_test_rmse:
    rmse_improvement = (basic_test_rmse - opt_test_rmse) / basic_test_rmse * 100
    print(f"  RMSE reduction: {rmse_improvement:.2f}%")

# ============================== Feature Importance Analysis ==============================
print("\n" + "=" * 80)
print("Step 8: Feature Importance Analysis")
print("=" * 80)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance_Basic': rf_basic.feature_importances_,
    'Importance_Optimized': rf_optimized.feature_importances_
})

# Normalize importance (percentage)
feature_importance['Importance_Basic_Norm'] = (feature_importance['Importance_Basic'] / 
                                                feature_importance['Importance_Basic'].sum() * 100)
feature_importance['Importance_Optimized_Norm'] = (feature_importance['Importance_Optimized'] / 
                                                     feature_importance['Importance_Optimized'].sum() * 100)

feature_importance = feature_importance.sort_values('Importance_Optimized', ascending=False)

print(f"\nTop 15 important features (optimized model):")
print(feature_importance.head(15)[['Feature', 'Importance_Optimized_Norm']].to_string(index=False))

# ============================== Visualization ==============================
print("\n" + "=" * 80)
print("Step 9: Generate Visualization Charts")
print("=" * 80)

# 9.1 Prediction vs Actual Scatter plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

models_data = [
    ('RF_Basic', y_test_pred_basic, 'blue'),
    ('RF_Optimized', y_test_pred_opt, 'green')
]

for i, (name, pred, color) in enumerate(models_data):
    test_result = results_df[(results_df['Model'] == name) & 
                             (results_df['Dataset'] == 'Test')].iloc[0]
    
    axes[i].scatter(y_test, pred, alpha=0.5, s=30, color=color, edgecolors='black', linewidth=0.5)
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction Line')
    axes[i].set_xlabel('Actual PM2.5 Concentration (μg/m³)', fontsize=12)
    axes[i].set_ylabel('Predicted PM2.5 Concentration (μg/m³)', fontsize=12)
    axes[i].set_title(f'{name}\nR²={test_result["R²"]:.4f}, RMSE={test_result["RMSE"]:.2f}, MAE={test_result["MAE"]:.2f}', 
                      fontsize=13, fontweight='bold')
    axes[i].legend(fontsize=11)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'rf_prediction_scatter.png', dpi=300, bbox_inches='tight')
print("Saved: rf_prediction_scatter.png")
plt.close()

# 9.2 Time series comparison plot
fig, ax = plt.subplots(figsize=(18, 6))

plot_range = min(365, len(y_test))  # Display last year data
plot_idx = range(len(y_test) - plot_range, len(y_test))
time_idx = y_test.index[plot_idx]

ax.plot(time_idx, y_test.iloc[plot_idx], 'k-', label='Actual values', linewidth=2, alpha=0.8)
ax.plot(time_idx, y_test_pred_basic[plot_idx], '--', color='blue', 
        label='RF_Basic', linewidth=1.5, alpha=0.7)
ax.plot(time_idx, y_test_pred_opt[plot_idx], '--', color='green', 
        label='RF_Optimized', linewidth=1.5, alpha=0.7)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
ax.set_title('PM2.5 Concentration Prediction Time Series Comparison (Last year of test set)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / 'rf_timeseries.png', dpi=300, bbox_inches='tight')
print("Saved: rf_timeseries.png")
plt.close()

# 9.3 Residual analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, (name, pred, color) in enumerate(models_data):
    residuals = y_test - pred
    
    axes[i].scatter(pred, residuals, alpha=0.5, s=30, color=color, edgecolors='black', linewidth=0.5)
    axes[i].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[i].set_xlabel('Predicted values (μg/m³)', fontsize=12)
    axes[i].set_ylabel('Residuals (μg/m³)', fontsize=12)
    axes[i].set_title(f'{name} - Residual Analysis', fontsize=13, fontweight='bold')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'rf_residuals.png', dpi=300, bbox_inches='tight')
print("Saved: rf_residuals.png")
plt.close()

# 9.4 Feature importance plot
fig, ax = plt.subplots(figsize=(10, 8))

top_n = min(15, len(feature_importance))  # Display top 15 features
top_features = feature_importance.head(top_n)

y_pos = np.arange(len(top_features))
width = 0.35

ax.barh(y_pos - width/2, top_features['Importance_Basic_Norm'], width, 
        label='RF_Basic', color='blue', alpha=0.7)
ax.barh(y_pos + width/2, top_features['Importance_Optimized_Norm'], width, 
        label='RF_Optimized', color='green', alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_features['Feature'])
ax.set_xlabel('Feature Importance (%)', fontsize=12)
ax.set_title(f'Random Forest Feature Importance Comparison (Top {top_n})', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'rf_feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: rf_feature_importance.png")
plt.close()

# 9.5 Model performance comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

test_df = results_df[results_df['Dataset'] == 'Test']
models = test_df['Model'].tolist()
x_pos = np.arange(len(models))

colors = ['blue', 'green']

for i, metric in enumerate(['R²', 'RMSE', 'MAE']):
    axes[i].bar(x_pos, test_df[metric], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[i].set_xticks(x_pos)
    axes[i].set_xticklabels(['Basic', 'Optimized'], rotation=0, fontsize=11)
    axes[i].set_ylabel(metric, fontsize=12)
    
    if metric == 'R²':
        axes[i].set_title(f'{metric} Comparison\n(Higher is better)', fontsize=12, fontweight='bold')
    else:
        axes[i].set_title(f'{metric} Comparison\n(Lower is better)', fontsize=12, fontweight='bold')
    
    axes[i].grid(True, alpha=0.3, axis='y')
    
    # Display values on bar chart
    for j, v in enumerate(test_df[metric]):
        axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'rf_model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: rf_model_comparison.png")
plt.close()

# 9.6 Prediction error distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, (name, pred, color) in enumerate(models_data):
    errors = y_test - pred
    
    axes[i].hist(errors, bins=50, color=color, alpha=0.7, edgecolor='black')
    axes[i].axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Zero Error')
    axes[i].set_xlabel('Prediction Error (μg/m³)', fontsize=12)
    axes[i].set_ylabel('Frequency', fontsize=12)
    axes[i].set_title(f'{name} - Prediction Error Distribution\nMean={errors.mean():.2f}, Std Dev={errors.std():.2f}', 
                      fontsize=13, fontweight='bold')
    axes[i].legend(fontsize=11)
    axes[i].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'rf_error_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: rf_error_distribution.png")
plt.close()

# ============================== Save Results ==============================
print("\n" + "=" * 80)
print("Step 10: Save Results")
print("=" * 80)

# Save model performance
results_df.to_csv(output_dir / 'rf_model_performance.csv', index=False, encoding='utf-8-sig')
print("Saved: rf_model_performance.csv")

# Save feature importance
feature_importance.to_csv(output_dir / 'rf_feature_importance.csv', index=False, encoding='utf-8-sig')
print("Saved: rf_feature_importance.csv")

# Save prediction results
predictions_df = pd.DataFrame({
    'Date': y_test.index,
    'Actual_PM25': y_test.values,
    'Predicted_Basic': y_test_pred_basic,
    'Predicted_Optimized': y_test_pred_opt,
    'Error_Basic': y_test.values - y_test_pred_basic,
    'Error_Optimized': y_test.values - y_test_pred_opt
})
predictions_df.to_csv(output_dir / 'rf_predictions.csv', index=False, encoding='utf-8-sig')
print("Saved: rf_predictions.csv")

# Save best parameters
best_params_df = pd.DataFrame([grid_search.best_params_])
best_params_df.to_csv(output_dir / 'rf_best_parameters.csv', index=False, encoding='utf-8-sig')
print("Saved: rf_best_parameters.csv")

# Save model (using pickle)
import pickle
with open(model_dir / 'rf_optimized.pkl', 'wb') as f:
    pickle.dump(rf_optimized, f)
print("Saved: rf_optimized.pkl")

# ============================== Summary Report ==============================
print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)

print(f"\nOutput directory: {output_dir}")
print(f"Model directory: {model_dir}")

print("\nGenerated files:")
print("\nCSV Files:")
print("  - rf_model_performance.csv       Model performance metrics")
print("  - rf_feature_importance.csv      Feature importance")
print("  - rf_predictions.csv             Prediction results")
print("  - rf_best_parameters.csv         Best parameters")

print("\nChart Files:")
print("  - rf_prediction_scatter.png      Prediction vs Actual scatter plot")
print("  - rf_timeseries.png              Time series comparison")
print("  - rf_residuals.png               Residual analysis")
print("  - rf_feature_importance.png      Feature importance plot")
print("  - rf_model_comparison.png        Model performance comparison")
print("  - rf_error_distribution.png      Error distribution")

print("\nModel Files:")
print("  - rf_optimized.pkl               Random Forest optimized model")

# Output best model information
best_model = test_results.iloc[0]
print(f"\nBest model: {best_model['Model']}")
print(f"  R² Score: {best_model['R²']:.4f}")
print(f"  RMSE: {best_model['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model['MAPE']:.2f}%")

print(f"\nTop 5 most important features:")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance_Optimized_Norm']:.2f}%")

print(f"\nDataset information:")
print(f"  Training set samples: {len(X_train)}")
print(f"  Validation set samples: {len(X_val)}")
print(f"  Test set samples: {len(X_test)}")
print(f"  Number of features: {len(selected_features)}")

print("\n" + "=" * 80)
print("Random Forest PM2.5 Concentration Prediction Complete!")
print("=" * 80)

