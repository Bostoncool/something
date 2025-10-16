"""
Support Vector Regression (SVR) for PM2.5 Concentration Prediction
===================================================================
This script implements an SVR-based PM2.5 concentration prediction model
Using pollutant data and ERA5 meteorological data as features
"""

# Part 1: Import necessary libraries
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import warnings
import glob
import multiprocessing
from pathlib import Path
warnings.filterwarnings('ignore')

# Get CPU core count
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)  # Reserve 1 core for system

# Try importing tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm is not installed, progress display will use simplified version.")
    print("      You can use 'pip install tqdm' to install for better progress bar display.")

# Machine learning related imports
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set Chinese font display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("SVR PM2.5 Concentration Prediction Model")
print("=" * 80)

# Part 2: Define data paths
pollution_all_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)'
era5_path = r'C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV'

# Output path - Use relative path, create in current script directory
script_dir = Path(__file__).parent
output_dir = script_dir / 'output'
model_dir = script_dir / 'models'
output_dir.mkdir(exist_ok=True)
model_dir.mkdir(exist_ok=True)

# Define date range
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

# Define pollutants
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# Define ERA5 variables
era5_vars = [
    'd2m', 't2m', 'u10', 'v10', 'u100', 'v100', 'blh', 'cvh', 'lsm', 'cvl',
    'avg_tprate', 'mn2t', 'sd', 'str', 'sp', 'tisr', 'tcwv', 'tp'
]

# Beijing boundaries
beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

print(f"Data time range: {start_date.date()} to {end_date.date()}")
print(f"Pollutants: {', '.join(pollutants)}")
print(f"Number of meteorological variables: {len(era5_vars)}")
print(f"Output directory: {output_dir}")
print(f"Model save directory: {model_dir}")
print(f"CPU cores: {CPU_COUNT}, parallel worker threads: {MAX_WORKERS}")

# Part 3: Helper functions
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

# Part 4: Read pollution data
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
        
        # Filter 24-hour average and AQI
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
        
        # Calculate average by pollutant and date
        df_daily = df_poll.groupby(['date', 'type'])['value'].mean().reset_index()
        
        # Convert to wide format
        df_daily = df_daily.pivot(index='date', columns='type', values='value')
        
        # Convert index to datetime format - Critical fix
        df_daily.index = pd.to_datetime(df_daily.index, format='%Y%m%d', errors='coerce')
        
        # Keep only required pollutants
        available_pollutants = [p for p in pollutants if p in df_daily.columns]
        df_daily = df_daily[available_pollutants]
        
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

# Part 5: Read ERA5 data
def read_era5_month(year, month):
    """Read ERA5 data for a single month - Handle folder structure by variables"""
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
                # Use join to merge, keep all dates
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
    print("\nLoading meteorological data...")
    print(f"Using {MAX_WORKERS} parallel worker threads")
    print(f"Meteorological data directory: {era5_path}")
    print(f"Check if directory exists: {os.path.exists(era5_path)}")
    
    # First check what files are in the directory
    if os.path.exists(era5_path):
        all_csv = glob.glob(os.path.join(era5_path, "**", "*.csv"), recursive=True)
        print(f"Found {len(all_csv)} CSV files")
        if all_csv:
            print(f"Sample files: {[os.path.basename(f) for f in all_csv[:5]]}")
    
    era5_dfs = []
    years = range(2015, 2025)
    months = range(1, 13)
    
    # Prepare all tasks
    month_tasks = [(year, month) for year in years for month in months 
                   if not (year == 2024 and month > 12)]
    total_months = len(month_tasks)
    print(f"Attempting to load {total_months} months of data...")
    
    # Use more parallel threads to load ERA5 data
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_era5_month, year, month): (year, month) 
                  for year, month in month_tasks}
        
        successful_reads = 0
        if TQDM_AVAILABLE:
            # Using tqdm progress bar
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Loading meteorological data", unit="months"):
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
            
        print(f"  Total successful reads: {successful_reads}/{len(futures)} months")
    
    if era5_dfs:
        print("\nMerging meteorological data...")
        df_era5_all = pd.concat(era5_dfs, axis=0)
        
        # Remove duplicates (may have duplicate dates)
        print("  Deduplication processing...")
        df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep='first')]
        
        # Sort
        print("  Sorting processing...")
        df_era5_all.sort_index(inplace=True)
        
        print(f"Merged shape: {df_era5_all.shape}")
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
        print("\nError: No meteorological data files loaded successfully!")
        print("Possible causes:")
        print("1. File naming format does not match (Expected format: *YYYYMM*.csv)")
        print("2. File content format is incorrect (Missing time column)")
        print("3. File path is incorrect")
        return pd.DataFrame()

# Part 6: Data loading
df_pollution = read_all_pollution()
df_era5 = read_all_era5()
gc.collect()

# Check data loading status
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

# Part 7: Data merging and feature engineering
print("\n" + "=" * 80)
print("Data Merging and Feature Engineering")
print("=" * 80)

# Ensure index is datetime
df_pollution.index = pd.to_datetime(df_pollution.index)
df_era5.index = pd.to_datetime(df_era5.index)

print(f"  Pollution data time range: {df_pollution.index.min()} to {df_pollution.index.max()}")
print(f"  Meteorological data time range: {df_era5.index.min()} to {df_era5.index.max()}")

# Merge data
print("\nMerging data...")
df_combined = df_pollution.join(df_era5, how='inner')

if df_combined.empty:
    print("\nError: Data is empty after merging!")
    print("   Possible cause: Pollution data and meteorological data date indices have no overlap.")
    print(f"   Pollution data has {len(df_pollution)} rows")
    print(f"   Meteorological data has {len(df_era5)} rows")
    print(f"   After merging has {len(df_combined)} rows")
    import sys
    sys.exit(1)

# Feature engineering
# 1. Calculate wind speed
if 'u10' in df_combined and 'v10' in df_combined:
    df_combined['wind_speed_10m'] = np.sqrt(df_combined['u10']**2 + df_combined['v10']**2)
    df_combined['wind_dir_10m'] = np.arctan2(df_combined['v10'], df_combined['u10']) * 180 / np.pi

if 'u100' in df_combined and 'v100' in df_combined:
    df_combined['wind_speed_100m'] = np.sqrt(df_combined['u100']**2 + df_combined['v100']**2)

# 2. Temperature related features (convert to Celsius)
if 't2m' in df_combined:
    df_combined['t2m_celsius'] = df_combined['t2m'] - 273.15

if 'd2m' in df_combined:
    df_combined['d2m_celsius'] = df_combined['d2m'] - 273.15

# 3. Relative humidity (if dew point temperature available)
if 't2m' in df_combined and 'd2m' in df_combined:
    # Simplified relative humidity calculation
    df_combined['relative_humidity'] = 100 * np.exp((17.625 * df_combined['d2m_celsius']) / 
                                                      (243.04 + df_combined['d2m_celsius'])) / \
                                       np.exp((17.625 * df_combined['t2m_celsius']) / 
                                              (243.04 + df_combined['t2m_celsius']))

# 4. Time features
df_combined['year'] = df_combined.index.year
df_combined['month'] = df_combined.index.month
df_combined['day'] = df_combined.index.day
df_combined['dayofyear'] = df_combined.index.dayofyear
df_combined['season'] = df_combined['month'].apply(lambda x: (x % 12 + 3) // 3)
df_combined['is_winter'] = df_combined['month'].isin([12, 1, 2]).astype(int)

# 5. Cyclic features (using sine and cosine encoding)
df_combined['month_sin'] = np.sin(2 * np.pi * df_combined['month'] / 12)
df_combined['month_cos'] = np.cos(2 * np.pi * df_combined['month'] / 12)
df_combined['day_sin'] = np.sin(2 * np.pi * df_combined['dayofyear'] / 365)
df_combined['day_cos'] = np.cos(2 * np.pi * df_combined['dayofyear'] / 365)

# Clean data
print("\nCleaning data...")
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows containing NaN (mainly top few rows caused by lag features)
initial_rows = len(df_combined)
df_combined.dropna(inplace=True)
final_rows = len(df_combined)
print(f"Removed {initial_rows - final_rows} rows containing missing values")

print(f"\nMerged data shape: {df_combined.shape}")
print(f"Time range: {df_combined.index.min().date()} to {df_combined.index.max().date()}")
print(f"Number of samples: {len(df_combined)}")
print(f"Number of features: {len(df_combined.columns)}")
print(f"\nFirst 5 rows of data:")
print(df_combined.head())

# Part 8: Prepare training data
print("\n" + "=" * 80)
print("Preparing Training Data")
print("=" * 80)

# Check if PM2.5 exists
if 'PM2.5' not in df_combined.columns:
    raise ValueError("Data does not contain PM2.5 column!")

# Define features and target variable
target = 'PM2.5'
# Exclude other pollutants and year
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']
feature_cols = [col for col in df_combined.columns if col not in exclude_cols]

X = df_combined[feature_cols]
y = df_combined[target]

# ============================== Data validation ==============================
# Check if data is empty
if len(X) == 0 or len(y) == 0:
    print("\n" + "=" * 80)
    print("Error: No available data!")
    print("=" * 80)
    print("\nPossible causes:")
    print("1. Data path is incorrect, unable to find data files")
    print("2. Pollution data or meteorological data loading failed")
    print("3. After data merging, indices have no overlap (check if date ranges match)")
    print("4. Data cleaning process removed all rows")
    print("\nPlease check:")
    print(f"- Pollution data path: {pollution_all_path}")
    print(f"- Meteorological data path: {era5_path}")
    print(f"- Date range: {start_date.date()} to {end_date.date()}")
    import sys
    sys.exit(1)

print(f"Feature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")
print(f"Number of selected features: {len(feature_cols)}")
print(f"\nPM2.5 Statistics:")
print(f"  Mean: {y.mean():.2f} μg/m³")
print(f"  Std Dev: {y.std():.2f} μg/m³")
print(f"  Min: {y.min():.2f} μg/m³")
print(f"  Max: {y.max():.2f} μg/m³")
print(f"  Median: {y.median():.2f} μg/m³")

# Data split (70% training, 15% validation, 15% testing)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set size: {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

# Standardization
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

print("\nFeature standardization complete")

# Part 9: SVR model training
print("\n" + "=" * 80)
print("SVR Model Training")
print("=" * 80)

# Define SVR models with different kernel functions
models = {
    'SVR-RBF': SVR(kernel='rbf', cache_size=1000),
    'SVR-Linear': SVR(kernel='linear', cache_size=1000),
    'SVR-Poly': SVR(kernel='poly', degree=3, cache_size=1000)
}

# Grid search parameters
param_grids = {
    'SVR-RBF': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.001, 0.01, 0.1],
        'epsilon': [0.01, 0.1, 0.2]
    },
    'SVR-Linear': {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2]
    },
    'SVR-Poly': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1],
        'epsilon': [0.1, 0.2]
    }
}

# Train and evaluate each model
best_models = {}
results = []

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training model: {name}")
    print(f"{'='*60}")
    
    # Grid search
    print(f"Starting grid search...")
    grid_search = GridSearchCV(
        model, 
        param_grids[name], 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train_scaled)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score (negative MSE): {grid_search.best_score_:.4f}")
    
    # Use best model
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    
    # Evaluate on validation set
    y_val_pred_scaled = best_model.predict(X_val_scaled)
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
    
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\nValidation set performance:")
    print(f"  RMSE: {val_rmse:.4f}")
    print(f"  MAE: {val_mae:.4f}")
    print(f"  R²: {val_r2:.4f}")
    
    # Evaluate on test set
    y_test_pred_scaled = best_model.predict(X_test_scaled)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTest set performance:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  R²: {test_r2:.4f}")
    
    # Save results
    results.append({
        'Model': name,
        'Validation RMSE': val_rmse,
        'Validation MAE': val_mae,
        'Validation R²': val_r2,
        'Test RMSE': test_rmse,
        'Test MAE': test_mae,
        'Test R²': test_r2,
        'Best Parameters': grid_search.best_params_
    })
    
    gc.collect()

# Part 10: Results summary
print("\n" + "=" * 80)
print("Model Performance Comparison")
print("=" * 80)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(output_dir / 'svr_model_comparison.csv', 
                  index=False, encoding='utf-8-sig')

# Part 11: Select best model and perform detailed analysis
best_model_name = results_df.loc[results_df['Test R²'].idxmax(), 'Model']
best_model = best_models[best_model_name]

print(f"\nBest model: {best_model_name}")
print(f"Test set R²: {results_df.loc[results_df['Model']==best_model_name, 'Test R²'].values[0]:.4f}")

# Use best model for predictions
y_train_pred_scaled = best_model.predict(X_train_scaled)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()

y_test_pred_scaled = best_model.predict(X_test_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()

# Part 12: Visualize results
print("\n" + "=" * 80)
print("Generating Visualization Charts")
print("=" * 80)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Figure 1: Model performance comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# RMSE comparison
axes[0, 0].bar(results_df['Model'], results_df['Test RMSE'], color='steelblue', alpha=0.7)
axes[0, 0].set_title('Test Set RMSE Comparison', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('RMSE', fontsize=10)
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# MAE comparison
axes[0, 1].bar(results_df['Model'], results_df['Test MAE'], color='coral', alpha=0.7)
axes[0, 1].set_title('Test Set MAE Comparison', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('MAE', fontsize=10)
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# R² comparison
axes[1, 0].bar(results_df['Model'], results_df['Test R²'], color='seagreen', alpha=0.7)
axes[1, 0].set_title('Test Set R² Comparison', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('R²', fontsize=10)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)

# Training set vs Test set R²
x_pos = np.arange(len(results_df))
width = 0.35
axes[1, 1].bar(x_pos - width/2, results_df['Validation R²'], width, label='Validation Set', 
               color='skyblue', alpha=0.7)
axes[1, 1].bar(x_pos + width/2, results_df['Test R²'], width, label='Test Set', 
               color='lightcoral', alpha=0.7)
axes[1, 1].set_title('Validation and Test Set R² Comparison', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('R²', fontsize=10)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(results_df['Model'], rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: model_comparison.png")

# Figure 2: Best model prediction results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training set scatter plot
axes[0, 0].scatter(y_train, y_train_pred, alpha=0.3, s=10, color='blue')
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                'r--', lw=2, label='Ideal Prediction')
axes[0, 0].set_xlabel('Actual PM2.5 (μg/m³)', fontsize=10)
axes[0, 0].set_ylabel('Predicted PM2.5 (μg/m³)', fontsize=10)
axes[0, 0].set_title(f'Training Set Prediction Results ({best_model_name})', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Test set scatter plot
axes[0, 1].scatter(y_test, y_test_pred, alpha=0.5, s=20, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Ideal Prediction')
axes[0, 1].set_xlabel('Actual PM2.5 (μg/m³)', fontsize=10)
axes[0, 1].set_ylabel('Predicted PM2.5 (μg/m³)', fontsize=10)
axes[0, 1].set_title(f'Test Set Prediction Results (R²={test_r2:.4f})', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Residual plot
residuals = y_test - y_test_pred
axes[1, 0].scatter(y_test_pred, residuals, alpha=0.5, s=20, color='purple')
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted PM2.5 (μg/m³)', fontsize=10)
axes[1, 0].set_ylabel('Residuals (μg/m³)', fontsize=10)
axes[1, 0].set_title('Residual Distribution Plot', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Residual histogram
axes[1, 1].hist(residuals, bins=50, color='orange', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Residuals (μg/m³)', fontsize=10)
axes[1, 1].set_ylabel('Frequency', fontsize=10)
axes[1, 1].set_title(f'Residual Distribution Histogram (Mean={residuals.mean():.2f})', 
                     fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'prediction_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: prediction_results.png")

# Figure 3: Time series prediction comparison (test set data only)
fig, ax = plt.subplots(figsize=(16, 6))

# Select first 500 samples from test set for visualization
n_samples = min(500, len(y_test))
test_indices = y_test.index[:n_samples]

ax.plot(test_indices, y_test.iloc[:n_samples], 
        label='Actual Values', color='blue', linewidth=1.5, alpha=0.7)
ax.plot(test_indices, y_test_pred[:n_samples], 
        label='Predicted Values', color='red', linewidth=1.5, alpha=0.7)
ax.fill_between(test_indices, y_test.iloc[:n_samples], y_test_pred[:n_samples],
                alpha=0.2, color='gray')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
ax.set_title(f'PM2.5 Concentration Prediction Time Series Comparison ({best_model_name})', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / 'time_series_prediction.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: time_series_prediction.png")

# Part 13: Save model and scaler
print("\n" + "=" * 80)
print("Saving Models")
print("=" * 80)

# Save best model to models directory
joblib.dump(best_model, model_dir / f'{best_model_name}_best.pkl')
print(f"  Saved best model: {best_model_name}_best.pkl")

# Save all models to models directory
for name, model in best_models.items():
    joblib.dump(model, model_dir / f'{name}.pkl')
print(f"  Saved all models to: {model_dir}")

# Save scaler to models directory
joblib.dump(scaler_X, model_dir / 'scaler_X.pkl')
joblib.dump(scaler_y, model_dir / 'scaler_y.pkl')
print(f"  Saved scalers")

# Save feature names to output directory
feature_names_df = pd.DataFrame({'Feature': feature_cols})
feature_names_df.to_csv(output_dir / 'feature_names.csv', 
                        index=False, encoding='utf-8-sig')
print(f"  Saved feature names")

# Part 14: Generate detailed report
print("\n" + "=" * 80)
print("Generating Detailed Report")
print("=" * 80)

report_content = f"""
SVR PM2.5 Concentration Prediction Model Report
{'='*80}

1. Data Overview
   - Data time range: {start_date.date()} to {end_date.date()}
   - Total number of samples: {len(df_combined)}
   - Number of features: {len(feature_cols)}
   - Target variable: PM2.5 Concentration (μg/m³)

2. Dataset Split
   - Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)
   - Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)
   - Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)

3. PM2.5 Statistics
   - Mean: {y.mean():.2f} μg/m³
   - Std Dev: {y.std():.2f} μg/m³
   - Min: {y.min():.2f} μg/m³
   - Max: {y.max():.2f} μg/m³
   - Median: {y.median():.2f} μg/m³

4. Model Performance Comparison
"""

for _, row in results_df.iterrows():
    report_content += f"""
   {row['Model']}:
   - Test RMSE: {row['Test RMSE']:.4f}
   - Test MAE: {row['Test MAE']:.4f}
   - Test R²: {row['Test R²']:.4f}
   - Best Parameters: {row['Best Parameters']}
"""

report_content += f"""
5. Best Model
   - Model name: {best_model_name}
   - Test R²: {test_r2:.4f}
   - Test RMSE: {test_rmse:.4f}
   - Test MAE: {test_mae:.4f}

6. Feature Engineering
   - Original pollutant features: {', '.join(pollutants)}
   - ERA5 meteorological features: {', '.join([v for v in era5_vars if v in df_combined.columns])}
   - Derived features: Wind speed, wind direction, temperature conversion, relative humidity, time features, etc.

7. Model Description
   Support Vector Regression (SVR) is a powerful nonlinear regression method that maps features to
   high-dimensional space through kernel functions, effectively fitting complex nonlinear relationships
   between PM2.5 and meteorological factors. This study compared RBF, Linear and Poly kernel functions,
   and optimized hyperparameters through grid search.

8. File List
   - SVR_PM25_Prediction.py: Main program
   - svr_model_comparison.csv: Model comparison results
   - model_comparison.png: Model performance comparison chart
   - prediction_results.png: Detailed prediction result analysis chart
   - time_series_prediction.png: Time series prediction comparison chart
   - {best_model_name}_best.pkl: Best model file
   - scaler_X.pkl, scaler_y.pkl: Scalers
   - feature_names.csv: Feature name list

{'='*80}
Report generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Save report
with open(output_dir / 'model_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_content)

print(report_content)
print("\n  Saved detailed report: model_report.txt")

# Part 15: Clean memory and summary
print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)

print("\nGenerated files:")
print("\nCSV files (output directory):")
print("  - svr_model_comparison.csv    Model performance comparison")
print("  - feature_names.csv           Feature name list")

print("\nChart files (output directory):")
print("  - model_comparison.png        Model performance comparison chart")
print("  - prediction_results.png      Detailed prediction result analysis chart")
print("  - time_series_prediction.png  Time series prediction comparison chart")

print("\nModel files (models directory):")
print(f"  - {best_model_name}_best.pkl  Best model")
print("  - SVR-RBF.pkl, SVR-Linear.pkl, SVR-Poly.pkl  All SVR models")
print("  - scaler_X.pkl, scaler_y.pkl  Scalers")

print("\nReport files (output directory):")
print("  - model_report.txt            Detailed analysis report")

print(f"\nOutput directory: {output_dir}")
print(f"Model directory: {model_dir}")

del df_pollution, df_era5, df_combined, X_train_scaled, X_val_scaled, X_test_scaled
gc.collect()

print("\n" + "=" * 80)
print("SVR PM2.5 Concentration Prediction Complete!")
print("=" * 80)

