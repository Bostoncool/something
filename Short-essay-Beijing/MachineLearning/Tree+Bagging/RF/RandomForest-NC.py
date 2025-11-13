"""
Beijing PM2.5 Concentration Prediction - Random Forest Model (NetCDF ERA5)
------------------------------------------------------------------------------
This script replicates the Random Forest workflow using ERA5 meteorological
variables stored as NetCDF files. The pollution observations remain in CSV
format. The ERA5 loading logic follows the LightGBM-NC pipeline to ensure
consistent preprocessing across models.
"""

import os
import glob
import gc
import calendar
import warnings
import multiprocessing
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from netCDF4 import Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle

warnings.filterwarnings("ignore")

CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TQDM_AVAILABLE = False
    print("Note: tqdm is not installed, progress display will use simplified version.")
    print("      You can use 'pip install tqdm' to install for better progress bar display.")

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

np.random.seed(42)

print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - Random Forest Model (NetCDF ERA5)")
print("=" * 80)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
pollution_all_path = r'E:\DATA Science\Benchmark\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'E:\DATA Science\Benchmark\extra(SO2+NO2+CO+O3)'
era5_path = r'E:\DATA Science\ERA5-Beijing-NC'

output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)

model_dir = Path('./models')
model_dir.mkdir(exist_ok=True)

start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

era5_vars = [
    'd2m', 't2m', 'u10', 'v10', 'u100', 'v100',
    'blh', 'sp', 'tcwv',
    'tp', 'avg_tprate',
    'tisr', 'str',
    'cvh', 'cvl',
    'mn2t', 'sd', 'lsm'
]

print(f"\nConfiguration Parameters:")
print(f"Data time range: {start_date.date()} to {end_date.date()}")
print(f"Target variable: PM2.5 concentration")
print(f"Output directory: {output_dir}")
print(f"Model save directory: {model_dir}")
print(f"CPU cores: {CPU_COUNT}, Parallel worker threads: {MAX_WORKERS}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def daterange(start: datetime, end: datetime):
    """Yield each day between start and end (inclusive)."""
    for offset in range((end - start).days + 1):
        yield start + timedelta(days=offset)


def find_file(base_path: str, date_str: str, prefix: str):
    """Locate a CSV file following the naming convention."""
    filename = f"{prefix}_{date_str}.csv"
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


# ---------------------------------------------------------------------------
# Pollution data loading (CSV files remain unchanged)
# ---------------------------------------------------------------------------
def read_pollution_day(date: datetime):
    date_str = date.strftime('%Y%m%d')
    all_file = find_file(pollution_all_path, date_str, 'beijing_all')
    extra_file = find_file(pollution_extra_path, date_str, 'beijing_extra')

    if not all_file or not extra_file:
        return None

    try:
        df_all = pd.read_csv(all_file, encoding='utf-8', on_bad_lines='skip')
        df_extra = pd.read_csv(extra_file, encoding='utf-8', on_bad_lines='skip')

        df_all = df_all[~df_all['type'].str.contains('_24h|AQI', na=False)]
        df_extra = df_extra[~df_extra['type'].str.contains('_24h', na=False)]

        df_poll = pd.concat([df_all, df_extra], ignore_index=True)
        df_poll = df_poll.melt(
            id_vars=['date', 'hour', 'type'],
            var_name='station',
            value_name='value'
        )
        df_poll['value'] = pd.to_numeric(df_poll['value'], errors='coerce')
        df_poll = df_poll[df_poll['value'] >= 0]

        df_daily = (
            df_poll.groupby(['date', 'type'])['value']
            .mean()
            .reset_index()
            .pivot(index='date', columns='type', values='value')
        )
        df_daily.index = pd.to_datetime(df_daily.index, format='%Y%m%d', errors='coerce')
        df_daily = df_daily[[col for col in pollutants if col in df_daily.columns]]

        return df_daily
    except Exception:
        return None


def read_all_pollution():
    print("\n" + "=" * 80)
    print("Step 1: Load Pollution Data")
    print("=" * 80)
    print(f"\nUsing {MAX_WORKERS} parallel worker threads")

    dates = list(daterange(start_date, end_date))
    pollution_dfs = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, date): date for date in dates}

        if TQDM_AVAILABLE:
            iterator = tqdm(as_completed(futures), total=len(futures), desc="Loading pollution data", unit="day")
        else:
            iterator = as_completed(futures)

        for idx, future in enumerate(iterator, 1):
            result = future.result()
            if result is not None:
                pollution_dfs.append(result)
            if not TQDM_AVAILABLE and (idx % 500 == 0 or idx == len(futures)):
                print(f"  Processed {idx}/{len(futures)} days ({idx / len(futures) * 100:.1f}%)")

    if not pollution_dfs:
        return pd.DataFrame()

    print(f"\n  Successfully read {len(pollution_dfs)}/{len(dates)} days data")
    print("  Merging data...")
    df_poll_all = pd.concat(pollution_dfs)
    df_poll_all.ffill(inplace=True)
    df_poll_all.fillna(df_poll_all.mean(), inplace=True)
    print(f"Pollution data loaded, shape: {df_poll_all.shape}")
    return df_poll_all


# ---------------------------------------------------------------------------
# ERA5 data loading (NetCDF)
# ---------------------------------------------------------------------------
def read_era5_month(year: int, month: int):
    """Read and aggregate ERA5 data for a single month using NetCDF files."""
    month_str = f"{year}{month:02d}"
    all_files = glob.glob(os.path.join(era5_path, "**", f"*{month_str}*.nc"), recursive=True)

    fallback_used = False
    if not all_files:
        all_files = glob.glob(os.path.join(era5_path, "**", "*.nc"), recursive=True)
        fallback_used = True
        if not all_files:
            return None

    monthly_datasets = []

    month_start = pd.to_datetime(f"{year}-{month:02d}-01")
    month_end = month_start + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    for file_path in all_files:
        try:
            with Dataset(file_path, mode='r') as nc_file:
                available_vars = [var for var in era5_vars if var in nc_file.variables]
            if not available_vars:
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
                    continue
                ds_subset = ds_subset.sortby('time')

                if fallback_used:
                    try:
                        ds_subset = ds_subset.sel(time=slice(month_start, month_end))
                    except Exception:
                        continue
                    if ds_subset.sizes.get('time', 0) == 0:
                        continue

                if {'latitude', 'longitude'} <= set(ds_subset.coords):
                    lat_values = ds_subset['latitude']
                    if lat_values[0] > lat_values[-1]:
                        lat_slice = slice(beijing_lats.max(), beijing_lats.min())
                    else:
                        lat_slice = slice(beijing_lats.min(), beijing_lats.max())
                    ds_subset = ds_subset.sel(
                        latitude=lat_slice,
                        longitude=slice(beijing_lons.min(), beijing_lons.max())
                    )
                    if {'latitude', 'longitude'} <= set(ds_subset.dims):
                        ds_subset = ds_subset.mean(dim=['latitude', 'longitude'], skipna=True)

                ds_daily = ds_subset.resample(time='1D').mean(keep_attrs=False)
                ds_daily = ds_daily.dropna('time', how='all')
                if ds_daily.sizes.get('time', 0) == 0:
                    continue

                monthly_datasets.append(ds_daily.load())
        except Exception:
            continue

    if not monthly_datasets:
        return None

    merged_ds = xr.merge(monthly_datasets, compat='override', join='outer')
    df_month = merged_ds.to_dataframe()
    df_month.index = pd.to_datetime(df_month.index)
    df_month = df_month.groupby(df_month.index).mean()
    if df_month.empty:
        return None

    return df_month


def read_all_era5():
    print("\n" + "=" * 80)
    print("Step 2: Load Meteorological Data (NetCDF)")
    print("=" * 80)
    print(f"\nUsing {MAX_WORKERS} parallel worker threads")
    print(f"Meteorological data directory: {era5_path}")
    print(f"Directory exists: {os.path.exists(era5_path)}")

    era5_dfs = []
    years = range(2015, 2025)
    months = range(1, 13)

    month_tasks = [(year, month) for year in years for month in months if not (year == 2024 and month > 12)]
    print(f"Attempting to load {len(month_tasks)} months of data...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_era5_month, year, month): (year, month) for year, month in month_tasks}

        if TQDM_AVAILABLE:
            iterator = tqdm(as_completed(futures), total=len(futures), desc="Loading meteorological data", unit="month")
        else:
            iterator = as_completed(futures)

        successful_reads = 0
        for idx, future in enumerate(iterator, 1):
            result = future.result()
            if result is not None and not result.empty:
                era5_dfs.append(result)
                successful_reads += 1
            if not TQDM_AVAILABLE and (idx % 20 == 0 or idx == len(futures)):
                print(f"  Progress: {idx}/{len(futures)} months (Success: {successful_reads}, {idx/len(futures)*100:.1f}%)")

        print(f"  Total successfully read: {successful_reads}/{len(futures)} months")

    if not era5_dfs:
        print("\n❌ Error: No meteorological data files loaded successfully!")
        return pd.DataFrame()

    print("\nMerging meteorological data...")
    df_era5_all = pd.concat(era5_dfs, axis=0)
    df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep='first')]
    df_era5_all.sort_index(inplace=True)

    print(f"Shape after merge: {df_era5_all.shape}")
    print(f"Time range: {df_era5_all.index.min()} to {df_era5_all.index.max()}")

    print("  Handling missing values...")
    initial_na = df_era5_all.isna().sum().sum()
    df_era5_all.ffill(inplace=True)
    df_era5_all.bfill(inplace=True)
    df_era5_all.fillna(df_era5_all.mean(), inplace=True)
    final_na = df_era5_all.isna().sum().sum()
    print(f"Missing value handling: {initial_na} -> {final_na}")

    print(f"Meteorological data loaded, shape: {df_era5_all.shape}")
    return df_era5_all


# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------
df_pollution = read_all_pollution()
df_era5 = read_all_era5()

print("\n" + "=" * 80)
print("Step 3: Data Merge and Validation")
print("=" * 80)

print("\nData loading check:")
print(f"  Pollution data shape: {df_pollution.shape}")
print(f"  Meteorological data shape: {df_era5.shape}")

if df_pollution.empty:
    print("\n⚠️ Warning: Pollution data is empty! Please check data path and files.")
    raise SystemExit(1)

if df_era5.empty:
    print("\n⚠️ Warning: Meteorological data is empty! Please check data path and files.")
    raise SystemExit(1)

df_pollution.index = pd.to_datetime(df_pollution.index)
df_era5.index = pd.to_datetime(df_era5.index)

print(f"\n  Pollution data time range: {df_pollution.index.min()} to {df_pollution.index.max()}")
print(f"  Meteorological data time range: {df_era5.index.min()} to {df_era5.index.max()}")

print("\nMerging data...")
df_combined = df_pollution.join(df_era5, how='inner')

if df_combined.empty:
    print("\n❌ Error: Data is empty after merge! No overlapping indices.")
    raise SystemExit(1)

print(f"Data shape after merge: {df_combined.shape}")
print(f"Time range: {df_combined.index.min().date()} to {df_combined.index.max().date()}")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
print("\nCreating features...")

if {'u10', 'v10'} <= set(df_combined.columns):
    df_combined['wind_speed_10m'] = np.sqrt(df_combined['u10'] ** 2 + df_combined['v10'] ** 2)
    df_combined['wind_dir_10m'] = np.degrees(np.arctan2(df_combined['v10'], df_combined['u10']))
    df_combined['wind_dir_10m'] = (df_combined['wind_dir_10m'] + 360) % 360

if {'u100', 'v100'} <= set(df_combined.columns):
    df_combined['wind_speed_100m'] = np.sqrt(df_combined['u100'] ** 2 + df_combined['v100'] ** 2)
    df_combined['wind_dir_100m'] = np.degrees(np.arctan2(df_combined['v100'], df_combined['u100']))
    df_combined['wind_dir_100m'] = (df_combined['wind_dir_100m'] + 360) % 360

index_calendar = df_combined.index.isocalendar()
df_combined['year'] = df_combined.index.year
df_combined['month'] = df_combined.index.month
df_combined['day'] = df_combined.index.day
df_combined['day_of_year'] = df_combined.index.dayofyear
df_combined['day_of_week'] = df_combined.index.dayofweek
df_combined['week_of_year'] = index_calendar.week.astype(int)

df_combined['season'] = df_combined['month'].apply(
    lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
)

df_combined['is_heating_season'] = ((df_combined['month'] >= 11) | (df_combined['month'] <= 3)).astype(int)

if {'t2m', 'd2m'} <= set(df_combined.columns):
    df_combined['temp_dewpoint_diff'] = df_combined['t2m'] - df_combined['d2m']
    df_combined['relative_humidity'] = 100 * np.exp((17.625 * (df_combined['d2m'] - 273.15)) /
                                                     (243.04 + (df_combined['d2m'] - 273.15))) / \
                                        np.exp((17.625 * (df_combined['t2m'] - 273.15)) /
                                               (243.04 + (df_combined['t2m'] - 273.15)))
    df_combined['relative_humidity'] = df_combined['relative_humidity'].clip(0, 100)

print("\nCleaning data...")
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)

initial_rows = len(df_combined)
df_combined.dropna(inplace=True)
final_rows = len(df_combined)
print(f"Removed {initial_rows - final_rows} rows containing missing values")

print(f"\nFinal data shape: {df_combined.shape}")
print(f"Number of samples: {len(df_combined)}")

if 'PM2.5' in df_combined.columns:
    print(f"\nPM2.5 Statistics:")
    print(f"  Mean: {df_combined['PM2.5'].mean():.2f} μg/m³")
    print(f"  Std Dev: {df_combined['PM2.5'].std():.2f} μg/m³")
    print(f"  Min: {df_combined['PM2.5'].min():.2f} μg/m³")
    print(f"  Max: {df_combined['PM2.5'].max():.2f} μg/m³")
    print(f"  Median: {df_combined['PM2.5'].median():.2f} μg/m³")

gc.collect()

print("\n" + "=" * 80)
print("Step 4: Feature Selection and Data Preparation")
print("=" * 80)

target = 'PM2.5'
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']

selected_features = []
for feature in ['t2m', 'd2m', 'temp_dewpoint_diff', 'tcwv', 'relative_humidity',
                'wind_speed_10m', 'wind_speed_100m', 'blh', 'tp', 'sp', 'str',
                'tisr', 'avg_tprate', 'month', 'season', 'day_of_year',
                'day_of_week', 'is_heating_season']:
    if feature in df_combined.columns and feature not in selected_features:
        selected_features.append(feature)

print(f"\nNumber of selected features: {len(selected_features)}")
print(f"Target variable: {target}")

print(f"\nFeature list:")
for idx, feat in enumerate(selected_features, 1):
    print(f"  {idx}. {feat}")

X = df_combined[selected_features].copy()
y = df_combined[target].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

if len(X) == 0 or len(y) == 0:
    print("\n❌ Error: No available data after preprocessing!")
    raise SystemExit(1)

print(f"\nPM2.5 Statistics:")
print(f"  Mean: {y.mean():.2f} μg/m³")
print(f"  Std Dev: {y.std():.2f} μg/m³")
print(f"  Min: {y.min():.2f} μg/m³")
print(f"  Max: {y.max():.2f} μg/m³")
print(f"  Median: {y.median():.2f} μg/m³")

print("\n" + "=" * 80)
print("Step 5: Dataset Split")
print("=" * 80)

n_samples = len(X)
train_size = int(n_samples * 0.70)
val_size = int(n_samples * 0.15)

X_train = X.iloc[:train_size]
X_val = X.iloc[train_size:train_size + val_size]
X_test = X.iloc[train_size + val_size:]

y_train = y.iloc[:train_size]
y_val = y.iloc[train_size:train_size + val_size]
y_test = y.iloc[train_size + val_size:]

print(f"\nTraining set: {len(X_train)} samples ({len(X_train) / n_samples * 100:.1f}%)")
print(f"  Time range: {X_train.index.min().date()} to {X_train.index.max().date()}")
print(f"  PM2.5: {y_train.mean():.2f} ± {y_train.std():.2f} μg/m³")

print(f"\nValidation set: {len(X_val)} samples ({len(X_val) / n_samples * 100:.1f}%)")
print(f"  Time range: {X_val.index.min().date()} to {X_val.index.max().date()}")
print(f"  PM2.5: {y_val.mean():.2f} ± {y_val.std():.2f} μg/m³")

print(f"\nTest set: {len(X_test)} samples ({len(X_test) / n_samples * 100:.1f}%)")
print(f"  Time range: {X_test.index.min().date()} to {X_test.index.max().date()}")
print(f"  PM2.5: {y_test.mean():.2f} ± {y_test.std():.2f} μg/m³")

print("\n" + "=" * 80)
print("Step 6: Random Forest Model Training")
print("=" * 80)

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

y_train_pred_basic = rf_basic.predict(X_train)
y_val_pred_basic = rf_basic.predict(X_val)
y_test_pred_basic = rf_basic.predict(X_test)

print("\n6.2 Grid search optimized Random Forest (NetCDF features)...")
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

y_train_pred_opt = rf_optimized.predict(X_train)
y_val_pred_opt = rf_optimized.predict(X_val)
y_test_pred_opt = rf_optimized.predict(X_test)

print("\n" + "=" * 80)
print("Step 7: Model Evaluation")
print("=" * 80)


def evaluate_model(y_true, y_pred, model_name, dataset):
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


results = [
    evaluate_model(y_train, y_train_pred_basic, 'RF_Basic', 'Train'),
    evaluate_model(y_val, y_val_pred_basic, 'RF_Basic', 'Validation'),
    evaluate_model(y_test, y_test_pred_basic, 'RF_Basic', 'Test'),
    evaluate_model(y_train, y_train_pred_opt, 'RF_Optimized', 'Train'),
    evaluate_model(y_val, y_val_pred_opt, 'RF_Optimized', 'Validation'),
    evaluate_model(y_test, y_test_pred_opt, 'RF_Optimized', 'Test'),
]

results_df = pd.DataFrame(results)
print("\nModel performance comparison:")
print(results_df.to_string(index=False))

test_results = results_df[results_df['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\nTest set performance ranking:")
print(test_results.to_string(index=False))

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

print("\n" + "=" * 80)
print("Step 8: Feature Importance Analysis")
print("=" * 80)

feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance_Basic': rf_basic.feature_importances_,
    'Importance_Optimized': rf_optimized.feature_importances_
})

feature_importance['Importance_Basic_Norm'] = (
    feature_importance['Importance_Basic'] / feature_importance['Importance_Basic'].sum() * 100
)
feature_importance['Importance_Optimized_Norm'] = (
    feature_importance['Importance_Optimized'] / feature_importance['Importance_Optimized'].sum() * 100
)

feature_importance = feature_importance.sort_values('Importance_Optimized', ascending=False)

print(f"\nTop 15 important features (optimized model):")
print(feature_importance.head(15)[['Feature', 'Importance_Optimized_Norm']].to_string(index=False))

print("\n" + "=" * 80)
print("Step 9: Generate Visualization Charts")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
models_data = [
    ('RF_Basic', y_test_pred_basic, 'blue'),
    ('RF_Optimized', y_test_pred_opt, 'green')
]

for idx, (name, pred, color) in enumerate(models_data):
    test_result = results_df[(results_df['Model'] == name) & (results_df['Dataset'] == 'Test')].iloc[0]
    axes[idx].scatter(y_test, pred, alpha=0.5, s=30, color=color, edgecolors='black', linewidth=0.5)
    axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction Line')
    axes[idx].set_xlabel('Actual PM2.5 Concentration (μg/m³)', fontsize=12)
    axes[idx].set_ylabel('Predicted PM2.5 Concentration (μg/m³)', fontsize=12)
    axes[idx].set_title(
        f"{name}\nR²={test_result['R²']:.4f}, RMSE={test_result['RMSE']:.2f}, MAE={test_result['MAE']:.2f}",
        fontsize=13,
        fontweight='bold'
    )
    axes[idx].legend(fontsize=11)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'rf_prediction_scatter_nc.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(18, 6))
plot_range = min(365, len(y_test))
plot_idx = range(len(y_test) - plot_range, len(y_test))
time_idx = y_test.index[plot_idx]

ax.plot(time_idx, y_test.iloc[plot_idx], 'k-', label='Actual values', linewidth=2, alpha=0.8)
ax.plot(time_idx, y_test_pred_basic[plot_idx], '--', color='blue', label='RF_Basic', linewidth=1.5, alpha=0.7)
ax.plot(time_idx, y_test_pred_opt[plot_idx], '--', color='green', label='RF_Optimized', linewidth=1.5, alpha=0.7)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
ax.set_title('PM2.5 Concentration Prediction Time Series Comparison (Last year of test set)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / 'rf_timeseries_nc.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, (name, pred, color) in enumerate(models_data):
    residuals = y_test - pred
    axes[idx].scatter(pred, residuals, alpha=0.5, s=30, color=color, edgecolors='black', linewidth=0.5)
    axes[idx].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[idx].set_xlabel('Predicted values (μg/m³)', fontsize=12)
    axes[idx].set_ylabel('Residuals (μg/m³)', fontsize=12)
    axes[idx].set_title(f'{name} - Residual Analysis', fontsize=13, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'rf_residuals_nc.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
top_n = min(15, len(feature_importance))
top_features = feature_importance.head(top_n)
y_pos = np.arange(len(top_features))
width = 0.35

ax.barh(y_pos - width / 2, top_features['Importance_Basic_Norm'], width, label='RF_Basic', color='blue', alpha=0.7)
ax.barh(y_pos + width / 2, top_features['Importance_Optimized_Norm'], width, label='RF_Optimized', color='green', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_features['Feature'])
ax.set_xlabel('Feature Importance (%)', fontsize=12)
ax.set_title(f'Random Forest Feature Importance Comparison (Top {top_n})', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'rf_feature_importance_nc.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['blue', 'green']

for idx, metric in enumerate(['R²', 'RMSE', 'MAE']):
    axes[idx].bar(range(len(test_results)), test_results[metric], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[idx].set_xticks(range(len(test_results)))
    axes[idx].set_xticklabels(['Basic', 'Optimized'], fontsize=11)
    axes[idx].set_ylabel(metric, fontsize=12)
    axes[idx].set_title(f"{metric} Comparison\n{'Higher' if metric == 'R²' else 'Lower'} is better", fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3, axis='y')
    for j, value in enumerate(test_results[metric]):
        axes[idx].text(j, value, f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'rf_model_comparison_nc.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, (name, pred, color) in enumerate(models_data):
    errors = y_test - pred
    axes[idx].hist(errors, bins=50, color=color, alpha=0.7, edgecolor='black')
    axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='Zero Error')
    axes[idx].set_xlabel('Prediction Error (μg/m³)', fontsize=12)
    axes[idx].set_ylabel('Frequency', fontsize=12)
    axes[idx].set_title(
        f"{name} - Prediction Error Distribution\nMean={errors.mean():.2f}, Std Dev={errors.std():.2f}",
        fontsize=13,
        fontweight='bold'
    )
    axes[idx].legend(fontsize=11)
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'rf_error_distribution_nc.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 80)
print("Step 10: Save Results")
print("=" * 80)

results_df.to_csv(output_dir / 'rf_model_performance_nc.csv', index=False, encoding='utf-8-sig')
feature_importance.to_csv(output_dir / 'rf_feature_importance_nc.csv', index=False, encoding='utf-8-sig')

predictions_df = pd.DataFrame({
    'Date': y_test.index,
    'Actual_PM25': y_test.values,
    'Predicted_Basic': y_test_pred_basic,
    'Predicted_Optimized': y_test_pred_opt,
    'Error_Basic': y_test.values - y_test_pred_basic,
    'Error_Optimized': y_test.values - y_test_pred_opt
})
predictions_df.to_csv(output_dir / 'rf_predictions_nc.csv', index=False, encoding='utf-8-sig')

best_params_df = pd.DataFrame([grid_search.best_params_])
best_params_df.to_csv(output_dir / 'rf_best_parameters_nc.csv', index=False, encoding='utf-8-sig')

with open(model_dir / 'rf_optimized_nc.pkl', 'wb') as f:
    pickle.dump(rf_optimized, f)

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)

print(f"\nOutput directory: {output_dir}")
print(f"Model directory: {model_dir}")

print("\nGenerated files:")
print("\nCSV Files:")
print("  - rf_model_performance_nc.csv       Model performance metrics")
print("  - rf_feature_importance_nc.csv      Feature importance")
print("  - rf_predictions_nc.csv             Prediction results")
print("  - rf_best_parameters_nc.csv         Best parameters")

print("\nChart Files:")
print("  - rf_prediction_scatter_nc.png      Prediction vs Actual scatter plot")
print("  - rf_timeseries_nc.png              Time series comparison")
print("  - rf_residuals_nc.png               Residual analysis")
print("  - rf_feature_importance_nc.png      Feature importance plot")
print("  - rf_model_comparison_nc.png        Model performance comparison")
print("  - rf_error_distribution_nc.png      Error distribution")

print("\nModel Files:")
print("  - rf_optimized_nc.pkl               Random Forest optimized model")

best_model = test_results.iloc[0]
print(f"\nBest model: {best_model['Model']}")
print(f"  R² Score: {best_model['R²']:.4f}")
print(f"  RMSE: {best_model['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model['MAPE']:.2f}%")

print(f"\nTop 5 most important features:")
for _, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance_Optimized_Norm']:.2f}%")

print(f"\nDataset information:")
print(f"  Training set samples: {len(X_train)}")
print(f"  Validation set samples: {len(X_val)}")
print(f"  Test set samples: {len(X_test)}")
print(f"  Number of features: {len(selected_features)}")

print("\n" + "=" * 80)
print("Random Forest PM2.5 Concentration Prediction Complete (NetCDF ERA5)!")
print("=" * 80)
