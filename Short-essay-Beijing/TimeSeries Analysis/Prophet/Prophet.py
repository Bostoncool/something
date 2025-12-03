import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import glob
import multiprocessing

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import xarray as xr
from netCDF4 import Dataset as NetCDFDataset

plt.rcParams['figure.dpi'] = 100

warnings.filterwarnings('ignore')

CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm not installed, using simplified progress display.")

print("=" * 80)
print("Beijing PM2.5 Prediction - Prophet Time Series Model")
print("=" * 80)

print("\nConfiguring parameters...")

pollution_all_path = r'/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)'
pollution_extra_path = r'/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)'
era5_path = r'/root/autodl-tmp/ERA5-Beijing-NC'

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
    't2m', 'd2m', 'u10', 'v10', 'u100', 'v100',
    'blh', 'sp', 'tcwv', 'tp', 'str', 'lsm'
]

PROPHET_PARAMS = {
    'seasonality_mode': 'multiplicative',
    'yearly_seasonality': True,
    'weekly_seasonality': True,
    'daily_seasonality': True,
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0,
    'holidays_prior_scale': 10.0,
}

CROSS_VALIDATION_PARAMS = {
    'initial': '730 days',
    'period': '180 days',
    'horizon': '365 days'
}

print(f"Data time range: {start_date.date()} to {end_date.date()}")
print(f"Target variable: PM2.5 concentration")
print(f"Output directory: {output_dir}")
print(f"Model save directory: {model_dir}")
print(f"CPU cores: {CPU_COUNT}, Parallel workers: {MAX_WORKERS}")
def daterange(start, end):
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def find_file(base_path, date_str, prefix):
    filename = f"{prefix}_{date_str}.csv"
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def read_pollution_day(date, pollution_all_path_local=None, pollution_extra_path_local=None,
                      pollutants_list=None, file_map_all=None, file_map_extra=None):
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
    file_map = {}
    filename_pattern = f"{prefix}_"
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename.startswith(filename_pattern) and filename.endswith('.csv'):
                file_map[filename] = os.path.join(root, filename)
    return file_map

def read_all_pollution():
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
        print(f"  Successfully read {len(pollution_dfs)}/{len(dates)} days of valid data")
        print("  Merging data...")
        df_poll_all = pd.concat(pollution_dfs)
        df_poll_all.ffill(inplace=True)
        df_poll_all.fillna(df_poll_all.mean(), inplace=True)
        print(f"Pollution data loaded, shape: {df_poll_all.shape}")
        return df_poll_all
    return pd.DataFrame()

def read_single_nc_file(file_path, era5_vars_list=None, beijing_lats_array=None, beijing_lons_array=None):
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
        futures = {executor.submit(read_single_nc_file, file_path, era5_vars_list, beijing_lats_array, beijing_lons_array):
                  file_path for file_path in all_nc_files}

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
                    print(f"  Progress: {i}/{len(futures)} files (Success: {successful_reads}, Failed: {failed_reads}, {i/len(futures)*100:.1f}%)")

    print(f"Reading complete: {successful_reads} successful, {failed_reads} failed")
    print(f"Found {len(file_results)} unique variables")

    if not file_results:
        print("Error: Failed to read any data!")
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
    print(f"Meteorological data loaded, shape: {df_era5_all.shape}")

    return df_era5_all
def create_features(df):
    df_copy = df.copy()

    if 'u10' in df_copy and 'v10' in df_copy:
        df_copy['wind_speed_10m'] = np.sqrt(df_copy['u10']**2 + df_copy['v10']**2)
        df_copy['wind_dir_10m'] = np.arctan2(df_copy['v10'], df_copy['u10']) * 180 / np.pi
        df_copy['wind_dir_10m'] = (df_copy['wind_dir_10m'] + 360) % 360

    if 'u100' in df_copy and 'v100' in df_copy:
        df_copy['wind_speed_100m'] = np.sqrt(df_copy['u100']**2 + df_copy['v100']**2)
        df_copy['wind_dir_100m'] = np.arctan2(df_copy['v100'], df_copy['u100']) * 180 / np.pi
        df_copy['wind_dir_100m'] = (df_copy['wind_dir_100m'] + 360) % 360

    df_copy['year'] = df_copy.index.year
    df_copy['month'] = df_copy.index.month
    df_copy['day'] = df_copy.index.day
    df_copy['day_of_year'] = df_copy.index.dayofyear
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['week_of_year'] = df_copy.index.isocalendar().week

    df_copy['season'] = df_copy['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )

    df_copy['is_heating_season'] = ((df_copy['month'] >= 11) | (df_copy['month'] <= 3)).astype(int)

    if 't2m' in df_copy and 'd2m' in df_copy:
        df_copy['temp_dewpoint_diff'] = df_copy['t2m'] - df_copy['d2m']

    if 'PM2.5' in df_copy:
        df_copy['PM2.5_lag1'] = df_copy['PM2.5'].shift(1)
        df_copy['PM2.5_lag3'] = df_copy['PM2.5'].shift(3)
        df_copy['PM2.5_lag7'] = df_copy['PM2.5'].shift(7)
        df_copy['PM2.5_ma3'] = df_copy['PM2.5'].rolling(window=3, min_periods=1).mean()
        df_copy['PM2.5_ma7'] = df_copy['PM2.5'].rolling(window=7, min_periods=1).mean()

    if 't2m' in df_copy and 'd2m' in df_copy:
        df_copy['relative_humidity'] = 100 * np.exp((17.625 * (df_copy['d2m'] - 273.15)) /
                                                      (243.04 + (df_copy['d2m'] - 273.15))) / \
                                        np.exp((17.625 * (df_copy['t2m'] - 273.15)) /
                                               (243.04 + (df_copy['t2m'] - 273.15)))
        df_copy['relative_humidity'] = df_copy['relative_humidity'].clip(0, 100)

    if 'wind_dir_10m' in df_copy:
        df_copy['wind_dir_category'] = pd.cut(df_copy['wind_dir_10m'],
                                                bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                                                labels=[0, 1, 2, 3, 4, 5, 6, 7],
                                                include_lowest=True).astype(int)

    return df_copy

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

df_pollution.index = pd.to_datetime(df_pollution.index)
df_era5.index = pd.to_datetime(df_era5.index)

print(f"  Pollution data time range: {df_pollution.index.min()} to {df_pollution.index.max()}")
print(f"  Meteorological data time range: {df_era5.index.min()} to {df_era5.index.max()}")

print("\nMerging data...")
df_combined = df_pollution.join(df_era5, how='inner')

if df_combined.empty:
    print("\nError: Merged data is empty!")
    sys.exit(1)

print("\nCreating features...")
df_combined = create_features(df_combined)

print("\nCleaning data...")
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
initial_rows = len(df_combined)
df_combined.dropna(inplace=True)
final_rows = len(df_combined)
print(f"Removed {initial_rows - final_rows} rows with missing values")

print(f"\nMerged data shape: {df_combined.shape}")
print(f"Time range: {df_combined.index.min().date()} to {df_combined.index.max().date()}")
print(f"Sample count: {len(df_combined)}")

print("\n" + "=" * 80)
print("Step 2: Prophet Data Preparation")
print("=" * 80)

target = 'PM2.5'
df_prophet = df_combined[[target]].copy()
df_prophet = df_prophet.reset_index()
df_prophet.columns = ['ds', 'y']

print(f"Prophet data shape: {df_prophet.shape}")
print(f"Target variable: {target}")
print(f"  PM2.5 statistics:")
print(f"    Mean: {df_prophet['y'].mean():.2f} μg/m³")
print(f"    Std: {df_prophet['y'].std():.2f} μg/m³")
print(f"    Range: [{df_prophet['y'].min():.2f}, {df_prophet['y'].max():.2f}] μg/m³")

print("\n" + "=" * 80)
print("Step 3: Prophet Model Training")
print("=" * 80)

def create_prophet_model(params=None):
    if params is None:
        params = PROPHET_PARAMS

    model = Prophet(
        seasonality_mode=params['seasonality_mode'],
        yearly_seasonality=params['yearly_seasonality'],
        weekly_seasonality=params['weekly_seasonality'],
        daily_seasonality=params['daily_seasonality'],
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        holidays_prior_scale=params['holidays_prior_scale']
    )

    return model

def train_prophet_model(df_train, params=None):
    print("Creating and training Prophet model...")

    model = create_prophet_model(params)

    print("Model parameters:")
    for key, value in PROPHET_PARAMS.items():
        print(f"  {key}: {value}")

    train_start = pd.Timestamp.now()
    model.fit(df_train)
    train_time = (pd.Timestamp.now() - train_start).total_seconds()
    print(f"Training time: {train_time:.2f} seconds")
    
    return model

def make_future_dataframe(model, periods=365):
    future = model.make_future_dataframe(periods=periods, freq='D')
    return future

def predict_prophet(model, future_df):
    print("Making predictions...")
    forecast = model.predict(future_df)
    return forecast

def perform_cross_validation(model, df_train):
    print("\nPerforming cross-validation...")
    print(f"Cross-validation parameters: {CROSS_VALIDATION_PARAMS}")

    cv_start = pd.Timestamp.now()
    df_cv = cross_validation(
        model,
        initial=CROSS_VALIDATION_PARAMS['initial'],
        period=CROSS_VALIDATION_PARAMS['period'],
        horizon=CROSS_VALIDATION_PARAMS['horizon']
    )
    cv_time = (pd.Timestamp.now() - cv_start).total_seconds()
    print(f"Cross-validation time: {cv_time:.2f} seconds")
    
    df_p = performance_metrics(df_cv)
    print("\nCross-validation performance metrics:")
    print(f"  RMSE: {df_p['rmse'].mean():.4f}")
    print(f"  MAE: {df_p['mae'].mean():.4f}")
    print(f"  MAPE: {df_p['mape'].mean():.4f}")

    return df_cv, df_p

model = train_prophet_model(df_prophet)

future = make_future_dataframe(model, periods=365)

forecast = predict_prophet(model, future)

df_cv, df_p = perform_cross_validation(model, df_prophet)

print("\n" + "=" * 80)
print("Step 4: Model Evaluation and Visualization")
print("=" * 80)

def evaluate_prophet_model(df_prophet, forecast):
    print("Evaluating model performance...")

    df_eval = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    df_eval = df_eval.merge(df_prophet[['ds', 'y']], on='ds', how='inner')

    y_true = df_eval['y']
    y_pred = df_eval['yhat']

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print("Model performance metrics:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f} μg/m³")
    print(f"  MAE: {mae:.4f} μg/m³")
    print(f"  MAPE: {mape:.4f}%")

    return df_eval, {
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def plot_prophet_forecast(model, forecast, df_prophet):
    print("Generating prediction visualizations...")

    fig1, ax1 = plt.subplots(figsize=(15, 8))
    model.plot(forecast, ax=ax1)
    ax1.set_title('Prophet PM2.5 Concentration Forecast', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'prophet_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: prophet_forecast.png")

    fig2 = model.plot_components(forecast)
    fig2.suptitle('Prophet Seasonality Decomposition', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'prophet_components.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: prophet_components.png")

    df_eval, metrics = evaluate_prophet_model(df_prophet, forecast)

    fig3, ax3 = plt.subplots(figsize=(10, 8))
    scatter = ax3.scatter(df_eval['y'], df_eval['yhat'], alpha=0.6, s=30, edgecolors='black', linewidth=0.3)

    min_val = min(df_eval['y'].min(), df_eval['yhat'].min())
    max_val = max(df_eval['y'].max(), df_eval['yhat'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Line')

    ax3.set_xlabel('Actual PM2.5 Concentration (μg/m³)', fontsize=12)
    ax3.set_ylabel('Predicted PM2.5 Concentration (μg/m³)', fontsize=12)
    ax3.set_title(f'Prophet Predictions vs Actual\nR²={metrics["R²"]:.4f}, RMSE={metrics["RMSE"]:.2f}', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'prophet_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: prophet_scatter.png")

    fig4 = plot_cross_validation_metric(df_cv, metric='rmse')
    fig4.suptitle('Prophet Cross-Validation RMSE Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'prophet_cv_rmse.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: prophet_cv_rmse.png")

    fig5, ax5 = plt.subplots(figsize=(15, 8))

    recent_year = df_eval[df_eval['ds'] >= (df_eval['ds'].max() - pd.DateOffset(years=1))]

    ax5.plot(recent_year['ds'], recent_year['y'], 'k-', label='Actual', linewidth=2, alpha=0.8)
    ax5.plot(recent_year['ds'], recent_year['yhat'], 'r-', label='Prophet Prediction', linewidth=1.5, alpha=0.8)
    ax5.fill_between(recent_year['ds'], recent_year['yhat_lower'], recent_year['yhat_upper'],
                     color='red', alpha=0.2, label='95% CI')

    ax5.set_xlabel('Date', fontsize=12)
    ax5.set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
    ax5.set_title('Prophet Time Series Forecast Comparison (Last Year)', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'prophet_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: prophet_timeseries.png")

    return metrics

metrics = plot_prophet_forecast(model, forecast, df_prophet)

print("\n" + "=" * 80)
print("Step 5: Saving Results")
print("=" * 80)

def save_prophet_results(model, forecast, df_prophet, metrics, df_cv, df_p):
    print("Saving model results...")

    forecast_to_save = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend',
                                 'yearly', 'weekly']].copy()
    forecast_to_save['ds'] = forecast_to_save['ds'].dt.date
    forecast_to_save.to_csv(output_dir / 'prophet_forecast_results.csv', index=False, encoding='utf-8-sig')
    print("Saved: prophet_forecast_results.csv")

    performance_df = pd.DataFrame([{
        'Model': 'Prophet',
        'R²': metrics['R²'],
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'MAPE': metrics['MAPE'],
        'Training_Time_Seconds': None,
        'CV_RMSE_Mean': df_p['rmse'].mean(),
        'CV_MAE_Mean': df_p['mae'].mean(),
        'CV_MAPE_Mean': df_p['mape'].mean()
    }])
    performance_df.to_csv(output_dir / 'prophet_performance.csv', index=False, encoding='utf-8-sig')
    print("Saved: prophet_performance.csv")

    df_cv.to_csv(output_dir / 'prophet_cross_validation.csv', index=False, encoding='utf-8-sig')
    df_p.to_csv(output_dir / 'prophet_cv_performance.csv', index=False, encoding='utf-8-sig')
    print("Saved: prophet_cross_validation.csv and prophet_cv_performance.csv")

    params_df = pd.DataFrame(list(PROPHET_PARAMS.items()), columns=['Parameter', 'Value'])
    params_df.to_csv(output_dir / 'prophet_parameters.csv', index=False, encoding='utf-8-sig')
    print("Saved: prophet_parameters.csv")

    future_only = forecast[forecast['ds'] > df_prophet['ds'].max()].copy()
    future_only = future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    future_only['ds'] = future_only['ds'].dt.date
    future_only.to_csv(output_dir / 'prophet_future_predictions.csv', index=False, encoding='utf-8-sig')
    print("Saved: prophet_future_predictions.csv")

    import joblib
    model_path = model_dir / 'prophet_pm25_model.pkl'
    joblib.dump(model, model_path)
    print(f"Saved model: {model_path}")

    return True

save_prophet_results(model, forecast, df_prophet, metrics, df_cv, df_p)

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)

print("\nGenerated plot files:")
print("  - prophet_forecast.png         Prophet forecast plot")
print("  - prophet_components.png       Seasonality decomposition")
print("  - prophet_scatter.png          Predictions vs actual scatter")
print("  - prophet_cv_rmse.png          Cross-validation RMSE")
print("  - prophet_timeseries.png       Time series comparison")

print("\nGenerated CSV files:")
print("  - prophet_forecast_results.csv     Complete forecast results")
print("  - prophet_performance.csv          Model performance metrics")
print("  - prophet_cross_validation.csv     Cross-validation results")
print("  - prophet_cv_performance.csv       CV performance metrics")
print("  - prophet_parameters.csv           Model parameters")
print("  - prophet_future_predictions.csv   Future predictions")

print("\nModel files:")
print("  - prophet_pm25_model.pkl           Prophet model file")

print(f"\nProphet Model Performance Summary:")
print(f"  R² Score: {metrics['R²']:.4f}")
print(f"  RMSE: {metrics['RMSE']:.2f} μg/m³")
print(f"  MAE: {metrics['MAE']:.2f} μg/m³")
print(f"  MAPE: {metrics['MAPE']:.2f}%")

print("\nCross-Validation Performance:")
print(f"  CV RMSE: {df_p['rmse'].mean():.4f}")
print(f"  CV MAE: {df_p['mae'].mean():.4f}")
print(f"  CV MAPE: {df_p['mape'].mean():.4f}")

print("\nProphet Model Features:")
print("  ✓ Automatic handling of seasonality and trends")
print("  ✓ Handles missing data")
print("  ✓ Provides prediction confidence intervals")
print("  ✓ Highly interpretable")

print("\n" + "=" * 80)
print("Beijing PM2.5 Prophet Time Series Forecast Analysis Complete!")
print("=" * 80)
