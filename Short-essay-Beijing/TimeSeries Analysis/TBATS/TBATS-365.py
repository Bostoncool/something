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

# Try to import TBATS and other time series libraries
try:
    from tbats import TBATS
    TBATS_AVAILABLE = True
except ImportError:
    TBATS_AVAILABLE = False
    print("Warning: TBATS library not installed. Please install with: pip install tbats")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm not installed, progress display will use simplified version.")

# Machine learning libraries for comparison and preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Statistical libraries for time series analysis
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

# Set English font for matplotlib
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus signs are displayed correctly
plt.rcParams['figure.dpi'] = 100

# Set random seed for reproducibility
np.random.seed(42)

# Get CPU core count
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - TBATS Model")
print("=" * 80)
print(f"CPU cores: {CPU_COUNT}, Parallel worker threads: {MAX_WORKERS}")

if not TBATS_AVAILABLE:
    print("\nError: TBATS library is required but not installed.")
    print("Please install it with: pip install tbats")
    sys.exit(1)

# ============================== Part 1: Configuration and Path Setup ==============================
print("\nConfiguring parameters...")

# Data paths (same as LSTM model)
pollution_all_path = r'/tmp/Benchmark/all(AQI+PM2.5+PM10)'
pollution_extra_path = r'/tmp/Benchmark/extra(SO2+NO2+CO+O3)'
era5_path = r'/tmp/ERA5-Beijing-NC'

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

# TBATS specific configuration
TBATS_TRAIN_RATIO = 0.7
TBATS_VAL_RATIO = 0.15

# Seasonal patterns for TBATS (Beijing PM2.5 - yearly seasonality only)
SEASONAL_PERIODS = [365]  # Yearly seasonality only

print(f"Data time range: {start_date.date()} to {end_date.date()}")
print(f"Target variable: PM2.5 concentration")
print(f"TBATS seasonal periods: {SEASONAL_PERIODS} days (365-day yearly cycle only)")
print(f"Output directory: {output_dir}")
print(f"Model save directory: {model_dir}")

# ============================== Part 2: Data Loading Functions (Reused from LSTM) ==============================
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

# ============================== Part 3: Feature Engineering (Adapted for TBATS) ==============================
def create_features(df):
    """Create additional features (adapted for TBATS time series analysis)"""
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

    # Time features (important for TBATS seasonal analysis)
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

    # Lag features (for TBATS exogenous variables)
    if 'PM2.5' in df_copy:
        for lag in [1, 2, 3, 7, 14, 30]:
            df_copy[f'PM2.5_lag{lag}'] = df_copy['PM2.5'].shift(lag)
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

# ============================== Part 4: TBATS Model Implementation ==============================
def check_stationarity(timeseries, title=''):
    """Check time series stationarity using Augmented Dickey-Fuller test"""
    print(f'\nStationarity check for {title}:')

    # Perform Dickey-Fuller test
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    print(dfoutput)

    if dftest[1] <= 0.05:
        print("Result: Series is stationary")
        return True
    else:
        print("Result: Series is non-stationary")
        return False

def decompose_time_series(ts, title=''):
    """Perform seasonal decomposition of time series"""
    print(f'\nDecomposing {title}...')

    # Try different decomposition methods
    try:
        # First try additive decomposition
        decomposition = seasonal_decompose(ts.dropna(), model='additive', period=365)
        print("Using additive decomposition")
    except:
        try:
            # Try multiplicative decomposition
            decomposition = seasonal_decompose(ts.dropna(), model='multiplicative', period=365)
            print("Using multiplicative decomposition")
        except Exception as e:
            print(f"Decomposition failed: {e}")
            return None

    return decomposition

def prepare_tbats_data(X, y, exogenous_vars=None, train_ratio=0.7, val_ratio=0.15):
    """Prepare data for TBATS model training"""
    print("\nPreparing data for TBATS...")

    # Ensure data is sorted by time
    combined_data = pd.DataFrame({'PM2.5': y}, index=X.index)
    if exogenous_vars:
        for var in exogenous_vars:
            if var in X.columns:
                combined_data[var] = X[var]

    combined_data = combined_data.sort_index()

    # Remove any remaining missing values
    combined_data = combined_data.dropna()

    # Split data by time order (important for time series)
    n_samples = len(combined_data)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    train_data = combined_data.iloc[:train_size]
    val_data = combined_data.iloc[train_size:train_size + val_size]
    test_data = combined_data.iloc[train_size + val_size:]

    print(f"Training set: {len(train_data)} samples ({train_data.index.min().date()} to {train_data.index.max().date()})")
    print(f"Validation set: {len(val_data)} samples ({val_data.index.min().date()} to {val_data.index.max().date()})")
    print(f"Test set: {len(test_data)} samples ({test_data.index.min().date()} to {test_data.index.max().date()})")

    return train_data, val_data, test_data

def train_tbats_model(train_data, seasonal_periods=None, use_boxcox=True, use_arma_errors=True,
                     use_trend=True, use_damped_trend=True, verbose=False, show_progress=True):
    """Train TBATS model"""
    print("\nTraining TBATS model...")

    if seasonal_periods is None:
        seasonal_periods = SEASONAL_PERIODS

    # Prepare target variable
    # Note: TBATS does not support exogenous variables in fit() method
    y_train = train_data['PM2.5'].values

    # Create and fit TBATS model
    estimator = TBATS(
        seasonal_periods=seasonal_periods,
        use_box_cox=use_boxcox,  # Note: TBATS uses use_box_cox (with underscore)
        use_arma_errors=use_arma_errors,
        use_trend=use_trend,
        use_damped_trend=use_damped_trend
    )

    print("TBATS model configuration:")
    print(f"  Seasonal periods: {seasonal_periods}")
    print(f"  Box-Cox transformation: {use_boxcox}")
    print(f"  ARMA errors: {use_arma_errors}")
    print(f"  Trend component: {use_trend}")
    print(f"  Damped trend: {use_damped_trend}")
    print(f"  Note: TBATS does not support exogenous variables")

    # TBATS训练进度可视化
    print(f"Training TBATS model on {len(y_train)} data points...")
    print(f"Note: TBATS training may take several minutes, please be patient...")
    start_time = time.time()

    if TQDM_AVAILABLE and show_progress:
        # 使用tqdm显示不确定进度的训练状态
        print("Training in progress (progress indicator is estimated)...")
        import threading
        import time as time_module

        training_complete = False
        training_result = [None]  # 使用列表来存储结果，以便在线程间共享
        training_error = [None]

        def train_model():
            """在单独的线程中训练模型"""
            try:
                training_result[0] = estimator.fit(y_train)
            except Exception as e:
                training_error[0] = e
            finally:
                training_complete = True

        def show_progress():
            """显示训练进度（基于时间的估算）"""
            with tqdm(total=100, desc="TBATS Training", unit="%", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]') as pbar:
                while not training_complete:
                    elapsed = time_module.time() - start_time
                    # 根据数据量估算训练时间（更保守的估计）
                    estimated_total_time = max(60, len(y_train) / 30)
                    # 进度不会超过90%，直到训练真正完成
                    progress = min(90, int((elapsed / estimated_total_time) * 100))
                    pbar.n = progress
                    pbar.refresh()
                    time_module.sleep(1.5)
                
                # 训练完成，设置为100%
                pbar.n = 100
                pbar.refresh()

        # 启动训练线程
        train_thread = threading.Thread(target=train_model, daemon=False)
        progress_thread = threading.Thread(target=show_progress, daemon=True)
        
        train_thread.start()
        progress_thread.start()
        
        # 等待训练完成
        train_thread.join()
        
        # 检查是否有错误
        if training_error[0] is not None:
            raise training_error[0]
        
        model = training_result[0]
    else:
        # 如果没有tqdm或不显示进度，直接训练
        model = estimator.fit(y_train)

    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Model AIC: {model.aic:.2f}")

    return model, training_time

def predict_tbats_model(model, data, steps_ahead=None):
    """Make predictions using trained TBATS model"""
    if steps_ahead is None:
        steps_ahead = len(data)

    # Make predictions
    # Note: TBATS.forecast() does not support exogenous variables
    predictions = model.forecast(steps=steps_ahead)  # TBATS uses 'steps' parameter, not 'steps_ahead'

    return predictions

def evaluate_tbats_model(y_true, y_pred):
    """Evaluate TBATS model performance"""
    # Remove any NaN values
    valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_idx]
    y_pred_clean = y_pred[valid_idx]

    if len(y_true_clean) == 0:
        return {'R²': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}

    r2 = r2_score(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100

    return {
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'n_samples': len(y_true_clean)
    }

def grid_search_tbats(train_data, val_data, seasonal_periods_options=None, verbose=False):
    """Perform grid search for TBATS hyperparameters"""
    print("\nPerforming TBATS hyperparameter grid search...")

    if seasonal_periods_options is None:
        seasonal_periods_options = [
            [365]  # Only 365-day yearly seasonality
        ]

    # Define parameter grid
    param_grid = {
        'seasonal_periods': seasonal_periods_options,
        'use_boxcox': [True, False],
        'use_arma_errors': [True, False],
        'use_trend': [True, False],
        'use_damped_trend': [True, False]
    }

    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    print(f"Testing {len(param_combinations)} parameter combinations...")

    best_model = None
    best_score = float('inf')
    best_params = None
    results = []

    iterable = tqdm(param_combinations, desc="Grid Search", unit="combo") if TQDM_AVAILABLE else param_combinations
    for i, params in enumerate(iterable):
        param_dict = dict(zip(param_names, params))

        try:
            # Train model (在网格搜索中不显示单个模型的进度条，避免冲突)
            model, _ = train_tbats_model(
                train_data,
                seasonal_periods=param_dict['seasonal_periods'],
                use_boxcox=param_dict['use_boxcox'],
                use_arma_errors=param_dict['use_arma_errors'],
                use_trend=param_dict['use_trend'],
                use_damped_trend=param_dict['use_damped_trend'],
                verbose=False,
                show_progress=False
            )

            # Validate model
            val_predictions = predict_tbats_model(model, val_data)
            val_metrics = evaluate_tbats_model(val_data['PM2.5'].values, val_predictions)

            rmse_score = val_metrics['RMSE']

            results.append({
                **param_dict,
                'val_rmse': rmse_score,
                'val_r2': val_metrics['R²']
            })

            if rmse_score < best_score:
                best_score = rmse_score
                best_model = model
                best_params = param_dict

            if verbose and (i + 1) % 5 == 0:
                print(f"  Tested {i+1}/{len(param_combinations)} combinations, current best RMSE: {best_score:.4f}")

        except Exception as e:
            if verbose:
                print(f"  Parameter combination {i+1} failed: {e}")
            continue

    print(f"\nGrid search complete. Best validation RMSE: {best_score:.4f}")

    return best_model, best_params, results

# ============================== Part 5: Main Execution ==============================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Step 1: Data Loading and Preprocessing")
    print("=" * 80)

    # Load data
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

    # ============================== Part 6: TBATS Feature Selection ==============================
    print("\n" + "=" * 80)
    print("Step 2: Feature Selection for TBATS")
    print("=" * 80)

    target = 'PM2.5'

    # For TBATS exogenous variables, select weather-related features that might influence PM2.5
    exogenous_candidates = [
        # Meteorological features
        't2m', 'd2m', 'u10', 'v10', 'u100', 'v100',
        'blh', 'sp', 'tcwv', 'tp', 'avg_tprate',
        'tisr', 'str', 'cvh', 'cvl', 'mn2t', 'sd', 'lsm',
        # Derived features
        'wind_speed_10m', 'wind_dir_10m', 'wind_speed_100m', 'wind_dir_100m',
        'temp_dewpoint_diff', 'relative_humidity', 'wind_dir_category',
        # Time features
        'month', 'day_of_year', 'day_of_week', 'season', 'is_heating_season'
    ]

    # Filter to available features
    available_exogenous = [col for col in exogenous_candidates if col in df_combined.columns]

    print(f"\nTarget variable: {target}")
    print(f"Selected exogenous features ({len(available_exogenous)}):")
    for i, feature in enumerate(available_exogenous, 1):
        print("2d")

    X = df_combined[available_exogenous].copy()
    y = df_combined[target].copy()

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")

    print(f"\nPM2.5 statistics:")
    print(f"  Mean: {y.mean():.2f} μg/m³")
    print(f"  Std Dev: {y.std():.2f} μg/m³")
    print(f"  Range: [{y.min():.2f}, {y.max():.2f}] μg/m³")

    # ============================== Part 7: Time Series Analysis ==============================
    print("\n" + "=" * 80)
    print("Step 3: Time Series Analysis")
    print("=" * 80)

    # Check stationarity
    is_stationary = check_stationarity(y, "PM2.5 concentration")

    # Seasonal decomposition
    decomposition = decompose_time_series(y, "PM2.5 time series")

    # ============================== Part 8: Prepare Data for TBATS ==============================
    print("\n" + "=" * 80)
    print("Step 4: Prepare Data for TBATS")
    print("=" * 80)

    # Prepare data with exogenous variables
    train_data, val_data, test_data = prepare_tbats_data(
        X, y,
        exogenous_vars=available_exogenous,
        train_ratio=TBATS_TRAIN_RATIO,
        val_ratio=TBATS_VAL_RATIO
    )

    # ============================== Part 9: TBATS Model Training ==============================
    print("\n" + "=" * 80)
    print("Step 5: TBATS Model Training and Evaluation")
    print("=" * 80)

    # Option 1: Train basic TBATS model
    print("\n--- Basic TBATS Model ---")
    basic_model, basic_training_time = train_tbats_model(train_data)

    # Evaluate basic model
    basic_results = []

    # Training set evaluation (in-sample)
    train_predictions = basic_model.y_hat
    train_metrics = evaluate_tbats_model(train_data['PM2.5'].values, train_predictions)
    basic_results.append({
        'Model': 'TBATS_Basic',
        'Dataset': 'Train',
        'R²': train_metrics['R²'],
        'RMSE': train_metrics['RMSE'],
        'MAE': train_metrics['MAE'],
        'MAPE': train_metrics['MAPE']
    })

    # Validation set evaluation
    val_predictions = predict_tbats_model(basic_model, val_data)
    val_metrics = evaluate_tbats_model(val_data['PM2.5'].values, val_predictions)
    basic_results.append({
        'Model': 'TBATS_Basic',
        'Dataset': 'Validation',
        'R²': val_metrics['R²'],
        'RMSE': val_metrics['RMSE'],
        'MAE': val_metrics['MAE'],
        'MAPE': val_metrics['MAPE']
    })

    # Test set evaluation
    test_predictions = predict_tbats_model(basic_model, test_data)
    test_metrics = evaluate_tbats_model(test_data['PM2.5'].values, test_predictions)
    basic_results.append({
        'Model': 'TBATS_Basic',
        'Dataset': 'Test',
        'R²': test_metrics['R²'],
        'RMSE': test_metrics['RMSE'],
        'MAE': test_metrics['MAE'],
        'MAPE': test_metrics['MAPE']
    })

    basic_results_df = pd.DataFrame(basic_results)
    print("\nBasic TBATS model performance:")
    print(basic_results_df.to_string(index=False))

    # ============================== Part 10: Hyperparameter Optimization ==============================
    print("\n" + "=" * 80)
    print("Step 6: TBATS Hyperparameter Optimization")
    print("=" * 80)

    # Perform grid search
    optimized_model, best_params, grid_results = grid_search_tbats(
        train_data, val_data, verbose=True
    )

    # Evaluate optimized model
    optimized_results = []

    # Training set evaluation (in-sample)
    train_predictions_opt = optimized_model.y_hat
    train_metrics_opt = evaluate_tbats_model(train_data['PM2.5'].values, train_predictions_opt)
    optimized_results.append({
        'Model': 'TBATS_Optimized',
        'Dataset': 'Train',
        'R²': train_metrics_opt['R²'],
        'RMSE': train_metrics_opt['RMSE'],
        'MAE': train_metrics_opt['MAE'],
        'MAPE': train_metrics_opt['MAPE']
    })

    # Validation set evaluation
    val_predictions_opt = predict_tbats_model(optimized_model, val_data)
    val_metrics_opt = evaluate_tbats_model(val_data['PM2.5'].values, val_predictions_opt)
    optimized_results.append({
        'Model': 'TBATS_Optimized',
        'Dataset': 'Validation',
        'R²': val_metrics_opt['R²'],
        'RMSE': val_metrics_opt['RMSE'],
        'MAE': val_metrics_opt['MAE'],
        'MAPE': val_metrics_opt['MAPE']
    })

    # Test set evaluation
    test_predictions_opt = predict_tbats_model(optimized_model, test_data)
    test_metrics_opt = evaluate_tbats_model(test_data['PM2.5'].values, test_predictions_opt)
    optimized_results.append({
        'Model': 'TBATS_Optimized',
        'Dataset': 'Test',
        'R²': test_metrics_opt['R²'],
        'RMSE': test_metrics_opt['RMSE'],
        'MAE': test_metrics_opt['MAE'],
        'MAPE': test_metrics_opt['MAPE']
    })

    optimized_results_df = pd.DataFrame(optimized_results)
    print("\nOptimized TBATS model performance:")
    print(optimized_results_df.to_string(index=False))

    print(f"\nBest hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # ============================== Part 11: Model Comparison ==============================
    print("\n" + "=" * 80)
    print("Step 7: Model Comparison")
    print("=" * 80)

    all_results = pd.concat([basic_results_df, optimized_results_df])
    print("\nAll TBATS model performance comparison:")
    print(all_results.to_string(index=False))

    # Test set performance ranking
    test_results = all_results[all_results['Dataset'] == 'Test'].sort_values('R²', ascending=False)
    print("\nTest set performance ranking:")
    print(test_results.to_string(index=False))

    # Find best model
    best_model_info = test_results.iloc[0]
    print(f"\nBest model: {best_model_info['Model']}")
    print(f"  R² Score: {best_model_info['R²']:.4f}")
    print(f"  RMSE: {best_model_info['RMSE']:.2f} μg/m³")
    print(f"  MAE: {best_model_info['MAE']:.2f} μg/m³")
    print(f"  MAPE: {best_model_info['MAPE']:.2f}%")

    # ============================== Part 12: Visualization ==============================
    print("\n" + "=" * 80)
    print("Step 8: Generate Visualization Charts")
    print("=" * 80)

    # 12.1 Time series decomposition
    if decomposition is not None:
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))

        axes[0].plot(decomposition.observed, linewidth=1.5)
        axes[0].set_title('Observed PM2.5 Concentration', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('PM2.5 (μg/m³)')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(decomposition.trend, linewidth=1.5, color='orange')
        axes[1].set_title('Trend Component', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(decomposition.seasonal, linewidth=1.5, color='green')
        axes[2].set_title('Seasonal Component', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Seasonal')
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(decomposition.resid, linewidth=1, color='red', alpha=0.7)
        axes[3].set_title('Residual Component', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Residual')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'tbats_decomposition.png', dpi=300, bbox_inches='tight')
        print("Saved: tbats_decomposition.png")
        plt.close()

    # 12.2 Prediction vs Actual scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    datasets = [
        ('Train', train_data['PM2.5'].values, basic_model.y_hat),
        ('Validation', val_data['PM2.5'].values, val_predictions),
        ('Test', test_data['PM2.5'].values, test_predictions)
    ]

    for i, (name, y_true, y_pred) in enumerate(datasets):
        ax = axes[i]

        # Remove NaN values for plotting
        valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[valid_idx]
        y_pred_clean = y_pred[valid_idx]

        ax.scatter(y_true_clean, y_pred_clean, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)

        min_val = min(y_true_clean.min(), y_pred_clean.min())
        max_val = max(y_true_clean.max(), y_pred_clean.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal prediction line')

        # Calculate metrics for this dataset
        metrics = evaluate_tbats_model(y_true_clean, y_pred_clean)

        ax.set_xlabel('Actual PM2.5 Concentration (μg/m³)', fontsize=11)
        ax.set_ylabel('Predicted PM2.5 Concentration (μg/m³)', fontsize=11)
        ax.set_title(f'TBATS Basic - {name}\nR²={metrics["R²"]:.4f}, RMSE={metrics["RMSE"]:.2f}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'tbats_prediction_scatter.png', dpi=300, bbox_inches='tight')
    print("Saved: tbats_prediction_scatter.png")
    plt.close()

    # 12.3 Time series prediction comparison (test set)
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))

    # Plot last 180 days of test data for clarity
    plot_days = min(180, len(test_data))
    plot_idx = range(len(test_data) - plot_days, len(test_data))
    time_idx = test_data.index[plot_idx]

    ax.plot(time_idx, test_data['PM2.5'].values[plot_idx], 'k-', label='Actual',
            linewidth=2, alpha=0.8)
    ax.plot(time_idx, test_predictions[plot_idx], 'r--', label='TBATS Basic Prediction',
            linewidth=1.5, alpha=0.7)
    ax.plot(time_idx, test_predictions_opt[plot_idx], 'b--', label='TBATS Optimized Prediction',
            linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
    ax.set_title(f'TBATS Time Series Prediction Comparison (Last {plot_days} days of test set)\n'
                 f'Basic: R²={test_metrics["R²"]:.4f}, RMSE={test_metrics["RMSE"]:.2f} | '
                 f'Optimized: R²={test_metrics_opt["R²"]:.4f}, RMSE={test_metrics_opt["RMSE"]:.2f}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'tbats_timeseries_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: tbats_timeseries_comparison.png")
    plt.close()

    # 12.4 Residual analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    datasets = [
        ('Train', train_data['PM2.5'].values, basic_model.y_hat),
        ('Validation', val_data['PM2.5'].values, val_predictions),
        ('Test', test_data['PM2.5'].values, test_predictions)
    ]

    for i, (name, y_true, y_pred) in enumerate(datasets):
        ax = axes[i]

        valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[valid_idx]
        y_pred_clean = y_pred[valid_idx]
        residuals = y_true_clean - y_pred_clean

        ax.scatter(y_pred_clean, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted value (μg/m³)', fontsize=11)
        ax.set_ylabel('Residuals (μg/m³)', fontsize=11)
        ax.set_title(f'TBATS Basic - {name} Residuals\nMean={residuals.mean():.2f}, Std dev={residuals.std():.2f}',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'tbats_residuals_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: tbats_residuals_analysis.png")
    plt.close()

    # 12.5 Model performance comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    test_results_sorted = test_results.copy()
    models_list = test_results_sorted['Model'].tolist()
    x_pos = np.arange(len(models_list))

    bars = ax.bar(x_pos, test_results_sorted['R²'], alpha=0.7,
                  edgecolor='black', linewidth=1.5, color=['skyblue', 'lightcoral'])

    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('TBATS_', '') for m in models_list], fontsize=11)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('TBATS Model Performance Comparison (Test Set)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Display values on bars
    for i, (bar, r2) in enumerate(zip(bars, test_results_sorted['R²'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{r2:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'tbats_model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: tbats_model_comparison.png")
    plt.close()

    # ============================== Part 13: Save Results ==============================
    print("\n" + "=" * 80)
    print("Step 9: Save Results")
    print("=" * 80)

    # Save model performance
    all_results.to_csv(output_dir / 'tbats_model_performance.csv', index=False, encoding='utf-8-sig')
    print("Saved: tbats_model_performance.csv")

    # Save best parameters
    best_params_df = pd.DataFrame([best_params])
    best_params_df.to_csv(output_dir / 'tbats_best_parameters.csv', index=False, encoding='utf-8-sig')
    print("Saved: tbats_best_parameters.csv")

    # Save grid search results
    grid_results_df = pd.DataFrame(grid_results)
    grid_results_df.to_csv(output_dir / 'tbats_grid_search_results.csv', index=False, encoding='utf-8-sig')
    print("Saved: tbats_grid_search_results.csv")

    # Save prediction results
    predictions_df = pd.DataFrame({
        'Date': test_data.index,
        'Actual': test_data['PM2.5'].values,
        'TBATS_Basic': test_predictions,
        'TBATS_Optimized': test_predictions_opt,
        'Basic_Error': test_data['PM2.5'].values - test_predictions,
        'Optimized_Error': test_data['PM2.5'].values - test_predictions_opt
    })
    predictions_df.to_csv(output_dir / 'tbats_predictions.csv', index=False, encoding='utf-8-sig')
    print("Saved: tbats_predictions.csv")

    # Save models (using pickle since TBATS models are not easily serializable with torch.save)
    import pickle

    with open(model_dir / 'tbats_basic_model.pkl', 'wb') as f:
        pickle.dump(basic_model, f)
    print("Saved: tbats_basic_model.pkl")

    with open(model_dir / 'tbats_optimized_model.pkl', 'wb') as f:
        pickle.dump(optimized_model, f)
    print("Saved: tbats_optimized_model.pkl")

    # ============================== Part 14: Summary Report ==============================
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

    print("\nGenerated files:")
    print("\nCSV Files:")
    print("  - tbats_model_performance.csv       Model performance comparison")
    print("  - tbats_best_parameters.csv         Best hyperparameters")
    print("  - tbats_grid_search_results.csv     Grid search results")
    print("  - tbats_predictions.csv             Prediction results")

    print("\nChart Files:")
    print("  - tbats_decomposition.png           Time series decomposition")
    print("  - tbats_prediction_scatter.png      Prediction vs Actual scatter plots")
    print("  - tbats_timeseries_comparison.png   Time series comparison")
    print("  - tbats_residuals_analysis.png      Residual analysis")
    print("  - tbats_model_comparison.png        Model performance comparison")

    print("\nModel Files:")
    print("  - tbats_basic_model.pkl             Basic TBATS model")
    print("  - tbats_optimized_model.pkl         Optimized TBATS model")

    print(f"\nBest model: {best_model_info['Model']}")
    print(f"  R² Score: {best_model_info['R²']:.4f}")
    print(f"  RMSE: {best_model_info['RMSE']:.2f} μg/m³")
    print(f"  MAE: {best_model_info['MAE']:.2f} μg/m³")
    print(f"  MAPE: {best_model_info['MAPE']:.2f}%")

    print(f"\nBest hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    print("\nTBATS model summary:")
    print(f"  Seasonal periods: {SEASONAL_PERIODS} (365-day yearly cycle only)")
    print(f"  Exogenous variables: {len(available_exogenous)}")
    print(f"  Training data: {len(train_data)} days")
    print(f"  Validation data: {len(val_data)} days")
    print(f"  Test data: {len(test_data)} days")

    print("\n" + "=" * 80)
    print("TBATS PM2.5 Concentration Prediction Complete!")
    print("=" * 80)
