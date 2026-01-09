import os
import glob
import gc
import warnings
import multiprocessing
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

np.random.seed(42)

print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - Random Forest Model (NetCDF ERA5)")
print("=" * 80)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
pollution_all_path = '/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)'
pollution_extra_path = '/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)'
era5_path = '/root/autodl-tmp/ERA5-Beijing-NC'

output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)

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
print(f"CPU cores: {CPU_COUNT}, Parallel worker processes: {MAX_WORKERS}")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def daterange(start: datetime, end: datetime):
    """Yield each day between start and end (inclusive)."""
    for offset in range((end - start).days + 1):
        yield start + timedelta(days=offset)


def build_file_path_dict(base_path: str, prefix: str):
    """构建文件路径字典，键为日期字符串(YYYYMMDD)，值为文件路径。时间复杂度O(1)查找。"""
    file_dict = {}
    filename_pattern = f"{prefix}_"
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename.startswith(filename_pattern) and filename.endswith('.csv'):
                # 提取日期部分: beijing_all_YYYYMMDD.csv -> YYYYMMDD
                date_str = filename[len(filename_pattern):-4]  # 去掉前缀和后缀.csv
                if len(date_str) == 8 and date_str.isdigit():  # 确保是8位数字日期
                    file_path = os.path.join(root, filename)
                    # 如果同一日期有多个文件，保留第一个找到的
                    if date_str not in file_dict:
                        file_dict[date_str] = file_path
    return file_dict


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, path: Path, index: bool = False):
    """统一 CSV 输出（UTF-8-SIG），便于后续脚本读取与绘图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding='utf-8-sig')
    print(f"✓ Saved CSV: {path}")


# ---------------------------------------------------------------------------
# Pollution data loading (CSV files remain unchanged)
# ---------------------------------------------------------------------------
def read_pollution_day(args):
    """读取单日污染数据。参数为(date, all_file_dict, extra_file_dict)元组。"""
    date, all_file_dict, extra_file_dict = args
    date_str = date.strftime('%Y%m%d')
    
    # 使用字典O(1)查找，替代原来的O(n)遍历
    all_file = all_file_dict.get(date_str)
    extra_file = extra_file_dict.get(date_str)

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
    print(f"\nUsing {MAX_WORKERS} parallel worker processes")
    
    print("  Building file path dictionaries (O(1) lookup)...")
    all_file_dict = build_file_path_dict(pollution_all_path, 'beijing_all')
    extra_file_dict = build_file_path_dict(pollution_extra_path, 'beijing_extra')
    print(f"  Found {len(all_file_dict)} files in all directory")
    print(f"  Found {len(extra_file_dict)} files in extra directory")

    dates = list(daterange(start_date, end_date))
    pollution_dfs = []
    
    # 准备参数列表，每个元素是(date, all_file_dict, extra_file_dict)元组
    task_args = [(date, all_file_dict, extra_file_dict) for date in dates]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, args): args[0] for args in task_args}

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
    df_poll_all = df_poll_all.ffill()
    df_poll_all = df_poll_all.fillna(df_poll_all.mean())
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
    print(f"\nUsing {MAX_WORKERS} parallel worker processes")
    print(f"Meteorological data directory: {era5_path}")
    print(f"Directory exists: {os.path.exists(era5_path)}")

    era5_dfs = []
    years = range(2015, 2025)
    months = range(1, 13)

    month_tasks = [(year, month) for year in years for month in months if not (year == 2024 and month > 12)]
    print(f"Attempting to load {len(month_tasks)} months of data...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
    df_era5_all = df_era5_all.sort_index()

    print(f"Shape after merge: {df_era5_all.shape}")
    print(f"Time range: {df_era5_all.index.min()} to {df_era5_all.index.max()}")

    print("  Handling missing values...")
    initial_na = df_era5_all.isna().sum().sum()
    df_era5_all = df_era5_all.ffill()
    df_era5_all = df_era5_all.bfill()
    df_era5_all = df_era5_all.fillna(df_era5_all.mean())
    final_na = df_era5_all.isna().sum().sum()
    print(f"Missing value handling: {initial_na} -> {final_na}")

    print(f"Meteorological data loaded, shape: {df_era5_all.shape}")
    return df_era5_all


# ---------------------------------------------------------------------------
# Model evaluation function
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------
if __name__ == '__main__':
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
    df_combined = df_combined.replace([np.inf, -np.inf], np.nan)

    initial_rows = len(df_combined)
    df_combined = df_combined.dropna()
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
    print("Step 4: Advanced Feature Engineering (Time Series + Lags)")
    print("=" * 80)

    # ============================================
    # C1. 强化特征工程（重构版本）
    # ============================================

    df = df_combined.copy()

    # ====== 1. 加入核心滞后特征（必须） ======
    lags = [1, 2, 3, 7, 14]
    for lag in lags:
        df[f'pm25_lag_{lag}'] = df['PM2.5'].shift(lag)

    # 污染物滞后项（绝不能使用当天的，以免目标泄漏）
    pollutant_cols = ['PM10', 'NO2', 'SO2', 'CO', 'O3']
    for col in pollutant_cols:
        for lag in [1, 2, 7]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # ====== 2. 滚动窗口特征 ======
    df['pm25_ma_3'] = df['PM2.5'].rolling(3).mean()
    df['pm25_ma_7'] = df['PM2.5'].rolling(7).mean()
    df['pm25_std_7'] = df['PM2.5'].rolling(7).std()

    # ====== 3. 季节性特征 ======
    df['sin_day'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['cos_day'] = np.cos(2 * np.pi * df.index.dayofyear / 365)

    # ====== 4. 删除当日污染物列（避免泄漏） ======
    df = df.drop(columns=['PM10', 'NO2', 'SO2', 'CO', 'O3'], errors='ignore')

    # ====== 5. 删除无意义特征 ======
    df = df.drop(columns=['year'], errors='ignore')

    # ====== 6. 删除含 NaN（由 lag/rolling 引起） ======
    df = df.dropna()
    print("Final feature shape:", df.shape)

    target = 'PM2.5'
    X = df.drop(columns=[target])
    y = df[target]

    print(f"\nTarget variable: {target}")
    print(f"Feature matrix shape: {X.shape}")
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
    print("Step 5: Time Series Dataset Split")
    print("=" * 80)

    # ============================================
    # C2. 时间序列拆分（更科学）
    # ============================================

    # 按时间序列划分：训练集 = 前 80%，测试 = 后 20%
    split_idx = int(len(X) * 0.80)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(f"\nTraining set: {len(X_train)} samples ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"  Time range: {X_train.index.min().date()} to {X_train.index.max().date()}")
    print(f"  PM2.5: {y_train.mean():.2f} ± {y_train.std():.2f} μg/m³")

    print(f"\nTest set: {len(X_test)} samples ({len(X_test) / len(X) * 100:.1f}%)")
    print(f"  Time range: {X_test.index.min().date()} to {X_test.index.max().date()}")
    print(f"  PM2.5: {y_test.mean():.2f} ± {y_test.std():.2f} μg/m³")

    print("\n" + "=" * 80)
    print("Step 6: Random Forest Model Training")
    print("=" * 80)

    print("\n6.1 Training basic Random Forest model (for comparison)...")
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
    print("  ✓ Basic RF model training complete")

    y_train_pred_basic = rf_basic.predict(X_train)
    y_test_pred_basic = rf_basic.predict(X_test)

    print("\n6.2 Random Forest Model Training (Optimized)...")
    # ============================================
    # C3. 优化的 Random Forest 模型
    # ============================================

    rf_optimized = RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    print("  Starting optimized Random Forest training...")
    rf_optimized.fit(X_train, y_train)
    print("  ✓ Optimized RF model training complete")

    y_train_pred_opt = rf_optimized.predict(X_train)
    y_test_pred_opt = rf_optimized.predict(X_test)

    print("\n" + "=" * 80)
    print("Step 7: Model Evaluation")
    print("=" * 80)

    results = [
        evaluate_model(y_train, y_train_pred_basic, 'RF_Basic', 'Train'),
        evaluate_model(y_test, y_test_pred_basic, 'RF_Basic', 'Test'),
        evaluate_model(y_train, y_train_pred_opt, 'RF_Optimized', 'Train'),
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
        'Feature': X_train.columns,
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

    print(f"\nTop 15 important features (Optimized RF model):")
    print(feature_importance.head(15)[['Feature', 'Importance_Optimized_Norm']].to_string(index=False))

    print("\n" + "=" * 80)
    print("Step 8.5: Simple Time Series Visualization")
    print("=" * 80)

    # ============================================
    # C4. Visualizing Predictions
    # ============================================

    # 1) Simple time series (按模型拆分输出)
    plot_df_simple = pd.DataFrame({
        'time': y_test.index,
        'y_true': y_test.values,
        'y_pred_rf_basic': y_test_pred_basic,
        'y_pred_rf_optimized': y_test_pred_opt,
    }).sort_values('time').reset_index(drop=True)

    step = 4
    plot_df_simple_sampled = plot_df_simple.iloc[::step].copy()

    save_csv(
        plot_df_simple_sampled[['time', 'y_true', 'y_pred_rf_basic']].rename(columns={'y_pred_rf_basic': 'y_pred'}),
        output_dir / 'plot_ts_simple_sampled__rf_basic.csv'
    )
    save_csv(
        plot_df_simple_sampled[['time', 'y_true', 'y_pred_rf_optimized']].rename(columns={'y_pred_rf_optimized': 'y_pred'}),
        output_dir / 'plot_ts_simple_sampled__rf_optimized.csv'
    )

    print("\n" + "=" * 80)
    print("Step 9: Generate Visualization Charts")
    print("=" * 80)

    models_data = [
        ('rf_basic', y_test_pred_basic),
        ('rf_optimized', y_test_pred_opt),
    ]

    # 2) Scatter plot data（按模型拆分）
    for model_name, pred in models_data:
        scatter_df = pd.DataFrame({
            'Date': y_test.index,
            'Actual_PM25': y_test.values,
            'Predicted_PM25': pred,
        })
        save_csv(scatter_df, output_dir / f'plot_scatter__{model_name}.csv')

    # 3) Time series compare（按模型拆分，取测试集最后一年/指定范围 + 采样）
    plot_df = pd.DataFrame({
        'time': y_test.index,
        'y_true': y_test.values,
        'y_pred_basic': y_test_pred_basic,
        'y_pred_opt': y_test_pred_opt
    })
    
    # 按时间排序，确保时间序列单调递增
    plot_df = plot_df.sort_values('time').reset_index(drop=True)
    
    # 选择最后一年（或指定范围）的数据
    plot_range = min(365, len(plot_df))
    plot_df_subset = plot_df.iloc[-plot_range:].copy()
    
    # 使用采样绘图避免打结现象，每4个点采样一个
    step = 4
    plot_df_sampled = plot_df_subset.iloc[::step].copy()
    
    # 确保所有数据长度一致
    assert len(plot_df_sampled) == len(plot_df_sampled['time']) == len(plot_df_sampled['y_true']) == len(plot_df_sampled['y_pred_basic']) == len(plot_df_sampled['y_pred_opt']), \
        "数据长度不一致，请检查数据对齐"
    
    # 使用整数索引作为 x 轴，避免日期不连续导致的打结
    x_axis = np.arange(len(plot_df_sampled))

    ts_common = pd.DataFrame({
        'x_axis': x_axis,
        'time': plot_df_sampled['time'].values,
        'y_true': plot_df_sampled['y_true'].values,
    })
    save_csv(ts_common, output_dir / 'plot_ts_lastyear_sampled__actual.csv')

    ts_basic = ts_common.assign(y_pred=plot_df_sampled['y_pred_basic'].values)
    save_csv(ts_basic, output_dir / 'plot_ts_lastyear_sampled__rf_basic.csv')

    ts_opt = ts_common.assign(y_pred=plot_df_sampled['y_pred_opt'].values)
    save_csv(ts_opt, output_dir / 'plot_ts_lastyear_sampled__rf_optimized.csv')

    # 4) Residuals（按模型拆分）
    for model_name, pred in models_data:
        residuals = y_test - pred
        residuals_df = pd.DataFrame({
            'Date': y_test.index,
            'Actual_PM25': y_test.values,
            'Predicted_PM25': pred,
            'Residual': residuals.values,
        })
        save_csv(residuals_df, output_dir / f'plot_residuals__{model_name}.csv')

    # 5) Feature importance（按模型拆分 + TopN）
    fi_basic = (
        feature_importance[['Feature', 'Importance_Basic', 'Importance_Basic_Norm']]
        .rename(columns={'Importance_Basic': 'Importance', 'Importance_Basic_Norm': 'Importance_Norm'})
        .sort_values('Importance', ascending=False)
        .reset_index(drop=True)
    )
    fi_opt = (
        feature_importance[['Feature', 'Importance_Optimized', 'Importance_Optimized_Norm']]
        .rename(columns={'Importance_Optimized': 'Importance', 'Importance_Optimized_Norm': 'Importance_Norm'})
        .sort_values('Importance', ascending=False)
        .reset_index(drop=True)
    )
    save_csv(fi_basic, output_dir / 'plot_feature_importance__rf_basic.csv')
    save_csv(fi_opt, output_dir / 'plot_feature_importance__rf_optimized.csv')

    top_n = min(15, len(fi_opt))
    save_csv(fi_basic.head(top_n), output_dir / f'plot_feature_importance_top{top_n}__rf_basic.csv')
    save_csv(fi_opt.head(top_n), output_dir / f'plot_feature_importance_top{top_n}__rf_optimized.csv')

    # 6) Metrics（按模型/按数据集拆分）
    save_csv(results_df, output_dir / 'metrics__all_models_train_test.csv')
    for model in results_df['Model'].unique():
        model_slug = model.lower()
        save_csv(results_df[results_df['Model'] == model], output_dir / f'metrics__{model_slug}__train_test.csv')
        for ds_name in results_df['Dataset'].unique():
            part = results_df[(results_df['Model'] == model) & (results_df['Dataset'] == ds_name)]
            if not part.empty:
                save_csv(part, output_dir / f'metrics__{model_slug}__{ds_name.lower()}.csv')

    # Test 集排序（便于“指标对比图”）
    save_csv(test_results, output_dir / 'plot_metrics_ranking__test_only.csv')

    # 7) Error distribution（按模型拆分）
    for model_name, pred in models_data:
        errors = y_test - pred
        error_df = pd.DataFrame({
            'Date': y_test.index,
            'Actual_PM25': y_test.values,
            'Predicted_PM25': pred,
            'Error': errors.values,
        })
        save_csv(error_df, output_dir / f'plot_error_distribution__{model_name}.csv')

    print("\n" + "=" * 80)
    print("Step 11: Save Results")
    print("=" * 80)

    # 结果汇总（保留原始“全量/合并”输出，兼容旧脚本）
    save_csv(results_df, output_dir / 'rf_model_performance_nc.csv')
    save_csv(feature_importance, output_dir / 'rf_feature_importance_nc.csv')

    predictions_df = pd.DataFrame({
        'Date': y_test.index,
        'Actual_PM25': y_test.values,
        'Predicted_RF_Basic': y_test_pred_basic,
        'Predicted_RF_Optimized': y_test_pred_opt,
        'Error_RF_Basic': y_test.values - y_test_pred_basic,
        'Error_RF_Optimized': y_test.values - y_test_pred_opt
    })
    save_csv(predictions_df, output_dir / 'rf_predictions_all_models_nc.csv')

    # 按模型拆分预测结果
    pred_basic_df = pd.DataFrame({
        'Date': y_test.index,
        'Actual_PM25': y_test.values,
        'Predicted_PM25': y_test_pred_basic,
        'Error': y_test.values - y_test_pred_basic,
    })
    save_csv(pred_basic_df, output_dir / 'rf_predictions__rf_basic.csv')

    pred_opt_df = pd.DataFrame({
        'Date': y_test.index,
        'Actual_PM25': y_test.values,
        'Predicted_PM25': y_test_pred_opt,
        'Error': y_test.values - y_test_pred_opt,
    })
    save_csv(pred_opt_df, output_dir / 'rf_predictions__rf_optimized.csv')

    # Save Random Forest model parameters（按模型拆分）
    save_csv(pd.DataFrame([rf_basic.get_params()]), output_dir / 'rf_parameters__rf_basic.csv')
    save_csv(pd.DataFrame([rf_optimized.get_params()]), output_dir / 'rf_parameters__rf_optimized.csv')

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

    print(f"\nOutput directory: {output_dir}")

    print("\nGenerated files:")
    print("\nCSV Files:")
    print("  - rf_model_performance_nc.csv              Model performance metrics (all models, Train/Test)")
    print("  - rf_feature_importance_nc.csv             Feature importance (combined table)")
    print("  - rf_predictions_all_models_nc.csv         Prediction results (combined)")
    print("  - rf_predictions__rf_basic.csv             Prediction results (RF_Basic)")
    print("  - rf_predictions__rf_optimized.csv         Prediction results (RF_Optimized)")
    print("  - rf_parameters__rf_basic.csv              Random Forest parameters (RF_Basic)")
    print("  - rf_parameters__rf_optimized.csv          Random Forest parameters (RF_Optimized)")
    print("  - metrics__all_models_train_test.csv       Metrics table (all models)")
    print("  - metrics__rf_basic__train_test.csv        Metrics table (RF_Basic)")
    print("  - metrics__rf_optimized__train_test.csv    Metrics table (RF_Optimized)")
    print("  - metrics__rf_basic__train.csv             Metrics row (RF_Basic, Train)")
    print("  - metrics__rf_basic__test.csv              Metrics row (RF_Basic, Test)")
    print("  - metrics__rf_optimized__train.csv         Metrics row (RF_Optimized, Train)")
    print("  - metrics__rf_optimized__test.csv          Metrics row (RF_Optimized, Test)")
    print("  - plot_metrics_ranking__test_only.csv      Plot data (test-only metrics ranking)")
    print("  - plot_ts_simple_sampled__rf_basic.csv     Plot data (simple time series, sampled, RF_Basic)")
    print("  - plot_ts_simple_sampled__rf_optimized.csv Plot data (simple time series, sampled, RF_Optimized)")
    print("  - plot_ts_lastyear_sampled__actual.csv     Plot data (last-year sampled, actual)")
    print("  - plot_ts_lastyear_sampled__rf_basic.csv   Plot data (last-year sampled, RF_Basic)")
    print("  - plot_ts_lastyear_sampled__rf_optimized.csv Plot data (last-year sampled, RF_Optimized)")
    print("  - plot_scatter__rf_basic.csv               Plot data (scatter, RF_Basic)")
    print("  - plot_scatter__rf_optimized.csv           Plot data (scatter, RF_Optimized)")
    print("  - plot_residuals__rf_basic.csv             Plot data (residuals, RF_Basic)")
    print("  - plot_residuals__rf_optimized.csv         Plot data (residuals, RF_Optimized)")
    print("  - plot_error_distribution__rf_basic.csv    Plot data (error distribution, RF_Basic)")
    print("  - plot_error_distribution__rf_optimized.csv Plot data (error distribution, RF_Optimized)")
    print("  - plot_feature_importance__rf_basic.csv    Plot data (feature importance, RF_Basic)")
    print("  - plot_feature_importance__rf_optimized.csv Plot data (feature importance, RF_Optimized)")
    print("  - plot_feature_importance_top15__rf_basic.csv Plot data (feature importance Top15, RF_Basic)")
    print("  - plot_feature_importance_top15__rf_optimized.csv Plot data (feature importance Top15, RF_Optimized)")

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
    print(f"  Test set samples: {len(X_test)}")
    print(f"  Number of features: {len(X_train.columns)}")

    print("\n" + "=" * 80)
    print("Random Forest PM2.5 Concentration Prediction Complete (NetCDF ERA5 + Time Series Features)!")
    print("=" * 80)
