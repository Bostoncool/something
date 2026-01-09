import os
import gc
import glob
import calendar
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm is not installed, progress display will use simplified version.")
    print("      You can use 'pip install tqdm' to install for better progress bar display.")

# 只忽略常见的弃用警告和未来警告，保留其他重要警告
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
np.random.seed(42)

CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

print("=" * 80)
print("SVR PM2.5 Concentration Prediction Model (NetCDF ERA5)")
print("=" * 80)

pollution_all_path = '/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)'
pollution_extra_path = '/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)'
era5_path = '/root/autodl-tmp/ERA5-Beijing-NC'

script_dir = Path(__file__).parent
output_dir = script_dir / 'output'
output_dir.mkdir(exist_ok=True)

start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
era5_vars = [
    'd2m', 't2m', 'u10', 'v10', 'u100', 'v100', 'blh', 'cvh', 'lsm', 'cvl',
    'avg_tprate', 'mn2t', 'sd', 'str', 'sp', 'tisr', 'tcwv', 'tp'
]

beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

print(f"Data time range: {start_date.date()} to {end_date.date()}")
print(f"Pollutants: {', '.join(pollutants)}")
print(f"Number of meteorological variables: {len(era5_vars)}")
print(f"Output directory: {output_dir}")
print(f"CPU cores: {CPU_COUNT}, parallel worker processes: {MAX_WORKERS}")


def slugify_model_name(name: str) -> str:
    """用于文件名的模型标识：小写 + 将特殊字符转为下划线。"""
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def save_csv(df: pd.DataFrame, path: Path, index: bool = False):
    """统一 CSV 输出（UTF-8-SIG），便于后续脚本读取与绘图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"✓ Saved CSV: {path}")


def evaluate_model(y_true, y_pred, model_name: str, dataset: str):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        "Model": model_name,
        "Dataset": dataset,
        "R²": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
    }


def daterange(start: datetime, end: datetime):
    """Generate date range inclusive of both start and end."""
    for n in range((end - start).days + 1):
        yield start + timedelta(days=n)


def build_file_dict(base_path: str, prefix: str):
    """Build a dictionary mapping date strings to file paths for O(1) lookup."""
    file_dict = {}
    filename_pattern = f"{prefix}_"
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename.startswith(filename_pattern) and filename.endswith('.csv'):
                # Extract date string from filename (format: prefix_YYYYMMDD.csv)
                date_str = filename[len(filename_pattern):-4]  # Remove prefix_ and .csv
                if len(date_str) == 8 and date_str.isdigit():  # Validate YYYYMMDD format
                    file_dict[date_str] = os.path.join(root, filename)
    return file_dict


def read_pollution_day(args):
    """Read pollution data for a single day.
    
    Args:
        args: tuple of (date, all_file_dict, extra_file_dict, pollutants)
    """
    date, all_file_dict, extra_file_dict, pollutants = args
    date_str = date.strftime('%Y%m%d')
    
    # O(1) dictionary lookup instead of O(n) file traversal
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
        df_poll = df_poll.melt(id_vars=['date', 'hour', 'type'],
                               var_name='station', value_name='value')
        df_poll['value'] = pd.to_numeric(df_poll['value'], errors='coerce')
        df_poll = df_poll[df_poll['value'] >= 0]

        df_daily = (
            df_poll
            .groupby(['date', 'type'])['value']
            .mean()
            .reset_index()
            .pivot(index='date', columns='type', values='value')
        )
        df_daily.index = pd.to_datetime(df_daily.index, format='%Y%m%d', errors='coerce')
        available_pollutants = [p for p in pollutants if p in df_daily.columns]
        return df_daily[available_pollutants]
    except Exception:
        return None


def read_all_pollution():
    """Read all pollution data in parallel using multiprocessing."""
    print("\nLoading pollution data...")
    print(f"Using {MAX_WORKERS} parallel worker processes")
    
    # Build file dictionaries once for O(1) lookup
    print("  Building file path dictionaries...")
    all_file_dict = build_file_dict(pollution_all_path, 'beijing_all')
    extra_file_dict = build_file_dict(pollution_extra_path, 'beijing_extra')
    print(f"  Found {len(all_file_dict)} all files and {len(extra_file_dict)} extra files")

    dates = list(daterange(start_date, end_date))
    pollution_dfs = []
    
    # Prepare arguments for multiprocessing
    tasks = [(date, all_file_dict, extra_file_dict, pollutants) for date in dates]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, task): task[0] for task in tasks}

        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Loading pollution data", unit="days"):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
        else:
            for idx, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
                if idx % 500 == 0 or idx == len(futures):
                    print(f"  Processed {idx}/{len(futures)} days "
                          f"({idx / len(futures) * 100:.1f}%)")

    if not pollution_dfs:
        return pd.DataFrame()

    print(f"  Successfully read {len(pollution_dfs)}/{len(dates)} days of data")
    print("  Merging data...")
    df_poll_all = pd.concat(pollution_dfs)
    df_poll_all.ffill(inplace=True)
    # 使用numeric_only=True避免对非数值列的警告
    df_poll_all.fillna(df_poll_all.mean(numeric_only=True), inplace=True)
    print(f"Pollution data loading complete, shape: {df_poll_all.shape}")
    return df_poll_all


def read_era5_month(args):
    """Load ERA5 data for a single month from NetCDF files.
    
    Args:
        args: tuple of (year, month, era5_path, era5_vars, beijing_lats, beijing_lons)
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
    end_day = calendar.monthrange(year, month)[1]
    month_start = pd.to_datetime(f"{year}-{month:02d}-01")
    month_end = pd.to_datetime(f"{year}-{month:02d}-{end_day:02d}") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    for file_path in all_files:
        try:
            with Dataset(file_path, mode='r') as nc_file:
                available_vars = [v for v in era5_vars if v in nc_file.variables]
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

                drop_vars = [var for var in ("expver", "surface") if var in ds]
                if drop_vars:
                    ds = ds.drop_vars(drop_vars)

                if "number" in ds.dims:
                    ds = ds.mean(dim="number", skipna=True)

                ds_subset = ds[available_vars]
                if "time" not in ds_subset.coords:
                    continue

                ds_subset = ds_subset.sortby("time")

                if fallback_used:
                    try:
                        ds_subset = ds_subset.sel(time=slice(month_start, month_end))
                    except Exception:
                        continue
                    if ds_subset.sizes.get("time", 0) == 0:
                        continue

                if "latitude" in ds_subset.coords and "longitude" in ds_subset.coords:
                    lat_values = ds_subset["latitude"]
                    if lat_values[0] > lat_values[-1]:
                        lat_slice = slice(beijing_lats.max(), beijing_lats.min())
                    else:
                        lat_slice = slice(beijing_lats.min(), beijing_lats.max())
                    ds_subset = ds_subset.sel(
                        latitude=lat_slice,
                        longitude=slice(beijing_lons.min(), beijing_lons.max())
                    )
                    if {'latitude', 'longitude'}.issubset(ds_subset.dims):
                        ds_subset = ds_subset.mean(dim=['latitude', 'longitude'], skipna=True)

                ds_daily = ds_subset.resample(time='1D').mean(keep_attrs=False)
                ds_daily = ds_daily.dropna('time', how='all')
                if ds_daily.sizes.get('time', 0) == 0:
                    continue

                monthly_datasets.append(ds_daily.load())
                print(f"  [+] {os.path.basename(file_path)} -> {year}-{month:02d} "
                      f"days: {ds_daily.sizes.get('time', 0)}, vars: {len(ds_daily.data_vars)}")
        except Exception as exc:
            print(f"[WARN] Failed to read {os.path.basename(file_path)}: {type(exc).__name__}: {exc}")
            continue

    if not monthly_datasets:
        return None

    merged_ds = xr.merge(monthly_datasets, compat='override', join='outer')
    df_month = merged_ds.to_dataframe()
    df_month.index = pd.to_datetime(df_month.index)
    df_month = df_month.groupby(df_month.index).mean()
    return df_month if not df_month.empty else None


def read_all_era5():
    """Read all ERA5 data in parallel from NetCDF files using multiprocessing."""
    print("\nLoading meteorological data (NetCDF)...")
    print(f"Using {MAX_WORKERS} parallel worker processes")
    print(f"Meteorological data directory: {era5_path}")
    print(f"Directory exists: {os.path.exists(era5_path)}")

    if os.path.exists(era5_path):
        all_nc = glob.glob(os.path.join(era5_path, "**", "*.nc"), recursive=True)
        print(f"Found {len(all_nc)} NetCDF files")
        if all_nc:
            print(f"Example files: {[os.path.basename(f) for f in all_nc[:5]]}")

    era5_dfs = []

    years = range(start_date.year, end_date.year + 1)
    month_tasks = []
    for year in years:
        for month in range(1, 13):
            if year == start_date.year and month < start_date.month:
                continue
            if year == end_date.year and month > end_date.month:
                continue
            month_tasks.append((year, month))

    print(f"Attempting to load {len(month_tasks)} months of data...")
    
    # Prepare arguments for multiprocessing
    tasks = [(year, month, era5_path, era5_vars, beijing_lats, beijing_lons) 
             for year, month in month_tasks]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_era5_month, task): (task[0], task[1])
                   for task in tasks}
        successful_reads = 0

        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Loading meteorological data", unit="month"):
                result = future.result()
                if result is not None and not result.empty:
                    era5_dfs.append(result)
                    successful_reads += 1
        else:
            for idx, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None and not result.empty:
                    era5_dfs.append(result)
                    successful_reads += 1
                if idx % 20 == 0 or idx == len(futures):
                    print(f"  Progress: {idx}/{len(futures)} months "
                          f"(Success: {successful_reads}, {idx / len(futures) * 100:.1f}%)")

        print(f"  Total successfully read: {successful_reads}/{len(futures)} months")

    if not era5_dfs:
        print("\n❌ Error: No meteorological data files loaded successfully!")
        print("Possible reasons:")
        print("1. File naming format does not match (Expected format: *YYYYMM*.nc)")
        print("2. File content format is incorrect (Missing time coordinate)")
        print("3. File path is incorrect")
        return pd.DataFrame()

    print("\nMerging meteorological data...")
    df_era5_all = pd.concat(era5_dfs, axis=0)

    print("  Deduplicating...")
    df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep='first')]

    print("  Sorting...")
    df_era5_all.sort_index(inplace=True)

    print(f"Shape after merge: {df_era5_all.shape}")
    print(f"Time range: {df_era5_all.index.min()} to {df_era5_all.index.max()}")
    preview_cols = list(df_era5_all.columns[:10])
    print(f"Available variables: {preview_cols}..."
          if len(df_era5_all.columns) > 10 else f"Available variables: {preview_cols}")

    print("  Handling missing values...")
    initial_na = df_era5_all.isna().sum().sum()
    df_era5_all.ffill(inplace=True)
    df_era5_all.bfill(inplace=True)
    # 使用numeric_only=True避免对非数值列的警告
    df_era5_all.fillna(df_era5_all.mean(numeric_only=True), inplace=True)
    final_na = df_era5_all.isna().sum().sum()

    print(f"Missing value handling: {initial_na} -> {final_na}")
    print(f"Meteorological data loading complete, shape: {df_era5_all.shape}")

    return df_era5_all


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("Step 1: Data Loading and Preprocessing")
    print("=" * 80)

    df_pollution = read_all_pollution()
    df_era5 = read_all_era5()
    gc.collect()

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

    print(f"  Pollution data time range: {df_pollution.index.min()} to {df_pollution.index.max()}")
    print(f"  Meteorological data time range: {df_era5.index.min()} to {df_era5.index.max()}")

    print("\nMerging data...")
    df_combined = df_pollution.join(df_era5, how='inner')

    if df_combined.empty:
        print("\n❌ Error: Data is empty after merging!")
        print("   Possible reason: No overlapping date indices between pollution and meteorological data.")
        print(f"   Pollution data rows: {len(df_pollution)}")
        print(f"   Meteorological data rows: {len(df_era5)}")
        print(f"   After merge rows: {len(df_combined)}")
        raise SystemExit(1)

    df_combined.sort_index(inplace=True)

    if {'u10', 'v10'}.issubset(df_combined.columns):
        df_combined['wind_speed_10m'] = np.sqrt(df_combined['u10'] ** 2 + df_combined['v10'] ** 2)
        df_combined['wind_dir_10m'] = np.degrees(np.arctan2(df_combined['v10'], df_combined['u10']))
        df_combined['wind_dir_10m'] = (df_combined['wind_dir_10m'] + 360) % 360

    if {'u100', 'v100'}.issubset(df_combined.columns):
        df_combined['wind_speed_100m'] = np.sqrt(df_combined['u100'] ** 2 + df_combined['v100'] ** 2)

    if 't2m' in df_combined.columns:
        df_combined['t2m_celsius'] = df_combined['t2m'] - 273.15

    if 'd2m' in df_combined.columns:
        df_combined['d2m_celsius'] = df_combined['d2m'] - 273.15

    if {'t2m_celsius', 'd2m_celsius'}.issubset(df_combined.columns):
        df_combined['relative_humidity'] = 100 * np.exp(
            (17.625 * df_combined['d2m_celsius']) / (243.04 + df_combined['d2m_celsius'])
        ) / np.exp(
            (17.625 * df_combined['t2m_celsius']) / (243.04 + df_combined['t2m_celsius'])
        )
        df_combined['relative_humidity'] = df_combined['relative_humidity'].clip(0, 100)

    df_combined['year'] = df_combined.index.year
    df_combined['month'] = df_combined.index.month
    df_combined['day'] = df_combined.index.day
    df_combined['dayofyear'] = df_combined.index.dayofyear
    df_combined['dayofweek'] = df_combined.index.dayofweek
    df_combined['season'] = df_combined['month'].apply(lambda x: (x % 12 + 3) // 3)
    df_combined['is_winter'] = df_combined['month'].isin([12, 1, 2]).astype(int)
    df_combined['month_sin'] = np.sin(2 * np.pi * df_combined['month'] / 12)
    df_combined['month_cos'] = np.cos(2 * np.pi * df_combined['month'] / 12)
    df_combined['day_sin'] = np.sin(2 * np.pi * df_combined['dayofyear'] / 365)
    df_combined['day_cos'] = np.cos(2 * np.pi * df_combined['dayofyear'] / 365)

    print("\nCleaning data...")
    df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
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

    print("\n" + "=" * 80)
    print("Preparing Training Data")
    print("=" * 80)

    if 'PM2.5' not in df_combined.columns:
        raise ValueError("Data does not contain PM2.5 column!")

    target = 'PM2.5'
    exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']
    feature_cols = [col for col in df_combined.columns if col not in exclude_cols]

    X = df_combined[feature_cols]
    y = df_combined[target]

    if len(X) == 0 or len(y) == 0:
        print("\n" + "=" * 80)
        print("❌ Error: No available data!")
        print("=" * 80)
        print("\nPossible reasons:")
        print("1. Data path is incorrect, unable to find data files")
        print("2. Pollution or meteorological data loading failed")
        print("3. No overlapping indices after data merge (check if date ranges match)")
        print("4. Data cleaning removed all rows")
        raise SystemExit(1)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"Number of selected features: {len(feature_cols)}")
    print(f"\nPM2.5 Statistics:")
    print(f"  Mean: {y.mean():.2f} μg/m³")
    print(f"  Std Dev: {y.std():.2f} μg/m³")
    print(f"  Min: {y.min():.2f} μg/m³")
    print(f"  Max: {y.max():.2f} μg/m³")
    print(f"  Median: {y.median():.2f} μg/m³")

    print("\nUsing time series split method (70% train, 15% validation, 15% test)")
    df_combined_sorted = df_combined.sort_index()
    X = df_combined_sorted[feature_cols]
    y = df_combined_sorted[target]

    n_total = len(X)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    X_val = X.iloc[n_train:n_train + n_val]
    y_val = y.iloc[n_train:n_train + n_val]
    X_test = X.iloc[n_train + n_val:]
    y_test = y.iloc[n_train + n_val:]

    print(f"\nTraining Set: {X_train.shape[0]} samples ({X_train.shape[0] / len(X) * 100:.1f}%)")
    print(f"  Time Range: {y_train.index.min().date()} to {y_train.index.max().date()}")
    print(f"Validation Set: {X_val.shape[0]} samples ({X_val.shape[0] / len(X) * 100:.1f}%)")
    print(f"  Time Range: {y_val.index.min().date()} to {y_val.index.max().date()}")
    print(f"Test Set: {X_test.shape[0]} samples ({X_test.shape[0] / len(X) * 100:.1f}%)")
    print(f"  Time Range: {y_test.index.min().date()} to {y_test.index.max().date()}")

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    print("\nFeature standardization complete")

    print("\n" + "=" * 80)
    print("SVR Model Training")
    print("=" * 80)

    models = {
        'SVR-RBF': SVR(kernel='rbf', cache_size=1000),
        'SVR-Linear': SVR(kernel='linear', cache_size=1000),
        'SVR-Poly': SVR(kernel='poly', degree=3, cache_size=1000)
    }

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

    best_models = {}
    results = []
    metrics_rows = []
    predictions_by_model = {}

    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Training model: {name}")
        print(f"{'=' * 60}")

        print("Starting grid search...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train_scaled, y_train_scaled)
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score (negative MSE): {grid_search.best_score_:.4f}")

        best_model = grid_search.best_estimator_
        best_models[name] = best_model

        def inv_scale(pred_scaled: np.ndarray) -> np.ndarray:
            return scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

        y_train_pred = inv_scale(best_model.predict(X_train_scaled))
        y_val_pred = inv_scale(best_model.predict(X_val_scaled))
        y_test_pred = inv_scale(best_model.predict(X_test_scaled))

        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        print("\nValidation set performance:")
        print(f"  RMSE: {val_rmse:.4f}")
        print(f"  MAE: {val_mae:.4f}")
        print(f"  R²: {val_r2:.4f}")

        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print("\nTest set performance:")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  R²: {test_r2:.4f}")

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

        metrics_rows.extend([
            evaluate_model(y_train, y_train_pred, name, "Train"),
            evaluate_model(y_val, y_val_pred, name, "Validation"),
            evaluate_model(y_test, y_test_pred, name, "Test"),
        ])
        predictions_by_model[name] = {
            "Train": y_train_pred,
            "Validation": y_val_pred,
            "Test": y_test_pred,
        }
        gc.collect()

    print("\n" + "=" * 80)
    print("Model Performance Comparison")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    save_csv(results_df, output_dir / "svr_model_comparison.csv")

    metrics_df = pd.DataFrame(metrics_rows)
    save_csv(metrics_df, output_dir / "metrics__all_models_train_validation_test.csv")

    for model in metrics_df["Model"].unique():
        model_slug = slugify_model_name(model)
        save_csv(metrics_df[metrics_df["Model"] == model], output_dir / f"metrics__{model_slug}__train_validation_test.csv")
        for ds_name in metrics_df["Dataset"].unique():
            part = metrics_df[(metrics_df["Model"] == model) & (metrics_df["Dataset"] == ds_name)]
            if not part.empty:
                save_csv(part, output_dir / f"metrics__{model_slug}__{ds_name.lower()}.csv")

    test_ranking = (
        metrics_df[metrics_df["Dataset"] == "Test"]
        .sort_values("R²", ascending=False)
        .reset_index(drop=True)
    )
    save_csv(test_ranking, output_dir / "plot_metrics_ranking__test_only.csv")

    best_model_name = test_ranking.iloc[0]["Model"]
    best_row = results_df.loc[results_df["Model"] == best_model_name].iloc[0]

    best_test_rmse = float(test_ranking.iloc[0]["RMSE"])
    best_test_mae = float(test_ranking.iloc[0]["MAE"])
    best_test_r2 = float(test_ranking.iloc[0]["R²"])
    best_params = best_row["Best Parameters"]

    print(f"\nBest model: {best_model_name}")
    print(f"Test set R²: {best_test_r2:.4f}")

    print("\n" + "=" * 80)
    print("Export CSV Files for Plotting (No figures will be generated)")
    print("=" * 80)

    # 1) feature names
    save_csv(pd.DataFrame({"Feature": feature_cols}), output_dir / "feature_names.csv")

    # 2) per-model / per-split scatter & residual & error distribution CSV
    for model_name, preds in predictions_by_model.items():
        model_slug = slugify_model_name(model_name)
        for ds_name, y_pred in preds.items():
            ds_slug = ds_name.lower()
            y_true = {"Train": y_train, "Validation": y_val, "Test": y_test}[ds_name]

            scatter_df = pd.DataFrame({
                "Date": y_true.index,
                "Actual_PM25": y_true.values,
                "Predicted_PM25": y_pred,
            })
            save_csv(scatter_df, output_dir / f"plot_scatter__{model_slug}__{ds_slug}.csv")

            residual = y_true.values - y_pred
            residual_df = scatter_df.assign(Residual=residual)
            save_csv(residual_df, output_dir / f"plot_residuals__{model_slug}__{ds_slug}.csv")

            error_df = scatter_df.assign(Error=residual)
            save_csv(error_df, output_dir / f"plot_error_distribution__{model_slug}__{ds_slug}.csv")

    # 3) time series CSV (test only) — simple sampled
    step = 4
    for model_name, preds in predictions_by_model.items():
        model_slug = slugify_model_name(model_name)
        ts_df = pd.DataFrame({
            "time": y_test.index,
            "y_true": y_test.values,
            "y_pred": preds["Test"],
        }).sort_values("time").reset_index(drop=True)
        ts_df_sampled = ts_df.iloc[::step].copy()
        save_csv(ts_df_sampled, output_dir / f"plot_ts_simple_sampled__{model_slug}.csv")

    # 4) time series CSV (test only) — last year sampled + common x_axis
    plot_df = pd.DataFrame({"time": y_test.index, "y_true": y_test.values}).sort_values("time").reset_index(drop=True)
    plot_range = min(365, len(plot_df))
    plot_df_subset = plot_df.iloc[-plot_range:].copy()
    plot_df_sampled = plot_df_subset.iloc[::step].copy().reset_index(drop=True)
    x_axis = np.arange(len(plot_df_sampled))

    ts_common = pd.DataFrame({
        "x_axis": x_axis,
        "time": plot_df_sampled["time"].values,
        "y_true": plot_df_sampled["y_true"].values,
    })
    save_csv(ts_common, output_dir / "plot_ts_lastyear_sampled__actual.csv")

    for model_name, preds in predictions_by_model.items():
        model_slug = slugify_model_name(model_name)
        preds_test = pd.Series(preds["Test"], index=y_test.index).sort_index()
        preds_test_df = pd.DataFrame({"time": preds_test.index, "y_pred": preds_test.values}).sort_values("time").reset_index(drop=True)
        preds_test_subset = preds_test_df.iloc[-plot_range:].copy()
        preds_test_sampled = preds_test_subset.iloc[::step].copy().reset_index(drop=True)
        ts_model = ts_common.assign(y_pred=preds_test_sampled["y_pred"].values)
        save_csv(ts_model, output_dir / f"plot_ts_lastyear_sampled__{model_slug}.csv")

    print("\n" + "=" * 80)
    print("Generating Detailed Report")
    print("=" * 80)

    report_content = f"""
    SVR PM2.5 Concentration Prediction Model Report (NetCDF ERA5)
    {'=' * 80}

    1. Data Overview
       - Data time range: {start_date.date()} to {end_date.date()}
       - Total number of samples: {len(df_combined)}
       - Number of features: {len(feature_cols)}
       - Target variable: PM2.5 Concentration (μg/m³)

    2. Dataset Split
       - Training set: {X_train.shape[0]} samples ({X_train.shape[0] / len(X) * 100:.1f}%)
       - Validation set: {X_val.shape[0]} samples ({X_val.shape[0] / len(X) * 100:.1f}%)
       - Test set: {X_test.shape[0]} samples ({X_test.shape[0] / len(X) * 100:.1f}%)

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
       - Test R²: {best_test_r2:.4f}
       - Test RMSE: {best_test_rmse:.4f}
       - Test MAE: {best_test_mae:.4f}
       - Best Parameters: {best_params}

    6. Feature Engineering
       - Original pollutant features: {', '.join(pollutants)}
       - ERA5 meteorological features: {', '.join([v for v in era5_vars if v in df_combined.columns])}
       - Derived features: Wind speed, wind direction, temperature conversion, relative humidity, seasonal cycles

    7. Model Description
       Support Vector Regression (SVR) maps features to high-dimensional space via kernel
       functions to capture nonlinear relationships between PM2.5 and meteorological
       factors. Three kernel variants (RBF, Linear, Polynomial) are tuned via grid search.

    8. File List
       - SVR-NC.py: Main program
       - svr_model_comparison.csv: Model performance comparison (legacy table)
       - metrics__*.csv: Metrics per model/split (Train/Validation/Test)
       - plot_scatter__*.csv: Scatter data per model/split
       - plot_residuals__*.csv: Residual data per model/split
       - plot_error_distribution__*.csv: Error distribution data per model/split
       - plot_ts_simple_sampled__*.csv: Simple test time series (sampled) per model
       - plot_ts_lastyear_sampled__*.csv: Last-year test time series (sampled) per model + actual
       - feature_names.csv: Feature name list

    {'=' * 80}
    Report generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    with open(output_dir / 'model_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(report_content)
    print("\n  Saved detailed report: model_report.txt")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

    print("\nGenerated files:")
    print("\nCSV files (output directory):")
    print("  - svr_model_comparison.csv                    Model performance comparison (legacy table)")
    print("  - metrics__all_models_train_validation_test.csv  Metrics table (all models, Train/Validation/Test)")
    print("  - metrics__{model}__*.csv                     Metrics per model/split")
    print("  - plot_scatter__{model}__{split}.csv          Scatter data per model/split")
    print("  - plot_residuals__{model}__{split}.csv        Residual data per model/split")
    print("  - plot_error_distribution__{model}__{split}.csv Error distribution data per model/split")
    print("  - plot_ts_simple_sampled__{model}.csv         Simple test time series (sampled) per model")
    print("  - plot_ts_lastyear_sampled__actual.csv        Last-year sampled (actual)")
    print("  - plot_ts_lastyear_sampled__{model}.csv       Last-year sampled (per model)")
    print("  - feature_names.csv                           Feature name list")

    print("\nReport files (output directory):")
    print("  - model_report.txt            Detailed analysis report")

    print(f"\nOutput directory: {output_dir}")

    del df_pollution, df_era5, df_combined, X_train_scaled, X_val_scaled, X_test_scaled
    gc.collect()

    print("\n" + "=" * 80)
    print("SVR PM2.5 Concentration Prediction Complete!")
    print("=" * 80)

