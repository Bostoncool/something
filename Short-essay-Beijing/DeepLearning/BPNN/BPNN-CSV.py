import os
import glob
import time
import warnings
import multiprocessing
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from itertools import product

warnings.filterwarnings("ignore")

CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

# 设置多进程启动方法，确保使用真正的多进程而非多线程（netcdf4 相关库非线程安全）
if hasattr(multiprocessing, "set_start_method"):
    try:
        if os.name != "nt":
            multiprocessing.set_start_method("fork", force=True)
            print("Multiprocessing start method: fork")
        else:
            multiprocessing.set_start_method("spawn", force=True)
            print("Multiprocessing start method: spawn")
    except RuntimeError:
        pass

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TQDM_AVAILABLE = False

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - BPNN Model (CSV outputs only)")
print("=" * 80)

# ---------------------------------------------------------------------------
# Data paths (保持与原脚本一致)
# ---------------------------------------------------------------------------
pollution_all_path = "/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)"
pollution_extra_path = "/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)"
era5_path = "/root/autodl-tmp/ERA5-Beijing-NC"

output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

WINDOW_SIZE = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
NON_BLOCKING = PIN_MEMORY

DATALOADER_WORKERS = min(16, MAX_WORKERS) if DEVICE.type == "cuda" else 0
PERSISTENT_WORKERS = DATALOADER_WORKERS > 0
PREFETCH_FACTOR = 4 if DATALOADER_WORKERS > 0 else 2

USE_AMP = True if DEVICE.type == "cuda" else False


# ---------------------------------------------------------------------------
# Output helpers (统一 UTF-8-SIG)
# ---------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, path: Path, index: bool = False):
    """统一 CSV 输出（UTF-8-SIG），便于后续脚本读取与绘图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"✓ Saved CSV: {path}")


# ---------------------------------------------------------------------------
# GPU batch size probe (保留原逻辑)
# ---------------------------------------------------------------------------
def get_optimal_batch_size(
    model_class,
    window_size,
    num_features,
    device,
    min_batch=64,
    max_batch=512,
    step=32,
    **model_kwargs,
):
    if device.type != "cuda":
        return 32

    torch.cuda.empty_cache()
    test_model = model_class(window_size=window_size, num_features=num_features, **model_kwargs).to(device)
    test_model.train()

    test_X = torch.randn(1, window_size, num_features, dtype=torch.float32).to(device)
    test_y = torch.randn(1, dtype=torch.float32).to(device)

    optimal_batch = min_batch
    current_batch = min_batch
    test_optimizer = optim.Adam(test_model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    while current_batch <= max_batch:
        try:
            torch.cuda.empty_cache()
            test_model.zero_grad()
            batch_X = test_X.repeat(current_batch, 1, 1)
            batch_y = test_y.repeat(current_batch)

            if USE_AMP:
                with torch.cuda.amp.autocast():
                    y_pred = test_model(batch_X)
                    loss = nn.MSELoss()(y_pred, batch_y)
                scaler.scale(loss).backward()
                scaler.step(test_optimizer)
                scaler.update()
            else:
                y_pred = test_model(batch_X)
                loss = nn.MSELoss()(y_pred, batch_y)
                loss.backward()
                test_optimizer.step()

            optimal_batch = current_batch
            current_batch += step
            del batch_X, batch_y, y_pred, loss
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                test_model.zero_grad()
                if scaler:
                    scaler.update()
                break
            test_model.zero_grad()
            if scaler:
                scaler.update()
            raise

    optimal_batch = int(optimal_batch * 0.9)
    if optimal_batch < min_batch:
        optimal_batch = min_batch

    del test_model, test_X, test_y, test_optimizer
    if scaler:
        del scaler
    torch.cuda.empty_cache()
    return optimal_batch


BATCH_SIZE = 128
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


def daterange(start, end):
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)


def build_file_index(base_path, prefix):
    file_index = {}
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename.startswith(prefix) and filename.endswith(".csv"):
                date_str = filename.replace(f"{prefix}_", "").replace(".csv", "")
                if len(date_str) == 8 and date_str.isdigit():
                    file_index[date_str] = os.path.join(root, filename)
    return file_index


def read_pollution_day(args):
    date, file_index_all, file_index_extra, pollutants_local = args
    date_str = date.strftime("%Y%m%d")
    all_file = file_index_all.get(date_str)
    extra_file = file_index_extra.get(date_str)
    if not all_file or not extra_file:
        return None
    if not os.path.exists(all_file) or not os.path.exists(extra_file):
        return None

    try:
        df_all = pd.read_csv(all_file, encoding="utf-8", on_bad_lines="skip")
        df_extra = pd.read_csv(extra_file, encoding="utf-8", on_bad_lines="skip")

        df_all = df_all[~df_all["type"].str.contains("_24h|AQI", na=False)]
        df_extra = df_extra[~df_extra["type"].str.contains("_24h", na=False)]

        df_poll = pd.concat([df_all, df_extra], ignore_index=True)
        df_poll = df_poll.melt(id_vars=["date", "hour", "type"], var_name="station", value_name="value")
        df_poll["value"] = pd.to_numeric(df_poll["value"], errors="coerce")
        df_poll = df_poll[df_poll["value"] >= 0]

        df_daily = df_poll.groupby(["date", "type"])["value"].mean().reset_index()
        df_daily = df_daily.pivot(index="date", columns="type", values="value")
        df_daily.index = pd.to_datetime(df_daily.index, format="%Y%m%d", errors="coerce")
        df_daily = df_daily[[c for c in pollutants_local if c in df_daily.columns]]
        return df_daily
    except Exception:
        return None


def read_all_pollution():
    print("\nLoading pollution data...")
    print(f"Using {MAX_WORKERS} parallel workers (multiprocessing)")

    print("Building file index dictionary...")
    file_index_all = build_file_index(pollution_all_path, "beijing_all")
    file_index_extra = build_file_index(pollution_extra_path, "beijing_extra")
    print(f"  Found {len(file_index_all)} files in all directory")
    print(f"  Found {len(file_index_extra)} files in extra directory")

    dates = list(daterange(start_date, end_date))
    args_list = [(date, file_index_all, file_index_extra, pollutants) for date in dates]

    pollution_dfs = []
    with Pool(processes=MAX_WORKERS) as pool:
        if TQDM_AVAILABLE:
            results = list(
                tqdm(pool.imap(read_pollution_day, args_list), total=len(args_list), desc="Loading pollution data", unit="day")
            )
        else:
            results = pool.map(read_pollution_day, args_list)

        for result in results:
            if result is not None:
                pollution_dfs.append(result)

    if not pollution_dfs:
        return pd.DataFrame()

    df_poll_all = pd.concat(pollution_dfs)
    df_poll_all.ffill(inplace=True)
    df_poll_all.fillna(df_poll_all.mean(), inplace=True)
    print(f"Pollution data loaded: {df_poll_all.shape}")
    return df_poll_all


def read_single_era5_file(args):
    file_path, beijing_lat_min, beijing_lat_max, beijing_lon_min, beijing_lon_max = args
    try:
        with xr.open_dataset(file_path, engine="netcdf4", decode_times=True, lock=False) as ds:
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

            drop_vars = [c for c in ("expver", "surface") if c in ds]
            if drop_vars:
                ds = ds.drop_vars(drop_vars)

            if "number" in ds.dims:
                ds = ds.mean(dim="number", skipna=True)

            if "time" not in ds.coords:
                return None

            data_vars = [v for v in ds.data_vars if v not in drop_vars]
            if not data_vars:
                return None

            ds = ds.sortby("time")

            if "latitude" in ds.coords and "longitude" in ds.coords:
                lat_values = ds["latitude"]
                if len(lat_values) > 0:
                    if lat_values[0] > lat_values[-1]:
                        lat_slice = slice(beijing_lat_max, beijing_lat_min)
                    else:
                        lat_slice = slice(beijing_lat_min, beijing_lat_max)
                    ds = ds.sel(latitude=lat_slice, longitude=slice(beijing_lon_min, beijing_lon_max))
                    if "latitude" in ds.dims and "longitude" in ds.dims:
                        ds = ds.mean(dim=["latitude", "longitude"], skipna=True)

            ds_daily = ds.resample(time="1D").mean(keep_attrs=False)
            ds_daily = ds_daily.dropna("time", how="all")
            if ds_daily.sizes.get("time", 0) == 0:
                return None

            ds_daily = ds_daily.load()

            result = {}
            for var in data_vars:
                if var in ds_daily.data_vars:
                    result[var] = ds_daily[[var]]
            return result if result else None
    except Exception:
        return None


def read_all_era5():
    print("\nLoading meteorological data...")
    print(f"Using {MAX_WORKERS} parallel workers (multiprocessing)")
    print(f"Meteorological data directory: {era5_path}")
    print(f"Directory exists: {os.path.exists(era5_path)}")

    if not os.path.exists(era5_path):
        print(f"Error: Directory {era5_path} does not exist!")
        return pd.DataFrame()

    print("\nSearching for NetCDF files...")
    all_nc_files = glob.glob(os.path.join(era5_path, "**", "*.nc"), recursive=True)
    print(f"Found {len(all_nc_files)} NetCDF files")
    if not all_nc_files:
        print("Error: No NetCDF files found!")
        return pd.DataFrame()

    print(f"\nReading {len(all_nc_files)} files in parallel...")
    variable_datasets = {}
    successful_files = 0
    failed_files = 0

    beijing_lat_min = float(beijing_lats.min())
    beijing_lat_max = float(beijing_lats.max())
    beijing_lon_min = float(beijing_lons.min())
    beijing_lon_max = float(beijing_lons.max())

    args_list = [(file_path, beijing_lat_min, beijing_lat_max, beijing_lon_min, beijing_lon_max) for file_path in all_nc_files]

    with Pool(processes=MAX_WORKERS) as pool:
        if TQDM_AVAILABLE:
            results = list(
                tqdm(
                    pool.imap(read_single_era5_file, args_list),
                    total=len(args_list),
                    desc="Reading NetCDF files (multiprocessing)",
                    unit="file",
                )
            )
        else:
            results = pool.map(read_single_era5_file, args_list)

        for i, result in enumerate(results, 1):
            try:
                if result is not None:
                    for var_name, var_ds in result.items():
                        variable_datasets.setdefault(var_name, []).append(var_ds)
                    successful_files += 1
                else:
                    failed_files += 1
            except Exception:
                failed_files += 1
            if not TQDM_AVAILABLE and (i % 200 == 0 or i == len(results)):
                print(
                    f"  Progress: {i}/{len(results)} files (success: {successful_files}, failed: {failed_files}, {i/len(results)*100:.1f}%)"
                )

    print("\nFile reading complete:")
    print(f"  Successfully read: {successful_files} files")
    print(f"  Failed: {failed_files} files")
    print(f"  Variables found: {len(variable_datasets)}")
    if not variable_datasets:
        print("\nError: No variables were extracted from files!")
        return pd.DataFrame()

    print("\nMerging datasets by variable...")
    merged_variables = {}
    for var_name, ds_list in variable_datasets.items():
        if not ds_list:
            continue
        try:
            merged_ds = xr.merge(ds_list, compat="override", join="outer")
            df_var = merged_ds.to_dataframe()
            if "time" in df_var.index.names:
                df_var.index = pd.to_datetime(df_var.index.get_level_values("time"))
            elif isinstance(df_var.index, pd.DatetimeIndex):
                df_var.index = pd.to_datetime(df_var.index)
            df_var = df_var[~df_var.index.duplicated(keep="first")]
            df_var.sort_index(inplace=True)
            var_cols = [col for col in df_var.columns if col == var_name]
            if var_cols:
                merged_variables[var_name] = df_var[var_cols]
        except Exception as exc:
            print(f"  [ERROR] Failed to merge {var_name}: {type(exc).__name__}: {exc}")
            continue

    if not merged_variables:
        print("\nError: No variables were successfully merged!")
        return pd.DataFrame()

    print("\nMerging all variables...")
    df_era5_all = pd.concat(list(merged_variables.values()), axis=1, join="outer")
    df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep="first")]
    df_era5_all.sort_index(inplace=True)

    print(f"\nMerged shape: {df_era5_all.shape}")
    print(f"Time range: {df_era5_all.index.min()} to {df_era5_all.index.max()}")

    print(f"\nFiltering by date range ({start_date.date()} to {end_date.date()})...")
    df_era5_all = df_era5_all.loc[(df_era5_all.index >= start_date) & (df_era5_all.index <= end_date)]

    print("\nHandling missing values...")
    df_era5_all.ffill(inplace=True)
    df_era5_all.bfill(inplace=True)
    df_era5_all.fillna(df_era5_all.mean(), inplace=True)
    print(f"Meteorological data loading complete, shape: {df_era5_all.shape}")
    return df_era5_all


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    if "u10" in df_copy and "v10" in df_copy:
        df_copy["wind_speed_10m"] = np.sqrt(df_copy["u10"] ** 2 + df_copy["v10"] ** 2)
        df_copy["wind_dir_10m"] = np.arctan2(df_copy["v10"], df_copy["u10"]) * 180 / np.pi
        df_copy["wind_dir_10m"] = (df_copy["wind_dir_10m"] + 360) % 360

    if "u100" in df_copy and "v100" in df_copy:
        df_copy["wind_speed_100m"] = np.sqrt(df_copy["u100"] ** 2 + df_copy["v100"] ** 2)
        df_copy["wind_dir_100m"] = np.arctan2(df_copy["v100"], df_copy["u100"]) * 180 / np.pi
        df_copy["wind_dir_100m"] = (df_copy["wind_dir_100m"] + 360) % 360

    df_copy["year"] = df_copy.index.year
    df_copy["month"] = df_copy.index.month
    df_copy["day"] = df_copy.index.day
    df_copy["day_of_year"] = df_copy.index.dayofyear
    df_copy["day_of_week"] = df_copy.index.dayofweek
    df_copy["week_of_year"] = df_copy.index.isocalendar().week.astype(int)

    df_copy["season"] = df_copy["month"].apply(lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4)
    df_copy["is_heating_season"] = ((df_copy["month"] >= 11) | (df_copy["month"] <= 3)).astype(int)

    if "t2m" in df_copy and "d2m" in df_copy:
        df_copy["temp_dewpoint_diff"] = df_copy["t2m"] - df_copy["d2m"]

    if "PM2.5" in df_copy:
        df_copy["PM2.5_lag1"] = df_copy["PM2.5"].shift(1)
        df_copy["PM2.5_lag3"] = df_copy["PM2.5"].shift(3)
        df_copy["PM2.5_lag7"] = df_copy["PM2.5"].shift(7)
        df_copy["PM2.5_ma3"] = df_copy["PM2.5"].rolling(window=3, min_periods=1).mean()
        df_copy["PM2.5_ma7"] = df_copy["PM2.5"].rolling(window=7, min_periods=1).mean()
        df_copy["PM2.5_ma30"] = df_copy["PM2.5"].rolling(window=30, min_periods=1).mean()

    if "t2m" in df_copy and "d2m" in df_copy:
        df_copy["relative_humidity"] = (
            100
            * np.exp((17.625 * (df_copy["d2m"] - 273.15)) / (243.04 + (df_copy["d2m"] - 273.15)))
            / np.exp((17.625 * (df_copy["t2m"] - 273.15)) / (243.04 + (df_copy["t2m"] - 273.15)))
        )
        df_copy["relative_humidity"] = df_copy["relative_humidity"].clip(0, 100)

    if "wind_dir_10m" in df_copy:
        df_copy["wind_dir_category"] = pd.cut(
            df_copy["wind_dir_10m"],
            bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
            labels=[0, 1, 2, 3, 4, 5, 6, 7],
            include_lowest=True,
        ).astype(int)

    return df_copy


print("\n" + "=" * 80)
print("Step 1: Data Loading and Preprocessing")
print("=" * 80)

df_era5 = read_all_era5()
df_pollution = read_all_pollution()

print("\nData loading check:")
print(f"  Pollution data shape: {df_pollution.shape}")
print(f"  Meteorological data shape: {df_era5.shape}")

if df_pollution.empty:
    print("\nWarning: Pollution data is empty! Please check data path and files.")
    raise SystemExit(1)
if df_era5.empty:
    print("\nWarning: Meteorological data is empty! Please check data path and files.")
    raise SystemExit(1)

df_pollution.index = pd.to_datetime(df_pollution.index)
df_era5.index = pd.to_datetime(df_era5.index)

print(f"  Pollution data time range: {df_pollution.index.min()} to {df_pollution.index.max()}")
print(f"  Meteorological data time range: {df_era5.index.min()} to {df_era5.index.max()}")

print("\nMerging data...")
df_combined = df_pollution.join(df_era5, how="inner")
if df_combined.empty:
    print("\nError: Data is empty after merging!")
    raise SystemExit(1)

print("\nCreating features...")
df_combined = create_features(df_combined)

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


print("\n" + "=" * 80)
print("Step 2: BPNN Data Preparation (Sliding Window)")
print("=" * 80)

target = "PM2.5"
exclude_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "year"]
numeric_features = [col for col in df_combined.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

print(f"\nNumber of selected features: {len(numeric_features)}")
print(f"Target variable: {target}")

X_raw = df_combined[numeric_features].values
y_raw = df_combined[target].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()


def create_sliding_windows(X, y, window_size):
    num_samples = len(X)
    num_windows = num_samples - window_size + 1
    num_features_local = X.shape[1]
    X_windows = np.zeros((num_windows, window_size, num_features_local))
    y_windows = np.zeros(num_windows)
    for i in range(num_windows):
        X_windows[i] = X[i : i + window_size]
        y_windows[i] = y[i + window_size - 1]
    return X_windows, y_windows


print(f"\nCreating {WINDOW_SIZE} day sliding windows...")
X_windows, y_windows = create_sliding_windows(X_scaled, y_scaled, WINDOW_SIZE)

feature_names = numeric_features
date_index = df_combined.index[WINDOW_SIZE - 1 :]

print(f"Sliding window data shape: X_windows={X_windows.shape}, y_windows={y_windows.shape}")


print("\n" + "=" * 80)
print("Step 3: Creating PyTorch Dataset")
print("=" * 80)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


n_samples = len(X_windows)
train_size = int(n_samples * 0.70)
val_size = int(n_samples * 0.15)

X_train = X_windows[:train_size]
X_val = X_windows[train_size : train_size + val_size]
X_test = X_windows[train_size + val_size :]

y_train = y_windows[:train_size]
y_val = y_windows[train_size : train_size + val_size]
y_test = y_windows[train_size + val_size :]

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
    drop_last=False,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=DATALOADER_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else None,
    drop_last=False,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=DATALOADER_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else None,
    drop_last=False,
)

print(f"\nTraining set: {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/n_samples*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")


print("\n" + "=" * 80)
print("Step 4: Defining BPNN Model")
print("=" * 80)


class PM25BPNN(nn.Module):
    """多层感知机（BPNN）用于 PM2.5 预测：输入[batch, window, features] -> 输出[batch]"""

    def __init__(self, window_size, num_features, hidden_units=(512, 256, 64), dropout_rate=0.3, activation="relu", use_batchnorm=True):
        super().__init__()
        self.window_size = window_size
        self.num_features = num_features
        self.input_dim = window_size * num_features
        self.activation = activation.lower()
        self.use_batchnorm = use_batchnorm

        layers = []
        in_dim = self.input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._build_activation_layer())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _build_activation_layer(self):
        if self.activation == "gelu":
            return nn.GELU()
        if self.activation == "selu":
            return nn.SELU()
        return nn.ReLU(inplace=True)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x).squeeze()


num_features = X_train.shape[2]


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device, non_blocking=NON_BLOCKING)
        y_batch = y_batch.to(device, non_blocking=NON_BLOCKING)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
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
    model.eval()
    total_loss = 0.0
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
    return avg_loss, np.array(predictions), np.array(actuals)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=20, verbose=True, use_amp=False):
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"\nTraining {num_epochs} epochs...")
    if use_amp:
        print("  Mixed precision training (AMP): Enabled")

    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, _, _ = validate(model, val_loader, criterion, device, use_amp)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"Training complete! Best epoch={best_epoch}, best val loss={best_val_loss:.4f}, time={total_time/60:.2f} min")
    return train_losses, val_losses, best_epoch


def evaluate_model(y_true, y_pred, dataset_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0.0
    return {"Dataset": dataset_name, "R²": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}


print("\n" + "=" * 80)
print("Step 5: Training Basic BPNN Model")
print("=" * 80)

basic_params = {
    "hidden_units": (512, 256, 64),
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "activation": "relu",
    "use_batchnorm": True,
    "num_epochs": 200,
    "patience": 20,
}

model_basic = PM25BPNN(
    window_size=WINDOW_SIZE,
    num_features=num_features,
    hidden_units=basic_params["hidden_units"],
    dropout_rate=basic_params["dropout_rate"],
    activation=basic_params["activation"],
    use_batchnorm=basic_params["use_batchnorm"],
).to(DEVICE)

total_params = sum(p.numel() for p in model_basic.parameters())
trainable_params = sum(p.numel() for p in model_basic.parameters() if p.requires_grad)
print(f"\nModel parameter statistics: total={total_params:,}, trainable={trainable_params:,}")

if DEVICE.type == "cuda":
    print("\n" + "=" * 80)
    print("Optimizing Batch Size for GPU Memory")
    print("=" * 80)
    optimal_batch_size = get_optimal_batch_size(
        PM25BPNN,
        WINDOW_SIZE,
        num_features,
        DEVICE,
        hidden_units=basic_params["hidden_units"],
        dropout_rate=basic_params["dropout_rate"],
        activation=basic_params["activation"],
        use_batchnorm=basic_params["use_batchnorm"],
    )
    if optimal_batch_size != BATCH_SIZE:
        print(f"\nUpdating batch size: {BATCH_SIZE} -> {optimal_batch_size}")
        BATCH_SIZE = optimal_batch_size
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=DATALOADER_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else None,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=DATALOADER_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else None,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=DATALOADER_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR if DATALOADER_WORKERS > 0 else None,
            drop_last=False,
        )

criterion = nn.MSELoss()
optimizer_basic = optim.Adam(model_basic.parameters(), lr=basic_params["learning_rate"])

train_losses_basic, val_losses_basic, best_epoch_basic = train_model(
    model_basic,
    train_loader,
    val_loader,
    criterion,
    optimizer_basic,
    num_epochs=basic_params["num_epochs"],
    device=DEVICE,
    patience=basic_params["patience"],
    verbose=True,
    use_amp=USE_AMP,
)

print("\nEvaluating basic model...")
_, y_train_pred_basic_scaled, y_train_actual_scaled = validate(model_basic, train_loader, criterion, DEVICE, USE_AMP)
_, y_val_pred_basic_scaled, y_val_actual_scaled = validate(model_basic, val_loader, criterion, DEVICE, USE_AMP)
_, y_test_pred_basic_scaled, y_test_actual_scaled = validate(model_basic, test_loader, criterion, DEVICE, USE_AMP)

y_train_pred_basic = scaler_y.inverse_transform(y_train_pred_basic_scaled.reshape(-1, 1)).flatten()
y_train_actual_basic = scaler_y.inverse_transform(y_train_actual_scaled.reshape(-1, 1)).flatten()
y_val_pred_basic = scaler_y.inverse_transform(y_val_pred_basic_scaled.reshape(-1, 1)).flatten()
y_val_actual_basic = scaler_y.inverse_transform(y_val_actual_scaled.reshape(-1, 1)).flatten()
y_test_pred_basic = scaler_y.inverse_transform(y_test_pred_basic_scaled.reshape(-1, 1)).flatten()
y_test_actual_basic = scaler_y.inverse_transform(y_test_actual_scaled.reshape(-1, 1)).flatten()

results_basic_df = pd.DataFrame(
    [
        evaluate_model(y_train_actual_basic, y_train_pred_basic, "Train"),
        evaluate_model(y_val_actual_basic, y_val_pred_basic, "Validation"),
        evaluate_model(y_test_actual_basic, y_test_pred_basic, "Test"),
    ]
)


print("\n" + "=" * 80)
print("Step 6: Hyperparameter Optimization (Grid Search)")
print("=" * 80)

param_grid = {
    "hidden_units": [(512, 256, 64), (256, 128, 64), (512, 256), (256, 128)],
    "dropout_rate": [0.2, 0.3, 0.4],
    "learning_rate": [0.0005, 0.001, 0.002],
    "activation": ["relu", "gelu"],
    "use_batchnorm": [True, False],
}

total_combinations = int(np.prod([len(v) for v in param_grid.values()]))
grid_search_results = []
best_val_loss_grid = float("inf")
best_params = {}

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

grid_search_start_time = time.time()
param_combinations = list(product(*param_grid.values()))
iterator = tqdm(enumerate(param_combinations, 1), total=total_combinations, desc="Grid search progress", unit="combination") if TQDM_AVAILABLE else enumerate(param_combinations, 1)

for i, combo in iterator:
    params_test = dict(zip(param_grid.keys(), combo))
    try:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        model_temp = PM25BPNN(
            window_size=WINDOW_SIZE,
            num_features=num_features,
            hidden_units=params_test["hidden_units"],
            dropout_rate=params_test["dropout_rate"],
            activation=params_test["activation"],
            use_batchnorm=params_test["use_batchnorm"],
        ).to(DEVICE)
        optimizer_temp = optim.Adam(model_temp.parameters(), lr=params_test["learning_rate"])
        train_start_time = time.time()
        _, _, best_epoch_temp = train_model(
            model_temp,
            train_loader,
            val_loader,
            criterion,
            optimizer_temp,
            num_epochs=100,
            device=DEVICE,
            patience=15,
            verbose=False,
            use_amp=USE_AMP,
        )
        train_time = time.time() - train_start_time
        val_loss, _, _ = validate(model_temp, val_loader, criterion, DEVICE, USE_AMP)

        result_entry = params_test.copy()
        result_entry["val_loss"] = float(val_loss)
        result_entry["best_epoch"] = int(best_epoch_temp)
        result_entry["train_time"] = float(train_time)
        grid_search_results.append(result_entry)

        if val_loss < best_val_loss_grid:
            best_val_loss_grid = float(val_loss)
            best_params = params_test.copy()

        del model_temp, optimizer_temp
    except Exception as e:
        result_entry = params_test.copy()
        result_entry["val_loss"] = float("inf")
        result_entry["best_epoch"] = 0
        result_entry["train_time"] = 0.0
        result_entry["error"] = str(e)
        grid_search_results.append(result_entry)

grid_search_total_time = time.time() - grid_search_start_time
grid_search_df = pd.DataFrame(grid_search_results).sort_values("val_loss", ascending=True)
save_csv(grid_search_df, output_dir / "grid_search_results__bpnn.csv")
print(f"Grid search total time: {grid_search_total_time/60:.2f} min")


print("\n" + "=" * 80)
print("Step 7: Training Optimized Model with Best Parameters")
print("=" * 80)

model_optimized = PM25BPNN(
    window_size=WINDOW_SIZE,
    num_features=num_features,
    hidden_units=best_params["hidden_units"],
    dropout_rate=best_params["dropout_rate"],
    activation=best_params["activation"],
    use_batchnorm=best_params["use_batchnorm"],
).to(DEVICE)
optimizer_opt = optim.Adam(model_optimized.parameters(), lr=best_params["learning_rate"])

train_losses_opt, val_losses_opt, best_epoch_opt = train_model(
    model_optimized,
    train_loader,
    val_loader,
    criterion,
    optimizer_opt,
    num_epochs=300,
    device=DEVICE,
    patience=30,
    verbose=True,
    use_amp=USE_AMP,
)

print("\nEvaluating optimized model...")
_, y_train_pred_opt_scaled, _ = validate(model_optimized, train_loader, criterion, DEVICE, USE_AMP)
_, y_val_pred_opt_scaled, _ = validate(model_optimized, val_loader, criterion, DEVICE, USE_AMP)
_, y_test_pred_opt_scaled, _ = validate(model_optimized, test_loader, criterion, DEVICE, USE_AMP)

y_train_pred_opt = scaler_y.inverse_transform(y_train_pred_opt_scaled.reshape(-1, 1)).flatten()
y_val_pred_opt = scaler_y.inverse_transform(y_val_pred_opt_scaled.reshape(-1, 1)).flatten()
y_test_pred_opt = scaler_y.inverse_transform(y_test_pred_opt_scaled.reshape(-1, 1)).flatten()

results_opt_df = pd.DataFrame(
    [
        evaluate_model(y_train_actual_basic, y_train_pred_opt, "Train"),
        evaluate_model(y_val_actual_basic, y_val_pred_opt, "Validation"),
        evaluate_model(y_test_actual_basic, y_test_pred_opt, "Test"),
    ]
)


print("\n" + "=" * 80)
print("Step 8: Model Performance Comparison")
print("=" * 80)

results_basic_df = results_basic_df.copy()
results_opt_df = results_opt_df.copy()
results_basic_df["Model"] = "BPNN_Basic"
results_opt_df["Model"] = "BPNN_Optimized"
all_results = pd.concat([results_basic_df, results_opt_df], ignore_index=True)[["Model", "Dataset", "R²", "RMSE", "MAE", "MAPE"]]

test_results = all_results[all_results["Dataset"] == "Test"].sort_values("R²", ascending=False)


print("\n" + "=" * 80)
print("Step 9: Feature Importance Analysis (Gradient×Input Method)")
print("=" * 80)


def compute_gradient_importance(model, X_samples, device, num_samples=500):
    model.eval()
    if len(X_samples) > num_samples:
        indices = np.random.choice(len(X_samples), num_samples, replace=False)
        X_samples = X_samples[indices]
    X_tensor = torch.FloatTensor(X_samples).to(device)
    X_tensor.requires_grad = True
    outputs = model(X_tensor)
    gradients = torch.autograd.grad(outputs=outputs.sum(), inputs=X_tensor, create_graph=False)[0]
    importance = (gradients * X_tensor).abs()
    importance = importance.mean(dim=[0, 1])
    return importance.detach().cpu().numpy()


feature_importance_scores = compute_gradient_importance(model_optimized, X_train, DEVICE, num_samples=500)
feature_importance_scores_norm = (feature_importance_scores / feature_importance_scores.sum()) * 100
feature_importance = (
    pd.DataFrame({"Feature": feature_names, "Importance": feature_importance_scores, "Importance_Norm": feature_importance_scores_norm})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)


print("\n" + "=" * 80)
print("Step 10: Export CSV for plotting (NO image generation)")
print("=" * 80)

# 1) 训练过程曲线（按模型拆分）
train_curves_basic_df = pd.DataFrame(
    {
        "epoch": np.arange(1, len(train_losses_basic) + 1),
        "train_loss": train_losses_basic,
        "val_loss": val_losses_basic,
        "best_epoch": best_epoch_basic,
    }
)
save_csv(train_curves_basic_df, output_dir / "plot_training_curves__bpnn_basic.csv")

train_curves_opt_df = pd.DataFrame(
    {
        "epoch": np.arange(1, len(train_losses_opt) + 1),
        "train_loss": train_losses_opt,
        "val_loss": val_losses_opt,
        "best_epoch": best_epoch_opt,
    }
)
save_csv(train_curves_opt_df, output_dir / "plot_training_curves__bpnn_optimized.csv")

# 2) Scatter plot 数据（按模型/按数据集拆分 + 兼容 test-only）
split_map = {
    "train": (y_train_actual_basic, y_train_pred_basic, y_train_pred_opt),
    "validation": (y_val_actual_basic, y_val_pred_basic, y_val_pred_opt),
    "test": (y_test_actual_basic, y_test_pred_basic, y_test_pred_opt),
}

for split_name, (y_true_split, y_pred_basic_split, y_pred_opt_split) in split_map.items():
    scatter_basic = pd.DataFrame({"Actual_PM25": y_true_split, "Predicted_PM25": y_pred_basic_split})
    save_csv(scatter_basic, output_dir / f"plot_scatter__bpnn_basic__{split_name}.csv")

    scatter_opt = pd.DataFrame({"Actual_PM25": y_true_split, "Predicted_PM25": y_pred_opt_split})
    save_csv(scatter_opt, output_dir / f"plot_scatter__bpnn_optimized__{split_name}.csv")

save_csv(pd.DataFrame({"Actual_PM25": y_test_actual_basic, "Predicted_PM25": y_test_pred_basic}), output_dir / "plot_scatter__bpnn_basic.csv")
save_csv(pd.DataFrame({"Actual_PM25": y_test_actual_basic, "Predicted_PM25": y_test_pred_opt}), output_dir / "plot_scatter__bpnn_optimized.csv")

# 3) Time series：simple sampled（按模型拆分，测试集）
test_date_index = date_index[train_size + val_size :]
plot_df_simple = pd.DataFrame(
    {
        "time": test_date_index,
        "y_true": y_test_actual_basic,
        "y_pred_bpnn_basic": y_test_pred_basic,
        "y_pred_bpnn_optimized": y_test_pred_opt,
    }
).sort_values("time").reset_index(drop=True)

step = 4
plot_df_simple_sampled = plot_df_simple.iloc[::step].copy()
save_csv(
    plot_df_simple_sampled[["time", "y_true", "y_pred_bpnn_basic"]].rename(columns={"y_pred_bpnn_basic": "y_pred"}),
    output_dir / "plot_ts_simple_sampled__bpnn_basic.csv",
)
save_csv(
    plot_df_simple_sampled[["time", "y_true", "y_pred_bpnn_optimized"]].rename(columns={"y_pred_bpnn_optimized": "y_pred"}),
    output_dir / "plot_ts_simple_sampled__bpnn_optimized.csv",
)

# 4) Time series：last-year sampled（按 RF 命名规则拆分 actual + 模型）
plot_range = min(365, len(plot_df_simple))
plot_df_subset = plot_df_simple.iloc[-plot_range:].copy()
plot_df_sampled = plot_df_subset.iloc[::step].copy()

x_axis = np.arange(len(plot_df_sampled))
ts_common = pd.DataFrame({"x_axis": x_axis, "time": plot_df_sampled["time"].values, "y_true": plot_df_sampled["y_true"].values})
save_csv(ts_common, output_dir / "plot_ts_lastyear_sampled__actual.csv")
save_csv(ts_common.assign(y_pred=plot_df_sampled["y_pred_bpnn_basic"].values), output_dir / "plot_ts_lastyear_sampled__bpnn_basic.csv")
save_csv(ts_common.assign(y_pred=plot_df_sampled["y_pred_bpnn_optimized"].values), output_dir / "plot_ts_lastyear_sampled__bpnn_optimized.csv")

# 5) Residuals（按模型/按数据集拆分 + 兼容 test-only）
for split_name, (y_true_split, y_pred_basic_split, y_pred_opt_split) in split_map.items():
    res_basic = pd.DataFrame({"Actual_PM25": y_true_split, "Predicted_PM25": y_pred_basic_split, "Residual": (y_true_split - y_pred_basic_split)})
    save_csv(res_basic, output_dir / f"plot_residuals__bpnn_basic__{split_name}.csv")
    res_opt = pd.DataFrame({"Actual_PM25": y_true_split, "Predicted_PM25": y_pred_opt_split, "Residual": (y_true_split - y_pred_opt_split)})
    save_csv(res_opt, output_dir / f"plot_residuals__bpnn_optimized__{split_name}.csv")

save_csv(
    pd.DataFrame({"Actual_PM25": y_test_actual_basic, "Predicted_PM25": y_test_pred_basic, "Residual": (y_test_actual_basic - y_test_pred_basic)}),
    output_dir / "plot_residuals__bpnn_basic.csv",
)
save_csv(
    pd.DataFrame({"Actual_PM25": y_test_actual_basic, "Predicted_PM25": y_test_pred_opt, "Residual": (y_test_actual_basic - y_test_pred_opt)}),
    output_dir / "plot_residuals__bpnn_optimized.csv",
)

# 6) Error distribution（按模型/按数据集拆分 + 兼容 test-only）
for split_name, (y_true_split, y_pred_basic_split, y_pred_opt_split) in split_map.items():
    save_csv(pd.DataFrame({"Error": (y_true_split - y_pred_basic_split)}), output_dir / f"plot_error_distribution__bpnn_basic__{split_name}.csv")
    save_csv(pd.DataFrame({"Error": (y_true_split - y_pred_opt_split)}), output_dir / f"plot_error_distribution__bpnn_optimized__{split_name}.csv")

save_csv(pd.DataFrame({"Error": (y_test_actual_basic - y_test_pred_basic)}), output_dir / "plot_error_distribution__bpnn_basic.csv")
save_csv(pd.DataFrame({"Error": (y_test_actual_basic - y_test_pred_opt)}), output_dir / "plot_error_distribution__bpnn_optimized.csv")

# 7) Feature importance（按模型拆分：此处基于 optimized；同时输出 TopN）
save_csv(feature_importance, output_dir / "bpnn_feature_importance.csv")
save_csv(feature_importance, output_dir / "plot_feature_importance__bpnn_optimized.csv")
top_n = min(15, len(feature_importance))
save_csv(feature_importance.head(top_n), output_dir / f"plot_feature_importance_top{top_n}__bpnn_optimized.csv")

# 8) Metrics（按模型/按数据集拆分 + test-only 排名）
save_csv(all_results, output_dir / "metrics__all_models_train_val_test.csv")
for model in all_results["Model"].unique():
    model_slug = model.lower()
    save_csv(all_results[all_results["Model"] == model], output_dir / f"metrics__{model_slug}__train_val_test.csv")
    for ds_name in all_results["Dataset"].unique():
        part = all_results[(all_results["Model"] == model) & (all_results["Dataset"] == ds_name)]
        if not part.empty:
            save_csv(part, output_dir / f"metrics__{model_slug}__{ds_name.lower()}.csv")

save_csv(test_results, output_dir / "plot_metrics_ranking__test_only.csv")

# 9) Predictions（按 RF 命名规则：all + 按模型）
pred_all = pd.DataFrame(
    {
        "Date": test_date_index,
        "Actual_PM25": y_test_actual_basic,
        "Predicted_BPNN_Basic": y_test_pred_basic,
        "Predicted_BPNN_Optimized": y_test_pred_opt,
        "Error_BPNN_Basic": y_test_actual_basic - y_test_pred_basic,
        "Error_BPNN_Optimized": y_test_actual_basic - y_test_pred_opt,
    }
)
save_csv(pred_all, output_dir / "bpnn_predictions_all_models.csv")

pred_basic = pd.DataFrame(
    {
        "Date": test_date_index,
        "Actual_PM25": y_test_actual_basic,
        "Predicted_PM25": y_test_pred_basic,
        "Error": y_test_actual_basic - y_test_pred_basic,
    }
)
save_csv(pred_basic, output_dir / "bpnn_predictions__bpnn_basic.csv")

pred_opt = pd.DataFrame(
    {
        "Date": test_date_index,
        "Actual_PM25": y_test_actual_basic,
        "Predicted_PM25": y_test_pred_opt,
        "Error": y_test_actual_basic - y_test_pred_opt,
    }
)
save_csv(pred_opt, output_dir / "bpnn_predictions__bpnn_optimized.csv")

# 10) Parameters（按模型拆分，便于后续脚本读取/汇总）
save_csv(pd.DataFrame([basic_params]), output_dir / "bpnn_parameters__bpnn_basic.csv")
save_csv(pd.DataFrame([best_params]), output_dir / "bpnn_parameters__bpnn_optimized.csv")

# 11) 兼容“整体表现”输出（但不生成任何图片/模型文件）
save_csv(all_results, output_dir / "bpnn_model_performance.csv")

print("\n" + "=" * 80)
print("Analysis Complete! (CSV only, no plots, no model files)")
print("=" * 80)
