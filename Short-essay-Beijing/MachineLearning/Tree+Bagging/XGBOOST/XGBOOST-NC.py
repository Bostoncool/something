import os
import glob
import warnings
import calendar
import pickle
import multiprocessing
import contextlib
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import xgboost as xgb

warnings.filterwarnings("ignore")

CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)
RANDOM_SEED = 42

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm not detected, progress display will use simple mode.")
    print("      Run `pip install tqdm` for a better progress bar experience.")

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100

# 使用numpy的新随机数生成器API（避免弃用警告）
# 注意：XGBoost使用random_state参数控制随机性，这里设置全局种子主要用于其他随机操作
np.random.default_rng(RANDOM_SEED)


class XGBoostProgressBar(xgb.callback.TrainingCallback):
    def __init__(self, total_rounds, description="Training"):
        self.total_rounds = total_rounds
        self.description = description
        self.pbar = None

    def before_training(self, model):
        if TQDM_AVAILABLE:
            self.pbar = tqdm(total=self.total_rounds, desc=self.description, unit="iter")
        return model

    def after_iteration(self, model, epoch, evals_log):
        if self.pbar:
            self.pbar.update(1)
            if evals_log:
                postfix = {}
                for dataset, metrics in evals_log.items():
                    for metric, values in metrics.items():
                        if values:
                            # Simplify dataset names for display
                            if dataset == "validation_0":
                                ds_name = "Train"
                            elif dataset == "validation_1":
                                ds_name = "Val"
                            else:
                                ds_name = dataset
                            postfix[f"{ds_name}-{metric}"] = f"{values[-1]:.4f}"
                self.pbar.set_postfix(postfix)
        return False

    def after_training(self, model):
        if self.pbar:
            self.pbar.close()
        return model


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar class instance"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def daterange(start: datetime, end: datetime):
    for n_days in range(int((end - start).days) + 1):
        yield start + timedelta(n_days)


def build_file_path_dict(base_path: str, prefix: str) -> dict[str, str]:
    """预先扫描目录，构建日期到文件路径的字典，时间复杂度O(1)查找"""
    file_dict: dict[str, str] = {}
    print(f"  正在扫描 {base_path} 目录...")
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename.startswith(f"{prefix}_") and filename.endswith(".csv"):
                # 提取日期部分：beijing_all_YYYYMMDD.csv -> YYYYMMDD
                date_str = filename[len(prefix) + 1:-4]  # 去掉前缀和下划线，去掉.csv
                if len(date_str) == 8 and date_str.isdigit():  # 确保是8位日期格式
                    file_path = os.path.join(root, filename)
                    if date_str not in file_dict:
                        file_dict[date_str] = file_path
                    else:
                        # 如果存在重复，保留第一个找到的
                        pass
    print(f"  找到 {len(file_dict)} 个文件")
    return file_dict


def read_pollution_day(
    date: datetime,
    file_dict_all: dict[str, str],
    file_dict_extra: dict[str, str],
    pollutants: list[str],
) -> pd.DataFrame | None:
    date_str = date.strftime("%Y%m%d")
    all_file = file_dict_all.get(date_str)
    extra_file = file_dict_extra.get(date_str)

    if not all_file or not extra_file:
        return None

    try:
        df_all = pd.read_csv(all_file, encoding="utf-8", on_bad_lines="skip")
        df_extra = pd.read_csv(extra_file, encoding="utf-8", on_bad_lines="skip")

        df_all = df_all[~df_all["type"].str.contains("_24h|AQI", na=False)]
        df_extra = df_extra[~df_extra["type"].str.contains("_24h", na=False)]

        df_poll = pd.concat([df_all, df_extra], ignore_index=True)

        df_poll = df_poll.melt(
            id_vars=["date", "hour", "type"],
            var_name="station",
            value_name="value",
        )
        df_poll["value"] = pd.to_numeric(df_poll["value"], errors="coerce")
        df_poll = df_poll[df_poll["value"] >= 0]

        df_daily = (
            df_poll.groupby(["date", "type"])["value"].mean().reset_index().pivot(
                index="date",
                columns="type",
                values="value",
            )
        )
        df_daily.index = pd.to_datetime(df_daily.index, format="%Y%m%d", errors="coerce")
        df_daily = df_daily[[col for col in pollutants if col in df_daily.columns]]
        return df_daily
    except Exception:
        return None


def read_all_pollution(
    start_date: datetime,
    end_date: datetime,
    pollution_all_path: str,
    pollution_extra_path: str,
    pollutants: list[str],
) -> pd.DataFrame:
    print("\nLoading pollution data...")
    print(f"Using {MAX_WORKERS} parallel processes")
    
    # 预先构建文件路径字典，避免每次遍历查找
    print("\n构建文件路径字典...")
    file_dict_all = build_file_path_dict(pollution_all_path, "beijing_all")
    file_dict_extra = build_file_path_dict(pollution_extra_path, "beijing_extra")
    
    dates = list(daterange(start_date, end_date))
    pollution_dfs: list[pd.DataFrame] = []

    # 使用多进程而不是多线程，避免netcdf4库的线程安全问题
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                read_pollution_day,
                date,
                file_dict_all,
                file_dict_extra,
                pollutants,
            ): date
            for date in dates
        }

        if TQDM_AVAILABLE:
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading pollution data",
                unit="day",
            ):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
        else:
            for idx, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
                if idx % 500 == 0 or idx == len(futures):
                    print(
                        f"  Processed {idx}/{len(futures)} days of data "
                        f"({idx / len(futures) * 100:.1f}%)"
                    )

    if pollution_dfs:
        print(f"  Successfully read {len(pollution_dfs)}/{len(dates)} days of data")
        df_poll_all = pd.concat(pollution_dfs)
        df_poll_all = df_poll_all.ffill()
        df_poll_all = df_poll_all.fillna(df_poll_all.mean())
        print(f"Pollution data loading completed, shape: {df_poll_all.shape}")
        return df_poll_all

    print("⚠️ Failed to load any pollution data!")
    return pd.DataFrame()


def read_era5_month(
    year: int,
    month: int,
    era5_path: str,
    era5_vars: list[str],
    beijing_lats: list[float] | np.ndarray,
    beijing_lons: list[float] | np.ndarray,
) -> pd.DataFrame | None:
    # 确保是numpy数组（多进程传递时可能是列表）
    if isinstance(beijing_lats, list):
        beijing_lats = np.array(beijing_lats)
    if isinstance(beijing_lons, list):
        beijing_lons = np.array(beijing_lons)
    month_str = f"{year}{month:02d}"
    all_files = glob.glob(os.path.join(era5_path, "**", f"*{month_str}*.nc"), recursive=True)
    fallback_used = False

    if not all_files:
        all_files = glob.glob(os.path.join(era5_path, "**", "*.nc"), recursive=True)
        fallback_used = True
        if not all_files:
            return None

    monthly_datasets: list[xr.Dataset] = []

    start_day = 1
    end_day = calendar.monthrange(year, month)[1]
    month_start = pd.to_datetime(f"{year}-{month:02d}-{start_day:02d}")
    month_end = (
        pd.to_datetime(f"{year}-{month:02d}-{end_day:02d}")
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )

    for file_path in all_files:
        try:
            with Dataset(file_path, mode="r") as nc_file:
                available_vars = [v for v in era5_vars if v in nc_file.variables]
            if not available_vars:
                print(
                    f"[WARN] {os.path.basename(file_path)} missing variables from target list, skipping"
                )
                continue

            with xr.open_dataset(
                file_path, engine="netcdf4", decode_times=True
            ) as ds:
                rename_map: dict[str, str] = {}

                for tkey in (
                    "valid_time",
                    "forecast_time",
                    "verification_time",
                    "time1",
                    "time2",
                ):
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
                    print(f"[WARN] {os.path.basename(file_path)} missing time coordinate, skipping")
                    continue
                ds_subset = ds_subset.sortby("time")

                if fallback_used:
                    try:
                        ds_subset = ds_subset.sel(time=slice(month_start, month_end))
                    except Exception as err:
                        print(
                            f"[WARN] {os.path.basename(file_path)} time filtering failed: {err}"
                        )
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
                        longitude=slice(beijing_lons.min(), beijing_lons.max()),
                    )

                    if "latitude" in ds_subset.dims and "longitude" in ds_subset.dims:
                        ds_subset = ds_subset.mean(dim=["latitude", "longitude"], skipna=True)

                ds_daily = ds_subset.resample(time="1D").mean(keep_attrs=False)
                ds_daily = ds_daily.dropna("time", how="all")
                if ds_daily.sizes.get("time", 0) == 0:
                    continue

                monthly_datasets.append(ds_daily.load())
                print(
                    f"  [+] {os.path.basename(file_path)} -> {year}-{month:02d} "
                    f"Days: {ds_daily.sizes.get('time', 0)}, Variables: {len(ds_daily.data_vars)}"
                )
        except Exception as err:
            print(f"[ERROR] Failed to read {os.path.basename(file_path)}: {type(err).__name__}: {err}")
            continue

    if not monthly_datasets:
        return None

    merged_ds = xr.merge(monthly_datasets, compat="override", join="outer")
    df_month = merged_ds.to_dataframe()
    df_month.index = pd.to_datetime(df_month.index)
    df_month = df_month.groupby(df_month.index).mean()

    if df_month.empty:
        return None

    print(
        f"  Successfully read ERA5 data: {year}-{month:02d}, Days: {len(df_month)}, Variables: {len(df_month.columns)}"
    )
    return df_month


def read_all_era5(
    start_year: int,
    end_year: int,
    era5_path: str,
    era5_vars: list[str],
    beijing_lats: np.ndarray,
    beijing_lons: np.ndarray,
) -> pd.DataFrame:
    print("\nLoading ERA5 meteorological data (NetCDF)...")
    print(f"Using {MAX_WORKERS} parallel processes")
    print(f"Meteorological data directory: {era5_path}")
    print(f"Directory exists: {os.path.exists(era5_path)}")

    if os.path.exists(era5_path):
        all_nc = glob.glob(os.path.join(era5_path, "**", "*.nc"), recursive=True)
        print(f"Found {len(all_nc)} NetCDF files")
        if all_nc:
            print(f"Sample files: {[os.path.basename(f) for f in all_nc[:5]]}")

    era5_dfs: list[pd.DataFrame] = []
    years = range(start_year, end_year + 1)
    months = range(1, 13)

    month_tasks = [
        (year, month)
        for year in years
        for month in months
        if not (year == end_year and month > 12)
    ]
    print(f"Planning to load {len(month_tasks)} months of data...")

    # 使用多进程而不是多线程，避免netcdf4库的线程安全问题
    # 注意：需要将numpy数组转换为列表以便序列化传递给子进程
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                read_era5_month,
                year,
                month,
                era5_path,
                era5_vars,
                beijing_lats.tolist(),  # 转换为列表以便序列化
                beijing_lons.tolist(),  # 转换为列表以便序列化
            ): (year, month)
            for year, month in month_tasks
        }

        success_count = 0
        if TQDM_AVAILABLE:
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading meteorological data",
                unit="month",
            ):
                result = future.result()
                if result is not None and not result.empty:
                    era5_dfs.append(result)
                    success_count += 1
        else:
            for idx, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None and not result.empty:
                    era5_dfs.append(result)
                    success_count += 1
                if idx % 20 == 0 or idx == len(futures):
                    print(
                        f"  Progress: {idx}/{len(futures)} months (successful {success_count}, "
                        f"{idx / len(futures) * 100:.1f}%)"
                    )

        print(f"  Successfully loaded {success_count}/{len(futures)} months of data")

    if not era5_dfs:
        print("\n❌ Error: Failed to load any meteorological data files!")
        print("Troubleshooting suggestions:")
        print("1. Verify file naming contains YYYYMM or check if .nc files exist in the directory")
        print("2. Check if files contain time, latitude/longitude coordinates and target variables")
        print("3. Verify the ERA5 data path is correct")
        return pd.DataFrame()

    print("\nMerging meteorological data...")
    df_era5_all = pd.concat(era5_dfs, axis=0)
    print("  Removing duplicates...")
    df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep="first")]
    print("  Sorting...")
    df_era5_all = df_era5_all.sort_index()

    print(f"Merged shape: {df_era5_all.shape}")
    print(f"Time range: {df_era5_all.index.min()} to {df_era5_all.index.max()}")
    preview_cols = list(df_era5_all.columns[:10])
    print(f"Sample variables: {preview_cols}{'...' if len(df_era5_all.columns) > 10 else ''}")

    print("  Processing missing values...")
    initial_na = df_era5_all.isna().sum().sum()
    df_era5_all = df_era5_all.ffill()
    df_era5_all = df_era5_all.bfill()
    df_era5_all = df_era5_all.fillna(df_era5_all.mean())
    final_na = df_era5_all.isna().sum().sum()
    print(f"Missing values: {initial_na} -> {final_na}")

    print("Meteorological data loading completed.")
    return df_era5_all


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    if {"u10", "v10"}.issubset(df_copy.columns):
        df_copy["wind_speed_10m"] = np.sqrt(df_copy["u10"] ** 2 + df_copy["v10"] ** 2)
        df_copy["wind_dir_10m"] = np.degrees(np.arctan2(df_copy["v10"], df_copy["u10"]))
        df_copy["wind_dir_10m"] = (df_copy["wind_dir_10m"] + 360) % 360

    if {"u100", "v100"}.issubset(df_copy.columns):
        df_copy["wind_speed_100m"] = np.sqrt(df_copy["u100"] ** 2 + df_copy["v100"] ** 2)
        df_copy["wind_dir_100m"] = np.degrees(np.arctan2(df_copy["v100"], df_copy["u100"]))
        df_copy["wind_dir_100m"] = (df_copy["wind_dir_100m"] + 360) % 360

    df_copy["year"] = df_copy.index.year
    df_copy["month"] = df_copy.index.month
    df_copy["day"] = df_copy.index.day
    df_copy["day_of_year"] = df_copy.index.day_of_year  # 使用新的API替代已弃用的dayofyear
    df_copy["day_of_week"] = df_copy.index.day_of_week  # 使用新的API替代已弃用的dayofweek
    df_copy["week_of_year"] = df_copy.index.isocalendar().week.astype(int)

    df_copy["season"] = df_copy["month"].apply(
        lambda x: 1
        if x in (12, 1, 2)
        else 2
        if x in (3, 4, 5)
        else 3
        if x in (6, 7, 8)
        else 4
    )
    df_copy["is_heating_season"] = (
        ((df_copy["month"] >= 11) | (df_copy["month"] <= 3)).astype(int)
    )

    if {"t2m", "d2m"}.issubset(df_copy.columns):
        df_copy["temp_dewpoint_diff"] = df_copy["t2m"] - df_copy["d2m"]
        numerator = np.exp(
            (17.625 * (df_copy["d2m"] - 273.15))
            / (243.04 + (df_copy["d2m"] - 273.15))
        )
        denominator = np.exp(
            (17.625 * (df_copy["t2m"] - 273.15))
            / (243.04 + (df_copy["t2m"] - 273.15))
        )
        df_copy["relative_humidity"] = (100 * numerator / denominator).clip(0, 100)

    if "PM2.5" in df_copy.columns:
        df_copy["PM2.5_lag1"] = df_copy["PM2.5"].shift(1)
        df_copy["PM2.5_lag3"] = df_copy["PM2.5"].shift(3)
        df_copy["PM2.5_lag7"] = df_copy["PM2.5"].shift(7)
        df_copy["PM2.5_ma3"] = df_copy["PM2.5"].rolling(window=3, min_periods=1).mean()
        df_copy["PM2.5_ma7"] = df_copy["PM2.5"].rolling(window=7, min_periods=1).mean()
        df_copy["PM2.5_ma30"] = df_copy["PM2.5"].rolling(window=30, min_periods=1).mean()

    if "wind_dir_10m" in df_copy.columns:
        df_copy["wind_dir_category"] = pd.cut(
            df_copy["wind_dir_10m"],
            bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
            labels=list(range(8)),
            include_lowest=True,
        ).astype("Int64")

    return df_copy


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, dataset_name: str) -> dict:
    mask = y_true != 0
    safe_true = y_true[mask]
    safe_pred = y_pred[mask.values]  # Use boolean array to directly index numpy array

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = (
        np.mean(np.abs((safe_true - safe_pred) / safe_true)) * 100
        if len(safe_true) > 0
        else np.nan
    )
    return {
        "Dataset": dataset_name,
        "R²": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
    }


def plot_training_curves(
    evals_result: dict,
    model_basic: xgb.XGBRegressor,
    evals_result_opt: dict,
    model_opt: xgb.XGBRegressor,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    def _plot_curve(ax, evals, model, title: str):
        train_key = "validation_0"
        valid_key = "validation_1"
        if train_key in evals:
            ax.plot(evals[train_key]["rmse"], label="Training Set", linewidth=2)
            ax.plot(evals[valid_key]["rmse"], label="Validation Set", linewidth=2)
        elif "train" in evals:
            ax.plot(evals["train"]["rmse"], label="Training Set", linewidth=2)
            ax.plot(evals["valid"]["rmse"], label="Validation Set", linewidth=2)
        if hasattr(model, "best_iteration") and model.best_iteration is not None:
            ax.axvline(
                x=model.best_iteration,
                color="r",
                linestyle="--",
                linewidth=1.5,
                label=f"Best Iteration ({model.best_iteration})",
            )
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    _plot_curve(axes[0], evals_result, model_basic, "XGBoost Basic Model - Training Process")
    _plot_curve(axes[1], evals_result_opt, model_opt, "XGBoost Optimized Model - Training Process")

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_prediction_scatter(
    models_data: list[tuple[str, np.ndarray, pd.Series, str]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for idx, (model_name, y_pred, y_true, dataset) in enumerate(models_data):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        ax.scatter(
            y_true,
            y_pred,
            alpha=0.5,
            s=20,
            edgecolors="black",
            linewidth=0.3,
        )
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Prediction")

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ax.set_xlabel("Actual PM2.5 Concentration (μg/m³)", fontsize=11)
        ax.set_ylabel("Predicted PM2.5 Concentration (μg/m³)", fontsize=11)
        ax.set_title(
            f"XGBoost_{model_name} - {dataset}\nR²={r2:.4f}, RMSE={rmse:.2f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "prediction_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_time_series(
    y_test: pd.Series,
    y_test_pred_basic: np.ndarray,
    y_test_pred_opt: np.ndarray,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    plot_range = min(300, len(y_test))
    plot_idx = range(len(y_test) - plot_range, len(y_test))
    time_idx = y_test.index[plot_idx]

    axes[0].plot(time_idx, y_test.iloc[plot_idx], "k-", label="Actual", linewidth=2, alpha=0.8)
    axes[0].plot(
        time_idx,
        y_test_pred_basic[plot_idx],
        "b--",
        label="Basic Model Prediction",
        linewidth=1.5,
        alpha=0.7,
    )
    axes[0].set_xlabel("Date", fontsize=12)
    axes[0].set_ylabel("PM2.5 Concentration (μg/m³)", fontsize=12)
    axes[0].set_title(
        "XGBoost Basic Model - Last 300 Days Prediction Comparison",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

    axes[1].plot(time_idx, y_test.iloc[plot_idx], "k-", label="Actual", linewidth=2, alpha=0.8)
    axes[1].plot(
        time_idx,
        y_test_pred_opt[plot_idx],
        "g--",
        label="Optimized Model Prediction",
        linewidth=1.5,
        alpha=0.7,
    )
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].set_ylabel("PM2.5 Concentration (μg/m³)", fontsize=12)
    axes[1].set_title(
        "XGBoost Optimized Model - Last 300 Days Prediction Comparison",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "timeseries_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_residuals(
    models_data: list[tuple[str, np.ndarray, pd.Series, str]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for idx, (model_name, y_pred, y_true, dataset) in enumerate(models_data):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        residuals = y_true - y_pred
        ax.scatter(
            y_pred,
            residuals,
            alpha=0.5,
            s=20,
            edgecolors="black",
            linewidth=0.3,
        )
        ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Predicted Value (μg/m³)", fontsize=11)
        ax.set_ylabel("Residual (μg/m³)", fontsize=11)
        ax.set_title(
            f"XGBoost_{model_name} - {dataset}\nMean Residual={residuals.mean():.2f}, "
            f"Std Dev={residuals.std():.2f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(feature_importance: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    top_n = 20
    top_features_gain = feature_importance.head(top_n)
    top_features_weight = feature_importance.sort_values(
        "Importance_Weight", ascending=False
    ).head(top_n)

    axes[0].barh(range(top_n), top_features_gain["Importance_Gain_Norm"], color="steelblue")
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top_features_gain["Feature"], fontsize=10)
    axes[0].set_xlabel("Importance (%)", fontsize=12)
    axes[0].set_title(f"Top {top_n} Important Features (by Gain)", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="x")
    axes[0].invert_yaxis()

    axes[1].barh(
        range(top_n),
        top_features_weight["Importance_Weight_Norm"],
        color="coral",
    )
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(top_features_weight["Feature"], fontsize=10)
    axes[1].set_xlabel("Importance (%)", fontsize=12)
    axes[1].set_title(f"Top {top_n} Important Features (by Weight)", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="x")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_comparison(test_results: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    models = test_results["Model"].tolist()
    x_pos = np.arange(len(models))
    colors = ["blue", "green"]
    metrics = ["R²", "RMSE", "MAE", "MAPE"]

    for idx, metric in enumerate(metrics):
        axes[idx].bar(
            x_pos,
            test_results[metric],
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(["Basic", "Optimized"], fontsize=11)
        axes[idx].set_ylabel(metric, fontsize=12)
        if metric == "R²":
            axes[idx].set_title(f"{metric} Comparison\n(Higher is Better)", fontsize=12, fontweight="bold")
        else:
            axes[idx].set_title(f"{metric} Comparison\n(Lower is Better)", fontsize=12, fontweight="bold")
        axes[idx].grid(True, alpha=0.3, axis="y")
        for j, value in enumerate(test_results[metric]):
            if metric == "MAPE":
                axes[idx].text(
                    j,
                    value,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
            else:
                axes[idx].text(
                    j,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_distribution(
    errors_basic: np.ndarray, errors_opt: np.ndarray, output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].hist(errors_basic, bins=50, color="blue", alpha=0.7, edgecolor="black")
    axes[0].axvline(x=0, color="r", linestyle="--", linewidth=2.5, label="Zero Error")
    axes[0].set_xlabel("Prediction Error (μg/m³)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title(
        f"Basic Model - Error Distribution\nMean={errors_basic.mean():.2f}, Std Dev={errors_basic.std():.2f}",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].hist(errors_opt, bins=50, color="green", alpha=0.7, edgecolor="black")
    axes[1].axvline(x=0, color="r", linestyle="--", linewidth=2.5, label="Zero Error")
    axes[1].set_xlabel("Prediction Error (μg/m³)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title(
        f"Optimized Model - Error Distribution\nMean={errors_opt.mean():.2f}, Std Dev={errors_opt.std():.2f}",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_results(
    output_dir: Path,
    model_dir: Path,
    all_results: pd.DataFrame,
    feature_importance: pd.DataFrame,
    params_optimized: dict,
    y_test: pd.Series,
    y_test_pred_basic: np.ndarray,
    y_test_pred_opt: np.ndarray,
    model_optimized: xgb.XGBRegressor,
) -> None:
    all_results.to_csv(output_dir / "model_performance.csv", index=False, encoding="utf-8-sig")
    feature_importance.to_csv(
        output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([params_optimized]).to_csv(
        output_dir / "best_parameters.csv", index=False, encoding="utf-8-sig"
    )

    predictions_df = pd.DataFrame(
        {
            "Date": y_test.index,
            "Actual": y_test.values,
            "Prediction_Basic": y_test_pred_basic,
            "Prediction_Optimized": y_test_pred_opt,
            "Error_Basic": y_test.values - y_test_pred_basic,
            "Error_Optimized": y_test.values - y_test_pred_opt,
        }
    )
    predictions_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    model_optimized.save_model(str(model_dir / "xgboost_optimized.txt"))
    
    # 清除回调函数以避免pickle错误
    # 回调函数中的tqdm进度条包含文件句柄（TextIOWrapper），无法被pickle
    def clear_callbacks_from_model(model):
        """递归清除模型中的回调函数和进度条对象"""
        if hasattr(model, 'callbacks'):
            model.callbacks = None
        
        # 清除模型属性中可能包含的回调对象
        if hasattr(model, '__dict__'):
            for key, value in list(model.__dict__.items()):
                if value is None:
                    continue
                # 如果是列表或元组，检查其中的元素
                if isinstance(value, (list, tuple)):
                    for i, item in enumerate(value):
                        if hasattr(item, '__class__'):
                            class_name = item.__class__.__name__
                            if 'XGBoostProgressBar' in class_name or 'ProgressBar' in class_name:
                                # 清除进度条对象中的文件句柄
                                if hasattr(item, 'pbar') and item.pbar is not None:
                                    item.pbar = None
                # 如果是回调对象本身
                elif hasattr(value, '__class__'):
                    class_name = value.__class__.__name__
                    if 'XGBoostProgressBar' in class_name or 'ProgressBar' in class_name:
                        if hasattr(value, 'pbar') and value.pbar is not None:
                            value.pbar = None
    
    # 清除回调函数
    clear_callbacks_from_model(model_optimized)
    
    # 保存模型
    with open(model_dir / "xgboost_optimized.pkl", "wb") as file:
        pickle.dump(model_optimized, file)


def main():
    print("=" * 80)
    print("Beijing PM2.5 Concentration Prediction - XGBoost Model (NetCDF Data Version)")
    print("=" * 80)

    print("\nConfiguring parameters...")
    print("Using GPU acceleration (device='cuda', tree_method='hist')")

    pollution_all_path = '/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)'
    pollution_extra_path = '/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)'
    era5_path = '/root/autodl-tmp/ERA5-Beijing-NC'

    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    model_dir = Path("./models")
    model_dir.mkdir(exist_ok=True)

    start_date = datetime(2015, 1, 1)
    end_date = datetime(2024, 12, 31)

    beijing_lats = np.arange(39.0, 41.25, 0.25)
    beijing_lons = np.arange(115.0, 117.25, 0.25)

    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    era5_vars = [
        "d2m",
        "t2m",
        "u10",
        "v10",
        "u100",
        "v100",
        "blh",
        "sp",
        "tcwv",
        "tp",
        "avg_tprate",
        "tisr",
        "str",
        "cvh",
        "cvl",
        "mn2t",
        "sd",
        "lsm",
    ]

    print(f"Data time range: {start_date.date()} to {end_date.date()}")
    print(f"Target variable: PM2.5 Concentration")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Model save directory: {model_dir.resolve()}")
    print(f"CPU cores: {CPU_COUNT}, Parallel threads: {MAX_WORKERS}")

    print("\n" + "=" * 80)
    print("Step 1: Data Loading and Preprocessing")
    print("=" * 80)

    df_pollution = read_all_pollution(
        start_date,
        end_date,
        pollution_all_path,
        pollution_extra_path,
        pollutants,
    )
    df_era5 = read_all_era5(
        start_date.year,
        end_date.year,
        era5_path,
        era5_vars,
        beijing_lats,
        beijing_lons,
    )

    print("\nData loading check:")
    print(f"  Pollution data shape: {df_pollution.shape}")
    print(f"  Meteorological data shape: {df_era5.shape}")

    if df_pollution.empty:
        print("\n⚠️ Warning: Pollution data is empty, please check data paths and files.")
        raise SystemExit(1)
    if df_era5.empty:
        print("\n⚠️ Warning: Meteorological data is empty, please check data paths and files.")
        raise SystemExit(1)

    df_pollution.index = pd.to_datetime(df_pollution.index)
    df_era5.index = pd.to_datetime(df_era5.index)

    print(
        f"  Pollution data time range: {df_pollution.index.min()} to {df_pollution.index.max()}"
    )
    print(f"  Meteorological data time range: {df_era5.index.min()} to {df_era5.index.max()}")

    print("\nMerging data...")
    df_combined = df_pollution.join(df_era5, how="inner")

    if df_combined.empty:
        print("\n❌ Error: Merged data is empty, dates may not overlap between the two datasets.")
        print(f"  Pollution data rows: {len(df_pollution)}")
        print(f"  Meteorological data rows: {len(df_era5)}")
        print(f"  Merged rows: {len(df_combined)}")
        raise SystemExit(1)

    print("\nCreating features...")
    df_combined = create_features(df_combined)

    print("\nData cleaning...")
    df_combined = df_combined.replace([np.inf, -np.inf], np.nan)
    initial_rows = len(df_combined)
    df_combined = df_combined.dropna()
    final_rows = len(df_combined)
    print(f"Removed missing rows: {initial_rows - final_rows}")

    print(f"\nMerged data shape: {df_combined.shape}")
    print(
        f"Time range: {df_combined.index.min().date()} to "
        f"{df_combined.index.max().date()}"
    )
    print(f"Sample size: {len(df_combined)}")
    print(f"Number of features: {df_combined.shape[1]}")

    print("\nSample features (first 20):")
    for idx, col in enumerate(df_combined.columns[:20], 1):
        print(f"  {idx}. {col}")
    if len(df_combined.columns) > 20:
        print(f"  ... and {len(df_combined.columns) - 20} more features")

    print("\n" + "=" * 80)
    print("Step 2: Feature Selection and Data Preparation")
    print("=" * 80)

    target = "PM2.5"
    exclude_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "year"]
    numeric_features = [
        col
        for col in df_combined.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]

    print(f"\nNumber of selected features: {len(numeric_features)}")
    print(f"Target variable: {target}")

    X = df_combined[numeric_features].copy()
    y = df_combined[target].copy()

    if X.empty or y.empty:
        print("\n❌ Error: Valid modeling data is empty, please check data sources and preprocessing pipeline.")
        raise SystemExit(1)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print("\nPM2.5 Statistics:")
    print(f"  Mean: {y.mean():.2f} μg/m³")
    print(f"  Std Dev: {y.std():.2f} μg/m³")
    print(f"  Min: {y.min():.2f} μg/m³")
    print(f"  Max: {y.max():.2f} μg/m³")
    print(f"  Median: {y.median():.2f} μg/m³")

    print("\n" + "=" * 80)
    print("Step 3: Dataset Splitting")
    print("=" * 80)

    n_samples = len(X)
    train_size = int(n_samples * 0.70)
    val_size = int(n_samples * 0.15)

    X_train = X.iloc[:train_size]
    X_val = X.iloc[train_size : train_size + val_size]
    X_test = X.iloc[train_size + val_size :]

    y_train = y.iloc[:train_size]
    y_val = y.iloc[train_size : train_size + val_size]
    y_test = y.iloc[train_size + val_size :]

    print(f"\nTraining set: {len(X_train)} samples ({len(X_train) / n_samples * 100:.1f}%)")
    print(
        f"  Time range: {X_train.index.min().date()} to {X_train.index.max().date()}"
    )
    print(f"  PM2.5: {y_train.mean():.2f} ± {y_train.std():.2f} μg/m³")

    print(f"\nValidation set: {len(X_val)} samples ({len(X_val) / n_samples * 100:.1f}%)")
    print(f"  Time range: {X_val.index.min().date()} to {X_val.index.max().date()}")
    print(f"  PM2.5: {y_val.mean():.2f} ± {y_val.std():.2f} μg/m³")

    print(f"\nTest set: {len(X_test)} samples ({len(X_test) / n_samples * 100:.1f}%)")
    print(f"  Time range: {X_test.index.min().date()} to {X_test.index.max().date()}")
    print(f"  PM2.5: {y_test.mean():.2f} ± {y_test.std():.2f} μg/m³")

    print("\n" + "=" * 80)
    print("Step 4: XGBoost Basic Model Training")
    print("=" * 80)

    params_basic = {
        "objective": "reg:squarederror",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "n_jobs": MAX_WORKERS,
        "eval_metric": "rmse",
        "device": "cuda",
        "tree_method": "hist",
    }

    print("\nBasic model parameters:")
    for key, value in params_basic.items():
        print(f"  {key}: {value}")

    print("\nStarting basic model training...")
    model_basic = xgb.XGBRegressor(
        **params_basic,
        early_stopping_rounds=50,
        callbacks=[
            XGBoostProgressBar(
                params_basic["n_estimators"], description="Basic Model Training"
            )
        ],
    )
    evals_result_basic: dict = {}
    model_basic.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    # Get training history
    evals_result_basic = model_basic.evals_result()

    print("\n✓ Basic model training completed")
    if hasattr(model_basic, "best_iteration") and model_basic.best_iteration is not None:
        print(f"  Best iteration: {model_basic.best_iteration}")

    y_train_pred_basic = model_basic.predict(X_train)
    y_val_pred_basic = model_basic.predict(X_val)
    y_test_pred_basic = model_basic.predict(X_test)

    results_basic = [
        evaluate_model(y_train, y_train_pred_basic, "Train"),
        evaluate_model(y_val, y_val_pred_basic, "Validation"),
        evaluate_model(y_test, y_test_pred_basic, "Test"),
    ]
    results_basic_df = pd.DataFrame(results_basic)
    print("\nBasic model performance:")
    print(results_basic_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Step 5: Hyperparameter Optimization (Optional)")
    print("=" * 80)

    optimize_input = input("\nExecute grid search optimization? (y/n, default is n): ").strip().lower()
    optimize = optimize_input == "y"

    if optimize:
        print("\nExecuting grid search, please wait...\n")
        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200, 300],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }

        base_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=MAX_WORKERS,
            device="cuda",
            tree_method="hist",
        )
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            verbose=0 if TQDM_AVAILABLE else 2,
            n_jobs=min(4, MAX_WORKERS),
        )
        
        if TQDM_AVAILABLE:
            # Calculate total fits for progress bar
            n_candidates = 1
            for v in param_grid.values():
                n_candidates *= len(v)
            n_fits = n_candidates * 3  # cv=3
            print(f"  Total fits to perform: {n_fits}")

            with tqdm_joblib(tqdm(desc="Grid Search", total=n_fits, unit="fit")):
                grid_search.fit(X_train, y_train)
        else:
            grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print("\nBest parameter combination:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

        params_optimized = {
            **best_params,
            "objective": "reg:squarederror",
            "random_state": RANDOM_SEED,
            "n_jobs": MAX_WORKERS,
            "eval_metric": "rmse",
            "device": "cuda",
            "tree_method": "hist",
        }
        model_optimized = xgb.XGBRegressor(
            **params_optimized,
            early_stopping_rounds=50,
            callbacks=[
                XGBoostProgressBar(
                    params_optimized["n_estimators"],
                    description="Optimized Model Training",
                )
            ],
        )
        evals_result_opt: dict = {}
        model_optimized.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        # Get training history
        evals_result_opt = model_optimized.evals_result()
        print("\n✓ Retrained model with optimal parameters completed")
    else:
        print("\nSkipping hyperparameter optimization, using basic model parameters.")
        params_optimized = params_basic
        model_optimized = model_basic
        evals_result_opt = evals_result_basic

    print("\n" + "=" * 80)
    print("Step 6: Optimized Model Evaluation")
    print("=" * 80)

    y_train_pred_opt = model_optimized.predict(X_train)
    y_val_pred_opt = model_optimized.predict(X_val)
    y_test_pred_opt = model_optimized.predict(X_test)

    results_opt = [
        evaluate_model(y_train, y_train_pred_opt, "Train"),
        evaluate_model(y_val, y_val_pred_opt, "Validation"),
        evaluate_model(y_test, y_test_pred_opt, "Test"),
    ]
    results_opt_df = pd.DataFrame(results_opt)
    print("\nOptimized model performance:")
    print(results_opt_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Step 7: Model Performance Comparison")
    print("=" * 80)

    results_basic_df["Model"] = "XGBoost_Basic"
    results_opt_df["Model"] = "XGBoost_Optimized"
    all_results = pd.concat([results_basic_df, results_opt_df])
    all_results = all_results[["Model", "Dataset", "R²", "RMSE", "MAE", "MAPE"]]

    print("\nModel performance comparison:")
    print(all_results.to_string(index=False))

    test_results = all_results[all_results["Dataset"] == "Test"].sort_values(
        "R²", ascending=False
    )
    print("\nTest set performance ranking:")
    print(test_results.to_string(index=False))

    basic_test_r2 = results_basic_df.loc[
        results_basic_df["Dataset"] == "Test", "R²"
    ].values[0]
    opt_test_r2 = results_opt_df.loc[
        results_opt_df["Dataset"] == "Test", "R²"
    ].values[0]
    basic_test_rmse = results_basic_df.loc[
        results_basic_df["Dataset"] == "Test", "RMSE"
    ].values[0]
    opt_test_rmse = results_opt_df.loc[
        results_opt_df["Dataset"] == "Test", "RMSE"
    ].values[0]

    r2_improvement = (
        (opt_test_r2 - basic_test_r2) / abs(basic_test_r2) * 100
        if basic_test_r2 != 0
        else np.nan
    )
    rmse_improvement = (
        (basic_test_rmse - opt_test_rmse) / basic_test_rmse * 100
        if basic_test_rmse != 0
        else np.nan
    )

    print("\nOptimization effect:")
    print(f"  R² improvement: {r2_improvement:.2f}%")
    print(f"  RMSE reduction: {rmse_improvement:.2f}%")

    print("\n" + "=" * 80)
    print("Step 8: Feature Importance Analysis")
    print("=" * 80)

    feature_names = X_train.columns.tolist()
    importance_weight = model_optimized.get_booster().get_score(importance_type="weight")
    importance_gain = model_optimized.get_booster().get_score(importance_type="gain")

    feature_importance_data = []
    for feature in feature_names:
        feature_idx = f"f{feature_names.index(feature)}"
        weight = importance_weight.get(feature_idx, 0)
        gain = importance_gain.get(feature_idx, 0)
        feature_importance_data.append(
            {
                "Feature": feature,
                "Importance_Weight": weight,
                "Importance_Gain": gain,
            }
        )

    feature_importance = pd.DataFrame(feature_importance_data)
    feature_importance["Importance_Weight_Norm"] = (
        feature_importance["Importance_Weight"]
        / feature_importance["Importance_Weight"].sum()
        * 100
    )
    feature_importance["Importance_Gain_Norm"] = (
        feature_importance["Importance_Gain"]
        / feature_importance["Importance_Gain"].sum()
        * 100
    )
    feature_importance = feature_importance.sort_values("Importance_Gain", ascending=False)

    print("\nTop 20 Important Features (by Gain):")
    print(
        feature_importance.head(20)[["Feature", "Importance_Gain_Norm"]].to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("Step 9: Generate Visualization Results")
    print("=" * 80)

    models_data = [
        ("Basic", y_train_pred_basic, y_train, "Train"),
        ("Basic", y_val_pred_basic, y_val, "Val"),
        ("Basic", y_test_pred_basic, y_test, "Test"),
        ("Optimized", y_train_pred_opt, y_train, "Train"),
        ("Optimized", y_val_pred_opt, y_val, "Val"),
        ("Optimized", y_test_pred_opt, y_test, "Test"),
    ]

    plot_training_curves(evals_result_basic, model_basic, evals_result_opt, model_optimized, output_dir)
    plot_prediction_scatter(models_data, output_dir)
    plot_time_series(y_test, y_test_pred_basic, y_test_pred_opt, output_dir)
    plot_residuals(models_data, output_dir)
    plot_feature_importance(feature_importance, output_dir)
    plot_model_comparison(test_results, output_dir)
    plot_error_distribution(y_test - y_test_pred_basic, y_test - y_test_pred_opt, output_dir)

    print("\n" + "=" * 80)
    print("Step 10: Save Results")
    print("=" * 80)

    save_results(
        output_dir,
        model_dir,
        all_results,
        feature_importance,
        params_optimized,
        y_test,
        y_test_pred_basic,
        y_test_pred_opt,
        model_optimized,
    )

    print("\nGenerated files:")
    print("\nCSV files:")
    print("  - model_performance.csv       Model performance comparison")
    print("  - feature_importance.csv     Feature importance")
    print("  - best_parameters.csv        Best parameters")
    print("  - predictions.csv             Prediction results")

    print("\nChart files:")
    print("  - training_curves.png        Training curves")
    print("  - prediction_scatter.png      Prediction vs Actual scatter plot")
    print("  - timeseries_comparison.png   Time series prediction comparison")
    print("  - residuals_analysis.png      Residual analysis")
    print("  - feature_importance.png      Feature importance bar chart")
    print("  - model_comparison.png        Model performance comparison bar chart")
    print("  - error_distribution.png      Prediction error distribution")

    print("\nModel files:")
    print("  - xgboost_optimized.txt      XGBoost model (text format)")
    print("  - xgboost_optimized.pkl       XGBoost model (pickle format)")

    best_model = test_results.iloc[0]
    print(f"\nBest model: {best_model['Model']}")
    print(f"  R²: {best_model['R²']:.4f}")
    print(f"  RMSE: {best_model['RMSE']:.2f} μg/m³")
    print(f"  MAE: {best_model['MAE']:.2f} μg/m³")
    print(f"  MAPE: {best_model['MAPE']:.2f}%")

    print("\nTop 5 Most Important Features:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Importance_Gain_Norm']:.2f}%")

    print("\n" + "=" * 80)
    print("XGBoost PM2.5 Concentration Prediction Completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

