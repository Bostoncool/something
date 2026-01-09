import os
import glob
import re
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
    df = df.copy()

    # -------------------------
    # 1. 风速风向（保留你原来的）
    # -------------------------
    if {"u10", "v10"}.issubset(df.columns):
        df["wind_speed_10m"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
        df["wind_dir_10m"] = (np.degrees(np.arctan2(df["v10"], df["u10"])) + 360) % 360

    if {"u100", "v100"}.issubset(df.columns):
        df["wind_speed_100m"] = np.sqrt(df["u100"]**2 + df["v100"]**2)
        df["wind_dir_100m"] = (np.degrees(np.arctan2(df["v100"], df["u100"])) + 360) % 360

    # -------------------------
    # 2. 基础时间特征
    # -------------------------
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["day_of_week"] = df.index.day_of_week
    df["day_of_year"] = df.index.day_of_year
    df["week_of_year"] = df.index.isocalendar().week.astype("int")

    # 季节
    df["season"] = df["month"].map(
        {12: 1, 1: 1, 2: 1,
         3: 2, 4: 2, 5: 2,
         6: 3, 7: 3, 8: 3,
         9: 4, 10: 4, 11: 4}
    )

    # 供暖季（强特征）
    df["is_heating"] = ((df["month"] >= 11) | (df["month"] <= 3)).astype(int)

    # -------------------------
    # 3. PM2.5 的 lag & rolling（最重要）
    # -------------------------
    if "PM2.5" in df.columns:
        for lag in [1, 2, 3, 5, 7, 14, 30]:
            df[f"PM25_lag{lag}"] = df["PM2.5"].shift(lag)

        for win in [3, 7, 14, 30]:
            df[f"PM25_ma{win}"] = df["PM2.5"].rolling(win, min_periods=1).mean()

        # 差分（捕获上升或下降趋势）
        df["PM25_diff1"] = df["PM2.5"].diff()
        df["PM25_diff7"] = df["PM2.5"].diff(7)

    # -------------------------
    # 4. 天气变化率（比绝对值更有预测力）
    # -------------------------
    if "t2m" in df.columns:
        df["t2m_diff1"] = df["t2m"].diff()
        df["t2m_diff3"] = df["t2m"].diff(3)

    if "d2m" in df.columns:
        df["humidity_diff1"] = df["d2m"].diff()
        df["humidity_diff3"] = df["d2m"].diff(3)

    if "wind_speed_10m" in df.columns:
        df["wind_change1"] = df["wind_speed_10m"].diff()

    # -------------------------
    # 5. 湿度计算（保留原来逻辑）
    # -------------------------
    if {"t2m", "d2m"}.issubset(df.columns):
        df["temp_dewpoint_diff"] = df["t2m"] - df["d2m"]
        numerator = np.exp(
            (17.625 * (df["d2m"] - 273.15))
            / (243.04 + (df["d2m"] - 273.15))
        )
        denominator = np.exp(
            (17.625 * (df["t2m"] - 273.15))
            / (243.04 + (df["t2m"] - 273.15))
        )
        df["relative_humidity"] = (100 * numerator / denominator).clip(0, 100)

    # -------------------------
    # 6. 风向分类（保留原来逻辑）
    # -------------------------
    if "wind_dir_10m" in df.columns:
        df["wind_dir_category"] = pd.cut(
            df["wind_dir_10m"],
            bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
            labels=list(range(8)),
            include_lowest=True,
        ).astype("Int64")

    # -------------------------
    # 7. 清理
    # -------------------------
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


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


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """统一 CSV 输出（UTF-8-SIG），便于后续脚本读取与绘图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"✓ Saved CSV: {path}")


def _slug_model_name(model_name: str) -> str:
    return model_name.strip().lower().replace(" ", "_").replace("-", "_")


def _build_training_curve_df(evals_result: dict) -> pd.DataFrame:
    """将 XGBoost evals_result 转为可绘图 CSV：iteration / train_rmse / val_rmse。"""
    if not evals_result:
        return pd.DataFrame()

    train_key = "validation_0" if "validation_0" in evals_result else "train"
    val_key = "validation_1" if "validation_1" in evals_result else "valid"

    train_rmse = evals_result.get(train_key, {}).get("rmse", [])
    val_rmse = evals_result.get(val_key, {}).get("rmse", [])
    n = min(len(train_rmse), len(val_rmse)) if val_rmse else len(train_rmse)

    df = pd.DataFrame(
        {
            "iteration": np.arange(1, n + 1),
            "train_rmse": train_rmse[:n],
            "val_rmse": val_rmse[:n] if val_rmse else [np.nan] * n,
        }
    )
    return df


def _get_xgb_feature_importance(
    model: xgb.XGBRegressor, feature_names: list[str]
) -> pd.DataFrame:
    """
    自动适配 XGBoost 的特征重要性 key：
    - 若 booster.get_score() 返回 key 为真实列名 -> 用列名对齐
    - 若返回 key 为 'f0','f1',... -> 用 f{idx} 对齐
    输出 weight / gain 及其归一化百分比
    """
    booster = model.get_booster()

    importance_weight = booster.get_score(importance_type="weight") or {}
    importance_gain = booster.get_score(importance_type="gain") or {}

    # 判断 get_score() 的 key 形式：更偏向列名还是 f0/f1
    keys = list(set(list(importance_weight.keys()) + list(importance_gain.keys())))
    has_f_style = any(re.fullmatch(r"f\d+", k) for k in keys)
    has_name_style = any((k in feature_names) for k in keys)

    # 优先策略：
    # 1) 明显是列名 -> 用列名
    # 2) 明显是 f0/f1 -> 用 f{idx}
    # 3) 两者都有或都不明显 -> 每个特征先试列名，再回退 f{idx}
    rows = []
    for idx, feat in enumerate(feature_names):
        fkey = f"f{idx}"

        if has_name_style and not has_f_style:
            # 典型：DataFrame + 列名
            w = float(importance_weight.get(feat, 0.0))
            g = float(importance_gain.get(feat, 0.0))
        elif has_f_style and not has_name_style:
            # 典型：f0/f1...
            w = float(importance_weight.get(fkey, 0.0))
            g = float(importance_gain.get(fkey, 0.0))
        else:
            # 兜底：两种都尝试（先列名后 fkey）
            w = float(importance_weight.get(feat, importance_weight.get(fkey, 0.0)))
            g = float(importance_gain.get(feat, importance_gain.get(fkey, 0.0)))

        rows.append(
            {"Feature": feat, "Importance_Weight": w, "Importance_Gain": g}
        )

    df = pd.DataFrame(rows)

    weight_sum = df["Importance_Weight"].sum()
    gain_sum = df["Importance_Gain"].sum()

    df["Importance_Weight_Norm"] = (
        df["Importance_Weight"] / weight_sum * 100 if weight_sum != 0 else 0.0
    )
    df["Importance_Gain_Norm"] = (
        df["Importance_Gain"] / gain_sum * 100 if gain_sum != 0 else 0.0
    )

    df = df.sort_values("Importance_Gain", ascending=False).reset_index(drop=True)

    # 可选：快速自检（你也可以删掉这几行）
    # print(f"[FI] keys sample: {keys[:10]}")
    # print(f"[FI] detected: has_f_style={has_f_style}, has_name_style={has_name_style}, "
    #       f"nonzero_gain={int((df['Importance_Gain']>0).sum())}/{len(df)}")

    return df


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
    print("Step 3: Dataset Splitting (Time-based)")
    print("=" * 80)

    # 确保索引是单调的（排序）
    if not X.index.is_monotonic_increasing:
        print("Warning: Index is not monotonic, sorting...")
        X = X.sort_index()
        y = y.sort_index()

    # 按年份划分数据集（避免数据泄漏）
    train_end = pd.Timestamp("2020-12-31")
    val_end = pd.Timestamp("2021-12-31")

    # 使用布尔索引进行更安全的日期切片
    train_mask = X.index <= train_end
    val_mask = (X.index > train_end) & (X.index <= val_end)
    test_mask = X.index > val_end

    X_train = X[train_mask].copy()
    X_val = X[val_mask].copy()
    X_test = X[test_mask].copy()

    y_train = y[train_mask].copy()
    y_val = y[val_mask].copy()
    y_test = y[test_mask].copy()

    # 对目标变量进行log1p变换
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    print(f"\nTraining set: {len(X_train)} samples (2015-2020)")
    print(
        f"  Time range: {X_train.index.min().date()} to {X_train.index.max().date()}"
    )
    print(f"  PM2.5: {y_train.mean():.2f} ± {y_train.std():.2f} μg/m³ (log-transformed for training)")

    print(f"\nValidation set: {len(X_val)} samples (2021)")
    print(f"  Time range: {X_val.index.min().date()} to {X_val.index.max().date()}")
    print(f"  PM2.5: {y_val.mean():.2f} ± {y_val.std():.2f} μg/m³ (log-transformed for training)")

    print(f"\nTest set: {len(X_test)} samples (2022-2024)")
    print(f"  Time range: {X_test.index.min().date()} to {X_test.index.max().date()}")
    print(f"  PM2.5: {y_test.mean():.2f} ± {y_test.std():.2f} μg/m³ (original scale for evaluation)")

    print("\n" + "=" * 80)
    print("Step 4: XGBoost Basic Model Training")
    print("=" * 80)

    params_basic = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 8,
        "learning_rate": 0.03,
        "n_estimators": 600,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_lambda": 1.2,
        "reg_alpha": 0.1,
        "random_state": RANDOM_SEED,
        "eval_metric": "rmse",
    }

    print("\nBasic model parameters:")
    for key, value in params_basic.items():
        print(f"  {key}: {value}")

    print("\nStarting basic model training...")
    model_basic = xgb.XGBRegressor(
        **params_basic,
        early_stopping_rounds=80,
        callbacks=[
            XGBoostProgressBar(
                params_basic["n_estimators"], description="Basic Model Training"
            )
        ],
    )
    evals_result_basic: dict = {}
    model_basic.fit(
        X_train,
        y_train_log,
        eval_set=[(X_train, y_train_log), (X_val, y_val_log)],
        verbose=False,
    )
    # Get training history
    evals_result_basic = model_basic.evals_result()

    print("\n✓ Basic model training completed")
    if hasattr(model_basic, "best_iteration") and model_basic.best_iteration is not None:
        print(f"  Best iteration: {model_basic.best_iteration}")

    # 预测结果转换回原始尺度
    y_train_pred_basic = np.expm1(model_basic.predict(X_train))
    y_val_pred_basic = np.expm1(model_basic.predict(X_val))
    y_test_pred_basic = np.expm1(model_basic.predict(X_test))

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
            early_stopping_rounds=80,
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
            y_train_log,
            eval_set=[(X_train, y_train_log), (X_val, y_val_log)],
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

    # 预测结果转换回原始尺度
    y_train_pred_opt = np.expm1(model_optimized.predict(X_train))
    y_val_pred_opt = np.expm1(model_optimized.predict(X_val))
    y_test_pred_opt = np.expm1(model_optimized.predict(X_test))

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
    fi_basic = _get_xgb_feature_importance(model_basic, feature_names)
    fi_opt = _get_xgb_feature_importance(model_optimized, feature_names)

    print("\nTop 20 Important Features (Optimized, by Gain):")
    print(fi_opt.head(20)[["Feature", "Importance_Gain_Norm"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("Step 9: Export Plot Data as CSV (No Images)")
    print("=" * 80)

    # -------------------------
    # A) Training curves (rmse vs iteration)
    # -------------------------
    curve_basic = _build_training_curve_df(model_basic.evals_result())
    curve_opt = _build_training_curve_df(model_optimized.evals_result())
    save_csv(curve_basic, output_dir / "plot_training_curve__xgboost_basic.csv")
    save_csv(curve_opt, output_dir / "plot_training_curve__xgboost_optimized.csv")

    # -------------------------
    # B) Metrics tables (按模型/按数据集拆分)
    # -------------------------
    results_basic_df["Model"] = "XGBoost_Basic"
    results_opt_df["Model"] = "XGBoost_Optimized"
    metrics_all = pd.concat([results_basic_df, results_opt_df], ignore_index=True)
    metrics_all = metrics_all[["Model", "Dataset", "R²", "RMSE", "MAE", "MAPE"]]

    save_csv(metrics_all, output_dir / "metrics__all_models_train_val_test.csv")
    for model_name in metrics_all["Model"].unique():
        model_slug = _slug_model_name(model_name)
        part_model = metrics_all[metrics_all["Model"] == model_name].copy()
        save_csv(part_model, output_dir / f"metrics__{model_slug}__train_val_test.csv")
        for ds_name in metrics_all["Dataset"].unique():
            part = part_model[part_model["Dataset"] == ds_name]
            if not part.empty:
                save_csv(part, output_dir / f"metrics__{model_slug}__{ds_name.lower()}.csv")

    test_ranking = metrics_all[metrics_all["Dataset"] == "Test"].sort_values("R²", ascending=False)
    save_csv(test_ranking, output_dir / "plot_metrics_ranking__test_only.csv")

    # -------------------------
    # C) Predictions / Scatter / Residuals / Error distribution (按模型 + 按数据集拆分)
    # -------------------------
    preds_pack = [
        ("xgboost_basic", "Train", y_train, y_train_pred_basic),
        ("xgboost_basic", "Val", y_val, y_val_pred_basic),
        ("xgboost_basic", "Test", y_test, y_test_pred_basic),
        ("xgboost_optimized", "Train", y_train, y_train_pred_opt),
        ("xgboost_optimized", "Val", y_val, y_val_pred_opt),
        ("xgboost_optimized", "Test", y_test, y_test_pred_opt),
    ]

    # 合并预测（兼容“统一读取”场景）
    # 注意：不同数据集索引不连续，直接按行拼接保存
    predictions_all_rows: list[pd.DataFrame] = []
    for model_slug, dataset, y_true_s, y_pred_arr in preds_pack:
        df_pred = pd.DataFrame(
            {
                "Date": y_true_s.index,
                "Actual_PM25": y_true_s.values,
                "Predicted_PM25": y_pred_arr,
            }
        )
        df_pred["Dataset"] = dataset
        df_pred["Model"] = model_slug
        df_pred["Error"] = df_pred["Actual_PM25"] - df_pred["Predicted_PM25"]
        predictions_all_rows.append(df_pred)

        # 1) scatter plot data（按模型/数据集）
        save_csv(df_pred[["Date", "Actual_PM25", "Predicted_PM25", "Dataset", "Model"]], output_dir / f"plot_scatter__{model_slug}__{dataset.lower()}.csv")

        # 2) residuals plot data（按模型/数据集）
        df_res = df_pred[["Date", "Actual_PM25", "Predicted_PM25", "Error", "Dataset", "Model"]].rename(columns={"Error": "Residual"})
        save_csv(df_res, output_dir / f"plot_residuals__{model_slug}__{dataset.lower()}.csv")

        # 3) error distribution plot data（按模型/数据集）
        df_err = df_pred[["Date", "Actual_PM25", "Predicted_PM25", "Error", "Dataset", "Model"]]
        save_csv(df_err, output_dir / f"plot_error_distribution__{model_slug}__{dataset.lower()}.csv")

        # 4) predictions（按模型/数据集）
        save_csv(df_pred, output_dir / f"xgboost_predictions__{model_slug}__{dataset.lower()}.csv")

    predictions_all = pd.concat(predictions_all_rows, ignore_index=True)
    save_csv(predictions_all, output_dir / "xgboost_predictions_all_models__train_val_test.csv")

    # -------------------------
    # D) Time series plot data（参考 RF：last-year sampled + x_axis，使用 Test 集）
    # -------------------------
    plot_df = pd.DataFrame(
        {
            "time": y_test.index,
            "y_true": y_test.values,
            "y_pred_basic": y_test_pred_basic,
            "y_pred_opt": y_test_pred_opt,
        }
    ).sort_values("time").reset_index(drop=True)

    plot_range = min(365, len(plot_df))
    plot_df_subset = plot_df.iloc[-plot_range:].copy()

    step = 4
    plot_df_sampled = plot_df_subset.iloc[::step].copy().reset_index(drop=True)

    x_axis = np.arange(len(plot_df_sampled))
    ts_common = pd.DataFrame(
        {
            "x_axis": x_axis,
            "time": plot_df_sampled["time"].values,
            "y_true": plot_df_sampled["y_true"].values,
        }
    )
    save_csv(ts_common, output_dir / "plot_ts_lastyear_sampled__actual.csv")
    save_csv(ts_common.assign(y_pred=plot_df_sampled["y_pred_basic"].values), output_dir / "plot_ts_lastyear_sampled__xgboost_basic.csv")
    save_csv(ts_common.assign(y_pred=plot_df_sampled["y_pred_opt"].values), output_dir / "plot_ts_lastyear_sampled__xgboost_optimized.csv")

    # 同时输出一个“简单时序 sampled”（完整 Test 集采样），便于快速画线
    plot_df_simple = pd.DataFrame(
        {
            "time": y_test.index,
            "y_true": y_test.values,
            "y_pred_xgboost_basic": y_test_pred_basic,
            "y_pred_xgboost_optimized": y_test_pred_opt,
        }
    ).sort_values("time").reset_index(drop=True)
    plot_df_simple_sampled = plot_df_simple.iloc[::step].copy()
    save_csv(
        plot_df_simple_sampled[["time", "y_true", "y_pred_xgboost_basic"]].rename(columns={"y_pred_xgboost_basic": "y_pred"}),
        output_dir / "plot_ts_simple_sampled__xgboost_basic.csv",
    )
    save_csv(
        plot_df_simple_sampled[["time", "y_true", "y_pred_xgboost_optimized"]].rename(columns={"y_pred_xgboost_optimized": "y_pred"}),
        output_dir / "plot_ts_simple_sampled__xgboost_optimized.csv",
    )

    # -------------------------
    # E) Feature importance（按模型拆分 + TopN）
    # -------------------------
    save_csv(fi_basic, output_dir / "plot_feature_importance__xgboost_basic.csv")
    save_csv(fi_opt, output_dir / "plot_feature_importance__xgboost_optimized.csv")

    top_n = 20
    save_csv(fi_basic.head(min(top_n, len(fi_basic))), output_dir / f"plot_feature_importance_top{min(top_n, len(fi_basic))}__xgboost_basic.csv")
    save_csv(fi_opt.head(min(top_n, len(fi_opt))), output_dir / f"plot_feature_importance_top{min(top_n, len(fi_opt))}__xgboost_optimized.csv")

    # -------------------------
    # F) Best params（仅当做“参数表”输出；不保存模型文件夹/模型文件）
    # -------------------------
    save_csv(pd.DataFrame([params_optimized]), output_dir / "best_parameters__xgboost_optimized.csv")

    print("\nGenerated files:")
    print("\nCSV files (plot data + metrics):")
    print("  - plot_training_curve__xgboost_basic.csv / plot_training_curve__xgboost_optimized.csv")
    print("  - metrics__all_models_train_val_test.csv / metrics__xgboost_*__*.csv")
    print("  - plot_metrics_ranking__test_only.csv")
    print("  - xgboost_predictions__{model}__{dataset}.csv / xgboost_predictions_all_models__train_val_test.csv")
    print("  - plot_scatter__{model}__{dataset}.csv")
    print("  - plot_residuals__{model}__{dataset}.csv")
    print("  - plot_error_distribution__{model}__{dataset}.csv")
    print("  - plot_ts_lastyear_sampled__actual.csv / plot_ts_lastyear_sampled__xgboost_*.csv")
    print("  - plot_ts_simple_sampled__xgboost_*.csv")
    print("  - plot_feature_importance__xgboost_*.csv / plot_feature_importance_top{N}__xgboost_*.csv")
    print("  - best_parameters__xgboost_optimized.csv")

    best_model = test_results.iloc[0]
    print(f"\nBest model: {best_model['Model']}")
    print(f"  R²: {best_model['R²']:.4f}")
    print(f"  RMSE: {best_model['RMSE']:.2f} μg/m³")
    print(f"  MAE: {best_model['MAE']:.2f} μg/m³")
    print(f"  MAPE: {best_model['MAPE']:.2f}%")

    print("\nTop 5 Most Important Features:")
    for _, row in fi_opt.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Importance_Gain_Norm']:.2f}%")

    print("\n" + "=" * 80)
    print("XGBoost PM2.5 Concentration Prediction Completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

