"""
Beijing PM2.5 Concentration Prediction - XGBoost Model (NetCDF Version)
读取 ERA5 气象数据的逻辑参考自 `LightGBM-NC.py`，实现基于 NetCDF 文件的高效加载流程
"""

import os
import glob
import warnings
import calendar
import pickle
import multiprocessing
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    print("提示：未检测到 tqdm，进度显示将使用简易模式。")
    print("      可执行 `pip install tqdm` 获得更友好的进度条体验。")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100

np.random.seed(RANDOM_SEED)


def daterange(start: datetime, end: datetime):
    for n_days in range(int((end - start).days) + 1):
        yield start + timedelta(n_days)


def find_file(base_path: str, date_str: str, prefix: str) -> str | None:
    filename = f"{prefix}_{date_str}.csv"
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


def read_pollution_day(
    date: datetime,
    pollution_all_path: str,
    pollution_extra_path: str,
    pollutants: list[str],
) -> pd.DataFrame | None:
    date_str = date.strftime("%Y%m%d")
    all_file = find_file(pollution_all_path, date_str, "beijing_all")
    extra_file = find_file(pollution_extra_path, date_str, "beijing_extra")

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
    print("\n加载污染物数据...")
    print(f"使用 {MAX_WORKERS} 个并行线程")
    dates = list(daterange(start_date, end_date))
    pollution_dfs: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                read_pollution_day,
                date,
                pollution_all_path,
                pollution_extra_path,
                pollutants,
            ): date
            for date in dates
        }

        if TQDM_AVAILABLE:
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="污染物数据加载",
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
                        f"  已处理 {idx}/{len(futures)} 天数据 "
                        f"({idx / len(futures) * 100:.1f}%)"
                    )

    if pollution_dfs:
        print(f"  成功读取 {len(pollution_dfs)}/{len(dates)} 天数据")
        df_poll_all = pd.concat(pollution_dfs)
        df_poll_all.ffill(inplace=True)
        df_poll_all.fillna(df_poll_all.mean(), inplace=True)
        print(f"污染物数据加载完成，形状：{df_poll_all.shape}")
        return df_poll_all

    print("⚠️ 未成功加载任何污染物数据！")
    return pd.DataFrame()


def read_era5_month(
    year: int,
    month: int,
    era5_path: str,
    era5_vars: list[str],
    beijing_lats: np.ndarray,
    beijing_lons: np.ndarray,
) -> pd.DataFrame | None:
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
                    f"[WARN] {os.path.basename(file_path)} 缺少目标变量列表中的字段，跳过"
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
                    print(f"[WARN] {os.path.basename(file_path)} 缺少时间坐标，跳过")
                    continue
                ds_subset = ds_subset.sortby("time")

                if fallback_used:
                    try:
                        ds_subset = ds_subset.sel(time=slice(month_start, month_end))
                    except Exception as err:
                        print(
                            f"[WARN] {os.path.basename(file_path)} 时间筛选失败：{err}"
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
                    f"天数: {ds_daily.sizes.get('time', 0)}, 变量数: {len(ds_daily.data_vars)}"
                )
        except Exception as err:
            print(f"[ERROR] 读取 {os.path.basename(file_path)} 失败：{type(err).__name__}: {err}")
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
        f"  成功读取 ERA5 数据：{year}-{month:02d}，天数: {len(df_month)}，变量: {len(df_month.columns)}"
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
    print("\n加载 ERA5 气象数据 (NetCDF)...")
    print(f"使用 {MAX_WORKERS} 个并行线程")
    print(f"气象数据目录: {era5_path}")
    print(f"目录是否存在: {os.path.exists(era5_path)}")

    if os.path.exists(era5_path):
        all_nc = glob.glob(os.path.join(era5_path, "**", "*.nc"), recursive=True)
        print(f"已发现 {len(all_nc)} 个 NetCDF 文件")
        if all_nc:
            print(f"示例文件: {[os.path.basename(f) for f in all_nc[:5]]}")

    era5_dfs: list[pd.DataFrame] = []
    years = range(start_year, end_year + 1)
    months = range(1, 13)

    month_tasks = [
        (year, month)
        for year in years
        for month in months
        if not (year == end_year and month > 12)
    ]
    print(f"计划加载 {len(month_tasks)} 个月的数据...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                read_era5_month,
                year,
                month,
                era5_path,
                era5_vars,
                beijing_lats,
                beijing_lons,
            ): (year, month)
            for year, month in month_tasks
        }

        success_count = 0
        if TQDM_AVAILABLE:
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="气象数据加载",
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
                        f"  进度: {idx}/{len(futures)} 月份 (成功 {success_count} 个, "
                        f"{idx / len(futures) * 100:.1f}%)"
                    )

        print(f"  共成功加载 {success_count}/{len(futures)} 个月份的数据")

    if not era5_dfs:
        print("\n❌ 错误：没有成功加载任何气象数据文件！")
        print("排查建议：")
        print("1. 确认文件命名是否包含 YYYYMM 或者目录下是否存在 .nc 文件")
        print("2. 检查文件内容是否包含时间、经纬度坐标以及目标变量")
        print("3. 核对 ERA5 数据路径是否正确")
        return pd.DataFrame()

    print("\n合并气象数据...")
    df_era5_all = pd.concat(era5_dfs, axis=0)
    print("  去重...")
    df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep="first")]
    print("  排序...")
    df_era5_all.sort_index(inplace=True)

    print(f"合并后形状: {df_era5_all.shape}")
    print(f"时间范围: {df_era5_all.index.min()} 至 {df_era5_all.index.max()}")
    preview_cols = list(df_era5_all.columns[:10])
    print(f"示例变量: {preview_cols}{'...' if len(df_era5_all.columns) > 10 else ''}")

    print("  处理缺失值...")
    initial_na = df_era5_all.isna().sum().sum()
    df_era5_all.ffill(inplace=True)
    df_era5_all.bfill(inplace=True)
    df_era5_all.fillna(df_era5_all.mean(), inplace=True)
    final_na = df_era5_all.isna().sum().sum()
    print(f"缺失值: {initial_na} -> {final_na}")

    print("气象数据加载完成。")
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
    df_copy["day_of_year"] = df_copy.index.dayofyear
    df_copy["day_of_week"] = df_copy.index.dayofweek
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
    safe_pred = y_pred[mask.values]  # 使用布尔数组直接索引 numpy 数组

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
            ax.plot(evals[train_key]["rmse"], label="训练集", linewidth=2)
            ax.plot(evals[valid_key]["rmse"], label="验证集", linewidth=2)
        elif "train" in evals:
            ax.plot(evals["train"]["rmse"], label="训练集", linewidth=2)
            ax.plot(evals["valid"]["rmse"], label="验证集", linewidth=2)
        if hasattr(model, "best_iteration") and model.best_iteration is not None:
            ax.axvline(
                x=model.best_iteration,
                color="r",
                linestyle="--",
                linewidth=1.5,
                label=f"最佳迭代 ({model.best_iteration})",
            )
        ax.set_xlabel("迭代轮数", fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    _plot_curve(axes[0], evals_result, model_basic, "XGBoost 基础模型 - 训练过程")
    _plot_curve(axes[1], evals_result_opt, model_opt, "XGBoost 优化模型 - 训练过程")

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
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="理想预测")

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ax.set_xlabel("真实 PM2.5 浓度 (μg/m³)", fontsize=11)
        ax.set_ylabel("预测 PM2.5 浓度 (μg/m³)", fontsize=11)
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

    axes[0].plot(time_idx, y_test.iloc[plot_idx], "k-", label="真实值", linewidth=2, alpha=0.8)
    axes[0].plot(
        time_idx,
        y_test_pred_basic[plot_idx],
        "b--",
        label="基础模型预测",
        linewidth=1.5,
        alpha=0.7,
    )
    axes[0].set_xlabel("日期", fontsize=12)
    axes[0].set_ylabel("PM2.5 浓度 (μg/m³)", fontsize=12)
    axes[0].set_title(
        "XGBoost 基础模型 - 测试集后 300 天预测对比",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

    axes[1].plot(time_idx, y_test.iloc[plot_idx], "k-", label="真实值", linewidth=2, alpha=0.8)
    axes[1].plot(
        time_idx,
        y_test_pred_opt[plot_idx],
        "g--",
        label="优化模型预测",
        linewidth=1.5,
        alpha=0.7,
    )
    axes[1].set_xlabel("日期", fontsize=12)
    axes[1].set_ylabel("PM2.5 浓度 (μg/m³)", fontsize=12)
    axes[1].set_title(
        "XGBoost 优化模型 - 测试集后 300 天预测对比",
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
        ax.set_xlabel("预测值 (μg/m³)", fontsize=11)
        ax.set_ylabel("残差 (μg/m³)", fontsize=11)
        ax.set_title(
            f"XGBoost_{model_name} - {dataset}\n残差均值={residuals.mean():.2f}, "
            f"标准差={residuals.std():.2f}",
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
    axes[0].set_xlabel("重要性 (%)", fontsize=12)
    axes[0].set_title(f"Top {top_n} 重要特征 (按 Gain)", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="x")
    axes[0].invert_yaxis()

    axes[1].barh(
        range(top_n),
        top_features_weight["Importance_Weight_Norm"],
        color="coral",
    )
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(top_features_weight["Feature"], fontsize=10)
    axes[1].set_xlabel("重要性 (%)", fontsize=12)
    axes[1].set_title(f"Top {top_n} 重要特征 (按 Weight)", fontsize=13, fontweight="bold")
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
        axes[idx].set_xticklabels(["基础", "优化"], fontsize=11)
        axes[idx].set_ylabel(metric, fontsize=12)
        if metric == "R²":
            axes[idx].set_title(f"{metric} 对比\n(越高越好)", fontsize=12, fontweight="bold")
        else:
            axes[idx].set_title(f"{metric} 对比\n(越低越好)", fontsize=12, fontweight="bold")
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
    axes[0].axvline(x=0, color="r", linestyle="--", linewidth=2.5, label="零误差")
    axes[0].set_xlabel("预测误差 (μg/m³)", fontsize=12)
    axes[0].set_ylabel("频数", fontsize=12)
    axes[0].set_title(
        f"基础模型 - 误差分布\n均值={errors_basic.mean():.2f}, 标准差={errors_basic.std():.2f}",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].hist(errors_opt, bins=50, color="green", alpha=0.7, edgecolor="black")
    axes[1].axvline(x=0, color="r", linestyle="--", linewidth=2.5, label="零误差")
    axes[1].set_xlabel("预测误差 (μg/m³)", fontsize=12)
    axes[1].set_ylabel("频数", fontsize=12)
    axes[1].set_title(
        f"优化模型 - 误差分布\n均值={errors_opt.mean():.2f}, 标准差={errors_opt.std():.2f}",
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
    with open(model_dir / "xgboost_optimized.pkl", "wb") as file:
        pickle.dump(model_optimized, file)


def main():
    print("=" * 80)
    print("北京 PM2.5 浓度预测 - XGBoost 模型（NetCDF 数据版本）")
    print("=" * 80)

    print("\n配置参数...")

    pollution_all_path = r"E:\DATA Science\Benchmark\all(AQI+PM2.5+PM10)"
    pollution_extra_path = r"E:\DATA Science\Benchmark\extra(SO2+NO2+CO+O3)"
    era5_path = r"E:\DATA Science\ERA5-Beijing-NC"

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

    print(f"数据时间范围: {start_date.date()} 至 {end_date.date()}")
    print(f"目标变量: PM2.5 浓度")
    print(f"输出目录: {output_dir.resolve()}")
    print(f"模型保存目录: {model_dir.resolve()}")
    print(f"CPU 核心数: {CPU_COUNT}, 并行线程数: {MAX_WORKERS}")

    print("\n" + "=" * 80)
    print("步骤 1: 数据加载与预处理")
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

    print("\n数据加载检查:")
    print(f"  污染物数据形状: {df_pollution.shape}")
    print(f"  气象数据形状: {df_era5.shape}")

    if df_pollution.empty:
        print("\n⚠️ 警告: 污染物数据为空，请检查数据路径和文件。")
        raise SystemExit(1)
    if df_era5.empty:
        print("\n⚠️ 警告: 气象数据为空，请检查数据路径和文件。")
        raise SystemExit(1)

    df_pollution.index = pd.to_datetime(df_pollution.index)
    df_era5.index = pd.to_datetime(df_era5.index)

    print(
        f"  污染物数据时间范围: {df_pollution.index.min()} 至 {df_pollution.index.max()}"
    )
    print(f"  气象数据时间范围: {df_era5.index.min()} 至 {df_era5.index.max()}")

    print("\n数据合并...")
    df_combined = df_pollution.join(df_era5, how="inner")

    if df_combined.empty:
        print("\n❌ 错误: 合并后数据为空，可能两类数据日期没有交集。")
        print(f"  污染物数据行数: {len(df_pollution)}")
        print(f"  气象数据行数: {len(df_era5)}")
        print(f"  合并后行数: {len(df_combined)}")
        raise SystemExit(1)

    print("\n创建特征...")
    df_combined = create_features(df_combined)

    print("\n数据清洗...")
    df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    initial_rows = len(df_combined)
    df_combined.dropna(inplace=True)
    final_rows = len(df_combined)
    print(f"移除缺失行数: {initial_rows - final_rows}")

    print(f"\n合并后数据形状: {df_combined.shape}")
    print(
        f"时间范围: {df_combined.index.min().date()} 至 "
        f"{df_combined.index.max().date()}"
    )
    print(f"样本量: {len(df_combined)}")
    print(f"特征数量: {df_combined.shape[1]}")

    print("\n特征示例 (前 20 个):")
    for idx, col in enumerate(df_combined.columns[:20], 1):
        print(f"  {idx}. {col}")
    if len(df_combined.columns) > 20:
        print(f"  ... 以及另外 {len(df_combined.columns) - 20} 个特征")

    print("\n" + "=" * 80)
    print("步骤 2: 特征选择与数据准备")
    print("=" * 80)

    target = "PM2.5"
    exclude_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "year"]
    numeric_features = [
        col
        for col in df_combined.select_dtypes(include=[np.number]).columns
        if col not in exclude_cols
    ]

    print(f"\n选取的特征数量: {len(numeric_features)}")
    print(f"目标变量: {target}")

    X = df_combined[numeric_features].copy()
    y = df_combined[target].copy()

    if X.empty or y.empty:
        print("\n❌ 错误: 有效建模数据为空，请检查数据源与预处理流程。")
        raise SystemExit(1)

    print(f"\n特征矩阵形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    print("\nPM2.5 统计信息:")
    print(f"  均值: {y.mean():.2f} μg/m³")
    print(f"  标准差: {y.std():.2f} μg/m³")
    print(f"  最小值: {y.min():.2f} μg/m³")
    print(f"  最大值: {y.max():.2f} μg/m³")
    print(f"  中位数: {y.median():.2f} μg/m³")

    print("\n" + "=" * 80)
    print("步骤 3: 数据集划分")
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

    print(f"\n训练集: {len(X_train)} 样本 ({len(X_train) / n_samples * 100:.1f}%)")
    print(
        f"  时间范围: {X_train.index.min().date()} 至 {X_train.index.max().date()}"
    )
    print(f"  PM2.5: {y_train.mean():.2f} ± {y_train.std():.2f} μg/m³")

    print(f"\n验证集: {len(X_val)} 样本 ({len(X_val) / n_samples * 100:.1f}%)")
    print(f"  时间范围: {X_val.index.min().date()} 至 {X_val.index.max().date()}")
    print(f"  PM2.5: {y_val.mean():.2f} ± {y_val.std():.2f} μg/m³")

    print(f"\n测试集: {len(X_test)} 样本 ({len(X_test) / n_samples * 100:.1f}%)")
    print(f"  时间范围: {X_test.index.min().date()} 至 {X_test.index.max().date()}")
    print(f"  PM2.5: {y_test.mean():.2f} ± {y_test.std():.2f} μg/m³")

    print("\n" + "=" * 80)
    print("步骤 4: XGBoost 基础模型训练")
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
    }

    print("\n基础模型参数:")
    for key, value in params_basic.items():
        print(f"  {key}: {value}")

    print("\n开始训练基础模型...")
    model_basic = xgb.XGBRegressor(**params_basic, early_stopping_rounds=50)
    evals_result_basic: dict = {}
    model_basic.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )
    # 获取训练历史
    evals_result_basic = model_basic.evals_result()

    print("\n✓ 基础模型训练完成")
    if hasattr(model_basic, "best_iteration") and model_basic.best_iteration is not None:
        print(f"  最佳迭代轮数: {model_basic.best_iteration}")

    y_train_pred_basic = model_basic.predict(X_train)
    y_val_pred_basic = model_basic.predict(X_val)
    y_test_pred_basic = model_basic.predict(X_test)

    results_basic = [
        evaluate_model(y_train, y_train_pred_basic, "Train"),
        evaluate_model(y_val, y_val_pred_basic, "Validation"),
        evaluate_model(y_test, y_test_pred_basic, "Test"),
    ]
    results_basic_df = pd.DataFrame(results_basic)
    print("\n基础模型表现:")
    print(results_basic_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("步骤 5: 超参数优化 (可选)")
    print("=" * 80)

    optimize_input = input("\n是否执行网格搜索优化? (y/n, 默认为 n): ").strip().lower()
    optimize = optimize_input == "y"

    if optimize:
        print("\n执行网格搜索，请稍候...\n")
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
        )
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            verbose=2,
            n_jobs=min(4, MAX_WORKERS),
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print("\n最佳参数组合:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

        params_optimized = {
            **best_params,
            "objective": "reg:squarederror",
            "random_state": RANDOM_SEED,
            "n_jobs": MAX_WORKERS,
            "eval_metric": "rmse",
        }
        model_optimized = xgb.XGBRegressor(**params_optimized, early_stopping_rounds=50)
        evals_result_opt: dict = {}
        model_optimized.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=50,
        )
        # 获取训练历史
        evals_result_opt = model_optimized.evals_result()
        print("\n✓ 使用最优参数重新训练模型完成")
    else:
        print("\n跳过超参数优化，沿用基础模型参数。")
        params_optimized = params_basic
        model_optimized = model_basic
        evals_result_opt = evals_result_basic

    print("\n" + "=" * 80)
    print("步骤 6: 优化模型评估")
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
    print("\n优化模型表现:")
    print(results_opt_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("步骤 7: 模型效果对比")
    print("=" * 80)

    results_basic_df["Model"] = "XGBoost_Basic"
    results_opt_df["Model"] = "XGBoost_Optimized"
    all_results = pd.concat([results_basic_df, results_opt_df])
    all_results = all_results[["Model", "Dataset", "R²", "RMSE", "MAE", "MAPE"]]

    print("\n模型性能对比:")
    print(all_results.to_string(index=False))

    test_results = all_results[all_results["Dataset"] == "Test"].sort_values(
        "R²", ascending=False
    )
    print("\n测试集表现排名:")
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

    print("\n优化效果:")
    print(f"  R² 提升: {r2_improvement:.2f}%")
    print(f"  RMSE 降低: {rmse_improvement:.2f}%")

    print("\n" + "=" * 80)
    print("步骤 8: 特征重要性分析")
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
    feature_importance.sort_values("Importance_Gain", ascending=False, inplace=True)

    print("\nTop 20 重要特征 (按 Gain):")
    print(
        feature_importance.head(20)[["Feature", "Importance_Gain_Norm"]].to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("步骤 9: 生成可视化结果")
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
    print("步骤 10: 保存结果")
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

    print("\n生成的文件:")
    print("\nCSV 文件:")
    print("  - model_performance.csv       模型性能对比")
    print("  - feature_importance.csv      特征重要性")
    print("  - best_parameters.csv         最优参数")
    print("  - predictions.csv             预测结果")

    print("\n图表文件:")
    print("  - training_curves.png         训练曲线")
    print("  - prediction_scatter.png      预测 vs 实际散点图")
    print("  - timeseries_comparison.png   时间序列预测对比")
    print("  - residuals_analysis.png      残差分析")
    print("  - feature_importance.png      特征重要性柱状图")
    print("  - model_comparison.png        模型性能对比柱状图")
    print("  - error_distribution.png      预测误差分布")

    print("\n模型文件:")
    print("  - xgboost_optimized.txt       XGBoost 模型 (文本格式)")
    print("  - xgboost_optimized.pkl       XGBoost 模型 (pickle 格式)")

    best_model = test_results.iloc[0]
    print(f"\n最佳模型: {best_model['Model']}")
    print(f"  R²: {best_model['R²']:.4f}")
    print(f"  RMSE: {best_model['RMSE']:.2f} μg/m³")
    print(f"  MAE: {best_model['MAE']:.2f} μg/m³")
    print(f"  MAPE: {best_model['MAPE']:.2f}%")

    print("\nTop 5 最重要特征:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Importance_Gain_Norm']:.2f}%")

    print("\n" + "=" * 80)
    print("XGBoost PM2.5 浓度预测完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

