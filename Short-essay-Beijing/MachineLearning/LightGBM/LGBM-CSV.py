import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import warnings
from pathlib import Path
import glob
import multiprocessing
from importlib import import_module
from importlib.util import find_spec
import xarray as xr
from netCDF4 import Dataset
import calendar

warnings.filterwarnings('ignore')

CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm is not installed, progress display will use simplified version.")
    print("      You can use 'pip install tqdm' to install for better progress bar display.")

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import lightgbm as lgb

# ---------------------------------------------------------------------------
# Output helpers (统一 UTF-8-SIG)
# ---------------------------------------------------------------------------
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


# 检测GPU可用性（强制要求GPU）
def check_gpu_availability():
    """检查LightGBM GPU支持是否可用，如果不可用则直接退出"""
    import subprocess
    import sys
    
    try:
        # 检查CUDA驱动
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("❌ GPU不可用: nvidia-smi执行失败")
            sys.exit(1)
    except Exception as e:
        print(f"❌ GPU不可用: {e}")
        sys.exit(1)
    
    # 测试LightGBM GPU功能
    try:
        test_X = np.random.rand(100, 10).astype(np.float32)
        test_y = np.random.rand(100).astype(np.float32)
        
        gpu_params = {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'max_bin': 255,
            'force_col_wise': True,
            'gpu_page_size': 2048
        }
        test_data = lgb.Dataset(test_X, label=test_y, params=gpu_params)
        
        test_params = {
            'objective': 'regression',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'num_gpu': 1,
            'max_bin': 255,
            'tree_learner': 'data',
            'force_col_wise': True,
            'gpu_page_size': 2048,
            'verbose': -1
        }
        
        lgb.train(test_params, test_data, num_boost_round=1, callbacks=[lgb.log_evaluation(period=0)])
        print("✓ GPU加速可用")
        return True
        
    except Exception as e:
        print(f"❌ GPU训练测试失败: {e}")
        print("请检查LightGBM GPU支持是否正确安装")
        sys.exit(1)

GPU_AVAILABLE = check_gpu_availability()

bayes_opt_spec = find_spec("bayes_opt")
if bayes_opt_spec is not None:
    BayesianOptimization = import_module("bayes_opt").BayesianOptimization
    BAYESIAN_OPT_AVAILABLE = True
else:
    print("Note: bayesian-optimization is not installed, grid search will be used.")
    print("      You can use 'pip install bayesian-optimization' to install and enable Bayesian optimization.")
    BAYESIAN_OPT_AVAILABLE = False
    BayesianOptimization = None

np.random.seed(42)

print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - LightGBM Model")
print("=" * 80)

print("\nConfiguring parameters...")

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

print(f"Data time range: {start_date.date()} to {end_date.date()}")
print(f"Target variable: PM2.5 concentration")
print(f"Output directory: {output_dir}")
print(f"CPU core count: {CPU_COUNT}, parallel worker processes: {MAX_WORKERS}")
print("GPU acceleration: Enabled (Required)")
print("GPU device: RTX 5090 (32GB)")

def daterange(start, end):
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def build_file_path_dict(base_path, prefix):
    """预先构建文件路径字典，避免每次遍历文件系统"""
    file_dict = {}
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename.startswith(prefix) and filename.endswith('.csv'):
                try:
                    date_str = filename.replace(f"{prefix}_", "").replace(".csv", "")
                    if len(date_str) == 8 and date_str.isdigit():
                        file_dict[date_str] = os.path.join(root, filename)
                except Exception:
                    continue
    return file_dict

def read_pollution_day(args):
    """读取单日污染数据，使用字典查找文件路径（O(1)时间复杂度）"""
    date, file_dict_all, file_dict_extra, pollutants_list = args
    date_str = date.strftime('%Y%m%d')
    
    # 使用字典查找，O(1)时间复杂度
    all_file = file_dict_all.get(date_str)
    extra_file = file_dict_extra.get(date_str)
    
    if not all_file or not extra_file:
        return None
    
    try:
        # pandas 1.3.0+使用on_bad_lines，旧版本使用error_bad_lines=False
        try:
            df_all = pd.read_csv(all_file, encoding='utf-8', on_bad_lines='skip')
            df_extra = pd.read_csv(extra_file, encoding='utf-8', on_bad_lines='skip')
        except TypeError:
            # 兼容pandas < 1.3.0
            df_all = pd.read_csv(all_file, encoding='utf-8', error_bad_lines=False, warn_bad_lines=False)
            df_extra = pd.read_csv(extra_file, encoding='utf-8', error_bad_lines=False, warn_bad_lines=False)
        
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
    except Exception:
        return None

def read_all_pollution():
    print("\n加载污染数据...")
    file_dict_all = build_file_path_dict(pollution_all_path, 'beijing_all')
    file_dict_extra = build_file_path_dict(pollution_extra_path, 'beijing_extra')
    
    dates = list(daterange(start_date, end_date))
    pollution_dfs = []
    
    # 准备参数列表，将文件字典和污染物列表一起传递
    args_list = [(date, file_dict_all, file_dict_extra, pollutants) for date in dates]
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, args): args[0] for args in args_list}
        
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
        df_poll_all = pd.concat(pollution_dfs)
        df_poll_all.ffill(inplace=True)
        df_poll_all.fillna(df_poll_all.mean(), inplace=True)
        print(f"✓ 污染数据加载完成: {df_poll_all.shape}")
        return df_poll_all
    return pd.DataFrame()

def read_era5_month(args):
    """读取单月ERA5气象数据，使用多进程"""
    year, month, era5_path_val, era5_vars_list, beijing_lats_val, beijing_lons_val = args
    month_str = f"{year}{month:02d}"
    all_files = glob.glob(os.path.join(era5_path_val, "**", f"*{month_str}*.nc"), recursive=True)
    fallback_used = False
    if not all_files:
        all_files = glob.glob(os.path.join(era5_path_val, "**", "*.nc"), recursive=True)
        fallback_used = True
        if not all_files:
            return None
    
    monthly_datasets = []
    
    # 计算当月时间窗口
    start_day = 1
    end_day = calendar.monthrange(year, month)[1]
    month_start = pd.to_datetime(f"{year}-{month:02d}-{start_day:02d}")
    month_end = pd.to_datetime(f"{year}-{month:02d}-{end_day:02d}") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    for file_path in all_files:
        try:
            with Dataset(file_path, mode='r') as nc_file:
                available_vars = [v for v in era5_vars_list if v in nc_file.variables]
            if not available_vars:
                continue
            with xr.open_dataset(file_path, engine="netcdf4", decode_times=True) as ds:
                rename_map = {}
                # 兼容多种时间坐标命名
                for tkey in ("valid_time", "forecast_time", "verification_time", "time1", "time2"):
                    if tkey in ds.coords and "time" not in ds.coords:
                        rename_map[tkey] = "time"
                if "lat" in ds.coords and "latitude" not in ds.coords:
                    rename_map["lat"] = "latitude"
                if "lon" in ds.coords and "longitude" not in ds.coords:
                    rename_map["lon"] = "longitude"
                if rename_map:
                    ds = ds.rename(rename_map)

                # 尝试全面解码 CF，以确保 time/坐标可用
                try:
                    ds = xr.decode_cf(ds)
                except Exception:
                    pass

                drop_vars = []
                for extra_coord in ("expver", "surface"):
                    if extra_coord in ds:
                        drop_vars.append(extra_coord)
                if drop_vars:
                    ds = ds.drop_vars(drop_vars)

                if "number" in ds.dims:
                    ds = ds.mean(dim="number", skipna=True)

                ds_subset = ds[available_vars]
                if "time" not in ds_subset.coords:
                    continue
                ds_subset = ds_subset.sortby('time')

                # 若使用了回退（全量 *.nc），则需要在数据内部按月份筛选
                if fallback_used:
                    try:
                        ds_subset = ds_subset.sel(time=slice(month_start, month_end))
                    except Exception:
                        continue
                    if ds_subset.sizes.get('time', 0) == 0:
                        # 文件不含目标月份数据
                        continue

                if 'latitude' in ds_subset.coords and 'longitude' in ds_subset.coords:
                    lat_values = ds_subset['latitude']
                    if lat_values[0] > lat_values[-1]:
                        lat_slice = slice(beijing_lats_val.max(), beijing_lats_val.min())
                    else:
                        lat_slice = slice(beijing_lats_val.min(), beijing_lats_val.max())
                    ds_subset = ds_subset.sel(latitude=lat_slice, longitude=slice(beijing_lons_val.min(), beijing_lons_val.max()))
                    if 'latitude' in ds_subset.dims and 'longitude' in ds_subset.dims:
                        ds_subset = ds_subset.mean(dim=['latitude', 'longitude'], skipna=True)
                ds_daily = ds_subset.resample(time='1D').mean(keep_attrs=False)
                ds_daily = ds_daily.dropna('time', how='all')
                if ds_daily.sizes.get('time', 0) == 0:
                    # 可能时间窗口或重采样为空
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
    print("\n加载气象数据...")
    era5_dfs = []
    years = range(2015, 2025)
    months = range(1, 13)
    
    month_tasks = [(year, month, era5_path, era5_vars, beijing_lats, beijing_lons) 
                   for year in years for month in months 
                   if not (year == 2024 and month > 12)]
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_era5_month, task): task 
                  for task in month_tasks}
        
        successful_reads = 0
        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Loading meteorological data", unit="month"):
                result = future.result()
                if result is not None and not result.empty:
                    era5_dfs.append(result)
                    successful_reads += 1
        else:
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None and not result.empty:
                    era5_dfs.append(result)
    
    if era5_dfs:
        df_era5_all = pd.concat(era5_dfs, axis=0)
        df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep='first')]
        df_era5_all.sort_index(inplace=True)
        df_era5_all.ffill(inplace=True)
        df_era5_all.bfill(inplace=True)
        df_era5_all.fillna(df_era5_all.mean(), inplace=True)
        print(f"✓ 气象数据加载完成: {df_era5_all.shape}")
        return df_era5_all
    else:
        print("❌ 错误: 未能加载气象数据文件")
        return pd.DataFrame()

print("\n" + "=" * 80)
print("Step 1: 数据加载和预处理")
print("=" * 80)

df_era5 = read_all_era5()
df_pollution = read_all_pollution()

if df_pollution.empty or df_era5.empty:
    print("❌ 错误: 数据加载失败")
    import sys
    sys.exit(1)

df_pollution.index = pd.to_datetime(df_pollution.index)
df_era5.index = pd.to_datetime(df_era5.index)
df_combined = df_pollution.join(df_era5, how='inner')

if df_combined.empty:
    print("❌ 错误: 数据合并后为空")
    import sys
    sys.exit(1)
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
    # 使用兼容的方式获取日期特征
    # pandas 2.0+推荐使用day_of_year/day_of_week，但dayofyear/dayofweek仍然可用
    try:
        df_copy['day_of_year'] = df_copy.index.day_of_year
    except AttributeError:
        df_copy['day_of_year'] = df_copy.index.dayofyear
    
    try:
        df_copy['day_of_week'] = df_copy.index.day_of_week
    except AttributeError:
        df_copy['day_of_week'] = df_copy.index.dayofweek
    
    # isocalendar() 返回 ISO 日历元组
    iso_calendar = df_copy.index.isocalendar()
    df_copy['week_of_year'] = iso_calendar.week
    
    df_copy['season'] = df_copy['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )
    
    df_copy['is_heating_season'] = ((df_copy['month'] >= 11) | (df_copy['month'] <= 3)).astype(int)
    
    if 't2m' in df_copy and 'd2m' in df_copy:
        df_copy['temp_dewpoint_diff'] = df_copy['t2m'] - df_copy['d2m']
    
    if 'PM2.5' in df_copy:
        # 扩展 lag 特征
        df_copy['PM2.5_lag1'] = df_copy['PM2.5'].shift(1)
        df_copy['PM2.5_lag2'] = df_copy['PM2.5'].shift(2)  # 新增
        df_copy['PM2.5_lag3'] = df_copy['PM2.5'].shift(3)
        df_copy['PM2.5_lag5'] = df_copy['PM2.5'].shift(5)  # 新增
        df_copy['PM2.5_lag7'] = df_copy['PM2.5'].shift(7)
        df_copy['PM2.5_lag14'] = df_copy['PM2.5'].shift(14)  # 新增，周周期

        # 移动平均特征
        df_copy['PM2.5_ma3'] = df_copy['PM2.5'].rolling(window=3, min_periods=1).mean()
        df_copy['PM2.5_ma7'] = df_copy['PM2.5'].rolling(window=7, min_periods=1).mean()
        df_copy['PM2.5_ma30'] = df_copy['PM2.5'].rolling(window=30, min_periods=1).mean()

        # 新增滚动统计特征
        df_copy['PM2.5_rolling_std7'] = df_copy['PM2.5'].rolling(window=7, min_periods=1).std()
        df_copy['PM2.5_rolling_max7'] = df_copy['PM2.5'].rolling(window=7, min_periods=1).max()
    
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

    # 添加污染物交叉项特征（化学反应相关）
    if 'PM2.5' in df_copy and 'NO2' in df_copy:
        df_copy['PM25_NO2_interaction'] = df_copy['PM2.5'] * df_copy['NO2']

    if 'PM2.5' in df_copy and 'SO2' in df_copy:
        df_copy['PM25_SO2_interaction'] = df_copy['PM2.5'] * df_copy['SO2']

    if 'PM2.5' in df_copy and 'O3' in df_copy:
        df_copy['PM25_O3_interaction'] = df_copy['PM2.5'] * df_copy['O3']

    # 添加气象特征组合
    if 'relative_humidity' in df_copy and 'wind_speed_10m' in df_copy:
        df_copy['humidity_wind_interaction'] = df_copy['relative_humidity'] * df_copy['wind_speed_10m']

    if 'blh' in df_copy and 'wind_speed_10m' in df_copy:
        df_copy['blh_wind_interaction'] = df_copy['blh'] * df_copy['wind_speed_10m']

    if 'temp_dewpoint_diff' in df_copy and 'blh' in df_copy:
        df_copy['temp_diff_blh_interaction'] = df_copy['temp_dewpoint_diff'] * df_copy['blh']

    return df_copy

df_combined = create_features(df_combined)

df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
df_combined.dropna(inplace=True)

print("\n" + "=" * 80)
print("Step 2: 特征选择和数据准备")
print("=" * 80)

target = 'PM2.5'
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']
numeric_features = [col for col in df_combined.select_dtypes(include=[np.number]).columns
                    if col not in exclude_cols]

X = df_combined[numeric_features].copy()
y = df_combined[target].copy()

if len(X) == 0 or len(y) == 0:
    print("❌ 错误: 无可用数据")
    import sys
    sys.exit(1)

# 对数变换目标变量以处理长尾分布
print("\n应用对数变换到目标变量...")
y_log = np.log1p(y)  # 使用log1p避免log(0)问题
print(f"原始目标变量范围: {y.min():.2f} - {y.max():.2f}")
print(f"对数变换后范围: {y_log.min():.2f} - {y_log.max():.2f}")

print("\n" + "=" * 80)
print("Step 3: 数据集划分")
print("=" * 80)

n_samples = len(X)
train_size = int(n_samples * 0.70)
val_size = int(n_samples * 0.15)

X_train = X.iloc[:train_size]
X_val = X.iloc[train_size:train_size + val_size]
X_test = X.iloc[train_size + val_size:]

# 使用原始目标变量用于评估和绘图，使用对数变换后的用于训练
y_train_orig = y.iloc[:train_size]
y_val_orig = y.iloc[train_size:train_size + val_size]
y_test_orig = y.iloc[train_size + val_size:]

# 使用对数变换后的目标变量进行训练
y_train_log = y_log.iloc[:train_size]
y_val_log = y_log.iloc[train_size:train_size + val_size]
y_test_log = y_log.iloc[train_size + val_size:]

X_train_gpu = X_train.values.astype(np.float32)
X_val_gpu = X_val.values.astype(np.float32)
y_train_gpu = y_train_log.values.astype(np.float32)  # 使用对数变换后的
y_val_gpu = y_val_log.values.astype(np.float32)     # 使用对数变换后的

gpu_dataset_params = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'max_bin': 255,
    'feature_pre_filter': False,
    'force_col_wise': True,
    'gpu_page_size': 2048
}

lgb_train = lgb.Dataset(X_train_gpu, label=y_train_gpu, feature_name=list(X_train.columns), params=gpu_dataset_params)
lgb_val = lgb.Dataset(X_val_gpu, label=y_val_gpu, reference=lgb_train, feature_name=list(X_val.columns), params=gpu_dataset_params)

# 保存原始目标变量用于后续评估
y_train_orig_values = y_train_orig.values
y_val_orig_values = y_val_orig.values
y_test_orig_values = y_test_orig.values

print("\n" + "=" * 80)
print("Step 4: 训练LightGBM基础模型")
print("=" * 80)

params_basic = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'num_gpu': 1,
    'max_bin': 255,
    'gpu_use_dp': False,
    'tree_learner': 'data',
    'force_col_wise': True,
    'gpu_page_size': 2048,
    'num_threads': 0
}
evals_result_basic = {}
model_basic = lgb.train(
    params_basic,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100),
        lgb.record_evaluation(evals_result_basic)
    ]
)

print(f"✓ 基础模型训练完成 (最佳迭代: {model_basic.best_iteration})")

# 预测时也需要使用numpy数组格式（预测结果需要反变换）
y_train_pred_basic_log = model_basic.predict(X_train_gpu, num_iteration=model_basic.best_iteration)
y_val_pred_basic_log = model_basic.predict(X_val_gpu, num_iteration=model_basic.best_iteration)
y_test_pred_basic_log = model_basic.predict(X_test.values.astype(np.float32), num_iteration=model_basic.best_iteration)

# 反变换回原始尺度
y_train_pred_basic = np.expm1(y_train_pred_basic_log)
y_val_pred_basic = np.expm1(y_val_pred_basic_log)
y_test_pred_basic = np.expm1(y_test_pred_basic_log)

def evaluate_model(y_true, y_pred, dataset_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'Dataset': dataset_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

results_basic = []
results_basic.append(evaluate_model(y_train_orig_values, y_train_pred_basic, 'Train'))
results_basic.append(evaluate_model(y_val_orig_values, y_val_pred_basic, 'Validation'))
results_basic.append(evaluate_model(y_test_orig_values, y_test_pred_basic, 'Test'))

results_basic_df = pd.DataFrame(results_basic)

print("\n" + "=" * 80)
print("Step 5: 超参数优化")
print("=" * 80)

if BAYESIAN_OPT_AVAILABLE:
    
    def lgb_evaluate(num_leaves, max_depth, learning_rate, feature_fraction, 
                     bagging_fraction, min_child_samples):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': int(num_leaves),
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': 5,
            'min_child_samples': int(min_child_samples),
            'feature_pre_filter': False,
            'verbose': -1,
            'seed': 42
        }
        
        params.update({
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'num_gpu': 1,
            'max_bin': 255,
            'gpu_use_dp': False,
            'tree_learner': 'data',
            'force_col_wise': True,
            'gpu_page_size': 2048,
            'num_threads': 0
        })
        
        # 使用GPU优化的数据集
        lgb_train_temp = lgb.Dataset(
            X_train_gpu, 
            label=y_train_gpu, 
            feature_name=list(X_train.columns),
            params={**gpu_dataset_params, 'feature_pre_filter': False}
        )
        lgb_val_temp = lgb.Dataset(
            X_val_gpu, 
            label=y_val_gpu, 
            reference=lgb_train_temp, 
            feature_name=list(X_val.columns),
            params=gpu_dataset_params
        )
        
        model = lgb.train(
            params,
            lgb_train_temp,
            num_boost_round=500,
            valid_sets=[lgb_val_temp],
            callbacks=[lgb.early_stopping(stopping_rounds=30)]
        )
        
        y_pred_log = model.predict(X_val_gpu, num_iteration=model.best_iteration)
        y_pred = np.expm1(y_pred_log)  # 反变换
        rmse = np.sqrt(mean_squared_error(y_val_orig_values, y_pred))
        
        return -rmse
    
    pbounds = {
        'num_leaves': (20, 100),
        'max_depth': (3, 12),
        'learning_rate': (0.01, 0.1),
        'feature_fraction': (0.5, 1.0),
        'bagging_fraction': (0.5, 1.0),
        'min_child_samples': (10, 50)
    }
    
    optimizer = BayesianOptimization(
        f=lgb_evaluate,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    optimizer.maximize(init_points=5, n_iter=15)
    
    best_params = optimizer.max['params']
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_samples'] = int(best_params['min_child_samples'])
    
else:
    
    param_grid = {
        'num_leaves': [31, 50, 70],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.03, 0.05, 0.07],
        'feature_fraction': [0.7, 0.8, 0.9],
    }
    
    from itertools import product
    
    def evaluate_params(combo):
        params_test = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': combo[0],
            'max_depth': combo[1],
            'learning_rate': combo[2],
            'feature_fraction': combo[3],
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'feature_pre_filter': False,
            'verbose': -1,
            'seed': 42
        }
        
        params_test.update({
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'num_gpu': 1,
            'max_bin': 255,
            'gpu_use_dp': False,
            'tree_learner': 'data',
            'force_col_wise': True,
            'gpu_page_size': 2048,
            'num_threads': 0
        })
        
        # 使用GPU优化的数据集
        lgb_train_temp = lgb.Dataset(
            X_train_gpu, 
            label=y_train_gpu, 
            feature_name=list(X_train.columns),
            params={**gpu_dataset_params, 'feature_pre_filter': False}
        )
        lgb_val_temp = lgb.Dataset(
            X_val_gpu, 
            label=y_val_gpu, 
            reference=lgb_train_temp,
            feature_name=list(X_val.columns),
            params=gpu_dataset_params
        )
        
        model_temp = lgb.train(
            params_test,
            lgb_train_temp,
            num_boost_round=500,
            valid_sets=[lgb_val_temp],
            callbacks=[lgb.early_stopping(stopping_rounds=30)]
        )
        
        y_pred_temp_log = model_temp.predict(X_val_gpu, num_iteration=model_temp.best_iteration)
        y_pred_temp = np.expm1(y_pred_temp_log)  # 反变换
        rmse_temp = np.sqrt(mean_squared_error(y_val_orig_values, y_pred_temp))
        
        return combo, rmse_temp
    
    all_combos = list(product(*param_grid.values()))
    
    best_rmse = float('inf')
    best_params = {}
    
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, 4)) as executor:
        futures = {executor.submit(evaluate_params, combo): combo for combo in all_combos}
        
        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Grid search", unit=" combination"):
                combo, rmse_temp = future.result()
                if rmse_temp < best_rmse:
                    best_rmse = rmse_temp
                    best_params = {
                        'num_leaves': combo[0],
                        'max_depth': combo[1],
                        'learning_rate': combo[2],
                        'feature_fraction': combo[3]
                    }
        else:
            for i, future in enumerate(as_completed(futures), 1):
                combo, rmse_temp = future.result()
                if rmse_temp < best_rmse:
                    best_rmse = rmse_temp
                    best_params = {
                        'num_leaves': combo[0],
                        'max_depth': combo[1],
                        'learning_rate': combo[2],
                        'feature_fraction': combo[3]
                    }

print("\n" + "=" * 80)
print("Step 6: 使用最佳参数训练优化模型")
print("=" * 80)

params_optimized = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': max(best_params.get('num_leaves', 50), 63),
    'max_depth': best_params.get('max_depth', 7),
    'learning_rate': best_params.get('learning_rate', 0.05),
    'feature_fraction': best_params.get('feature_fraction', 0.8),
    'bagging_fraction': best_params.get('bagging_fraction', 0.8),
    'bagging_freq': 5,
    'min_child_samples': best_params.get('min_child_samples', 20),
    'feature_pre_filter': False,
    'verbose': -1,
    'seed': 42,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'num_gpu': 1,
    'max_bin': 255,
    'gpu_use_dp': False,
    'tree_learner': 'data',
    'force_col_wise': True,
    'gpu_page_size': 2048,
    'num_threads': 0
}
evals_result_opt = {}
model_optimized = lgb.train(
    params_optimized,
    lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100),
        lgb.record_evaluation(evals_result_opt)
    ]
)

print(f"✓ 优化模型训练完成 (最佳迭代: {model_optimized.best_iteration})")

# 预测时也需要使用numpy数组格式（预测结果需要反变换）
y_train_pred_opt_log = model_optimized.predict(X_train_gpu, num_iteration=model_optimized.best_iteration)
y_val_pred_opt_log = model_optimized.predict(X_val_gpu, num_iteration=model_optimized.best_iteration)
y_test_pred_opt_log = model_optimized.predict(X_test.values.astype(np.float32), num_iteration=model_optimized.best_iteration)

# 反变换回原始尺度
y_train_pred_opt = np.expm1(y_train_pred_opt_log)
y_val_pred_opt = np.expm1(y_val_pred_opt_log)
y_test_pred_opt = np.expm1(y_test_pred_opt_log)

results_opt = []
results_opt.append(evaluate_model(y_train_orig_values, y_train_pred_opt, 'Train'))
results_opt.append(evaluate_model(y_val_orig_values, y_val_pred_opt, 'Validation'))
results_opt.append(evaluate_model(y_test_orig_values, y_test_pred_opt, 'Test'))

results_opt_df = pd.DataFrame(results_opt)

print("\n" + "=" * 80)
print("Step 7: 模型性能比较")
print("=" * 80)

results_basic_df['Model'] = 'LightGBM_Basic'
results_opt_df['Model'] = 'LightGBM_Optimized'
all_results = pd.concat([results_basic_df, results_opt_df])
all_results = all_results[['Model', 'Dataset', 'R²', 'RMSE', 'MAE', 'MAPE']]

test_results = all_results[all_results['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\n测试集性能:")
print(test_results.to_string(index=False))

basic_test_r2 = results_basic_df[results_basic_df['Dataset'] == 'Test']['R²'].values[0]
opt_test_r2 = results_opt_df[results_opt_df['Dataset'] == 'Test']['R²'].values[0]
basic_test_rmse = results_basic_df[results_basic_df['Dataset'] == 'Test']['RMSE'].values[0]
opt_test_rmse = results_opt_df[results_opt_df['Dataset'] == 'Test']['RMSE'].values[0]

print("\n对数变换优化效果:")
print("  使用对数变换处理长尾分布，提升对峰值预测能力")

r2_improvement = (opt_test_r2 - basic_test_r2) / basic_test_r2 * 100
rmse_improvement = (basic_test_rmse - opt_test_rmse) / basic_test_rmse * 100

print(f"\nOptimization Effect:")
print(f"  R² improvement: {r2_improvement:.2f}%")
print(f"  RMSE reduction: {rmse_improvement:.2f}%")

print("\n" + "=" * 80)
print("Step 8: 特征重要性分析")
print("=" * 80)

feature_importance = pd.DataFrame({
    'Feature': model_optimized.feature_name(),
    'Importance_Split': model_optimized.feature_importance(importance_type='split'),
    'Importance_Gain': model_optimized.feature_importance(importance_type='gain')
})

feature_importance['Importance_Split_Norm'] = (feature_importance['Importance_Split'] / 
                                                feature_importance['Importance_Split'].sum() * 100)
feature_importance['Importance_Gain_Norm'] = (feature_importance['Importance_Gain'] / 
                                               feature_importance['Importance_Gain'].sum() * 100)

feature_importance = feature_importance.sort_values('Importance_Gain', ascending=False)

print("\n" + "=" * 80)
print("Step 9: Export CSV Files for Plotting (No figures will be generated)")
print("=" * 80)

# 1) Training curve data（替代训练曲线图片）
def _export_training_curve(evals_result: dict, model_slug: str, best_iteration: int):
    train_rmse = evals_result.get("train", {}).get("rmse", [])
    valid_rmse = evals_result.get("valid", {}).get("rmse", [])
    n = max(len(train_rmse), len(valid_rmse))
    iters = np.arange(1, n + 1)
    df_curve = pd.DataFrame(
        {
            "iteration": iters,
            "train_rmse": pd.Series(train_rmse, dtype="float64"),
            "valid_rmse": pd.Series(valid_rmse, dtype="float64"),
            "best_iteration": best_iteration,
        }
    )
    save_csv(df_curve, output_dir / f"plot_training_curve__{model_slug}.csv")


_export_training_curve(evals_result_basic, "lightgbm_basic", int(model_basic.best_iteration))
_export_training_curve(evals_result_opt, "lightgbm_optimized", int(model_optimized.best_iteration))

# 2) Metrics（按模型/按数据集拆分）
save_csv(all_results, output_dir / "metrics__all_models_train_validation_test.csv")
for model in all_results["Model"].unique():
    model_slug = slugify_model_name(model)
    save_csv(
        all_results[all_results["Model"] == model],
        output_dir / f"metrics__{model_slug}__train_validation_test.csv",
    )
    for ds_name in all_results["Dataset"].unique():
        part = all_results[(all_results["Model"] == model) & (all_results["Dataset"] == ds_name)]
        if not part.empty:
            save_csv(part, output_dir / f"metrics__{model_slug}__{ds_name.lower()}.csv")

test_ranking = (
    all_results[all_results["Dataset"] == "Test"]
    .sort_values("R²", ascending=False)
    .reset_index(drop=True)
)
save_csv(test_ranking, output_dir / "plot_metrics_ranking__test_only.csv")

# 3) Scatter / Residuals / Error distribution（按模型 + 数据集拆分）
models_data = [
    ("LightGBM_Basic", y_train_pred_basic, y_train_orig_values, "Train"),
    ("LightGBM_Basic", y_val_pred_basic, y_val_orig_values, "Validation"),
    ("LightGBM_Basic", y_test_pred_basic, y_test_orig_values, "Test"),
    ("LightGBM_Optimized", y_train_pred_opt, y_train_orig_values, "Train"),
    ("LightGBM_Optimized", y_val_pred_opt, y_val_orig_values, "Validation"),
    ("LightGBM_Optimized", y_test_pred_opt, y_test_orig_values, "Test"),
]

for model_name, y_pred, y_true, dataset in models_data:
    model_slug = slugify_model_name(model_name)
    ds_slug = dataset.lower()

    if dataset == "Train":
        date_index = y_train_orig.index
    elif dataset == "Validation":
        date_index = y_val_orig.index
    else:
        date_index = y_test_orig.index

    scatter_df = pd.DataFrame(
        {
            "Date": date_index,
            "Actual_PM25": y_true,
            "Predicted_PM25": y_pred,
        }
    )
    save_csv(scatter_df, output_dir / f"plot_scatter__{model_slug}__{ds_slug}.csv")

    residuals = y_true - y_pred
    residuals_df = scatter_df.assign(Residual=residuals)
    save_csv(residuals_df, output_dir / f"plot_residuals__{model_slug}__{ds_slug}.csv")

    error_df = scatter_df.assign(Error=residuals)
    save_csv(error_df, output_dir / f"plot_error_distribution__{model_slug}__{ds_slug}.csv")

# 4) Time series（按 RF 命名规则：simple sampled + lastyear sampled）
step = 4

def _export_ts_simple_sampled(model_slug: str, y_pred_test: np.ndarray):
    ts_df = (
        pd.DataFrame({"time": y_test_orig.index, "y_true": y_test_orig_values, "y_pred": y_pred_test})
        .sort_values("time")
        .reset_index(drop=True)
    )
    ts_df_sampled = ts_df.iloc[::step].copy()
    save_csv(ts_df_sampled, output_dir / f"plot_ts_simple_sampled__{model_slug}.csv")


_export_ts_simple_sampled("lightgbm_basic", y_test_pred_basic)
_export_ts_simple_sampled("lightgbm_optimized", y_test_pred_opt)

plot_df = pd.DataFrame({"time": y_test_orig.index, "y_true": y_test_orig_values}).sort_values("time").reset_index(drop=True)
plot_range = min(365, len(plot_df))
plot_df_subset = plot_df.iloc[-plot_range:].copy()
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

def _export_ts_lastyear_sampled(model_slug: str, y_pred_test: np.ndarray):
    y_pred_series = pd.Series(y_pred_test, index=y_test_orig.index).sort_index()
    pred_df = pd.DataFrame({"time": y_pred_series.index, "y_pred": y_pred_series.values}).sort_values("time").reset_index(drop=True)
    pred_subset = pred_df.iloc[-plot_range:].copy()
    pred_sampled = pred_subset.iloc[::step].copy().reset_index(drop=True)
    ts_model = ts_common.assign(y_pred=pred_sampled["y_pred"].values)
    save_csv(ts_model, output_dir / f"plot_ts_lastyear_sampled__{model_slug}.csv")


_export_ts_lastyear_sampled("lightgbm_basic", y_test_pred_basic)
_export_ts_lastyear_sampled("lightgbm_optimized", y_test_pred_opt)

# 5) Feature importance（按图类型导出 CSV）
fi_gain = (
    feature_importance[["Feature", "Importance_Gain", "Importance_Gain_Norm"]]
    .rename(columns={"Importance_Gain": "Importance", "Importance_Gain_Norm": "Importance_Norm"})
    .reset_index(drop=True)
)
save_csv(fi_gain, output_dir / "plot_feature_importance__lightgbm_optimized.csv")

fi_split = (
    feature_importance[["Feature", "Importance_Split", "Importance_Split_Norm"]]
    .rename(columns={"Importance_Split": "Importance", "Importance_Split_Norm": "Importance_Norm"})
    .reset_index(drop=True)
)
save_csv(fi_split, output_dir / "plot_feature_importance__lightgbm_optimized__split.csv")

top_n = min(20, len(fi_gain))
save_csv(fi_gain.head(top_n), output_dir / f"plot_feature_importance_top{top_n}__lightgbm_optimized.csv")
save_csv(fi_split.head(top_n), output_dir / f"plot_feature_importance_top{top_n}__lightgbm_optimized__split.csv")

print("\n" + "=" * 80)
print("Step 10: 保存结果")
print("=" * 80)

save_csv(all_results, output_dir / "lgbm_model_performance.csv")
save_csv(feature_importance, output_dir / "lgbm_feature_importance__raw.csv")
save_csv(pd.DataFrame([params_basic]), output_dir / "lgbm_parameters__lightgbm_basic.csv")
save_csv(pd.DataFrame([params_optimized]), output_dir / "lgbm_parameters__lightgbm_optimized.csv")

pred_all = pd.DataFrame(
    {
        "Date": y_test_orig.index,
        "Actual_PM25": y_test_orig_values,
        "Predicted_LightGBM_Basic": y_test_pred_basic,
        "Predicted_LightGBM_Optimized": y_test_pred_opt,
        "Error_LightGBM_Basic": y_test_orig_values - y_test_pred_basic,
        "Error_LightGBM_Optimized": y_test_orig_values - y_test_pred_opt,
    }
)
save_csv(pred_all, output_dir / "lgbm_predictions_all_models.csv")

pred_basic = pd.DataFrame(
    {
        "Date": y_test_orig.index,
        "Actual_PM25": y_test_orig_values,
        "Predicted_PM25": y_test_pred_basic,
        "Error": y_test_orig_values - y_test_pred_basic,
    }
)
save_csv(pred_basic, output_dir / "lgbm_predictions__lightgbm_basic.csv")

pred_opt = pd.DataFrame(
    {
        "Date": y_test_orig.index,
        "Actual_PM25": y_test_orig_values,
        "Predicted_PM25": y_test_pred_opt,
        "Error": y_test_orig_values - y_test_pred_opt,
    }
)
save_csv(pred_opt, output_dir / "lgbm_predictions__lightgbm_optimized.csv")

print("\n" + "=" * 80)
print("分析完成!")
print("=" * 80)

best_model = test_results.iloc[0]
print(f"\n最佳模型: {best_model['Model']}")
print(f"  R²: {best_model['R²']:.4f}")
print(f"  RMSE: {best_model['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model['MAPE']:.2f}%")

print("\n前5个最重要特征:")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance_Gain_Norm']:.2f}%")