import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import pickle
from pathlib import Path
import multiprocessing

warnings.filterwarnings('ignore')

# Get CPU core count
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

# Try to import tqdm progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: tqdm not installed, progress display will use simplified version.")

# Time series libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from scipy import stats

# auto_arima
try:
    import pmdarima as pm
    AUTO_ARIMA_AVAILABLE = True
except ImportError:
    print("Warning: pmdarima not installed, auto_arima optimization will be skipped.")
    print("      Use 'pip install pmdarima' to install.")
    AUTO_ARIMA_AVAILABLE = False

# Evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Set English fonts
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# Set random seed
np.random.seed(42)

print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - IMPROVED ARIMA Model")
print("=" * 80)

# ============================== Part 1: Configuration and Path Setup ==============================
print("\nConfiguring parameters...")

# Data paths
pollution_all_path = '/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)'
pollution_extra_path = '/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)'

# Output path
output_dir = Path('./improved_output')
output_dir.mkdir(exist_ok=True)

# Model save path
model_dir = Path('./improved_models')
model_dir.mkdir(exist_ok=True)

# Date range
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

print(f"Data time range: {start_date.date()} to {end_date.date()}")
print(f"Target variable: PM2.5 concentration (time series)")
print(f"Output directory: {output_dir}")
print(f"Model save directory: {model_dir}")
print(f"CPU cores: {CPU_COUNT}, parallel workers: {MAX_WORKERS}")

# ============================== Part 2: Improved Data Loading Functions ==============================
def daterange(start, end):
    """Generate date sequence"""
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def build_file_path_dict(base_path, prefix):
    """
    预先构建文件路径字典，实现O(1)查找
    返回字典：{date_str: file_path}
    """
    print(f"  正在扫描目录构建文件路径字典: {base_path}")
    file_dict = {}
    filename_prefix = f"{prefix}_"
    
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename.startswith(filename_prefix) and filename.endswith('.csv'):
                # 提取日期字符串 (格式: prefix_YYYYMMDD.csv)
                date_str = filename[len(filename_prefix):-4]  # 去掉前缀和后缀
                if len(date_str) == 8 and date_str.isdigit():  # 验证日期格式
                    full_path = os.path.join(root, filename)
                    file_dict[date_str] = full_path
    
    print(f"  找到 {len(file_dict)} 个文件")
    return file_dict

def read_pollution_day(args):
    """
    读取单日污染数据（多进程版本）
    args: (date, file_path_dict) 元组
    """
    date, file_path_dict = args
    date_str = date.strftime('%Y%m%d')
    
    # 使用字典O(1)查找文件路径
    all_file = file_path_dict.get(date_str)
    
    if not all_file:
        return None
    
    try:
        df_all = pd.read_csv(all_file, encoding='utf-8', on_bad_lines='skip')
        
        # Filter out 24-hour average and AQI
        df_all = df_all[~df_all['type'].str.contains('_24h|AQI', na=False)]
        
        # Keep only PM2.5
        df_pm25 = df_all[df_all['type'] == 'PM2.5']
        
        if df_pm25.empty:
            return None
        
        # Convert to long format
        df_pm25 = df_pm25.melt(id_vars=['date', 'hour', 'type'], 
                                var_name='station', value_name='value')
        df_pm25['value'] = pd.to_numeric(df_pm25['value'], errors='coerce')
        
        # Remove negative values
        df_pm25 = df_pm25[df_pm25['value'] >= 0]
        
        # Aggregate by date (average all stations)
        daily_mean = df_pm25.groupby('date')['value'].mean()
        
        if len(daily_mean) == 0:
            return None
        
        # Convert index to datetime format
        daily_mean.index = pd.to_datetime(daily_mean.index, format='%Y%m%d', errors='coerce')
        
        return daily_mean
    except Exception as e:
        return None

def read_all_pollution():
    """使用多进程并行读取所有污染数据"""
    print("\nLoading pollution data...")
    print(f"使用 {MAX_WORKERS} 个进程并行读取")
    
    # 预先构建文件路径字典（O(1)查找）
    print("正在构建文件路径字典...")
    file_path_dict = build_file_path_dict(pollution_all_path, 'beijing_all')
    
    if not file_path_dict:
        print("⚠️ 警告: 未找到任何数据文件！")
        return pd.Series()
    
    dates = list(daterange(start_date, end_date))
    pollution_series = []
    
    # 准备参数：将日期和文件路径字典打包
    task_args = [(date, file_path_dict) for date in dates]
    
    # 使用多进程池
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, args): args[0] for args in task_args}
        
        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Loading pollution data", unit="day"):
                result = future.result()
                if result is not None:
                    pollution_series.append(result)
        else:
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    pollution_series.append(result)
                if i % 500 == 0 or i == len(futures):
                    print(f"  已处理 {i}/{len(futures)} 天 ({i/len(futures)*100:.1f}%)")
    
    if pollution_series:
        print(f"  Successfully read {len(pollution_series)}/{len(dates)} days of data")
        print("  Merging data...")
        pm25_series = pd.concat(pollution_series)
        pm25_series = pm25_series.sort_index()
        
        # Remove duplicates (keep first)
        pm25_series = pm25_series[~pm25_series.index.duplicated(keep='first')]
        
        # Fill missing dates
        full_index = pd.date_range(start=pm25_series.index.min(), 
                                    end=pm25_series.index.max(), 
                                    freq='D')
        pm25_series = pm25_series.reindex(full_index)
        
        # 改进的缺失值处理
        print("  Applying improved missing value handling...")
        pm25_series = improved_data_preprocessing(pm25_series)
        
        print(f"Pollution data loading complete, shape: {pm25_series.shape}")
        return pm25_series
    return pd.Series()

def improved_data_preprocessing(series):
    """改进的数据预处理方法"""
    print("  应用改进的数据预处理...")
    
    # 1. 首先使用时间序列插值
    series_interpolated = series.interpolate(method='time', limit_direction='both')
    
    # 2. 检测并温和处理异常值
    Q1 = series_interpolated.quantile(0.25)
    Q3 = series_interpolated.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # 使用3倍IQR，更宽松
    upper_bound = Q3 + 3 * IQR
    
    # 温和的异常值处理：缩尾而不是删除
    series_cleaned = series_interpolated.clip(lower=lower_bound, upper=upper_bound)
    
    # 3. 对剩余缺失值使用前向后向填充
    series_filled = series_cleaned.ffill().bfill()
    
    # 4. 如果还有缺失值，使用滚动均值填充
    if series_filled.isna().sum() > 0:
        rolling_mean = series_filled.rolling(window=7, min_periods=1).mean()
        series_filled = series_filled.fillna(rolling_mean)
    
    print(f"  缺失值处理: {series.isna().sum()} -> {series_filled.isna().sum()}")
    
    return series_filled

def create_time_features(series):
    """创建时间特征（用于分析）"""
    df = series.reset_index()
    df.columns = ['date', 'value']
    
    # 时间特征
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 季节性特征
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df.set_index('date')['value'], df.drop(['date', 'value'], axis=1)

# ============================== Part 3: Data Loading and Preprocessing ==============================
print("\n" + "=" * 80)
print("Step 1: Data Loading and Preprocessing")
print("=" * 80)

pm25_series = read_all_pollution()

# Check data loading
print("\nData loading check:")
print(f"  PM2.5 series shape: {pm25_series.shape}")

if pm25_series.empty:
    print("\n⚠️ Warning: Pollution data is empty! Please check data path and files.")
    import sys
    sys.exit(1)

print(f"  Time range: {pm25_series.index.min()} to {pm25_series.index.max()}")
print(f"  Number of data points: {len(pm25_series)}")

# 创建时间特征（用于分析）
pm25_series, time_features = create_time_features(pm25_series)

# Basic statistics
print(f"\nPM2.5 Statistics:")
print(f"  Mean: {pm25_series.mean():.2f} μg/m³")
print(f"  Std Dev: {pm25_series.std():.2f} μg/m³")
print(f"  Min: {pm25_series.min():.2f} μg/m³")
print(f"  Max: {pm25_series.max():.2f} μg/m³")
print(f"  Median: {pm25_series.median():.2f} μg/m³")
print(f"  Missing values: {pm25_series.isna().sum()}")

# ============================== Part 4: Enhanced Time Series Analysis ==============================
print("\n" + "=" * 80)
print("Step 2: Enhanced Time Series Stationarity Analysis")
print("=" * 80)

def enhanced_stationarity_test(series):
    """增强的平稳性测试"""
    print("Performing enhanced stationarity analysis...")
    
    # 原始序列ADF检验
    adf_original = adfuller(series.dropna())
    print(f"\nOriginal Series ADF Test:")
    print(f"  ADF Statistic: {adf_original[0]:.4f}")
    print(f"  p-value: {adf_original[1]:.4f}")
    
    # 一阶差分后检验
    diff_1 = series.diff().dropna()
    adf_diff1 = adfuller(diff_1)
    print(f"\nFirst Difference ADF Test:")
    print(f"  ADF Statistic: {adf_diff1[0]:.4f}")
    print(f"  p-value: {adf_diff1[1]:.4f}")
    
    # 季节性差分（周期7天）
    seasonal_diff = series.diff(7).dropna()
    adf_seasonal = adfuller(seasonal_diff)
    print(f"\nSeasonal Difference (7 days) ADF Test:")
    print(f"  ADF Statistic: {adf_seasonal[0]:.4f}")
    print(f"  p-value: {adf_seasonal[1]:.4f}")
    
    # 建议差分阶数
    if adf_original[1] < 0.05:
        print(f"\n✓ Original series is stationary")
        recommended_d = 0
    elif adf_diff1[1] < 0.05:
        print(f"\n✓ First difference makes series stationary")
        recommended_d = 1
    else:
        print(f"\n⚠ Series may need higher order differencing")
        recommended_d = 1  # 默认使用一阶差分
    
    return recommended_d

recommended_d = enhanced_stationarity_test(pm25_series)

# ============================== Part 5: Improved Dataset Splitting ==============================
print("\n" + "=" * 80)
print("Step 3: Improved Dataset Splitting")
print("=" * 80)

# 使用时间序列交叉验证确保训练集和测试集的时间连续性
n_samples = len(pm25_series)
train_size = int(n_samples * 0.70)
val_size = int(n_samples * 0.15)

y_train = pm25_series.iloc[:train_size]
y_val = pm25_series.iloc[train_size:train_size + val_size]
y_test = pm25_series.iloc[train_size + val_size:]

print(f"\nTraining set: {len(y_train)} samples ({len(y_train)/n_samples*100:.1f}%)")
print(f"  Time range: {y_train.index.min().date()} to {y_train.index.max().date()}")
print(f"  PM2.5: {y_train.mean():.2f} ± {y_train.std():.2f} μg/m³")

print(f"\nValidation set: {len(y_val)} samples ({len(y_val)/n_samples*100:.1f}%)")
print(f"  Time range: {y_val.index.min().date()} to {y_val.index.max().date()}")
print(f"  PM2.5: {y_val.mean():.2f} ± {y_val.std():.2f} μg/m³")

print(f"\nTest set: {len(y_test)} samples ({len(y_test)/n_samples*100:.1f}%)")
print(f"  Time range: {y_test.index.min().date()} to {y_test.index.max().date()}")
print(f"  PM2.5: {y_test.mean():.2f} ± {y_test.std():.2f} μg/m³")

# ============================== Part 6: Improved Basic ARIMA Model ==============================
print("\n" + "=" * 80)
print("Step 4: Improved Basic SARIMA Model Training")
print("=" * 80)

# 改进的基本模型参数 - 更简单更稳定
order_basic = (2, 1, 2)  # 降低复杂度
seasonal_order_basic = (1, 1, 1, 7)  # 使用周季节性而非月季节性

print(f"\nImproved basic model parameters:")
print(f"  ARIMA order (p,d,q): {order_basic}")
print(f"  Seasonal order (P,D,Q,s): {seasonal_order_basic}")

print("\nStarting improved basic model training...")
try:
    model_basic = SARIMAX(y_train, 
                          order=order_basic,
                          seasonal_order=seasonal_order_basic,
                          enforce_stationarity=False,
                          enforce_invertibility=False)

    fit_basic = model_basic.fit(disp=False, maxiter=200)

    print(f"\n✓ Improved basic model training complete")
    print(f"  AIC: {fit_basic.aic:.2f}")
    print(f"  BIC: {fit_basic.bic:.2f}")
    print(f"  Log-likelihood: {fit_basic.llf:.2f}")

    # Prediction
    print("\nMaking predictions...")
    y_train_pred_basic = fit_basic.fittedvalues
    y_val_pred_basic = fit_basic.forecast(steps=len(y_val))
    y_test_pred_basic = fit_basic.forecast(steps=len(y_val) + len(y_test)).iloc[len(y_val):]
    
    basic_model_success = True
    
except Exception as e:
    print(f"\n⚠ Basic model training failed: {e}")
    print("  Using fallback parameters...")
    
    # 备用参数
    order_basic = (1, 1, 1)
    seasonal_order_basic = (1, 1, 1, 7)
    
    model_basic = SARIMAX(y_train, 
                          order=order_basic,
                          seasonal_order=seasonal_order_basic,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
    
    fit_basic = model_basic.fit(disp=False, maxiter=200)
    y_train_pred_basic = fit_basic.fittedvalues
    y_val_pred_basic = fit_basic.forecast(steps=len(y_val))
    y_test_pred_basic = fit_basic.forecast(steps=len(y_val) + len(y_test)).iloc[len(y_val):]
    
    basic_model_success = True

# ============================== Part 7: Enhanced Hyperparameter Optimization ==============================
print("\n" + "=" * 80)
print("Step 5: Enhanced Hyperparameter Optimization")
print("=" * 80)

def improved_auto_arima_search(y_train):
    """改进的自动参数搜索"""
    if not AUTO_ARIMA_AVAILABLE:
        print("⚠ auto_arima not available, using improved manual parameters")
        return (2, 1, 2), (1, 1, 1, 7)
    
    print("\nUsing improved auto_arima with better parameter ranges...")
    print("This may take a few minutes...")
    
    try:
        # 尝试周季节性
        auto_model_weekly = pm.auto_arima(
            y_train,
            start_p=0, start_q=0,
            max_p=4, max_q=4,  # 限制最大阶数
            d=None,
            seasonal=True,
            m=7,  # 周季节性
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            D=None,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            n_jobs=-1,
            information_criterion='aic',
            with_intercept=True
        )
        
        print(f"\n✓ Weekly seasonal model found:")
        print(f"  Order: {auto_model_weekly.order}")
        print(f"  Seasonal order: {auto_model_weekly.seasonal_order}")
        print(f"  AIC: {auto_model_weekly.aic():.2f}")
        
        return auto_model_weekly.order, auto_model_weekly.seasonal_order
        
    except Exception as e:
        print(f"⚠ Weekly seasonal search failed: {e}")
        print("  Trying non-seasonal model...")
        
        try:
            auto_model_nonseasonal = pm.auto_arima(
                y_train,
                start_p=1, start_q=1,
                max_p=3, max_q=3,
                d=None,
                seasonal=False,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                n_jobs=-1
            )
            
            print(f"\n✓ Non-seasonal model found:")
            print(f"  Order: {auto_model_nonseasonal.order}")
            print(f"  AIC: {auto_model_nonseasonal.aic():.2f}")
            
            return auto_model_nonseasonal.order, (0, 0, 0, 0)
            
        except Exception as e2:
            print(f"⚠ Non-seasonal search also failed: {e2}")
            print("  Using fallback parameters...")
            return (1, 1, 1), (1, 1, 1, 7)

best_order, best_seasonal_order = improved_auto_arima_search(y_train)

print(f"\nFinal best parameters:")
print(f"  ARIMA order (p,d,q): {best_order}")
print(f"  Seasonal order (P,D,Q,s): {best_seasonal_order}")

# ============================== Part 8: Training Optimized Model ==============================
print("\n" + "=" * 80)
print("Step 6: Training Optimized Model with Best Parameters")
print("=" * 80)

print(f"\nOptimized model parameters:")
print(f"  ARIMA order (p,d,q): {best_order}")
print(f"  Seasonal order (P,D,Q,s): {best_seasonal_order}")

print("\nStarting optimized model training...")
try:
    model_optimized = SARIMAX(y_train,
                              order=best_order,
                              seasonal_order=best_seasonal_order,
                              enforce_stationarity=False,
                              enforce_invertibility=False)

    fit_optimized = model_optimized.fit(disp=False, maxiter=200)

    print(f"\n✓ Optimized model training complete")
    print(f"  AIC: {fit_optimized.aic:.2f}")
    print(f"  BIC: {fit_optimized.bic:.2f}")
    print(f"  Log-likelihood: {fit_optimized.llf:.2f}")

    # Prediction
    print("\nMaking predictions...")
    y_train_pred_opt = fit_optimized.fittedvalues
    y_val_pred_opt = fit_optimized.forecast(steps=len(y_val))
    y_test_pred_opt = fit_optimized.forecast(steps=len(y_val) + len(y_test)).iloc[len(y_val):]
    
    optimized_model_success = True
    
except Exception as e:
    print(f"⚠ Optimized model training failed: {e}")
    print("  Using basic model as fallback...")
    fit_optimized = fit_basic
    y_train_pred_opt = y_train_pred_basic
    y_val_pred_opt = y_val_pred_basic
    y_test_pred_opt = y_test_pred_basic
    optimized_model_success = False

# ============================== Part 9: Enhanced Model Evaluation ==============================
print("\n" + "=" * 80)
print("Step 7: Enhanced Model Evaluation")
print("=" * 80)

def enhanced_evaluate_model(y_true, y_pred, dataset_name):
    """增强的模型评估"""
    # Ensure index alignment
    y_true_aligned = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred_aligned = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    
    # 确保长度一致
    min_len = min(len(y_true_aligned), len(y_pred_aligned))
    y_true_aligned = y_true_aligned[:min_len]
    y_pred_aligned = y_pred_aligned[:min_len]
    
    r2 = r2_score(y_true_aligned, y_pred_aligned)
    rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
    mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
    mape = np.mean(np.abs((y_true_aligned - y_pred_aligned) / np.maximum(y_true_aligned, 1))) * 100  # 避免除零
    
    # 方向准确性
    direction_accuracy = np.mean(np.sign(np.diff(y_true_aligned)) == np.sign(np.diff(y_pred_aligned))) * 100
    
    return {
        'Dataset': dataset_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Direction_Accuracy': direction_accuracy
    }

# Basic model evaluation
if basic_model_success:
    results_basic = []
    results_basic.append(enhanced_evaluate_model(y_train, y_train_pred_basic, 'Train'))
    results_basic.append(enhanced_evaluate_model(y_val, y_val_pred_basic, 'Validation'))
    results_basic.append(enhanced_evaluate_model(y_test, y_test_pred_basic, 'Test'))
    results_basic_df = pd.DataFrame(results_basic)
else:
    results_basic_df = pd.DataFrame()

# Optimized model evaluation
results_opt = []
results_opt.append(enhanced_evaluate_model(y_train, y_train_pred_opt, 'Train'))
results_opt.append(enhanced_evaluate_model(y_val, y_val_pred_opt, 'Validation'))
results_opt.append(enhanced_evaluate_model(y_test, y_test_pred_opt, 'Test'))
results_opt_df = pd.DataFrame(results_opt)

print("\nImproved basic model performance:")
print(results_basic_df.to_string(index=False, float_format='%.4f'))

print("\nOptimized model performance:")
print(results_opt_df.to_string(index=False, float_format='%.4f'))

# ============================== Part 10: Enhanced Model Comparison ==============================
print("\n" + "=" * 80)
print("Step 8: Enhanced Model Performance Comparison")
print("=" * 80)

# Merge results
results_basic_df['Model'] = 'ARIMA_Basic'
results_opt_df['Model'] = 'ARIMA_Optimized'
all_results = pd.concat([results_basic_df, results_opt_df])

# Rearrange column order
all_results = all_results[['Model', 'Dataset', 'R²', 'RMSE', 'MAE', 'MAPE', 'Direction_Accuracy']]

print("\nAll models performance comparison:")
print(all_results.to_string(index=False, float_format='%.4f'))

# Test set performance comparison
test_results = all_results[all_results['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\nTest set performance ranking:")
print(test_results.to_string(index=False, float_format='%.4f'))

# Performance improvement
if basic_model_success:
    basic_test_r2 = results_basic_df[results_basic_df['Dataset'] == 'Test']['R²'].values[0]
    opt_test_r2 = results_opt_df[results_opt_df['Dataset'] == 'Test']['R²'].values[0]
    basic_test_rmse = results_basic_df[results_basic_df['Dataset'] == 'Test']['RMSE'].values[0]
    opt_test_rmse = results_opt_df[results_opt_df['Dataset'] == 'Test']['RMSE'].values[0]

    if basic_test_r2 != 0:
        r2_improvement = (opt_test_r2 - basic_test_r2) / abs(basic_test_r2) * 100
    else:
        r2_improvement = 0

    if basic_test_rmse != 0:
        rmse_improvement = (basic_test_rmse - opt_test_rmse) / basic_test_rmse * 100
    else:
        rmse_improvement = 0

    print(f"\nOptimization effect:")
    print(f"  R² change: {r2_improvement:+.2f}%")
    print(f"  RMSE change: {rmse_improvement:+.2f}%")

# ============================== Part 11: Enhanced Residual Diagnostics ==============================
print("\n" + "=" * 80)
print("Step 9: Enhanced Residual Diagnostics")
print("=" * 80)

def enhanced_residual_analysis(residuals, model_name):
    """增强的残差分析"""
    print(f"\n{model_name} Residual Analysis:")
    print(f"  Mean: {residuals.mean():.6f}")
    print(f"  Std Dev: {residuals.std():.6f}")
    print(f"  Skewness: {stats.skew(residuals.dropna()):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(residuals.dropna()):.4f}")
    
    # Ljung-Box test at multiple lags
    lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    print(f"  Ljung-Box p-values:")
    for lag in [10, 20, 30]:
        p_val = lb_test.loc[lb_test.index == lag, 'lb_pvalue'].values[0]
        print(f"    Lag {lag}: {p_val:.4f}")
    
    # Jarque-Bera normality test
    jb_test = stats.jarque_bera(residuals.dropna())
    print(f"  Jarque-Bera test:")
    print(f"    Statistic: {jb_test[0]:.4f}")
    print(f"    p-value: {jb_test[1]:.4f}")
    
    # Normality assessment
    if jb_test[1] > 0.05:
        print("  ✓ Residuals follow normal distribution")
    else:
        print("  ⚠ Residuals do not follow normal distribution")
    
    # White noise assessment
    if all(lb_test['lb_pvalue'] > 0.05):
        print("  ✓ No significant residual autocorrelation (white noise)")
    else:
        print("  ⚠ Residuals have autocorrelation")

# Basic model residuals
if basic_model_success:
    residuals_basic = fit_basic.resid
    enhanced_residual_analysis(residuals_basic, "Basic Model")

# Optimized model residuals
residuals_opt = fit_optimized.resid
enhanced_residual_analysis(residuals_opt, "Optimized Model")

# ============================== Part 12: Enhanced Visualization ==============================
print("\n" + "=" * 80)
print("Step 10: Generating Enhanced Visualization Charts")
print("=" * 80)

# 12.1 Enhanced ACF and PACF plots
print("Generating enhanced ACF/PACF plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Original series ACF
plot_acf(pm25_series.dropna(), lags=40, ax=axes[0, 0], alpha=0.05)
axes[0, 0].set_title('Original Series - Autocorrelation Function (ACF)', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Lag (Days)', fontsize=11)
axes[0, 0].set_ylabel('Autocorrelation', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# Original series PACF
plot_pacf(pm25_series.dropna(), lags=40, ax=axes[0, 1], alpha=0.05)
axes[0, 1].set_title('Original Series - Partial Autocorrelation Function (PACF)', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Lag (Days)', fontsize=11)
axes[0, 1].set_ylabel('Partial Autocorrelation', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# First-order differenced series ACF
pm25_diff = pm25_series.diff().dropna()
plot_acf(pm25_diff, lags=40, ax=axes[1, 0], alpha=0.05)
axes[1, 0].set_title('First-order Differenced Series - ACF', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Lag (Days)', fontsize=11)
axes[1, 0].set_ylabel('Autocorrelation', fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# First-order differenced series PACF
plot_pacf(pm25_diff, lags=40, ax=axes[1, 1], alpha=0.05)
axes[1, 1].set_title('First-order Differenced Series - PACF', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Lag (Days)', fontsize=11)
axes[1, 1].set_ylabel('Partial Autocorrelation', fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'enhanced_acf_pacf_plots.png', dpi=300, bbox_inches='tight')
print("Saved: enhanced_acf_pacf_plots.png")
plt.close()

# 12.2 Enhanced time series prediction comparison
print("Generating enhanced time series comparison plots...")
fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# Test set - basic model
if basic_model_success:
    plot_range = min(200, len(y_test))  # 减少点数使图表更清晰
    plot_idx = range(len(y_test) - plot_range, len(y_test))
    time_idx = y_test.index[plot_idx]

    axes[0].plot(time_idx, y_test.iloc[plot_idx], 'k-', label='Actual', 
                 linewidth=2, alpha=0.9, marker='o', markersize=2)
    axes[0].plot(time_idx, y_test_pred_basic.iloc[plot_idx], 'b--', label='Basic Model Prediction', 
                 linewidth=1.5, alpha=0.8, marker='s', markersize=2)
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
    axes[0].set_title('ARIMA Basic Model - Enhanced Time Series Prediction (Last 200 Days)', 
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
    
    # 添加性能指标
    r2_basic = r2_score(y_test.iloc[plot_idx], y_test_pred_basic.iloc[plot_idx])
    rmse_basic = np.sqrt(mean_squared_error(y_test.iloc[plot_idx], y_test_pred_basic.iloc[plot_idx]))
    axes[0].text(0.02, 0.98, f'R² = {r2_basic:.3f}\nRMSE = {rmse_basic:.1f}', 
                 transform=axes[0].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=10)

# Test set - optimized model
plot_range = min(200, len(y_test))
plot_idx = range(len(y_test) - plot_range, len(y_test))
time_idx = y_test.index[plot_idx]

axes[1].plot(time_idx, y_test.iloc[plot_idx], 'k-', label='Actual', 
             linewidth=2, alpha=0.9, marker='o', markersize=2)
axes[1].plot(time_idx, y_test_pred_opt.iloc[plot_idx], 'g--', label='Optimized Model Prediction', 
             linewidth=1.5, alpha=0.8, marker='s', markersize=2)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
axes[1].set_title('ARIMA Optimized Model - Enhanced Time Series Prediction (Last 200 Days)', 
                  fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

# 添加性能指标
r2_opt = r2_score(y_test.iloc[plot_idx], y_test_pred_opt.iloc[plot_idx])
rmse_opt = np.sqrt(mean_squared_error(y_test.iloc[plot_idx], y_test_pred_opt.iloc[plot_idx]))
axes[1].text(0.02, 0.98, f'R² = {r2_opt:.3f}\nRMSE = {rmse_opt:.1f}', 
             transform=axes[1].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'enhanced_timeseries_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: enhanced_timeseries_comparison.png")
plt.close()

# 12.3 Enhanced residual diagnostics
print("Generating enhanced residual diagnostics plots...")
fig = plt.figure(figsize=(20, 12))

# Optimized model residual diagnostics
# QQ plot
ax1 = plt.subplot(2, 3, 1)
stats.probplot(residuals_opt.dropna(), dist="norm", plot=plt)
ax1.set_title('Optimized Model - QQ Plot', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Residual time series
ax2 = plt.subplot(2, 3, 2)
residuals_opt.plot(ax=ax2, color='green', alpha=0.7, linewidth=1)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_title('Optimized Model - Residual Time Series', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Residual', fontsize=11)
ax2.grid(True, alpha=0.3)

# Residual ACF
ax3 = plt.subplot(2, 3, 3)
plot_acf(residuals_opt.dropna(), lags=30, ax=ax3, alpha=0.05)
ax3.set_title('Optimized Model - Residual ACF', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Residual distribution
ax4 = plt.subplot(2, 3, 4)
residuals_opt.hist(bins=50, ax=ax4, color='green', alpha=0.7, edgecolor='black')
ax4.axvline(x=residuals_opt.mean(), color='r', linestyle='--', linewidth=2, 
            label=f'Mean: {residuals_opt.mean():.3f}')
ax4.set_xlabel('Residual Value', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Optimized Model - Residual Distribution', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Residuals vs Fitted
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(y_train_pred_opt, residuals_opt[:len(y_train_pred_opt)], alpha=0.5, s=20, color='green')
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Fitted Values', fontsize=11)
ax5.set_ylabel('Residuals', fontsize=11)
ax5.set_title('Optimized Model - Residuals vs Fitted', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Normality test comparison
ax6 = plt.subplot(2, 3, 6)
models = ['Optimized Model']
jb_stats = [stats.jarque_bera(residuals_opt.dropna())[0]]
jb_pvalues = [stats.jarque_bera(residuals_opt.dropna())[1]]

x_pos = np.arange(len(models))
bars = ax6.bar(x_pos, jb_pvalues, color=['green'], alpha=0.7, edgecolor='black')
ax6.axhline(y=0.05, color='r', linestyle='--', linewidth=2, label='Significance Level (0.05)')
ax6.set_ylabel('Jarque-Bera p-value', fontsize=11)
ax6.set_title('Normality Test Comparison', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(models)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# Add p-value labels
for bar, pval in zip(bars, jb_pvalues):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height, f'{pval:.4f}', 
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'enhanced_residual_diagnostics.png', dpi=300, bbox_inches='tight')
print("Saved: enhanced_residual_diagnostics.png")
plt.close()

# 12.4 Enhanced model comparison
print("Generating enhanced model comparison plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

metrics = ['R²', 'RMSE', 'MAE', 'MAPE', 'Direction_Accuracy']
colors = ['blue', 'green']

for i, metric in enumerate(metrics[:3]):  # 前3个指标在上排
    model_values = [
        results_basic_df[results_basic_df['Dataset'] == 'Test'][metric].values[0] if basic_model_success else 0,
        results_opt_df[results_opt_df['Dataset'] == 'Test'][metric].values[0]
    ]
    
    bars = axes[0, i].bar(['Basic', 'Optimized'], model_values, color=colors, alpha=0.7, 
                         edgecolor='black', linewidth=1.5)
    axes[0, i].set_ylabel(metric, fontsize=12)
    
    if metric == 'R²':
        axes[0, i].set_title(f'{metric} Comparison\n(Higher is Better)', fontsize=12, fontweight='bold')
    else:
        axes[0, i].set_title(f'{metric} Comparison\n(Lower is Better)', fontsize=12, fontweight='bold')
    
    axes[0, i].grid(True, alpha=0.3, axis='y')
    
    # Display values
    for j, v in enumerate(model_values):
        if metric == 'MAPE':
            axes[0, i].text(j, v, f'{v:.1f}%', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
        elif metric == 'Direction_Accuracy':
            axes[0, i].text(j, v, f'{v:.1f}%', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
        else:
            axes[0, i].text(j, v, f'{v:.3f}', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')

for i, metric in enumerate(metrics[3:], 0):  # 后2个指标在下排
    model_values = [
        results_basic_df[results_basic_df['Dataset'] == 'Test'][metric].values[0] if basic_model_success else 0,
        results_opt_df[results_opt_df['Dataset'] == 'Test'][metric].values[0]
    ]
    
    bars = axes[1, i].bar(['Basic', 'Optimized'], model_values, color=colors, alpha=0.7, 
                         edgecolor='black', linewidth=1.5)
    axes[1, i].set_ylabel(metric, fontsize=12)
    
    if metric == 'Direction_Accuracy':
        axes[1, i].set_title(f'{metric} Comparison\n(Higher is Better)', fontsize=12, fontweight='bold')
    else:
        axes[1, i].set_title(f'{metric} Comparison\n(Lower is Better)', fontsize=12, fontweight='bold')
    
    axes[1, i].grid(True, alpha=0.3, axis='y')
    
    # Display values
    for j, v in enumerate(model_values):
        axes[1, i].text(j, v, f'{v:.1f}%', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')

# 隐藏多余的子图
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'enhanced_model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: enhanced_model_comparison.png")
plt.close()

# ============================== Part 13: Save Enhanced Results ==============================
print("\n" + "=" * 80)
print("Step 11: Saving Enhanced Results")
print("=" * 80)

# Save model performance
all_results.to_csv(output_dir / 'enhanced_model_performance.csv', index=False, encoding='utf-8-sig')
print("Saved: enhanced_model_performance.csv")

# Save best parameters
best_params_df = pd.DataFrame([{
    'ARIMA_order_p': best_order[0],
    'ARIMA_order_d': best_order[1],
    'ARIMA_order_q': best_order[2],
    'Seasonal_order_P': best_seasonal_order[0],
    'Seasonal_order_D': best_seasonal_order[1],
    'Seasonal_order_Q': best_seasonal_order[2],
    'Seasonal_order_s': best_seasonal_order[3],
    'AIC': fit_optimized.aic,
    'BIC': fit_optimized.bic,
    'LogLikelihood': fit_optimized.llf
}])
best_params_df.to_csv(output_dir / 'enhanced_best_parameters.csv', index=False, encoding='utf-8-sig')
print("Saved: enhanced_best_parameters.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'Date': y_test.index,
    'Actual': y_test.values,
    'Prediction_Basic': y_test_pred_basic.values if basic_model_success else [np.nan] * len(y_test),
    'Prediction_Optimized': y_test_pred_opt.values,
    'Error_Basic': y_test.values - y_test_pred_basic.values if basic_model_success else [np.nan] * len(y_test),
    'Error_Optimized': y_test.values - y_test_pred_opt.values
})
predictions_df.to_csv(output_dir / 'enhanced_predictions.csv', index=False, encoding='utf-8-sig')
print("Saved: enhanced_predictions.csv")

# Save residual diagnostics results
diagnostics_data = {
    'Model': ['Optimized'],
    'Residual_Mean': [residuals_opt.mean()],
    'Residual_Std': [residuals_opt.std()],
    'Residual_Skewness': [stats.skew(residuals_opt.dropna())],
    'Residual_Kurtosis': [stats.kurtosis(residuals_opt.dropna())],
    'Ljungbox_pvalue_lag10': [acorr_ljungbox(residuals_opt, lags=[10], return_df=True)['lb_pvalue'].iloc[-1]],
    'Ljungbox_pvalue_lag20': [acorr_ljungbox(residuals_opt, lags=[20], return_df=True)['lb_pvalue'].iloc[-1]],
    'JarqueBera_statistic': [stats.jarque_bera(residuals_opt.dropna())[0]],
    'JarqueBera_pvalue': [stats.jarque_bera(residuals_opt.dropna())[1]]
}

if basic_model_success:
    diagnostics_data['Model'].insert(0, 'Basic')
    diagnostics_data['Residual_Mean'].insert(0, residuals_basic.mean())
    diagnostics_data['Residual_Std'].insert(0, residuals_basic.std())
    diagnostics_data['Residual_Skewness'].insert(0, stats.skew(residuals_basic.dropna()))
    diagnostics_data['Residual_Kurtosis'].insert(0, stats.kurtosis(residuals_basic.dropna()))
    diagnostics_data['Ljungbox_pvalue_lag10'].insert(0, acorr_ljungbox(residuals_basic, lags=[10], return_df=True)['lb_pvalue'].iloc[-1])
    diagnostics_data['Ljungbox_pvalue_lag20'].insert(0, acorr_ljungbox(residuals_basic, lags=[20], return_df=True)['lb_pvalue'].iloc[-1])
    diagnostics_data['JarqueBera_statistic'].insert(0, stats.jarque_bera(residuals_basic.dropna())[0])
    diagnostics_data['JarqueBera_pvalue'].insert(0, stats.jarque_bera(residuals_basic.dropna())[1])

diagnostics_df = pd.DataFrame(diagnostics_data)
diagnostics_df.to_csv(output_dir / 'enhanced_diagnostics.csv', index=False, encoding='utf-8-sig')
print("Saved: enhanced_diagnostics.csv")

# Save models
if basic_model_success:
    with open(model_dir / 'enhanced_arima_basic.pkl', 'wb') as f:
        pickle.dump(fit_basic, f)
    print("Saved: enhanced_arima_basic.pkl")

with open(model_dir / 'enhanced_arima_optimized.pkl', 'wb') as f:
    pickle.dump(fit_optimized, f)
print("Saved: enhanced_arima_optimized.pkl")

# ============================== Part 14: Enhanced Summary Report ==============================
print("\n" + "=" * 80)
print("Enhanced Analysis Complete!")
print("=" * 80)

print("\nKey Improvements Applied:")
print("✓ Enhanced data preprocessing with interpolation and outlier handling")
print("✓ Improved parameter selection with simpler, more stable models")
print("✓ Better seasonal pattern detection (weekly vs monthly)")
print("✓ Enhanced residual diagnostics and model validation")
print("✓ More informative visualizations with performance metrics")
print("✓ Robust error handling with fallback mechanisms")

print("\nGenerated Enhanced Files:")
print("\nCSV files:")
print("  - enhanced_model_performance.csv    Enhanced model performance comparison")
print("  - enhanced_best_parameters.csv      Enhanced best ARIMA parameters")
print("  - enhanced_predictions.csv          Enhanced prediction results")
print("  - enhanced_diagnostics.csv          Enhanced residual diagnostics results")

print("\nChart files:")
print("  - enhanced_acf_pacf_plots.png       Enhanced ACF/PACF analysis plots")
print("  - enhanced_residual_diagnostics.png Enhanced residual diagnostics plots")
print("  - enhanced_timeseries_comparison.png Enhanced time series comparison")
print("  - enhanced_model_comparison.png     Enhanced model performance comparison")

print("\nModel files:")
if basic_model_success:
    print("  - enhanced_arima_basic.pkl        Enhanced basic ARIMA model")
print("  - enhanced_arima_optimized.pkl      Enhanced optimized ARIMA model")

# Best model information
best_model = test_results.iloc[0]
print(f"\nBest model: {best_model['Model']}")
print(f"  R² Score: {best_model['R²']:.4f}")
print(f"  RMSE: {best_model['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model['MAPE']:.2f}%")
print(f"  Direction Accuracy: {best_model['Direction_Accuracy']:.2f}%")

print(f"\nBest ARIMA parameters:")
print(f"  ARIMA order (p,d,q): {best_order}")
print(f"  Seasonal order (P,D,Q,s): {best_seasonal_order}")
print(f"  AIC: {fit_optimized.aic:.2f}")
print(f"  BIC: {fit_optimized.bic:.2f}")

print("\n" + "=" * 80)
print("ENHANCED ARIMA PM2.5 Concentration Prediction Complete!")
print("=" * 80)