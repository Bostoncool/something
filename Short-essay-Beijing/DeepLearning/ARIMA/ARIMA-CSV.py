"""
Beijing PM2.5 Concentration Prediction - ARIMA Model
Using ARIMA/SARIMA time series models for univariate prediction

Features:
- Pure time series approach (using only PM2.5 historical values)
- SARIMA supports seasonal analysis
- auto_arima automatic parameter optimization
- Complete residual diagnostics
- ACF/PACF analysis
- ADF stationarity test

Data Sources:
- Pollution data: Benchmark dataset (PM2.5 concentration)
"""

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

# Set English fonts
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# Set random seed
np.random.seed(42)

print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - ARIMA Model")
print("=" * 80)

# ============================== Part 1: Configuration and Path Setup ==============================
print("\nConfiguring parameters...")

# Data paths
pollution_all_path = '/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)'
pollution_extra_path = '/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)'

# Output path
output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)

# Model save path
model_dir = Path('./models')
model_dir.mkdir(exist_ok=True)

# Date range
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

print(f"Data time range: {start_date.date()} to {end_date.date()}")
print(f"Target variable: PM2.5 concentration (time series)")
print(f"Output directory: {output_dir}")
print(f"Model save directory: {model_dir}")
print(f"CPU cores: {CPU_COUNT}, parallel workers: {MAX_WORKERS}")

# ============================== Part 2: Data Loading Functions ==============================
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
    # 注意：多进程需要传递可序列化的参数
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
        
        # Forward fill then mean fill
        pm25_series = pm25_series.ffill().bfill()
        pm25_series = pm25_series.fillna(pm25_series.mean())
        
        print(f"Pollution data loading complete, shape: {pm25_series.shape}")
        return pm25_series
    return pd.Series()

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

# Basic statistics
print(f"\nPM2.5 Statistics:")
print(f"  Mean: {pm25_series.mean():.2f} μg/m³")
print(f"  Std Dev: {pm25_series.std():.2f} μg/m³")
print(f"  Min: {pm25_series.min():.2f} μg/m³")
print(f"  Max: {pm25_series.max():.2f} μg/m³")
print(f"  Median: {pm25_series.median():.2f} μg/m³")
print(f"  Missing values: {pm25_series.isna().sum()}")

# ============================== Part 4: Time Series Analysis ==============================
print("\n" + "=" * 80)
print("Step 2: Time Series Stationarity Analysis")
print("=" * 80)

# ADF test (Augmented Dickey-Fuller test)
print("\nPerforming ADF stationarity test...")
adf_result = adfuller(pm25_series.dropna())
print(f"\nADF Test Results:")
print(f"  ADF Statistic: {adf_result[0]:.4f}")
print(f"  p-value: {adf_result[1]:.4f}")
print(f"  Critical values:")
for key, value in adf_result[4].items():
    print(f"    {key}: {value:.4f}")

if adf_result[1] < 0.05:
    print(f"\n✓ Series is stationary (p={adf_result[1]:.4f} < 0.05)")
else:
    print(f"\n⚠ Series is non-stationary (p={adf_result[1]:.4f} >= 0.05), ARIMA will automatically perform differencing")

# ============================== Part 5: Dataset Splitting ==============================
print("\n" + "=" * 80)
print("Step 3: Dataset Splitting")
print("=" * 80)

# Split by time order: 70% training, 15% validation, 15% test
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

# ============================== Part 6: Basic ARIMA Model ==============================
print("\n" + "=" * 80)
print("Step 4: Basic SARIMA Model Training")
print("=" * 80)

# Basic model parameters
order_basic = (5, 1, 5)  # (p, d, q)
seasonal_order_basic = (1, 1, 1, 12)  # (P, D, Q, s)

print(f"\nBasic model parameters:")
print(f"  ARIMA order (p,d,q): {order_basic}")
print(f"  Seasonal order (P,D,Q,s): {seasonal_order_basic}")

print("\nStarting basic model training...")
model_basic = SARIMAX(y_train, 
                      order=order_basic,
                      seasonal_order=seasonal_order_basic,
                      enforce_stationarity=False,
                      enforce_invertibility=False)

fit_basic = model_basic.fit(disp=False, maxiter=200)

print(f"\n✓ Basic model training complete")
print(f"  AIC: {fit_basic.aic:.2f}")
print(f"  BIC: {fit_basic.bic:.2f}")
print(f"  Log-likelihood: {fit_basic.llf:.2f}")

# Prediction
print("\nMaking predictions...")
y_train_pred_basic = fit_basic.fittedvalues
y_val_pred_basic = fit_basic.forecast(steps=len(y_val))
y_test_pred_basic = fit_basic.forecast(steps=len(y_val) + len(y_test)).iloc[len(y_val):]

# Evaluation
def evaluate_model(y_true, y_pred, dataset_name):
    """Evaluate model performance"""
    # Ensure index alignment
    y_true_aligned = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred_aligned = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    
    r2 = r2_score(y_true_aligned, y_pred_aligned)
    rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
    mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
    mape = np.mean(np.abs((y_true_aligned - y_pred_aligned) / y_true_aligned)) * 100
    return {
        'Dataset': dataset_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

results_basic = []
results_basic.append(evaluate_model(y_train, y_train_pred_basic, 'Train'))
results_basic.append(evaluate_model(y_val, y_val_pred_basic, 'Validation'))
results_basic.append(evaluate_model(y_test, y_test_pred_basic, 'Test'))

results_basic_df = pd.DataFrame(results_basic)
print("\nBasic model performance:")
print(results_basic_df.to_string(index=False))

# ============================== Part 7: Hyperparameter Optimization ==============================
print("\n" + "=" * 80)
print("Step 5: Hyperparameter Optimization (auto_arima)")
print("=" * 80)

if AUTO_ARIMA_AVAILABLE:
    print("\nUsing auto_arima to automatically search for best parameters...")
    print("This may take a few minutes...")
    
    auto_model = pm.auto_arima(
        y_train,
        start_p=1, start_q=1,
        max_p=7, max_q=7,
        d=None,  # Automatically determine differencing order
        seasonal=True, m=12,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        D=None,  # Automatically determine seasonal differencing order
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        n_jobs=-1
    )
    
    print(f"\n✓ auto_arima search complete")
    print(f"\nBest model: {auto_model.order} x {auto_model.seasonal_order}")
    print(f"  AIC: {auto_model.aic():.2f}")
    print(f"  BIC: {auto_model.bic():.2f}")
    
    best_order = auto_model.order
    best_seasonal_order = auto_model.seasonal_order
    
else:
    print("\n⚠ auto_arima not available, using manual optimization parameters")
    best_order = (3, 1, 3)
    best_seasonal_order = (1, 1, 1, 12)
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

# Evaluation
results_opt = []
results_opt.append(evaluate_model(y_train, y_train_pred_opt, 'Train'))
results_opt.append(evaluate_model(y_val, y_val_pred_opt, 'Validation'))
results_opt.append(evaluate_model(y_test, y_test_pred_opt, 'Test'))

results_opt_df = pd.DataFrame(results_opt)
print("\nOptimized model performance:")
print(results_opt_df.to_string(index=False))

# ============================== Part 9: Model Comparison ==============================
print("\n" + "=" * 80)
print("Step 7: Model Performance Comparison")
print("=" * 80)

# Merge results
results_basic_df['Model'] = 'ARIMA_Basic'
results_opt_df['Model'] = 'ARIMA_Optimized'
all_results = pd.concat([results_basic_df, results_opt_df])

# Rearrange column order
all_results = all_results[['Model', 'Dataset', 'R²', 'RMSE', 'MAE', 'MAPE']]

print("\nAll models performance comparison:")
print(all_results.to_string(index=False))

# Test set performance comparison
test_results = all_results[all_results['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\nTest set performance ranking:")
print(test_results.to_string(index=False))

# Performance improvement
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

# ============================== Part 10: Residual Diagnostics ==============================
print("\n" + "=" * 80)
print("Step 8: Residual Diagnostics")
print("=" * 80)

# Basic model residuals
residuals_basic = fit_basic.resid
print("\nBasic model residual statistics:")
print(f"  Mean: {residuals_basic.mean():.4f}")
print(f"  Std Dev: {residuals_basic.std():.4f}")

# Ljung-Box test (test residual autocorrelation)
lb_test_basic = acorr_ljungbox(residuals_basic, lags=10, return_df=True)
print(f"\nLjung-Box test (basic model):")
print(f"  p-value (lag 10): {lb_test_basic['lb_pvalue'].iloc[-1]:.4f}")
if lb_test_basic['lb_pvalue'].iloc[-1] > 0.05:
    print("  ✓ No significant residual autocorrelation (white noise)")
else:
    print("  ⚠ Residuals have autocorrelation")

# Jarque-Bera normality test
jb_test_basic = stats.jarque_bera(residuals_basic.dropna())
print(f"\nJarque-Bera normality test (basic model):")
print(f"  Statistic: {jb_test_basic[0]:.4f}")
print(f"  p-value: {jb_test_basic[1]:.4f}")
if jb_test_basic[1] > 0.05:
    print("  ✓ Residuals follow normal distribution")
else:
    print("  ⚠ Residuals do not follow normal distribution")

# Optimized model residuals
residuals_opt = fit_optimized.resid
print("\nOptimized model residual statistics:")
print(f"  Mean: {residuals_opt.mean():.4f}")
print(f"  Std Dev: {residuals_opt.std():.4f}")

lb_test_opt = acorr_ljungbox(residuals_opt, lags=10, return_df=True)
print(f"\nLjung-Box test (optimized model):")
print(f"  p-value (lag 10): {lb_test_opt['lb_pvalue'].iloc[-1]:.4f}")
if lb_test_opt['lb_pvalue'].iloc[-1] > 0.05:
    print("  ✓ No significant residual autocorrelation (white noise)")
else:
    print("  ⚠ Residuals have autocorrelation")

jb_test_opt = stats.jarque_bera(residuals_opt.dropna())
print(f"\nJarque-Bera normality test (optimized model):")
print(f"  Statistic: {jb_test_opt[0]:.4f}")
print(f"  p-value: {jb_test_opt[1]:.4f}")
if jb_test_opt[1] > 0.05:
    print("  ✓ Residuals follow normal distribution")
else:
    print("  ⚠ Residuals do not follow normal distribution")

# ============================== Part 11: Visualization ==============================
print("\n" + "=" * 80)
print("Step 9: Generating Visualization Charts")
print("=" * 80)

# 11.1 ACF and PACF plots (original series)
print("Generating ACF/PACF plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Original series ACF
plot_acf(pm25_series.dropna(), lags=40, ax=axes[0, 0])
axes[0, 0].set_title('Original Series - Autocorrelation Function (ACF)', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Lag', fontsize=11)
axes[0, 0].set_ylabel('Autocorrelation', fontsize=11)

# Original series PACF
plot_pacf(pm25_series.dropna(), lags=40, ax=axes[0, 1])
axes[0, 1].set_title('Original Series - Partial Autocorrelation Function (PACF)', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Lag', fontsize=11)
axes[0, 1].set_ylabel('Partial Autocorrelation', fontsize=11)

# First-order differenced series ACF
pm25_diff = pm25_series.diff().dropna()
plot_acf(pm25_diff, lags=40, ax=axes[1, 0])
axes[1, 0].set_title('First-order Differenced Series - Autocorrelation Function (ACF)', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Lag', fontsize=11)
axes[1, 0].set_ylabel('Autocorrelation', fontsize=11)

# First-order differenced series PACF
plot_pacf(pm25_diff, lags=40, ax=axes[1, 1])
axes[1, 1].set_title('First-order Differenced Series - Partial Autocorrelation Function (PACF)', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Lag', fontsize=11)
axes[1, 1].set_ylabel('Partial Autocorrelation', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'acf_pacf_plots.png', dpi=300, bbox_inches='tight')
print("Saved: acf_pacf_plots.png")
plt.close()

# 11.2 Residual diagnostics plots
print("Generating residual diagnostics plots...")
fig = plt.figure(figsize=(18, 12))

# Basic model residual diagnostics
# QQ plot
ax1 = plt.subplot(3, 2, 1)
stats.probplot(residuals_basic.dropna(), dist="norm", plot=plt)
ax1.set_title('Basic Model - QQ Plot', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Residual time series
ax2 = plt.subplot(3, 2, 2)
residuals_basic.plot(ax=ax2, color='blue', alpha=0.7)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_title('Basic Model - Residual Time Series', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Residual', fontsize=11)
ax2.grid(True, alpha=0.3)

# Residual ACF
ax3 = plt.subplot(3, 2, 3)
plot_acf(residuals_basic.dropna(), lags=30, ax=ax3)
ax3.set_title('Basic Model - Residual ACF', fontsize=12, fontweight='bold')

# Optimized model residual diagnostics
# QQ plot
ax4 = plt.subplot(3, 2, 4)
stats.probplot(residuals_opt.dropna(), dist="norm", plot=plt)
ax4.set_title('Optimized Model - QQ Plot', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Residual time series
ax5 = plt.subplot(3, 2, 5)
residuals_opt.plot(ax=ax5, color='green', alpha=0.7)
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_title('Optimized Model - Residual Time Series', fontsize=12, fontweight='bold')
ax5.set_xlabel('Date', fontsize=11)
ax5.set_ylabel('Residual', fontsize=11)
ax5.grid(True, alpha=0.3)

# Residual ACF
ax6 = plt.subplot(3, 2, 6)
plot_acf(residuals_opt.dropna(), lags=30, ax=ax6)
ax6.set_title('Optimized Model - Residual ACF', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'residual_diagnostics.png', dpi=300, bbox_inches='tight')
print("Saved: residual_diagnostics.png")
plt.close()

# 11.3 Prediction vs actual scatter plots
print("Generating prediction scatter plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

models_data = [
    ('Basic', y_train_pred_basic, y_train, 'Train'),
    ('Basic', y_val_pred_basic, y_val, 'Val'),
    ('Basic', y_test_pred_basic, y_test, 'Test'),
    ('Optimized', y_train_pred_opt, y_train, 'Train'),
    ('Optimized', y_val_pred_opt, y_val, 'Val'),
    ('Optimized', y_test_pred_opt, y_test, 'Test')
]

for idx, (model_name, y_pred, y_true, dataset) in enumerate(models_data):
    row = idx // 3
    col = idx % 3
    
    ax = axes[row, col]
    
    y_true_vals = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred_vals = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    
    # Scatter plot
    ax.scatter(y_true_vals, y_pred_vals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    
    # Ideal prediction line
    min_val = min(y_true_vals.min(), y_pred_vals.min())
    max_val = max(y_true_vals.max(), y_pred_vals.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')
    
    # Calculate metrics
    r2 = r2_score(y_true_vals, y_pred_vals)
    rmse = np.sqrt(mean_squared_error(y_true_vals, y_pred_vals))
    
    ax.set_xlabel('Actual PM2.5 Concentration (μg/m³)', fontsize=11)
    ax.set_ylabel('Predicted PM2.5 Concentration (μg/m³)', fontsize=11)
    ax.set_title(f'ARIMA_{model_name} - {dataset}\nR²={r2:.4f}, RMSE={rmse:.2f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
print("Saved: prediction_scatter.png")
plt.close()

# 11.4 Time series prediction comparison
print("Generating time series comparison plots...")
fig, axes = plt.subplots(2, 1, figsize=(18, 10))

# Test set - basic model
plot_range = min(300, len(y_test))
plot_idx = range(len(y_test) - plot_range, len(y_test))
time_idx = y_test.index[plot_idx]

axes[0].plot(time_idx, y_test.iloc[plot_idx], 'k-', label='Actual', 
             linewidth=2, alpha=0.8)
axes[0].plot(time_idx, y_test_pred_basic.iloc[plot_idx], 'b--', label='Basic Model Prediction', 
             linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
axes[0].set_title('ARIMA Basic Model - Time Series Prediction Comparison (Last 300 Days of Test Set)', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

# Test set - optimized model
axes[1].plot(time_idx, y_test.iloc[plot_idx], 'k-', label='Actual', 
             linewidth=2, alpha=0.8)
axes[1].plot(time_idx, y_test_pred_opt.iloc[plot_idx], 'g--', label='Optimized Model Prediction', 
             linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
axes[1].set_title('ARIMA Optimized Model - Time Series Prediction Comparison (Last 300 Days of Test Set)', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(output_dir / 'timeseries_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: timeseries_comparison.png")
plt.close()

# 11.5 Residual analysis scatter plots
print("Generating residual analysis plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (model_name, y_pred, y_true, dataset) in enumerate(models_data):
    row = idx // 3
    col = idx % 3
    
    ax = axes[row, col]
    
    y_true_vals = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred_vals = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    
    residuals = y_true_vals - y_pred_vals
    
    ax.scatter(y_pred_vals, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Value (μg/m³)', fontsize=11)
    ax.set_ylabel('Residual (μg/m³)', fontsize=11)
    ax.set_title(f'ARIMA_{model_name} - {dataset}\nMean Residual={residuals.mean():.2f}, Std={residuals.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: residuals_analysis.png")
plt.close()

# 11.6 Model performance comparison bar charts
print("Generating model comparison plots...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

test_results_plot = all_results[all_results['Dataset'] == 'Test']
models = test_results_plot['Model'].tolist()
x_pos = np.arange(len(models))
colors = ['blue', 'green']

metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
for i, metric in enumerate(metrics):
    axes[i].bar(x_pos, test_results_plot[metric], color=colors, alpha=0.7, 
                edgecolor='black', linewidth=1.5)
    axes[i].set_xticks(x_pos)
    axes[i].set_xticklabels(['Basic', 'Optimized'], fontsize=11)
    axes[i].set_ylabel(metric, fontsize=12)
    
    if metric == 'R²':
        axes[i].set_title(f'{metric} Comparison\n(Higher is Better)', fontsize=12, fontweight='bold')
    else:
        axes[i].set_title(f'{metric} Comparison\n(Lower is Better)', fontsize=12, fontweight='bold')
    
    axes[i].grid(True, alpha=0.3, axis='y')
    
    # Display values
    for j, v in enumerate(test_results_plot[metric]):
        if metric == 'MAPE':
            axes[i].text(j, v, f'{v:.1f}%', ha='center', va='bottom', 
                         fontsize=10, fontweight='bold')
        else:
            axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom', 
                         fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison.png")
plt.close()

# 11.7 Error distribution histograms
print("Generating error distribution plots...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

errors_basic = y_test.values - y_test_pred_basic.values
errors_opt = y_test.values - y_test_pred_opt.values

axes[0].hist(errors_basic, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
axes[0].set_xlabel('Prediction Error (μg/m³)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title(f'Basic Model - Prediction Error Distribution\nMean={errors_basic.mean():.2f}, Std={errors_basic.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].hist(errors_opt, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
axes[1].set_xlabel('Prediction Error (μg/m³)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title(f'Optimized Model - Prediction Error Distribution\nMean={errors_opt.mean():.2f}, Std={errors_opt.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: error_distribution.png")
plt.close()

# 11.8 AIC/BIC comparison plots
print("Generating AIC/BIC comparison plots...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

models_aic_bic = ['Basic', 'Optimized']
aic_values = [fit_basic.aic, fit_optimized.aic]
bic_values = [fit_basic.bic, fit_optimized.bic]

x = np.arange(len(models_aic_bic))
width = 0.35

bars1 = ax.bar(x - width/2, aic_values, width, label='AIC', color='steelblue', 
               edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, bic_values, width, label='BIC', color='coral', 
               edgecolor='black', linewidth=1.5)

ax.set_ylabel('Information Criterion Value', fontsize=12)
ax.set_title('Model Information Criterion Comparison (AIC/BIC)\nLower is Better', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_aic_bic, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Display values
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'aic_bic_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: aic_bic_comparison.png")
plt.close()

# ============================== Part 12: Save Results ==============================
print("\n" + "=" * 80)
print("Step 10: Saving Results")
print("=" * 80)

# Save model performance
all_results.to_csv(output_dir / 'model_performance.csv', index=False, encoding='utf-8-sig')
print("Saved: model_performance.csv")

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
    'BIC': fit_optimized.bic
}])
best_params_df.to_csv(output_dir / 'best_parameters.csv', index=False, encoding='utf-8-sig')
print("Saved: best_parameters.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'Date': y_test.index,
    'Actual': y_test.values,
    'Prediction_Basic': y_test_pred_basic.values,
    'Prediction_Optimized': y_test_pred_opt.values,
    'Error_Basic': y_test.values - y_test_pred_basic.values,
    'Error_Optimized': y_test.values - y_test_pred_opt.values
})
predictions_df.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
print("Saved: predictions.csv")

# Save residual diagnostics results
diagnostics_df = pd.DataFrame({
    'Model': ['Basic', 'Optimized'],
    'Residual_Mean': [residuals_basic.mean(), residuals_opt.mean()],
    'Residual_Std': [residuals_basic.std(), residuals_opt.std()],
    'Ljungbox_pvalue': [lb_test_basic['lb_pvalue'].iloc[-1], lb_test_opt['lb_pvalue'].iloc[-1]],
    'JarqueBera_statistic': [jb_test_basic[0], jb_test_opt[0]],
    'JarqueBera_pvalue': [jb_test_basic[1], jb_test_opt[1]]
})
diagnostics_df.to_csv(output_dir / 'diagnostics.csv', index=False, encoding='utf-8-sig')
print("Saved: diagnostics.csv")

# Save models
with open(model_dir / 'arima_basic.pkl', 'wb') as f:
    pickle.dump(fit_basic, f)
print("Saved: arima_basic.pkl")

with open(model_dir / 'arima_optimized.pkl', 'wb') as f:
    pickle.dump(fit_optimized, f)
print("Saved: arima_optimized.pkl")

# ============================== Part 13: Summary Report ==============================
print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)

print("\nGenerated files:")
print("\nCSV files:")
print("  - model_performance.csv       Model performance comparison")
print("  - best_parameters.csv         Best ARIMA parameters")
print("  - predictions.csv             Prediction results")
print("  - diagnostics.csv             Residual diagnostics results")

print("\nChart files:")
print("  - acf_pacf_plots.png          ACF/PACF analysis plots")
print("  - residual_diagnostics.png    Residual diagnostics plots")
print("  - prediction_scatter.png      Prediction vs actual scatter plots")
print("  - timeseries_comparison.png   Time series comparison")
print("  - residuals_analysis.png      Residual analysis")
print("  - model_comparison.png        Model performance comparison")
print("  - error_distribution.png      Error distribution")
print("  - aic_bic_comparison.png      AIC/BIC comparison")

print("\nModel files:")
print("  - arima_basic.pkl             Basic ARIMA model")
print("  - arima_optimized.pkl         Optimized ARIMA model")

# Best model information
best_model = test_results.iloc[0]
print(f"\nBest model: {best_model['Model']}")
print(f"  R² Score: {best_model['R²']:.4f}")
print(f"  RMSE: {best_model['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model['MAPE']:.2f}%")

print(f"\nBest ARIMA parameters:")
print(f"  ARIMA order (p,d,q): {best_order}")
print(f"  Seasonal order (P,D,Q,s): {best_seasonal_order}")
print(f"  AIC: {fit_optimized.aic:.2f}")
print(f"  BIC: {fit_optimized.bic:.2f}")

print("\n" + "=" * 80)
print("ARIMA PM2.5 Concentration Prediction Complete!")
print("=" * 80)
