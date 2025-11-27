import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime, timedelta
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import pickle
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

# Set English fonts
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - SARIMAX Model")
print("=" * 80)

# ============================== Part 1: Configuration and Path Setup ==============================
print("\n配置参数...")

# Data paths
pollution_all_path = '/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)'

# Date range
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

print(f"数据时间范围: {start_date.date()} 至 {end_date.date()}")
print(f"目标变量: PM2.5 浓度 (时间序列)")
print(f"CPU 核心数: {CPU_COUNT}, 并行工作进程数: {MAX_WORKERS}")

# ============================== Part 2: Data Loading Functions ==============================
def daterange(start, end):
    """生成日期序列"""
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
    print("\n加载污染数据...")
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
                             desc="加载污染数据", unit="天"):
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
        print(f"  成功读取 {len(pollution_series)}/{len(dates)} 天的数据")
        print("  合并数据...")
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
        
        print(f"污染数据加载完成, 形状: {pm25_series.shape}")
        return pm25_series
    return pd.Series()

# ============================== Part 3: Data Loading and Preprocessing ==============================
print("\n" + "=" * 80)
print("步骤 1: 数据加载和预处理")
print("=" * 80)

pm25_series = read_all_pollution()

# Check data loading
print("\n数据加载检查:")
print(f"  PM2.5 序列形状: {pm25_series.shape}")

if pm25_series.empty:
    print("\n⚠️ 警告: 污染数据为空！请检查数据路径和文件。")
    import sys
    sys.exit(1)

print(f"  时间范围: {pm25_series.index.min()} 至 {pm25_series.index.max()}")
print(f"  数据点数量: {len(pm25_series)}")

# Basic statistics
print(f"\nPM2.5 统计信息:")
print(f"  均值: {pm25_series.mean():.2f} μg/m³")
print(f"  标准差: {pm25_series.std():.2f} μg/m³")
print(f"  最小值: {pm25_series.min():.2f} μg/m³")
print(f"  最大值: {pm25_series.max():.2f} μg/m³")
print(f"  中位数: {pm25_series.median():.2f} μg/m³")
print(f"  缺失值: {pm25_series.isna().sum()}")

# 数据已经是日度数据，无需重采样
pm25_series = pm25_series.dropna()  # 删除任何剩余的 NaN 值

# ============================== Part 4: Train-Test Split ==============================
print("\n" + "=" * 80)
print("步骤 2: 数据集划分")
print("=" * 80)

# Split by time order: 70% training, 15% validation, 15% test
train_size = int(len(pm25_series) * 0.7)
val_size = int(len(pm25_series) * 0.15)

y_train = pm25_series[:train_size]
y_val = pm25_series[train_size:train_size + val_size]
y_test = pm25_series[train_size + val_size:]

print(f"\n训练集: {len(y_train)} 个样本 ({len(y_train)/len(pm25_series)*100:.1f}%)")
print(f"  时间范围: {y_train.index.min().date()} 至 {y_train.index.max().date()}")
print(f"验证集: {len(y_val)} 个样本 ({len(y_val)/len(pm25_series)*100:.1f}%)")
print(f"  时间范围: {y_val.index.min().date()} 至 {y_val.index.max().date()}")
print(f"测试集: {len(y_test)} 个样本 ({len(y_test)/len(pm25_series)*100:.1f}%)")
print(f"  时间范围: {y_test.index.min().date()} 至 {y_test.index.max().date()}")

# ============================== Part 5: SARIMAX Model ==============================
print("\n" + "=" * 80)
print("步骤 3: SARIMAX 模型训练")
print("=" * 80)

# Define the SARIMAX model
order = (3, 1, 3)  # ARIMA (p, d, q) order
seasonal_order = (1, 1, 1, 12)  # Seasonal order (P, D, Q, s)

print(f"\n模型参数:")
print(f"  ARIMA 阶数 (p,d,q): {order}")
print(f"  季节阶数 (P,D,Q,s): {seasonal_order}")

print("\n开始模型训练...")
# Fit the model using SARIMAX
sarimax_model = SARIMAX(y_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)

sarimax_fit = sarimax_model.fit(disp=False)

print(f"\n✓ 模型训练完成")
print(f"  AIC: {sarimax_fit.aic:.2f}")
print(f"  BIC: {sarimax_fit.bic:.2f}")
print(f"  对数似然值: {sarimax_fit.llf:.2f}")

# ============================== Part 6: Predictions ==============================
print("\n进行预测...")
# Make predictions on training, validation, and test sets
y_train_pred = sarimax_fit.fittedvalues
y_val_pred = sarimax_fit.forecast(steps=len(y_val))
y_test_pred = sarimax_fit.forecast(steps=len(y_test))

# ============================== Part 7: Model Evaluation ==============================
print("\n" + "=" * 80)
print("步骤 4: 模型评估")
print("=" * 80)

def evaluate_model(y_true, y_pred, dataset_name):
    """评估模型性能"""
    # 确保索引对齐
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

# Evaluate the model
results = []
results.append(evaluate_model(y_train, y_train_pred, 'Train'))
results.append(evaluate_model(y_val, y_val_pred, 'Validation'))
results.append(evaluate_model(y_test, y_test_pred, 'Test'))

# Convert results to DataFrame for easy viewing
results_df = pd.DataFrame(results)
print("\n模型性能评估:")
print(results_df.to_string(index=False))

# ============================== Part 8: Visualization ==============================
print("\n" + "=" * 80)
print("步骤 5: 生成可视化图表")
print("=" * 80)

print("生成时间序列预测对比图...")
# Time Series Plot for Prediction Comparison
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual', color='black', linewidth=2)
# forecast 返回的 Series 索引可能不同，使用 values 确保对齐
y_test_pred_values = y_test_pred.values if isinstance(y_test_pred, pd.Series) else y_test_pred
plt.plot(y_test.index, y_test_pred_values, label='SARIMAX Prediction', linestyle='--', color='blue', linewidth=1.5)
plt.title('SARIMAX Model - Time Series Prediction Comparison (Test Set)')
plt.xlabel('Date')
plt.ylabel('PM2.5 Concentration (μg/m³)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sarimax_timeseries_comparison.png', dpi=300, bbox_inches='tight')
print("已保存: sarimax_timeseries_comparison.png")
plt.close()

# ============================== Part 9: Save Results ==============================
print("\n" + "=" * 80)
print("步骤 6: 保存结果")
print("=" * 80)

# Save the model
with open('sarimax_model.pkl', 'wb') as f:
    pickle.dump(sarimax_fit, f)
print("已保存: sarimax_model.pkl")

# Save predictions to a CSV file
predictions_df = pd.DataFrame({
    'Date': y_test.index,
    'Actual': y_test.values,
    'Prediction': y_test_pred.values if isinstance(y_test_pred, pd.Series) else y_test_pred
})
predictions_df.to_csv('sarimax_predictions.csv', index=False, encoding='utf-8-sig')
print("已保存: sarimax_predictions.csv")

# Save model performance
results_df.to_csv('sarimax_performance.csv', index=False, encoding='utf-8-sig')
print("已保存: sarimax_performance.csv")

print("\n" + "=" * 80)
print("SARIMAX PM2.5 浓度预测完成！")
print("=" * 80)
