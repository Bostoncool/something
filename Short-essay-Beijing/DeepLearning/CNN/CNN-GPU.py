"""
北京PM2.5浓度预测 - 2D CNN模型 (GPU加速版本)
使用2D卷积神经网络进行时间序列预测

特点:
- 2D CNN架构，将时间窗口和特征作为二维图像处理
- GPU加速训练，支持混合精度训练（AMP）
- 动态批次大小调整，根据GPU显存自动优化
- 使用30天历史数据预测当天PM2.5浓度
- 支持超参数优化（贝叶斯优化）
- 梯度×输入方法分析特征重要性
- 完整的模型评估和可视化

数据来源:
- 污染数据: Benchmark数据集 (PM2.5, PM10, SO2, NO2, CO, O3)
- 气象数据: ERA5再分析数据

注意: 此版本需要GPU支持。如需使用CPU，请运行CNN-CPU.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import pickle
from pathlib import Path
import glob
import multiprocessing
import time

warnings.filterwarnings('ignore')

# PyTorch相关
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# 获取CPU核心数
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

# 尝试导入tqdm进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("提示: tqdm未安装，进度显示将使用简化版本。")
    print("      可使用 'pip install tqdm' 安装以获得更好的进度条显示。")

# 机器学习库
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 贝叶斯优化 (可选)
try:
    from bayes_opt import BayesianOptimization
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    print("提示: bayesian-optimization未安装，将使用网格搜索。")
    print("      可使用 'pip install bayesian-optimization' 安装以启用贝叶斯优化。")
    BAYESIAN_OPT_AVAILABLE = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

print("=" * 80)
print("北京PM2.5浓度预测 - 2D CNN模型 (GPU加速版本)")
print("=" * 80)

# ============================== 第1部分: GPU检测和配置 ==============================
print("\n检测GPU环境...")

# 检测GPU
if not torch.cuda.is_available():
    print("\n" + "=" * 80)
    print("❌ 错误: 未检测到可用的GPU！")
    print("=" * 80)
    print("\n此版本需要GPU支持才能运行。")
    print("\n可能的原因:")
    print("  1. 系统中没有NVIDIA GPU")
    print("  2. CUDA未正确安装")
    print("  3. PyTorch未安装GPU版本")
    print("\n解决方案:")
    print("  - 如需使用CPU运行，请使用: CNN-CPU.py")
    print("  - 如需安装CUDA和PyTorch GPU版本，请访问: https://pytorch.org/")
    print("=" * 80)
    import sys
    sys.exit(1)

# 启用CUDA
DEVICE = torch.device('cuda')
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 启用确定性算法（可能会略微降低性能，但结果可复现）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 获取GPU信息
gpu_name = torch.cuda.get_device_name(0)
gpu_props = torch.cuda.get_device_properties(0)
total_memory_gb = gpu_props.total_memory / 1e9
compute_capability = f"{gpu_props.major}.{gpu_props.minor}"

print(f"\n✓ GPU检测成功！")
print(f"  设备名称: {gpu_name}")
print(f"  显存容量: {total_memory_gb:.2f} GB")
print(f"  计算能力: {compute_capability}")
print(f"  CUDA版本: {torch.version.cuda}")
print(f"  PyTorch版本: {torch.__version__}")

# ============================== GPU显存管理函数 ==============================
def print_gpu_memory(prefix=""):
    """打印GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"{prefix}GPU显存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB, 总容量={total:.2f}GB, 使用率={allocated/total*100:.1f}%")

def clear_gpu_memory():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ============================== 动态批次大小调整 ==============================
def get_optimal_batch_size(model_class, window_size, num_features, 
                           sample_data_shape, device, min_batch=16, max_batch=256):
    """
    根据GPU显存自动确定最优批次大小
    
    Args:
        model_class: 模型类
        window_size: 时间窗口大小
        num_features: 特征数量
        sample_data_shape: 样本数据形状
        device: 计算设备
        min_batch: 最小批次大小
        max_batch: 最大批次大小
    
    Returns:
        optimal_batch_size: 最优批次大小
    """
    print("\n正在测试最优批次大小...")
    clear_gpu_memory()
    
    # 创建测试模型
    test_model = model_class(
        window_size=window_size,
        num_features=num_features,
        num_conv_layers=3,
        base_filters=32,
        kernel_size=3,
        dropout_rate=0.3
    ).to(device)
    
    # 创建测试数据
    test_X = torch.randn(1, window_size, num_features).to(device)
    
    optimal_batch = min_batch
    current_batch = min_batch
    
    while current_batch <= max_batch:
        try:
            clear_gpu_memory()
            
            # 测试当前批次大小
            batch_X = test_X.repeat(current_batch, 1, 1)
            
            # 前向传播
            with autocast():
                _ = test_model(batch_X)
            
            # 如果成功，更新最优批次
            optimal_batch = current_batch
            print(f"  批次大小 {current_batch}: ✓ 通过")
            
            # 增加批次大小
            current_batch *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  批次大小 {current_batch}: ✗ 显存不足")
                break
            else:
                raise e
    
    # 使用80%的最大可用批次作为安全值
    optimal_batch = int(optimal_batch * 0.8)
    optimal_batch = max(min_batch, optimal_batch)
    
    # 清理测试模型
    del test_model, test_X
    if 'batch_X' in locals():
        del batch_X
    clear_gpu_memory()
    
    print(f"\n推荐批次大小: {optimal_batch}")
    return optimal_batch

# ============================== 第2部分: 配置和路径设置 ==============================
print("\n配置参数...")

# 数据路径
pollution_all_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)'
era5_path = r'C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV'

# 输出路径
output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)

# 模型保存路径
model_dir = Path('./models')
model_dir.mkdir(exist_ok=True)

# 日期范围
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

# 北京地理范围
beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

# 污染物列表
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# ERA5变量
era5_vars = [
    'd2m', 't2m', 'u10', 'v10', 'u100', 'v100',  # 温度、风速
    'blh', 'sp', 'tcwv',  # 边界层高度、气压、水汽
    'tp', 'avg_tprate',  # 降水
    'tisr', 'str',  # 辐射
    'cvh', 'cvl',  # 云覆盖
    'mn2t', 'sd', 'lsm'  # 其他
]

# CNN特定参数
WINDOW_SIZE = 30  # 使用过去30天数据
BATCH_SIZE = None  # 将自动确定

print(f"\n数据时间范围: {start_date.date()} 至 {end_date.date()}")
print(f"目标变量: PM2.5浓度")
print(f"时间窗口大小: {WINDOW_SIZE}天")
print(f"设备: {DEVICE}")
print(f"混合精度训练: 已启用 (AMP)")
print(f"输出目录: {output_dir}")
print(f"模型保存目录: {model_dir}")
print(f"CPU核心数: {CPU_COUNT}, 数据加载线程: 4")

# ============================== 第3部分: 数据加载函数 ==============================
def daterange(start, end):
    """生成日期序列"""
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def find_file(base_path, date_str, prefix):
    """查找指定日期的文件"""
    filename = f"{prefix}_{date_str}.csv"
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def read_pollution_day(date):
    """读取单日污染数据"""
    date_str = date.strftime('%Y%m%d')
    all_file = find_file(pollution_all_path, date_str, 'beijing_all')
    extra_file = find_file(pollution_extra_path, date_str, 'beijing_extra')
    
    if not all_file or not extra_file:
        return None
    
    try:
        df_all = pd.read_csv(all_file, encoding='utf-8', on_bad_lines='skip')
        df_extra = pd.read_csv(extra_file, encoding='utf-8', on_bad_lines='skip')
        
        # 过滤掉24小时平均和AQI
        df_all = df_all[~df_all['type'].str.contains('_24h|AQI', na=False)]
        df_extra = df_extra[~df_extra['type'].str.contains('_24h', na=False)]
        
        # 合并
        df_poll = pd.concat([df_all, df_extra], ignore_index=True)
        
        # 转换为长格式
        df_poll = df_poll.melt(id_vars=['date', 'hour', 'type'], 
                                var_name='station', value_name='value')
        df_poll['value'] = pd.to_numeric(df_poll['value'], errors='coerce')
        
        # 删除负值和异常值
        df_poll = df_poll[df_poll['value'] >= 0]
        
        # 按日期和类型聚合（所有站点平均）
        df_daily = df_poll.groupby(['date', 'type'])['value'].mean().reset_index()
        
        # 转换为宽格式
        df_daily = df_daily.pivot(index='date', columns='type', values='value')
        
        # 将索引转换为datetime格式
        df_daily.index = pd.to_datetime(df_daily.index, format='%Y%m%d', errors='coerce')
        
        # 只保留需要的污染物
        df_daily = df_daily[[col for col in pollutants if col in df_daily.columns]]
        
        return df_daily
    except Exception as e:
        return None

def read_all_pollution():
    """并行读取所有污染数据"""
    print("\n正在加载污染数据...")
    print(f"使用 {MAX_WORKERS} 个并行工作线程")
    dates = list(daterange(start_date, end_date))
    pollution_dfs = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, date): date for date in dates}
        
        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="加载污染数据", unit="天"):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
        else:
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
                if i % 500 == 0 or i == len(futures):
                    print(f"  已处理 {i}/{len(futures)} 天 ({i/len(futures)*100:.1f}%)")
    
    if pollution_dfs:
        print(f"  成功读取 {len(pollution_dfs)}/{len(dates)} 天的数据")
        print("  正在合并数据...")
        df_poll_all = pd.concat(pollution_dfs)
        df_poll_all.ffill(inplace=True)
        df_poll_all.fillna(df_poll_all.mean(), inplace=True)
        print(f"污染数据加载完成，形状: {df_poll_all.shape}")
        return df_poll_all
    return pd.DataFrame()

def read_era5_month(year, month):
    """读取单月ERA5数据"""
    month_str = f"{year}{month:02d}"
    all_files = glob.glob(os.path.join(era5_path, "**", f"*{month_str}*.csv"), recursive=True)
    
    if not all_files:
        return None
    
    monthly_data = None
    loaded_vars = []
    
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', 
                            low_memory=False, comment='#')
            
            if df.empty or 'time' not in df.columns:
                continue
            
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.dropna(subset=['time'])
            
            if len(df) == 0:
                continue
            
            if 'latitude' in df.columns and 'longitude' in df.columns:
                df = df[(df['latitude'] >= beijing_lats.min()) & 
                       (df['latitude'] <= beijing_lats.max()) &
                       (df['longitude'] >= beijing_lons.min()) & 
                       (df['longitude'] <= beijing_lons.max())]
                
                if len(df) == 0:
                    continue
            
            if 'expver' in df.columns:
                if '0001' in df['expver'].values:
                    df = df[df['expver'] == '0001']
                else:
                    first_expver = df['expver'].iloc[0]
                    df = df[df['expver'] == first_expver]
            
            df['date'] = df['time'].dt.date
            avail_vars = [v for v in era5_vars if v in df.columns]
            
            if not avail_vars:
                continue
            
            for col in avail_vars:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df_daily = df.groupby('date')[avail_vars].mean().reset_index()
            df_daily.set_index('date', inplace=True)
            df_daily.index = pd.to_datetime(df_daily.index)
            
            if monthly_data is None:
                monthly_data = df_daily
            else:
                monthly_data = monthly_data.join(df_daily, how='outer')
            
            loaded_vars.extend(avail_vars)
            
        except Exception as e:
            continue
    
    if monthly_data is not None and not monthly_data.empty:
        print(f"  成功读取: {year}-{month:02d}, 日数: {len(monthly_data)}, 变量数: {len(loaded_vars)}")
        return monthly_data
    else:
        return None

def read_all_era5():
    """并行读取所有ERA5数据"""
    print("\n正在加载气象数据...")
    print(f"使用 {MAX_WORKERS} 个并行工作线程")
    print(f"气象数据目录: {era5_path}")
    
    if os.path.exists(era5_path):
        all_csv = glob.glob(os.path.join(era5_path, "**", "*.csv"), recursive=True)
        print(f"找到 {len(all_csv)} 个CSV文件")
        if all_csv:
            print(f"示例文件: {[os.path.basename(f) for f in all_csv[:5]]}")
    
    era5_dfs = []
    years = range(2015, 2025)
    months = range(1, 13)
    
    month_tasks = [(year, month) for year in years for month in months 
                   if not (year == 2024 and month > 12)]
    total_months = len(month_tasks)
    print(f"尝试加载 {total_months} 个月的数据...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_era5_month, year, month): (year, month) 
                  for year, month in month_tasks}
        
        successful_reads = 0
        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="加载气象数据", unit="月"):
                result = future.result()
                if result is not None and not result.empty:
                    era5_dfs.append(result)
                    successful_reads += 1
        else:
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None and not result.empty:
                    era5_dfs.append(result)
                    successful_reads += 1
                if i % 20 == 0 or i == len(futures):
                    print(f"  进度: {i}/{len(futures)} 个月 (成功: {successful_reads}, {i/len(futures)*100:.1f}%)")
        
        print(f"  总计成功读取: {successful_reads}/{len(futures)} 个月")
    
    if era5_dfs:
        print("\n正在合并气象数据...")
        df_era5_all = pd.concat(era5_dfs, axis=0)
        print("  去重处理...")
        df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep='first')]
        print("  排序处理...")
        df_era5_all.sort_index(inplace=True)
        
        print(f"合并后形状: {df_era5_all.shape}")
        print(f"时间范围: {df_era5_all.index.min()} 至 {df_era5_all.index.max()}")
        
        print("  处理缺失值...")
        initial_na = df_era5_all.isna().sum().sum()
        df_era5_all.ffill(inplace=True)
        df_era5_all.bfill(inplace=True)
        df_era5_all.fillna(df_era5_all.mean(), inplace=True)
        final_na = df_era5_all.isna().sum().sum()
        
        print(f"缺失值处理: {initial_na} -> {final_na}")
        print(f"气象数据加载完成，形状: {df_era5_all.shape}")
        
        return df_era5_all
    else:
        print("\n❌ 错误: 没有成功加载任何气象数据文件！")
        return pd.DataFrame()

# ============================== 第4部分: 特征工程 ==============================
def create_features(df):
    """创建额外特征"""
    df_copy = df.copy()
    
    # 1. 风速特征
    if 'u10' in df_copy and 'v10' in df_copy:
        df_copy['wind_speed_10m'] = np.sqrt(df_copy['u10']**2 + df_copy['v10']**2)
        df_copy['wind_dir_10m'] = np.arctan2(df_copy['v10'], df_copy['u10']) * 180 / np.pi
        df_copy['wind_dir_10m'] = (df_copy['wind_dir_10m'] + 360) % 360
    
    if 'u100' in df_copy and 'v100' in df_copy:
        df_copy['wind_speed_100m'] = np.sqrt(df_copy['u100']**2 + df_copy['v100']**2)
        df_copy['wind_dir_100m'] = np.arctan2(df_copy['v100'], df_copy['u100']) * 180 / np.pi
        df_copy['wind_dir_100m'] = (df_copy['wind_dir_100m'] + 360) % 360
    
    # 2. 时间特征
    df_copy['year'] = df_copy.index.year
    df_copy['month'] = df_copy.index.month
    df_copy['day'] = df_copy.index.day
    df_copy['day_of_year'] = df_copy.index.dayofyear
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['week_of_year'] = df_copy.index.isocalendar().week
    
    # 季节特征
    df_copy['season'] = df_copy['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )
    
    # 是否供暖季
    df_copy['is_heating_season'] = ((df_copy['month'] >= 11) | (df_copy['month'] <= 3)).astype(int)
    
    # 3. 温度相关特征
    if 't2m' in df_copy and 'd2m' in df_copy:
        df_copy['temp_dewpoint_diff'] = df_copy['t2m'] - df_copy['d2m']
    
    # 4. 滞后特征
    if 'PM2.5' in df_copy:
        df_copy['PM2.5_lag1'] = df_copy['PM2.5'].shift(1)
        df_copy['PM2.5_lag3'] = df_copy['PM2.5'].shift(3)
        df_copy['PM2.5_lag7'] = df_copy['PM2.5'].shift(7)
        
        df_copy['PM2.5_ma3'] = df_copy['PM2.5'].rolling(window=3, min_periods=1).mean()
        df_copy['PM2.5_ma7'] = df_copy['PM2.5'].rolling(window=7, min_periods=1).mean()
        df_copy['PM2.5_ma30'] = df_copy['PM2.5'].rolling(window=30, min_periods=1).mean()
    
    # 5. 相对湿度估算
    if 't2m' in df_copy and 'd2m' in df_copy:
        df_copy['relative_humidity'] = 100 * np.exp((17.625 * (df_copy['d2m'] - 273.15)) / 
                                                      (243.04 + (df_copy['d2m'] - 273.15))) / \
                                        np.exp((17.625 * (df_copy['t2m'] - 273.15)) / 
                                               (243.04 + (df_copy['t2m'] - 273.15)))
        df_copy['relative_humidity'] = df_copy['relative_humidity'].clip(0, 100)
    
    # 6. 风向分类
    if 'wind_dir_10m' in df_copy:
        df_copy['wind_dir_category'] = pd.cut(df_copy['wind_dir_10m'], 
                                                bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                                                labels=[0, 1, 2, 3, 4, 5, 6, 7],
                                                include_lowest=True).astype(int)
    
    return df_copy

# ============================== 第5部分: 数据加载和预处理 ==============================
print("\n" + "=" * 80)
print("第1步: 数据加载和预处理")
print("=" * 80)

df_pollution = read_all_pollution()
df_era5 = read_all_era5()

# 检查数据加载情况
print("\n数据加载检查:")
print(f"  污染数据形状: {df_pollution.shape}")
print(f"  气象数据形状: {df_era5.shape}")

if df_pollution.empty:
    print("\n⚠️ 警告: 污染数据为空！请检查数据路径和文件。")
    import sys
    sys.exit(1)

if df_era5.empty:
    print("\n⚠️ 警告: 气象数据为空！请检查数据路径和文件。")
    import sys
    sys.exit(1)

# 确保索引是datetime类型
df_pollution.index = pd.to_datetime(df_pollution.index)
df_era5.index = pd.to_datetime(df_era5.index)

print(f"  污染数据时间范围: {df_pollution.index.min()} 至 {df_pollution.index.max()}")
print(f"  气象数据时间范围: {df_era5.index.min()} 至 {df_era5.index.max()}")

# 合并数据
print("\n正在合并数据...")
df_combined = df_pollution.join(df_era5, how='inner')

if df_combined.empty:
    print("\n❌ 错误: 数据合并后为空！")
    import sys
    sys.exit(1)

# 创建特征
print("\n正在创建特征...")
df_combined = create_features(df_combined)

# 清理数据
print("\n正在清理数据...")
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
initial_rows = len(df_combined)
df_combined.dropna(inplace=True)
final_rows = len(df_combined)
print(f"删除了 {initial_rows - final_rows} 行包含缺失值的数据")

print(f"\n合并后数据形状: {df_combined.shape}")
print(f"时间范围: {df_combined.index.min().date()} 至 {df_combined.index.max().date()}")
print(f"样本数: {len(df_combined)}")
print(f"特征数: {df_combined.shape[1]}")

print(f"\n特征列表（前20个）:")
for i, col in enumerate(df_combined.columns[:20], 1):
    print(f"  {i}. {col}")
if len(df_combined.columns) > 20:
    print(f"  ... 还有 {len(df_combined.columns) - 20} 个特征")

# ============================== 第6部分: CNN数据准备 ==============================
print("\n" + "=" * 80)
print("第2步: CNN数据准备（滑动窗口）")
print("=" * 80)

# 定义目标变量
target = 'PM2.5'

# 排除的列
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']

# 选择数值型特征
numeric_features = [col for col in df_combined.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_cols]

print(f"\n选择的特征数量: {len(numeric_features)}")
print(f"目标变量: {target}")

# 准备数据
X_raw = df_combined[numeric_features].values
y_raw = df_combined[target].values

print(f"\n原始数据形状:")
print(f"  X: {X_raw.shape}")
print(f"  y: {y_raw.shape}")

# 标准化特征
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

print(f"\n标准化后数据形状:")
print(f"  X: {X_scaled.shape}")
print(f"  y: {y_scaled.shape}")

# 创建滑动窗口数据集
def create_sliding_windows(X, y, window_size):
    """
    创建滑动窗口数据集
    
    Args:
        X: 特征数据 [samples, features]
        y: 目标数据 [samples]
        window_size: 窗口大小
    
    Returns:
        X_windows: [num_windows, window_size, features]
        y_windows: [num_windows]
    """
    num_samples = len(X)
    num_windows = num_samples - window_size + 1
    num_features = X.shape[1]
    
    X_windows = np.zeros((num_windows, window_size, num_features))
    y_windows = np.zeros(num_windows)
    
    for i in range(num_windows):
        X_windows[i] = X[i:i+window_size]
        y_windows[i] = y[i+window_size-1]  # 预测窗口最后一天的PM2.5
    
    return X_windows, y_windows

print(f"\n创建 {WINDOW_SIZE} 天滑动窗口...")
X_windows, y_windows = create_sliding_windows(X_scaled, y_scaled, WINDOW_SIZE)

print(f"滑动窗口数据形状:")
print(f"  X_windows: {X_windows.shape}  # [样本数, 时间步, 特征数]")
print(f"  y_windows: {y_windows.shape}")

# 保存特征名称和日期索引（用于后续分析）
feature_names = numeric_features
date_index = df_combined.index[WINDOW_SIZE-1:]

print(f"\nPM2.5统计信息:")
print(f"  均值: {y_raw.mean():.2f} μg/m³")
print(f"  标准差: {y_raw.std():.2f} μg/m³")
print(f"  最小值: {y_raw.min():.2f} μg/m³")
print(f"  最大值: {y_raw.max():.2f} μg/m³")
print(f"  中位数: {np.median(y_raw):.2f} μg/m³")

# ============================== 第7部分: 2D CNN模型定义 ==============================
print("\n" + "=" * 80)
print("第3步: 定义2D CNN模型")
print("=" * 80)

class PM25CNN2D(nn.Module):
    """
    2D CNN模型用于PM2.5预测（GPU优化版本）
    输入: [batch, 1, window_size, num_features] 
    输出: [batch] (单个PM2.5值)
    """
    def __init__(self, window_size, num_features, num_conv_layers=3, 
                 base_filters=32, kernel_size=3, dropout_rate=0.3):
        super(PM25CNN2D, self).__init__()
        
        self.window_size = window_size
        self.num_features = num_features
        
        # 卷积层
        conv_layers = []
        in_channels = 1
        
        for i in range(num_conv_layers):
            out_channels = base_filters * (2 ** i)
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # 计算卷积后的特征图大小
        self.feature_size = self._get_conv_output_size()
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(64, 1)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _get_conv_output_size(self):
        """计算卷积层输出的特征尺寸"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.window_size, self.num_features)
            dummy_output = self.conv_layers(dummy_input)
            return int(np.prod(dummy_output.shape[1:]))
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [batch, window_size, num_features]
        # 添加通道维度
        x = x.unsqueeze(1)  # [batch, 1, window_size, num_features]
        
        # 卷积层
        x = self.conv_layers(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc_layers(x)
        
        return x.squeeze()

num_features = X_windows.shape[2]
print(f"\n模型输入维度:")
print(f"  窗口大小: {WINDOW_SIZE}")
print(f"  特征数: {num_features}")

# 确定最优批次大小
if BATCH_SIZE is None:
    BATCH_SIZE = get_optimal_batch_size(
        PM25CNN2D, WINDOW_SIZE, num_features, 
        X_windows.shape, DEVICE
    )
print(f"\n最终批次大小: {BATCH_SIZE}")

# ============================== 第8部分: PyTorch数据集和数据加载器 ==============================
print("\n" + "=" * 80)
print("第4步: 创建PyTorch数据集")
print("=" * 80)

class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 按时间顺序划分：训练集70%，验证集15%，测试集15%
n_samples = len(X_windows)
train_size = int(n_samples * 0.70)
val_size = int(n_samples * 0.15)

X_train = X_windows[:train_size]
X_val = X_windows[train_size:train_size + val_size]
X_test = X_windows[train_size + val_size:]

y_train = y_windows[:train_size]
y_val = y_windows[train_size:train_size + val_size]
y_test = y_windows[train_size + val_size:]

print(f"\n训练集: {len(X_train)} 样本 ({len(X_train)/n_samples*100:.1f}%)")
print(f"  时间范围: {date_index[0].date()} 至 {date_index[train_size-1].date()}")

print(f"\n验证集: {len(X_val)} 样本 ({len(X_val)/n_samples*100:.1f}%)")
print(f"  时间范围: {date_index[train_size].date()} 至 {date_index[train_size+val_size-1].date()}")

print(f"\n测试集: {len(X_test)} 样本 ({len(X_test)/n_samples*100:.1f}%)")
print(f"  时间范围: {date_index[train_size+val_size].date()} 至 {date_index[-1].date()}")

# 创建数据集和数据加载器（GPU优化配置）
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)

# 启用pin_memory和多进程加载
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=4, pin_memory=True, persistent_workers=True)

print(f"\n数据加载器创建完成:")
print(f"  训练批次数: {len(train_loader)}")
print(f"  验证批次数: {len(val_loader)}")
print(f"  测试批次数: {len(test_loader)}")
print(f"  多进程加载: 4 workers")
print(f"  Pin Memory: 已启用")

# ============================== 第9部分: 训练和评估函数（AMP优化） ==============================
def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """训练一个epoch（混合精度训练）"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
        
        # 混合精度训练
        with autocast():
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
        
        # 缩放损失并反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * len(X_batch)
    
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    """验证模型（混合精度）"""
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            with autocast():
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
            
            total_loss += loss.item() * len(X_batch)
            predictions.extend(y_pred.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    return avg_loss, predictions, actuals

def train_model(model, train_loader, val_loader, criterion, optimizer, scaler,
                num_epochs, device, patience=20, verbose=True):
    """训练模型（带早停和混合精度）"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    print(f"\n开始训练 {num_epochs} 个epochs...")
    print_gpu_memory("训练前 ")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, _, _ = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
            if (epoch + 1) % 30 == 0:
                print_gpu_memory("  ")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n早停触发于 epoch {epoch+1}")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    total_time = time.time() - start_time
    print(f"训练完成！最佳模型在 epoch {best_epoch}，验证损失: {best_val_loss:.4f}")
    print(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print_gpu_memory("训练后 ")
    
    return train_losses, val_losses, best_epoch

def evaluate_model(y_true, y_pred, dataset_name):
    """评估模型性能"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # 避免除零错误
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    return {
        'Dataset': dataset_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

# ============================== 第10部分: 基础模型训练 ==============================
print("\n" + "=" * 80)
print("第5步: 训练基础CNN模型")
print("=" * 80)

# 基础模型参数
basic_params = {
    'num_conv_layers': 3,
    'base_filters': 32,
    'kernel_size': 3,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'num_epochs': 200,
    'patience': 20
}

print("\n基础模型参数:")
for key, value in basic_params.items():
    print(f"  {key}: {value}")

# 清理GPU缓存
clear_gpu_memory()

# 创建基础模型
model_basic = PM25CNN2D(
    window_size=WINDOW_SIZE,
    num_features=num_features,
    num_conv_layers=basic_params['num_conv_layers'],
    base_filters=basic_params['base_filters'],
    kernel_size=basic_params['kernel_size'],
    dropout_rate=basic_params['dropout_rate']
).to(DEVICE)

# 统计模型参数
total_params = sum(p.numel() for p in model_basic.parameters())
trainable_params = sum(p.numel() for p in model_basic.parameters() if p.requires_grad)
print(f"\n模型参数统计:")
print(f"  总参数: {total_params:,}")
print(f"  可训练参数: {trainable_params:,}")

# 定义损失函数、优化器和混合精度缩放器
criterion = nn.MSELoss()
optimizer_basic = optim.Adam(model_basic.parameters(), lr=basic_params['learning_rate'])
scaler_basic = GradScaler()

# 训练基础模型
train_losses_basic, val_losses_basic, best_epoch_basic = train_model(
    model_basic, train_loader, val_loader, criterion, optimizer_basic, scaler_basic,
    num_epochs=basic_params['num_epochs'],
    device=DEVICE,
    patience=basic_params['patience'],
    verbose=True
)

print(f"\n✓ 基础模型训练完成")
print(f"  最佳epoch: {best_epoch_basic}")
print(f"  最终训练损失: {train_losses_basic[best_epoch_basic-1]:.4f}")
print(f"  最终验证损失: {val_losses_basic[best_epoch_basic-1]:.4f}")

# 评估基础模型
print("\n评估基础模型...")

_, y_train_pred_basic_scaled, y_train_actual_scaled = validate(model_basic, train_loader, criterion, DEVICE)
_, y_val_pred_basic_scaled, y_val_actual_scaled = validate(model_basic, val_loader, criterion, DEVICE)
_, y_test_pred_basic_scaled, y_test_actual_scaled = validate(model_basic, test_loader, criterion, DEVICE)

# 反标准化
y_train_pred_basic = scaler_y.inverse_transform(y_train_pred_basic_scaled.reshape(-1, 1)).flatten()
y_train_actual_basic = scaler_y.inverse_transform(y_train_actual_scaled.reshape(-1, 1)).flatten()

y_val_pred_basic = scaler_y.inverse_transform(y_val_pred_basic_scaled.reshape(-1, 1)).flatten()
y_val_actual_basic = scaler_y.inverse_transform(y_val_actual_scaled.reshape(-1, 1)).flatten()

y_test_pred_basic = scaler_y.inverse_transform(y_test_pred_basic_scaled.reshape(-1, 1)).flatten()
y_test_actual_basic = scaler_y.inverse_transform(y_test_actual_scaled.reshape(-1, 1)).flatten()

# 计算性能指标
results_basic = []
results_basic.append(evaluate_model(y_train_actual_basic, y_train_pred_basic, 'Train'))
results_basic.append(evaluate_model(y_val_actual_basic, y_val_pred_basic, 'Validation'))
results_basic.append(evaluate_model(y_test_actual_basic, y_test_pred_basic, 'Test'))

results_basic_df = pd.DataFrame(results_basic)
print("\n基础模型性能:")
print(results_basic_df.to_string(index=False))

# ============================== 第11部分: 超参数优化（GPU优化） ==============================
print("\n" + "=" * 80)
print("第6步: 超参数优化")
print("=" * 80)

if BAYESIAN_OPT_AVAILABLE:
    print("\n使用贝叶斯优化进行超参数搜索...")
    
    def cnn_evaluate(num_conv_layers, base_filters, kernel_size, 
                     learning_rate, dropout_rate):
        """贝叶斯优化的目标函数（GPU优化版）"""
        try:
            # 参数转换
            num_conv_layers = int(num_conv_layers)
            base_filters = int(base_filters)
            kernel_size = int(kernel_size)
            
            # 清理GPU缓存
            clear_gpu_memory()
            
            # 创建模型
            model_temp = PM25CNN2D(
                window_size=WINDOW_SIZE,
                num_features=num_features,
                num_conv_layers=num_conv_layers,
                base_filters=base_filters,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate
            ).to(DEVICE)
            
            # 训练
            optimizer_temp = optim.Adam(model_temp.parameters(), lr=learning_rate)
            scaler_temp = GradScaler()
            
            _, _, _ = train_model(
                model_temp, train_loader, val_loader, criterion, optimizer_temp, scaler_temp,
                num_epochs=100, device=DEVICE, patience=15, verbose=False
            )
            
            # 评估
            val_loss, _, _ = validate(model_temp, val_loader, criterion, DEVICE)
            
            # 清理
            del model_temp, optimizer_temp, scaler_temp
            clear_gpu_memory()
            
            # 返回负损失（贝叶斯优化是最大化）
            return -val_loss
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  显存不足，跳过此参数组合")
                clear_gpu_memory()
                return -999999  # 返回很差的分数
            else:
                raise e
    
    # 参数搜索空间
    pbounds = {
        'num_conv_layers': (2, 4),
        'base_filters': (16, 64),
        'kernel_size': (3, 5),
        'learning_rate': (0.0001, 0.01),
        'dropout_rate': (0.2, 0.5)
    }
    
    optimizer_bo = BayesianOptimization(
        f=cnn_evaluate,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    optimizer_bo.maximize(init_points=5, n_iter=10)
    
    # 获取最佳参数
    best_params = optimizer_bo.max['params']
    best_params['num_conv_layers'] = int(best_params['num_conv_layers'])
    best_params['base_filters'] = int(best_params['base_filters'])
    best_params['kernel_size'] = int(best_params['kernel_size'])
    
    print(f"\n最佳参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  最佳验证损失: {-optimizer_bo.max['target']:.4f}")
    
else:
    # 网格搜索（简化版，GPU优化）
    print("\n使用网格搜索进行超参数优化...")
    
    param_grid = {
        'num_conv_layers': [2, 3],
        'base_filters': [32, 64],
        'kernel_size': [3, 5],
        'learning_rate': [0.001, 0.005],
        'dropout_rate': [0.3, 0.4]
    }
    
    from itertools import product
    total_combinations = int(np.prod([len(v) for v in param_grid.values()]))
    print(f"总共 {total_combinations} 种参数组合")
    
    best_val_loss_grid = float('inf')
    best_params = {}
    
    for i, combo in enumerate(product(*param_grid.values()), 1):
        print(f"\n测试组合 {i}/{total_combinations}...")
        params_test = dict(zip(param_grid.keys(), combo))
        
        try:
            # 清理GPU缓存
            clear_gpu_memory()
            
            model_temp = PM25CNN2D(
                window_size=WINDOW_SIZE,
                num_features=num_features,
                num_conv_layers=params_test['num_conv_layers'],
                base_filters=params_test['base_filters'],
                kernel_size=params_test['kernel_size'],
                dropout_rate=params_test['dropout_rate']
            ).to(DEVICE)
            
            optimizer_temp = optim.Adam(model_temp.parameters(), lr=params_test['learning_rate'])
            scaler_temp = GradScaler()
            
            _, _, _ = train_model(
                model_temp, train_loader, val_loader, criterion, optimizer_temp, scaler_temp,
                num_epochs=100, device=DEVICE, patience=15, verbose=False
            )
            
            val_loss, _, _ = validate(model_temp, val_loader, criterion, DEVICE)
            print(f"  验证损失: {val_loss:.4f}")
            
            if val_loss < best_val_loss_grid:
                best_val_loss_grid = val_loss
                best_params = params_test.copy()
            
            # 清理
            del model_temp, optimizer_temp, scaler_temp
            clear_gpu_memory()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  显存不足，跳过此组合")
                clear_gpu_memory()
                continue
            else:
                raise e
    
    print(f"\n最佳参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  最佳验证损失: {best_val_loss_grid:.4f}")

# ============================== 第12部分: 训练优化模型 ==============================
print("\n" + "=" * 80)
print("第7步: 使用最佳参数训练优化模型")
print("=" * 80)

print("\n优化模型参数:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# 清理GPU缓存
clear_gpu_memory()

# 创建优化模型
model_optimized = PM25CNN2D(
    window_size=WINDOW_SIZE,
    num_features=num_features,
    num_conv_layers=best_params['num_conv_layers'],
    base_filters=best_params['base_filters'],
    kernel_size=best_params['kernel_size'],
    dropout_rate=best_params['dropout_rate']
).to(DEVICE)

optimizer_opt = optim.Adam(model_optimized.parameters(), lr=best_params['learning_rate'])
scaler_opt = GradScaler()

# 训练优化模型
train_losses_opt, val_losses_opt, best_epoch_opt = train_model(
    model_optimized, train_loader, val_loader, criterion, optimizer_opt, scaler_opt,
    num_epochs=300, device=DEVICE, patience=30, verbose=True
)

print(f"\n✓ 优化模型训练完成")
print(f"  最佳epoch: {best_epoch_opt}")
print(f"  最终训练损失: {train_losses_opt[best_epoch_opt-1]:.4f}")
print(f"  最终验证损失: {val_losses_opt[best_epoch_opt-1]:.4f}")

# 评估优化模型
print("\n评估优化模型...")

_, y_train_pred_opt_scaled, _ = validate(model_optimized, train_loader, criterion, DEVICE)
_, y_val_pred_opt_scaled, _ = validate(model_optimized, val_loader, criterion, DEVICE)
_, y_test_pred_opt_scaled, _ = validate(model_optimized, test_loader, criterion, DEVICE)

# 反标准化
y_train_pred_opt = scaler_y.inverse_transform(y_train_pred_opt_scaled.reshape(-1, 1)).flatten()
y_val_pred_opt = scaler_y.inverse_transform(y_val_pred_opt_scaled.reshape(-1, 1)).flatten()
y_test_pred_opt = scaler_y.inverse_transform(y_test_pred_opt_scaled.reshape(-1, 1)).flatten()

# 计算性能指标
results_opt = []
results_opt.append(evaluate_model(y_train_actual_basic, y_train_pred_opt, 'Train'))
results_opt.append(evaluate_model(y_val_actual_basic, y_val_pred_opt, 'Validation'))
results_opt.append(evaluate_model(y_test_actual_basic, y_test_pred_opt, 'Test'))

results_opt_df = pd.DataFrame(results_opt)
print("\n优化模型性能:")
print(results_opt_df.to_string(index=False))

# ============================== 第13部分: 模型比较 ==============================
print("\n" + "=" * 80)
print("第8步: 模型性能比较")
print("=" * 80)

# 合并结果
results_basic_df['Model'] = 'CNN_Basic'
results_opt_df['Model'] = 'CNN_Optimized'
all_results = pd.concat([results_basic_df, results_opt_df])

# 重新排列列顺序
all_results = all_results[['Model', 'Dataset', 'R²', 'RMSE', 'MAE', 'MAPE']]

print("\n所有模型性能对比:")
print(all_results.to_string(index=False))

# 测试集性能对比
test_results = all_results[all_results['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\n测试集性能排名:")
print(test_results.to_string(index=False))

# 性能提升
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

print(f"\n优化效果:")
print(f"  R²提升: {r2_improvement:.2f}%")
print(f"  RMSE降低: {rmse_improvement:.2f}%")

# ============================== 第14部分: 特征重要性分析（GPU优化） ==============================
print("\n" + "=" * 80)
print("第9步: 特征重要性分析（梯度×输入方法）")
print("=" * 80)

def compute_gradient_importance(model, X_samples, device, num_samples=500, batch_size=100):
    """
    使用梯度×输入方法计算特征重要性（GPU优化，分批处理）
    
    Args:
        model: 训练好的模型
        X_samples: 样本数据 [num_samples, window_size, num_features]
        device: 计算设备
        num_samples: 使用的样本数量
        batch_size: 分批处理的批次大小
    
    Returns:
        feature_importance: [num_features] 每个特征的重要性分数
    """
    model.eval()
    
    # 随机选择样本
    if len(X_samples) > num_samples:
        indices = np.random.choice(len(X_samples), num_samples, replace=False)
        X_samples = X_samples[indices]
    
    # 分批处理以避免显存溢出
    num_batches = (len(X_samples) + batch_size - 1) // batch_size
    importance_sum = None
    
    print(f"  分 {num_batches} 批处理...")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_samples))
        X_batch = X_samples[start_idx:end_idx]
        
        X_tensor = torch.FloatTensor(X_batch).to(device)
        X_tensor.requires_grad = True
        
        # 前向传播（使用混合精度）
        with autocast():
            outputs = model(X_tensor)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=outputs.sum(),
            inputs=X_tensor,
            create_graph=False
        )[0]
        
        # 计算重要性：|梯度 × 输入|
        importance_batch = (gradients * X_tensor).abs()
        
        # 在时间维度和样本维度上求平均
        importance_batch = importance_batch.mean(dim=[0, 1])  # [num_features]
        
        if importance_sum is None:
            importance_sum = importance_batch
        else:
            importance_sum += importance_batch
        
        # 清理
        del X_tensor, outputs, gradients, importance_batch
        
        if (i + 1) % 5 == 0:
            clear_gpu_memory()
    
    # 平均所有批次
    importance = importance_sum / num_batches
    
    clear_gpu_memory()
    
    return importance.detach().cpu().numpy()

print("\n计算特征重要性...")
feature_importance_scores = compute_gradient_importance(
    model_optimized, X_train, DEVICE, num_samples=500, batch_size=100
)

# 归一化重要性分数
feature_importance_scores_norm = (feature_importance_scores / feature_importance_scores.sum()) * 100

# 创建特征重要性DataFrame
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance_scores,
    'Importance_Norm': feature_importance_scores_norm
})

# 排序
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print(f"\nTop 20 重要特征:")
print(feature_importance.head(20)[['Feature', 'Importance_Norm']].to_string(index=False))

# ============================== 第15部分: 可视化 ==============================
print("\n" + "=" * 80)
print("第10步: 生成可视化图表")
print("=" * 80)

# 15.1 训练过程曲线
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 基础模型
axes[0].plot(train_losses_basic, label='训练损失', linewidth=2)
axes[0].plot(val_losses_basic, label='验证损失', linewidth=2)
axes[0].axvline(x=best_epoch_basic-1, color='r', linestyle='--', 
                label=f'最佳epoch({best_epoch_basic})', linewidth=1.5)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('CNN基础模型 - 训练过程', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 优化模型
axes[1].plot(train_losses_opt, label='训练损失', linewidth=2)
axes[1].plot(val_losses_opt, label='验证损失', linewidth=2)
axes[1].axvline(x=best_epoch_opt-1, color='r', linestyle='--',
                label=f'最佳epoch({best_epoch_opt})', linewidth=1.5)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss (MSE)', fontsize=12)
axes[1].set_title('CNN优化模型 - 训练过程', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
print("保存: training_curves.png")
plt.close()

# 15.2 预测vs实际值散点图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

models_data = [
    ('Basic', y_train_pred_basic, y_train_actual_basic, 'Train'),
    ('Basic', y_val_pred_basic, y_val_actual_basic, 'Val'),
    ('Basic', y_test_pred_basic, y_test_actual_basic, 'Test'),
    ('Optimized', y_train_pred_opt, y_train_actual_basic, 'Train'),
    ('Optimized', y_val_pred_opt, y_val_actual_basic, 'Val'),
    ('Optimized', y_test_pred_opt, y_test_actual_basic, 'Test')
]

for idx, (model_name, y_pred, y_true, dataset) in enumerate(models_data):
    row = idx // 3
    col = idx % 3
    
    ax = axes[row, col]
    
    # 散点图
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    
    # 理想预测线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测线')
    
    # 计算指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    ax.set_xlabel('实际PM2.5浓度 (μg/m³)', fontsize=11)
    ax.set_ylabel('预测PM2.5浓度 (μg/m³)', fontsize=11)
    ax.set_title(f'CNN_{model_name} - {dataset}\nR²={r2:.4f}, RMSE={rmse:.2f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
print("保存: prediction_scatter.png")
plt.close()

# 15.3 时间序列预测对比
fig, axes = plt.subplots(2, 1, figsize=(18, 10))

# 测试集索引
test_date_index = date_index[train_size+val_size:]

# 绘制最后300个点
plot_range = min(300, len(y_test_actual_basic))
plot_idx = range(len(y_test_actual_basic) - plot_range, len(y_test_actual_basic))
time_idx = test_date_index[plot_idx]

axes[0].plot(time_idx, y_test_actual_basic[plot_idx], 'k-', label='实际值', 
             linewidth=2, alpha=0.8)
axes[0].plot(time_idx, y_test_pred_basic[plot_idx], 'b--', label='基础模型预测', 
             linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('日期', fontsize=12)
axes[0].set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
axes[0].set_title('CNN基础模型 - 时间序列预测对比（测试集最后300天）', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

axes[1].plot(time_idx, y_test_actual_basic[plot_idx], 'k-', label='实际值', 
             linewidth=2, alpha=0.8)
axes[1].plot(time_idx, y_test_pred_opt[plot_idx], 'g--', label='优化模型预测', 
             linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('日期', fontsize=12)
axes[1].set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
axes[1].set_title('CNN优化模型 - 时间序列预测对比（测试集最后300天）', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(output_dir / 'timeseries_comparison.png', dpi=300, bbox_inches='tight')
print("保存: timeseries_comparison.png")
plt.close()

# 15.4 残差分析
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (model_name, y_pred, y_true, dataset) in enumerate(models_data):
    row = idx // 3
    col = idx % 3
    
    ax = axes[row, col]
    
    residuals = y_true - y_pred
    
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('预测值 (μg/m³)', fontsize=11)
    ax.set_ylabel('残差 (μg/m³)', fontsize=11)
    ax.set_title(f'CNN_{model_name} - {dataset}\n残差均值={residuals.mean():.2f}, 标准差={residuals.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
print("保存: residuals_analysis.png")
plt.close()

# 15.5 特征重要性图
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

top_n = 20
top_features = feature_importance.head(top_n)

ax.barh(range(top_n), top_features['Importance_Norm'], color='steelblue')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['Feature'], fontsize=10)
ax.set_xlabel('重要性 (%)', fontsize=12)
ax.set_title(f'Top {top_n} 重要特征 (梯度×输入方法)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
print("保存: feature_importance.png")
plt.close()

# 15.6 模型性能对比柱状图
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
        axes[i].set_title(f'{metric} 对比\n(越大越好)', fontsize=12, fontweight='bold')
    else:
        axes[i].set_title(f'{metric} 对比\n(越小越好)', fontsize=12, fontweight='bold')
    
    axes[i].grid(True, alpha=0.3, axis='y')
    
    # 显示数值
    for j, v in enumerate(test_results_plot[metric]):
        if metric == 'MAPE':
            axes[i].text(j, v, f'{v:.1f}%', ha='center', va='bottom', 
                         fontsize=10, fontweight='bold')
        else:
            axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom', 
                         fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
print("保存: model_comparison.png")
plt.close()

# 15.7 误差分布直方图
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

errors_basic = y_test_actual_basic - y_test_pred_basic
errors_opt = y_test_actual_basic - y_test_pred_opt

axes[0].hist(errors_basic, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='零误差')
axes[0].set_xlabel('预测误差 (μg/m³)', fontsize=12)
axes[0].set_ylabel('频数', fontsize=12)
axes[0].set_title(f'基础模型 - 预测误差分布\n均值={errors_basic.mean():.2f}, 标准差={errors_basic.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].hist(errors_opt, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='零误差')
axes[1].set_xlabel('预测误差 (μg/m³)', fontsize=12)
axes[1].set_ylabel('频数', fontsize=12)
axes[1].set_title(f'优化模型 - 预测误差分布\n均值={errors_opt.mean():.2f}, 标准差={errors_opt.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
print("保存: error_distribution.png")
plt.close()

# ============================== 第16部分: 保存结果 ==============================
print("\n" + "=" * 80)
print("第11步: 保存结果")
print("=" * 80)

# 保存模型性能
all_results.to_csv(output_dir / 'model_performance.csv', index=False, encoding='utf-8-sig')
print("保存: model_performance.csv")

# 保存特征重要性
feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False, encoding='utf-8-sig')
print("保存: feature_importance.csv")

# 保存最佳参数
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv(output_dir / 'best_parameters.csv', index=False, encoding='utf-8-sig')
print("保存: best_parameters.csv")

# 保存预测结果
predictions_df = pd.DataFrame({
    'Date': test_date_index,
    'Actual': y_test_actual_basic,
    'Prediction_Basic': y_test_pred_basic,
    'Prediction_Optimized': y_test_pred_opt,
    'Error_Basic': y_test_actual_basic - y_test_pred_basic,
    'Error_Optimized': y_test_actual_basic - y_test_pred_opt
})
predictions_df.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
print("保存: predictions.csv")

# 保存模型（PyTorch格式）
torch.save({
    'model_state_dict': model_optimized.state_dict(),
    'optimizer_state_dict': optimizer_opt.state_dict(),
    'scaler_state_dict': scaler_opt.state_dict(),
    'best_epoch': best_epoch_opt,
    'train_losses': train_losses_opt,
    'val_losses': val_losses_opt,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'feature_names': feature_names,
    'hyperparameters': best_params,
    'batch_size': BATCH_SIZE
}, model_dir / 'cnn_optimized_gpu.pth')
print("保存: cnn_optimized_gpu.pth")

# 保存模型架构信息
model_info = {
    'window_size': WINDOW_SIZE,
    'num_features': num_features,
    'total_params': total_params,
    'trainable_params': trainable_params,
    'best_params': best_params,
    'batch_size': BATCH_SIZE,
    'device': 'cuda',
    'amp_enabled': True
}

with open(model_dir / 'model_info_gpu.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("保存: model_info_gpu.pkl")

# ============================== 第17部分: 总结报告 ==============================
print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)

print("\n生成的文件:")
print("\nCSV文件:")
print("  - model_performance.csv       模型性能对比")
print("  - feature_importance.csv      特征重要性")
print("  - best_parameters.csv         最佳参数")
print("  - predictions.csv             预测结果")

print("\n图表文件:")
print("  - training_curves.png         训练过程曲线")
print("  - prediction_scatter.png      预测vs实际散点图")
print("  - timeseries_comparison.png   时间序列对比")
print("  - residuals_analysis.png      残差分析")
print("  - feature_importance.png      特征重要性图")
print("  - model_comparison.png        模型性能对比")
print("  - error_distribution.png      误差分布")

print("\n模型文件:")
print("  - cnn_optimized_gpu.pth       CNN模型（GPU版本，含AMP）")
print("  - model_info_gpu.pkl          模型信息")

# 最佳模型信息
best_model = test_results.iloc[0]
print(f"\n最佳模型: {best_model['Model']}")
print(f"  R² Score: {best_model['R²']:.4f}")
print(f"  RMSE: {best_model['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model['MAPE']:.2f}%")

print("\nTop 5 最重要特征:")
for i, (idx, row) in enumerate(feature_importance.head(5).iterrows(), 1):
    print(f"  {i}. {row['Feature']}: {row['Importance_Norm']:.2f}%")

print(f"\n模型架构:")
print(f"  时间窗口: {WINDOW_SIZE} 天")
print(f"  特征数: {num_features}")
print(f"  批次大小: {BATCH_SIZE}")
print(f"  总参数: {total_params:,}")
print(f"  可训练参数: {trainable_params:,}")

print(f"\nGPU加速信息:")
print(f"  设备: {gpu_name}")
print(f"  混合精度训练: 已启用 (FP16)")
print_gpu_memory("  最终")

print("\n" + "=" * 80)
print("CNN PM2.5浓度预测完成！(GPU加速版)")
print("=" * 80)

