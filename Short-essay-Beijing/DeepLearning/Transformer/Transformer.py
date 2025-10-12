"""
北京PM2.5浓度预测 - Transformer模型
使用时间序列Transformer进行多步预测

特点:
- 时间序列专用Transformer架构
- 多头自注意力机制
- 位置编码和时间编码
- 多步预测（预测未来7天）
- 完整超参数优化
- SHAP特征重要性分析
- 全流程自动化执行

数据来源:
- 污染数据: Benchmark数据集 (PM2.5, PM10, SO2, NO2, CO, O3)
- 气象数据: ERA5再分析数据
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
from itertools import product
import time

warnings.filterwarnings('ignore')

# PyTorch相关
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

# SHAP (可选)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("提示: shap未安装，将使用排列重要性。")
    print("      可使用 'pip install shap' 安装以启用SHAP分析。")
    SHAP_AVAILABLE = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("北京PM2.5浓度预测 - Transformer模型")
print("=" * 80)
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")

# ============================== 第1部分: 配置和路径设置 ==============================
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

# Transformer特定参数
SEQ_LENGTH = 30  # 输入序列长度（30天）
PRED_LENGTH = 7  # 预测长度（7天）
BATCH_SIZE = 32  # 批量大小

print(f"数据时间范围: {start_date.date()} 至 {end_date.date()}")
print(f"目标变量: PM2.5浓度")
print(f"输入序列长度: {SEQ_LENGTH}天")
print(f"预测长度: {PRED_LENGTH}天")
print(f"批量大小: {BATCH_SIZE}")
print(f"输出目录: {output_dir}")
print(f"模型保存目录: {model_dir}")
print(f"CPU核心数: {CPU_COUNT}, 并行工作线程: {MAX_WORKERS}")

# ============================== 第2部分: 数据加载函数 ==============================
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

# ============================== 第3部分: 特征工程 ==============================
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
        
        # 滚动平均特征
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

# ============================== 第4部分: 数据加载和预处理 ==============================
print("\n" + "=" * 80)
print("第1步: 数据加载和预处理")
print("=" * 80)

df_pollution = read_all_pollution()
df_era5 = read_all_era5()

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

df_pollution.index = pd.to_datetime(df_pollution.index)
df_era5.index = pd.to_datetime(df_era5.index)

print(f"  污染数据时间范围: {df_pollution.index.min()} 至 {df_pollution.index.max()}")
print(f"  气象数据时间范围: {df_era5.index.min()} 至 {df_era5.index.max()}")

print("\n正在合并数据...")
df_combined = df_pollution.join(df_era5, how='inner')

if df_combined.empty:
    print("\n❌ 错误: 数据合并后为空！")
    import sys
    sys.exit(1)

print("\n正在创建特征...")
df_combined = create_features(df_combined)

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

# ============================== 第5部分: 序列数据准备 ==============================
print("\n" + "=" * 80)
print("第2步: 序列数据准备")
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
X = df_combined[numeric_features].values
y = df_combined[target].values

print(f"\n特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

print(f"\nPM2.5统计信息:")
print(f"  均值: {y.mean():.2f} μg/m³")
print(f"  标准差: {y.std():.2f} μg/m³")
print(f"  最小值: {y.min():.2f} μg/m³")
print(f"  最大值: {y.max():.2f} μg/m³")
print(f"  中位数: {np.median(y):.2f} μg/m³")

# 创建序列数据
class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    def __init__(self, X, y, seq_length, pred_length):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        self.pred_length = pred_length
        
    def __len__(self):
        return len(self.X) - self.seq_length - self.pred_length + 1
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.seq_length]
        y_seq = self.y[idx + self.seq_length:idx + self.seq_length + self.pred_length]
        return torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)

# 按时间顺序划分
n_samples = len(X_scaled)
train_size = int(n_samples * 0.70)
val_size = int(n_samples * 0.15)

X_train = X_scaled[:train_size]
X_val = X_scaled[train_size:train_size + val_size]
X_test = X_scaled[train_size + val_size:]

y_train = y_scaled[:train_size]
y_val = y_scaled[train_size:train_size + val_size]
y_test = y_scaled[train_size + val_size:]

# 创建数据集
train_dataset = TimeSeriesDataset(X_train, y_train, SEQ_LENGTH, PRED_LENGTH)
val_dataset = TimeSeriesDataset(X_val, y_val, SEQ_LENGTH, PRED_LENGTH)
test_dataset = TimeSeriesDataset(X_test, y_test, SEQ_LENGTH, PRED_LENGTH)

print(f"\n数据集划分:")
print(f"  训练集序列数: {len(train_dataset)} ({len(train_dataset)/(len(train_dataset)+len(val_dataset)+len(test_dataset))*100:.1f}%)")
print(f"  验证集序列数: {len(val_dataset)} ({len(val_dataset)/(len(train_dataset)+len(val_dataset)+len(test_dataset))*100:.1f}%)")
print(f"  测试集序列数: {len(test_dataset)} ({len(test_dataset)/(len(train_dataset)+len(val_dataset)+len(test_dataset))*100:.1f}%)")

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\n批次数:")
print(f"  训练批次: {len(train_loader)}")
print(f"  验证批次: {len(val_loader)}")
print(f"  测试批次: {len(test_loader)}")

# ============================== 第6部分: Transformer模型定义 ==============================
print("\n" + "=" * 80)
print("第3步: Transformer模型定义")
print("=" * 80)

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TimeSeriesTransformer(nn.Module):
    """时间序列Transformer模型"""
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.1, pred_length=7):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_length = pred_length
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层 - 多步预测
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, pred_length)
        )
        
        self.init_weights()
        
    def init_weights(self):
        """初始化权重"""
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        
    def forward(self, src):
        # src: (batch_size, seq_length, input_dim)
        
        # 输入投影
        src = self.input_projection(src)  # (batch_size, seq_length, d_model)
        
        # 位置编码
        src = self.pos_encoder(src)
        
        # Transformer编码
        memory = self.transformer_encoder(src)  # (batch_size, seq_length, d_model)
        
        # 使用最后一个时间步的输出进行预测
        output = memory[:, -1, :]  # (batch_size, d_model)
        
        # 多步预测
        output = self.decoder(output)  # (batch_size, pred_length)
        
        return output

print("✓ Transformer模型架构定义完成")
print(f"  输入特征维度: {len(numeric_features)}")
print(f"  序列长度: {SEQ_LENGTH}")
print(f"  预测长度: {PRED_LENGTH}")

# ============================== 第7部分: 训练和评估函数 ==============================
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=50, verbose=True):
    """训练模型"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                       patience=10, verbose=verbose)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"  早停触发于 Epoch {epoch+1}")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, epoch + 1

def predict_model(model, data_loader):
    """模型预测"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            output = model(batch_X)
            predictions.append(output.cpu().numpy())
            actuals.append(batch_y.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    return predictions, actuals

def evaluate_predictions(y_true, y_pred, dataset_name):
    """评估预测结果"""
    # 对于多步预测，计算每一步的平均指标
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    r2 = r2_score(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
    
    return {
        'Dataset': dataset_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

# ============================== 第8部分: 基础模型训练 ==============================
print("\n" + "=" * 80)
print("第4步: Transformer基础模型训练")
print("=" * 80)

# 基础模型参数
d_model_basic = 128
nhead_basic = 8
num_layers_basic = 3
dim_feedforward_basic = 512
dropout_basic = 0.1
lr_basic = 0.001

print("\n基础模型参数:")
print(f"  d_model: {d_model_basic}")
print(f"  nhead: {nhead_basic}")
print(f"  num_layers: {num_layers_basic}")
print(f"  dim_feedforward: {dim_feedforward_basic}")
print(f"  dropout: {dropout_basic}")
print(f"  learning_rate: {lr_basic}")

# 创建模型
model_basic = TimeSeriesTransformer(
    input_dim=len(numeric_features),
    d_model=d_model_basic,
    nhead=nhead_basic,
    num_layers=num_layers_basic,
    dim_feedforward=dim_feedforward_basic,
    dropout=dropout_basic,
    pred_length=PRED_LENGTH
).to(device)

print(f"\n模型参数数量: {sum(p.numel() for p in model_basic.parameters()):,}")

print("\n开始训练基础模型...")
start_time = time.time()
model_basic, train_losses_basic, val_losses_basic, epochs_trained_basic = train_model(
    model_basic, train_loader, val_loader, 
    epochs=200, lr=lr_basic, patience=50, verbose=True
)
training_time_basic = time.time() - start_time

print(f"\n✓ 基础模型训练完成")
print(f"  训练轮数: {epochs_trained_basic}")
print(f"  训练时间: {training_time_basic:.2f}秒")
print(f"  最终训练损失: {train_losses_basic[-1]:.6f}")
print(f"  最终验证损失: {val_losses_basic[-1]:.6f}")

# 预测
print("\n进行预测...")
train_pred_basic, train_actual_basic = predict_model(model_basic, train_loader)
val_pred_basic, val_actual_basic = predict_model(model_basic, val_loader)
test_pred_basic, test_actual_basic = predict_model(model_basic, test_loader)

# 反标准化
train_pred_basic_orig = scaler_y.inverse_transform(train_pred_basic)
train_actual_basic_orig = scaler_y.inverse_transform(train_actual_basic)
val_pred_basic_orig = scaler_y.inverse_transform(val_pred_basic)
val_actual_basic_orig = scaler_y.inverse_transform(val_actual_basic)
test_pred_basic_orig = scaler_y.inverse_transform(test_pred_basic)
test_actual_basic_orig = scaler_y.inverse_transform(test_actual_basic)

# 评估
results_basic = []
results_basic.append(evaluate_predictions(train_actual_basic_orig, train_pred_basic_orig, 'Train'))
results_basic.append(evaluate_predictions(val_actual_basic_orig, val_pred_basic_orig, 'Validation'))
results_basic.append(evaluate_predictions(test_actual_basic_orig, test_pred_basic_orig, 'Test'))

results_basic_df = pd.DataFrame(results_basic)
print("\n基础模型性能:")
print(results_basic_df.to_string(index=False))

# ============================== 第9部分: 超参数优化 ==============================
print("\n" + "=" * 80)
print("第5步: 超参数优化")
print("=" * 80)

if BAYESIAN_OPT_AVAILABLE:
    print("\n使用贝叶斯优化进行超参数搜索...")
    
    def transformer_evaluate(d_model, nhead, num_layers, dim_feedforward, dropout, learning_rate):
        """贝叶斯优化目标函数"""
        # 确保nhead能整除d_model
        d_model = int(d_model)
        nhead = int(nhead)
        if d_model % nhead != 0:
            # 调整d_model使其能被nhead整除
            d_model = (d_model // nhead) * nhead
            if d_model < nhead:
                d_model = nhead
        
        try:
            model_temp = TimeSeriesTransformer(
                input_dim=len(numeric_features),
                d_model=d_model,
                nhead=nhead,
                num_layers=int(num_layers),
                dim_feedforward=int(dim_feedforward),
                dropout=dropout,
                pred_length=PRED_LENGTH
            ).to(device)
            
            _, _, val_losses_temp, _ = train_model(
                model_temp, train_loader, val_loader,
                epochs=100, lr=learning_rate, patience=20, verbose=False
            )
            
            best_val_loss = min(val_losses_temp)
            
            # 清理内存
            del model_temp
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return -best_val_loss  # 负数因为要最大化
        except Exception as e:
            print(f"  错误: {e}")
            return -999999
    
    # 定义搜索空间
    pbounds = {
        'd_model': (64, 256),
        'nhead': (4, 8),
        'num_layers': (2, 4),
        'dim_feedforward': (256, 1024),
        'dropout': (0.05, 0.3),
        'learning_rate': (0.0001, 0.01)
    }
    
    optimizer = BayesianOptimization(
        f=transformer_evaluate,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    optimizer.maximize(init_points=5, n_iter=15)
    
    best_params = optimizer.max['params']
    best_params['d_model'] = int(best_params['d_model'])
    best_params['nhead'] = int(best_params['nhead'])
    best_params['num_layers'] = int(best_params['num_layers'])
    best_params['dim_feedforward'] = int(best_params['dim_feedforward'])
    
    # 确保d_model能被nhead整除
    if best_params['d_model'] % best_params['nhead'] != 0:
        best_params['d_model'] = (best_params['d_model'] // best_params['nhead']) * best_params['nhead']
    
    print(f"\n最佳参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  最佳验证损失: {-optimizer.max['target']:.6f}")

else:
    # 网格搜索
    print("\n使用网格搜索进行超参数优化...")
    
    param_grid = {
        'd_model': [64, 128, 192],
        'nhead': [4, 8],
        'num_layers': [2, 3],
        'dim_feedforward': [256, 512],
        'dropout': [0.1, 0.2],
        'learning_rate': [0.001, 0.0005]
    }
    
    total_combinations = int(np.prod([len(v) for v in param_grid.values()]))
    print(f"总共 {total_combinations} 种参数组合")
    
    all_combos = list(product(*param_grid.values()))
    
    best_val_loss = float('inf')
    best_params = {}
    
    if TQDM_AVAILABLE:
        pbar = tqdm(all_combos, desc="网格搜索", unit="组合")
    else:
        pbar = all_combos
    
    for i, combo in enumerate(pbar):
        d_model, nhead, num_layers, dim_feedforward, dropout, lr = combo
        
        # 确保d_model能被nhead整除
        if d_model % nhead != 0:
            continue
        
        try:
            model_temp = TimeSeriesTransformer(
                input_dim=len(numeric_features),
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                pred_length=PRED_LENGTH
            ).to(device)
            
            _, _, val_losses_temp, _ = train_model(
                model_temp, train_loader, val_loader,
                epochs=100, lr=lr, patience=20, verbose=False
            )
            
            current_val_loss = min(val_losses_temp)
            
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_params = {
                    'd_model': d_model,
                    'nhead': nhead,
                    'num_layers': num_layers,
                    'dim_feedforward': dim_feedforward,
                    'dropout': dropout,
                    'learning_rate': lr
                }
            
            del model_temp
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            if not TQDM_AVAILABLE and (i + 1) % 5 == 0:
                print(f"  已测试 {i+1}/{total_combinations} 组合，当前最佳验证损失: {best_val_loss:.6f}")
        
        except Exception as e:
            continue
    
    print(f"\n最佳参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  最佳验证损失: {best_val_loss:.6f}")

# ============================== 第10部分: 训练优化模型 ==============================
print("\n" + "=" * 80)
print("第6步: 使用最佳参数训练优化模型")
print("=" * 80)

print("\n优化模型参数:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# 创建优化模型
model_optimized = TimeSeriesTransformer(
    input_dim=len(numeric_features),
    d_model=best_params['d_model'],
    nhead=best_params['nhead'],
    num_layers=best_params['num_layers'],
    dim_feedforward=best_params['dim_feedforward'],
    dropout=best_params['dropout'],
    pred_length=PRED_LENGTH
).to(device)

print(f"\n优化模型参数数量: {sum(p.numel() for p in model_optimized.parameters()):,}")

print("\n开始训练优化模型...")
start_time = time.time()
model_optimized, train_losses_opt, val_losses_opt, epochs_trained_opt = train_model(
    model_optimized, train_loader, val_loader,
    epochs=300, lr=best_params['learning_rate'], patience=100, verbose=True
)
training_time_opt = time.time() - start_time

print(f"\n✓ 优化模型训练完成")
print(f"  训练轮数: {epochs_trained_opt}")
print(f"  训练时间: {training_time_opt:.2f}秒")
print(f"  最终训练损失: {train_losses_opt[-1]:.6f}")
print(f"  最终验证损失: {val_losses_opt[-1]:.6f}")

# 预测
print("\n进行预测...")
train_pred_opt, train_actual_opt = predict_model(model_optimized, train_loader)
val_pred_opt, val_actual_opt = predict_model(model_optimized, val_loader)
test_pred_opt, test_actual_opt = predict_model(model_optimized, test_loader)

# 反标准化
train_pred_opt_orig = scaler_y.inverse_transform(train_pred_opt)
train_actual_opt_orig = scaler_y.inverse_transform(train_actual_opt)
val_pred_opt_orig = scaler_y.inverse_transform(val_pred_opt)
val_actual_opt_orig = scaler_y.inverse_transform(val_actual_opt)
test_pred_opt_orig = scaler_y.inverse_transform(test_pred_opt)
test_actual_opt_orig = scaler_y.inverse_transform(test_actual_opt)

# 评估
results_opt = []
results_opt.append(evaluate_predictions(train_actual_opt_orig, train_pred_opt_orig, 'Train'))
results_opt.append(evaluate_predictions(val_actual_opt_orig, val_pred_opt_orig, 'Validation'))
results_opt.append(evaluate_predictions(test_actual_opt_orig, test_pred_opt_orig, 'Test'))

results_opt_df = pd.DataFrame(results_opt)
print("\n优化模型性能:")
print(results_opt_df.to_string(index=False))

# ============================== 第11部分: 模型比较 ==============================
print("\n" + "=" * 80)
print("第7步: 模型性能比较")
print("=" * 80)

results_basic_df['Model'] = 'Transformer_Basic'
results_opt_df['Model'] = 'Transformer_Optimized'
all_results = pd.concat([results_basic_df, results_opt_df])

all_results = all_results[['Model', 'Dataset', 'R²', 'RMSE', 'MAE', 'MAPE']]

print("\n所有模型性能对比:")
print(all_results.to_string(index=False))

test_results = all_results[all_results['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\n测试集性能排名:")
print(test_results.to_string(index=False))

# 性能提升
basic_test_r2 = results_basic_df[results_basic_df['Dataset'] == 'Test']['R²'].values[0]
opt_test_r2 = results_opt_df[results_opt_df['Dataset'] == 'Test']['R²'].values[0]
basic_test_rmse = results_basic_df[results_basic_df['Dataset'] == 'Test']['RMSE'].values[0]
opt_test_rmse = results_opt_df[results_opt_df['Dataset'] == 'Test']['RMSE'].values[0]

r2_improvement = (opt_test_r2 - basic_test_r2) / (abs(basic_test_r2) + 1e-8) * 100
rmse_improvement = (basic_test_rmse - opt_test_rmse) / basic_test_rmse * 100

print(f"\n优化效果:")
print(f"  R²提升: {r2_improvement:.2f}%")
print(f"  RMSE降低: {rmse_improvement:.2f}%")

# ============================== 第12部分: 特征重要性分析 ==============================
print("\n" + "=" * 80)
print("第8步: 特征重要性分析")
print("=" * 80)

if SHAP_AVAILABLE:
    print("\n使用SHAP进行特征重要性分析...")
    print("注意: SHAP计算较慢，使用样本子集...")
    
    # 使用少量样本进行SHAP分析
    sample_size = min(500, len(test_dataset))
    sample_indices = np.random.choice(len(test_dataset), sample_size, replace=False)
    
    X_sample = []
    for idx in sample_indices:
        X_seq, _ = test_dataset[idx]
        X_sample.append(X_seq.numpy())
    X_sample = np.array(X_sample)
    X_sample_tensor = torch.FloatTensor(X_sample).to(device)
    
    try:
        # 使用GradientExplainer
        explainer = shap.GradientExplainer(model_optimized, X_sample_tensor)
        shap_values = explainer.shap_values(X_sample_tensor[:100])  # 使用更少样本计算
        
        # 计算平均绝对SHAP值
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # 对序列和样本取平均
        mean_shap = np.abs(shap_values).mean(axis=(0, 1))
        
        feature_importance = pd.DataFrame({
            'Feature': numeric_features,
            'Importance': mean_shap
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        feature_importance['Importance_Norm'] = (feature_importance['Importance'] / 
                                                  feature_importance['Importance'].sum() * 100)
        
        print(f"\nTop 20 重要特征 (SHAP):")
        print(feature_importance.head(20)[['Feature', 'Importance_Norm']].to_string(index=False))
        
    except Exception as e:
        print(f"SHAP分析失败: {e}")
        print("使用排列重要性作为替代...")
        SHAP_AVAILABLE = False

if not SHAP_AVAILABLE:
    print("\n使用排列重要性进行特征分析...")
    
    # 获取基线性能
    test_pred_baseline, test_actual_baseline = predict_model(model_optimized, test_loader)
    baseline_rmse = np.sqrt(mean_squared_error(test_actual_baseline.flatten(), 
                                                 test_pred_baseline.flatten()))
    
    feature_importances = []
    
    print("计算排列重要性...")
    for i, feat_name in enumerate(numeric_features):
        if i % 5 == 0 and i > 0:
            print(f"  已处理 {i}/{len(numeric_features)} 个特征")
        
        # 创建扰动数据集
        X_test_permuted = X_test.copy()
        np.random.shuffle(X_test_permuted[:, i])
        
        test_dataset_permuted = TimeSeriesDataset(X_test_permuted, y_test, SEQ_LENGTH, PRED_LENGTH)
        test_loader_permuted = DataLoader(test_dataset_permuted, batch_size=BATCH_SIZE, 
                                           shuffle=False, num_workers=0)
        
        # 预测
        test_pred_permuted, test_actual_permuted = predict_model(model_optimized, test_loader_permuted)
        permuted_rmse = np.sqrt(mean_squared_error(test_actual_permuted.flatten(), 
                                                     test_pred_permuted.flatten()))
        
        # 重要性 = 扰动后RMSE增加量
        importance = permuted_rmse - baseline_rmse
        feature_importances.append(importance)
    
    feature_importance = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': feature_importances
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    feature_importance['Importance_Norm'] = (feature_importance['Importance'] / 
                                              (feature_importance['Importance'].sum() + 1e-8) * 100)
    
    print(f"\nTop 20 重要特征 (排列重要性):")
    print(feature_importance.head(20)[['Feature', 'Importance_Norm']].to_string(index=False))

# ============================== 第13部分: 可视化 ==============================
print("\n" + "=" * 80)
print("第9步: 生成可视化图表")
print("=" * 80)

# 13.1 训练过程曲线
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].plot(train_losses_basic, label='训练集', linewidth=2)
axes[0].plot(val_losses_basic, label='验证集', linewidth=2)
axes[0].axvline(x=len(train_losses_basic)-1, color='r', linestyle='--',
                label=f'最终轮次({len(train_losses_basic)})', linewidth=1.5)
axes[0].set_xlabel('轮次', fontsize=12)
axes[0].set_ylabel('损失 (MSE)', fontsize=12)
axes[0].set_title('Transformer基础模型 - 训练过程', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_losses_opt, label='训练集', linewidth=2)
axes[1].plot(val_losses_opt, label='验证集', linewidth=2)
axes[1].axvline(x=len(train_losses_opt)-1, color='r', linestyle='--',
                label=f'最终轮次({len(train_losses_opt)})', linewidth=1.5)
axes[1].set_xlabel('轮次', fontsize=12)
axes[1].set_ylabel('损失 (MSE)', fontsize=12)
axes[1].set_title('Transformer优化模型 - 训练过程', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
print("保存: training_curves.png")
plt.close()

# 13.2 预测vs实际值散点图 (使用第1天预测)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

models_data = [
    ('Basic', train_pred_basic_orig[:, 0], train_actual_basic_orig[:, 0], 'Train'),
    ('Basic', val_pred_basic_orig[:, 0], val_actual_basic_orig[:, 0], 'Val'),
    ('Basic', test_pred_basic_orig[:, 0], test_actual_basic_orig[:, 0], 'Test'),
    ('Optimized', train_pred_opt_orig[:, 0], train_actual_opt_orig[:, 0], 'Train'),
    ('Optimized', val_pred_opt_orig[:, 0], val_actual_opt_orig[:, 0], 'Val'),
    ('Optimized', test_pred_opt_orig[:, 0], test_actual_opt_orig[:, 0], 'Test')
]

for idx, (model_name, y_pred, y_true, dataset) in enumerate(models_data):
    row = idx // 3
    col = idx % 3
    
    ax = axes[row, col]
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测线')
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    ax.set_xlabel('实际PM2.5浓度 (μg/m³)', fontsize=11)
    ax.set_ylabel('预测PM2.5浓度 (μg/m³)', fontsize=11)
    ax.set_title(f'Transformer_{model_name} - {dataset}\nR²={r2:.4f}, RMSE={rmse:.2f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
print("保存: prediction_scatter.png")
plt.close()

# 13.3 时间序列预测对比 (第1天预测)
fig, axes = plt.subplots(2, 1, figsize=(18, 10))

plot_range = min(300, len(test_pred_basic_orig))
plot_idx = range(len(test_pred_basic_orig) - plot_range, len(test_pred_basic_orig))

axes[0].plot(plot_idx, test_actual_basic_orig[plot_idx, 0], 'k-', label='实际值',
             linewidth=2, alpha=0.8)
axes[0].plot(plot_idx, test_pred_basic_orig[plot_idx, 0], 'b--', label='基础模型预测',
             linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('样本索引', fontsize=12)
axes[0].set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
axes[0].set_title('Transformer基础模型 - 时间序列预测对比（测试集最后300个样本，第1天预测）',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(plot_idx, test_actual_opt_orig[plot_idx, 0], 'k-', label='实际值',
             linewidth=2, alpha=0.8)
axes[1].plot(plot_idx, test_pred_opt_orig[plot_idx, 0], 'g--', label='优化模型预测',
             linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('样本索引', fontsize=12)
axes[1].set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
axes[1].set_title('Transformer优化模型 - 时间序列预测对比（测试集最后300个样本，第1天预测）',
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'timeseries_comparison.png', dpi=300, bbox_inches='tight')
print("保存: timeseries_comparison.png")
plt.close()

# 13.4 残差分析
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
    ax.set_title(f'Transformer_{model_name} - {dataset}\n残差均值={residuals.mean():.2f}, 标准差={residuals.std():.2f}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
print("保存: residuals_analysis.png")
plt.close()

# 13.5 特征重要性图
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

top_n = 20
top_features = feature_importance.head(top_n)

ax.barh(range(top_n), top_features['Importance_Norm'], color='steelblue')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['Feature'], fontsize=10)
ax.set_xlabel('重要性 (%)', fontsize=12)
ax.set_title(f'Top {top_n} 重要特征', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
print("保存: feature_importance.png")
plt.close()

# 13.6 模型性能对比柱状图
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

# 13.7 误差分布直方图
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

errors_basic = test_actual_basic_orig[:, 0] - test_pred_basic_orig[:, 0]
errors_opt = test_actual_opt_orig[:, 0] - test_pred_opt_orig[:, 0]

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

# ============================== 第14部分: 保存结果 ==============================
print("\n" + "=" * 80)
print("第10步: 保存结果")
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

# 保存预测结果 (第1天预测)
predictions_df = pd.DataFrame({
    'Sample_Index': range(len(test_actual_basic_orig)),
    'Actual_Day1': test_actual_basic_orig[:, 0],
    'Prediction_Basic_Day1': test_pred_basic_orig[:, 0],
    'Prediction_Optimized_Day1': test_pred_opt_orig[:, 0],
    'Error_Basic': test_actual_basic_orig[:, 0] - test_pred_basic_orig[:, 0],
    'Error_Optimized': test_actual_opt_orig[:, 0] - test_pred_opt_orig[:, 0]
})
predictions_df.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
print("保存: predictions.csv")

# 保存模型
torch.save(model_optimized.state_dict(), model_dir / 'transformer_optimized.pth')
print("保存: transformer_optimized.pth")

# 保存完整模型（包括架构）
torch.save({
    'model_state_dict': model_optimized.state_dict(),
    'model_params': best_params,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'feature_names': numeric_features,
    'seq_length': SEQ_LENGTH,
    'pred_length': PRED_LENGTH
}, model_dir / 'transformer_optimized_full.pth')
print("保存: transformer_optimized_full.pth")

# 使用pickle保存（可选）
with open(model_dir / 'transformer_optimized.pkl', 'wb') as f:
    pickle.dump({
        'model': model_optimized,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_names': numeric_features
    }, f)
print("保存: transformer_optimized.pkl")

# ============================== 第15部分: 总结报告 ==============================
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
print("  - transformer_optimized.pth       Transformer模型（权重）")
print("  - transformer_optimized_full.pth  Transformer模型（完整）")
print("  - transformer_optimized.pkl       Transformer模型（pickle格式）")

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

print(f"\n模型训练时间:")
print(f"  基础模型: {training_time_basic:.2f}秒")
print(f"  优化模型: {training_time_opt:.2f}秒")

print("\n多步预测信息:")
print(f"  输入序列长度: {SEQ_LENGTH}天")
print(f"  预测未来天数: {PRED_LENGTH}天")

print("\n" + "=" * 80)
print("Transformer PM2.5浓度预测完成！")
print("=" * 80)

