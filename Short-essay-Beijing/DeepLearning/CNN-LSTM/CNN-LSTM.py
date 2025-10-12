"""
北京PM2.5浓度预测 - CNN-LSTM + Attention模型
使用1D CNN + LSTM + Attention进行时间序列预测

特点:
- 1D CNN提取局部时间模式
- LSTM捕捉长期时间依赖
- Attention机制提供特征重要性分析
- 多序列长度对比（7/14/30天）
- 超参数优化（贝叶斯优化/网格搜索）
- 完整的训练、评估和可视化流程

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
print("北京PM2.5浓度预测 - CNN-LSTM + Attention模型")
print("=" * 80)
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
    'd2m', 't2m', 'u10', 'v10', 'u100', 'v100',
    'blh', 'sp', 'tcwv',
    'tp', 'avg_tprate',
    'tisr', 'str',
    'cvh', 'cvl',
    'mn2t', 'sd', 'lsm'
]

# 序列长度（多窗口对比）
sequence_lengths = [7, 14, 30]

print(f"数据时间范围: {start_date.date()} 至 {end_date.date()}")
print(f"目标变量: PM2.5浓度")
print(f"序列长度: {sequence_lengths} 天")
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
    print(f"检查目录是否存在: {os.path.exists(era5_path)}")
    
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
        print(f"可用变量: {list(df_era5_all.columns[:10])}..." if len(df_era5_all.columns) > 10 else f"可用变量: {list(df_era5_all.columns)}")
        
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
        print("可能的原因:")
        print("1. 文件命名格式不匹配（期望格式: *YYYYMM*.csv）")
        print("2. 文件内容格式不正确（缺少time列）")
        print("3. 文件路径不正确")
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

# ============================== 第4部分: 时序数据集类 ==============================
class TimeSeriesDataset(Dataset):
    """时间序列数据集类"""
    def __init__(self, X, y, seq_length):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.X) - self.seq_length
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.seq_length]
        y_target = self.y[idx + self.seq_length]
        return torch.FloatTensor(X_seq), torch.FloatTensor([y_target])

# ============================== 第5部分: CNN-LSTM-Attention模型 ==============================
class CNNLSTMAttention(nn.Module):
    """CNN-LSTM-Attention模型"""
    def __init__(self, input_size, hidden_size, num_layers, num_filters, kernel_size, dropout=0.2):
        super(CNNLSTMAttention, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1D CNN层
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, 
                               kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)
        self.dropout1 = nn.Dropout(dropout)
        
        # LSTM层
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention层
        self.attention = nn.Linear(hidden_size, 1)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x, return_attention=False):
        # x: (batch, seq_len, features)
        batch_size, seq_len, features = x.size()
        
        # CNN: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # (batch, num_filters, seq_len) -> (batch, seq_len, num_filters)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)
        
        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # attention_weights: (batch, seq_len, 1)
        
        # Weighted sum
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        # context_vector: (batch, hidden_size)
        
        # Output
        output = self.fc(context_vector)
        
        if return_attention:
            return output, attention_weights
        return output

# ============================== 第6部分: 训练和评估函数 ==============================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, device):
    """训练模型"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_model(model, data_loader, device):
    """评估模型"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(y_batch.numpy().flatten())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
    
    return {
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'predictions': predictions,
        'actuals': actuals
    }

def predict_with_attention(model, data_loader, device):
    """预测并提取attention权重"""
    model.eval()
    predictions = []
    actuals = []
    attention_weights_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs, attn_weights = model(X_batch, return_attention=True)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(y_batch.numpy().flatten())
            attention_weights_list.append(attn_weights.cpu().numpy())
    
    # 平均attention权重
    avg_attention = np.concatenate(attention_weights_list, axis=0).mean(axis=0).flatten()
    
    return np.array(predictions), np.array(actuals), avg_attention

# ============================== 第7部分: 数据加载和预处理 ==============================
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

# ============================== 第8部分: 特征选择和数据准备 ==============================
print("\n" + "=" * 80)
print("第2步: 特征选择和数据准备")
print("=" * 80)

target = 'PM2.5'
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']

numeric_features = [col for col in df_combined.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_cols]

print(f"\n选择的特征数量: {len(numeric_features)}")
print(f"目标变量: {target}")

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

# 保存scaler
with open(model_dir / 'scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open(model_dir / 'scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
print("\n保存StandardScaler对象")

# ============================== 第9部分: 多窗口模型训练 ==============================
print("\n" + "=" * 80)
print("第3步: 多窗口CNN-LSTM模型训练")
print("=" * 80)

# 存储所有结果
all_results = []
all_models = {}
all_predictions = {}
all_attention_weights = {}

for seq_length in sequence_lengths:
    print(f"\n{'='*80}")
    print(f"训练窗口长度: {seq_length}天")
    print(f"{'='*80}")
    
    # 创建时序数据集
    n_samples = len(X_scaled)
    train_size = int(n_samples * 0.70)
    val_size = int(n_samples * 0.15)
    
    X_train = X_scaled[:train_size]
    X_val = X_scaled[train_size:train_size + val_size]
    X_test = X_scaled[train_size + val_size:]
    
    y_train = y_scaled[:train_size]
    y_val = y_scaled[train_size:train_size + val_size]
    y_test = y_scaled[train_size + val_size:]
    
    print(f"\n数据集划分（原始样本数: {n_samples}）:")
    print(f"  训练集: {len(X_train)} ({len(X_train)/n_samples*100:.1f}%)")
    print(f"  验证集: {len(X_val)} ({len(X_val)/n_samples*100:.1f}%)")
    print(f"  测试集: {len(X_test)} ({len(X_test)/n_samples*100:.1f}%)")
    
    # 创建Dataset和DataLoader
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_length)
    val_dataset = TimeSeriesDataset(X_val, y_val, seq_length)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_length)
    
    print(f"\n时序数据集大小（窗口长度={seq_length}）:")
    print(f"  训练集: {len(train_dataset)} 序列")
    print(f"  验证集: {len(val_dataset)} 序列")
    print(f"  测试集: {len(test_dataset)} 序列")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # ========== 基础模型 ==========
    print(f"\n{'─'*80}")
    print("训练基础模型")
    print(f"{'─'*80}")
    
    input_size = X_scaled.shape[1]
    basic_params = {
        'input_size': input_size,
        'hidden_size': 64,
        'num_layers': 2,
        'num_filters': 32,
        'kernel_size': 3,
        'dropout': 0.2
    }
    
    print("\n基础模型参数:")
    for key, value in basic_params.items():
        print(f"  {key}: {value}")
    
    model_basic = CNNLSTMAttention(**basic_params).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_basic.parameters(), lr=0.001)
    
    print("\n开始训练...")
    model_basic, train_losses_basic, val_losses_basic = train_model(
        model_basic, train_loader, val_loader, criterion, optimizer,
        num_epochs=100, patience=15, device=device
    )
    
    print("\n评估基础模型...")
    train_results_basic = evaluate_model(model_basic, train_loader, device)
    val_results_basic = evaluate_model(model_basic, val_loader, device)
    test_results_basic = evaluate_model(model_basic, test_loader, device)
    
    # 逆标准化
    train_pred_basic = scaler_y.inverse_transform(train_results_basic['predictions'].reshape(-1, 1)).flatten()
    val_pred_basic = scaler_y.inverse_transform(val_results_basic['predictions'].reshape(-1, 1)).flatten()
    test_pred_basic = scaler_y.inverse_transform(test_results_basic['predictions'].reshape(-1, 1)).flatten()
    
    train_actual = scaler_y.inverse_transform(train_results_basic['actuals'].reshape(-1, 1)).flatten()
    val_actual = scaler_y.inverse_transform(val_results_basic['actuals'].reshape(-1, 1)).flatten()
    test_actual = scaler_y.inverse_transform(test_results_basic['actuals'].reshape(-1, 1)).flatten()
    
    # 重新计算指标
    train_r2_basic = r2_score(train_actual, train_pred_basic)
    train_rmse_basic = np.sqrt(mean_squared_error(train_actual, train_pred_basic))
    train_mae_basic = mean_absolute_error(train_actual, train_pred_basic)
    train_mape_basic = np.mean(np.abs((train_actual - train_pred_basic) / (train_actual + 1e-8))) * 100
    
    val_r2_basic = r2_score(val_actual, val_pred_basic)
    val_rmse_basic = np.sqrt(mean_squared_error(val_actual, val_pred_basic))
    val_mae_basic = mean_absolute_error(val_actual, val_pred_basic)
    val_mape_basic = np.mean(np.abs((val_actual - val_pred_basic) / (val_actual + 1e-8))) * 100
    
    test_r2_basic = r2_score(test_actual, test_pred_basic)
    test_rmse_basic = np.sqrt(mean_squared_error(test_actual, test_pred_basic))
    test_mae_basic = mean_absolute_error(test_actual, test_pred_basic)
    test_mape_basic = np.mean(np.abs((test_actual - test_pred_basic) / (test_actual + 1e-8))) * 100
    
    print(f"\n基础模型性能 (窗口={seq_length}天):")
    print(f"  训练集 - R²: {train_r2_basic:.4f}, RMSE: {train_rmse_basic:.2f}, MAE: {train_mae_basic:.2f}, MAPE: {train_mape_basic:.2f}%")
    print(f"  验证集 - R²: {val_r2_basic:.4f}, RMSE: {val_rmse_basic:.2f}, MAE: {val_mae_basic:.2f}, MAPE: {val_mape_basic:.2f}%")
    print(f"  测试集 - R²: {test_r2_basic:.4f}, RMSE: {test_rmse_basic:.2f}, MAE: {test_mae_basic:.2f}, MAPE: {test_mape_basic:.2f}%")
    
    # 保存结果
    all_results.append({
        'Model': f'CNN-LSTM_Basic_W{seq_length}',
        'Window': seq_length,
        'Dataset': 'Train',
        'R²': train_r2_basic,
        'RMSE': train_rmse_basic,
        'MAE': train_mae_basic,
        'MAPE': train_mape_basic
    })
    all_results.append({
        'Model': f'CNN-LSTM_Basic_W{seq_length}',
        'Window': seq_length,
        'Dataset': 'Validation',
        'R²': val_r2_basic,
        'RMSE': val_rmse_basic,
        'MAE': val_mae_basic,
        'MAPE': val_mape_basic
    })
    all_results.append({
        'Model': f'CNN-LSTM_Basic_W{seq_length}',
        'Window': seq_length,
        'Dataset': 'Test',
        'R²': test_r2_basic,
        'RMSE': test_rmse_basic,
        'MAE': test_mae_basic,
        'MAPE': test_mape_basic
    })
    
    all_models[f'basic_w{seq_length}'] = {
        'model': model_basic,
        'train_losses': train_losses_basic,
        'val_losses': val_losses_basic,
        'params': basic_params
    }
    
    all_predictions[f'basic_w{seq_length}'] = {
        'train': (train_actual, train_pred_basic),
        'val': (val_actual, val_pred_basic),
        'test': (test_actual, test_pred_basic)
    }
    
    # ========== 超参数优化 ==========
    print(f"\n{'─'*80}")
    print("超参数优化")
    print(f"{'─'*80}")
    
    if BAYESIAN_OPT_AVAILABLE:
        print("\n使用贝叶斯优化...")
        
        def cnn_lstm_evaluate(hidden_size, num_layers, num_filters, kernel_size, learning_rate, dropout):
            """贝叶斯优化目标函数"""
            params_test = {
                'input_size': input_size,
                'hidden_size': int(hidden_size),
                'num_layers': int(num_layers),
                'num_filters': int(num_filters),
                'kernel_size': int(kernel_size),
                'dropout': dropout
            }
            
            model_temp = CNNLSTMAttention(**params_test).to(device)
            criterion_temp = nn.MSELoss()
            optimizer_temp = optim.Adam(model_temp.parameters(), lr=learning_rate)
            
            model_temp, _, _ = train_model(
                model_temp, train_loader, val_loader, criterion_temp, optimizer_temp,
                num_epochs=50, patience=10, device=device
            )
            
            val_results_temp = evaluate_model(model_temp, val_loader, device)
            rmse = val_results_temp['RMSE']
            
            return -rmse  # 负RMSE（最大化）
        
        pbounds = {
            'hidden_size': (32, 128),
            'num_layers': (1, 3),
            'num_filters': (16, 64),
            'kernel_size': (3, 7),
            'learning_rate': (0.0001, 0.01),
            'dropout': (0.1, 0.5)
        }
        
        optimizer_bayes = BayesianOptimization(
            f=cnn_lstm_evaluate,
            pbounds=pbounds,
            random_state=42,
            verbose=0
        )
        
        optimizer_bayes.maximize(init_points=3, n_iter=10)
        
        best_params_opt = optimizer_bayes.max['params']
        best_params_opt['hidden_size'] = int(best_params_opt['hidden_size'])
        best_params_opt['num_layers'] = int(best_params_opt['num_layers'])
        best_params_opt['num_filters'] = int(best_params_opt['num_filters'])
        best_params_opt['kernel_size'] = int(best_params_opt['kernel_size'])
        
        print(f"\n最佳参数 (贝叶斯优化):")
        for key, value in best_params_opt.items():
            print(f"  {key}: {value}")
        print(f"  最佳验证RMSE: {-optimizer_bayes.max['target']:.4f}")
        
        best_lr = best_params_opt.pop('learning_rate')
        
    else:
        print("\n使用网格搜索...")
        
        param_grid = {
            'hidden_size': [48, 64, 96],
            'num_layers': [2, 3],
            'num_filters': [32, 48],
            'kernel_size': [3, 5],
            'learning_rate': [0.001, 0.0005],
            'dropout': [0.2, 0.3]
        }
        
        total_combinations = int(np.prod([len(v) for v in param_grid.values()]))
        print(f"总共 {total_combinations} 种参数组合")
        
        best_val_rmse = float('inf')
        best_params_opt = {}
        best_lr = 0.001
        
        tested = 0
        for hs in param_grid['hidden_size']:
            for nl in param_grid['num_layers']:
                for nf in param_grid['num_filters']:
                    for ks in param_grid['kernel_size']:
                        for lr in param_grid['learning_rate']:
                            for dr in param_grid['dropout']:
                                tested += 1
                                
                                params_test = {
                                    'input_size': input_size,
                                    'hidden_size': hs,
                                    'num_layers': nl,
                                    'num_filters': nf,
                                    'kernel_size': ks,
                                    'dropout': dr
                                }
                                
                                model_temp = CNNLSTMAttention(**params_test).to(device)
                                criterion_temp = nn.MSELoss()
                                optimizer_temp = optim.Adam(model_temp.parameters(), lr=lr)
                                
                                model_temp, _, _ = train_model(
                                    model_temp, train_loader, val_loader, criterion_temp, optimizer_temp,
                                    num_epochs=50, patience=10, device=device
                                )
                                
                                val_results_temp = evaluate_model(model_temp, val_loader, device)
                                val_rmse = val_results_temp['RMSE']
                                
                                if val_rmse < best_val_rmse:
                                    best_val_rmse = val_rmse
                                    best_params_opt = params_test.copy()
                                    best_params_opt.pop('input_size')
                                    best_lr = lr
                                
                                if tested % 5 == 0:
                                    print(f"  已测试 {tested}/{total_combinations} 组合，当前最佳RMSE: {best_val_rmse:.4f}")
        
        print(f"\n最佳参数 (网格搜索):")
        for key, value in best_params_opt.items():
            print(f"  {key}: {value}")
        print(f"  learning_rate: {best_lr}")
        print(f"  最佳验证RMSE: {best_val_rmse:.4f}")
    
    # ========== 训练优化模型 ==========
    print(f"\n{'─'*80}")
    print("使用最佳参数训练优化模型")
    print(f"{'─'*80}")
    
    optimized_params = {
        'input_size': input_size,
        **best_params_opt
    }
    
    print("\n优化模型参数:")
    for key, value in optimized_params.items():
        print(f"  {key}: {value}")
    print(f"  learning_rate: {best_lr}")
    
    model_optimized = CNNLSTMAttention(**optimized_params).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_optimized.parameters(), lr=best_lr)
    
    print("\n开始训练...")
    model_optimized, train_losses_opt, val_losses_opt = train_model(
        model_optimized, train_loader, val_loader, criterion, optimizer,
        num_epochs=150, patience=20, device=device
    )
    
    print("\n评估优化模型...")
    train_results_opt = evaluate_model(model_optimized, train_loader, device)
    val_results_opt = evaluate_model(model_optimized, val_loader, device)
    test_results_opt = evaluate_model(model_optimized, test_loader, device)
    
    # 逆标准化
    train_pred_opt = scaler_y.inverse_transform(train_results_opt['predictions'].reshape(-1, 1)).flatten()
    val_pred_opt = scaler_y.inverse_transform(val_results_opt['predictions'].reshape(-1, 1)).flatten()
    test_pred_opt = scaler_y.inverse_transform(test_results_opt['predictions'].reshape(-1, 1)).flatten()
    
    # 重新计算指标
    train_r2_opt = r2_score(train_actual, train_pred_opt)
    train_rmse_opt = np.sqrt(mean_squared_error(train_actual, train_pred_opt))
    train_mae_opt = mean_absolute_error(train_actual, train_pred_opt)
    train_mape_opt = np.mean(np.abs((train_actual - train_pred_opt) / (train_actual + 1e-8))) * 100
    
    val_r2_opt = r2_score(val_actual, val_pred_opt)
    val_rmse_opt = np.sqrt(mean_squared_error(val_actual, val_pred_opt))
    val_mae_opt = mean_absolute_error(val_actual, val_pred_opt)
    val_mape_opt = np.mean(np.abs((val_actual - val_pred_opt) / (val_actual + 1e-8))) * 100
    
    test_r2_opt = r2_score(test_actual, test_pred_opt)
    test_rmse_opt = np.sqrt(mean_squared_error(test_actual, test_pred_opt))
    test_mae_opt = mean_absolute_error(test_actual, test_pred_opt)
    test_mape_opt = np.mean(np.abs((test_actual - test_pred_opt) / (test_actual + 1e-8))) * 100
    
    print(f"\n优化模型性能 (窗口={seq_length}天):")
    print(f"  训练集 - R²: {train_r2_opt:.4f}, RMSE: {train_rmse_opt:.2f}, MAE: {train_mae_opt:.2f}, MAPE: {train_mape_opt:.2f}%")
    print(f"  验证集 - R²: {val_r2_opt:.4f}, RMSE: {val_rmse_opt:.2f}, MAE: {val_mae_opt:.2f}, MAPE: {val_mape_opt:.2f}%")
    print(f"  测试集 - R²: {test_r2_opt:.4f}, RMSE: {test_rmse_opt:.2f}, MAE: {test_mae_opt:.2f}, MAPE: {test_mape_opt:.2f}%")
    
    # 保存结果
    all_results.append({
        'Model': f'CNN-LSTM_Optimized_W{seq_length}',
        'Window': seq_length,
        'Dataset': 'Train',
        'R²': train_r2_opt,
        'RMSE': train_rmse_opt,
        'MAE': train_mae_opt,
        'MAPE': train_mape_opt
    })
    all_results.append({
        'Model': f'CNN-LSTM_Optimized_W{seq_length}',
        'Window': seq_length,
        'Dataset': 'Validation',
        'R²': val_r2_opt,
        'RMSE': val_rmse_opt,
        'MAE': val_mae_opt,
        'MAPE': val_mape_opt
    })
    all_results.append({
        'Model': f'CNN-LSTM_Optimized_W{seq_length}',
        'Window': seq_length,
        'Dataset': 'Test',
        'R²': test_r2_opt,
        'RMSE': test_rmse_opt,
        'MAE': test_mae_opt,
        'MAPE': test_mape_opt
    })
    
    all_models[f'optimized_w{seq_length}'] = {
        'model': model_optimized,
        'train_losses': train_losses_opt,
        'val_losses': val_losses_opt,
        'params': optimized_params
    }
    
    all_predictions[f'optimized_w{seq_length}'] = {
        'train': (train_actual, train_pred_opt),
        'val': (val_actual, val_pred_opt),
        'test': (test_actual, test_pred_opt)
    }
    
    # 提取Attention权重
    print("\n提取Attention特征重要性...")
    _, _, avg_attention = predict_with_attention(model_optimized, test_loader, device)
    all_attention_weights[f'w{seq_length}'] = avg_attention
    
    # 保存模型
    torch.save(model_optimized.state_dict(), model_dir / f'cnn_lstm_window{seq_length}_optimized.pth')
    with open(model_dir / f'cnn_lstm_window{seq_length}_optimized.pkl', 'wb') as f:
        pickle.dump(model_optimized, f)
    print(f"保存模型: cnn_lstm_window{seq_length}_optimized.pth/pkl")

# ============================== 第10部分: 结果汇总 ==============================
print("\n" + "=" * 80)
print("第4步: 结果汇总和比较")
print("=" * 80)

results_df = pd.DataFrame(all_results)
print("\n所有模型性能对比:")
print(results_df.to_string(index=False))

# 测试集性能排名
test_results = results_df[results_df['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\n测试集性能排名:")
print(test_results.to_string(index=False))

# 最佳模型
best_model_row = test_results.iloc[0]
print(f"\n最佳模型: {best_model_row['Model']}")
print(f"  窗口大小: {best_model_row['Window']}天")
print(f"  R² Score: {best_model_row['R²']:.4f}")
print(f"  RMSE: {best_model_row['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model_row['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model_row['MAPE']:.2f}%")

# ============================== 第11部分: 可视化 ==============================
print("\n" + "=" * 80)
print("第5步: 生成可视化图表")
print("=" * 80)

# 11.1 训练过程曲线（每个窗口）
for seq_length in sequence_lengths:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # 基础模型
    train_losses = all_models[f'basic_w{seq_length}']['train_losses']
    val_losses = all_models[f'basic_w{seq_length}']['val_losses']
    
    axes[0].plot(train_losses, label='训练Loss', linewidth=2)
    axes[0].plot(val_losses, label='验证Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title(f'CNN-LSTM基础模型训练过程 (窗口={seq_length}天)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 优化模型
    train_losses = all_models[f'optimized_w{seq_length}']['train_losses']
    val_losses = all_models[f'optimized_w{seq_length}']['val_losses']
    
    axes[1].plot(train_losses, label='训练Loss', linewidth=2)
    axes[1].plot(val_losses, label='验证Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss (MSE)', fontsize=12)
    axes[1].set_title(f'CNN-LSTM优化模型训练过程 (窗口={seq_length}天)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_window{seq_length}.png', dpi=300, bbox_inches='tight')
    print(f"保存: training_curves_window{seq_length}.png")
    plt.close()

# 11.2 预测vs实际值散点图（所有模型）
n_models = len(sequence_lengths) * 2
rows = len(sequence_lengths)
fig, axes = plt.subplots(rows, 2, figsize=(16, 5*rows))

plot_idx = 0
for i, seq_length in enumerate(sequence_lengths):
    for j, model_type in enumerate(['basic', 'optimized']):
        ax = axes[i, j] if rows > 1 else axes[j]
        
        test_actual, test_pred = all_predictions[f'{model_type}_w{seq_length}']['test']
        
        ax.scatter(test_actual, test_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
        
        min_val = min(test_actual.min(), test_pred.min())
        max_val = max(test_actual.max(), test_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测线')
        
        r2 = r2_score(test_actual, test_pred)
        rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
        
        ax.set_xlabel('实际PM2.5浓度 (μg/m³)', fontsize=11)
        ax.set_ylabel('预测PM2.5浓度 (μg/m³)', fontsize=11)
        ax.set_title(f'CNN-LSTM-{"基础" if model_type=="basic" else "优化"} (窗口={seq_length}天)\nR²={r2:.4f}, RMSE={rmse:.2f}', 
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
print("保存: prediction_scatter.png")
plt.close()

# 11.3 时间序列预测对比（最佳窗口）
best_window = best_model_row['Window']
fig, axes = plt.subplots(2, 1, figsize=(18, 10))

test_actual_best, test_pred_basic_best = all_predictions[f'basic_w{best_window}']['test']
_, test_pred_opt_best = all_predictions[f'optimized_w{best_window}']['test']

plot_range = min(300, len(test_actual_best))
plot_idx_range = range(len(test_actual_best) - plot_range, len(test_actual_best))

axes[0].plot(plot_idx_range, test_actual_best[plot_idx_range], 'k-', label='实际值', 
             linewidth=2, alpha=0.8)
axes[0].plot(plot_idx_range, test_pred_basic_best[plot_idx_range], 'b--', label='基础模型预测', 
             linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('样本索引', fontsize=12)
axes[0].set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
axes[0].set_title(f'CNN-LSTM基础模型 - 时间序列预测对比（窗口={best_window}天，最后{plot_range}个样本）', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(plot_idx_range, test_actual_best[plot_idx_range], 'k-', label='实际值', 
             linewidth=2, alpha=0.8)
axes[1].plot(plot_idx_range, test_pred_opt_best[plot_idx_range], 'g--', label='优化模型预测', 
             linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('样本索引', fontsize=12)
axes[1].set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
axes[1].set_title(f'CNN-LSTM优化模型 - 时间序列预测对比（窗口={best_window}天，最后{plot_range}个样本）', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'timeseries_comparison.png', dpi=300, bbox_inches='tight')
print("保存: timeseries_comparison.png")
plt.close()

# 11.4 残差分析
fig, axes = plt.subplots(rows, 2, figsize=(16, 5*rows))

for i, seq_length in enumerate(sequence_lengths):
    for j, model_type in enumerate(['basic', 'optimized']):
        ax = axes[i, j] if rows > 1 else axes[j]
        
        test_actual, test_pred = all_predictions[f'{model_type}_w{seq_length}']['test']
        residuals = test_actual - test_pred
        
        ax.scatter(test_pred, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('预测值 (μg/m³)', fontsize=11)
        ax.set_ylabel('残差 (μg/m³)', fontsize=11)
        ax.set_title(f'CNN-LSTM-{"基础" if model_type=="basic" else "优化"} (窗口={seq_length}天)\n残差均值={residuals.mean():.2f}, 标准差={residuals.std():.2f}', 
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
print("保存: residuals_analysis.png")
plt.close()

# 11.5 Attention特征重要性（平均序列位置权重）
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, seq_length in enumerate(sequence_lengths):
    avg_attention = all_attention_weights[f'w{seq_length}']
    
    axes[i].bar(range(len(avg_attention)), avg_attention, color='steelblue')
    axes[i].set_xlabel('序列位置（天数）', fontsize=12)
    axes[i].set_ylabel('Attention权重', fontsize=12)
    axes[i].set_title(f'Attention权重分布 (窗口={seq_length}天)', fontsize=13, fontweight='bold')
    axes[i].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'attention_weights.png', dpi=300, bbox_inches='tight')
print("保存: attention_weights.png")
plt.close()

# 11.6 模型性能对比（所有窗口和模型）
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

test_results_sorted = test_results.sort_values('Model')
x_pos = np.arange(len(test_results_sorted))
colors = ['blue', 'green', 'cyan', 'orange', 'purple', 'red']

metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
for i, metric in enumerate(metrics):
    axes[i].bar(x_pos, test_results_sorted[metric], color=colors[:len(test_results_sorted)], 
                alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[i].set_xticks(x_pos)
    axes[i].set_xticklabels([f"{m.split('_')[2]}\n{m.split('_')[1]}" for m in test_results_sorted['Model']], 
                             fontsize=9, rotation=0)
    axes[i].set_ylabel(metric, fontsize=12)
    
    if metric == 'R²':
        axes[i].set_title(f'{metric} 对比\n(越大越好)', fontsize=12, fontweight='bold')
    else:
        axes[i].set_title(f'{metric} 对比\n(越小越好)', fontsize=12, fontweight='bold')
    
    axes[i].grid(True, alpha=0.3, axis='y')
    
    for j, v in enumerate(test_results_sorted[metric]):
        if metric == 'MAPE':
            axes[i].text(j, v, f'{v:.1f}%', ha='center', va='bottom', 
                         fontsize=8, fontweight='bold')
        else:
            axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom', 
                         fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
print("保存: model_comparison.png")
plt.close()

# 11.7 误差分布（最佳窗口）
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

test_actual_best, test_pred_basic_best = all_predictions[f'basic_w{best_window}']['test']
_, test_pred_opt_best = all_predictions[f'optimized_w{best_window}']['test']

errors_basic = test_actual_best - test_pred_basic_best
errors_opt = test_actual_best - test_pred_opt_best

axes[0].hist(errors_basic, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='零误差')
axes[0].set_xlabel('预测误差 (μg/m³)', fontsize=12)
axes[0].set_ylabel('频数', fontsize=12)
axes[0].set_title(f'基础模型 - 预测误差分布 (窗口={best_window}天)\n均值={errors_basic.mean():.2f}, 标准差={errors_basic.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].hist(errors_opt, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='零误差')
axes[1].set_xlabel('预测误差 (μg/m³)', fontsize=12)
axes[1].set_ylabel('频数', fontsize=12)
axes[1].set_title(f'优化模型 - 预测误差分布 (窗口={best_window}天)\n均值={errors_opt.mean():.2f}, 标准差={errors_opt.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
print("保存: error_distribution.png")
plt.close()

# 11.8 窗口大小对比
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 提取每个窗口的测试集性能
window_comparison = []
for seq_length in sequence_lengths:
    basic_row = results_df[(results_df['Model'] == f'CNN-LSTM_Basic_W{seq_length}') & 
                           (results_df['Dataset'] == 'Test')].iloc[0]
    opt_row = results_df[(results_df['Model'] == f'CNN-LSTM_Optimized_W{seq_length}') & 
                         (results_df['Dataset'] == 'Test')].iloc[0]
    window_comparison.append({
        'Window': seq_length,
        'Basic_R2': basic_row['R²'],
        'Basic_RMSE': basic_row['RMSE'],
        'Opt_R2': opt_row['R²'],
        'Opt_RMSE': opt_row['RMSE']
    })

window_df = pd.DataFrame(window_comparison)

# R²对比
x = np.arange(len(sequence_lengths))
width = 0.35

axes[0].bar(x - width/2, window_df['Basic_R2'], width, label='基础模型', color='blue', alpha=0.7)
axes[0].bar(x + width/2, window_df['Opt_R2'], width, label='优化模型', color='green', alpha=0.7)
axes[0].set_xlabel('窗口大小（天）', fontsize=12)
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('不同窗口大小的R²对比', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'{w}天' for w in sequence_lengths])
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# RMSE对比
axes[1].bar(x - width/2, window_df['Basic_RMSE'], width, label='基础模型', color='blue', alpha=0.7)
axes[1].bar(x + width/2, window_df['Opt_RMSE'], width, label='优化模型', color='green', alpha=0.7)
axes[1].set_xlabel('窗口大小（天）', fontsize=12)
axes[1].set_ylabel('RMSE (μg/m³)', fontsize=12)
axes[1].set_title('不同窗口大小的RMSE对比', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'{w}天' for w in sequence_lengths])
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'window_size_comparison.png', dpi=300, bbox_inches='tight')
print("保存: window_size_comparison.png")
plt.close()

# ============================== 第12部分: 保存结果 ==============================
print("\n" + "=" * 80)
print("第6步: 保存结果")
print("=" * 80)

# 保存模型性能
results_df.to_csv(output_dir / 'model_performance.csv', index=False, encoding='utf-8-sig')
print("保存: model_performance.csv")

# 保存窗口对比
window_df.to_csv(output_dir / 'window_comparison.csv', index=False, encoding='utf-8-sig')
print("保存: window_comparison.csv")

# 保存Attention权重（每个窗口单独保存，因为长度不同）
for seq_length in sequence_lengths:
    attn_weights = all_attention_weights[f'w{seq_length}']
    actual_length = len(attn_weights)
    attention_df = pd.DataFrame({
        'Sequence_Position': range(actual_length),
        'Attention_Weight': attn_weights
    })
    attention_df.to_csv(output_dir / f'attention_weights_window{seq_length}.csv', 
                       index=False, encoding='utf-8-sig')
    print(f"保存: attention_weights_window{seq_length}.csv (实际长度: {actual_length})")

# 保存各窗口预测结果
for seq_length in sequence_lengths:
    for model_type in ['basic', 'optimized']:
        test_actual, test_pred = all_predictions[f'{model_type}_w{seq_length}']['test']
        pred_df = pd.DataFrame({
            'Actual': test_actual,
            'Prediction': test_pred,
            'Error': test_actual - test_pred
        })
        pred_df.to_csv(output_dir / f'predictions_{model_type}_window{seq_length}.csv', 
                      index=False, encoding='utf-8-sig')
        print(f"保存: predictions_{model_type}_window{seq_length}.csv")

# 保存最佳参数
best_params_list = []
for seq_length in sequence_lengths:
    params = all_models[f'optimized_w{seq_length}']['params'].copy()
    params['window'] = seq_length
    best_params_list.append(params)

best_params_df = pd.DataFrame(best_params_list)
best_params_df.to_csv(output_dir / 'best_parameters.csv', index=False, encoding='utf-8-sig')
print("保存: best_parameters.csv")

# ============================== 第13部分: 总结报告 ==============================
print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)

print("\n生成的文件:")
print("\nCSV文件:")
print("  - model_performance.csv           模型性能对比")
print("  - window_comparison.csv           窗口大小对比")
print("  - attention_weights_window*.csv   各窗口Attention权重")
print("  - predictions_*.csv               各模型预测结果")
print("  - best_parameters.csv             最佳参数")

print("\n图表文件:")
print("  - training_curves_window*.png     训练过程曲线（每个窗口）")
print("  - prediction_scatter.png          预测vs实际散点图")
print("  - timeseries_comparison.png       时间序列对比")
print("  - residuals_analysis.png          残差分析")
print("  - attention_weights.png           Attention权重分布")
print("  - model_comparison.png            模型性能对比")
print("  - error_distribution.png          误差分布")
print("  - window_size_comparison.png      窗口大小对比")

print("\n模型文件:")
for seq_length in sequence_lengths:
    print(f"  - cnn_lstm_window{seq_length}_optimized.pth   PyTorch模型权重")
    print(f"  - cnn_lstm_window{seq_length}_optimized.pkl   完整模型对象")
print("  - scaler_X.pkl                    特征标准化器")
print("  - scaler_y.pkl                    目标标准化器")

print(f"\n最佳模型: {best_model_row['Model']}")
print(f"  窗口大小: {best_model_row['Window']}天")
print(f"  R² Score: {best_model_row['R²']:.4f}")
print(f"  RMSE: {best_model_row['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model_row['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model_row['MAPE']:.2f}%")

print("\n各窗口测试集性能总结:")
for seq_length in sequence_lengths:
    opt_row = results_df[(results_df['Model'] == f'CNN-LSTM_Optimized_W{seq_length}') & 
                         (results_df['Dataset'] == 'Test')].iloc[0]
    print(f"  窗口={seq_length}天: R²={opt_row['R²']:.4f}, RMSE={opt_row['RMSE']:.2f}")

print("\n" + "=" * 80)
print("CNN-LSTM PM2.5浓度预测完成！")
print("=" * 80)

