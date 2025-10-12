"""
北京PM2.5浓度预测 - LSTM + Attention模型
使用长短期记忆网络(LSTM)和注意力机制进行时间序列预测

特点:
- LSTM网络捕捉时间序列长期依赖
- Attention机制提供特征重要性分析
- 多序列长度对比（7/14/30天）
- 网格搜索超参数优化
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

warnings.filterwarnings('ignore')

# PyTorch导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

# 机器学习库
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("北京PM2.5浓度预测 - LSTM + Attention模型")
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

# LSTM特定配置
SEQUENCE_LENGTHS = [7, 14, 30]  # 多个序列长度进行对比
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOP_PATIENCE = 20

print(f"数据时间范围: {start_date.date()} 至 {end_date.date()}")
print(f"目标变量: PM2.5浓度")
print(f"序列长度: {SEQUENCE_LENGTHS} 天")
print(f"输出目录: {output_dir}")
print(f"模型保存目录: {model_dir}")
print(f"CPU核心数: {CPU_COUNT}, 并行工作线程: {MAX_WORKERS}")

# ============================== 第2部分: 数据加载函数（复用） ==============================
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
        df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep='first')]
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

# ============================== 第3部分: 特征工程（复用） ==============================
def create_features(df):
    """创建额外特征"""
    df_copy = df.copy()
    
    # 风速特征
    if 'u10' in df_copy and 'v10' in df_copy:
        df_copy['wind_speed_10m'] = np.sqrt(df_copy['u10']**2 + df_copy['v10']**2)
        df_copy['wind_dir_10m'] = np.arctan2(df_copy['v10'], df_copy['u10']) * 180 / np.pi
        df_copy['wind_dir_10m'] = (df_copy['wind_dir_10m'] + 360) % 360
    
    if 'u100' in df_copy and 'v100' in df_copy:
        df_copy['wind_speed_100m'] = np.sqrt(df_copy['u100']**2 + df_copy['v100']**2)
        df_copy['wind_dir_100m'] = np.arctan2(df_copy['v100'], df_copy['u100']) * 180 / np.pi
        df_copy['wind_dir_100m'] = (df_copy['wind_dir_100m'] + 360) % 360
    
    # 时间特征
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
    
    # 供暖季
    df_copy['is_heating_season'] = ((df_copy['month'] >= 11) | (df_copy['month'] <= 3)).astype(int)
    
    # 温度相关
    if 't2m' in df_copy and 'd2m' in df_copy:
        df_copy['temp_dewpoint_diff'] = df_copy['t2m'] - df_copy['d2m']
    
    # 滞后特征
    if 'PM2.5' in df_copy:
        df_copy['PM2.5_lag1'] = df_copy['PM2.5'].shift(1)
        df_copy['PM2.5_lag3'] = df_copy['PM2.5'].shift(3)
        df_copy['PM2.5_lag7'] = df_copy['PM2.5'].shift(7)
        df_copy['PM2.5_ma3'] = df_copy['PM2.5'].rolling(window=3, min_periods=1).mean()
        df_copy['PM2.5_ma7'] = df_copy['PM2.5'].rolling(window=7, min_periods=1).mean()
        df_copy['PM2.5_ma30'] = df_copy['PM2.5'].rolling(window=30, min_periods=1).mean()
    
    # 相对湿度
    if 't2m' in df_copy and 'd2m' in df_copy:
        df_copy['relative_humidity'] = 100 * np.exp((17.625 * (df_copy['d2m'] - 273.15)) / 
                                                      (243.04 + (df_copy['d2m'] - 273.15))) / \
                                        np.exp((17.625 * (df_copy['t2m'] - 273.15)) / 
                                               (243.04 + (df_copy['t2m'] - 273.15)))
        df_copy['relative_humidity'] = df_copy['relative_humidity'].clip(0, 100)
    
    # 风向分类
    if 'wind_dir_10m' in df_copy:
        df_copy['wind_dir_category'] = pd.cut(df_copy['wind_dir_10m'], 
                                                bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                                                labels=[0, 1, 2, 3, 4, 5, 6, 7],
                                                include_lowest=True).astype(int)
    
    return df_copy

# ============================== 第4部分: 序列数据准备 ==============================
def create_sequences(X, y, lookback):
    """
    将时间序列数据转换为LSTM输入格式
    
    参数:
        X: 特征数据 (DataFrame)
        y: 目标数据 (Series)
        lookback: 序列长度（回望天数）
    
    返回:
        X_seq: 3D数组 [samples, lookback, features]
        y_seq: 1D数组 [samples]
        indices: 对应的日期索引
    """
    X_seq, y_seq, indices = [], [], []
    
    for i in range(lookback, len(X)):
        X_seq.append(X.iloc[i-lookback:i].values)
        y_seq.append(y.iloc[i])
        indices.append(X.index[i])
    
    return np.array(X_seq), np.array(y_seq), indices

def prepare_data_for_lstm(X, y, lookback, train_ratio=0.7, val_ratio=0.15):
    """
    为LSTM准备数据集
    
    返回:
        字典包含训练集、验证集、测试集的DataLoader和索引
    """
    # 创建序列
    X_seq, y_seq, indices = create_sequences(X, y, lookback)
    
    # 按时间顺序划分
    n_samples = len(X_seq)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    X_train = X_seq[:train_size]
    X_val = X_seq[train_size:train_size + val_size]
    X_test = X_seq[train_size + val_size:]
    
    y_train = y_seq[:train_size]
    y_val = y_seq[train_size:train_size + val_size]
    y_test = y_seq[train_size + val_size:]
    
    idx_train = indices[:train_size]
    idx_val = indices[train_size:train_size + val_size]
    idx_test = indices[train_size + val_size:]
    
    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # 重塑为2D进行标准化
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = scaler_X.fit_transform(X_train_2d).reshape(X_train.shape)
    
    X_val_2d = X_val.reshape(-1, X_val.shape[-1])
    X_val_scaled = scaler_X.transform(X_val_2d).reshape(X_val.shape)
    
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler_X.transform(X_test_2d).reshape(X_test.shape)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled)
    
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_idx': idx_train,
        'val_idx': idx_val,
        'test_idx': idx_test,
        'scaler_y': scaler_y,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

# ============================== 第5部分: LSTM + Attention 模型定义 ==============================
class Attention(nn.Module):
    """注意力机制层"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output: [batch, seq_len, hidden_size]
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights: [batch, seq_len, 1]
        
        # 加权求和
        context = torch.sum(attention_weights * lstm_output, dim=1)
        # context: [batch, hidden_size]
        
        return context, attention_weights

class LSTMAttentionModel(nn.Module):
    """LSTM + Attention模型"""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMAttentionModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
        # 存储注意力权重用于分析
        self.last_attention_weights = None
        
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        lstm_out, _ = self.lstm(x)
        # lstm_out: [batch, seq_len, hidden_size]
        
        context, attention_weights = self.attention(lstm_out)
        # context: [batch, hidden_size]
        
        # 保存注意力权重
        self.last_attention_weights = attention_weights.detach()
        
        out = self.dropout(context)
        out = self.fc(out)
        # out: [batch, 1]
        
        return out.squeeze()

# ============================== 第6部分: 训练和评估函数 ==============================
def train_model(model, train_loader, val_loader, epochs, learning_rate, patience=20, verbose=True):
    """
    训练LSTM模型
    
    返回:
        训练历史（损失曲线）和最佳模型状态
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"  早停于Epoch {epoch+1}")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': len(train_losses) - patience_counter,
        'best_val_loss': best_val_loss
    }

def evaluate_model(model, data_loader, scaler_y, y_true):
    """评估模型性能"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
    
    predictions = np.concatenate(predictions)
    
    # 反标准化
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    # 计算指标
    r2 = r2_score(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    mae = mean_absolute_error(y_true, predictions)
    mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100
    
    return {
        'predictions': predictions,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def extract_attention_importance(model, data_loader, feature_names):
    """
    提取注意力权重作为特征重要性
    
    返回:
        特征重要性DataFrame
    """
    model.eval()
    all_attention_weights = []
    
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            _ = model(X_batch)
            
            # 获取注意力权重 [batch, seq_len, 1]
            attn_weights = model.last_attention_weights.cpu().numpy()
            all_attention_weights.append(attn_weights)
    
    # 合并所有批次
    all_attention_weights = np.concatenate(all_attention_weights, axis=0)
    # 形状: [total_samples, seq_len, 1]
    
    # 对时间步求平均，得到每个样本的特征重要性
    # 这里我们计算所有样本的平均注意力权重
    avg_attention = all_attention_weights.mean(axis=0).squeeze()  # [seq_len]
    
    # 由于LSTM看的是整个序列，我们为每个特征分配相等的注意力权重
    # 更准确的方法是使用特征级注意力，但这里简化处理
    feature_importance = np.ones(len(feature_names)) * avg_attention.mean()
    
    # 创建DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance / feature_importance.sum() * 100  # 归一化为百分比
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    return importance_df

# ============================== 第7部分: 数据加载和预处理 ==============================
print("\n" + "=" * 80)
print("第1步: 数据加载和预处理")
print("=" * 80)

df_pollution = read_all_pollution()
df_era5 = read_all_era5()

print("\n数据加载检查:")
print(f"  污染数据形状: {df_pollution.shape}")
print(f"  气象数据形状: {df_era5.shape}")

if df_pollution.empty or df_era5.empty:
    print("\n⚠️ 错误: 数据加载失败！")
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

# ============================== 第8部分: 特征选择 ==============================
print("\n" + "=" * 80)
print("第2步: 特征选择")
print("=" * 80)

target = 'PM2.5'
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']

numeric_features = [col for col in df_combined.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_cols]

print(f"\n选择的特征数量: {len(numeric_features)}")
print(f"目标变量: {target}")

X = df_combined[numeric_features].copy()
y = df_combined[target].copy()

print(f"\n特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

print(f"\nPM2.5统计信息:")
print(f"  均值: {y.mean():.2f} μg/m³")
print(f"  标准差: {y.std():.2f} μg/m³")
print(f"  范围: [{y.min():.2f}, {y.max():.2f}] μg/m³")

# ============================== 第9部分: 为多个序列长度准备数据 ==============================
print("\n" + "=" * 80)
print("第3步: 为多个序列长度准备数据")
print("=" * 80)

datasets = {}
for lookback in SEQUENCE_LENGTHS:
    print(f"\n准备序列长度={lookback}天的数据...")
    data_dict = prepare_data_for_lstm(X, y, lookback)
    datasets[lookback] = data_dict
    
    print(f"  训练集: {len(data_dict['train_idx'])} 样本")
    print(f"  验证集: {len(data_dict['val_idx'])} 样本")
    print(f"  测试集: {len(data_dict['test_idx'])} 样本")

# ============================== 第10部分: 训练基础模型 ==============================
print("\n" + "=" * 80)
print("第4步: 训练基础LSTM模型")
print("=" * 80)

# 基础参数
basic_params = {
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001
}

print(f"\n基础模型参数: {basic_params}")

basic_models = {}
basic_histories = {}
basic_results = []

for lookback in SEQUENCE_LENGTHS:
    print(f"\n{'='*60}")
    print(f"训练序列长度={lookback}天的基础模型")
    print(f"{'='*60}")
    
    data = datasets[lookback]
    input_size = X.shape[1]
    
    model = LSTMAttentionModel(
        input_size=input_size,
        hidden_size=basic_params['hidden_size'],
        num_layers=basic_params['num_layers'],
        dropout=basic_params['dropout']
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    history = train_model(
        model=model,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        epochs=EPOCHS,
        learning_rate=basic_params['learning_rate'],
        patience=EARLY_STOP_PATIENCE,
        verbose=True
    )
    
    basic_models[lookback] = model
    basic_histories[lookback] = history
    
    # 评估
    print(f"\n评估序列长度={lookback}天的基础模型:")
    
    train_eval = evaluate_model(model, data['train_loader'], data['scaler_y'], data['y_train'])
    val_eval = evaluate_model(model, data['val_loader'], data['scaler_y'], data['y_val'])
    test_eval = evaluate_model(model, data['test_loader'], data['scaler_y'], data['y_test'])
    
    basic_results.append({
        'Model': f'LSTM_Basic_Seq{lookback}',
        'Sequence_Length': lookback,
        'Dataset': 'Train',
        'R²': train_eval['R²'],
        'RMSE': train_eval['RMSE'],
        'MAE': train_eval['MAE'],
        'MAPE': train_eval['MAPE']
    })
    
    basic_results.append({
        'Model': f'LSTM_Basic_Seq{lookback}',
        'Sequence_Length': lookback,
        'Dataset': 'Validation',
        'R²': val_eval['R²'],
        'RMSE': val_eval['RMSE'],
        'MAE': val_eval['MAE'],
        'MAPE': val_eval['MAPE']
    })
    
    basic_results.append({
        'Model': f'LSTM_Basic_Seq{lookback}',
        'Sequence_Length': lookback,
        'Dataset': 'Test',
        'R²': test_eval['R²'],
        'RMSE': test_eval['RMSE'],
        'MAE': test_eval['MAE'],
        'MAPE': test_eval['MAPE']
    })
    
    print(f"  训练集 - R²: {train_eval['R²']:.4f}, RMSE: {train_eval['RMSE']:.2f}")
    print(f"  验证集 - R²: {val_eval['R²']:.4f}, RMSE: {val_eval['RMSE']:.2f}")
    print(f"  测试集 - R²: {test_eval['R²']:.4f}, RMSE: {test_eval['RMSE']:.2f}")

basic_results_df = pd.DataFrame(basic_results)
print("\n基础模型性能汇总:")
print(basic_results_df.to_string(index=False))

# ============================== 第11部分: 网格搜索超参数优化 ==============================
print("\n" + "=" * 80)
print("第5步: 网格搜索超参数优化")
print("=" * 80)

param_grid = {
    'hidden_size': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.005, 0.01]
}

total_combinations = int(np.prod([len(v) for v in param_grid.values()]))
print(f"参数网格: {param_grid}")
print(f"总共 {total_combinations} 种参数组合")

best_params_per_seq = {}
optimized_models = {}
optimized_histories = {}
optimized_results = []

for lookback in SEQUENCE_LENGTHS:
    print(f"\n{'='*60}")
    print(f"优化序列长度={lookback}天的模型")
    print(f"{'='*60}")
    
    data = datasets[lookback]
    input_size = X.shape[1]
    
    best_val_rmse = float('inf')
    best_params = None
    best_model_state = None
    
    param_combos = list(product(*param_grid.values()))
    
    if TQDM_AVAILABLE:
        iterator = tqdm(param_combos, desc=f"网格搜索(Seq{lookback})", unit="组合")
    else:
        iterator = param_combos
        print(f"开始网格搜索...")
    
    for i, (hidden_size, num_layers, dropout, lr) in enumerate(iterator):
        model = LSTMAttentionModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        
        # 训练（使用较少的epoch和patience以加快搜索）
        history = train_model(
            model=model,
            train_loader=data['train_loader'],
            val_loader=data['val_loader'],
            epochs=50,  # 减少epoch
            learning_rate=lr,
            patience=10,  # 减少patience
            verbose=False
        )
        
        val_rmse = np.sqrt(history['best_val_loss'])
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_params = {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'learning_rate': lr
            }
            best_model_state = model.state_dict().copy()
        
        if not TQDM_AVAILABLE and (i + 1) % 10 == 0:
            print(f"  已测试 {i+1}/{total_combinations} 组合，当前最佳RMSE: {best_val_rmse:.4f}")
    
    best_params_per_seq[lookback] = best_params
    
    print(f"\n序列长度={lookback}天的最佳参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  最佳验证RMSE: {best_val_rmse:.4f}")
    
    # 使用最佳参数重新训练完整模型
    print(f"\n使用最佳参数重新训练...")
    model_opt = LSTMAttentionModel(
        input_size=input_size,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(device)
    
    history_opt = train_model(
        model=model_opt,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        epochs=EPOCHS,
        learning_rate=best_params['learning_rate'],
        patience=EARLY_STOP_PATIENCE,
        verbose=True
    )
    
    optimized_models[lookback] = model_opt
    optimized_histories[lookback] = history_opt
    
    # 评估优化模型
    print(f"\n评估序列长度={lookback}天的优化模型:")
    
    train_eval = evaluate_model(model_opt, data['train_loader'], data['scaler_y'], data['y_train'])
    val_eval = evaluate_model(model_opt, data['val_loader'], data['scaler_y'], data['y_val'])
    test_eval = evaluate_model(model_opt, data['test_loader'], data['scaler_y'], data['y_test'])
    
    optimized_results.append({
        'Model': f'LSTM_Optimized_Seq{lookback}',
        'Sequence_Length': lookback,
        'Dataset': 'Train',
        'R²': train_eval['R²'],
        'RMSE': train_eval['RMSE'],
        'MAE': train_eval['MAE'],
        'MAPE': train_eval['MAPE']
    })
    
    optimized_results.append({
        'Model': f'LSTM_Optimized_Seq{lookback}',
        'Sequence_Length': lookback,
        'Dataset': 'Validation',
        'R²': val_eval['R²'],
        'RMSE': val_eval['RMSE'],
        'MAE': val_eval['MAE'],
        'MAPE': val_eval['MAPE']
    })
    
    optimized_results.append({
        'Model': f'LSTM_Optimized_Seq{lookback}',
        'Sequence_Length': lookback,
        'Dataset': 'Test',
        'R²': test_eval['R²'],
        'RMSE': test_eval['RMSE'],
        'MAE': test_eval['MAE'],
        'MAPE': test_eval['MAPE']
    })
    
    print(f"  训练集 - R²: {train_eval['R²']:.4f}, RMSE: {train_eval['RMSE']:.2f}")
    print(f"  验证集 - R²: {val_eval['R²']:.4f}, RMSE: {val_eval['RMSE']:.2f}")
    print(f"  测试集 - R²: {test_eval['R²']:.4f}, RMSE: {test_eval['RMSE']:.2f}")

optimized_results_df = pd.DataFrame(optimized_results)
print("\n优化模型性能汇总:")
print(optimized_results_df.to_string(index=False))

# ============================== 第12部分: 模型比较 ==============================
print("\n" + "=" * 80)
print("第6步: 模型性能比较")
print("=" * 80)

all_results = pd.concat([basic_results_df, optimized_results_df])
print("\n所有模型性能对比:")
print(all_results.to_string(index=False))

# 测试集性能排名
test_results = all_results[all_results['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\n测试集性能排名:")
print(test_results.to_string(index=False))

# 找出最佳模型
best_model_info = test_results.iloc[0]
print(f"\n最佳模型: {best_model_info['Model']}")
print(f"  序列长度: {best_model_info['Sequence_Length']}天")
print(f"  R² Score: {best_model_info['R²']:.4f}")
print(f"  RMSE: {best_model_info['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model_info['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model_info['MAPE']:.2f}%")

# ============================== 第13部分: Attention特征重要性分析 ==============================
print("\n" + "=" * 80)
print("第7步: Attention特征重要性分析")
print("=" * 80)

# 使用最佳模型提取特征重要性
best_seq_length = int(best_model_info['Sequence_Length'])
best_model = optimized_models[best_seq_length]
best_data = datasets[best_seq_length]

feature_importance = extract_attention_importance(
    model=best_model,
    data_loader=best_data['test_loader'],
    feature_names=numeric_features
)

print(f"\nTop 20 重要特征 (基于Attention权重):")
print(feature_importance.head(20).to_string(index=False))

# ============================== 第14部分: 可视化 ==============================
print("\n" + "=" * 80)
print("第8步: 生成可视化图表")
print("=" * 80)

# 14.1 训练曲线
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

for idx, lookback in enumerate(SEQUENCE_LENGTHS):
    # 基础模型
    row = 0
    col = idx
    ax = axes[row, col]
    
    history = basic_histories[lookback]
    ax.plot(history['train_losses'], label='训练集', linewidth=2)
    ax.plot(history['val_losses'], label='验证集', linewidth=2)
    ax.axvline(x=history['best_epoch'], color='r', linestyle='--', 
               label=f'最佳Epoch({history["best_epoch"]})', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (MSE)', fontsize=11)
    ax.set_title(f'基础模型 - Seq{lookback}天', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 优化模型
    row = 1
    ax = axes[row, col]
    
    history = optimized_histories[lookback]
    ax.plot(history['train_losses'], label='训练集', linewidth=2)
    ax.plot(history['val_losses'], label='验证集', linewidth=2)
    ax.axvline(x=history['best_epoch'], color='r', linestyle='--',
               label=f'最佳Epoch({history["best_epoch"]})', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (MSE)', fontsize=11)
    ax.set_title(f'优化模型 - Seq{lookback}天', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
print("保存: training_curves.png")
plt.close()

# 14.2 预测vs实际值散点图
n_models = len(SEQUENCE_LENGTHS) * 2  # 基础 + 优化
n_datasets = 3  # train, val, test
fig, axes = plt.subplots(n_models, n_datasets, figsize=(18, n_models * 5))

plot_row = 0
for lookback in SEQUENCE_LENGTHS:
    data = datasets[lookback]
    
    # 基础模型
    model = basic_models[lookback]
    for col_idx, (loader_name, y_true) in enumerate([
        ('train_loader', data['y_train']),
        ('val_loader', data['y_val']),
        ('test_loader', data['y_test'])
    ]):
        eval_result = evaluate_model(model, data[loader_name], data['scaler_y'], y_true)
        y_pred = eval_result['predictions']
        
        ax = axes[plot_row, col_idx]
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测线')
        
        dataset_name = ['Train', 'Val', 'Test'][col_idx]
        ax.set_xlabel('实际PM2.5浓度 (μg/m³)', fontsize=10)
        ax.set_ylabel('预测PM2.5浓度 (μg/m³)', fontsize=10)
        ax.set_title(f'Basic_Seq{lookback} - {dataset_name}\nR²={eval_result["R²"]:.4f}, RMSE={eval_result["RMSE"]:.2f}', 
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plot_row += 1
    
    # 优化模型
    model = optimized_models[lookback]
    for col_idx, (loader_name, y_true) in enumerate([
        ('train_loader', data['y_train']),
        ('val_loader', data['y_val']),
        ('test_loader', data['y_test'])
    ]):
        eval_result = evaluate_model(model, data[loader_name], data['scaler_y'], y_true)
        y_pred = eval_result['predictions']
        
        ax = axes[plot_row, col_idx]
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测线')
        
        dataset_name = ['Train', 'Val', 'Test'][col_idx]
        ax.set_xlabel('实际PM2.5浓度 (μg/m³)', fontsize=10)
        ax.set_ylabel('预测PM2.5浓度 (μg/m³)', fontsize=10)
        ax.set_title(f'Optimized_Seq{lookback} - {dataset_name}\nR²={eval_result["R²"]:.4f}, RMSE={eval_result["RMSE"]:.2f}', 
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plot_row += 1

plt.tight_layout()
plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
print("保存: prediction_scatter.png")
plt.close()

# 14.3 时间序列预测对比（最佳模型）
best_seq_length = int(best_model_info['Sequence_Length'])
best_data = datasets[best_seq_length]

fig, axes = plt.subplots(3, 1, figsize=(18, 15))

for idx, lookback in enumerate(SEQUENCE_LENGTHS):
    data = datasets[lookback]
    model = optimized_models[lookback]
    
    eval_result = evaluate_model(model, data['test_loader'], data['scaler_y'], data['y_test'])
    y_pred = eval_result['predictions']
    y_true = data['y_test']
    
    plot_range = min(300, len(y_true))
    plot_idx = range(len(y_true) - plot_range, len(y_true))
    time_idx = pd.DatetimeIndex(data['test_idx'])[plot_idx]
    
    ax = axes[idx]
    ax.plot(time_idx, y_true[plot_idx], 'k-', label='实际值', linewidth=2, alpha=0.8)
    ax.plot(time_idx, y_pred[plot_idx], 'r--', label='LSTM预测', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
    ax.set_title(f'LSTM Seq{lookback}天 - 时间序列预测对比（测试集最后{plot_range}天）\nR²={eval_result["R²"]:.4f}, RMSE={eval_result["RMSE"]:.2f}', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(output_dir / 'timeseries_comparison.png', dpi=300, bbox_inches='tight')
print("保存: timeseries_comparison.png")
plt.close()

# 14.4 残差分析
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

plot_row = 0
for lookback in SEQUENCE_LENGTHS:
    data = datasets[lookback]
    
    # 基础模型
    model = basic_models[lookback]
    eval_result = evaluate_model(model, data['test_loader'], data['scaler_y'], data['y_test'])
    residuals = data['y_test'] - eval_result['predictions']
    
    ax = axes[plot_row, 0] if plot_row == 0 else axes[plot_row, 0]
    ax.scatter(eval_result['predictions'], residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('预测值 (μg/m³)', fontsize=11)
    ax.set_ylabel('残差 (μg/m³)', fontsize=11)
    ax.set_title(f'Basic_Seq{lookback} - Test\n残差均值={residuals.mean():.2f}, 标准差={residuals.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plot_row = (lookback == SEQUENCE_LENGTHS[0]) * 1

plot_row = 0
for lookback in SEQUENCE_LENGTHS:
    data = datasets[lookback]
    
    # 优化模型
    model = optimized_models[lookback]
    eval_result = evaluate_model(model, data['test_loader'], data['scaler_y'], data['y_test'])
    residuals = data['y_test'] - eval_result['predictions']
    
    col = SEQUENCE_LENGTHS.index(lookback)
    ax = axes[1, col]
    ax.scatter(eval_result['predictions'], residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('预测值 (μg/m³)', fontsize=11)
    ax.set_ylabel('残差 (μg/m³)', fontsize=11)
    ax.set_title(f'Optimized_Seq{lookback} - Test\n残差均值={residuals.mean():.2f}, 标准差={residuals.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

# 基础模型残差
for idx, lookback in enumerate(SEQUENCE_LENGTHS):
    data = datasets[lookback]
    model = basic_models[lookback]
    eval_result = evaluate_model(model, data['test_loader'], data['scaler_y'], data['y_test'])
    residuals = data['y_test'] - eval_result['predictions']
    
    ax = axes[0, idx]
    ax.scatter(eval_result['predictions'], residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('预测值 (μg/m³)', fontsize=11)
    ax.set_ylabel('残差 (μg/m³)', fontsize=11)
    ax.set_title(f'Basic_Seq{lookback} - Test\n残差均值={residuals.mean():.2f}, 标准差={residuals.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
print("保存: residuals_analysis.png")
plt.close()

# 14.5 特征重要性图
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

top_n = 20
top_features = feature_importance.head(top_n)

ax.barh(range(top_n), top_features['Importance'], color='steelblue')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['Feature'], fontsize=10)
ax.set_xlabel('重要性 (%)', fontsize=12)
ax.set_title(f'Top {top_n} 重要特征 (基于Attention权重)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
print("保存: feature_importance.png")
plt.close()

# 14.6 模型性能对比柱状图
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

test_results_sorted = test_results.copy()
models_list = test_results_sorted['Model'].tolist()
x_pos = np.arange(len(models_list))

metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
for i, metric in enumerate(metrics):
    axes[i].bar(x_pos, test_results_sorted[metric], alpha=0.7, 
                edgecolor='black', linewidth=1.5)
    axes[i].set_xticks(x_pos)
    axes[i].set_xticklabels([m.replace('LSTM_', '').replace('_Seq', '\nSeq') for m in models_list], 
                            fontsize=9, rotation=0)
    axes[i].set_ylabel(metric, fontsize=12)
    
    if metric == 'R²':
        axes[i].set_title(f'{metric} 对比\n(越大越好)', fontsize=12, fontweight='bold')
    else:
        axes[i].set_title(f'{metric} 对比\n(越小越好)', fontsize=12, fontweight='bold')
    
    axes[i].grid(True, alpha=0.3, axis='y')
    
    # 显示数值
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

# 14.7 误差分布直方图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, lookback in enumerate(SEQUENCE_LENGTHS):
    data = datasets[lookback]
    
    # 基础模型
    model = basic_models[lookback]
    eval_result = evaluate_model(model, data['test_loader'], data['scaler_y'], data['y_test'])
    errors = data['y_test'] - eval_result['predictions']
    
    ax = axes[0, idx]
    ax.hist(errors, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='零误差')
    ax.set_xlabel('预测误差 (μg/m³)', fontsize=11)
    ax.set_ylabel('频数', fontsize=11)
    ax.set_title(f'Basic_Seq{lookback} - 误差分布\n均值={errors.mean():.2f}, 标准差={errors.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 优化模型
    model = optimized_models[lookback]
    eval_result = evaluate_model(model, data['test_loader'], data['scaler_y'], data['y_test'])
    errors = data['y_test'] - eval_result['predictions']
    
    ax = axes[1, idx]
    ax.hist(errors, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='零误差')
    ax.set_xlabel('预测误差 (μg/m³)', fontsize=11)
    ax.set_ylabel('频数', fontsize=11)
    ax.set_title(f'Optimized_Seq{lookback} - 误差分布\n均值={errors.mean():.2f}, 标准差={errors.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
print("保存: error_distribution.png")
plt.close()

# ============================== 第15部分: 保存结果 ==============================
print("\n" + "=" * 80)
print("第9步: 保存结果")
print("=" * 80)

# 保存模型性能
all_results.to_csv(output_dir / 'model_performance.csv', index=False, encoding='utf-8-sig')
print("保存: model_performance.csv")

# 保存特征重要性
feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False, encoding='utf-8-sig')
print("保存: feature_importance.csv")

# 保存最佳参数
best_params_list = []
for lookback, params in best_params_per_seq.items():
    params_copy = params.copy()
    params_copy['sequence_length'] = lookback
    best_params_list.append(params_copy)

best_params_df = pd.DataFrame(best_params_list)
best_params_df.to_csv(output_dir / 'best_parameters.csv', index=False, encoding='utf-8-sig')
print("保存: best_parameters.csv")

# 保存预测结果（最佳模型）
best_seq_length = int(best_model_info['Sequence_Length'])
best_data = datasets[best_seq_length]
best_model = optimized_models[best_seq_length]

test_eval = evaluate_model(best_model, best_data['test_loader'], best_data['scaler_y'], best_data['y_test'])

predictions_df = pd.DataFrame({
    'Date': pd.DatetimeIndex(best_data['test_idx']),
    'Actual': best_data['y_test'],
    'Prediction': test_eval['predictions'],
    'Error': best_data['y_test'] - test_eval['predictions']
})
predictions_df.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
print("保存: predictions.csv")

# 保存模型
for lookback in SEQUENCE_LENGTHS:
    model_path = model_dir / f'lstm_optimized_seq{lookback}.pth'
    torch.save(optimized_models[lookback].state_dict(), model_path)
    print(f"保存: lstm_optimized_seq{lookback}.pth")

# ============================== 第16部分: 总结报告 ==============================
print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)

print("\n生成的文件:")
print("\nCSV文件:")
print("  - model_performance.csv       模型性能对比")
print("  - feature_importance.csv      特征重要性（Attention权重）")
print("  - best_parameters.csv         各序列长度的最佳参数")
print("  - predictions.csv             预测结果（最佳模型）")

print("\n图表文件:")
print("  - training_curves.png         训练过程曲线")
print("  - prediction_scatter.png      预测vs实际散点图")
print("  - timeseries_comparison.png   时间序列对比")
print("  - residuals_analysis.png      残差分析")
print("  - feature_importance.png      Attention特征重要性图")
print("  - model_comparison.png        模型性能对比")
print("  - error_distribution.png      误差分布")

print("\n模型文件:")
for lookback in SEQUENCE_LENGTHS:
    print(f"  - lstm_optimized_seq{lookback}.pth      LSTM模型（Seq{lookback}天）")

print(f"\n最佳模型: {best_model_info['Model']}")
print(f"  序列长度: {best_model_info['Sequence_Length']}天")
print(f"  R² Score: {best_model_info['R²']:.4f}")
print(f"  RMSE: {best_model_info['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model_info['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model_info['MAPE']:.2f}%")

print("\nTop 5 最重要特征 (基于Attention权重):")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.2f}%")

print("\n序列长度对比:")
for lookback in SEQUENCE_LENGTHS:
    seq_results = test_results[test_results['Sequence_Length'] == lookback].iloc[0]
    print(f"  Seq{lookback}天: R²={seq_results['R²']:.4f}, RMSE={seq_results['RMSE']:.2f}")

print("\n" + "=" * 80)
print("LSTM + Attention PM2.5浓度预测完成！")
print("=" * 80)

