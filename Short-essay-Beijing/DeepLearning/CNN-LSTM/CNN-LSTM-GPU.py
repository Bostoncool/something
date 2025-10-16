"""
Beijing PM2.5 Concentration Prediction - CNN-LSTM + Attention Model (GPU Accelerated Version)
Using 1D CNN + LSTM + Attention for time series prediction

GPU Acceleration Features:
- Automatic Mixed Precision Training (AMP) - 2-3x speed improvement
- Optimized data loading (pin_memory, multi-process)
- Larger batch size (128)
- GPU memory monitoring and management
- Gradient clipping to prevent explosion
- Training speed statistics

Features:
- 1D CNN extracts local temporal patterns
- LSTM captures long-term temporal dependencies
- Attention mechanism provides feature importance analysis
- Multiple sequence length comparison (7/14/30 days)
- Hyperparameter optimization (Bayesian optimization/grid search)
- Complete training, evaluation and visualization workflow

Data Sources:
- Pollution data: Benchmark dataset (PM2.5, PM10, SO2, NO2, CO, O3)
- Meteorological data: ERA5 reanalysis data
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

# PyTorch related
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

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
    print("      Use 'pip install tqdm' to get better progress bar display.")

# Machine learning libraries
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Bayesian optimization (optional)
try:
    from bayes_opt import BayesianOptimization
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    print("Note: bayesian-optimization not installed, will use grid search.")
    print("      Use 'pip install bayesian-optimization' to enable Bayesian optimization.")
    BAYESIAN_OPT_AVAILABLE = False

# Set English fonts
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# Set random seed
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # Multi-GPU support
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # GPU acceleration: auto find optimal convolution algorithm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - CNN-LSTM + Attention Model (GPU Accelerated Version)")
print("=" * 80)
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"Memory: {gpu_props.total_memory / 1e9:.2f} GB")
    print(f"Compute capability: {gpu_props.major}.{gpu_props.minor}")
    print(f"Multiprocessor count: {gpu_props.multi_processor_count}")
    print(f"Mixed precision training (AMP): Enabled ✓")
else:
    print("⚠️  Warning: GPU not detected, will use CPU mode (slower speed)")
    print("   Mixed precision training: Disabled")

# ============================== Part 1: Configuration and Path Setup ==============================
print("\nConfiguring parameters...")

# Data paths
pollution_all_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)'
era5_path = r'C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV'

# Output path
output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)

# Model save path
model_dir = Path('./models')
model_dir.mkdir(exist_ok=True)

# Date range
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

# Beijing geographic range
beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

# Pollutant list
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# ERA5 variables
era5_vars = [
    'd2m', 't2m', 'u10', 'v10', 'u100', 'v100',
    'blh', 'sp', 'tcwv',
    'tp', 'avg_tprate',
    'tisr', 'str',
    'cvh', 'cvl',
    'mn2t', 'sd', 'lsm'
]

# Sequence lengths (multi-window comparison)
sequence_lengths = [7, 14, 30]

# GPU optimization parameters
BATCH_SIZE = 128 if torch.cuda.is_available() else 64  # GPU uses larger batch
NUM_WORKERS = 4  # Number of data loading processes
PIN_MEMORY = torch.cuda.is_available()  # Pinned memory for faster transfer
PERSISTENT_WORKERS = True if torch.cuda.is_available() else False  # Keep worker processes alive

print(f"Data time range: {start_date.date()} to {end_date.date()}")
print(f"Target variable: PM2.5 concentration")
print(f"Sequence lengths: {sequence_lengths} days")
print(f"Output directory: {output_dir}")
print(f"Model save directory: {model_dir}")
print(f"CPU cores: {CPU_COUNT}, parallel workers: {MAX_WORKERS}")
print(f"\nGPU optimization parameters:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Data loading processes: {NUM_WORKERS}")
print(f"  Pin Memory: {PIN_MEMORY}")
print(f"  Persistent workers: {PERSISTENT_WORKERS}")

# ============================== Part 2: Data Loading Functions ==============================
def daterange(start, end):
    """Generate date sequence"""
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def find_file(base_path, date_str, prefix):
    """Find file for specified date"""
    filename = f"{prefix}_{date_str}.csv"
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def read_pollution_day(date):
    """Read single day pollution data"""
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
    """Read all pollution data in parallel"""
    print("\nLoading pollution data...")
    print(f"Using {MAX_WORKERS} parallel workers")
    dates = list(daterange(start_date, end_date))
    pollution_dfs = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, date): date for date in dates}
        
        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Loading pollution data", unit="day"):
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
        print(f"  Successfully read {len(pollution_dfs)}/{len(dates)} days of data")
        print("  Merging data...")
        df_poll_all = pd.concat(pollution_dfs)
        df_poll_all.ffill(inplace=True)
        df_poll_all.fillna(df_poll_all.mean(), inplace=True)
        print(f"Pollution data loading complete, shape: {df_poll_all.shape}")
        return df_poll_all
    return pd.DataFrame()

def read_era5_month(year, month):
    """Read single month ERA5 data"""
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
        return monthly_data
    else:
        return None

def read_all_era5():
    """Read all ERA5 data in parallel"""
    print("\nLoading meteorological data...")
    print(f"Using {MAX_WORKERS} parallel workers")
    
    era5_dfs = []
    years = range(2015, 2025)
    months = range(1, 13)
    
    month_tasks = [(year, month) for year in years for month in months 
                   if not (year == 2024 and month > 12)]
    total_months = len(month_tasks)
    print(f"Attempting to load {total_months} months of data...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_era5_month, year, month): (year, month) 
                  for year, month in month_tasks}
        
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
                    successful_reads += 1
                if i % 20 == 0 or i == len(futures):
                    print(f"  Progress: {i}/{len(futures)} months (success: {successful_reads}, {i/len(futures)*100:.1f}%)")
        
        print(f"  Total successfully read: {successful_reads}/{len(futures)} months")
    
    if era5_dfs:
        print("\nMerging meteorological data...")
        df_era5_all = pd.concat(era5_dfs, axis=0)
        
        print("  Removing duplicates...")
        df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep='first')]
        
        print("  Sorting...")
        df_era5_all.sort_index(inplace=True)
        
        print(f"Merged shape: {df_era5_all.shape}")
        print(f"Time range: {df_era5_all.index.min()} to {df_era5_all.index.max()}")
        
        print("  Handling missing values...")
        initial_na = df_era5_all.isna().sum().sum()
        df_era5_all.ffill(inplace=True)
        df_era5_all.bfill(inplace=True)
        df_era5_all.fillna(df_era5_all.mean(), inplace=True)
        final_na = df_era5_all.isna().sum().sum()
        
        print(f"Missing value handling: {initial_na} -> {final_na}")
        print(f"Meteorological data loading complete, shape: {df_era5_all.shape}")
        
        return df_era5_all
    else:
        print("\n❌ Error: Failed to load any meteorological data files!")
        return pd.DataFrame()

# ============================== Part 3: Feature Engineering ==============================
def create_features(df):
    """Create additional features"""
    df_copy = df.copy()
    
    # 1. Wind speed features
    if 'u10' in df_copy and 'v10' in df_copy:
        df_copy['wind_speed_10m'] = np.sqrt(df_copy['u10']**2 + df_copy['v10']**2)
        df_copy['wind_dir_10m'] = np.arctan2(df_copy['v10'], df_copy['u10']) * 180 / np.pi
        df_copy['wind_dir_10m'] = (df_copy['wind_dir_10m'] + 360) % 360
    
    if 'u100' in df_copy and 'v100' in df_copy:
        df_copy['wind_speed_100m'] = np.sqrt(df_copy['u100']**2 + df_copy['v100']**2)
        df_copy['wind_dir_100m'] = np.arctan2(df_copy['v100'], df_copy['u100']) * 180 / np.pi
        df_copy['wind_dir_100m'] = (df_copy['wind_dir_100m'] + 360) % 360
    
    # 2. Time features
    df_copy['year'] = df_copy.index.year
    df_copy['month'] = df_copy.index.month
    df_copy['day'] = df_copy.index.day
    df_copy['day_of_year'] = df_copy.index.dayofyear
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['week_of_year'] = df_copy.index.isocalendar().week
    
    # Season feature
    df_copy['season'] = df_copy['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )
    
    # Heating season indicator
    df_copy['is_heating_season'] = ((df_copy['month'] >= 11) | (df_copy['month'] <= 3)).astype(int)
    
    # 3. Temperature related features
    if 't2m' in df_copy and 'd2m' in df_copy:
        df_copy['temp_dewpoint_diff'] = df_copy['t2m'] - df_copy['d2m']
    
    # 4. Lag features
    if 'PM2.5' in df_copy:
        df_copy['PM2.5_lag1'] = df_copy['PM2.5'].shift(1)
        df_copy['PM2.5_lag3'] = df_copy['PM2.5'].shift(3)
        df_copy['PM2.5_lag7'] = df_copy['PM2.5'].shift(7)
        
        df_copy['PM2.5_ma3'] = df_copy['PM2.5'].rolling(window=3, min_periods=1).mean()
        df_copy['PM2.5_ma7'] = df_copy['PM2.5'].rolling(window=7, min_periods=1).mean()
        df_copy['PM2.5_ma30'] = df_copy['PM2.5'].rolling(window=30, min_periods=1).mean()
    
    # 5. Relative humidity estimation
    if 't2m' in df_copy and 'd2m' in df_copy:
        df_copy['relative_humidity'] = 100 * np.exp((17.625 * (df_copy['d2m'] - 273.15)) / 
                                                      (243.04 + (df_copy['d2m'] - 273.15))) / \
                                        np.exp((17.625 * (df_copy['t2m'] - 273.15)) / 
                                               (243.04 + (df_copy['t2m'] - 273.15)))
        df_copy['relative_humidity'] = df_copy['relative_humidity'].clip(0, 100)
    
    # 6. Wind direction category
    if 'wind_dir_10m' in df_copy:
        df_copy['wind_dir_category'] = pd.cut(df_copy['wind_dir_10m'], 
                                                bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                                                labels=[0, 1, 2, 3, 4, 5, 6, 7],
                                                include_lowest=True).astype(int)
    
    return df_copy

# ============================== Part 4: Time Series Dataset Class ==============================
class TimeSeriesDataset(Dataset):
    """Time series dataset class"""
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

# ============================== Part 5: CNN-LSTM-Attention Model ==============================
class CNNLSTMAttention(nn.Module):
    """CNN-LSTM-Attention model"""
    def __init__(self, input_size, hidden_size, num_layers, num_filters, kernel_size, dropout=0.2):
        super(CNNLSTMAttention, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1D CNN layer
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, 
                               kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)
        self.dropout1 = nn.Dropout(dropout)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention layer
        self.attention = nn.Linear(hidden_size, 1)
        
        # Fully connected layer
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

# ============================== Part 6: GPU Accelerated Training and Evaluation Functions ==============================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, device):
    """Train model (GPU accelerated version - supports mixed precision training)"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Mixed precision training
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    # Training speed statistics
    epoch_times = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # ========== Training Phase ==========
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        if TQDM_AVAILABLE:
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        else:
            train_iterator = train_loader
        
        for X_batch, y_batch in train_iterator:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient gradient clearing
            
            # Mixed precision forward pass
            if use_amp:
                with autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                
                # Gradient clipping (prevent gradient explosion)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training (CPU mode)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            batch_count += 1
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # ========== Validation Phase ==========
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            if TQDM_AVAILABLE:
                val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            else:
                val_iterator = val_loader
            
            for X_batch, y_batch in val_iterator:
                X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                
                if use_amp:
                    with autocast():
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                else:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # GPU memory statistics
        if use_amp:
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            samples_per_sec = len(train_loader.dataset) / epoch_time
            if use_amp:
                print(f"  Epoch [{epoch+1}/{num_epochs}], "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Time: {epoch_time:.1f}s, Speed: {samples_per_sec:.0f} samples/s, "
                      f"GPU Mem: {gpu_memory_allocated:.2f}/{gpu_memory_reserved:.2f} GB")
            else:
                print(f"  Epoch [{epoch+1}/{num_epochs}], "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Time: {epoch_time:.1f}s, Speed: {samples_per_sec:.0f} samples/s")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Training statistics
    if epoch_times:
        avg_epoch_time = np.mean(epoch_times)
        total_time = np.sum(epoch_times)
        print(f"\n  Training statistics:")
        print(f"    Total training time: {total_time/60:.1f} minutes")
        print(f"    Average per epoch: {avg_epoch_time:.1f} seconds")
        print(f"    Average speed: {len(train_loader.dataset)/avg_epoch_time:.0f} samples/s")
    
    # GPU memory cleanup
    if use_amp:
        torch.cuda.empty_cache()
    
    return model, train_losses, val_losses

def evaluate_model(model, data_loader, device):
    """Evaluate model (GPU accelerated version)"""
    model.eval()
    predictions = []
    actuals = []
    
    use_amp = device.type == 'cuda'
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            
            if use_amp:
                with autocast():
                    outputs = model(X_batch)
            else:
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
    """Predict and extract attention weights (GPU accelerated version)"""
    model.eval()
    predictions = []
    actuals = []
    attention_weights_list = []
    
    use_amp = device.type == 'cuda'
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            
            if use_amp:
                with autocast():
                    outputs, attn_weights = model(X_batch, return_attention=True)
            else:
                outputs, attn_weights = model(X_batch, return_attention=True)
            
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(y_batch.numpy().flatten())
            attention_weights_list.append(attn_weights.cpu().numpy())
    
    # Average attention weights
    avg_attention = np.concatenate(attention_weights_list, axis=0).mean(axis=0).flatten()
    
    return np.array(predictions), np.array(actuals), avg_attention

# ============================== Part 7: Data Loading and Preprocessing ==============================
print("\n" + "=" * 80)
print("Step 1: Data Loading and Preprocessing")
print("=" * 80)

df_pollution = read_all_pollution()
df_era5 = read_all_era5()

print("\nData loading check:")
print(f"  Pollution data shape: {df_pollution.shape}")
print(f"  Meteorological data shape: {df_era5.shape}")

if df_pollution.empty:
    print("\n⚠️ Warning: Pollution data is empty! Please check data path and files.")
    import sys
    sys.exit(1)

if df_era5.empty:
    print("\n⚠️ Warning: Meteorological data is empty! Please check data path and files.")
    import sys
    sys.exit(1)

df_pollution.index = pd.to_datetime(df_pollution.index)
df_era5.index = pd.to_datetime(df_era5.index)

print(f"  Pollution data time range: {df_pollution.index.min()} to {df_pollution.index.max()}")
print(f"  Meteorological data time range: {df_era5.index.min()} to {df_era5.index.max()}")

print("\nMerging data...")
df_combined = df_pollution.join(df_era5, how='inner')

if df_combined.empty:
    print("\n❌ Error: Data is empty after merging!")
    import sys
    sys.exit(1)

print("\nCreating features...")
df_combined = create_features(df_combined)

print("\nCleaning data...")
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
initial_rows = len(df_combined)
df_combined.dropna(inplace=True)
final_rows = len(df_combined)
print(f"Removed {initial_rows - final_rows} rows containing missing values")

print(f"\nMerged data shape: {df_combined.shape}")
print(f"Time range: {df_combined.index.min().date()} to {df_combined.index.max().date()}")
print(f"Number of samples: {len(df_combined)}")
print(f"Number of features: {df_combined.shape[1]}")

# ============================== Part 8: Feature Selection and Data Preparation ==============================
print("\n" + "=" * 80)
print("Step 2: Feature Selection and Data Preparation")
print("=" * 80)

target = 'PM2.5'
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']

numeric_features = [col for col in df_combined.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_cols]

print(f"\nNumber of selected features: {len(numeric_features)}")
print(f"Target variable: {target}")

X = df_combined[numeric_features].values
y = df_combined[target].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

# Data standardization
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

print(f"\nPM2.5 Statistics:")
print(f"  Mean: {y.mean():.2f} μg/m³")
print(f"  Std Dev: {y.std():.2f} μg/m³")
print(f"  Min: {y.min():.2f} μg/m³")
print(f"  Max: {y.max():.2f} μg/m³")
print(f"  Median: {np.median(y):.2f} μg/m³")

# Save scaler
with open(model_dir / 'scaler_X_gpu.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open(model_dir / 'scaler_y_gpu.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
print("\nSaved StandardScaler objects")

# ============================== Part 9: Multi-Window Model Training (GPU Accelerated) ==============================
print("\n" + "=" * 80)
print("Step 3: Multi-Window CNN-LSTM Model Training (GPU Accelerated Version)")
print("=" * 80)

# Store all results
all_results = []
all_models = {}
all_predictions = {}
all_attention_weights = {}

for seq_length in sequence_lengths:
    print(f"\n{'='*80}")
    print(f"Training window length: {seq_length} days")
    print(f"{'='*80}")
    
    # Create time series dataset
    n_samples = len(X_scaled)
    train_size = int(n_samples * 0.70)
    val_size = int(n_samples * 0.15)
    
    X_train = X_scaled[:train_size]
    X_val = X_scaled[train_size:train_size + val_size]
    X_test = X_scaled[train_size + val_size:]
    
    y_train = y_scaled[:train_size]
    y_val = y_scaled[train_size:train_size + val_size]
    y_test = y_scaled[train_size + val_size:]
    
    print(f"\nDataset split (original samples: {n_samples}):")
    print(f"  Training set: {len(X_train)} ({len(X_train)/n_samples*100:.1f}%)")
    print(f"  Validation set: {len(X_val)} ({len(X_val)/n_samples*100:.1f}%)")
    print(f"  Test set: {len(X_test)} ({len(X_test)/n_samples*100:.1f}%)")
    
    # Create Dataset and DataLoader (GPU optimized)
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_length)
    val_dataset = TimeSeriesDataset(X_val, y_val, seq_length)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_length)
    
    print(f"\nTime series dataset size (window length={seq_length}):")
    print(f"  Training set: {len(train_dataset)} sequences")
    print(f"  Validation set: {len(val_dataset)} sequences")
    print(f"  Test set: {len(test_dataset)} sequences")
    
    # GPU optimized DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, 
                              persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS,
                            persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS,
                             persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False)
    
    # ========== Basic Model ==========
    print(f"\n{'─'*80}")
    print("Training basic model")
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
    
    print("\nBasic model parameters:")
    for key, value in basic_params.items():
        print(f"  {key}: {value}")
    
    model_basic = CNNLSTMAttention(**basic_params).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_basic.parameters(), lr=0.001)
    
    print("\nStarting training...")
    model_basic, train_losses_basic, val_losses_basic = train_model(
        model_basic, train_loader, val_loader, criterion, optimizer,
        num_epochs=100, patience=15, device=device
    )
    
    print("\nEvaluating basic model...")
    train_results_basic = evaluate_model(model_basic, train_loader, device)
    val_results_basic = evaluate_model(model_basic, val_loader, device)
    test_results_basic = evaluate_model(model_basic, test_loader, device)
    
    # Denormalize
    train_pred_basic = scaler_y.inverse_transform(train_results_basic['predictions'].reshape(-1, 1)).flatten()
    val_pred_basic = scaler_y.inverse_transform(val_results_basic['predictions'].reshape(-1, 1)).flatten()
    test_pred_basic = scaler_y.inverse_transform(test_results_basic['predictions'].reshape(-1, 1)).flatten()
    
    train_actual = scaler_y.inverse_transform(train_results_basic['actuals'].reshape(-1, 1)).flatten()
    val_actual = scaler_y.inverse_transform(val_results_basic['actuals'].reshape(-1, 1)).flatten()
    test_actual = scaler_y.inverse_transform(test_results_basic['actuals'].reshape(-1, 1)).flatten()
    
    # Recalculate metrics
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
    
    print(f"\nBasic model performance (window={seq_length} days):")
    print(f"  Training set - R²: {train_r2_basic:.4f}, RMSE: {train_rmse_basic:.2f}, MAE: {train_mae_basic:.2f}, MAPE: {train_mape_basic:.2f}%")
    print(f"  Validation set - R²: {val_r2_basic:.4f}, RMSE: {val_rmse_basic:.2f}, MAE: {val_mae_basic:.2f}, MAPE: {val_mape_basic:.2f}%")
    print(f"  Test set - R²: {test_r2_basic:.4f}, RMSE: {test_rmse_basic:.2f}, MAE: {test_mae_basic:.2f}, MAPE: {test_mape_basic:.2f}%")
    
    # Save results
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
    
    # GPU memory cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # ========== Hyperparameter Optimization (GPU Accelerated) ==========
    print(f"\n{'─'*80}")
    print("Hyperparameter optimization (GPU accelerated version)")
    print(f"{'─'*80}")
    
    if BAYESIAN_OPT_AVAILABLE:
        print("\nUsing Bayesian optimization...")
        
        def cnn_lstm_evaluate(hidden_size, num_layers, num_filters, kernel_size, learning_rate, dropout):
            """Bayesian optimization objective function"""
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
                num_epochs=30, patience=10, device=device  # Reduce epochs, GPU trains faster
            )
            
            val_results_temp = evaluate_model(model_temp, val_loader, device)
            rmse = val_results_temp['RMSE']
            
            # Clean GPU memory
            del model_temp, criterion_temp, optimizer_temp
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return -rmse  # Negative RMSE (for maximization)
        
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
        
        print(f"\nBest parameters (Bayesian optimization):")
        for key, value in best_params_opt.items():
            print(f"  {key}: {value}")
        print(f"  Best validation RMSE: {-optimizer_bayes.max['target']:.4f}")
        
        best_lr = best_params_opt.pop('learning_rate')
        
    else:
        print("\nUsing grid search...")
        
        param_grid = {
            'hidden_size': [48, 64, 96],
            'num_layers': [2, 3],
            'num_filters': [32, 48],
            'kernel_size': [3, 5],
            'learning_rate': [0.001, 0.0005],
            'dropout': [0.2, 0.3]
        }
        
        total_combinations = int(np.prod([len(v) for v in param_grid.values()]))
        print(f"Total {total_combinations} parameter combinations")
        
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
                                    num_epochs=30, patience=10, device=device  # Reduce epochs
                                )
                                
                                val_results_temp = evaluate_model(model_temp, val_loader, device)
                                val_rmse = val_results_temp['RMSE']
                                
                                if val_rmse < best_val_rmse:
                                    best_val_rmse = val_rmse
                                    best_params_opt = params_test.copy()
                                    best_params_opt.pop('input_size')
                                    best_lr = lr
                                
                                # Clean GPU memory
                                del model_temp, criterion_temp, optimizer_temp
                                if device.type == 'cuda':
                                    torch.cuda.empty_cache()
                                
                                if tested % 5 == 0:
                                    print(f"  Tested {tested}/{total_combinations} combinations, current best RMSE: {best_val_rmse:.4f}")
        
        print(f"\nBest parameters (grid search):")
        for key, value in best_params_opt.items():
            print(f"  {key}: {value}")
        print(f"  learning_rate: {best_lr}")
        print(f"  Best validation RMSE: {best_val_rmse:.4f}")
    
    # ========== Train Optimized Model ==========
    print(f"\n{'─'*80}")
    print("Training optimized model with best parameters")
    print(f"{'─'*80}")
    
    optimized_params = {
        'input_size': input_size,
        **best_params_opt
    }
    
    print("\nOptimized model parameters:")
    for key, value in optimized_params.items():
        print(f"  {key}: {value}")
    print(f"  learning_rate: {best_lr}")
    
    model_optimized = CNNLSTMAttention(**optimized_params).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_optimized.parameters(), lr=best_lr)
    
    print("\nStarting training...")
    model_optimized, train_losses_opt, val_losses_opt = train_model(
        model_optimized, train_loader, val_loader, criterion, optimizer,
        num_epochs=150, patience=20, device=device
    )
    
    print("\nEvaluating optimized model...")
    train_results_opt = evaluate_model(model_optimized, train_loader, device)
    val_results_opt = evaluate_model(model_optimized, val_loader, device)
    test_results_opt = evaluate_model(model_optimized, test_loader, device)
    
    # Denormalize
    train_pred_opt = scaler_y.inverse_transform(train_results_opt['predictions'].reshape(-1, 1)).flatten()
    val_pred_opt = scaler_y.inverse_transform(val_results_opt['predictions'].reshape(-1, 1)).flatten()
    test_pred_opt = scaler_y.inverse_transform(test_results_opt['predictions'].reshape(-1, 1)).flatten()
    
    # Recalculate metrics
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
    
    print(f"\nOptimized model performance (window={seq_length} days):")
    print(f"  Training set - R²: {train_r2_opt:.4f}, RMSE: {train_rmse_opt:.2f}, MAE: {train_mae_opt:.2f}, MAPE: {train_mape_opt:.2f}%")
    print(f"  Validation set - R²: {val_r2_opt:.4f}, RMSE: {val_rmse_opt:.2f}, MAE: {val_mae_opt:.2f}, MAPE: {val_mape_opt:.2f}%")
    print(f"  Test set - R²: {test_r2_opt:.4f}, RMSE: {test_rmse_opt:.2f}, MAE: {test_mae_opt:.2f}, MAPE: {test_mape_opt:.2f}%")
    
    # Save results
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
    
    # Extract Attention weights
    print("\nExtracting Attention feature importance...")
    _, _, avg_attention = predict_with_attention(model_optimized, test_loader, device)
    all_attention_weights[f'w{seq_length}'] = avg_attention
    
    # Save model
    torch.save(model_optimized.state_dict(), model_dir / f'cnn_lstm_window{seq_length}_optimized_gpu.pth')
    with open(model_dir / f'cnn_lstm_window{seq_length}_optimized_gpu.pkl', 'wb') as f:
        pickle.dump(model_optimized, f)
    print(f"Saved model: cnn_lstm_window{seq_length}_optimized_gpu.pth/pkl")
    
    # Final GPU memory cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# ============================== Part 10: Results Summary ==============================
print("\n" + "=" * 80)
print("Step 4: Results Summary and Comparison")
print("=" * 80)

results_df = pd.DataFrame(all_results)
print("\nAll models performance comparison:")
print(results_df.to_string(index=False))

# Test set performance ranking
test_results = results_df[results_df['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\nTest set performance ranking:")
print(test_results.to_string(index=False))

# Best model
best_model_row = test_results.iloc[0]
print(f"\nBest model: {best_model_row['Model']}")
print(f"  Window size: {best_model_row['Window']} days")
print(f"  R² Score: {best_model_row['R²']:.4f}")
print(f"  RMSE: {best_model_row['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model_row['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model_row['MAPE']:.2f}%")

# ============================== Part 11: Visualization ==============================
print("\n" + "=" * 80)
print("Step 5: Generating Visualization Charts")
print("=" * 80)

# 11.1 Training process curves (for each window)
for seq_length in sequence_lengths:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Basic model
    train_losses = all_models[f'basic_w{seq_length}']['train_losses']
    val_losses = all_models[f'basic_w{seq_length}']['val_losses']
    
    axes[0].plot(train_losses, label='Training Loss', linewidth=2)
    axes[0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title(f'CNN-LSTM Basic Model Training Process (window={seq_length} days) [GPU]', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Optimized model
    train_losses = all_models[f'optimized_w{seq_length}']['train_losses']
    val_losses = all_models[f'optimized_w{seq_length}']['val_losses']
    
    axes[1].plot(train_losses, label='Training Loss', linewidth=2)
    axes[1].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss (MSE)', fontsize=12)
    axes[1].set_title(f'CNN-LSTM Optimized Model Training Process (window={seq_length} days) [GPU]', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_window{seq_length}_gpu.png', dpi=300, bbox_inches='tight')
    print(f"Saved: training_curves_window{seq_length}_gpu.png")
    plt.close()

# 11.2 Prediction vs actual scatter plots (all models)
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
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')
        
        r2 = r2_score(test_actual, test_pred)
        rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
        
        ax.set_xlabel('Actual PM2.5 Concentration (μg/m³)', fontsize=11)
        ax.set_ylabel('Predicted PM2.5 Concentration (μg/m³)', fontsize=11)
        ax.set_title(f'CNN-LSTM-{"Basic" if model_type=="basic" else "Optimized"} (window={seq_length} days) [GPU]\nR²={r2:.4f}, RMSE={rmse:.2f}', 
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'prediction_scatter_gpu.png', dpi=300, bbox_inches='tight')
print("Saved: prediction_scatter_gpu.png")
plt.close()

# 11.3 Time series prediction comparison (best window)
best_window = best_model_row['Window']
fig, axes = plt.subplots(2, 1, figsize=(18, 10))

test_actual_best, test_pred_basic_best = all_predictions[f'basic_w{best_window}']['test']
_, test_pred_opt_best = all_predictions[f'optimized_w{best_window}']['test']

plot_range = min(300, len(test_actual_best))
plot_idx_range = range(len(test_actual_best) - plot_range, len(test_actual_best))

axes[0].plot(plot_idx_range, test_actual_best[plot_idx_range], 'k-', label='Actual', 
             linewidth=2, alpha=0.8)
axes[0].plot(plot_idx_range, test_pred_basic_best[plot_idx_range], 'b--', label='Basic Model Prediction', 
             linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('Sample Index', fontsize=12)
axes[0].set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
axes[0].set_title(f'CNN-LSTM Basic Model - Time Series Prediction Comparison (window={best_window} days, last {plot_range} samples) [GPU]', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(plot_idx_range, test_actual_best[plot_idx_range], 'k-', label='Actual', 
             linewidth=2, alpha=0.8)
axes[1].plot(plot_idx_range, test_pred_opt_best[plot_idx_range], 'g--', label='Optimized Model Prediction', 
             linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('Sample Index', fontsize=12)
axes[1].set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
axes[1].set_title(f'CNN-LSTM Optimized Model - Time Series Prediction Comparison (window={best_window} days, last {plot_range} samples) [GPU]', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'timeseries_comparison_gpu.png', dpi=300, bbox_inches='tight')
print("Saved: timeseries_comparison_gpu.png")
plt.close()

# 11.4 Residual analysis
fig, axes = plt.subplots(rows, 2, figsize=(16, 5*rows))

for i, seq_length in enumerate(sequence_lengths):
    for j, model_type in enumerate(['basic', 'optimized']):
        ax = axes[i, j] if rows > 1 else axes[j]
        
        test_actual, test_pred = all_predictions[f'{model_type}_w{seq_length}']['test']
        residuals = test_actual - test_pred
        
        ax.scatter(test_pred, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Value (μg/m³)', fontsize=11)
        ax.set_ylabel('Residual (μg/m³)', fontsize=11)
        ax.set_title(f'CNN-LSTM-{"Basic" if model_type=="basic" else "Optimized"} (window={seq_length} days) [GPU]\nMean Residual={residuals.mean():.2f}, Std={residuals.std():.2f}', 
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'residuals_analysis_gpu.png', dpi=300, bbox_inches='tight')
print("Saved: residuals_analysis_gpu.png")
plt.close()

# 11.5 Attention feature importance (average sequence position weights)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, seq_length in enumerate(sequence_lengths):
    avg_attention = all_attention_weights[f'w{seq_length}']
    
    axes[i].bar(range(len(avg_attention)), avg_attention, color='steelblue')
    axes[i].set_xlabel('Sequence Position (days)', fontsize=12)
    axes[i].set_ylabel('Attention Weight', fontsize=12)
    axes[i].set_title(f'Attention Weight Distribution (window={seq_length} days) [GPU]', fontsize=13, fontweight='bold')
    axes[i].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'attention_weights_gpu.png', dpi=300, bbox_inches='tight')
print("Saved: attention_weights_gpu.png")
plt.close()

# 11.6 Model performance comparison (all windows and models)
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
        axes[i].set_title(f'{metric} Comparison [GPU]\n(Higher is Better)', fontsize=12, fontweight='bold')
    else:
        axes[i].set_title(f'{metric} Comparison [GPU]\n(Lower is Better)', fontsize=12, fontweight='bold')
    
    axes[i].grid(True, alpha=0.3, axis='y')
    
    for j, v in enumerate(test_results_sorted[metric]):
        if metric == 'MAPE':
            axes[i].text(j, v, f'{v:.1f}%', ha='center', va='bottom', 
                         fontsize=8, fontweight='bold')
        else:
            axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom', 
                         fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison_gpu.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison_gpu.png")
plt.close()

# 11.7 Error distribution (best window)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

test_actual_best, test_pred_basic_best = all_predictions[f'basic_w{best_window}']['test']
_, test_pred_opt_best = all_predictions[f'optimized_w{best_window}']['test']

errors_basic = test_actual_best - test_pred_basic_best
errors_opt = test_actual_best - test_pred_opt_best

axes[0].hist(errors_basic, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
axes[0].set_xlabel('Prediction Error (μg/m³)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title(f'Basic Model - Prediction Error Distribution (window={best_window} days) [GPU]\nMean={errors_basic.mean():.2f}, Std={errors_basic.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].hist(errors_opt, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero Error')
axes[1].set_xlabel('Prediction Error (μg/m³)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title(f'Optimized Model - Prediction Error Distribution (window={best_window} days) [GPU]\nMean={errors_opt.mean():.2f}, Std={errors_opt.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'error_distribution_gpu.png', dpi=300, bbox_inches='tight')
print("Saved: error_distribution_gpu.png")
plt.close()

# 11.8 Window size comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Extract test set performance for each window
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

# R² comparison
x = np.arange(len(sequence_lengths))
width = 0.35

axes[0].bar(x - width/2, window_df['Basic_R2'], width, label='Basic Model', color='blue', alpha=0.7)
axes[0].bar(x + width/2, window_df['Opt_R2'], width, label='Optimized Model', color='green', alpha=0.7)
axes[0].set_xlabel('Window Size (days)', fontsize=12)
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('R² Comparison for Different Window Sizes [GPU]', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'{w} days' for w in sequence_lengths])
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# RMSE comparison
axes[1].bar(x - width/2, window_df['Basic_RMSE'], width, label='Basic Model', color='blue', alpha=0.7)
axes[1].bar(x + width/2, window_df['Opt_RMSE'], width, label='Optimized Model', color='green', alpha=0.7)
axes[1].set_xlabel('Window Size (days)', fontsize=12)
axes[1].set_ylabel('RMSE (μg/m³)', fontsize=12)
axes[1].set_title('RMSE Comparison for Different Window Sizes [GPU]', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'{w} days' for w in sequence_lengths])
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'window_size_comparison_gpu.png', dpi=300, bbox_inches='tight')
print("Saved: window_size_comparison_gpu.png")
plt.close()

# ============================== Part 12: Save Results ==============================
print("\n" + "=" * 80)
print("Step 6: Saving Results")
print("=" * 80)

# Save model performance
results_df.to_csv(output_dir / 'model_performance_gpu.csv', index=False, encoding='utf-8-sig')
print("Saved: model_performance_gpu.csv")

# Save window comparison
window_df.to_csv(output_dir / 'window_comparison_gpu.csv', index=False, encoding='utf-8-sig')
print("Saved: window_comparison_gpu.csv")

# Save Attention weights (save separately for each window as lengths differ)
for seq_length in sequence_lengths:
    attn_weights = all_attention_weights[f'w{seq_length}']
    actual_length = len(attn_weights)
    attention_df = pd.DataFrame({
        'Sequence_Position': range(actual_length),
        'Attention_Weight': attn_weights
    })
    attention_df.to_csv(output_dir / f'attention_weights_window{seq_length}_gpu.csv', 
                       index=False, encoding='utf-8-sig')
    print(f"Saved: attention_weights_window{seq_length}_gpu.csv (actual length: {actual_length})")

# Save predictions for each window
for seq_length in sequence_lengths:
    for model_type in ['basic', 'optimized']:
        test_actual, test_pred = all_predictions[f'{model_type}_w{seq_length}']['test']
        pred_df = pd.DataFrame({
            'Actual': test_actual,
            'Prediction': test_pred,
            'Error': test_actual - test_pred
        })
        pred_df.to_csv(output_dir / f'predictions_{model_type}_window{seq_length}_gpu.csv', 
                      index=False, encoding='utf-8-sig')
        print(f"Saved: predictions_{model_type}_window{seq_length}_gpu.csv")

# Save best parameters
best_params_list = []
for seq_length in sequence_lengths:
    params = all_models[f'optimized_w{seq_length}']['params'].copy()
    params['window'] = seq_length
    best_params_list.append(params)

best_params_df = pd.DataFrame(best_params_list)
best_params_df.to_csv(output_dir / 'best_parameters_gpu.csv', index=False, encoding='utf-8-sig')
print("Saved: best_parameters_gpu.csv")

# ============================== Part 13: Summary Report ==============================
print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)

print("\nGenerated files:")
print("\nCSV files:")
print("  - model_performance_gpu.csv           Model performance comparison")
print("  - window_comparison_gpu.csv           Window size comparison")
print("  - attention_weights_window*_gpu.csv   Attention weights for each window")
print("  - predictions_*_gpu.csv               Prediction results for each model")
print("  - best_parameters_gpu.csv             Best parameters")

print("\nChart files:")
print("  - training_curves_window*_gpu.png     Training process curves (for each window)")
print("  - prediction_scatter_gpu.png          Prediction vs actual scatter plots")
print("  - timeseries_comparison_gpu.png       Time series comparison")
print("  - residuals_analysis_gpu.png          Residual analysis")
print("  - attention_weights_gpu.png           Attention weight distribution")
print("  - model_comparison_gpu.png            Model performance comparison")
print("  - error_distribution_gpu.png          Error distribution")
print("  - window_size_comparison_gpu.png      Window size comparison")

print("\nModel files:")
for seq_length in sequence_lengths:
    print(f"  - cnn_lstm_window{seq_length}_optimized_gpu.pth   PyTorch model weights")
    print(f"  - cnn_lstm_window{seq_length}_optimized_gpu.pkl   Complete model object")
print("  - scaler_X_gpu.pkl                    Feature scaler")
print("  - scaler_y_gpu.pkl                    Target scaler")

print(f"\nBest model: {best_model_row['Model']}")
print(f"  Window size: {best_model_row['Window']} days")
print(f"  R² Score: {best_model_row['R²']:.4f}")
print(f"  RMSE: {best_model_row['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model_row['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model_row['MAPE']:.2f}%")

print("\nTest set performance summary for each window:")
for seq_length in sequence_lengths:
    opt_row = results_df[(results_df['Model'] == f'CNN-LSTM_Optimized_W{seq_length}') & 
                         (results_df['Dataset'] == 'Test')].iloc[0]
    print(f"  Window={seq_length} days: R²={opt_row['R²']:.4f}, RMSE={opt_row['RMSE']:.2f}")

print("\n" + "=" * 80)
print("CNN-LSTM PM2.5 Concentration Prediction Complete! (GPU Accelerated Version)")
print("=" * 80)

if device.type == 'cuda':
    print(f"\n✓ GPU accelerated training enabled")
    print(f"✓ Mixed precision training (AMP)")
    print(f"✓ Batch size: {BATCH_SIZE}")
    print(f"✓ Optimized data loading")
else:
    print(f"\n⚠️  Running in CPU mode")
