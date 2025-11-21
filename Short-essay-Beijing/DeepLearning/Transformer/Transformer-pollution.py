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
from itertools import product
import time
import importlib

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

try:
    from bayes_opt import BayesianOptimization
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False

try:
    shap = importlib.import_module("shap")
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

if not torch.cuda.is_available():
    print("Error: CUDA/GPU device not detected!")
    print("Please install CUDA version of PyTorch or use CPU version.")
    import sys
    sys.exit(1)

device = torch.device('cuda')

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
compute_capability = torch.cuda.get_device_capability(0)
USE_AMP = compute_capability[0] >= 7

pollution_all_path = '/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)'
pollution_extra_path = '/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)'

output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)
model_dir = Path('./models')
model_dir.mkdir(exist_ok=True)

start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

SEQ_LENGTH = 30
PRED_LENGTH = 7
BATCH_SIZE = 2048

if gpu_memory < 6:
    BATCH_SIZE = 32
elif gpu_memory < 8:
    BATCH_SIZE = 64
elif gpu_memory < 16:
    BATCH_SIZE = 128
elif gpu_memory < 24:
    BATCH_SIZE = 256

def daterange(start, end):
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def build_file_path_dict(base_path, prefix):
    file_dict = {}
    filename_pattern = f"{prefix}_"
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename.startswith(filename_pattern) and filename.endswith('.csv'):
                date_str = filename[len(filename_pattern):-4]
                if len(date_str) == 8 and date_str.isdigit():
                    file_dict[date_str] = os.path.join(root, filename)
    return file_dict

def read_pollution_day(args):
    date, file_path_dict_all, file_path_dict_extra, pollutants_list = args
    date_str = date.strftime('%Y%m%d')
    all_file = file_path_dict_all.get(date_str)
    extra_file = file_path_dict_extra.get(date_str)
    
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
        df_daily = df_daily[[col for col in pollutants_list if col in df_daily.columns]]
        
        return df_daily
    except Exception:
        return None

def read_all_pollution():
    print("Loading pollution data...")
    file_path_dict_all = build_file_path_dict(pollution_all_path, 'beijing_all')
    file_path_dict_extra = build_file_path_dict(pollution_extra_path, 'beijing_extra')
    
    dates = list(daterange(start_date, end_date))
    pollution_dfs = []
    args_list = [(date, file_path_dict_all, file_path_dict_extra, pollutants) 
                  for date in dates]
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, args): args[0] for args in args_list}
        
        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading data"):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
        else:
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
                if i % 500 == 0:
                    print(f"Processed {i}/{len(futures)} days")
    
    if pollution_dfs:
        df_poll_all = pd.concat(pollution_dfs)
        df_poll_all = df_poll_all.ffill()
        df_poll_all = df_poll_all.fillna(df_poll_all.mean())
        print(f"Data loaded: {df_poll_all.shape}")
        return df_poll_all
    return pd.DataFrame()

def create_features(df):
    df_copy = df.copy()
    
    df_copy['year'] = df_copy.index.year
    df_copy['month'] = df_copy.index.month
    df_copy['day'] = df_copy.index.day
    df_copy['day_of_year'] = df_copy.index.dayofyear
    df_copy['day_of_week'] = df_copy.index.dayofweek
    try:
        df_copy['week_of_year'] = df_copy.index.isocalendar().week
    except AttributeError:
        df_copy['week_of_year'] = (df_copy.index.dayofyear - 1) // 7 + 1
    
    df_copy['season'] = df_copy['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )
    df_copy['is_heating_season'] = ((df_copy['month'] >= 11) | (df_copy['month'] <= 3)).astype(int)
    
    if 'PM2.5' in df_copy:
        df_copy['PM2.5_lag1'] = df_copy['PM2.5'].shift(1)
        df_copy['PM2.5_lag3'] = df_copy['PM2.5'].shift(3)
        df_copy['PM2.5_lag7'] = df_copy['PM2.5'].shift(7)
        df_copy['PM2.5_ma3'] = df_copy['PM2.5'].rolling(window=3, min_periods=1).mean()
        df_copy['PM2.5_ma7'] = df_copy['PM2.5'].rolling(window=7, min_periods=1).mean()
        df_copy['PM2.5_ma30'] = df_copy['PM2.5'].rolling(window=30, min_periods=1).mean()
    
    for pollutant in ['PM10', 'SO2', 'NO2', 'CO', 'O3']:
        if pollutant in df_copy:
            df_copy[f'{pollutant}_lag1'] = df_copy[pollutant].shift(1)
            df_copy[f'{pollutant}_lag7'] = df_copy[pollutant].shift(7)
            df_copy[f'{pollutant}_ma7'] = df_copy[pollutant].rolling(window=7, min_periods=1).mean()
    
    if 'PM2.5' in df_copy and 'PM10' in df_copy:
        df_copy['PM2.5_PM10_ratio'] = df_copy['PM2.5'] / (df_copy['PM10'] + 1e-8)
    
    if 'NO2' in df_copy and 'O3' in df_copy:
        df_copy['NO2_O3_ratio'] = df_copy['NO2'] / (df_copy['O3'] + 1e-8)
    
    return df_copy

df_pollution = read_all_pollution()

if df_pollution.empty:
    print("Error: Pollution data is empty!")
    import sys
    sys.exit(1)

df_pollution.index = pd.to_datetime(df_pollution.index)
df_combined = df_pollution.copy()
df_combined = create_features(df_combined)
df_combined = df_combined.replace([np.inf, -np.inf], np.nan)
df_combined = df_combined.dropna()

print(f"Final data shape: {df_combined.shape}")

target = 'PM2.5'
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']
numeric_features = [col for col in df_combined.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_cols]

X = df_combined[numeric_features].values
y = df_combined[target].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

class TimeSeriesDataset(Dataset):
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

n_samples = len(X_scaled)
train_size = int(n_samples * 0.70)
val_size = int(n_samples * 0.15)

X_train = X_scaled[:train_size]
X_val = X_scaled[train_size:train_size + val_size]
X_test = X_scaled[train_size + val_size:]
y_train = y_scaled[:train_size]
y_val = y_scaled[train_size:train_size + val_size]
y_test = y_scaled[train_size + val_size:]

train_dataset = TimeSeriesDataset(X_train, y_train, SEQ_LENGTH, PRED_LENGTH)
val_dataset = TimeSeriesDataset(X_val, y_val, SEQ_LENGTH, PRED_LENGTH)
test_dataset = TimeSeriesDataset(X_test, y_test, SEQ_LENGTH, PRED_LENGTH)

optimal_workers = min(32, max(16, CPU_COUNT // 2))
optimal_prefetch = 8

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=optimal_workers, pin_memory=True,
                         persistent_workers=True, prefetch_factor=optimal_prefetch, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=optimal_workers, pin_memory=True,
                       persistent_workers=True, prefetch_factor=optimal_prefetch, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=optimal_workers, pin_memory=True,
                        persistent_workers=True, prefetch_factor=optimal_prefetch, drop_last=False)

class PositionalEncoding(nn.Module):
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
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.1, pred_length=7):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_length = pred_length
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu',
            norm_first=False, bias=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, pred_length)
        )
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        
    def forward(self, src):
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        output = memory[:, -1, :]
        output = self.decoder(output)
        return output

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=50, verbose=True, 
                gradient_accumulation_steps=1):
    model_compiled = False
    try:
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
            model_compiled = True
    except Exception:
        pass
    
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda', enabled=USE_AMP)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            with autocast('cuda', enabled=USE_AMP):
                output = model(batch_X)
                loss = criterion(output, batch_y) / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item() * gradient_accumulation_steps
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                with autocast('cuda', enabled=USE_AMP):
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at Epoch {epoch+1}")
            break
        
        if (epoch + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, epoch + 1

def predict_model(model, data_loader):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            with autocast('cuda', enabled=USE_AMP):
                output = model(batch_X)
            predictions.append(output.cpu().numpy())
            actuals.append(batch_y.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    return predictions, actuals

def evaluate_predictions(y_true, y_pred, dataset_name):
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

d_model_basic = 128
nhead_basic = 8
num_layers_basic = 3
dim_feedforward_basic = 512
dropout_basic = 0.1
lr_basic = 0.001

model_basic = TimeSeriesTransformer(
    input_dim=len(numeric_features),
    d_model=d_model_basic,
    nhead=nhead_basic,
    num_layers=num_layers_basic,
    dim_feedforward=dim_feedforward_basic,
    dropout=dropout_basic,
    pred_length=PRED_LENGTH
).to(device)

gradient_accum_basic = max(1, 2048 // BATCH_SIZE)
model_basic, train_losses_basic, val_losses_basic, epochs_trained_basic = train_model(
    model_basic, train_loader, val_loader, 
    epochs=200, lr=lr_basic, patience=50, verbose=True,
    gradient_accumulation_steps=gradient_accum_basic
)
torch.cuda.empty_cache()

train_pred_basic, train_actual_basic = predict_model(model_basic, train_loader)
val_pred_basic, val_actual_basic = predict_model(model_basic, val_loader)
test_pred_basic, test_actual_basic = predict_model(model_basic, test_loader)

train_pred_basic_orig = scaler_y.inverse_transform(train_pred_basic)
train_actual_basic_orig = scaler_y.inverse_transform(train_actual_basic)
val_pred_basic_orig = scaler_y.inverse_transform(val_pred_basic)
val_actual_basic_orig = scaler_y.inverse_transform(val_actual_basic)
test_pred_basic_orig = scaler_y.inverse_transform(test_pred_basic)
test_actual_basic_orig = scaler_y.inverse_transform(test_actual_basic)

results_basic = [
    evaluate_predictions(train_actual_basic_orig, train_pred_basic_orig, 'Train'),
    evaluate_predictions(val_actual_basic_orig, val_pred_basic_orig, 'Validation'),
    evaluate_predictions(test_actual_basic_orig, test_pred_basic_orig, 'Test')
]
results_basic_df = pd.DataFrame(results_basic)
print("\nBasic model performance:")
print(results_basic_df.to_string(index=False))

if BAYESIAN_OPT_AVAILABLE:
    def transformer_evaluate(d_model, nhead, num_layers, dim_feedforward, dropout, learning_rate):
        d_model = int(d_model)
        nhead = int(nhead)
        if d_model % nhead != 0:
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
            del model_temp
            torch.cuda.empty_cache()
            return -best_val_loss
        except Exception:
            torch.cuda.empty_cache()
            return -999999
    
    pbounds = {
        'd_model': (64, 256),
        'nhead': (4, 8),
        'num_layers': (2, 4),
        'dim_feedforward': (256, 1024),
        'dropout': (0.05, 0.3),
        'learning_rate': (0.0001, 0.01)
    }
    
    optimizer = BayesianOptimization(f=transformer_evaluate, pbounds=pbounds, random_state=42, verbose=2)
    optimizer.maximize(init_points=5, n_iter=15)
    
    best_params = optimizer.max['params']
    best_params['d_model'] = int(best_params['d_model'])
    best_params['nhead'] = int(best_params['nhead'])
    best_params['num_layers'] = int(best_params['num_layers'])
    best_params['dim_feedforward'] = int(best_params['dim_feedforward'])
    
    if best_params['d_model'] % best_params['nhead'] != 0:
        best_params['d_model'] = (best_params['d_model'] // best_params['nhead']) * best_params['nhead']
else:
    param_grid = {
        'd_model': [64, 128, 192],
        'nhead': [4, 8],
        'num_layers': [2, 3],
        'dim_feedforward': [256, 512],
        'dropout': [0.1, 0.2],
        'learning_rate': [0.001, 0.0005]
    }
    
    all_combos = list(product(*param_grid.values()))
    best_val_loss = float('inf')
    best_params = {}
    
    pbar = tqdm(all_combos, desc="Grid search") if TQDM_AVAILABLE else all_combos
    
    for i, combo in enumerate(pbar):
        d_model, nhead, num_layers, dim_feedforward, dropout, lr = combo
        
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
            torch.cuda.empty_cache()
        except Exception:
            torch.cuda.empty_cache()
            continue
    
    print(f"Best validation loss: {best_val_loss:.6f}")

model_optimized = TimeSeriesTransformer(
    input_dim=len(numeric_features),
    d_model=best_params['d_model'],
    nhead=best_params['nhead'],
    num_layers=best_params['num_layers'],
    dim_feedforward=best_params['dim_feedforward'],
    dropout=best_params['dropout'],
    pred_length=PRED_LENGTH
).to(device)

gradient_accum_steps = max(1, 4096 // BATCH_SIZE)
model_optimized, train_losses_opt, val_losses_opt, epochs_trained_opt = train_model(
    model_optimized, train_loader, val_loader,
    epochs=300, lr=best_params['learning_rate'], patience=100, verbose=True,
    gradient_accumulation_steps=gradient_accum_steps
)
torch.cuda.empty_cache()

train_pred_opt, train_actual_opt = predict_model(model_optimized, train_loader)
val_pred_opt, val_actual_opt = predict_model(model_optimized, val_loader)
test_pred_opt, test_actual_opt = predict_model(model_optimized, test_loader)

train_pred_opt_orig = scaler_y.inverse_transform(train_pred_opt)
train_actual_opt_orig = scaler_y.inverse_transform(train_actual_opt)
val_pred_opt_orig = scaler_y.inverse_transform(val_pred_opt)
val_actual_opt_orig = scaler_y.inverse_transform(val_actual_opt)
test_pred_opt_orig = scaler_y.inverse_transform(test_pred_opt)
test_actual_opt_orig = scaler_y.inverse_transform(test_actual_opt)

results_opt = [
    evaluate_predictions(train_actual_opt_orig, train_pred_opt_orig, 'Train'),
    evaluate_predictions(val_actual_opt_orig, val_pred_opt_orig, 'Validation'),
    evaluate_predictions(test_actual_opt_orig, test_pred_opt_orig, 'Test')
]
results_opt_df = pd.DataFrame(results_opt)
print("\nOptimized model performance:")
print(results_opt_df.to_string(index=False))

results_basic_df['Model'] = 'Transformer_Basic_GPU'
results_opt_df['Model'] = 'Transformer_Optimized_GPU'
all_results = pd.concat([results_basic_df, results_opt_df])
all_results = all_results[['Model', 'Dataset', 'R²', 'RMSE', 'MAE', 'MAPE']]

print("\nModel performance comparison:")
print(all_results.to_string(index=False))

test_results = all_results[all_results['Dataset'] == 'Test'].sort_values('R²', ascending=False)
basic_test_r2 = results_basic_df[results_basic_df['Dataset'] == 'Test']['R²'].values[0]
opt_test_r2 = results_opt_df[results_opt_df['Dataset'] == 'Test']['R²'].values[0]
basic_test_rmse = results_basic_df[results_basic_df['Dataset'] == 'Test']['RMSE'].values[0]
opt_test_rmse = results_opt_df[results_opt_df['Dataset'] == 'Test']['RMSE'].values[0]

r2_improvement = (opt_test_r2 - basic_test_r2) / (abs(basic_test_r2) + 1e-8) * 100
rmse_improvement = (basic_test_rmse - opt_test_rmse) / basic_test_rmse * 100
print(f"R² improvement: {r2_improvement:.2f}%, RMSE reduction: {rmse_improvement:.2f}%")

if SHAP_AVAILABLE:
    sample_size = min(500, len(test_dataset))
    sample_indices = np.random.choice(len(test_dataset), sample_size, replace=False)
    
    X_sample = []
    for idx in sample_indices:
        X_seq, _ = test_dataset[idx]
        X_sample.append(X_seq.numpy())
    X_sample = np.array(X_sample)
    X_sample_tensor = torch.FloatTensor(X_sample).to(device)
    
    try:
        explainer = shap.GradientExplainer(model_optimized, X_sample_tensor)
        shap_values = explainer.shap_values(X_sample_tensor[:100])
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        mean_shap = np.abs(shap_values).mean(axis=(0, 1))
        
        feature_importance = pd.DataFrame({
            'Feature': numeric_features,
            'Importance': mean_shap
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        feature_importance['Importance_Norm'] = (feature_importance['Importance'] / 
                                                  feature_importance['Importance'].sum() * 100)
        SHAP_AVAILABLE = True
    except Exception:
        SHAP_AVAILABLE = False

if not SHAP_AVAILABLE:
    test_pred_baseline, test_actual_baseline = predict_model(model_optimized, test_loader)
    baseline_rmse = np.sqrt(mean_squared_error(test_actual_baseline.flatten(), 
                                                 test_pred_baseline.flatten()))
    
    feature_importances = []
    
    for i, feat_name in enumerate(numeric_features):
        X_test_permuted = X_test.copy()
        np.random.shuffle(X_test_permuted[:, i])
        
        test_dataset_permuted = TimeSeriesDataset(X_test_permuted, y_test, SEQ_LENGTH, PRED_LENGTH)
        test_loader_permuted = DataLoader(test_dataset_permuted, batch_size=BATCH_SIZE, 
                                           shuffle=False, num_workers=0)
        
        test_pred_permuted, test_actual_permuted = predict_model(model_optimized, test_loader_permuted)
        permuted_rmse = np.sqrt(mean_squared_error(test_actual_permuted.flatten(), 
                                                     test_pred_permuted.flatten()))
        
        importance = permuted_rmse - baseline_rmse
        feature_importances.append(importance)
    
    feature_importance = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': feature_importances
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    feature_importance['Importance_Norm'] = (feature_importance['Importance'] / 
                                              (feature_importance['Importance'].sum() + 1e-8) * 100)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].plot(train_losses_basic, label='Training set', linewidth=2)
axes[0].plot(val_losses_basic, label='Validation set', linewidth=2)
axes[0].axvline(x=len(train_losses_basic)-1, color='r', linestyle='--',
                label=f'Final epoch ({len(train_losses_basic)})', linewidth=1.5)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('Transformer Basic Model (GPU) - Training Process', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_losses_opt, label='Training set', linewidth=2)
axes[1].plot(val_losses_opt, label='Validation set', linewidth=2)
axes[1].axvline(x=len(train_losses_opt)-1, color='r', linestyle='--',
                label=f'Final epoch ({len(train_losses_opt)})', linewidth=1.5)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss (MSE)', fontsize=12)
axes[1].set_title('Transformer Optimized Model (GPU) - Training Process', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'training_curves_gpu.png', dpi=300, bbox_inches='tight')
plt.close()

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
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal prediction line')
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    ax.set_xlabel('Actual PM2.5 Concentration (μg/m³)', fontsize=11)
    ax.set_ylabel('Predicted PM2.5 Concentration (μg/m³)', fontsize=11)
    ax.set_title(f'Transformer_{model_name}(GPU) - {dataset}\nR²={r2:.4f}, RMSE={rmse:.2f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'prediction_scatter_gpu.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(18, 10))

plot_range = min(300, len(test_pred_basic_orig))
plot_idx = range(len(test_pred_basic_orig) - plot_range, len(test_pred_basic_orig))

axes[0].plot(plot_idx, test_actual_basic_orig[plot_idx, 0], 'k-', label='Actual',
             linewidth=2, alpha=0.8)
axes[0].plot(plot_idx, test_pred_basic_orig[plot_idx, 0], 'b--', label='Basic model prediction',
             linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('Sample index', fontsize=12)
axes[0].set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
axes[0].set_title('Transformer Basic Model (GPU) - Time Series Prediction Comparison (Last 300 samples of test set, Day 1 prediction)',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(plot_idx, test_actual_opt_orig[plot_idx, 0], 'k-', label='Actual',
             linewidth=2, alpha=0.8)
axes[1].plot(plot_idx, test_pred_opt_orig[plot_idx, 0], 'g--', label='Optimized model prediction',
             linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('Sample index', fontsize=12)
axes[1].set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
axes[1].set_title('Transformer Optimized Model (GPU) - Time Series Prediction Comparison (Last 300 samples of test set, Day 1 prediction)',
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'timeseries_comparison_gpu.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (model_name, y_pred, y_true, dataset) in enumerate(models_data):
    row = idx // 3
    col = idx % 3
    
    ax = axes[row, col]
    
    residuals = y_true - y_pred
    
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Value (μg/m³)', fontsize=11)
    ax.set_ylabel('Residual (μg/m³)', fontsize=11)
    ax.set_title(f'Transformer_{model_name}(GPU) - {dataset}\nResidual mean={residuals.mean():.2f}, Std dev={residuals.std():.2f}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'residuals_analysis_gpu.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

top_n = 20
top_features = feature_importance.head(top_n)

ax.barh(range(top_n), top_features['Importance_Norm'], color='steelblue')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['Feature'], fontsize=10)
ax.set_xlabel('Importance (%)', fontsize=12)
ax.set_title(f'Top {top_n} Important Features', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance_gpu.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

test_results_plot = all_results[all_results['Dataset'] == 'Test']
models = test_results_plot.loc[:, 'Model'].tolist()
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
        axes[i].set_title(f'{metric} Comparison\n(Higher is better)', fontsize=12, fontweight='bold')
    else:
        axes[i].set_title(f'{metric} Comparison\n(Lower is better)', fontsize=12, fontweight='bold')
    
    axes[i].grid(True, alpha=0.3, axis='y')
    
    for j, v in enumerate(test_results_plot[metric]):
        if metric == 'MAPE':
            axes[i].text(j, v, f'{v:.1f}%', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')
        else:
            axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison_gpu.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

errors_basic = test_actual_basic_orig[:, 0] - test_pred_basic_orig[:, 0]
errors_opt = test_actual_opt_orig[:, 0] - test_pred_opt_orig[:, 0]

axes[0].hist(errors_basic, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero error')
axes[0].set_xlabel('Prediction Error (μg/m³)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title(f'Basic Model (GPU) - Prediction Error Distribution\nMean={errors_basic.mean():.2f}, Std dev={errors_basic.std():.2f}',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].hist(errors_opt, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='Zero error')
axes[1].set_xlabel('Prediction Error (μg/m³)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title(f'Optimized Model (GPU) - Prediction Error Distribution\nMean={errors_opt.mean():.2f}, Std dev={errors_opt.std():.2f}',
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'error_distribution_gpu.png', dpi=300, bbox_inches='tight')
plt.close()

all_results.to_csv(output_dir / 'model_performance_gpu.csv', index=False, encoding='utf-8-sig')
feature_importance.to_csv(output_dir / 'feature_importance_gpu.csv', index=False, encoding='utf-8-sig')
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv(output_dir / 'best_parameters_gpu.csv', index=False, encoding='utf-8-sig')

predictions_df = pd.DataFrame({
    'Sample_Index': range(len(test_actual_basic_orig)),
    'Actual_Day1': test_actual_basic_orig[:, 0],
    'Prediction_Basic_Day1': test_pred_basic_orig[:, 0],
    'Prediction_Optimized_Day1': test_pred_opt_orig[:, 0],
    'Error_Basic': test_actual_basic_orig[:, 0] - test_pred_basic_orig[:, 0],
    'Error_Optimized': test_actual_opt_orig[:, 0] - test_pred_opt_orig[:, 0]
})
predictions_df.to_csv(output_dir / 'predictions_gpu.csv', index=False, encoding='utf-8-sig')

torch.save(model_optimized.state_dict(), model_dir / 'transformer_optimized_gpu.pth')
torch.save({
    'model_state_dict': model_optimized.state_dict(),
    'model_params': best_params,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'feature_names': numeric_features,
    'seq_length': SEQ_LENGTH,
    'pred_length': PRED_LENGTH,
    'use_amp': USE_AMP
}, model_dir / 'transformer_optimized_full_gpu.pth')

with open(model_dir / 'transformer_optimized_gpu.pkl', 'wb') as f:
    pickle.dump({
        'model': model_optimized,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_names': numeric_features
    }, f)

best_model = test_results.iloc[0]
print(f"\nBest model: {best_model['Model']}")
print(f"R² Score: {best_model['R²']:.4f}, RMSE: {best_model['RMSE']:.2f}, MAE: {best_model['MAE']:.2f}, MAPE: {best_model['MAPE']:.2f}%")
print("\nTop 5 important features:")
for i, (idx, row) in enumerate(feature_importance.head(5).iterrows(), 1):
    print(f"{i}. {row['Feature']}: {row['Importance_Norm']:.2f}%")