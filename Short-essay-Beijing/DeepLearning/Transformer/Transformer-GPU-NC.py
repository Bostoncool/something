import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from pathlib import Path
import glob
import multiprocessing
from itertools import product
import importlib

import xarray as xr

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

if not torch.cuda.is_available():
    print("Error: CUDA/GPU device not detected!")
    import sys
    sys.exit(1)

device = torch.device('cuda')

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
compute_capability = torch.cuda.get_device_capability(0)
USE_AMP = compute_capability[0] >= 7

print(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {gpu_memory:.2f}GB, AMP: {USE_AMP}")

pollution_all_path = '/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)'
pollution_extra_path = '/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)'
era5_path = '/root/autodl-tmp/ERA5-Beijing-NC'

output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)
model_dir = Path('./models')
model_dir.mkdir(exist_ok=True)

start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

SEQ_LENGTH = 30
PRED_LENGTH = 7
BATCH_SIZE = 512 if gpu_memory >= 24 else 256 if gpu_memory >= 16 else 128 if gpu_memory >= 8 else 64

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
    args_list = [(date, file_path_dict_all, file_path_dict_extra, pollutants) for date in dates]
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, args): args[0] for args in args_list}
        
        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading pollution data"):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
        else:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
    
    if pollution_dfs:
        df_poll_all = pd.concat(pollution_dfs)
        df_poll_all.ffill(inplace=True)
        df_poll_all.fillna(df_poll_all.mean(), inplace=True)
        print(f"Pollution data loaded: {df_poll_all.shape}")
        return df_poll_all
    return pd.DataFrame()

def read_era5_file_universal(args):
    file_path, start_date_val, end_date_val, beijing_lats_val, beijing_lons_val = args
    try:
        with xr.open_dataset(file_path, engine="netcdf4", decode_times=True) as ds:
            rename_map = {}
            for tkey in ("valid_time", "forecast_time", "verification_time", "time1", "time2"):
                if tkey in ds.coords and "time" not in ds.coords:
                    rename_map[tkey] = "time"
            if "lat" in ds.coords and "latitude" not in ds.coords:
                rename_map["lat"] = "latitude"
            if "lon" in ds.coords and "longitude" not in ds.coords:
                rename_map["lon"] = "longitude"
            if rename_map:
                ds = ds.rename(rename_map)
            
            try:
                ds = xr.decode_cf(ds)
            except Exception:
                pass
            
            drop_vars = [coord for coord in ("expver", "surface") if coord in ds]
            if drop_vars:
                ds = ds.drop_vars(drop_vars)
            
            if "number" in ds.dims:
                ds = ds.mean(dim="number", skipna=True)
            
            if "time" not in ds.coords:
                return None
            
            data_vars = [v for v in ds.data_vars if v not in ds.coords]
            if not data_vars:
                return None
            
            ds_subset = ds[data_vars]
            ds_subset = ds_subset.sortby('time')
            
            try:
                ds_subset = ds_subset.sel(time=slice(start_date_val, end_date_val))
            except Exception:
                pass
            if ds_subset.sizes.get('time', 0) == 0:
                return None
            
            if 'latitude' in ds_subset.coords and 'longitude' in ds_subset.coords:
                lat_values = ds_subset['latitude']
                if len(lat_values) > 0:
                    if lat_values[0] > lat_values[-1]:
                        lat_slice = slice(beijing_lats_val.max(), beijing_lats_val.min())
                    else:
                        lat_slice = slice(beijing_lats_val.min(), beijing_lats_val.max())
                    ds_subset = ds_subset.sel(
                        latitude=lat_slice,
                        longitude=slice(beijing_lons_val.min(), beijing_lons_val.max())
                    )
                    if 'latitude' in ds_subset.dims and 'longitude' in ds_subset.dims:
                        ds_subset = ds_subset.mean(dim=['latitude', 'longitude'], skipna=True)
            
            ds_daily = ds_subset.resample(time='1D').mean(keep_attrs=False)
            ds_daily = ds_daily.dropna('time', how='all')
            if ds_daily.sizes.get('time', 0) == 0:
                return None
            
            return ds_daily.load()
    except Exception:
        return None

def read_all_era5():
    print("Loading meteorological data...")
    if not os.path.exists(era5_path):
        print(f"Error: Directory {era5_path} does not exist!")
        return pd.DataFrame()
    
    all_nc = glob.glob(os.path.join(era5_path, "**", "*.nc"), recursive=True)
    if not all_nc:
        print("Error: No NetCDF files found!")
        return pd.DataFrame()
    
    var_datasets = {}
    files_read = 0
    
    args_list = [(file_path, start_date, end_date, beijing_lats, beijing_lons) for file_path in all_nc]
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_era5_file_universal, args): args[0] for args in args_list}
        
        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Reading NetCDF files"):
                ds_result = future.result()
                if ds_result is not None:
                    for var_name in ds_result.data_vars:
                        if var_name not in var_datasets:
                            var_datasets[var_name] = []
                        var_datasets[var_name].append(ds_result[[var_name]])
                    files_read += 1
        else:
            for future in as_completed(futures):
                ds_result = future.result()
                if ds_result is not None:
                    for var_name in ds_result.data_vars:
                        if var_name not in var_datasets:
                            var_datasets[var_name] = []
                        var_datasets[var_name].append(ds_result[[var_name]])
                    files_read += 1
    
    if not var_datasets:
        return pd.DataFrame()
    
    merged_datasets = []
    for var_name, datasets in var_datasets.items():
        if datasets:
            try:
                var_merged = xr.concat(datasets, dim='time', combine_attrs='override')
                var_merged = var_merged.sortby('time')
                var_merged = var_merged.groupby('time').mean()
                merged_datasets.append(var_merged)
            except Exception:
                continue
    
    if not merged_datasets:
        return pd.DataFrame()
    
    try:
        merged_ds = xr.merge(merged_datasets, compat='override', join='outer')
        df_era5_all = merged_ds.to_dataframe()
        df_era5_all.index = pd.to_datetime(df_era5_all.index)
        df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep='first')]
        df_era5_all.sort_index(inplace=True)
        
        df_era5_all.ffill(inplace=True)
        df_era5_all.bfill(inplace=True)
        df_era5_all.fillna(df_era5_all.mean(), inplace=True)
        
        print(f"Meteorological data loaded: {df_era5_all.shape}")
        return df_era5_all
    except Exception:
        return pd.DataFrame()

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
    df_copy['day_of_year'] = df_copy.index.dayofyear
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['week_of_year'] = df_copy.index.isocalendar().week
    
    df_copy['season'] = df_copy['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )
    
    df_copy['is_heating_season'] = ((df_copy['month'] >= 11) | (df_copy['month'] <= 3)).astype(int)
    
    if 't2m' in df_copy and 'd2m' in df_copy:
        df_copy['temp_dewpoint_diff'] = df_copy['t2m'] - df_copy['d2m']
    
    if 'PM2.5' in df_copy:
        df_copy['PM2.5_lag1'] = df_copy['PM2.5'].shift(1)
        df_copy['PM2.5_lag3'] = df_copy['PM2.5'].shift(3)
        df_copy['PM2.5_lag7'] = df_copy['PM2.5'].shift(7)
        df_copy['PM2.5_ma3'] = df_copy['PM2.5'].rolling(window=3, min_periods=1).mean()
        df_copy['PM2.5_ma7'] = df_copy['PM2.5'].rolling(window=7, min_periods=1).mean()
        df_copy['PM2.5_ma30'] = df_copy['PM2.5'].rolling(window=30, min_periods=1).mean()
    
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
    
    return df_copy

def clean_data(df):
    # 异常值：使用IQR方法过滤极端值
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df.clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR, axis=1)
    # 缺失值：使用线性插值代替简单填充
    df = df.interpolate(method='linear').ffill().bfill()
    return df

if __name__ == '__main__':
    print("Step 1: Data Loading and Preprocessing")

    df_pollution = read_all_pollution()
    df_era5 = read_all_era5()

    if df_pollution.empty or df_era5.empty:
        print("Error: Data loading failed!")
        import sys
        sys.exit(1)

    df_pollution.index = pd.to_datetime(df_pollution.index)
    df_era5.index = pd.to_datetime(df_era5.index)

    df_combined = df_pollution.join(df_era5, how='inner')

    if df_combined.empty:
        print("Error: Data merge failed!")
        import sys
        sys.exit(1)

    df_combined = clean_data(df_combined)
    df_combined = create_features(df_combined)
    df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_combined.dropna(inplace=True)

    # Insert correlation heatmap here
    import seaborn as sns
    import matplotlib.pyplot as plt

    corr = df_combined.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation')
    plt.savefig(output_dir / 'correlation_heatmap.png')
    plt.close()

    print(f"Combined data shape: {df_combined.shape}")

    print("Step 2: Sequence Data Preparation")

    target = 'PM2.5'
    exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']

    numeric_features = [col for col in df_combined.select_dtypes(include=[np.number]).columns 
                        if col not in exclude_cols]

    X = df_combined[numeric_features].values
    y = df_combined[target].values

    # Change to MinMaxScaler
    scaler_X = MinMaxScaler()
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

    optimal_workers = min(8, max(4, CPU_COUNT // 4))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False
    )

    print("Step 3: Transformer Model Definition")

    class EnhancedPositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
            self.time_embed = nn.Linear(3, d_model)  # 对于year, month, day

        def forward(self, x, time_features):  # time_features: [batch, seq_len, 3]
            x = x + self.pe[:, :x.size(1), :]
            time_emb = self.time_embed(time_features)
            return x + time_emb

    class TimeSeriesTransformerSeq2Seq(nn.Module):
        def __init__(
            self,
            input_dim,
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            pred_length=7
        ):
            super().__init__()

            self.pred_length = pred_length
            self.d_model = d_model

            # ----------- Encoder -----------
            self.encoder_input_projection = nn.Linear(input_dim, d_model)
            self.encoder_positional_encoding = EnhancedPositionalEncoding(d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # ----------- Decoder -----------
            self.decoder_input_projection = nn.Linear(1, d_model)
            self.decoder_positional_encoding = EnhancedPositionalEncoding(d_model)

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu"
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

            # Output head
            self.output_layer = nn.Linear(d_model, 1)

        def forward(self, src, src_time_features=None, tgt_time_features=None):
            # Assume src_time_features and tgt_time_features are provided
            # src: [batch, seq_len, input_dim]

            # ----- Encoder -----
            src_emb = self.encoder_input_projection(src)
            src_emb = self.encoder_positional_encoding(src_emb, src_time_features)
            memory = self.encoder(src_emb)

            # ----- Decoder -----
            tgt = torch.zeros(src.size(0), self.pred_length, 1, device=src.device)
            tgt_emb = self.decoder_input_projection(tgt)
            tgt_emb = self.decoder_positional_encoding(tgt_emb, tgt_time_features)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.pred_length).to(src.device)

            output = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask
            )

            out = self.output_layer(output)
            return out.squeeze(-1)

    print("Step 4: Model Training")

    def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=50, verbose=True, 
                    gradient_accumulation_steps=1):
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode='max-autotune')
        except Exception:
            pass
        
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
            
            scheduler.step()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 20 == 0:
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
        
        return np.concatenate(predictions, axis=0), np.concatenate(actuals, axis=0)

    def evaluate_predictions(y_true, y_pred, dataset_name):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        return {
            'Dataset': dataset_name,
            'R²': r2_score(y_true_flat, y_pred_flat),
            'RMSE': np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
            'MAE': mean_absolute_error(y_true_flat, y_pred_flat),
            'MAPE': np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
        }

    def evaluate_per_step(y_true, y_pred, dataset_name):
        results = []
        for step in range(y_true.shape[1]):
            t, p = y_true[:, step], y_pred[:, step]
            results.append({
                'Dataset': dataset_name,
                'Step': step + 1,
                'R²': r2_score(t, p),
                'RMSE': np.sqrt(mean_squared_error(t, p)),
                'MAE': mean_absolute_error(t, p),
                'MAPE': np.mean(np.abs((t - p) / (t + 1e-8))) * 100
            })
        return pd.DataFrame(results)

    d_model_basic = 128
    nhead_basic = 8
    num_layers_basic = 3
    dim_feedforward_basic = 512
    dropout_basic = 0.1
    lr_basic = 0.001

    model_basic = TimeSeriesTransformerSeq2Seq(
        input_dim=len(numeric_features),
        d_model=d_model_basic,
        nhead=nhead_basic,
        num_layers=num_layers_basic,
        dim_feedforward=dim_feedforward_basic,
        dropout=dropout_basic,
        pred_length=PRED_LENGTH
    ).to(device)

    print("Training basic model...")
    model_basic, train_losses_basic, val_losses_basic, epochs_trained_basic = train_model(
        model_basic, train_loader, val_loader, 
        epochs=200, lr=lr_basic, patience=50, verbose=True
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

    results_basic = []
    results_basic.append(evaluate_predictions(train_actual_basic_orig, train_pred_basic_orig, 'Train'))
    results_basic.append(evaluate_predictions(val_actual_basic_orig, val_pred_basic_orig, 'Validation'))
    results_basic.append(evaluate_predictions(test_actual_basic_orig, test_pred_basic_orig, 'Test'))

    results_basic_df = pd.DataFrame(results_basic)
    print("\nBasic model performance:")
    print(results_basic_df.to_string(index=False))

    print("Step 5: Hyperparameter Optimization")

    if BAYESIAN_OPT_AVAILABLE:
        def transformer_evaluate(d_model, nhead, num_layers, dim_feedforward, dropout, learning_rate):
            d_model = int(d_model)
            nhead = int(nhead)
            if d_model % nhead != 0:
                d_model = (d_model // nhead) * nhead
                if d_model < nhead:
                    d_model = nhead
            
            try:
                model_temp = TimeSeriesTransformerSeq2Seq(
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
        
        if TQDM_AVAILABLE:
            pbar = tqdm(all_combos, desc="Grid search")
        else:
            pbar = all_combos
        
        for combo in pbar:
            d_model, nhead, num_layers, dim_feedforward, dropout, lr = combo
            
            if d_model % nhead != 0:
                continue
            
            try:
                model_temp = TimeSeriesTransformerSeq2Seq(
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

    print("Step 6: Train Optimized Model")

    model_optimized = TimeSeriesTransformerSeq2Seq(
        input_dim=len(numeric_features),
        d_model=best_params['d_model'],
        nhead=best_params['nhead'],
        num_layers=best_params['num_layers'],
        dim_feedforward=best_params['dim_feedforward'],
        dropout=best_params['dropout'],
        pred_length=PRED_LENGTH
    ).to(device)

    gradient_accum_steps = max(1, 512 // BATCH_SIZE)
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

    results_opt = []
    results_opt.append(evaluate_predictions(train_actual_opt_orig, train_pred_opt_orig, 'Train'))
    results_opt.append(evaluate_predictions(val_actual_opt_orig, val_pred_opt_orig, 'Validation'))
    results_opt.append(evaluate_predictions(test_actual_opt_orig, test_pred_opt_orig, 'Test'))

    results_opt_df = pd.DataFrame(results_opt)
    print("\nOptimized model performance:")
    print(results_opt_df.to_string(index=False))

    per_step_results = evaluate_per_step(test_actual_opt_orig, test_pred_opt_orig, 'Test')
    print("\nPer-step performance:")
    print(per_step_results.to_string(index=False))

    print("Step 7: Feature Importance Analysis")

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
            
            feature_importances.append(permuted_rmse - baseline_rmse)
        
        feature_importance = pd.DataFrame({
            'Feature': numeric_features,
            'Importance': feature_importances
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        feature_importance['Importance_Norm'] = (feature_importance['Importance'] / 
                                                  (feature_importance['Importance'].sum() + 1e-8) * 100)

    def plot_scatter_multistep(y_true, y_pred, model_name, save_path):
        """
        y_true: [N, 7]
        y_pred: [N, 7]
        """
        num_steps = y_true.shape[1]
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        for i in range(num_steps):
            ax = axes[i//4, i%4]

            t = y_true[:, i]
            p = y_pred[:, i]

            ax.scatter(t, p, alpha=0.5, s=20)

            min_val, max_val = min(t.min(), p.min()), max(t.max(), p.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')

            r2 = r2_score(t, p)
            rmse = np.sqrt(mean_squared_error(t, p))

            ax.set_title(f"{model_name} - Day {i+1}\nR²={r2:.3f}, RMSE={rmse:.2f}")
            ax.set_xlabel("Actual PM2.5")
            ax.set_ylabel("Predicted PM2.5")

        axes[-1, -1].axis('off')  # 空白占位
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_residuals_multistep(y_true, y_pred, model_name, save_path):
        num_steps = y_true.shape[1]
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        for i in range(num_steps):
            ax = axes[i//4, i%4]

            t = y_true[:, i]
            p = y_pred[:, i]
            residual = p - t

            ax.scatter(p, residual, alpha=0.4, s=15)
            ax.axhline(0, color='red', linestyle='--')

            ax.set_title(f"{model_name} - Day {i+1}")
            ax.set_xlabel("Predicted PM2.5")
            ax.set_ylabel("Residual")

        axes[-1, -1].axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_timeseries_day1(test_true, test_pred, num_points, model_name, save_path):
        """
        仅绘制 Day1 预测，避免窗口冲突导致的曲线跳变
        """
        plt.figure(figsize=(18, 6))
        plt.plot(test_true[:num_points, 0], label='Actual', color='black')
        plt.plot(test_pred[:num_points, 0], label='Predicted (Day1)', linestyle='--')

        plt.title(f"{model_name} - Time Series (First {num_points} Samples, Day1 Prediction)")
        plt.xlabel("Sample Index")
        plt.ylabel("PM2.5 Concentration")

        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=300)
        plt.close()

    print("Step 8: Visualization and Save Results")

    # 训练曲线图（保持不变）
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(train_losses_basic, label='Training', linewidth=2)
    axes[0].plot(val_losses_basic, label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Basic Model Training Process')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_losses_opt, label='Training', linewidth=2)
    axes[1].plot(val_losses_opt, label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (MSE)')
    axes[1].set_title('Optimized Model Training Process')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_gpu.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 多步预测散点图（按Day1-Day7分开）
    plot_scatter_multistep(
        test_actual_basic_orig,
        test_pred_basic_orig,
        "Basic Model (GPU)",
        output_dir / "scatter_multistep_basic.png"
    )

    plot_scatter_multistep(
        test_actual_opt_orig,
        test_pred_opt_orig,
        "Optimized Model (GPU)",
        output_dir / "scatter_multistep_optimized.png"
    )

    # 多步预测残差图（按Day1-Day7分开）
    plot_residuals_multistep(
        test_actual_basic_orig,
        test_pred_basic_orig,
        "Basic Model",
        output_dir / "residuals_multistep_basic.png"
    )

    plot_residuals_multistep(
        test_actual_opt_orig,
        test_pred_opt_orig,
        "Optimized Model",
        output_dir / "residuals_multistep_optimized.png"
    )

    # 时间序列对比图（只绘制Day1预测，避免打结）
    plot_timeseries_day1(
        test_actual_basic_orig,
        test_pred_basic_orig,
        num_points=300,
        model_name="Basic Model",
        save_path=output_dir / "timeseries_basic_day1.png"
    )

    plot_timeseries_day1(
        test_actual_opt_orig,
        test_pred_opt_orig,
        num_points=300,
        model_name="Optimized Model",
        save_path=output_dir / "timeseries_optimized_day1.png"
    )

    results_basic_df['Model'] = 'Transformer_Basic_GPU'
    results_opt_df['Model'] = 'Transformer_Optimized_GPU'
    all_results = pd.concat([results_basic_df, results_opt_df])
    all_results = all_results[['Model', 'Dataset', 'R²', 'RMSE', 'MAE', 'MAPE']]

    all_results.to_csv(output_dir / 'model_performance_gpu.csv', index=False, encoding='utf-8-sig')
    feature_importance.to_csv(output_dir / 'feature_importance_gpu.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([best_params]).to_csv(output_dir / 'best_parameters_gpu.csv', index=False, encoding='utf-8-sig')

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

    print("Analysis completed!")
