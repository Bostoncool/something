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

# PyTorch related
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

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

# SHAP (optional)
try:
    shap = importlib.import_module("shap")
    SHAP_AVAILABLE = True
except ImportError:
    print("Note: shap not installed, will use permutation importance.")
    print("      Use 'pip install shap' to enable SHAP analysis.")
    SHAP_AVAILABLE = False

# Set English font for matplotlib
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus signs are displayed correctly
plt.rcParams['figure.dpi'] = 100

# ============================== GPU Optimization Configuration ==============================
print("=" * 80)
print("Beijing PM2.5 Concentration Prediction - Transformer Model (GPU Optimized Version)")
print("=" * 80)

# Mandatory GPU check
if not torch.cuda.is_available():
    print("\n" + "❌" * 40)
    print("Error: CUDA/GPU device not detected!")
    print("=" * 80)
    print("\nPossible reasons:")
    print("  1. System has no NVIDIA GPU")
    print("  2. CUDA driver not properly installed")
    print("  3. PyTorch CUDA version not installed")
    print("\nSolutions:")
    print("  1. Check GPU: Run 'nvidia-smi' command")
    print("  2. Install CUDA version of PyTorch:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("  3. To use CPU version, please run the original Transformer.py")
    print("=" * 80)
    import sys
    sys.exit(1)

# Set device to GPU
device = torch.device('cuda')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For multi-GPU case

# GPU performance optimization settings
torch.backends.cudnn.benchmark = True  # Automatically find optimal algorithms
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for better performance
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for RTX 5090 (faster matrix operations)
torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cuDNN operations

# Display GPU detailed information
print(f"\n✓ GPU detection successful!")
print(f"\nGPU Information:")
print(f"  Device name: {torch.cuda.get_device_name(0)}")
print(f"  CUDA version: {torch.version.cuda}")
print(f"  cuDNN version: {torch.backends.cudnn.version()}")
print(f"  GPU count: {torch.cuda.device_count()}")

# Memory information
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"  Total memory: {gpu_memory:.2f} GB")
print(f"  Currently available memory: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")

# Compute capability
compute_capability = torch.cuda.get_device_capability(0)
print(f"  Compute capability: {compute_capability[0]}.{compute_capability[1]}")

# Mixed precision training support check
if compute_capability[0] >= 7:
    print(f"  ✓ Supports mixed precision training (Tensor Cores)")
    USE_AMP = True
else:
    print(f"  ⚠️ Partial support for mixed precision training (GPU with compute capability 7.0+ recommended)")
    USE_AMP = False

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

# Pollutant list
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# Transformer specific parameters - GPU optimized for RTX 5090 (32GB)
SEQ_LENGTH = 30  # Input sequence length (30 days)
PRED_LENGTH = 7  # Prediction length (7 days)
BATCH_SIZE = 2048  # Batch size (GPU optimized for RTX 5090 32GB: increased to maximize GPU utilization)

# Dynamically adjust batch size recommendation based on memory size
if gpu_memory >= 24:
    recommended_batch_size = 2048  # RTX 5090 can handle much larger batches
    print(f"  ✓ High-end GPU ({gpu_memory:.1f}GB), recommended batch_size: {recommended_batch_size}")
elif gpu_memory >= 16:
    recommended_batch_size = 256
    print(f"  ✓ Sufficient memory ({gpu_memory:.1f}GB), recommended batch_size: {recommended_batch_size}")
elif gpu_memory >= 8:
    recommended_batch_size = 128
    print(f"  ✓ Moderate memory ({gpu_memory:.1f}GB), recommended batch_size: {recommended_batch_size}")
elif gpu_memory >= 6:
    recommended_batch_size = 64
    print(f"  ⚠️ Limited memory ({gpu_memory:.1f}GB), recommended batch_size: {recommended_batch_size}")
    if BATCH_SIZE > recommended_batch_size:
        print(f"  ⚠️ Current batch_size({BATCH_SIZE}) may cause memory shortage, recommend reducing")
else:
    recommended_batch_size = 32
    print(f"  ⚠️ Very limited memory ({gpu_memory:.1f}GB), recommended batch_size: {recommended_batch_size}")
    if BATCH_SIZE > recommended_batch_size:
        print(f"  ⚠️ Current batch_size({BATCH_SIZE}) may cause memory shortage, recommend reducing")

print(f"\nTraining configuration:")
print(f"  Data time range: {start_date.date()} to {end_date.date()}")
print(f"  Target variable: PM2.5 concentration")
print(f"  Input sequence length: {SEQ_LENGTH} days")
print(f"  Prediction length: {PRED_LENGTH} days")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Mixed precision training: {'Enabled' if USE_AMP else 'Disabled'}")
print(f"  Output directory: {output_dir}")
print(f"  Model save directory: {model_dir}")
print(f"  CPU cores: {CPU_COUNT}, Parallel worker threads: {MAX_WORKERS}")

# GPU memory monitoring function
def print_gpu_memory(prefix=""):
    """Print GPU memory usage"""
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"{prefix}GPU memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

# ============================== Part 2: Data Loading Functions ==============================
def daterange(start, end):
    """Generate date sequence"""
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def build_file_path_dict(base_path, prefix):
    """Build a dictionary mapping date strings to file paths for O(1) lookup"""
    file_dict = {}
    filename_pattern = f"{prefix}_"
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename.startswith(filename_pattern) and filename.endswith('.csv'):
                # Extract date string from filename (format: prefix_YYYYMMDD.csv)
                date_str = filename[len(filename_pattern):-4]  # Remove prefix_ and .csv
                if len(date_str) == 8 and date_str.isdigit():  # Validate date format
                    file_dict[date_str] = os.path.join(root, filename)
    return file_dict

def read_pollution_day(args):
    """Read pollution data for a single day (multiprocessing compatible)"""
    date, file_path_dict_all, file_path_dict_extra, pollutants_list = args
    date_str = date.strftime('%Y%m%d')
    all_file = file_path_dict_all.get(date_str)
    extra_file = file_path_dict_extra.get(date_str)
    
    if not all_file or not extra_file:
        return None
    
    try:
        df_all = pd.read_csv(all_file, encoding='utf-8', on_bad_lines='skip')
        df_extra = pd.read_csv(extra_file, encoding='utf-8', on_bad_lines='skip')
        
        # Filter out 24-hour averages and AQI
        df_all = df_all[~df_all['type'].str.contains('_24h|AQI', na=False)]
        df_extra = df_extra[~df_extra['type'].str.contains('_24h', na=False)]
        
        # Merge
        df_poll = pd.concat([df_all, df_extra], ignore_index=True)
        
        # Convert to long format
        df_poll = df_poll.melt(id_vars=['date', 'hour', 'type'], 
                                var_name='station', value_name='value')
        df_poll['value'] = pd.to_numeric(df_poll['value'], errors='coerce')
        
        # Remove negative and outlier values
        df_poll = df_poll[df_poll['value'] >= 0]
        
        # Aggregate by date and type (average across all stations)
        df_daily = df_poll.groupby(['date', 'type'])['value'].mean().reset_index()
        
        # Convert to wide format
        df_daily = df_daily.pivot(index='date', columns='type', values='value')
        
        # Convert index to datetime format
        df_daily.index = pd.to_datetime(df_daily.index, format='%Y%m%d', errors='coerce')
        
        # Keep only required pollutants
        df_daily = df_daily[[col for col in pollutants_list if col in df_daily.columns]]
        
        return df_daily
    except Exception as e:
        return None

def read_all_pollution():
    """Read all pollution data in parallel using multiprocessing"""
    print("\nLoading pollution data...")
    print(f"Building file path dictionaries for fast lookup...")
    
    # Build file path dictionaries (O(1) lookup)
    file_path_dict_all = build_file_path_dict(pollution_all_path, 'beijing_all')
    file_path_dict_extra = build_file_path_dict(pollution_extra_path, 'beijing_extra')
    
    print(f"  Found {len(file_path_dict_all)} files in all directory")
    print(f"  Found {len(file_path_dict_extra)} files in extra directory")
    print(f"Using {MAX_WORKERS} parallel worker processes")
    
    dates = list(daterange(start_date, end_date))
    pollution_dfs = []
    
    # Prepare arguments for multiprocessing
    args_list = [(date, file_path_dict_all, file_path_dict_extra, pollutants) 
                  for date in dates]
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, args): args[0] for args in args_list}
        
        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Loading pollution data", unit="days"):
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
        df_poll_all = df_poll_all.ffill()
        df_poll_all = df_poll_all.fillna(df_poll_all.mean())
        print(f"Pollution data loading complete, shape: {df_poll_all.shape}")
        return df_poll_all
    return pd.DataFrame()

# ============================== Part 3: Feature Engineering ==============================
def create_features(df):
    """Create additional features from pollution data only"""
    df_copy = df.copy()
    
    # 1. Time features
    df_copy['year'] = df_copy.index.year
    df_copy['month'] = df_copy.index.month
    df_copy['day'] = df_copy.index.day
    df_copy['day_of_year'] = df_copy.index.dayofyear
    df_copy['day_of_week'] = df_copy.index.dayofweek
    # Use isocalendar() if available (pandas 1.1.0+), otherwise calculate manually
    try:
        df_copy['week_of_year'] = df_copy.index.isocalendar().week
    except AttributeError:
        # Fallback for older pandas versions - calculate week number manually
        df_copy['week_of_year'] = (df_copy.index.dayofyear - 1) // 7 + 1
    
    # Season features
    df_copy['season'] = df_copy['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )
    
    # Heating season indicator
    df_copy['is_heating_season'] = ((df_copy['month'] >= 11) | (df_copy['month'] <= 3)).astype(int)
    
    # 2. Lag features for PM2.5
    if 'PM2.5' in df_copy:
        df_copy['PM2.5_lag1'] = df_copy['PM2.5'].shift(1)
        df_copy['PM2.5_lag3'] = df_copy['PM2.5'].shift(3)
        df_copy['PM2.5_lag7'] = df_copy['PM2.5'].shift(7)
        
        # Rolling average features
        df_copy['PM2.5_ma3'] = df_copy['PM2.5'].rolling(window=3, min_periods=1).mean()
        df_copy['PM2.5_ma7'] = df_copy['PM2.5'].rolling(window=7, min_periods=1).mean()
        df_copy['PM2.5_ma30'] = df_copy['PM2.5'].rolling(window=30, min_periods=1).mean()
    
    # 3. Lag features for other pollutants
    for pollutant in ['PM10', 'SO2', 'NO2', 'CO', 'O3']:
        if pollutant in df_copy:
            df_copy[f'{pollutant}_lag1'] = df_copy[pollutant].shift(1)
            df_copy[f'{pollutant}_lag7'] = df_copy[pollutant].shift(7)
            df_copy[f'{pollutant}_ma7'] = df_copy[pollutant].rolling(window=7, min_periods=1).mean()
    
    # 4. Pollutant ratios (useful features for pollution prediction)
    if 'PM2.5' in df_copy and 'PM10' in df_copy:
        df_copy['PM2.5_PM10_ratio'] = df_copy['PM2.5'] / (df_copy['PM10'] + 1e-8)
    
    if 'NO2' in df_copy and 'O3' in df_copy:
        df_copy['NO2_O3_ratio'] = df_copy['NO2'] / (df_copy['O3'] + 1e-8)
    
    return df_copy

# ============================== Part 4: Data Loading and Preprocessing ==============================
print("\n" + "=" * 80)
print("Step 1: Data Loading and Preprocessing")
print("=" * 80)

df_pollution = read_all_pollution()

print("\nData loading check:")
print(f"  Pollution data shape: {df_pollution.shape}")

if df_pollution.empty:
    print("\n⚠️ Warning: Pollution data is empty! Please check data path and files.")
    import sys
    sys.exit(1)

df_pollution.index = pd.to_datetime(df_pollution.index)

print(f"  Pollution data time range: {df_pollution.index.min()} to {df_pollution.index.max()}")

print("\nUsing pollution data only...")
df_combined = df_pollution.copy()

if df_combined.empty:
    print("\n❌ Error: Data is empty!")
    import sys
    sys.exit(1)

print("\nCreating features...")
df_combined = create_features(df_combined)

print("\nCleaning data...")
df_combined = df_combined.replace([np.inf, -np.inf], np.nan)

initial_rows = len(df_combined)
df_combined = df_combined.dropna()
final_rows = len(df_combined)
print(f"Removed {initial_rows - final_rows} rows containing missing values")

print(f"\nFinal data shape: {df_combined.shape}")
print(f"Time range: {df_combined.index.min().date()} to {df_combined.index.max().date()}")
print(f"Sample count: {len(df_combined)}")
print(f"Feature count: {df_combined.shape[1]}")

# ============================== Part 5: Sequence Data Preparation ==============================
print("\n" + "=" * 80)
print("Step 2: Sequence Data Preparation")
print("=" * 80)

# Define target variable
target = 'PM2.5'

# Exclude columns
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']

# Select numeric features
numeric_features = [col for col in df_combined.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_cols]

print(f"\nNumber of selected features: {len(numeric_features)}")
print(f"Target variable: {target}")

# Prepare data
X = df_combined[numeric_features].values
y = df_combined[target].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

# Data standardization
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

print(f"\nPM2.5 statistics:")
print(f"  Mean: {y.mean():.2f} μg/m³")
print(f"  Std dev: {y.std():.2f} μg/m³")
print(f"  Min: {y.min():.2f} μg/m³")
print(f"  Max: {y.max():.2f} μg/m³")
print(f"  Median: {np.median(y):.2f} μg/m³")

# Create sequence data
class TimeSeriesDataset(Dataset):
    """Time series dataset"""
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

# Split by time order
n_samples = len(X_scaled)
train_size = int(n_samples * 0.70)
val_size = int(n_samples * 0.15)

X_train = X_scaled[:train_size]
X_val = X_scaled[train_size:train_size + val_size]
X_test = X_scaled[train_size + val_size:]

y_train = y_scaled[:train_size]
y_val = y_scaled[train_size:train_size + val_size]
y_test = y_scaled[train_size + val_size:]

# Create datasets
train_dataset = TimeSeriesDataset(X_train, y_train, SEQ_LENGTH, PRED_LENGTH)
val_dataset = TimeSeriesDataset(X_val, y_val, SEQ_LENGTH, PRED_LENGTH)
test_dataset = TimeSeriesDataset(X_test, y_test, SEQ_LENGTH, PRED_LENGTH)

print(f"\nDataset split:")
print(f"  Training sequences: {len(train_dataset)} ({len(train_dataset)/(len(train_dataset)+len(val_dataset)+len(test_dataset))*100:.1f}%)")
print(f"  Validation sequences: {len(val_dataset)} ({len(val_dataset)/(len(train_dataset)+len(val_dataset)+len(test_dataset))*100:.1f}%)")
print(f"  Test sequences: {len(test_dataset)} ({len(test_dataset)/(len(train_dataset)+len(val_dataset)+len(test_dataset))*100:.1f}%)")

# Create data loaders - GPU optimized for high-end GPUs
print("\nCreating data loaders (GPU optimized for RTX 5090)...")
# Optimize num_workers based on CPU cores (use more workers for better GPU utilization)
# For RTX 5090, use more workers to keep GPU fed with data
optimal_workers = min(32, max(16, CPU_COUNT // 2))  # Use 16-32 workers for RTX 5090
optimal_prefetch = 8  # Increase prefetch for better pipeline (prefetch more batches)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=optimal_workers,  # GPU optimization: increased workers
    pin_memory=True,  # GPU optimization: pinned memory
    persistent_workers=True,  # GPU optimization: persistent workers
    prefetch_factor=optimal_prefetch,  # GPU optimization: increased prefetch
    drop_last=False  # Keep all data
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=optimal_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=optimal_prefetch,
    drop_last=False
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=optimal_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=optimal_prefetch,
    drop_last=False
)

print(f"\nBatch count:")
print(f"  Training batches: {len(train_loader)}")
print(f"  Validation batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")
print(f"\nData loading optimization:")
print(f"  num_workers: {optimal_workers} (asynchronous data loading, optimized for {CPU_COUNT} CPU cores)")
print(f"  pin_memory: True (accelerate data transfer to GPU)")
print(f"  persistent_workers: True (reuse worker processes)")
print(f"  prefetch_factor: {optimal_prefetch} (prefetch {optimal_prefetch} batches for better GPU utilization)")
print(f"\nGPU acceleration settings:")
print(f"  TF32 acceleration: {'Enabled' if torch.backends.cuda.matmul.allow_tf32 else 'Disabled'}")
print(f"  cuDNN benchmark: {'Enabled' if torch.backends.cudnn.benchmark else 'Disabled'}")
print(f"  cuDNN deterministic: {'Disabled' if not torch.backends.cudnn.deterministic else 'Enabled'} (non-deterministic for better performance)")

# ============================== Part 6: Transformer Model Definition ==============================
print("\n" + "=" * 80)
print("Step 3: Transformer Model Definition")
print("=" * 80)

class PositionalEncoding(nn.Module):
    """Positional encoding"""
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
    """Time series Transformer model"""
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.1, pred_length=7):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.pred_length = pred_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder - optimized for GPU performance
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',  # GELU activation for better performance
            norm_first=False,  # Post-norm (standard Transformer)
            bias=True  # Enable bias for better expressiveness
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer - multi-step prediction
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, pred_length)
        )
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        
    def forward(self, src):
        # src: (batch_size, seq_length, input_dim)
        
        # Input projection
        src = self.input_projection(src)  # (batch_size, seq_length, d_model)
        
        # Positional encoding
        src = self.pos_encoder(src)
        
        # Transformer encoding
        memory = self.transformer_encoder(src)  # (batch_size, seq_length, d_model)
        
        # Use output from last time step for prediction
        output = memory[:, -1, :]  # (batch_size, d_model)
        
        # Multi-step prediction
        output = self.decoder(output)  # (batch_size, pred_length)
        
        return output

print("✓ Transformer model architecture definition complete")
print(f"  Input feature dimension: {len(numeric_features)}")
print(f"  Sequence length: {SEQ_LENGTH}")
print(f"  Prediction length: {PRED_LENGTH}")

# ============================== Part 7: GPU Optimized Training and Evaluation Functions ==============================
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=50, verbose=True, 
                gradient_accumulation_steps=1):
    """Train model - GPU optimized version (mixed precision training + compile optimization)"""
    # Compile model for better performance (PyTorch 2.0+)
    # IMPORTANT: Compile BEFORE moving to GPU and BEFORE creating optimizer
    model_compiled = False
    try:
        if hasattr(torch, 'compile'):
            print("  Compiling model with torch.compile for optimal performance...")
            # Use 'reduce-overhead' mode for faster compilation, or 'max-autotune' for best performance
            model = torch.compile(model, mode='reduce-overhead', fullgraph=False)  # Faster compilation
            model_compiled = True
            print("  ✓ Model compilation successful")
        else:
            print("  ⚠️ torch.compile not available (requires PyTorch 2.0+)")
    except Exception as e:
        print(f"  ⚠️ Model compilation failed: {e}, continuing without compilation")
    
    # Note: channels_last is mainly for CNNs with 4D tensors (NCHW)
    # For Transformer with 3D tensors (N, L, D), we skip channels_last
    # but keep the model on GPU with optimal settings
    
    # Enable CUDA graph optimization for better performance (PyTorch 2.0+)
    use_cuda_graph = False
    if hasattr(torch.cuda, 'graph') and model_compiled:
        try:
            use_cuda_graph = True
            print("  ✓ CUDA graph optimization available")
        except:
            pass
    
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                       patience=10)
    criterion = nn.MSELoss()
    
    # Mixed precision training
    scaler = GradScaler('cuda', enabled=USE_AMP)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # For calculating training speed and GPU utilization
    total_samples = 0
    total_time = 0
    
    print(f"\nTraining configuration:")
    print(f"  Mixed precision training: {'Enabled' if USE_AMP else 'Disabled'}")
    print(f"  TF32 acceleration: {'Enabled' if torch.backends.cuda.matmul.allow_tf32 else 'Disabled'}")
    print(f"  Model compilation: {'Enabled' if model_compiled else 'Disabled'}")
    print(f"  Gradient clipping: Enabled (max_norm=1.0)")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {BATCH_SIZE * gradient_accumulation_steps}")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0
        batch_count = 0
        
        optimizer.zero_grad(set_to_none=True)  # Zero gradients at start of epoch
        
        # Warmup for CUDA graph (if enabled)
        if use_cuda_graph and epoch == 0:
            # Warmup iteration
            warmup_batch = next(iter(train_loader))
            warmup_X = warmup_batch[0].to(device, non_blocking=True)
            warmup_y = warmup_batch[1].to(device, non_blocking=True)
            with autocast('cuda', enabled=USE_AMP):
                _ = model(warmup_X)
            torch.cuda.synchronize()
        
        # Start GPU timing for the entire epoch
        if torch.cuda.is_available():
            epoch_start_event = torch.cuda.Event(enable_timing=True)
            epoch_end_event = torch.cuda.Event(enable_timing=True)
            epoch_start_event.record()
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            # Transfer to GPU with non-blocking for better performance
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            # Mixed precision training
            with autocast('cuda', enabled=USE_AMP):
                output = model(batch_X)
                loss = criterion(output, batch_y)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
            # Backward propagation
            scaler.scale(loss).backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping (prevent gradient explosion)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item() * gradient_accumulation_steps  # Scale back for logging
            batch_count += 1
            total_samples += batch_X.size(0)
        
        # End GPU timing and synchronize
        gpu_time_total = 0.0
        if torch.cuda.is_available():
            epoch_end_event.record()
            torch.cuda.synchronize()
            gpu_time_total = epoch_start_event.elapsed_time(epoch_end_event) / 1000.0  # Convert to seconds
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Transfer to GPU with non-blocking for better performance
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                with autocast('cuda', enabled=USE_AMP):
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            samples_per_sec = len(train_loader.dataset) / epoch_time
            # Calculate GPU utilization (GPU time / total epoch time)
            gpu_utilization = (gpu_time_total / epoch_time * 100) if epoch_time > 0 and torch.cuda.is_available() else 0
            print(f"  Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}, "
                  f"Speed: {samples_per_sec:.1f} samples/sec, Time: {epoch_time:.2f}s, "
                  f"GPU Util: {gpu_utilization:.1f}%")
            print_gpu_memory("    ")
        
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping triggered at Epoch {epoch+1}")
            break
        
        # Periodic GPU cache cleanup (less frequent for better performance)
        if (epoch + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final statistics
    if verbose and total_time > 0:
        avg_samples_per_sec = total_samples / total_time
        print(f"\nTraining statistics:")
        print(f"  Average speed: {avg_samples_per_sec:.1f} samples/sec")
        print(f"  Total training time: {total_time:.2f}s")
    
    return model, train_losses, val_losses, epoch + 1

def predict_model(model, data_loader):
    """Model prediction - GPU optimized version"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            # Transfer to GPU with non-blocking for better performance
            batch_X = batch_X.to(device, non_blocking=True)
            
            with autocast('cuda', enabled=USE_AMP):
                output = model(batch_X)
            
            predictions.append(output.cpu().numpy())
            actuals.append(batch_y.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    return predictions, actuals

def evaluate_predictions(y_true, y_pred, dataset_name):
    """Evaluate prediction results"""
    # For multi-step prediction, calculate average metrics for each step
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

# ============================== Part 8: Basic Model Training ==============================
print("\n" + "=" * 80)
print("Step 4: Transformer Basic Model Training (GPU Accelerated)")
print("=" * 80)

# Basic model parameters
d_model_basic = 128
nhead_basic = 8
num_layers_basic = 3
dim_feedforward_basic = 512
dropout_basic = 0.1
lr_basic = 0.001

print("\nBasic model parameters:")
print(f"  d_model: {d_model_basic}")
print(f"  nhead: {nhead_basic}")
print(f"  num_layers: {num_layers_basic}")
print(f"  dim_feedforward: {dim_feedforward_basic}")
print(f"  dropout: {dropout_basic}")
print(f"  learning_rate: {lr_basic}")

# Create model
model_basic = TimeSeriesTransformer(
    input_dim=len(numeric_features),
    d_model=d_model_basic,
    nhead=nhead_basic,
    num_layers=num_layers_basic,
    dim_feedforward=dim_feedforward_basic,
    dropout=dropout_basic,
    pred_length=PRED_LENGTH
).to(device)

print(f"\nModel parameter count: {sum(p.numel() for p in model_basic.parameters()):,}")
print_gpu_memory("After model loading ")

print("\nStarting basic model training...")
start_time = time.time()
# Use gradient accumulation to simulate larger batch size for better GPU utilization
gradient_accum_basic = max(1, 2048 // BATCH_SIZE)  # Target effective batch size of 2048
model_basic, train_losses_basic, val_losses_basic, epochs_trained_basic = train_model(
    model_basic, train_loader, val_loader, 
    epochs=200, lr=lr_basic, patience=50, verbose=True,
    gradient_accumulation_steps=gradient_accum_basic  # Use gradient accumulation for better GPU utilization
)
training_time_basic = time.time() - start_time

print(f"\nBasic model training completed")
print(f"  Training epochs: {epochs_trained_basic}")
print(f"  Training time: {training_time_basic:.2f}s")
print(f"  Final training loss: {train_losses_basic[-1]:.6f}")
print(f"  Final validation loss: {val_losses_basic[-1]:.6f}")
print_gpu_memory("After training ")

# Clean up GPU cache
torch.cuda.empty_cache()

# Prediction
print("\nMaking predictions...")
train_pred_basic, train_actual_basic = predict_model(model_basic, train_loader)
val_pred_basic, val_actual_basic = predict_model(model_basic, val_loader)
test_pred_basic, test_actual_basic = predict_model(model_basic, test_loader)

# Inverse transform
train_pred_basic_orig = scaler_y.inverse_transform(train_pred_basic)
train_actual_basic_orig = scaler_y.inverse_transform(train_actual_basic)
val_pred_basic_orig = scaler_y.inverse_transform(val_pred_basic)
val_actual_basic_orig = scaler_y.inverse_transform(val_actual_basic)
test_pred_basic_orig = scaler_y.inverse_transform(test_pred_basic)
test_actual_basic_orig = scaler_y.inverse_transform(test_actual_basic)

# Evaluation
results_basic = []
results_basic.append(evaluate_predictions(train_actual_basic_orig, train_pred_basic_orig, 'Train'))
results_basic.append(evaluate_predictions(val_actual_basic_orig, val_pred_basic_orig, 'Validation'))
results_basic.append(evaluate_predictions(test_actual_basic_orig, test_pred_basic_orig, 'Test'))

results_basic_df = pd.DataFrame(results_basic)
print("\nBasic model performance:")
print(results_basic_df.to_string(index=False))

# ============================== Part 9: Hyperparameter Optimization ==============================
print("\n" + "=" * 80)
print("Step 5: Hyperparameter Optimization (GPU Accelerated)")
print("=" * 80)

if BAYESIAN_OPT_AVAILABLE:
    print("\nUsing Bayesian optimization for hyperparameter search...")
    
    def transformer_evaluate(d_model, nhead, num_layers, dim_feedforward, dropout, learning_rate):
        """Bayesian optimization objective function"""
        # Ensure nhead divides d_model
        d_model = int(d_model)
        nhead = int(nhead)
        if d_model % nhead != 0:
            # Adjust d_model to be divisible by nhead
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
            
            # Clean up memory
            del model_temp
            torch.cuda.empty_cache()
            
            return -best_val_loss  # Negative because we want to maximize
        except Exception as e:
            print(f"  Error: {e}")
            torch.cuda.empty_cache()
            return -999999
    
    # Define search space
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
    
    # Ensure d_model is divisible by nhead
    if best_params['d_model'] % best_params['nhead'] != 0:
        best_params['d_model'] = (best_params['d_model'] // best_params['nhead']) * best_params['nhead']
    
    print(f"\nBest parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  Best validation loss: {-optimizer.max['target']:.6f}")

else:
    # Grid search
    print("\nUsing grid search for hyperparameter optimization...")
    
    param_grid = {
        'd_model': [64, 128, 192],
        'nhead': [4, 8],
        'num_layers': [2, 3],
        'dim_feedforward': [256, 512],
        'dropout': [0.1, 0.2],
        'learning_rate': [0.001, 0.0005]
    }
    
    total_combinations = int(np.prod([len(v) for v in param_grid.values()]))
    print(f"Total {total_combinations} parameter combinations")
    
    all_combos = list(product(*param_grid.values()))
    
    best_val_loss = float('inf')
    best_params = {}
    
    if TQDM_AVAILABLE:
        pbar = tqdm(all_combos, desc="Grid search", unit="combo")
    else:
        pbar = all_combos
    
    for i, combo in enumerate(pbar):
        d_model, nhead, num_layers, dim_feedforward, dropout, lr = combo
        
        # Ensure d_model is divisible by nhead
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
            
            if not TQDM_AVAILABLE and (i + 1) % 5 == 0:
                print(f"  Tested {i+1}/{total_combinations} combinations, current best validation loss: {best_val_loss:.6f}")
        
        except Exception as e:
            torch.cuda.empty_cache()
            continue
    
    print(f"\nBest parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  Best validation loss: {best_val_loss:.6f}")

# ============================== Part 10: Train Optimized Model ==============================
print("\n" + "=" * 80)
print("Step 6: Train Optimized Model with Best Parameters (GPU Accelerated)")
print("=" * 80)

print("\nOptimized model parameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# Create optimized model
model_optimized = TimeSeriesTransformer(
    input_dim=len(numeric_features),
    d_model=best_params['d_model'],
    nhead=best_params['nhead'],
    num_layers=best_params['num_layers'],
    dim_feedforward=best_params['dim_feedforward'],
    dropout=best_params['dropout'],
    pred_length=PRED_LENGTH
).to(device)

print(f"\nOptimized model parameter count: {sum(p.numel() for p in model_optimized.parameters()):,}")
print_gpu_memory("After optimized model loading ")

print("\nStarting optimized model training...")
start_time = time.time()
# Use gradient accumulation for larger effective batch size if needed
# For RTX 5090, use larger effective batch size
gradient_accum_steps = max(1, 4096 // BATCH_SIZE)  # Target effective batch size of 4096 for RTX 5090
model_optimized, train_losses_opt, val_losses_opt, epochs_trained_opt = train_model(
    model_optimized, train_loader, val_loader,
    epochs=300, lr=best_params['learning_rate'], patience=100, verbose=True,
    gradient_accumulation_steps=gradient_accum_steps
)
training_time_opt = time.time() - start_time

print(f"\nOptimized model training completed")
print(f"  Training epochs: {epochs_trained_opt}")
print(f"  Training time: {training_time_opt:.2f}s")
print(f"  Final training loss: {train_losses_opt[-1]:.6f}")
print(f"  Final validation loss: {val_losses_opt[-1]:.6f}")
print_gpu_memory("After training ")

# Clean up GPU cache
torch.cuda.empty_cache()

# Prediction
print("\nMaking predictions...")
train_pred_opt, train_actual_opt = predict_model(model_optimized, train_loader)
val_pred_opt, val_actual_opt = predict_model(model_optimized, val_loader)
test_pred_opt, test_actual_opt = predict_model(model_optimized, test_loader)

# Inverse transform
train_pred_opt_orig = scaler_y.inverse_transform(train_pred_opt)
train_actual_opt_orig = scaler_y.inverse_transform(train_actual_opt)
val_pred_opt_orig = scaler_y.inverse_transform(val_pred_opt)
val_actual_opt_orig = scaler_y.inverse_transform(val_actual_opt)
test_pred_opt_orig = scaler_y.inverse_transform(test_pred_opt)
test_actual_opt_orig = scaler_y.inverse_transform(test_actual_opt)

# Evaluation
results_opt = []
results_opt.append(evaluate_predictions(train_actual_opt_orig, train_pred_opt_orig, 'Train'))
results_opt.append(evaluate_predictions(val_actual_opt_orig, val_pred_opt_orig, 'Validation'))
results_opt.append(evaluate_predictions(test_actual_opt_orig, test_pred_opt_orig, 'Test'))

results_opt_df = pd.DataFrame(results_opt)
print("\nOptimized model performance:")
print(results_opt_df.to_string(index=False))

# ============================== Part 11: Model Comparison ==============================
print("\n" + "=" * 80)
print("Step 7: Model Performance Comparison")
print("=" * 80)

results_basic_df['Model'] = 'Transformer_Basic_GPU'
results_opt_df['Model'] = 'Transformer_Optimized_GPU'
all_results = pd.concat([results_basic_df, results_opt_df])

all_results = all_results[['Model', 'Dataset', 'R²', 'RMSE', 'MAE', 'MAPE']]

print("\nAll model performance comparison:")
print(all_results.to_string(index=False))

test_results = all_results[all_results['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\nTest set performance ranking:")
print(test_results.to_string(index=False))

# Performance improvement
basic_test_r2 = results_basic_df[results_basic_df['Dataset'] == 'Test']['R²'].values[0]
opt_test_r2 = results_opt_df[results_opt_df['Dataset'] == 'Test']['R²'].values[0]
basic_test_rmse = results_basic_df[results_basic_df['Dataset'] == 'Test']['RMSE'].values[0]
opt_test_rmse = results_opt_df[results_opt_df['Dataset'] == 'Test']['RMSE'].values[0]

r2_improvement = (opt_test_r2 - basic_test_r2) / (abs(basic_test_r2) + 1e-8) * 100
rmse_improvement = (basic_test_rmse - opt_test_rmse) / basic_test_rmse * 100

print(f"\nOptimization effect:")
print(f"  R² improvement: {r2_improvement:.2f}%")
print(f"  RMSE reduction: {rmse_improvement:.2f}%")

# ============================== Part 12: Feature Importance Analysis ==============================
print("\n" + "=" * 80)
print("Step 8: Feature Importance Analysis")
print("=" * 80)

if SHAP_AVAILABLE:
    print("\nUsing SHAP for feature importance analysis...")
    print("Note: SHAP computation is slow, using sample subset...")
    
    # Use small sample for SHAP analysis
    sample_size = min(500, len(test_dataset))
    sample_indices = np.random.choice(len(test_dataset), sample_size, replace=False)
    
    X_sample = []
    for idx in sample_indices:
        X_seq, _ = test_dataset[idx]
        X_sample.append(X_seq.numpy())
    X_sample = np.array(X_sample)
    X_sample_tensor = torch.FloatTensor(X_sample).to(device)
    
    try:
        # Use GradientExplainer
        explainer = shap.GradientExplainer(model_optimized, X_sample_tensor)
        shap_values = explainer.shap_values(X_sample_tensor[:100])  # Use fewer samples for computation
        
        # Calculate average absolute SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Average over sequences and samples
        mean_shap = np.abs(shap_values).mean(axis=(0, 1))
        
        feature_importance = pd.DataFrame({
            'Feature': numeric_features,
            'Importance': mean_shap
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        feature_importance['Importance_Norm'] = (feature_importance['Importance'] / 
                                                  feature_importance['Importance'].sum() * 100)
        
        print(f"\nTop 20 important features (SHAP):")
        print(feature_importance.head(20)[['Feature', 'Importance_Norm']].to_string(index=False))
        
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("Using permutation importance as alternative...")
        SHAP_AVAILABLE = False

if not SHAP_AVAILABLE:
    print("\nUsing permutation importance for feature analysis...")
    
    # Get baseline performance
    test_pred_baseline, test_actual_baseline = predict_model(model_optimized, test_loader)
    baseline_rmse = np.sqrt(mean_squared_error(test_actual_baseline.flatten(), 
                                                 test_pred_baseline.flatten()))
    
    feature_importances = []
    
    print("Calculating permutation importance...")
    for i, feat_name in enumerate(numeric_features):
        if i % 5 == 0 and i > 0:
            print(f"  Processed {i}/{len(numeric_features)} features")
        
        # Create permuted dataset
        X_test_permuted = X_test.copy()
        np.random.shuffle(X_test_permuted[:, i])
        
        test_dataset_permuted = TimeSeriesDataset(X_test_permuted, y_test, SEQ_LENGTH, PRED_LENGTH)
        test_loader_permuted = DataLoader(test_dataset_permuted, batch_size=BATCH_SIZE, 
                                           shuffle=False, num_workers=0)
        
        # Prediction
        test_pred_permuted, test_actual_permuted = predict_model(model_optimized, test_loader_permuted)
        permuted_rmse = np.sqrt(mean_squared_error(test_actual_permuted.flatten(), 
                                                     test_pred_permuted.flatten()))
        
        # Importance = increase in RMSE after permutation
        importance = permuted_rmse - baseline_rmse
        feature_importances.append(importance)
    
    feature_importance = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': feature_importances
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    feature_importance['Importance_Norm'] = (feature_importance['Importance'] / 
                                              (feature_importance['Importance'].sum() + 1e-8) * 100)
    
    print(f"\nTop 20 important features (Permutation Importance):")
    print(feature_importance.head(20)[['Feature', 'Importance_Norm']].to_string(index=False))

# ============================== Part 13: Visualization ==============================
print("\n" + "=" * 80)
print("Step 9: Generate Visualization Charts")
print("=" * 80)

# 13.1 Training process curves
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
print("Saved: training_curves_gpu.png")
plt.close()

# 13.2 Prediction vs Actual scatter plots (using day 1 prediction)
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
print("Saved: prediction_scatter_gpu.png")
plt.close()

# 13.3 Time series prediction comparison (day 1 prediction)
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
print("Saved: timeseries_comparison_gpu.png")
plt.close()

# 13.4 Residual analysis
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
print("Saved: residuals_analysis_gpu.png")
plt.close()

# 13.5 Feature importance plot
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
print("Saved: feature_importance_gpu.png")
plt.close()

# 13.6 Model performance comparison bar chart
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
print("Saved: model_comparison_gpu.png")
plt.close()

# 13.7 Error distribution histogram
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
print("Saved: error_distribution_gpu.png")
plt.close()

# ============================== Part 14: Save Results ==============================
print("\n" + "=" * 80)
print("Step 10: Save Results")
print("=" * 80)

# Save model performance
all_results.to_csv(output_dir / 'model_performance_gpu.csv', index=False, encoding='utf-8-sig')
print("Saved: model_performance_gpu.csv")

# Save feature importance
feature_importance.to_csv(output_dir / 'feature_importance_gpu.csv', index=False, encoding='utf-8-sig')
print("Saved: feature_importance_gpu.csv")

# Save best parameters
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv(output_dir / 'best_parameters_gpu.csv', index=False, encoding='utf-8-sig')
print("Saved: best_parameters_gpu.csv")

# Save prediction results (day 1 prediction)
predictions_df = pd.DataFrame({
    'Sample_Index': range(len(test_actual_basic_orig)),
    'Actual_Day1': test_actual_basic_orig[:, 0],
    'Prediction_Basic_Day1': test_pred_basic_orig[:, 0],
    'Prediction_Optimized_Day1': test_pred_opt_orig[:, 0],
    'Error_Basic': test_actual_basic_orig[:, 0] - test_pred_basic_orig[:, 0],
    'Error_Optimized': test_actual_opt_orig[:, 0] - test_pred_opt_orig[:, 0]
})
predictions_df.to_csv(output_dir / 'predictions_gpu.csv', index=False, encoding='utf-8-sig')
print("Saved: predictions_gpu.csv")

# Save model
torch.save(model_optimized.state_dict(), model_dir / 'transformer_optimized_gpu.pth')
print("Saved: transformer_optimized_gpu.pth")

# Save complete model (including architecture)
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
print("Saved: transformer_optimized_full_gpu.pth")

# Save using pickle (optional)
with open(model_dir / 'transformer_optimized_gpu.pkl', 'wb') as f:
    pickle.dump({
        'model': model_optimized,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_names': numeric_features
    }, f)
print("Saved: transformer_optimized_gpu.pkl")

# ============================== Part 15: Summary Report ==============================
print("\n" + "=" * 80)
print("Analysis completed!")
print("=" * 80)

print("\nGenerated files:")
print("\nCSV files:")
print("  - model_performance_gpu.csv       Model performance comparison")
print("  - feature_importance_gpu.csv      Feature importance")
print("  - best_parameters_gpu.csv         Best parameters")
print("  - predictions_gpu.csv             Prediction results")

print("\nChart files:")
print("  - training_curves_gpu.png         Training process curves")
print("  - prediction_scatter_gpu.png      Prediction vs Actual scatter plots")
print("  - timeseries_comparison_gpu.png   Time series comparison")
print("  - residuals_analysis_gpu.png      Residual analysis")
print("  - feature_importance_gpu.png      Feature importance plot")
print("  - model_comparison_gpu.png        Model performance comparison")
print("  - error_distribution_gpu.png      Error distribution")

print("\nModel files:")
print("  - transformer_optimized_gpu.pth       Transformer model (weights)")
print("  - transformer_optimized_full_gpu.pth  Transformer model (complete)")
print("  - transformer_optimized_gpu.pkl       Transformer model (pickle format)")

# Best model information
best_model = test_results.iloc[0]
print(f"\nBest model: {best_model['Model']}")
print(f"  R² Score: {best_model['R²']:.4f}")
print(f"  RMSE: {best_model['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model['MAPE']:.2f}%")

print("\nTop 5 most important features:")
for i, (idx, row) in enumerate(feature_importance.head(5).iterrows(), 1):
    print(f"  {i}. {row['Feature']}: {row['Importance_Norm']:.2f}%")

print(f"\nModel training time:")
print(f"  Basic model: {training_time_basic:.2f}s")
print(f"  Optimized model: {training_time_opt:.2f}s")

print("\nMulti-step prediction info:")
print(f"  Input sequence length: {SEQ_LENGTH} days")
print(f"  Prediction horizon: {PRED_LENGTH} days")

print("\nGPU optimization features:")
print(f"  Mixed precision training: {'Enabled' if USE_AMP else 'Disabled'}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Data loading workers: 4")
print(f"  Pin Memory: Enabled")
print(f"  Gradient clipping: Enabled")

print_gpu_memory("\nFinal ")

print("\n" + "=" * 80)
print("Transformer PM2.5 Concentration Prediction Completed! (GPU Optimized Version)")
print("=" * 80)