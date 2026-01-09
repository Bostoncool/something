import os
import sys
import re
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# -------------------------- Global Configuration -------------------------- #
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 2)

BASE_PATH = Path(".")
OUTPUT_DIR = BASE_PATH / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pollution_all_path = Path("/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)")
pollution_extra_path = Path("/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)")

start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

LOOKBACK = 30
BATCH_SIZE = 512  # RTX 5090 (32 GB) friendly
EPOCHS = 120
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 18
NUM_WORKERS = max(8, CPU_COUNT - 2)
PIN_MEMORY = torch.cuda.is_available()
PREFETCH_FACTOR = 6
USE_AMP = True
USE_COMPILE = hasattr(torch, "compile")

MODEL_SLUG = "transformer"

# -------------------------- Utility Functions ----------------------------- #


def print_header():
    separator = "=" * 80
    print(separator)
    print("Beijing PM2.5 Concentration Forecast - Transformer (Pollution Only)")
    print(separator)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    print(f"Date range: {start_date.date()} - {end_date.date()}")
    print(f"Sequence length: {LOOKBACK} days")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Num workers: {NUM_WORKERS}")
    print(f"torch.compile: {'Enabled' if USE_COMPILE else 'Disabled'}")
    print(separator)


def daterange(start, end):
    days = (end - start).days + 1
    for n in range(days):
        yield start + timedelta(days=n)


def build_file_map(base_path: Path, prefix: str):
    mapping = {}
    filename_prefix = f"{prefix}_"
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.startswith(filename_prefix) and file.endswith(".csv"):
                mapping[file] = Path(root) / file
    return mapping


def read_pollution_day(
    date_obj, file_map_all, file_map_extra, pollutants=None
):
    pollutants = pollutants or ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    date_str = date_obj.strftime("%Y%m%d")
    file_all = file_map_all.get(f"beijing_all_{date_str}.csv")
    file_extra = file_map_extra.get(f"beijing_extra_{date_str}.csv")

    if not file_all or not file_extra:
        return None

    try:
        df_all = pd.read_csv(file_all, encoding="utf-8", on_bad_lines="skip")
        df_extra = pd.read_csv(file_extra, encoding="utf-8", on_bad_lines="skip")

        df_all = df_all[~df_all["type"].str.contains("_24h|AQI", na=False)]
        df_extra = df_extra[~df_extra["type"].str.contains("_24h", na=False)]

        df_poll = pd.concat([df_all, df_extra], ignore_index=True)
        df_poll = df_poll.melt(
            id_vars=["date", "hour", "type"], var_name="station", value_name="value"
        )
        df_poll["value"] = pd.to_numeric(df_poll["value"], errors="coerce")
        df_poll = df_poll[df_poll["value"] >= 0]

        df_daily = (
            df_poll.groupby(["date", "type"])["value"].mean().reset_index()
        )
        df_daily = df_daily.pivot(index="date", columns="type", values="value")
        df_daily.index = pd.to_datetime(df_daily.index, format="%Y%m%d", errors="coerce")
        df_daily = df_daily[[col for col in pollutants if col in df_daily.columns]]
        return df_daily
    except Exception:
        return None


def read_all_pollution():
    print("\nLoading pollution data (pollution only)...")
    if not pollution_all_path.exists() or not pollution_extra_path.exists():
        print("Pollution directories not found.")
        return pd.DataFrame()

    file_map_all = build_file_map(pollution_all_path, "beijing_all")
    file_map_extra = build_file_map(pollution_extra_path, "beijing_extra")
    print(
        f"  Indexed {len(file_map_all)} 'all' files and {len(file_map_extra)} 'extra' files"
    )

    daily_frames = []
    dates = list(daterange(start_date, end_date))

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                read_pollution_day, date_obj, file_map_all, file_map_extra
            ): date_obj
            for date_obj in dates
        }

        iterator = (
            tqdm(as_completed(futures), total=len(futures), desc="Pollution days")
            if TQDM_AVAILABLE
            else as_completed(futures)
        )

        for future in iterator:
            daily_df = future.result()
            if daily_df is not None:
                daily_frames.append(daily_df)

    if not daily_frames:
        print("No pollution data read successfully.")
        return pd.DataFrame()

    df_poll = pd.concat(daily_frames)
    df_poll.sort_index(inplace=True)
    df_poll = df_poll[~df_poll.index.duplicated(keep="first")]
    df_poll.ffill(inplace=True)
    df_poll.bfill(inplace=True)
    print(f"  Pollution data shape: {df_poll.shape}")
    print(f"  Time span: {df_poll.index.min()} -> {df_poll.index.max()}")
    return df_poll


def engineer_features(df: pd.DataFrame):
    df_feat = df.copy()
    df_feat.index = pd.to_datetime(df_feat.index)

    df_feat["dayofyear"] = df_feat.index.dayofyear
    df_feat["month"] = df_feat.index.month
    df_feat["weekday"] = df_feat.index.weekday
    df_feat["is_weekend"] = df_feat["weekday"].isin([5, 6]).astype(int)
    df_feat["sin_doy"] = np.sin(2 * np.pi * df_feat["dayofyear"] / 365.25)
    df_feat["cos_doy"] = np.cos(2 * np.pi * df_feat["dayofyear"] / 365.25)

    lag_days = [1, 3, 7, 14]
    ma_windows = [3, 7, 30]

    for col in df.columns:
        for lag in lag_days:
            df_feat[f"{col}_lag{lag}"] = df_feat[col].shift(lag)
        for window in ma_windows:
            df_feat[f"{col}_ma{window}"] = (
                df_feat[col].rolling(window=window, min_periods=1).mean()
            )
        df_feat[f"{col}_diff1"] = df_feat[col].diff()

    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_feat.dropna(inplace=True)
    return df_feat


def create_sequences(features: pd.DataFrame, target: pd.Series, lookback: int):
    feature_values = features.values
    target_values = target.values
    indices = features.index

    if len(features) <= lookback:
        raise ValueError("Not enough rows to create sequences.")

    X_list, y_list, idx_list = [], [], []
    for i in range(lookback, len(features)):
        X_list.append(feature_values[i - lookback : i])
        y_list.append(target_values[i])
        idx_list.append(indices[i])

    X = np.stack(X_list)
    y = np.array(y_list)
    idx = pd.DatetimeIndex(idx_list)
    return X, y, idx


def split_train_val_test(X, y, idx, train_ratio=0.7, val_ratio=0.15):
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    splits = {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n_samples),
    }

    data = {}
    for split_name, slice_idx in splits.items():
        data[f"X_{split_name}"] = X[slice_idx]
        data[f"y_{split_name}"] = y[slice_idx]
        data[f"idx_{split_name}"] = idx[slice_idx]

    return data


def scale_and_dataloader(data, batch_size=BATCH_SIZE):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    n_features = X_train.shape[-1]
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, n_features)).reshape(
        X_train.shape
    )
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, n_features)).reshape(
        X_test.shape
    )

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    def make_loader(X_array, y_array, shuffle, drop_last):
        dataset = TensorDataset(
            torch.from_numpy(X_array).float(), torch.from_numpy(y_array).float()
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        )

    loaders = {
        "train": make_loader(X_train_scaled, y_train_scaled, True, True),
        "val": make_loader(X_val_scaled, y_val_scaled, False, False),
        "test": make_loader(X_test_scaled, y_test_scaled, False, False),
    }

    tensors = {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "idx_train": data["idx_train"],
        "idx_val": data["idx_val"],
        "idx_test": data["idx_test"],
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
    }

    return loaders, tensors


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class PollutionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.project = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, max_len=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.project(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.dropout(x)
        return self.regressor(x[:, -1, :]).squeeze(-1)


def train_model(model, loaders, epochs, learning_rate, patience):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    scaler = GradScaler(enabled=USE_AMP)

    history = {"train": [], "val": []}
    best_state = None
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        samples = 0

        for X_batch, y_batch in loaders["train"]:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                preds = model(X_batch)
                loss = criterion(preds, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * X_batch.size(0)
            samples += X_batch.size(0)

        train_loss /= samples

        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for X_batch, y_batch in loaders["val"]:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                with autocast(enabled=USE_AMP):
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_samples += X_batch.size(0)

        val_loss /= val_samples
        scheduler.step(val_loss)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1024 ** 3
                gpu_msg = f", GPU mem: {mem_alloc:.2f} GB"
            else:
                gpu_msg = ""
            print(f"Epoch {epoch:03d} - Train: {train_loss:.5f}, Val: {val_loss:.5f}{gpu_msg}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience reached).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def evaluate_model(model, X_array, scaler_y, y_true, batch_size=2048):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_array), batch_size):
            batch = torch.from_numpy(X_array[start : start + batch_size]).float()
            batch = batch.to(device, non_blocking=True)
            with autocast(enabled=USE_AMP):
                pred = model(batch)
            preds.append(pred.cpu())

    preds = torch.cat(preds).numpy()
    preds = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_true = np.array(y_true)

    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    mape = np.mean(np.abs((y_true - preds) / np.clip(y_true, 1e-6, None))) * 100
    return {"pred": preds, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def save_csv(df: pd.DataFrame, path: Path, index: bool = False):
    """统一 CSV 输出（UTF-8-SIG），便于后续脚本读取与绘图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"✓ Saved CSV: {path}")


def export_training_curves_csv(history: dict, model_slug: str = MODEL_SLUG):
    df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history.get("train", [])) + 1, dtype=int),
            "loss_train": history.get("train", []),
            "loss_val": history.get("val", []),
        }
    )
    save_csv(df, OUTPUT_DIR / f"plot_training_curves__{model_slug}.csv")


def export_metrics_csv(metrics: list[dict], model_slug: str = MODEL_SLUG):
    df = pd.DataFrame(metrics)
    save_csv(df, OUTPUT_DIR / f"metrics__{model_slug}__train_val_test.csv")
    for stage in df["Stage"].unique():
        part = df[df["Stage"] == stage].copy()
        save_csv(part, OUTPUT_DIR / f"metrics__{model_slug}__{stage.lower()}.csv")


def export_plot_csvs(idx, y_true, y_pred, stage: str, model_slug: str = MODEL_SLUG, sample_step: int = 4):
    # 1) Scatter plot data
    scatter_df = pd.DataFrame(
        {
            "Date": pd.DatetimeIndex(idx),
            "Actual_PM25": np.asarray(y_true),
            "Predicted_PM25": np.asarray(y_pred),
        }
    )
    save_csv(scatter_df, OUTPUT_DIR / f"plot_scatter__{model_slug}__{stage.lower()}.csv")

    # 2) Residuals / Error distribution
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    residuals_df = scatter_df.assign(Residual=residuals)
    save_csv(residuals_df, OUTPUT_DIR / f"plot_residuals__{model_slug}__{stage.lower()}.csv")
    save_csv(residuals_df.rename(columns={"Residual": "Error"}), OUTPUT_DIR / f"plot_error_distribution__{model_slug}__{stage.lower()}.csv")

    # 3) Simple time series (sampled to avoid too dense lines)
    ts_df = pd.DataFrame(
        {
            "time": pd.DatetimeIndex(idx),
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
        }
    ).sort_values("time").reset_index(drop=True)
    ts_df_sampled = ts_df.iloc[:: max(1, int(sample_step))].copy()
    save_csv(ts_df_sampled, OUTPUT_DIR / f"plot_ts_simple_sampled__{model_slug}__{stage.lower()}.csv")

    # 4) Time series compare (last-year sampled, integer x_axis to avoid “打结”)
    plot_range = min(365, len(ts_df))
    ts_last = ts_df.iloc[-plot_range:].copy()
    ts_last_sampled = ts_last.iloc[:: max(1, int(sample_step))].copy()
    x_axis = np.arange(len(ts_last_sampled))
    ts_common = pd.DataFrame(
        {
            "x_axis": x_axis,
            "time": ts_last_sampled["time"].values,
            "y_true": ts_last_sampled["y_true"].values,
        }
    )
    save_csv(ts_common, OUTPUT_DIR / f"plot_ts_lastyear_sampled__actual__{stage.lower()}.csv")
    save_csv(ts_common.assign(y_pred=ts_last_sampled["y_pred"].values), OUTPUT_DIR / f"plot_ts_lastyear_sampled__{model_slug}__{stage.lower()}.csv")


def export_predictions_csv(idx, actual, pred, stage: str, model_slug: str = MODEL_SLUG):
    df = pd.DataFrame({"Date": pd.DatetimeIndex(idx), "Actual": np.asarray(actual), "Prediction": np.asarray(pred)})
    df["Error"] = df["Actual"] - df["Prediction"]
    save_csv(df, OUTPUT_DIR / f"predictions__{model_slug}__{stage.lower()}.csv")


def permutation_importance_timeseries(
    model, X_test, y_test, scaler_y, feature_names, n_repeats=5, random_state=42, batch_size=2048
):
    """
    计算时间序列模型的 permutation importance。
    
    参数:
        model: 训练好的模型
        X_test: 测试集特征 (n_samples, lookback, n_features)
        y_test: 测试集真实值
        scaler_y: y 的标准化器
        feature_names: 特征名称列表
        n_repeats: 每个特征重复置换的次数
        random_state: 随机种子
        batch_size: 批处理大小
    
    返回:
        DataFrame with columns: feature, rmse_increase_mean, rmse_increase_std
    """
    model.eval()
    
    # 计算基准 RMSE
    baseline_preds = []
    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            batch = torch.from_numpy(X_test[start : start + batch_size]).float()
            batch = batch.to(device, non_blocking=True)
            with autocast(enabled=USE_AMP):
                pred = model(batch)
            baseline_preds.append(pred.cpu())
    
    baseline_preds = torch.cat(baseline_preds).numpy()
    baseline_preds = scaler_y.inverse_transform(baseline_preds.reshape(-1, 1)).flatten()
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
    
    print(f"\n计算 Permutation Importance (基准 RMSE: {baseline_rmse:.4f})...")
    print(f"  特征数量: {len(feature_names)}, 重复次数: {n_repeats}")
    
    results = []
    
    iterator = tqdm(enumerate(feature_names), total=len(feature_names), desc="特征重要性") if TQDM_AVAILABLE else enumerate(feature_names)
    
    for feat_idx, feat_name in iterator:
        rmse_increases = []
        
        for repeat in range(n_repeats):
            # 为每次重复设置不同的随机种子，确保每次置换都不同
            rng = np.random.RandomState(random_state + feat_idx * n_repeats + repeat)
            
            # 复制测试数据
            X_permuted = X_test.copy()
            
            # 找到特征在所有时间步中的索引
            # 对于时间序列，我们需要置换该特征在所有时间步的值
            # X_permuted shape: (n_samples, lookback, n_features)
            # 我们需要置换特征 feat_idx 在所有样本和时间步的值
            n_samples, lookback, n_features = X_permuted.shape
            
            # 随机打乱该特征的值
            permuted_values = X_permuted[:, :, feat_idx].flatten()
            rng.shuffle(permuted_values)
            X_permuted[:, :, feat_idx] = permuted_values.reshape(n_samples, lookback)
            
            # 评估置换后的性能
            permuted_preds = []
            with torch.no_grad():
                for start in range(0, len(X_permuted), batch_size):
                    batch = torch.from_numpy(X_permuted[start : start + batch_size]).float()
                    batch = batch.to(device, non_blocking=True)
                    with autocast(enabled=USE_AMP):
                        pred = model(batch)
                    permuted_preds.append(pred.cpu())
            
            permuted_preds = torch.cat(permuted_preds).numpy()
            permuted_preds = scaler_y.inverse_transform(permuted_preds.reshape(-1, 1)).flatten()
            permuted_rmse = np.sqrt(mean_squared_error(y_test, permuted_preds))
            
            rmse_increase = permuted_rmse - baseline_rmse
            rmse_increases.append(rmse_increase)
        
        results.append({
            "feature": feat_name,
            "rmse_increase_mean": np.mean(rmse_increases),
            "rmse_increase_std": np.std(rmse_increases),
        })
    
    fi_df = pd.DataFrame(results)
    fi_df = fi_df.sort_values("rmse_increase_mean", ascending=False).reset_index(drop=True)
    
    return fi_df


def summarize_permutation_importance(fi_df: pd.DataFrame):
    """
    Take fi_df from permutation_importance_timeseries and produce grouped summaries:
      1) by base variable (e.g., PM2.5, PM10, NO2, time)
      2) by feature type (lag/ma/diff/time/raw)
      3) by (base variable, feature type)
    Returns (by_var, by_type, by_var_type) as DataFrames.
    """
    df = fi_df.copy()

    # ---- helper parsers ----
    time_feats = {"dayofyear", "month", "weekday", "is_weekend", "sin_doy", "cos_doy"}

    def parse_base_and_type(name: str):
        # time features
        if name in time_feats:
            return ("time", "time")

        # patterns like "PM2.5_lag7", "NO2_ma30", "CO_diff1"
        m = re.match(r"^(?P<base>.+?)_(?P<kind>lag\d+|ma\d+|diff\d+)$", name)
        if m:
            base = m.group("base")
            kind = m.group("kind")
            if kind.startswith("lag"):
                return (base, "lag")
            if kind.startswith("ma"):
                return (base, "ma")
            if kind.startswith("diff"):
                return (base, "diff")
            return (base, "other")

        # raw pollutant columns could appear if ever included (here you exclude target PM2.5, but keep robust)
        # e.g., "PM10", "SO2", ...
        return (name, "raw")

    parsed = df["feature"].apply(parse_base_and_type)
    df["base_var"] = parsed.apply(lambda x: x[0])
    df["feat_type"] = parsed.apply(lambda x: x[1])

    # ---- summaries ----
    score_col = "rmse_increase_mean"

    by_var = (
        df.groupby("base_var", as_index=False)[score_col]
        .sum()
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
        .rename(columns={score_col: "rmse_increase_sum"})
    )

    by_type = (
        df.groupby("feat_type", as_index=False)[score_col]
        .sum()
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
        .rename(columns={score_col: "rmse_increase_sum"})
    )

    by_var_type = (
        df.groupby(["base_var", "feat_type"], as_index=False)[score_col]
        .sum()
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
        .rename(columns={score_col: "rmse_increase_sum"})
    )

    return by_var, by_type, by_var_type


def main():
    print_header()
    df_pollution = read_all_pollution()

    if df_pollution.empty:
        print("No pollution data available. Exiting.")
        sys.exit(1)

    df_features = engineer_features(df_pollution)
    target_col = "PM2.5"
    if target_col not in df_features.columns:
        print("Target column PM2.5 missing after feature engineering.")
        sys.exit(1)

    feature_cols = [col for col in df_features.columns if col != target_col]
    X = df_features[feature_cols]
    y = df_features[target_col]

    X_seq, y_seq, idx_seq = create_sequences(X, y, lookback=LOOKBACK)
    data_dict = split_train_val_test(X_seq, y_seq, idx_seq)
    loaders, tensors = scale_and_dataloader(data_dict)
    input_dim = X_seq.shape[-1]

    model = PollutionTransformer(
        input_dim=input_dim, model_dim=384, num_heads=8, num_layers=6, ff_dim=768, dropout=0.15
    ).to(device)

    if USE_COMPILE:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile.")
        except Exception as exc:
            print(f"torch.compile failed: {exc}. Continuing without compilation.")

    history = train_model(
        model,
        loaders=loaders,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=EARLY_STOP_PATIENCE,
    )
    export_training_curves_csv(history, model_slug=MODEL_SLUG)

    metrics = []
    for stage in ["train", "val", "test"]:
        eval_result = evaluate_model(
            model,
            X_array=tensors[f"X_{stage}"],
            scaler_y=tensors["scaler_y"],
            y_true=tensors[f"y_{stage}"],
        )
        metrics.append(
            {
                "Stage": stage,
                "RMSE": eval_result["rmse"],
                "MAE": eval_result["mae"],
                "MAPE": eval_result["mape"],
                "R2": eval_result["r2"],
            }
        )
        export_plot_csvs(
            idx=tensors[f"idx_{stage}"],
            y_true=tensors[f"y_{stage}"],
            y_pred=eval_result["pred"],
            stage=stage,
            model_slug=MODEL_SLUG,
            sample_step=4,
        )
        export_predictions_csv(
            idx=tensors[f"idx_{stage}"],
            actual=tensors[f"y_{stage}"],
            pred=eval_result["pred"],
            stage=stage,
            model_slug=MODEL_SLUG,
        )

    export_metrics_csv(metrics, model_slug=MODEL_SLUG)

    print("\nEvaluation Summary:")
    for item in metrics:
        print(
            f"  {item['Stage'].title():<5} | RMSE: {item['RMSE']:.2f} | MAE: {item['MAE']:.2f} | "
            f"MAPE: {item['MAPE']:.2f}% | R²: {item['R2']:.4f}"
        )

    # ---- Feature Importance Analysis ----
    print("\n" + "=" * 80)
    print("Feature Importance Analysis (Permutation Importance)")
    print("=" * 80)
    
    # 计算 permutation importance
    fi_df = permutation_importance_timeseries(
        model=model,
        X_test=tensors["X_test"],
        y_test=tensors["y_test"],
        scaler_y=tensors["scaler_y"],
        feature_names=feature_cols,
        n_repeats=5,
        random_state=SEED,
        batch_size=2048,
    )
    
    # 保存原始 importance 结果
    save_csv(fi_df, OUTPUT_DIR / f"feature_importance__{MODEL_SLUG}__raw.csv", index=False)
    
    # ---- grouped summaries (paper-friendly) ----
    fi_by_var, fi_by_type, fi_by_var_type = summarize_permutation_importance(fi_df)

    save_csv(fi_by_var, OUTPUT_DIR / f"feature_importance__{MODEL_SLUG}__group_by_variable.csv", index=False)
    save_csv(fi_by_type, OUTPUT_DIR / f"feature_importance__{MODEL_SLUG}__group_by_type.csv", index=False)
    save_csv(fi_by_var_type, OUTPUT_DIR / f"feature_importance__{MODEL_SLUG}__group_by_variable_type.csv", index=False)

    print("\nGrouped importance (sum of RMSE increase) — by variable:")
    print(fi_by_var.head(20).to_string(index=False))

    print("\nGrouped importance (sum of RMSE increase) — by feature type:")
    print(fi_by_type.to_string(index=False))

    print("\nTop 30 (variable, type) groups:")
    print(fi_by_var_type.head(30).to_string(index=False))

    print(f"\nCSV outputs created in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

