import os
import sys
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
import matplotlib.pyplot as plt
import seaborn as sns

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
MODEL_DIR = BASE_PATH / "models"
for path in (OUTPUT_DIR, MODEL_DIR):
    path.mkdir(parents=True, exist_ok=True)

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

sns.set_theme(style="whitegrid")

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


def plot_training_curves(history):
    plt.figure(figsize=(8, 8))
    plt.plot(history["train"], label="Train", linewidth=2)
    plt.plot(history["val"], label="Validation", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.tif", dpi=300, bbox_inches="tight")  # DPI从450改为300
    plt.close()


def plot_scatter(y_true, y_pred, stage):
    plt.figure(figsize=(8, 8))
    # 设置散点图：蓝色填充，黑色边缘
    plt.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors="black", linewidths=0.3, facecolors="tab:blue")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    # 理想预测线：红色虚线，线宽2
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Prediction")
    plt.xlabel("Actual PM2.5 (µg/m³)")
    plt.ylabel("Predicted PM2.5 (µg/m³)")
    plt.title(f"Prediction Scatter - {stage}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"prediction_scatter_{stage.lower()}.tif", dpi=300)  # DPI从450改为300
    plt.close()


def plot_timeseries(idx, y_true, y_pred, stage, window=300, y_pred_base=None):
    """
    绘制时序图
    参数:
        idx: 时间索引
        y_true: 实际值
        y_pred: 预测值（优化模型）
        stage: 阶段名称
        window: 显示窗口大小
        y_pred_base: 基础模型预测值（可选）
    """
    plt.figure(figsize=(18, 5))  # 输出比例改为18*5英寸
    time_idx = pd.DatetimeIndex(idx)[-window:]
    
    # 黑色实线：实际值，线宽2
    plt.plot(time_idx, y_true[-window:], color='black', linestyle='-', label="Actual", linewidth=2)
    
    # 如果有基础模型预测值，绘制蓝色虚线
    if y_pred_base is not None:
        plt.plot(time_idx, y_pred_base[-window:], color='blue', linestyle='--', 
                label="Base Model Prediction", linewidth=1.5)
    
    # 绿色虚线：优化模型预测值，线宽1.5
    plt.plot(time_idx, y_pred[-window:], color='green', linestyle='--', 
            label="Optimized Model Prediction", linewidth=1.5)
    
    plt.xlabel("Date")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.title(f"Time Series Comparison - {stage} (last {len(time_idx)} days)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"timeseries_{stage.lower()}.tif", dpi=300)  # DPI从450改为300
    plt.close()


def plot_residual_distribution(y_true, y_pred, stage):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 8))
    sns.histplot(residuals, bins=40, kde=True, color="steelblue")
    plt.axvline(0, color="r", linestyle="--", linewidth=2)  # 统一线宽为2
    plt.title(f"Residual Distribution - {stage}")
    plt.xlabel("Residual (µg/m³)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"residual_distribution_{stage.lower()}.tif", dpi=300)  # DPI从450改为300
    plt.close()


def plot_residual_vs_prediction(y_true, y_pred, stage):
    """
    绘制残差图，与散点图设置保持一致
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 8))
    # 与散点图保持一致的设置：蓝色填充，黑色边缘
    plt.scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors="black", linewidths=0.3, facecolors="tab:blue")
    # 理想预测线：红色虚线，线宽2
    plt.axhline(0, color="r", linestyle="--", linewidth=2)
    plt.xlabel("Predicted (µg/m³)")
    plt.ylabel("Residual (µg/m³)")
    plt.title(f"Residual vs Prediction - {stage}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"residual_vs_prediction_{stage.lower()}.tif", dpi=300)  # DPI从450改为300
    plt.close()


def save_metrics(metrics):
    df = pd.DataFrame(metrics)
    df.to_csv(OUTPUT_DIR / "transformer_metrics.csv", index=False, encoding="utf-8-sig")
    print("\nSaved metrics to transformer_metrics.csv")


def save_predictions(idx, actual, pred, stage):
    df = pd.DataFrame({"Date": idx, "Actual": actual, "Prediction": pred})
    df["Error"] = df["Actual"] - df["Prediction"]
    df.to_csv(
        OUTPUT_DIR / f"predictions_{stage.lower()}.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Saved predictions_{stage.lower()}.csv")


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
    plot_training_curves(history)

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
        plot_scatter(tensors[f"y_{stage}"], eval_result["pred"], stage)
        plot_timeseries(tensors[f"idx_{stage}"], tensors[f"y_{stage}"], eval_result["pred"], stage)
        plot_residual_distribution(tensors[f"y_{stage}"], eval_result["pred"], stage)
        plot_residual_vs_prediction(tensors[f"y_{stage}"], eval_result["pred"], stage)
        save_predictions(tensors[f"idx_{stage}"], tensors[f"y_{stage}"], eval_result["pred"], stage)

    save_metrics(metrics)

    torch.save(model.state_dict(), MODEL_DIR / "pollution_transformer.pth")
    print(f"\nModel saved to {MODEL_DIR / 'pollution_transformer.pth'}")

    print("\nEvaluation Summary:")
    for item in metrics:
        print(
            f"  {item['Stage'].title():<5} | RMSE: {item['RMSE']:.2f} | MAE: {item['MAE']:.2f} | "
            f"MAPE: {item['MAPE']:.2f}% | R²: {item['R2']:.4f}"
        )

    print("\nArtifacts created in output_transformer/.")


if __name__ == "__main__":
    main()

