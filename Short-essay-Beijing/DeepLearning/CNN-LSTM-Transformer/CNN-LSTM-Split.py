import os
import warnings
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Install tqdm for richer progress bars: pip install tqdm")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DATA_DIR = Path(os.getenv("POLLUTION_BASE_DIR", "/root/autodl-tmp/Benchmark"))
pollution_all_path = os.getenv(
    "POLLUTION_ALL_PATH", str(BASE_DATA_DIR / "all(AQI+PM2.5+PM10)")
)
pollution_extra_path = os.getenv(
    "POLLUTION_EXTRA_PATH", str(BASE_DATA_DIR / "extra(SO2+NO2+CO+O3)")
)

output_dir = Path("./output")
model_dir = Path("./models")
output_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)
pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

SEQUENCE_LENGTH = 30
BATCH_SIZE = 2048 if torch.cuda.is_available() else 128
NUM_EPOCHS = 120
PATIENCE = 20
LEARNING_RATE = 2e-4

CPU_COUNT = os.cpu_count() or 8
NUM_WORKERS = min(24, max(4, CPU_COUNT // 2))
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = NUM_WORKERS > 0
PREFETCH_FACTOR = 6 if NUM_WORKERS > 0 else None


def daterange(start: datetime, end: datetime):
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)


def build_file_path_dict(base_path: str, prefix: str):
    file_dict = {}
    filename_pattern = f"{prefix}_"
    for root, _, files in os.walk(base_path):
        for filename in files:
            if filename.startswith(filename_pattern) and filename.endswith(".csv"):
                date_str = filename[len(filename_pattern) : -4]
                if len(date_str) == 8 and date_str.isdigit():
                    file_dict[date_str] = os.path.join(root, filename)
    return file_dict


def read_pollution_day(args):
    date, all_dict, extra_dict, pollutant_cols = args
    date_str = date.strftime("%Y%m%d")
    all_file = all_dict.get(date_str)
    extra_file = extra_dict.get(date_str)
    if not all_file or not extra_file:
        return None
    try:
        df_all = pd.read_csv(all_file, encoding="utf-8", on_bad_lines="skip")
        df_extra = pd.read_csv(extra_file, encoding="utf-8", on_bad_lines="skip")
        df_all = df_all[~df_all["type"].str.contains("_24h|AQI", na=False)]
        df_extra = df_extra[~df_extra["type"].str.contains("_24h", na=False)]
        df_poll = pd.concat([df_all, df_extra], ignore_index=True)
        df_poll = df_poll.melt(
            id_vars=["date", "hour", "type"], var_name="station", value_name="value"
        )
        df_poll["value"] = pd.to_numeric(df_poll["value"], errors="coerce")
        df_poll = df_poll[df_poll["value"] >= 0]
        df_daily = (
            df_poll.groupby(["date", "type"])["value"].mean().reset_index().pivot(
                index="date", columns="type", values="value"
            )
        )
        df_daily.index = pd.to_datetime(df_daily.index, format="%Y%m%d", errors="coerce")
        df_daily = df_daily[[col for col in pollutant_cols if col in df_daily.columns]]
        return df_daily
    except Exception:
        return None


def read_all_pollution():
    print("\nLoading pollution data only...")
    all_dict = build_file_path_dict(pollution_all_path, "beijing_all")
    extra_dict = build_file_path_dict(pollution_extra_path, "beijing_extra")
    if not all_dict or not extra_dict:
        print("No pollution files detected, please verify the paths.")
        return pd.DataFrame()

    dates = list(daterange(start_date, end_date))
    args_list = [(date, all_dict, extra_dict, pollutants) for date in dates]
    pollution_frames = []

    with ProcessPoolExecutor(max_workers=min(32, CPU_COUNT)) as executor:
        futures = {executor.submit(read_pollution_day, args): args[0] for args in args_list}
        iterator = (
            tqdm(as_completed(futures), total=len(futures), desc="Pollution", unit="day")
            if TQDM_AVAILABLE
            else as_completed(futures)
        )
        for future in iterator:
            result = future.result()
            if result is not None:
                pollution_frames.append(result)

    if not pollution_frames:
        print("Pollution dataframe list is empty.")
        return pd.DataFrame()

    df_poll = pd.concat(pollution_frames).sort_index()
    df_poll.ffill(inplace=True)
    df_poll.bfill(inplace=True)
    df_poll = df_poll[(df_poll.index >= start_date) & (df_poll.index <= end_date)]
    print(f"Pollution data shape: {df_poll.shape}")
    return df_poll


def create_pollution_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat["year"] = df_feat.index.year
    df_feat["month"] = df_feat.index.month
    df_feat["day_of_year"] = df_feat.index.dayofyear
    df_feat["day_of_week"] = df_feat.index.dayofweek
    df_feat["is_weekend"] = (df_feat["day_of_week"] >= 5).astype(int)
    df_feat["season"] = df_feat["month"].apply(
        lambda x: 0 if x in [12, 1, 2] else 1 if x in [3, 4, 5] else 2 if x in [6, 7, 8] else 3
    )

    for pollutant in pollutants:
        if pollutant not in df_feat.columns:
            continue
        for lag in (1, 2, 3, 7, 14):
            df_feat[f"{pollutant}_lag{lag}"] = df_feat[pollutant].shift(lag)
        for window in (3, 7, 14, 30):
            df_feat[f"{pollutant}_ma{window}"] = (
                df_feat[pollutant].rolling(window, min_periods=1).mean()
            )
        df_feat[f"{pollutant}_diff1"] = df_feat[pollutant].diff()
        df_feat[f"{pollutant}_pct_change"] = df_feat[pollutant].pct_change().replace(
            [np.inf, -np.inf], np.nan
        )

    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_feat.dropna(inplace=True)
    return df_feat


class TensorTimeSeriesDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor, seq_length: int):
        self.features = features.contiguous()
        self.targets = targets.contiguous()
        if self.features.device.type != "cuda":
            self.features.share_memory_()
            self.targets.share_memory_()
        self.seq_length = seq_length

    def __len__(self):
        return self.features.shape[0] - self.seq_length

    def __getitem__(self, idx):
        return (
            self.features[idx : idx + self.seq_length],
            self.targets[idx + self.seq_length],
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PM25Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


def prepare_tensors(data: np.ndarray):
    return torch.from_numpy(np.ascontiguousarray(data)).float()


def standardize_data(df: pd.DataFrame, target_col: str):
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y).flatten()
    return (
        feature_cols,
        scaler_X,
        scaler_y,
        X_scaled,
        y_scaled,
    )


def split_tensors(X_scaled, y_scaled, seq_length):
    n_samples = len(X_scaled)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)

    if train_end <= seq_length or val_end - train_end <= seq_length or n_samples - val_end <= seq_length:
        raise ValueError(
            "Dataset too small for the chosen sequence length. Reduce SEQUENCE_LENGTH or provide more data."
        )

    def _to_tensor(slice_):
        return prepare_tensors(slice_)

    tensors = {
        "train": {
            "X": _to_tensor(X_scaled[:train_end]),
            "y": _to_tensor(y_scaled[:train_end]),
        },
        "val": {
            "X": _to_tensor(X_scaled[train_end:val_end]),
            "y": _to_tensor(y_scaled[train_end:val_end]),
        },
        "test": {
            "X": _to_tensor(X_scaled[val_end:]),
            "y": _to_tensor(y_scaled[val_end:]),
        },
    }
    return tensors


def create_dataloader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )


def train_model(model, train_loader, val_loader, epochs, patience, device):
    history = {"train_loss": [], "val_loss": []}
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = GradScaler(enabled=device.type == "cuda")
    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True).unsqueeze(-1)
            with autocast(enabled=device.type == "cuda"):
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.detach().item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True).unsqueeze(-1)
                with autocast(enabled=device.type == "cuda"):
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_model(model, data_loader, scaler_y, device):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.unsqueeze(-1)
            with autocast(enabled=device.type == "cuda"):
                outputs = model(X_batch).cpu()
            preds.append(outputs.numpy())
            actuals.append(y_batch.numpy())
    preds = np.concatenate(preds).reshape(-1, 1)
    actuals = np.concatenate(actuals).reshape(-1, 1)
    preds_denorm = scaler_y.inverse_transform(preds).flatten()
    actuals_denorm = scaler_y.inverse_transform(actuals).flatten()
    metrics = {
        "R2": r2_score(actuals_denorm, preds_denorm),
        "RMSE": np.sqrt(mean_squared_error(actuals_denorm, preds_denorm)),
        "MAE": mean_absolute_error(actuals_denorm, preds_denorm),
        "MAPE": np.mean(
            np.abs((actuals_denorm - preds_denorm) / (actuals_denorm + 1e-8))
        )
        * 100,
    }
    return metrics, preds_denorm, actuals_denorm


def plot_training_curves(history, path: Path):
    plt.figure(figsize=(6, 6))
    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transformer Training Curves (Pollution Only)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, format="tif")
    plt.close()


def plot_scatter(y_true, y_pred, path: Path):
    plt.figure(figsize=(6, 6))
    # 设置散点图：蓝色填充，黑色边缘
    plt.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors="black", linewidth=0.3, facecolors="blue")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    # 理想预测线：红色虚线，线宽为2
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2)
    plt.xlabel("Actual PM2.5 (μg/m³)")
    plt.ylabel("Predicted PM2.5 (μg/m³)")
    plt.title("Prediction Scatter (Test Set)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, format="tif")
    plt.close()


def plot_timeseries(y_true, y_pred, path: Path, window: int = 400, y_pred_base=None):
    # 时序图输出比例为18*5英寸
    plt.figure(figsize=(18, 5))
    start = max(0, len(y_true) - window)
    idx = np.arange(start, len(y_true))
    # 黑色实线为实际值，线宽为2
    plt.plot(idx, y_true[start:], label="Actual", linewidth=2, color="black", linestyle="-")
    # 蓝色虚线为基础模型的预测值，线宽为1.5
    if y_pred_base is not None:
        plt.plot(idx, y_pred_base[start:], label="Base Model Prediction", linewidth=1.5, color="blue", linestyle="--")
    # 绿色虚线为优化模型的预测值，线宽为1.5
    plt.plot(idx, y_pred[start:], label="Optimized Model Prediction", linewidth=1.5, color="green", linestyle="--")
    plt.xlabel("Sample Index")
    plt.ylabel("PM2.5 (μg/m³)")
    plt.title(f"PM2.5 Prediction vs Actual (last {len(idx)} samples)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, format="tif")
    plt.close()


def plot_error_histogram(y_true, y_pred, path: Path):
    errors = y_true - y_pred
    plt.figure(figsize=(6, 6))
    plt.hist(errors, bins=50, color="steelblue", alpha=0.8, edgecolor="black")
    plt.axvline(0, color="red", linestyle="--", linewidth=2)
    plt.title(f"Error Distribution | μ={errors.mean():.2f}, σ={errors.std():.2f}")
    plt.xlabel("Error (μg/m³)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, format="tif")
    plt.close()


def plot_residuals(y_true, y_pred, path: Path):
    errors = y_true - y_pred
    plt.figure(figsize=(6, 6))
    # 残差图与散点图设置保持一致：蓝色填充，黑色边缘
    plt.scatter(y_pred, errors, alpha=0.5, s=20, edgecolors="black", linewidth=0.3, facecolors="blue")
    # 理想预测线：红色虚线，线宽为2
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.xlabel("Predicted PM2.5")
    plt.ylabel("Residual (Actual - Pred)")
    plt.title("Residuals vs Predicted")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, format="tif")
    plt.close()


def save_predictions_csv(y_true, y_pred, path: Path):
    df_pred = pd.DataFrame(
        {"Actual_PM2.5": y_true, "Pred_PM2.5": y_pred, "Error": y_true - y_pred}
    )
    df_pred.to_csv(path, index=False, encoding="utf-8-sig")


def save_metrics_csv(metrics_dict, path: Path):
    rows = []
    for split, metrics in metrics_dict.items():
        row = {"Split": split}
        row.update(metrics)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def main():
    print("=" * 80)
    print("Beijing PM2.5 Transformer (Pollution Only)")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    df_pollution = read_all_pollution()
    if df_pollution.empty:
        raise RuntimeError("Pollution dataframe is empty. Check the file paths.")

    df_features = create_pollution_features(df_pollution)
    (
        feature_cols,
        scaler_X,
        scaler_y,
        X_scaled,
        y_scaled,
    ) = standardize_data(df_features, target_col="PM2.5")

    tensors = split_tensors(X_scaled, y_scaled, SEQUENCE_LENGTH)
    datasets = {
        split: TensorTimeSeriesDataset(
            tensors[split]["X"], tensors[split]["y"], SEQUENCE_LENGTH
        )
        for split in tensors
    }

    train_loader = create_dataloader(datasets["train"], BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(datasets["val"], BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(datasets["test"], BATCH_SIZE, shuffle=False)

    model = PM25Transformer(input_size=len(feature_cols)).to(device)
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled.")
        except Exception as exc:
            print(f"torch.compile failed: {exc}")

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=NUM_EPOCHS,
        patience=PATIENCE,
        device=device,
    )

    metrics = {}
    preds = {}
    splits = {"Train": train_loader, "Validation": val_loader, "Test": test_loader}
    for split_name, loader in splits.items():
        split_metrics, y_pred, y_true = evaluate_model(model, loader, scaler_y, device)
        metrics[split_name] = split_metrics
        preds[split_name] = {"y_true": y_true, "y_pred": y_pred}
        print(
            f"{split_name}: R2={split_metrics['R2']:.4f}, "
            f"RMSE={split_metrics['RMSE']:.2f}, MAE={split_metrics['MAE']:.2f}, "
            f"MAPE={split_metrics['MAPE']:.2f}%"
        )

    print("\nGenerating visualizations...")
    training_curve_path = output_dir / "training_curve_pollution.tif"
    plot_training_curves(history, training_curve_path)

    test_true = preds["Test"]["y_true"]
    test_pred = preds["Test"]["y_pred"]
    scatter_path = output_dir / "prediction_scatter_pollution.tif"
    timeseries_path = output_dir / "timeseries_pollution.tif"
    error_hist_path = output_dir / "error_histogram_pollution.tif"
    residuals_path = output_dir / "residuals_pollution.tif"

    plot_scatter(test_true, test_pred, scatter_path)
    plot_timeseries(test_true, test_pred, timeseries_path)
    plot_error_histogram(test_true, test_pred, error_hist_path)
    plot_residuals(test_true, test_pred, residuals_path)

    print("Saving evaluation artifacts...")
    metrics_csv_path = output_dir / "metrics_pollution.csv"
    save_metrics_csv(metrics, metrics_csv_path)

    for split_name, arrays in preds.items():
        pred_csv_path = output_dir / f"predictions_{split_name.lower()}_pollution.csv"
        save_predictions_csv(arrays["y_true"], arrays["y_pred"], pred_csv_path)

    torch.save(model.state_dict(), model_dir / "cnn_lstm.pth")
    with open(model_dir / "scaler_X_cnn_lstm.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open(model_dir / "scaler_y_cnn_lstm.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    print("\nArtifacts saved to:")
    print(f"  {output_dir.resolve()}")
    print(f"  {model_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()