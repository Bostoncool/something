import os
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
    "Bitstream Vera Sans",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

POLLUTION_ALL_PATH = Path(
    os.environ.get(
        "POLLUTION_ALL_PATH", "/root/autodl-tmp/Benchmark/all(AQI+PM2.5+PM10)"
    )
).expanduser()
POLLUTION_EXTRA_PATH = Path(
    os.environ.get(
        "POLLUTION_EXTRA_PATH", "/root/autodl-tmp/Benchmark/extra(SO2+NO2+CO+O3)"
    )
).expanduser()

OUTPUT_DIR = Path("./output")
MODEL_DIR = Path("./models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = datetime(2015, 1, 1)
END_DATE = datetime(2024, 12, 31)
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

SEQ_LENGTH = 45
PRED_LENGTH = 7
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

CPU_COUNT = multiprocessing.cpu_count()


def daterange(start_date: datetime, end_date: datetime):
    for n_days in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(days=n_days)


def build_file_path_dict(base_path: Path, prefix: str) -> Dict[str, Path]:
    file_dict: Dict[str, Path] = {}
    if not base_path.exists():
        return file_dict
    for root, _, files in os.walk(base_path):
        for file_name in files:
            if not file_name.startswith(prefix) or not file_name.endswith(".csv"):
                continue
            date_str = file_name[len(prefix) + 1 : -4]
            if len(date_str) == 8 and date_str.isdigit():
                file_dict[date_str] = Path(root) / file_name
    return file_dict


def read_pollution_day(
    args: Tuple[datetime, Dict[str, Path], Dict[str, Path], List[str]]
) -> pd.DataFrame:
    date_point, file_dict_all, file_dict_extra, pollutants = args
    date_str = date_point.strftime("%Y%m%d")
    all_file = file_dict_all.get(date_str)
    extra_file = file_dict_extra.get(date_str)

    if all_file is None or extra_file is None:
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
            df_poll.groupby(["date", "type"])["value"].mean().reset_index()
        )
        df_daily = df_daily.pivot(index="date", columns="type", values="value")
        df_daily.index = pd.to_datetime(
            df_daily.index, format="%Y%m%d", errors="coerce"
        )
        df_daily = df_daily[[col for col in pollutants if col in df_daily.columns]]
        return df_daily
    except Exception:
        return None


def load_pollution_dataset() -> pd.DataFrame:
    if not POLLUTION_ALL_PATH.exists() or not POLLUTION_EXTRA_PATH.exists():
        print("Error: pollution data directories are missing.")
        return pd.DataFrame()

    file_dict_all = build_file_path_dict(POLLUTION_ALL_PATH, "beijing_all")
    file_dict_extra = build_file_path_dict(POLLUTION_EXTRA_PATH, "beijing_extra")

    if not file_dict_all or not file_dict_extra:
        print("Error: pollution CSV index is empty.")
        return pd.DataFrame()

    dates = list(daterange(START_DATE, END_DATE))
    args_list = [
        (the_date, file_dict_all, file_dict_extra, POLLUTANTS) for the_date in dates
    ]

    pollution_frames: List[pd.DataFrame] = []
    worker_count = min(max(8, CPU_COUNT - 2), 32)

    print(f"Loading pollution data with {worker_count} workers ...")
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(read_pollution_day, arg): arg[0] for arg in args_list}
        iterator = (
            tqdm(as_completed(futures), total=len(futures), desc="Pollution days")
            if TQDM_AVAILABLE
            else as_completed(futures)
        )
        for future in iterator:
            result = future.result()
            if result is not None:
                pollution_frames.append(result)

    if not pollution_frames:
        print("Error: no pollution records were loaded.")
        return pd.DataFrame()

    df_poll = pd.concat(pollution_frames).sort_index()
    df_poll = df_poll[~df_poll.index.duplicated(keep="first")]
    df_poll = (
        df_poll.replace([np.inf, -np.inf], np.nan)
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )
    df_poll = df_poll.clip(lower=0)
    print(f"Pollution dataframe shape: {df_poll.shape}")
    return df_poll


def data_quality_report(df: pd.DataFrame) -> None:
    summary_path = OUTPUT_DIR / "pollution_summary_stats.csv"
    missing_path = OUTPUT_DIR / "pollution_missing_ratio.csv"
    quantile_path = OUTPUT_DIR / "pollution_quantiles.csv"

    df.describe().T.to_csv(summary_path, encoding="utf-8-sig")
    df.isna().mean().mul(100).to_frame("missing_pct").to_csv(
        missing_path, encoding="utf-8-sig"
    )
    df.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).T.to_csv(
        quantile_path, encoding="utf-8-sig"
    )
    print(f"Saved data quality reports to {OUTPUT_DIR}")


def clean_pollution_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    q1 = df_clean.quantile(0.25)
    q3 = df_clean.quantile(0.75)
    iqr = q3 - q1
    df_clean = df_clean.clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr, axis=1)
    df_clean = (
        df_clean.replace([np.inf, -np.inf], np.nan)
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )
    df_clean = df_clean.clip(lower=0)
    return df_clean


def engineer_pollution_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    lags = [1, 3, 7, 14]
    windows = [3, 7, 14, 30]

    for col in POLLUTANTS:
        if col not in features.columns:
            continue
        for lag in lags:
            features[f"{col}_lag{lag}"] = features[col].shift(lag)
        for window in windows:
            features[f"{col}_ma{window}"] = (
                features[col].rolling(window=window, min_periods=1).mean()
            )
        features[f"{col}_diff1"] = features[col].diff()

    if {"PM2.5", "PM10"}.issubset(features.columns):
        features["pm25_pm10_ratio"] = features["PM2.5"] / features["PM10"].clip(
            lower=1e-3
        )

    features["pollution_load"] = features.filter(POLLUTANTS).sum(axis=1)
    if all(col in features.columns for col in POLLUTANTS):
        features["dominant_pollutant"] = (
            features[POLLUTANTS].idxmax(axis=1).astype("category").cat.codes
        )
    else:
        features["dominant_pollutant"] = 0

    return features


def build_time_feature_matrix(index: pd.DatetimeIndex) -> pd.DataFrame:
    day_of_year = index.dayofyear.to_numpy()
    day_of_week = index.dayofweek.to_numpy()
    month = index.month.to_numpy()
    day_of_month = index.day.to_numpy()
    total_days = (index[-1] - index[0]).days + 1
    trend = ((index - index[0]).days / max(total_days, 1)).to_numpy()

    time_features = pd.DataFrame(
        {
            "sin_doy": np.sin(2 * np.pi * day_of_year / 365.25),
            "cos_doy": np.cos(2 * np.pi * day_of_year / 365.25),
            "sin_dow": np.sin(2 * np.pi * day_of_week / 7),
            "cos_dow": np.cos(2 * np.pi * day_of_week / 7),
            "sin_dom": np.sin(2 * np.pi * day_of_month / 31),
            "cos_dom": np.cos(2 * np.pi * day_of_month / 31),
            "sin_month": np.sin(2 * np.pi * month / 12),
            "cos_month": np.cos(2 * np.pi * month / 12),
            "is_weekend": (day_of_week >= 5).astype(float),
            "trend": trend,
        },
        index=index,
    )
    return time_features


def compute_batch_size(mem_gb: float) -> int:
    if mem_gb >= 64:
        return 1024
    if mem_gb >= 48:
        return 896
    if mem_gb >= 32:
        return 768
    if mem_gb >= 24:
        return 512
    if mem_gb >= 16:
        return 256
    if mem_gb >= 12:
        return 192
    if mem_gb >= 8:
        return 128
    return 64


def compute_worker_count() -> int:
    return min(max(8, CPU_COUNT // 2), 32)


def setup_device() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this script.")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    mem_gb = props.total_memory / (1024**3)
    batch_size = compute_batch_size(mem_gb)
    worker_count = compute_worker_count()
    use_amp = props.major >= 7
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(
        f"Detected GPU: {props.name}, {mem_gb:.2f} GB, capability {props.major}.{props.minor}"
    )
    print(
        f"Auto batch size: {batch_size}, DataLoader workers: {worker_count}, AMP: {use_amp}"
    )
    return {
        "device": device,
        "gpu_name": props.name,
        "mem_gb": mem_gb,
        "batch_size": batch_size,
        "num_workers": worker_count,
        "use_amp": use_amp,
    }


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        time_features: np.ndarray,
        seq_length: int,
        pred_length: int,
        start_idx: int,
        end_idx: int,
    ):
        if end_idx - start_idx < seq_length + pred_length:
            raise ValueError("Not enough samples for the requested window configuration.")
        self.X = X
        self.y = y
        self.time_features = time_features
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __len__(self):
        return self.end_idx - self.start_idx - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx: int):
        base_idx = self.start_idx + idx
        src_slice = slice(base_idx, base_idx + self.seq_length)
        tgt_slice = slice(base_idx + self.seq_length, base_idx + self.seq_length + self.pred_length)
        X_seq = self.X[src_slice]
        y_seq = self.y[tgt_slice]
        src_time = self.time_features[src_slice]
        tgt_time = self.time_features[tgt_slice]

        return (
            torch.from_numpy(X_seq),
            torch.from_numpy(y_seq),
            torch.from_numpy(src_time),
            torch.from_numpy(tgt_time),
        )


def create_data_loader(
    dataset: TimeSeriesDataset, batch_size: int, num_workers: int
) -> DataLoader:
    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(**loader_kwargs)


class EnhancedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, time_dim: int = None):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.time_embed = nn.Linear(time_dim, d_model) if time_dim else None

    def forward(self, x: torch.Tensor, time_features: torch.Tensor = None):
        x = x + self.pe[:, : x.size(1), :]
        if self.time_embed is not None and time_features is not None:
            x = x + self.time_embed(time_features)
        return x


class TimeSeriesTransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        input_dim: int,
        time_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pred_length: int = 7,
    ):
        super().__init__()
        self.pred_length = pred_length
        self.d_model = d_model

        self.encoder_input_projection = nn.Linear(input_dim, d_model)
        self.encoder_positional_encoding = EnhancedPositionalEncoding(
            d_model, max_len=5000, time_dim=time_dim
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.decoder_input_projection = nn.Linear(1, d_model)
        self.decoder_positional_encoding = EnhancedPositionalEncoding(
            d_model, max_len=5000, time_dim=time_dim
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(
        self,
        src: torch.Tensor,
        src_time_features: torch.Tensor,
        tgt_time_features: torch.Tensor,
    ) -> torch.Tensor:
        src_emb = self.encoder_input_projection(src)
        src_emb = self.encoder_positional_encoding(src_emb, src_time_features)
        memory = self.encoder(src_emb)

        tgt = torch.zeros(src.size(0), self.pred_length, 1, device=src.device)
        tgt_emb = self.decoder_input_projection(tgt)
        tgt_emb = self.decoder_positional_encoding(tgt_emb, tgt_time_features)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            self.pred_length, device=src.device
        )
        output = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        return self.output_layer(output).squeeze(-1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device_cfg: Dict[str, Any],
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 40,
    gradient_accumulation_steps: int = 1,
) -> Tuple[nn.Module, List[float], List[float], int]:
    device = device_cfg["device"]
    use_amp = device_cfg["use_amp"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    scaler = GradScaler("cuda", enabled=use_amp)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val = float("inf")
    patience_counter = 0
    best_state = None

    try:
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode="max-autotune")
            print("Enabled torch.compile for optimized kernels.")
    except Exception:
        pass

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        optimizer_step_called = False
        for batch_idx, (
            batch_X,
            batch_y,
            batch_src_time,
            batch_tgt_time,
        ) in enumerate(train_loader):
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_src_time = batch_src_time.to(device, non_blocking=True)
            batch_tgt_time = batch_tgt_time.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                preds = model(batch_X, batch_src_time, batch_tgt_time)
                loss = criterion(preds, batch_y) / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_called = True

            epoch_loss += loss.item() * gradient_accumulation_steps

        # Handle remaining gradients at the end of epoch
        if not optimizer_step_called and len(train_loader) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_called = True

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        if optimizer_step_called:
            scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y, batch_src_time, batch_tgt_time in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_src_time = batch_src_time.to(device, non_blocking=True)
                batch_tgt_time = batch_tgt_time.to(device, non_blocking=True)

                with autocast("cuda", enabled=use_amp):
                    preds = model(batch_X, batch_src_time, batch_tgt_time)
                    loss = criterion(preds, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:04d} | Train {epoch_loss:.6f} | Val {val_loss:.6f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses, epoch + 1


def predict_model(
    model: nn.Module, loader: DataLoader, device_cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    device = device_cfg["device"]
    use_amp = device_cfg["use_amp"]
    model.eval()
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    with torch.no_grad():
        for batch_X, batch_y, batch_src_time, batch_tgt_time in loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_src_time = batch_src_time.to(device, non_blocking=True)
            batch_tgt_time = batch_tgt_time.to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp):
                outputs = model(batch_X, batch_src_time, batch_tgt_time)
            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def inverse_transform(arr: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    shape = arr.shape
    flat = arr.reshape(-1, 1)
    inv = scaler.inverse_transform(flat).reshape(shape)
    return inv


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str
) -> Dict[str, Any]:
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    return {
        "Dataset": dataset_name,
        "R2": r2_score(y_true_flat, y_pred_flat),
        "RMSE": np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
        "MAE": mean_absolute_error(y_true_flat, y_pred_flat),
        "MAPE": np.mean(
            np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))
        )
        * 100,
    }


def evaluate_per_step(
    y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for step in range(y_true.shape[1]):
        t = y_true[:, step]
        p = y_pred[:, step]
        rows.append(
            {
                "Dataset": dataset_name,
                "Step": step + 1,
                "R2": r2_score(t, p),
                "RMSE": np.sqrt(mean_squared_error(t, p)),
                "MAE": mean_absolute_error(t, p),
                "MAPE": np.mean(np.abs((t - p) / (t + 1e-8))) * 100,
            }
        )
    return pd.DataFrame(rows)


def plot_correlation_heatmap(df: pd.DataFrame, path: Path) -> None:
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.1)
    plt.title("Pollution Feature Correlation")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_training_curves(
    train_losses: List[float], val_losses: List[float], path: Path
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train", linewidth=2)
    plt.plot(val_losses, label="Validation", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Transformer Training Curves (Pollution Only)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_scatter_multistep(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, path: Path
) -> None:
    steps = y_true.shape[1]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for i in range(steps):
        ax = axes[i // 4, i % 4]
        t = y_true[:, i]
        p = y_pred[:, i]
        ax.scatter(t, p, alpha=0.45, s=18)
        lim_min, lim_max = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1.2)
        r2 = r2_score(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        ax.set_title(f"{model_name} Day {i+1}\nR²={r2:.3f}, RMSE={rmse:.1f}")
        ax.set_xlabel("Actual PM2.5")
        ax.set_ylabel("Predicted PM2.5")
    axes[-1, -1].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_residuals_multistep(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, path: Path
) -> None:
    steps = y_true.shape[1]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for i in range(steps):
        ax = axes[i // 4, i % 4]
        t = y_true[:, i]
        p = y_pred[:, i]
        residual = p - t
        ax.scatter(p, residual, alpha=0.4, s=15)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{model_name} Day {i+1}")
        ax.set_xlabel("Predicted PM2.5")
        ax.set_ylabel("Residual")
    axes[-1, -1].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_timeseries_day1(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, path: Path, num_points: int = 300
) -> None:
    plt.figure(figsize=(16, 5))
    plt.plot(y_true[:num_points, 0], label="Actual", color="black", linewidth=1.5)
    plt.plot(y_pred[:num_points, 0], label="Predicted Day1", linestyle="--", linewidth=1.5)
    plt.title(f"{model_name} Day1 PM2.5 (First {num_points} Samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("PM2.5 Concentration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def permutation_importance(
    model: nn.Module,
    baseline_rmse: float,
    X_scaled: np.ndarray,
    y_scaled: np.ndarray,
    time_features: np.ndarray,
    feature_names: List[str],
    template_dataset: TimeSeriesDataset,
    device_cfg: Dict[str, Any],
    scaler_y: StandardScaler,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    segment = slice(template_dataset.start_idx, template_dataset.end_idx)
    importances: List[Dict[str, Any]] = []
    for feat_idx, feat_name in enumerate(feature_names):
        X_perm = X_scaled.copy()
        perm_slice = X_perm[segment, feat_idx]
        rng.shuffle(perm_slice)
        X_perm[segment, feat_idx] = perm_slice

        perm_dataset = TimeSeriesDataset(
            X_perm,
            y_scaled,
            time_features,
            SEQ_LENGTH,
            PRED_LENGTH,
            template_dataset.start_idx,
            template_dataset.end_idx,
        )
        perm_loader = create_data_loader(
            perm_dataset, device_cfg["batch_size"], device_cfg["num_workers"]
        )
        perm_pred_scaled, perm_true_scaled = predict_model(
            model, perm_loader, device_cfg
        )
        perm_pred = inverse_transform(perm_pred_scaled, scaler_y)
        perm_true = inverse_transform(perm_true_scaled, scaler_y)
        perm_rmse = np.sqrt(mean_squared_error(perm_true.flatten(), perm_pred.flatten()))
        importances.append(
            {
                "Feature": feat_name,
                "Delta_RMSE": perm_rmse - baseline_rmse,
            }
        )
    importance_df = pd.DataFrame(importances).sort_values(
        "Delta_RMSE", ascending=False
    )
    total = importance_df["Delta_RMSE"].clip(lower=0).sum() + 1e-8
    importance_df["Contribution_%"] = importance_df["Delta_RMSE"].clip(lower=0) / total * 100
    return importance_df


def main():
    device_cfg = setup_device()

    df_pollution = load_pollution_dataset()
    if df_pollution.empty:
        raise RuntimeError("Pollution dataset is empty. Aborting.")

    data_quality_report(df_pollution)
    df_clean = clean_pollution_dataframe(df_pollution)
    df_features = engineer_pollution_features(df_clean)
    time_features = build_time_feature_matrix(df_features.index)

    df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna()
    time_features = time_features.loc[df_features.index]

    plot_correlation_heatmap(
        df_features[POLLUTANTS], OUTPUT_DIR / "pollution_correlation.png"
    )

    target_col = "PM2.5"
    feature_cols = [col for col in df_features.columns if col != target_col]
    X = df_features[feature_cols].to_numpy(dtype=np.float32)
    y = df_features[target_col].to_numpy(dtype=np.float32)
    time_matrix = time_features.to_numpy(dtype=np.float32)

    scaler_X = MinMaxScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    total_samples = len(df_features)
    train_end = int(total_samples * TRAIN_RATIO)
    val_end = int(total_samples * (TRAIN_RATIO + VAL_RATIO))
    val_end = min(val_end, total_samples - PRED_LENGTH - 1)

    train_dataset = TimeSeriesDataset(
        X_scaled,
        y_scaled,
        time_matrix,
        SEQ_LENGTH,
        PRED_LENGTH,
        start_idx=0,
        end_idx=train_end,
    )
    val_dataset = TimeSeriesDataset(
        X_scaled,
        y_scaled,
        time_matrix,
        SEQ_LENGTH,
        PRED_LENGTH,
        start_idx=max(train_end - SEQ_LENGTH, 0),
        end_idx=val_end,
    )
    test_dataset = TimeSeriesDataset(
        X_scaled,
        y_scaled,
        time_matrix,
        SEQ_LENGTH,
        PRED_LENGTH,
        start_idx=max(val_end - SEQ_LENGTH, 0),
        end_idx=total_samples,
    )

    train_loader = create_data_loader(
        train_dataset, device_cfg["batch_size"], device_cfg["num_workers"]
    )
    val_loader = create_data_loader(
        val_dataset, device_cfg["batch_size"], device_cfg["num_workers"]
    )
    test_loader = create_data_loader(
        test_dataset, device_cfg["batch_size"], device_cfg["num_workers"]
    )

    gradient_accum = max(1, 1024 // device_cfg["batch_size"])
    model = TimeSeriesTransformerSeq2Seq(
        input_dim=len(feature_cols),
        time_dim=time_matrix.shape[1],
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        pred_length=PRED_LENGTH,
    ).to(device_cfg["device"])

    model, train_losses, val_losses, epochs_trained = train_model(
        model,
        train_loader,
        val_loader,
        device_cfg,
        epochs=250,
        lr=8e-4,
        patience=60,
        gradient_accumulation_steps=gradient_accum,
    )

    train_pred_scaled, train_true_scaled = predict_model(
        model, train_loader, device_cfg
    )
    val_pred_scaled, val_true_scaled = predict_model(model, val_loader, device_cfg)
    test_pred_scaled, test_true_scaled = predict_model(model, test_loader, device_cfg)

    train_pred = inverse_transform(train_pred_scaled, scaler_y)
    train_true = inverse_transform(train_true_scaled, scaler_y)
    val_pred = inverse_transform(val_pred_scaled, scaler_y)
    val_true = inverse_transform(val_true_scaled, scaler_y)
    test_pred = inverse_transform(test_pred_scaled, scaler_y)
    test_true = inverse_transform(test_true_scaled, scaler_y)

    metrics = [
        evaluate_predictions(train_true, train_pred, "Train"),
        evaluate_predictions(val_true, val_pred, "Validation"),
        evaluate_predictions(test_true, test_pred, "Test"),
    ]
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(
        OUTPUT_DIR / "pollution_transformer_metrics.csv", index=False, encoding="utf-8-sig"
    )
    per_step_df = evaluate_per_step(test_true, test_pred, "Test")
    per_step_df.to_csv(
        OUTPUT_DIR / "pollution_transformer_per_step.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_training_curves(
        train_losses, val_losses, OUTPUT_DIR / "pollution_training_curves.png"
    )
    plot_scatter_multistep(
        test_true, test_pred, "Pollution Transformer", OUTPUT_DIR / "pollution_scatter.png"
    )
    plot_residuals_multistep(
        test_true, test_pred, "Pollution Transformer", OUTPUT_DIR / "pollution_residuals.png"
    )
    plot_timeseries_day1(
        test_true, test_pred, "Pollution Transformer", OUTPUT_DIR / "pollution_timeseries_day1.png"
    )

    baseline_rmse = metrics_df.loc[metrics_df["Dataset"] == "Test", "RMSE"].iloc[0]
    feature_importance_df = permutation_importance(
        model,
        baseline_rmse,
        X_scaled,
        y_scaled,
        time_matrix,
        feature_cols,
        test_dataset,
        device_cfg,
        scaler_y,
    )
    feature_importance_df.to_csv(
        OUTPUT_DIR / "pollution_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    torch.save(
        model.state_dict(), MODEL_DIR / "pollution_transformer_weights.pth"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": feature_cols,
            "time_features": time_features.columns.tolist(),
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "seq_length": SEQ_LENGTH,
            "pred_length": PRED_LENGTH,
            "device_cfg": device_cfg,
            "epochs_trained": epochs_trained,
        },
        MODEL_DIR / "pollution_transformer_bundle.pth",
    )

    print("Training complete. All artifacts saved to output directories.")


if __name__ == "__main__":
    main()

