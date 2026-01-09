import os
import warnings
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

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
output_dir.mkdir(parents=True, exist_ok=True)

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


def save_csv(df: pd.DataFrame, path: Path, index: bool = False):
    """统一 CSV 输出（UTF-8-SIG），便于后续脚本读取与绘图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"✓ Saved CSV: {path}")


def build_history_csv(history: dict) -> pd.DataFrame:
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    return pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": history.get("train_loss", []),
            "val_loss": history.get("val_loss", []),
        }
    )


def build_metrics_csv(metrics_by_split: dict, model_name: str) -> pd.DataFrame:
    rows = []
    for split, m in metrics_by_split.items():
        row = {"Model": model_name, "Split": split}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


def build_predictions_csv(
    times: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split: str,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    df["error"] = df["y_true"] - df["y_pred"]
    df["model"] = model_name
    df["split"] = split
    return df.sort_values("time").reset_index(drop=True)


def build_scatter_csv(times: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(times),
            "Actual_PM25": y_true,
            "Predicted_PM25": y_pred,
        }
    ).sort_values("Date").reset_index(drop=True)


def build_residuals_csv(times: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    residuals = y_true - y_pred
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(times),
            "Actual_PM25": y_true,
            "Predicted_PM25": y_pred,
            "Residual": residuals,
        }
    ).sort_values("Date").reset_index(drop=True)


def build_error_distribution_csv(times: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    errors = y_true - y_pred
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(times),
            "Actual_PM25": y_true,
            "Predicted_PM25": y_pred,
            "Error": errors,
        }
    ).sort_values("Date").reset_index(drop=True)


def build_ts_lastyear_sampled_csv(
    times: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_days: int = 365,
    step: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    输出与 RF-CSV.py 风格一致的“时序图数据”CSV：
    - plot_ts_lastyear_sampled__actual.csv
    - plot_ts_lastyear_sampled__<model>.csv
    这里返回 actual/common、actual、pred 三份，便于按模型拆分落盘。
    """
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).sort_values("time").reset_index(drop=True)

    plot_range = min(max_days, len(df))
    df_subset = df.iloc[-plot_range:].copy()
    df_sampled = df_subset.iloc[::step].copy()

    x_axis = np.arange(len(df_sampled))
    ts_common = pd.DataFrame(
        {
            "x_axis": x_axis,
            "time": df_sampled["time"].values,
            "y_true": df_sampled["y_true"].values,
        }
    )
    ts_pred = ts_common.assign(y_pred=df_sampled["y_pred"].values)
    return ts_common, ts_common.copy(), ts_pred


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

    print("\nSaving CSV artifacts (no plotting, no model/scaler export)...")

    model_name = "transformer"

    # 1) Training curve data
    save_csv(
        build_history_csv(history),
        output_dir / f"plot_training_curve__{model_name}.csv",
    )

    # 2) Metrics（按模型/按数据集拆分 + 汇总表）
    metrics_df = build_metrics_csv(metrics, model_name=model_name)
    save_csv(metrics_df, output_dir / "metrics__all_models_train_val_test.csv")
    save_csv(metrics_df, output_dir / f"metrics__{model_name}__train_val_test.csv")
    for split_name in metrics_df["Split"].unique():
        part = metrics_df[metrics_df["Split"] == split_name]
        save_csv(part, output_dir / f"metrics__{model_name}__{split_name.lower()}.csv")

    # 3) Build timestamps aligned with seq_length and split boundaries
    all_times = df_features.index.to_numpy()
    n_samples = len(all_times)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)
    split_time_map = {
        "Train": all_times[SEQUENCE_LENGTH:train_end],
        "Validation": all_times[train_end + SEQUENCE_LENGTH:val_end],
        "Test": all_times[val_end + SEQUENCE_LENGTH:],
    }

    # 4) Predictions（按 split 拆分）
    for split_name, arrays in preds.items():
        times = split_time_map.get(split_name)
        if times is None or len(times) != len(arrays["y_true"]):
            # 兜底：若对齐异常，用样本序号代替时间（仍输出 CSV，避免中断）
            times = pd.date_range(start=start_date, periods=len(arrays["y_true"]), freq="D").to_numpy()

        pred_df = build_predictions_csv(
            times=times,
            y_true=arrays["y_true"],
            y_pred=arrays["y_pred"],
            model_name=model_name,
            split=split_name,
        )
        save_csv(pred_df, output_dir / f"predictions__{model_name}__{split_name.lower()}.csv")

    # 5) “作图数据 CSV”：仅对 Test 输出（更贴近常用评估）
    test_times = split_time_map["Test"]
    test_true = preds["Test"]["y_true"]
    test_pred = preds["Test"]["y_pred"]
    if len(test_times) != len(test_true):
        test_times = pd.date_range(start=start_date, periods=len(test_true), freq="D").to_numpy()

    save_csv(
        build_scatter_csv(test_times, test_true, test_pred),
        output_dir / f"plot_scatter__{model_name}.csv",
    )
    save_csv(
        build_residuals_csv(test_times, test_true, test_pred),
        output_dir / f"plot_residuals__{model_name}.csv",
    )
    save_csv(
        build_error_distribution_csv(test_times, test_true, test_pred),
        output_dir / f"plot_error_distribution__{model_name}.csv",
    )

    ts_common, ts_actual, ts_pred = build_ts_lastyear_sampled_csv(
        test_times, test_true, test_pred, max_days=365, step=4
    )
    # RF 风格：actual 与 model 分别落盘
    save_csv(ts_common, output_dir / "plot_ts_lastyear_sampled__actual.csv")
    save_csv(ts_pred, output_dir / f"plot_ts_lastyear_sampled__{model_name}.csv")

    print("\nArtifacts saved to:")
    print(f"  {output_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()