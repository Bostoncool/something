# -*- coding: utf-8 -*-
"""
LSTM预测北京PM2.5（单文件示例）
运行环境：Python 3.8+，PyTorch 1.12+
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. 数据读取与整合
DATA_PATH = "beijing_all_20150101.csv"  # 请放在当前目录或修改为实际路径
raw_df = pd.read_csv(DATA_PATH)

# 2. 数据预处理
def preprocess_pm25(df):
    """
    仅保留type为PM2.5的整点数据，按站点取平均作为城市均值
    """
    # 过滤PM2.5整点数据
    pm25_df = df[df['type'] == 'PM2.5'].copy()
    # 过滤出整点（hour=0~23），去掉24h滑动平均
    pm25_df = pm25_df[pm25_df['hour'] <= 23]

    # 按日期时间排序
    pm25_df['datetime'] = pd.to_datetime(pm25_df['date'].astype(str), format='%Y%m%d') + pd.to_timedelta(pm25_df['hour'], unit='h')
    pm25_df = pm25_df.sort_values('datetime')

    # 提取所有站点列（忽略空值列）
    station_cols = [c for c in pm25_df.columns if c not in ['date', 'hour', 'type', 'datetime']]
    pm25_df[station_cols] = pm25_df[station_cols].apply(pd.to_numeric, errors='coerce')

    # 计算城市平均PM2.5
    pm25_df['PM25_avg'] = pm25_df[station_cols].mean(axis=1, skipna=True)
    pm25_df = pm25_df[['datetime', 'PM25_avg']].dropna()

    return pm25_df[['PM25_avg']].values.astype(float)

pm25_series = preprocess_pm25(raw_df)
print("数据样本数:", len(pm25_series))

# 3. 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
pm25_scaled = scaler.fit_transform(pm25_series)

# 4. 构造数据集（滑动窗口）
class PM25Dataset(Dataset):
    def __init__(self, data, seq_len=24):
        self.seq_len = seq_len
        self.X, self.y = [], []
        for i in range(len(data) - seq_len):
            self.X.append(data[i:i+seq_len])
            self.y.append(data[i+seq_len])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

SEQ_LEN = 24  # 用过去24小时预测下一小时
dataset = PM25Dataset(pm25_scaled, seq_len=SEQ_LEN)

# 划分训练集/测试集（8:2）
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

BATCH_SIZE = 32
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# 5. 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # [batch, seq_len, hidden]
        out = self.fc(out[:, -1, :])  # 取最后一步
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel().to(device)

# 6. 训练与评估
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 100

def train(model, loader):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item() * x.size(0)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    targets = scaler.inverse_transform(np.array(targets).reshape(-1, 1))
    mse = np.mean((preds - targets) ** 2)
    return mse, preds, targets

train_losses = []
for epoch in tqdm(range(EPOCHS), desc="Training"):
    loss = train(model, train_loader)
    train_losses.append(loss)

test_mse, preds, targets = evaluate(model, test_loader)
print(f"测试MSE: {test_mse:.4f}")

# 7. 可视化
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train MSE')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(targets, label='True PM2.5')
plt.plot(preds, label='Predicted PM2.5')
plt.title("PM2.5 Prediction vs True")
plt.xlabel("Time")
plt.ylabel("PM2.5")
plt.legend()
plt.tight_layout()
plt.show()