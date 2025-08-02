import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import re
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def read_all_csv_files(root_folder):
    """
    读取指定文件夹及其子文件夹中的所有CSV文件
    """
    all_data = []
    
    print(f"开始读取文件夹：{root_folder}")
    
    # 遍历所有子文件夹
    for folder_name, subfolders, filenames in os.walk(root_folder):
        # 获取当前文件夹下所有CSV文件
        csv_files = [f for f in filenames if f.endswith('.csv')]
        
        for file in csv_files:
            file_path = os.path.join(folder_name, file)
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 检查是否为空
                if df.empty:
                    print(f"警告：文件为空 - {file_path}")
                    continue
                
                # 确保有日期列
                if 'date' not in df.columns:
                    # 尝试从文件名中提取日期信息
                    match = re.search(r'(\d{8})', file)
                    if match:
                        df['date'] = match.group(1)
                    else:
                        print(f"警告：无法获取日期信息 - {file_path}")
                        continue
                
                # 将所有空值填充为0
                df = df.fillna(0)
                
                # 添加文件源信息
                df['source_file'] = os.path.basename(file_path)
                
                # 将数据添加到列表
                all_data.append(df)
                print(f"成功读取：{file_path} - {len(df)} 条记录")
                
            except Exception as e:
                print(f"错误：读取文件时出错 - {file_path}")
                print(f"错误信息：{str(e)}")
    
    # 合并所有数据框
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # 确保所有空值都被填充为0
        combined_df = combined_df.fillna(0)
        print(f"数据合并完成，总共 {len(combined_df)} 条记录")
        return combined_df
    else:
        print("没有找到有效的数据文件")
        return pd.DataFrame()

# 1. 数据读取与整合
# 修改为批量读取模式
DATA_FOLDER = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\all'  # 请修改为实际的CSV文件所在文件夹路径
if not os.path.exists(DATA_FOLDER):
    print(f"错误：指定的文件夹不存在 - {DATA_FOLDER}")
    print("请修改 DATA_FOLDER 变量为正确的CSV文件所在路径")
    exit()

raw_df = read_all_csv_files(DATA_FOLDER)

if raw_df.empty:
    print("错误：未读取到有效数据，程序退出")
    exit()

# 2. 数据预处理
def preprocess_pm25(df):
    """
    仅保留type为PM2.5的整点数据，按站点取平均作为城市均值
    增强版本：支持批量数据和多格式处理
    """
    try:
        print("开始预处理PM2.5数据...")
        
        # 检查必要的列是否存在
        required_cols = ['type', 'date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"错误：缺少必要的列 - {missing_cols}")
            return None
        
        # 过滤PM2.5整点数据
        pm25_df = df[df['type'] == 'PM2.5'].copy()
        print(f"PM2.5数据筛选后记录数: {len(pm25_df)}")
        
        if len(pm25_df) == 0:
            print("警告：未找到PM2.5数据")
            return None
        
        # 检查是否有hour列，如果没有则尝试从其他列推断
        if 'hour' not in pm25_df.columns:
            # 尝试从其他列推断小时信息
            numeric_cols = pm25_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # 假设第一列是小时信息
                hour_col = numeric_cols[0]
                pm25_df['hour'] = pm25_df[hour_col].astype(int)
                print(f"从列 {hour_col} 推断小时信息")
            else:
                print("警告：无法确定小时信息，使用默认值0")
                pm25_df['hour'] = 0
        
        # 过滤出整点（hour=0~23），去掉24h滑动平均
        pm25_df = pm25_df[pm25_df['hour'] <= 23]
        print(f"整点数据筛选后记录数: {len(pm25_df)}")

        # 按日期时间排序
        pm25_df['datetime'] = pd.to_datetime(pm25_df['date'].astype(str), format='%Y%m%d') + pd.to_timedelta(pm25_df['hour'], unit='h')
        pm25_df = pm25_df.sort_values('datetime')

        # 提取所有站点列（忽略非数据列）
        exclude_cols = ['date', 'hour', 'type', 'datetime', 'source_file']
        station_cols = [c for c in pm25_df.columns if c not in exclude_cols]
        
        if len(station_cols) == 0:
            print("错误：未找到站点数据列")
            return None
        
        print(f"找到 {len(station_cols)} 个站点列")
        
        # 确保站点列为数值类型
        pm25_df[station_cols] = pm25_df[station_cols].apply(pd.to_numeric, errors='coerce')

        # 计算城市平均PM2.5
        pm25_df['PM25_avg'] = pm25_df[station_cols].mean(axis=1, skipna=True)
        
        # 移除包含NaN的行
        pm25_df = pm25_df[['datetime', 'PM25_avg']].dropna()
        
        print(f"最终PM2.5数据记录数: {len(pm25_df)}")
        
        return pm25_df[['PM25_avg']].values.astype(float)
        
    except Exception as e:
        print(f"数据预处理时出错：{str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

pm25_series = preprocess_pm25(raw_df)

if pm25_series is None:
    print("错误：数据预处理失败，程序退出")
    exit()

print(f"数据样本数: {len(pm25_series)}")

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

print(f"训练集大小: {len(train_set)}")
print(f"测试集大小: {len(test_set)}")

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
print(f"使用设备: {device}")

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

print("开始训练LSTM模型...")
train_losses = []
for epoch in tqdm(range(EPOCHS), desc="Training"):
    loss = train(model, train_loader)
    train_losses.append(loss)
    
    # 每10个epoch打印一次训练损失
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.6f}")

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

# 8. 保存模型和结果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = f"lstm_pm25_model_{timestamp}.pth"
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存至: {model_save_path}")

# 保存预测结果
results_df = pd.DataFrame({
    'True_PM25': targets.flatten(),
    'Predicted_PM25': preds.flatten()
})
results_save_path = f"lstm_pm25_results_{timestamp}.csv"
results_df.to_csv(results_save_path, index=False, encoding='utf-8-sig')
print(f"预测结果已保存至: {results_save_path}")

print("LSTM PM2.5预测完成！")