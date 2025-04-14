import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重复
torch.manual_seed(42)
np.random.seed(42)

# 自定义数据集类
class AirQualityDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.features) - self.seq_length
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx:idx+self.seq_length]),
            torch.FloatTensor(self.targets[idx+self.seq_length:idx+self.seq_length+1])
        )

# LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(data, seq_length, train_split=0.8):
    # 数据标准化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 准备特征和目标
    features = scaled_data[:, :]
    targets = scaled_data[:, 1:2]  # PM2.5作为目标
    
    # 划分训练集和测试集
    train_size = int(len(features) * train_split)
    train_features = features[:train_size]
    train_targets = targets[:train_size]
    test_features = features[train_size:]
    test_targets = targets[train_size:]
    
    # 创建数据加载器
    train_dataset = AirQualityDataset(train_features, train_targets, seq_length)
    test_dataset = AirQualityDataset(test_features, test_targets, seq_length)
    
    return train_dataset, test_dataset, scaler

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 评估模式
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                test_loss += loss.item()
        
        train_losses.append(train_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Test Loss: {test_loss/len(test_loader):.4f}')
    
    return train_losses, test_losses

def main():
    # 设置超参数
    seq_length = 24  # 使用前24小时的数据预测
    hidden_size = 64
    num_layers = 2
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001
    
    # 检查是否可以使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据（这里需要根据实际数据集格式进行修改）
    try:
        data = pd.read_csv('air_quality_data.csv')
        # 选择相关特征
        features = ['PM2.5', 'dew_point', 'temperature', 'wind_direction', 
                   'wind_speed', 'snow', 'rain', 'year', 'month', 'day']
        data = data[features].values
    except:
        print("请确保数据文件存在且格式正确")
        return
    
    # 准备数据
    train_dataset, test_dataset, scaler = prepare_data(data, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 初始化模型
    input_size = data.shape[1]  # 特征数量
    output_size = 1  # 预测PM2.5一个值
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    train_losses, test_losses = train_model(model, train_loader, test_loader, 
                                          criterion, optimizer, num_epochs, device)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
