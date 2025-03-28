import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import glob
import os

# 数据预处理类
class PM25Dataset(Dataset):
    def __init__(self, data, seq_length=24):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (self.data[idx:idx+self.seq_length],
                self.data[idx+self.seq_length])

# LSTM模型定义
class PM25LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(PM25LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_and_preprocess_data(data_path):
    # 读取所有Excel文件
    all_files = glob.glob(os.path.join(data_path, "beijing_all_*.xlsx"))
    all_data = []
    
    for file in sorted(all_files):
        df = pd.read_excel(file)
        pm25_values = df['PM2.5'].values
        all_data.extend(pm25_values)
    
    # 数据标准化
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(np.array(all_data).reshape(-1, 1))
    
    return data_normalized, scaler

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载和预处理数据
    data_path = "."  # 数据文件所在目录
    data_normalized, scaler = load_and_preprocess_data(data_path)
    
    # 创建数据集和数据加载器
    dataset = PM25Dataset(data_normalized)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 初始化模型
    model = PM25LSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 100
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)
    
    # 保存模型
    torch.save(model.state_dict(), 'pm25_lstm_model.pth')
    
    print("模型训练完成并已保存!")

if __name__ == "__main__":
    main()
