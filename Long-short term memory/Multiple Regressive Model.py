import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class PM25Dataset(Dataset):
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

class MultipleRegressiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(MultipleRegressiveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, output_size)
        )
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(data_path, seq_length=24, train_split=0.8):
    """准备训练数据"""
    # 读取数据
    df = pd.read_csv(data_path)
    
    # 打印列名以进行检查
    print("数据集中的列名：")
    print(df.columns.tolist())
    
    # 打印数据集前几行以检查列名
    print("数据集前几行:")
    print(df.head())
    
    # 选择特征 - 根据实际数据集的列名调整
    features = [col.strip() for col in ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']]
    
    # 验证所有特征是否存在
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"以下特征在数据集中不存在: {missing_features}")
    
    # 处理缺失值
    df[features] = df[features].fillna(method='ffill')
    
    # 数据标准化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # 准备特征和目标
    X = scaled_data
    y = scaled_data[:, 0:1]  # PM2.5作为目标
    
    # 划分训练集和测试集
    train_size = int(len(X) * train_split)
    train_X = X[:train_size]
    train_y = y[:train_size]
    test_X = X[train_size:]
    test_y = y[train_size:]
    
    # 创建数据集
    train_dataset = PM25Dataset(train_X, train_y, seq_length)
    test_dataset = PM25Dataset(test_X, test_y, seq_length)
    
    return train_dataset, test_dataset, scaler

def train_model(model, train_loader, test_loader, criterion, optimizer, 
                num_epochs, device, early_stopping_patience=10):
    """训练模型"""
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 评估阶段
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
        
        # 计算平均损失
        avg_train_loss = train_loss/len(train_loader)
        avg_test_loss = test_loss/len(test_loader)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        # 早停检查
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f'早停: epoch {epoch+1}')
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Test Loss: {avg_test_loss:.4f}')
    
    return train_losses, test_losses

def plot_results(train_losses, test_losses):
    """绘制训练结果"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和测试损失曲线')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 超参数设置
    seq_length = 24
    hidden_size = 128
    num_layers = 2
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 修改这里的数据路径
    try:
        data_path = r'C:\Users\IU\Desktop\Code\something\Long-short term memory\PRSA_Data\PRSA_data_2010.1.1-2014.12.31.csv'  # 使用相对路径
        print(f'尝试读取数据文件: {data_path}')
        train_dataset, test_dataset, scaler = prepare_data(
            data_path,
            seq_length=seq_length
        )
    except FileNotFoundError:
        print("错误：找不到数据文件，请确保数据文件路径正确")
        print("请检查以下几点：")
        print("1. 数据文件是否存在于正确的位置")
        print("2. 文件名是否正确")
        print("3. 文件路径是否���确")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 初始化模型
    input_size = 7  # 特征数量
    output_size = 1  # 预测PM2.5
    model = MultipleRegressiveLSTM(
        input_size, hidden_size, num_layers, output_size
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    train_losses, test_losses = train_model(
        model, train_loader, test_loader,
        criterion, optimizer, num_epochs, device
    )
    
    # 绘制结果
    plot_results(train_losses, test_losses)

if __name__ == '__main__':
    main()
