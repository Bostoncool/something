import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 自定义数据集类
class TifDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取TIF图像
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 定义CNN模型
class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# 数据加载和预处理函数
def load_data(data_dir):
    image_paths = []
    labels = []
    
    # 遍历数据目录获取图像路径和标签
    for filename in os.listdir(data_dir):
        if filename.endswith('.tif'):
            image_paths.append(os.path.join(data_dir, filename))
            
            # 提取文件名中的example部分
            example = filename.split('_')[3]
            
            # 根据长度判断标签类型
            if len(example) == 4:
                # 可能是年份
                label = float(example)
            elif len(example) == 6:
                # 可能是月份
                label = float(example)
            elif len(example) == 8:
                # 可能是日期
                label = float(example)
            else:
                raise ValueError(f"Unexpected format in filename: {filename}")
            
            labels.append(label)
    
    return image_paths, labels

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}')
    
    return train_losses, val_losses

def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 检查CUDA是否可用，如果不可用则抛出错误
    if not torch.cuda.is_available():
        raise RuntimeError("需要CUDA支持才能运行此程序！请确保您的GPU已正确配置。")
    
    # 设置设备
    device = torch.device('cuda')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 加载数据
    data_dir = r"G:\PM2.5(TIF)\2000"  # 请替换为实际的数据目录
    image_paths, labels = load_data(data_dir)
    
    # 划分训练集和验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    
    # 创建数据加载器
    train_dataset = TifDataset(train_paths, train_labels, transform=transform)
    val_dataset = TifDataset(val_paths, val_labels, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    model = CNNRegressor().to(device)
    
    # 定义损失函数和优化器
    
    criterion = nn.MSELoss()# 使用均方误差损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)# 使用Adam优化器
    
    # 训练模型
    num_epochs = 50
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    # 保存模型
    torch.save(model.state_dict(), 'cnn_regressor.pth')

if __name__ == '__main__':
    main() 