import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

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
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
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
    
    # 将标签转换为numpy数组并进行标准化
    labels = np.array(labels)
    labels = (labels - np.mean(labels)) / np.std(labels)
    
    return image_paths, labels.tolist()

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 确保模型在训练模式
    model.train()
    print("\n确认模型训练模式:", model.training)
    
    # 打印CUDA信息
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"当前设备: {torch.cuda.get_device_name(0)}")
        print(f"显存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"显存缓存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    torch.cuda.synchronize()  # 确保GPU同步
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        batch_times = []
        
        # 预热GPU
        if epoch == 0:
            print("预热GPU...")
            warmup_tensor = torch.randn(32, 1, 128, 128).to(device)
            with torch.no_grad():
                for _ in range(10):
                    model(warmup_tensor)
            torch.cuda.synchronize()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            
            # 将数据移动到GPU
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            
            # 反向传播
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 记录损失
            running_loss += loss.item()
            
            # 计算批次时间
            torch.cuda.synchronize()  # 确保GPU操作完成
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            if batch_idx % 5 == 0:  # 更频繁地打印进度
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Batch Time: {batch_time:.3f}s '
                      f'GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB')
        
        # 计算训练集平均损失
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
        
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        # 更新学习率
        scheduler.step(epoch_val_loss)
        
        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # 打印epoch统计信息
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        
        print(f'\nEpoch {epoch+1} 统计:')
        print(f'训练损失: {epoch_train_loss:.4f}')
        print(f'验证损失: {epoch_val_loss:.4f}')
        print(f'总耗时: {epoch_time:.2f}s')
        print(f'平均批次时间: {avg_batch_time:.3f}s')
        print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'GPU显存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB')
        print(f'GPU显存缓存: {torch.cuda.memory_reserved(0) / 1024**2:.2f}MB')
        
        # 清理GPU缓存
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
    
    return train_losses, val_losses

def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 检查CUDA是否可用，如果不可用则抛出错误
    print("CUDA是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前CUDA设备:", torch.cuda.get_device_name(0))
        print("CUDA设备数量:", torch.cuda.device_count())
    
    if not torch.cuda.is_available():
        raise RuntimeError("需要CUDA支持才能运行此程序！请确保您的GPU已正确配置。")
    
    # 设置设备
    device = torch.device('cuda')
    print("使用的设备:", device)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 加载数据
    data_dir = r"G:\PM2.5(TIF)\2000"
    print("正在加载数据从:", data_dir)
    image_paths, labels = load_data(data_dir)
    print("找到的图像数量:", len(image_paths))
    print("标签值范围:", min(labels), "到", max(labels))
    
    if len(image_paths) == 0:
        raise RuntimeError("没有找到任何.tif文件！请检查数据目录路径是否正确。")
    
    # 划分训练集和验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    print("训练集大小:", len(train_paths))
    print("验证集大小:", len(val_paths))
    
    # 创建数据加载器
    train_dataset = TifDataset(train_paths, train_labels, transform=transform)
    val_dataset = TifDataset(val_paths, val_labels, transform=transform)
    
    # 修改数据加载器配置
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # 减小批次大小
        shuffle=True,
        num_workers=0,  # 暂时不使用多进程
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,  # 减小批次大小
        shuffle=False,
        num_workers=0,  # 暂时不使用多进程
        pin_memory=True
    )
    
    print(f"\n数据加载器配置:")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"批次大小: 32")
    
    try:
        # 测试数据加载器
        print("\n测试数据加载:")
        test_batch = next(iter(train_loader))
        print(f"测试批次形状: {test_batch[0].shape}")
        print(f"测试标签形状: {test_batch[1].shape}")
        
        # 初始化模型
        model = CNNRegressor().to(device)
        print("\n模型结构:")
        print(model)
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 训练模型
        num_epochs = 50
        print("\n开始训练...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device
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
        
    except Exception as e:
        print(f"\n训练过程中出现错误:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        raise

if __name__ == '__main__':
    main() 