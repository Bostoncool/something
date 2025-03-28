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
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def setup_cudnn():
    """设置cuDNN以优化性能"""
    cudnn.enabled = True
    cudnn.benchmark = True
    print("\ncuDNN配置信息:")
    print(f"cuDNN是否可用: {cudnn.is_available()}")
    print(f"cuDNN版本: {cudnn.version()}")
    print(f"benchmark模式: {cudnn.benchmark}")

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
        # print(f"图像模式: {image.mode}, 图像大小: {image.size}")  # 添加调试信息
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 定义改进的CNN模型
class CNNRegressor(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CNNRegressor, self).__init__()
        
        # 使用改进的卷积层架构
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.SiLU(inplace=True),  # 替换ReLU为SiLU(Swish)激活函数
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 增加一层
            nn.BatchNorm2d(256, momentum=0.9),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # 计算卷积后的特征图大小
        self.feature_size = 256 * 8 * 8  # 128x128 -> 8x8 经过4次下采样
        
        # 优化的全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512, momentum=0.9),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128, momentum=0.9),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),  # 逐渐减少dropout
            
            nn.Linear(128, 1)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        # 使用Kaiming初始化来优化SiLU激活函数的权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
    
    # 初始化混合精度训练的scaler
    scaler = GradScaler('cuda')
    
    # 确保模型在训练模式
    model.train()
    print("\n确认模型训练模式:", model.training)
    
    # 打印CUDA信息
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"当前设备: {torch.cuda.get_device_name(0)}")
        print(f"显存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"显存缓存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    torch.cuda.synchronize()
    
    # 学习率预热
    warmup_epochs = 5
    
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
        
        # 学习率预热
        if epoch < warmup_epochs:
            # 线性预热
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * warmup_factor
            print(f"学习率预热: {optimizer.param_groups[0]['lr']:.6f}")
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            
            # 将数据移动到GPU
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            
            # 使用混合精度训练
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
            
            # 使用scaler进行反向传播
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # 记录损失
            running_loss += loss.item()
            
            # 计算批次时间
            torch.cuda.synchronize()
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            if batch_idx % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Batch Time: {batch_time:.3f}s '
                      f'GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB')
            
            # 定期清理缓存
            if batch_idx % 30 == 0:
                torch.cuda.empty_cache()
        
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
        
        # 更新学习率（预热期后）
        if epoch >= warmup_epochs:
            scheduler.step(epoch_val_loss)
        
        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
            }, 'best_model.pth')
            print(f"已保存新的最佳模型，验证损失: {best_val_loss:.4f}")
        
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
    np.random.seed(42)
    
    # 设置cuDNN
    setup_cudnn()
    
    # 检查CUDA是否可用
    print("CUDA是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前CUDA设备:", torch.cuda.get_device_name(0))
        print("CUDA设备数量:", torch.cuda.device_count())
    
    if not torch.cuda.is_available():
        raise RuntimeError("需要CUDA支持才能运行此程序！请确保您的GPU已正确配置。")
    
    # 设置设备
    device = torch.device('cuda')
    print("使用的设备:", device)
    
    # 增强的数据预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 验证集使用更简单的转换
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 加载数据
    data_dir = r"F:\2000"
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
    val_dataset = TifDataset(val_paths, val_labels, transform=val_transform)
    
    # 修改数据加载器配置 - 使用更优的配置
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=7, 
        pin_memory=True,
        prefetch_factor=3, # 预取因子
        persistent_workers=True,
        drop_last=True  # 丢弃最后不完整的批次
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=7,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True
    )
    
    print(f"\n数据加载器配置:")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"批次大小: 16")
    
    try:
        # 测试数据加载器
        print("\n测试数据加载:")
        test_batch = next(iter(train_loader))
        print(f"测试批次形状: {test_batch[0].shape}")
        print(f"测试标签形状: {test_batch[1].shape}")
        
        # 初始化改进的模型
        model = CNNRegressor(dropout_rate=0.5).to(device)
        print("\n模型结构:")
        print(model)
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 使用改进的损失函数 - Huber损失结合MSE
        criterion = nn.HuberLoss(delta=0.1)
        
        # 使用AdamW优化器 - 更好的权重衰减处理
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=0.01,  # 降低初始学习率
            weight_decay=1e-3,  # 增加权重衰减
            betas=(0.9, 0.999),  # 使用AdamW的默认参数
            eps=1e-8  # 增加epsilon以防止除零
        )
        
        # 保存初始学习率以用于预热
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']
        
        # 改进的学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8, 
            min_lr=1e-6, verbose=True
        )
        
        # 训练模型
        num_epochs = 5  # 增加训练轮数
        print("\n开始训练...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device
        )
        
        # 绘制损失曲线
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('训练和验证损失曲线')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.show()
        
        # 加载最佳模型进行测试
        print("\n加载最佳模型进行测试...")
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint['epoch']
        best_loss = checkpoint['val_loss']
        print(f"最佳模型来自第 {best_epoch+1} 轮, 验证损失: {best_loss:.4f}")
        
        # 在验证集上进行最终评估
        model.eval()
        val_loss = 0.0
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                
                # 收集预测和真实值
                predictions.extend(outputs.squeeze().cpu().numpy())
                true_values.extend(labels.cpu().numpy())
        
        final_val_loss = val_loss / len(val_loader)
        print(f"最终验证损失: {final_val_loss:.4f}")
        
        # 绘制预测vs真实值散点图
        plt.figure(figsize=(10, 8))
        plt.scatter(true_values, predictions, alpha=0.5)
        plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测值vs真实值')
        plt.grid(True)
        plt.savefig('predictions.png')
        plt.show()
        
    except Exception as e:
        print(f"\n训练过程中出现错误:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        raise

if __name__ == '__main__':
    main() 