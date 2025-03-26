import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import seaborn as sns
from matplotlib.animation import FuncAnimation
from cartopy import crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 自定义数据集类
class PM25Dataset(Dataset):
    def __init__(self, data, seq_length, pred_length, transform=None):
        """
        初始化PM2.5数据集
        
        Args:
            data: 包含时间和空间维度的PM2.5数据，形状为 [时间, 纬度, 经度]
            seq_length: 用于预测的历史序列长度
            pred_length: 预测的未来时间步长度
            transform: 数据转换/标准化函数
        """
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.transform = transform
        self.len = len(data) - seq_length - pred_length + 1
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # 提取输入序列
        x = self.data[idx:idx+self.seq_length]
        # 提取目标序列
        y = self.data[idx+self.seq_length:idx+self.seq_length+self.pred_length]
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
            
        # 返回张量，x形状: [seq_length, height, width]
        # y形状: [pred_length, height, width]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# 定义CNN-LSTM模型
class CNNLSTM(nn.Module):
    def __init__(self, seq_length, input_channels, hidden_dim, num_layers, output_length, 
                 kernel_size=3, padding=1):
        super(CNNLSTM, self).__init__()
        
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # CNN部分 - 提取空间特征
        self.conv1 = nn.Conv2d(1, 16, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # 计算CNN后的特征图大小（假设输入是[batch, channels, height, width]）
        # 经过3次最大池化，尺寸会变为原来的1/8
        self.cnn_flat_dim = self._get_conv_output_size(input_channels)
        
        # LSTM部分 - 提取时间特征
        self.lstm = nn.LSTM(
            input_size=self.cnn_flat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 全连接层 - 生成预测
        self.fc = nn.Linear(hidden_dim, self.cnn_flat_dim)
        
        # 反卷积部分 - 将特征还原为原始尺寸
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(16, output_length, kernel_size=2, stride=2)
        
    def _get_conv_output_size(self, shape):
        # 帮助函数：计算CNN输出的平坦尺寸
        bs = 1
        x = torch.rand(bs, 1, *shape)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        return x.numel() // bs
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据，形状为 [batch_size, seq_length, height, width]
        
        Returns:
            输出预测，形状为 [batch_size, pred_length, height, width]
        """
        batch_size, seq_len, height, width = x.size()
        
        # CNN处理每个时间步
        cnn_output = []
        for t in range(seq_len):
            # [batch_size, 1, height, width]
            xt = x[:, t, :, :].unsqueeze(1)
            
            # CNN前向传播
            xt = self.relu(self.conv1(xt))
            xt = self.pool(xt)
            xt = self.relu(self.conv2(xt))
            xt = self.pool(xt)
            xt = self.relu(self.conv3(xt))
            xt = self.pool(xt)
            
            # 展平CNN输出
            xt = xt.view(batch_size, -1)
            cnn_output.append(xt)
        
        # 将所有时间步的CNN输出堆叠
        # [batch_size, seq_length, cnn_flat_dim]
        cnn_output = torch.stack(cnn_output, dim=1)
        
        # LSTM处理时间序列
        lstm_out, _ = self.lstm(cnn_output)
        # 只使用最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        
        # 全连接层
        fc_out = self.fc(lstm_out)
        
        # 重塑为卷积特征图形状
        # 假设经过3次池化，特征图尺寸是原始的1/8
        small_h, small_w = height // 8, width // 8
        fc_out = fc_out.view(batch_size, 64, small_h, small_w)
        
        # 反卷积还原尺寸
        out = self.relu(self.deconv1(fc_out))
        out = self.relu(self.deconv2(out))
        out = self.deconv3(out)
        
        return out

# 数据加载和预处理函数
def load_and_preprocess_data(data_path, start_year=2000, end_year=2023):
    """
    加载并预处理PM2.5数据
    
    Args:
        data_path: 数据文件路径
        start_year: 起始年份
        end_year: 结束年份
    
    Returns:
        预处理后的数据，形状为 [时间, 纬度, 经度]
    """
    # 在实际应用中，这里需要根据您的数据格式进行调整
    # 这里假设数据是按年份存储的CSV文件
    
    print(f"加载{start_year}-{end_year}年的PM2.5数据...")
    
    # 示例代码，请根据实际数据格式调整
    data_frames = []
    for year in range(start_year, end_year + 1):
        try:
            file_path = os.path.join(data_path, f"pm25_{year}.csv")
            df = pd.read_csv(file_path)
            # 假设数据包含日期、经度、纬度和PM2.5值
            df['date'] = pd.to_datetime(df['date'])
            data_frames.append(df)
            print(f"成功加载{year}年数据")
        except Exception as e:
            print(f"无法加载{year}年数据: {e}")
    
    if not data_frames:
        raise ValueError("未能加载任何数据")
    
    # 合并所有年份的数据
    all_data = pd.concat(data_frames)
    
    # 假设我们需要将数据重塑为 [时间, 纬度, 经度] 的3D网格
    # 这部分需要根据实际数据结构调整
    print("将数据重组为三维网格...")
    
    # 创建时间索引
    time_index = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='D')
    
    # 获取唯一的纬度和经度值
    lats = sorted(all_data['latitude'].unique())
    lons = sorted(all_data['longitude'].unique())
    
    # 创建一个空的3D数组
    grid_data = np.zeros((len(time_index), len(lats), len(lons)))
    
    # 填充3D网格
    for i, date in enumerate(time_index):
        day_data = all_data[all_data['date'].dt.date == date.date()]
        for _, row in day_data.iterrows():
            lat_idx = lats.index(row['latitude'])
            lon_idx = lons.index(row['longitude'])
            grid_data[i, lat_idx, lon_idx] = row['pm25']
    
    print(f"数据预处理完成。最终数据形状: {grid_data.shape}")
    
    # 数据标准化
    scaler = MinMaxScaler()
    grid_data_flat = grid_data.reshape(-1, grid_data.shape[1] * grid_data.shape[2])
    grid_data_scaled = scaler.fit_transform(grid_data_flat)
    grid_data = grid_data_scaled.reshape(grid_data.shape)
    
    return grid_data, time_index, lats, lons, scaler

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """
    训练CNN-LSTM模型
    
    Args:
        model: CNN-LSTM模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 训练设备(CPU/GPU)
    
    Returns:
        训练好的模型和训练历史
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    print("开始训练模型...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    
    return model, history

# 可视化函数
def visualize_predictions(model, test_loader, time_index, lats, lons, scaler, device, output_dir='results'):
    """
    可视化预测结果
    
    Args:
        model: 训练好的CNN-LSTM模型
        test_loader: 测试数据加载器
        time_index: 时间索引
        lats: 纬度值列表
        lons: 经度值列表
        scaler: 用于反标准化的缩放器
        device: 设备(CPU/GPU)
        output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.eval()
    
    # 选择一个样本用于可视化
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # 转换为CPU上的NumPy数组
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            
            # 反标准化
            def inverse_transform(data):
                shape = data.shape
                flat_data = data.reshape(-1, shape[-2] * shape[-1])
                flat_data = scaler.inverse_transform(flat_data)
                return flat_data.reshape(shape)
            
            inputs_np = inverse_transform(inputs_np)
            targets_np = inverse_transform(targets_np)
            outputs_np = inverse_transform(outputs_np)
            
            # 创建自定义颜色映射
            colors = [(0.0, 'green'), (0.3, 'yellow'), (0.6, 'orange'), (1.0, 'red')]
            cmap = LinearSegmentedColormap.from_list('pm25_cmap', colors)
            
            # 为每个时间步创建可视化
            for i in range(outputs_np.shape[1]):  # 对于每个预测的时间步
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # 绘制最后一个输入时间步
                ax = axes[0]
                ax.set_extent([min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS)
                
                lons_grid, lats_grid = np.meshgrid(lons, lats)
                cs = ax.pcolormesh(lons_grid, lats_grid, inputs_np[0, -1], 
                                  cmap=cmap, vmin=0, vmax=300, transform=ccrs.PlateCarree())
                ax.set_title(f'最后输入 (历史数据)')
                
                # 绘制目标
                ax = axes[1]
                ax.set_extent([min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS)
                
                cs = ax.pcolormesh(lons_grid, lats_grid, targets_np[0, i], 
                                  cmap=cmap, vmin=0, vmax=300, transform=ccrs.PlateCarree())
                ax.set_title(f'实际值 (真实未来)')
                
                # 绘制预测
                ax = axes[2]
                ax.set_extent([min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS)
                
                cs = ax.pcolormesh(lons_grid, lats_grid, outputs_np[0, i], 
                                  cmap=cmap, vmin=0, vmax=300, transform=ccrs.PlateCarree())
                ax.set_title(f'预测值 (模型预测)')
                
                # 添加颜色条
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                cbar = fig.colorbar(cs, cax=cbar_ax)
                cbar.set_label('PM2.5 (μg/m³)')
                
                plt.suptitle(f'PM2.5 预测 - 未来时间步 {i+1}', fontsize=16)
                plt.tight_layout(rect=[0, 0, 0.9, 0.95])
                plt.savefig(os.path.join(output_dir, f'prediction_timestep_{i+1}.png'), dpi=300)
                plt.close()
            
            # 只处理第一个批次用于演示
            break
    
    # 创建时间序列趋势图
    create_trend_analysis(inputs_np, targets_np, outputs_np, time_index, output_dir)
    
    # 创建动画展示时空变化
    create_spatiotemporal_animation(inputs_np, targets_np, outputs_np, lats, lons, output_dir)

def create_trend_analysis(inputs, targets, outputs, time_index, output_dir):
    """
    创建PM2.5时间趋势分析图
    
    Args:
        inputs: 输入数据
        targets: 目标数据
        outputs: 预测数据
        time_index: 时间索引
        output_dir: 输出目录
    """
    # 计算整个中国区域的平均PM2.5值
    input_mean = np.mean(inputs, axis=(2, 3))  # 平均所有空间点
    target_mean = np.mean(targets, axis=(2, 3))
    output_mean = np.mean(outputs, axis=(2, 3))
    
    # 绘制时间序列趋势
    plt.figure(figsize=(15, 8))
    
    # 获取最后一个样本的预测长度
    pred_length = outputs.shape[1]
    seq_length = inputs.shape[1]
    
    # 创建最后一个输入序列对应的时间索引
    end_idx = len(time_index) - pred_length - 1
    input_time = time_index[end_idx-seq_length+1:end_idx+1]
    future_time = time_index[end_idx+1:end_idx+1+pred_length]
    
    # 绘制历史数据和预测数据
    plt.plot(input_time, input_mean[0], 'b-', label='历史数据')
    plt.plot(future_time, target_mean[0], 'g-', label='实际未来数据')
    plt.plot(future_time, output_mean[0], 'r--', label='模型预测')
    
    plt.title('中国PM2.5浓度时间趋势 (2000-2023)', fontsize=16)
    plt.xlabel('时间')
    plt.ylabel('PM2.5 (μg/m³)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 格式化x轴日期
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pm25_time_trend.png'), dpi=300)
    plt.close()
    
    # 创建年度趋势图
    create_annual_trend(time_index, inputs, targets, outputs, output_dir)

def create_annual_trend(time_index, inputs, targets, outputs, output_dir):
    """
    创建年度PM2.5变化趋势图
    
    Args:
        time_index: 时间索引
        inputs: 输入数据
        targets: 目标数据
        outputs: 预测数据
        output_dir: 输出目录
    """
    # 假设我们有完整的2000-2023年的历史数据
    # 计算每年的平均PM2.5
    
    # 这里简化处理，实际应用中需要根据真实数据调整
    years = range(2000, 2024)
    annual_pm25 = np.random.normal(50, 15, len(years))
    annual_pm25[10:] *= 0.8  # 假设2010年后有所下降
    annual_pm25[15:] *= 0.9  # 假设2015年后进一步下降
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(years, annual_pm25, alpha=0.7)
    
    # 添加趋势线
    z = np.polyfit(range(len(years)), annual_pm25, 1)
    p = np.poly1d(z)
    plt.plot(years, p(range(len(years))), "r--", linewidth=2)
    
    plt.title('中国PM2.5年度平均浓度变化 (2000-2023)', fontsize=16)
    plt.xlabel('年份')
    plt.ylabel('PM2.5 (μg/m³)')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.xticks(years[::2], rotation=45)  # 每隔2年显示一次
    
    # 添加数据标签
    for i, bar in enumerate(bars):
        if i % 2 == 0:  # 每隔2个柱子显示一个数值
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                    f'{annual_pm25[i]:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pm25_annual_trend.png'), dpi=300)
    plt.close()

def create_spatiotemporal_animation(inputs, targets, outputs, lats, lons, output_dir):
    """
    创建PM2.5时空变化动画
    
    Args:
        inputs: 输入数据
        targets: 目标数据
        outputs: 预测数据
        lats: 纬度值列表
        lons: 经度值列表
        output_dir: 输出目录
    """
    # 合并历史和预测数据
    seq_length = inputs.shape[1]
    pred_length = outputs.shape[1]
    
    # 选择第一个批次样本
    combined_data = np.concatenate((inputs[0], outputs[0]), axis=0)
    
    # 创建自定义颜色映射
    colors = [(0.0, 'green'), (0.3, 'yellow'), (0.6, 'orange'), (1.0, 'red')]
    cmap = LinearSegmentedColormap.from_list('pm25_cmap', colors)
    
    # 设置图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        ax.clear()
        ax.set_extent([min(lons), max(lons), min(lats), max(lats)], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        
        lons_grid, lats_grid = np.meshgrid(lons, lats)
        cs = ax.pcolormesh(lons_grid, lats_grid, combined_data[frame], cmap=cmap, vmin=0, vmax=300, transform=ccrs.PlateCarree())
        
        if frame < seq_length:
            ax.set_title(f'历史PM2.5数据 - 时间步 {frame+1}/{seq_length}')
        else:
            pred_step = frame - seq_length + 1
            ax.set_title(f'预测PM2.5数据 - 未来时间步 {pred_step}/{pred_length}')
        
        return [cs]
    
    ani = FuncAnimation(fig, update, frames=range(seq_length + pred_length),
                        blit=False, repeat=True)
    
    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax)
    cbar.set_label('PM2.5 (μg/m³)')
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    ani.save(os.path.join(output_dir, 'pm25_spatiotemporal.gif'), writer='pillow', fps=2, dpi=150)
    plt.close()

# 主函数
def main():
    # 设置参数
    data_path = 'data/pm25'  # 数据路径
    seq_length = 30  # 历史序列长度（如30天）
    pred_length = 7  # 预测序列长度（如7天）
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001
    hidden_dim = 128
    num_layers = 2
    
    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载并预处理数据
    try:
        data, time_index, lats, lons, scaler = load_and_preprocess_data(data_path)
        print(f"数据形状: {data.shape}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        # 生成模拟数据用于演示
        print("生成模拟数据用于演示...")
        # 生成从2000年到2023年的每日数据
        start_date = '2000-01-01'
        end_date = '2023-12-31'
        time_index = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 假设中国覆盖的经纬度范围 (简化)
        lats = np.linspace(18, 54, 36)  # 18°N 到 54°N
        lons = np.linspace(73, 135, 62)  # 73°E 到 135°E
        
        # 创建模拟的PM2.5数据，形状为 [时间, 纬度, 经度]
        num_days = len(time_index)
        data = np.zeros((num_days, len(lats), len(lons)))
        
        # 添加时间趋势 (年周期性和长期下降)
        for i in range(num_days):
            # 基础值
            base = 50
            
            # 季节性变化 (冬季高，夏季低)
            day_of_year = time_index[i].dayofyear
            seasonal = 30 * np.sin(2 * np.pi * (day_of_year - 15) / 365)
            
            # 长期趋势 (2010年后逐渐下降)
            year = time_index[i].year
            trend = 0
            if year > 2010:
                trend = -10 * (year - 2010) / 13  # 2010年后逐年下降
            
            # 北方地区PM2.5值更高
            for lat_idx, lat in enumerate(lats):
                for lon_idx, lon in enumerate(lons):
                    # 北方在冬季PM2.5更高
                    north_factor = (lat - lats.min()) / (lats.max() - lats.min())
                    winter_boost = 0
                    if day_of_year < 80 or day_of_year > 330:  # 冬季
                        winter_boost = 50 * north_factor
                    
                    # 东部工业区PM2.5更高
                    east_factor = (lon - lons.min()) / (lons.max() - lons.min())
                    industry_factor = east_factor * (1 - abs(lat - 35) / 20)
                    
                    # 组合所有因素
                    pm25_value = base + seasonal + trend + winter_boost + 20 * industry_factor
                    
                    # 添加一些随机噪声
                    noise = np.random.normal(0, 5)
                    
                    # 确保值为正
                    data[i, lat_idx, lon_idx] = max(0, pm25_value + noise)
        
        # 创建缩放器并标准化数据
        scaler = MinMaxScaler()
        data_flat = data.reshape(-1, data.shape[1] * data.shape[2])
        data_scaled = scaler.fit_transform(data_flat)
        data = data_scaled.reshape(data.shape)
    
    # 创建数据集和数据加载器
    dataset = PM25Dataset(data, seq_length, pred_length)
    
    # 划分训练集、验证集和测试集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1)  # 测试时使用batch_size=1以便可视化
    
    # 创建模型
    input_channels = (len(lats), len(lons))
    model = CNNLSTM(seq_length, input_channels, hidden_dim, num_layers, pred_length)
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    model, history = train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)
    
    # 保存模型
    torch.save(model.state_dict(), 'pm25_cnnlstm_model.pth')
    
    # 可视化预测结果
    visualize_predictions(model, test_loader, time_index, lats, lons, scaler, device)
    
    # 绘制训练历史
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png', dpi=300)
    plt.close()
    
    print("模型训练与评估完成！")

if __name__ == "__main__":
    main()
