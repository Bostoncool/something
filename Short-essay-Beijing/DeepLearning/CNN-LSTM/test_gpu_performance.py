"""
GPU性能测试脚本
用于测试GPU加速效果和验证环境配置

功能:
1. 检测GPU可用性
2. 测试混合精度训练速度
3. 对比CPU vs GPU训练时间
4. 显存使用监控
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import time
import numpy as np

print("=" * 80)
print("CNN-LSTM GPU性能测试")
print("=" * 80)

# ============================== 1. 环境检测 ==============================
print("\n【1. 环境检测】")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU显存: {gpu_props.total_memory / 1e9:.2f} GB")
    print(f"计算能力: {gpu_props.major}.{gpu_props.minor}")
    print(f"多核心数: {gpu_props.multi_processor_count}")
else:
    print("⚠️  警告: 未检测到CUDA GPU")
    print("   将无法进行GPU性能测试")

# ============================== 2. 简化模型定义 ==============================
class SimpleCNNLSTM(nn.Module):
    """简化的CNN-LSTM模型用于测试"""
    def __init__(self, input_size=50, hidden_size=64, num_filters=32):
        super(SimpleCNNLSTM, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.lstm = nn.LSTM(num_filters, hidden_size, num_layers=2, 
                           batch_first=True, dropout=0.2)
        
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(context_vector)
        
        return output

# ============================== 3. 生成测试数据 ==============================
print("\n【2. 生成测试数据】")

seq_length = 14
batch_size_cpu = 64
batch_size_gpu = 128
input_size = 50
num_batches = 50  # 测试批次数

print(f"序列长度: {seq_length}")
print(f"CPU批处理大小: {batch_size_cpu}")
print(f"GPU批处理大小: {batch_size_gpu}")
print(f"输入特征数: {input_size}")
print(f"测试批次数: {num_batches}")

# ============================== 4. CPU训练测试 ==============================
print("\n【3. CPU训练测试】")

model_cpu = SimpleCNNLSTM(input_size=input_size)
model_cpu = model_cpu.to('cpu')
criterion = nn.MSELoss()
optimizer_cpu = optim.Adam(model_cpu.parameters(), lr=0.001)

print("开始CPU训练测试...")
cpu_times = []

for batch_idx in range(num_batches):
    # 生成随机数据
    X_batch = torch.randn(batch_size_cpu, seq_length, input_size)
    y_batch = torch.randn(batch_size_cpu, 1)
    
    start_time = time.time()
    
    optimizer_cpu.zero_grad()
    outputs = model_cpu(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer_cpu.step()
    
    batch_time = time.time() - start_time
    cpu_times.append(batch_time)
    
    if (batch_idx + 1) % 10 == 0:
        print(f"  批次 {batch_idx + 1}/{num_batches}, 时间: {batch_time*1000:.2f}ms")

cpu_avg_time = np.mean(cpu_times)
cpu_total_time = np.sum(cpu_times)
cpu_throughput = batch_size_cpu / cpu_avg_time

print(f"\nCPU训练结果:")
print(f"  总时间: {cpu_total_time:.2f}s")
print(f"  平均每批次: {cpu_avg_time*1000:.2f}ms")
print(f"  吞吐量: {cpu_throughput:.0f} samples/s")

# ============================== 5. GPU训练测试 (不使用AMP) ==============================
if torch.cuda.is_available():
    print("\n【4. GPU训练测试 (FP32)】")
    
    model_gpu = SimpleCNNLSTM(input_size=input_size)
    model_gpu = model_gpu.to('cuda')
    criterion_gpu = nn.MSELoss()
    optimizer_gpu = optim.Adam(model_gpu.parameters(), lr=0.001)
    
    print("开始GPU训练测试 (FP32)...")
    
    # 预热
    for _ in range(5):
        X_warmup = torch.randn(batch_size_gpu, seq_length, input_size).to('cuda')
        y_warmup = torch.randn(batch_size_gpu, 1).to('cuda')
        outputs = model_gpu(X_warmup)
        loss = criterion_gpu(outputs, y_warmup)
        loss.backward()
        optimizer_gpu.step()
    
    torch.cuda.synchronize()
    
    gpu_times = []
    
    for batch_idx in range(num_batches):
        X_batch = torch.randn(batch_size_gpu, seq_length, input_size).to('cuda')
        y_batch = torch.randn(batch_size_gpu, 1).to('cuda')
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        optimizer_gpu.zero_grad()
        outputs = model_gpu(X_batch)
        loss = criterion_gpu(outputs, y_batch)
        loss.backward()
        optimizer_gpu.step()
        
        torch.cuda.synchronize()
        batch_time = time.time() - start_time
        gpu_times.append(batch_time)
        
        if (batch_idx + 1) % 10 == 0:
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  批次 {batch_idx + 1}/{num_batches}, 时间: {batch_time*1000:.2f}ms, "
                  f"显存: {mem_allocated:.2f}/{mem_reserved:.2f}GB")
    
    gpu_avg_time = np.mean(gpu_times)
    gpu_total_time = np.sum(gpu_times)
    gpu_throughput = batch_size_gpu / gpu_avg_time
    
    print(f"\nGPU训练结果 (FP32):")
    print(f"  总时间: {gpu_total_time:.2f}s")
    print(f"  平均每批次: {gpu_avg_time*1000:.2f}ms")
    print(f"  吞吐量: {gpu_throughput:.0f} samples/s")
    print(f"  峰值显存: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    
    # ============================== 6. GPU + AMP训练测试 ==============================
    print("\n【5. GPU训练测试 (FP16混合精度)】")
    
    model_amp = SimpleCNNLSTM(input_size=input_size)
    model_amp = model_amp.to('cuda')
    criterion_amp = nn.MSELoss()
    optimizer_amp = optim.Adam(model_amp.parameters(), lr=0.001)
    scaler = GradScaler()
    
    print("开始GPU+AMP训练测试 (FP16)...")
    
    torch.cuda.reset_peak_memory_stats()
    
    # 预热
    for _ in range(5):
        X_warmup = torch.randn(batch_size_gpu, seq_length, input_size).to('cuda')
        y_warmup = torch.randn(batch_size_gpu, 1).to('cuda')
        with autocast():
            outputs = model_amp(X_warmup)
            loss = criterion_amp(outputs, y_warmup)
        scaler.scale(loss).backward()
        scaler.step(optimizer_amp)
        scaler.update()
    
    torch.cuda.synchronize()
    
    amp_times = []
    
    for batch_idx in range(num_batches):
        X_batch = torch.randn(batch_size_gpu, seq_length, input_size).to('cuda')
        y_batch = torch.randn(batch_size_gpu, 1).to('cuda')
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        optimizer_amp.zero_grad(set_to_none=True)
        
        with autocast():
            outputs = model_amp(X_batch)
            loss = criterion_amp(outputs, y_batch)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer_amp)
        torch.nn.utils.clip_grad_norm_(model_amp.parameters(), max_norm=1.0)
        scaler.step(optimizer_amp)
        scaler.update()
        
        torch.cuda.synchronize()
        batch_time = time.time() - start_time
        amp_times.append(batch_time)
        
        if (batch_idx + 1) % 10 == 0:
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  批次 {batch_idx + 1}/{num_batches}, 时间: {batch_time*1000:.2f}ms, "
                  f"显存: {mem_allocated:.2f}/{mem_reserved:.2f}GB")
    
    amp_avg_time = np.mean(amp_times)
    amp_total_time = np.sum(amp_times)
    amp_throughput = batch_size_gpu / amp_avg_time
    
    print(f"\nGPU+AMP训练结果 (FP16):")
    print(f"  总时间: {amp_total_time:.2f}s")
    print(f"  平均每批次: {amp_avg_time*1000:.2f}ms")
    print(f"  吞吐量: {amp_throughput:.0f} samples/s")
    print(f"  峰值显存: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    
    # ============================== 7. 性能对比 ==============================
    print("\n" + "=" * 80)
    print("【6. 性能对比总结】")
    print("=" * 80)
    
    print(f"\n训练速度对比:")
    print(f"  CPU (batch={batch_size_cpu}):        {cpu_throughput:.0f} samples/s")
    print(f"  GPU FP32 (batch={batch_size_gpu}):   {gpu_throughput:.0f} samples/s  ({gpu_throughput/cpu_throughput:.2f}x)")
    print(f"  GPU FP16 (batch={batch_size_gpu}):   {amp_throughput:.0f} samples/s  ({amp_throughput/cpu_throughput:.2f}x)")
    
    print(f"\n单批次时间对比:")
    print(f"  CPU:      {cpu_avg_time*1000:.2f}ms")
    print(f"  GPU FP32: {gpu_avg_time*1000:.2f}ms  ({cpu_avg_time/gpu_avg_time:.2f}x 加速)")
    print(f"  GPU FP16: {amp_avg_time*1000:.2f}ms  ({cpu_avg_time/amp_avg_time:.2f}x 加速)")
    
    print(f"\n显存使用 (GPU):")
    print(f"  FP32峰值: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    print(f"  FP16相对节省: ~50% (估计)")
    
    print(f"\n推荐配置:")
    if gpu_throughput / cpu_throughput > 2:
        print("  ✓ GPU加速效果显著，建议使用GPU版本")
        print(f"  ✓ 使用混合精度可获得 {amp_throughput/gpu_throughput:.2f}x 额外加速")
    else:
        print("  ⚠️  GPU加速效果不明显，可能存在以下问题:")
        print("     - GPU性能较低")
        print("     - 数据传输开销过大")
        print("     - 模型规模较小")
    
    torch.cuda.empty_cache()

else:
    print("\n⚠️  跳过GPU测试（未检测到CUDA）")
    print("\n如需启用GPU加速，请:")
    print("  1. 安装支持CUDA的PyTorch")
    print("  2. 确保已安装NVIDIA驱动")
    print("  3. 检查CUDA环境配置")

print("\n" + "=" * 80)
print("性能测试完成！")
print("=" * 80)

print("\n提示:")
print("  - 如果加速比低于预期，尝试增大批处理大小")
print("  - 显存不足时可以减小批处理大小")
print("  - 混合精度训练可节省约50%显存")
print("  - 实际训练速度还受数据加载影响")

