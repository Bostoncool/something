"""
GPU Performance Testing Script
Used to test GPU acceleration effect and verify environment configuration

Features:
1. Detect GPU availability
2. Test mixed precision training speed
3. Compare CPU vs GPU training time
4. Memory usage monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import time
import numpy as np

print("=" * 80)
print("CNN-LSTM GPU Performance Test")
print("=" * 80)

# ============================== 1. Environment Detection ==============================
print("\n[1. Environment Detection]")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU memory: {gpu_props.total_memory / 1e9:.2f} GB")
    print(f"Compute capability: {gpu_props.major}.{gpu_props.minor}")
    print(f"Multiprocessor count: {gpu_props.multi_processor_count}")
else:
    print("⚠️  Warning: CUDA GPU not detected")
    print("   GPU performance testing will not be possible")

# ============================== 2. Simplified Model Definition ==============================
class SimpleCNNLSTM(nn.Module):
    """Simplified CNN-LSTM model for testing"""
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

# ============================== 3. Generate Test Data ==============================
print("\n[2. Generating Test Data]")

seq_length = 14
batch_size_cpu = 64
batch_size_gpu = 128
input_size = 50
num_batches = 50  # Number of test batches

print(f"Sequence length: {seq_length}")
print(f"CPU batch size: {batch_size_cpu}")
print(f"GPU batch size: {batch_size_gpu}")
print(f"Input feature count: {input_size}")
print(f"Test batch count: {num_batches}")

# ============================== 4. CPU Training Test ==============================
print("\n[3. CPU Training Test]")

model_cpu = SimpleCNNLSTM(input_size=input_size)
model_cpu = model_cpu.to('cpu')
criterion = nn.MSELoss()
optimizer_cpu = optim.Adam(model_cpu.parameters(), lr=0.001)

print("Starting CPU training test...")
cpu_times = []

for batch_idx in range(num_batches):
    # Generate random data
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
        print(f"  Batch {batch_idx + 1}/{num_batches}, time: {batch_time*1000:.2f}ms")

cpu_avg_time = np.mean(cpu_times)
cpu_total_time = np.sum(cpu_times)
cpu_throughput = batch_size_cpu / cpu_avg_time

print(f"\nCPU Training Results:")
print(f"  Total time: {cpu_total_time:.2f}s")
print(f"  Average per batch: {cpu_avg_time*1000:.2f}ms")
print(f"  Throughput: {cpu_throughput:.0f} samples/s")

# ============================== 5. GPU Training Test (without AMP) ==============================
if torch.cuda.is_available():
    print("\n[4. GPU Training Test (FP32)]")
    
    model_gpu = SimpleCNNLSTM(input_size=input_size)
    model_gpu = model_gpu.to('cuda')
    criterion_gpu = nn.MSELoss()
    optimizer_gpu = optim.Adam(model_gpu.parameters(), lr=0.001)
    
    print("Starting GPU training test (FP32)...")
    
    # Warmup
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
            print(f"  Batch {batch_idx + 1}/{num_batches}, time: {batch_time*1000:.2f}ms, "
                  f"memory: {mem_allocated:.2f}/{mem_reserved:.2f}GB")
    
    gpu_avg_time = np.mean(gpu_times)
    gpu_total_time = np.sum(gpu_times)
    gpu_throughput = batch_size_gpu / gpu_avg_time
    
    print(f"\nGPU Training Results (FP32):")
    print(f"  Total time: {gpu_total_time:.2f}s")
    print(f"  Average per batch: {gpu_avg_time*1000:.2f}ms")
    print(f"  Throughput: {gpu_throughput:.0f} samples/s")
    print(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    
    # ============================== 6. GPU + AMP Training Test ==============================
    print("\n[5. GPU Training Test (FP16 Mixed Precision)]")
    
    model_amp = SimpleCNNLSTM(input_size=input_size)
    model_amp = model_amp.to('cuda')
    criterion_amp = nn.MSELoss()
    optimizer_amp = optim.Adam(model_amp.parameters(), lr=0.001)
    scaler = GradScaler()
    
    print("Starting GPU+AMP training test (FP16)...")
    
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
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
            print(f"  Batch {batch_idx + 1}/{num_batches}, time: {batch_time*1000:.2f}ms, "
                  f"memory: {mem_allocated:.2f}/{mem_reserved:.2f}GB")
    
    amp_avg_time = np.mean(amp_times)
    amp_total_time = np.sum(amp_times)
    amp_throughput = batch_size_gpu / amp_avg_time
    
    print(f"\nGPU+AMP Training Results (FP16):")
    print(f"  Total time: {amp_total_time:.2f}s")
    print(f"  Average per batch: {amp_avg_time*1000:.2f}ms")
    print(f"  Throughput: {amp_throughput:.0f} samples/s")
    print(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    
    # ============================== 7. Performance Comparison ==============================
    print("\n" + "=" * 80)
    print("[6. Performance Comparison Summary]")
    print("=" * 80)
    
    print(f"\nTraining Speed Comparison:")
    print(f"  CPU (batch={batch_size_cpu}):        {cpu_throughput:.0f} samples/s")
    print(f"  GPU FP32 (batch={batch_size_gpu}):   {gpu_throughput:.0f} samples/s  ({gpu_throughput/cpu_throughput:.2f}x)")
    print(f"  GPU FP16 (batch={batch_size_gpu}):   {amp_throughput:.0f} samples/s  ({amp_throughput/cpu_throughput:.2f}x)")
    
    print(f"\nBatch Time Comparison:")
    print(f"  CPU:      {cpu_avg_time*1000:.2f}ms")
    print(f"  GPU FP32: {gpu_avg_time*1000:.2f}ms  ({cpu_avg_time/gpu_avg_time:.2f}x speedup)")
    print(f"  GPU FP16: {amp_avg_time*1000:.2f}ms  ({cpu_avg_time/amp_avg_time:.2f}x speedup)")
    
    print(f"\nMemory Usage (GPU):")
    print(f"  FP32 peak: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    print(f"  FP16 relative saving: ~50% (estimated)")
    
    print(f"\nRecommended Configuration:")
    if gpu_throughput / cpu_throughput > 2:
        print("  ✓ Significant GPU acceleration, GPU version recommended")
        print(f"  ✓ Using mixed precision can achieve {amp_throughput/gpu_throughput:.2f}x additional speedup")
    else:
        print("  ⚠️  GPU acceleration effect not significant, possible issues:")
        print("     - Low GPU performance")
        print("     - Excessive data transfer overhead")
        print("     - Small model size")
    
    torch.cuda.empty_cache()

else:
    print("\n⚠️  Skipping GPU test (CUDA not detected)")
    print("\nTo enable GPU acceleration, please:")
    print("  1. Install PyTorch with CUDA support")
    print("  2. Ensure NVIDIA drivers are installed")
    print("  3. Check CUDA environment configuration")

print("\n" + "=" * 80)
print("Performance Testing Complete!")
print("=" * 80)

print("\nTips:")
print("  - If speedup is lower than expected, try increasing batch size")
print("  - Reduce batch size if memory is insufficient")
print("  - Mixed precision training can save about 50% memory")
print("  - Actual training speed is also affected by data loading")
