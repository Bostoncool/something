# CNN-LSTM GPU Accelerated Version - Quick Start Guide

## 📋 Table of Contents
1. [Environment Check](#environment-check)
2. [Quick Run](#quick-run)
3. [Performance Testing](#performance-testing)
4. [Common Issues](#common-issues)

---

## 🔍 Environment Check

### Step 1: Check if CUDA is Installed

Run in command line:
```bash
nvidia-smi
```

If GPU information is displayed, CUDA is correctly installed.

### Step 2: Check PyTorch CUDA Support

Run in Python:
```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Display GPU name
```

### Step 3: Install Dependencies

```bash
# Install PyTorch (CUDA 11.8 version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install pandas numpy matplotlib scikit-learn tqdm

# Optional: Bayesian optimization
pip install bayesian-optimization
```

---

## 🚀 Quick Run

### Method 1: Run GPU Version Directly

```bash
cd Short-Essay-Beijing/DeepLearning/CNN-LSTM
python CNN-LSTM-GPU.py
```

The program will automatically:
- ✓ Detect GPU availability
- ✓ Enable mixed precision training
- ✓ Optimize data loading
- ✓ Display training speed statistics
- ✓ Monitor GPU memory usage

### Method 2: Test Performance First

It's recommended to run performance test first to confirm GPU acceleration effect:

```bash
python test_gpu_performance.py
```

This will show:
- CPU vs GPU training speed comparison
- FP32 vs FP16 performance difference
- Memory usage
- Speedup ratio statistics

---

## 📊 Performance Testing

### Run Performance Test Script

```bash
python test_gpu_performance.py
```

### Expected Results Example

```
[Performance Comparison Summary]
================================================================================

Training Speed Comparison:
  CPU (batch=64):        150 samples/s
  GPU FP32 (batch=128):  800 samples/s  (5.33x)
  GPU FP16 (batch=128):  1200 samples/s (8.00x)

Batch Time Comparison:
  CPU:      426.67ms
  GPU FP32: 160.00ms  (2.67x speedup)
  GPU FP16: 106.67ms  (4.00x speedup)

Memory Usage (GPU):
  FP32 peak: 3.42GB
  FP16 relative saving: ~50% (estimated)

Recommended Configuration:
  ✓ Significant GPU acceleration, GPU version recommended
  ✓ Using mixed precision can achieve 1.50x additional speedup
```

### Performance Metrics Explanation

| Speedup | Rating | Recommendation |
|---------|--------|----------------|
| > 5x | Excellent | Strongly recommend using GPU |
| 3-5x | Good | Recommend using GPU |
| 2-3x | Fair | Can use GPU |
| < 2x | Poor | Check configuration or consider CPU |

---

## 🎯 Actual Training

### Modify Data Paths

Modify the following paths in `CNN-LSTM-GPU.py`:

```python
# Lines 111-113
pollution_all_path = r'your_path\Benchmark\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'your_path\Benchmark\extra(SO2+NO2+CO+O3)'
era5_path = r'your_path\ERA5-Beijing-CSV'
```

### Adjust GPU Parameters (Optional)

If memory is insufficient, you can modify batch size:

```python
# Line 161
BATCH_SIZE = 64  # Reduce batch size (default 128)
```

### Start Training

```bash
python CNN-LSTM-GPU.py
```

During training, it will display in real-time:
- Training/validation loss
- Training speed (samples/s)
- GPU memory usage
- Estimated remaining time

---

## ❓ Common Issues

### Q1: Shows "CUDA not available"

**Solutions:**
1. Check if PyTorch with CUDA support is installed
2. Confirm graphics driver is installed
3. Run `nvidia-smi` to check GPU status

**Reinstall PyTorch:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Q2: Out of Memory

**Solutions (by priority):**

1. **Reduce batch size**
   ```python
   BATCH_SIZE = 64  # Or smaller: 32, 16
   ```

2. **Reduce sequence length**
   ```python
   sequence_lengths = [7, 14]  # Only train 2 windows
   ```

3. **Reduce model complexity** (in hyperparameter search)
   ```python
   hidden_size = 32  # Reduce hidden layer size
   num_filters = 16  # Reduce number of convolutional kernels
   ```

4. **Reduce hyperparameter search**
   ```python
   # Use fewer search iterations
   optimizer_bayes.maximize(init_points=2, n_iter=5)
   ```

### Q3: Training is still slow

**Possible causes and solutions:**

1. **Data loading bottleneck**
   - Increase `NUM_WORKERS` (line 162)
   - Check data storage location (SSD is faster)

2. **Insufficient GPU performance**
   - Check GPU utilization: run `nvidia-smi`
   - Confirm GPU is not being used by other programs

3. **Batch size too small**
   - GPU is suitable for large batch processing
   - Increase batch size when memory allows

### Q4: How to use trained model on machine without GPU?

GPU-trained models can be loaded on CPU:

```python
# Load model
model = torch.load('models/cnn_lstm_window14_optimized_gpu.pkl', 
                   map_location='cpu')

# Or only load weights
model = CNNLSTMAttention(**params)
model.load_state_dict(torch.load('models/cnn_lstm_window14_optimized_gpu.pth',
                                  map_location='cpu'))
```

### Q5: How to train only one window size?

Modify sequence length list (line 126):

```python
sequence_lengths = [14]  # Only train 14-day window
```

This can significantly reduce training time.

### Q6: Can I run CPU and GPU versions simultaneously?

Yes! They use different output file names (GPU version has `_gpu` suffix):
- CPU version: `CNN-LSTM.py` → `model_performance.csv`
- GPU version: `CNN-LSTM-GPU.py` → `model_performance_gpu.csv`

But note:
- More system resources consumed
- GPU memory will be occupied by GPU version

---

## 💡 Optimization Recommendations

### 1. First-time Usage Recommendation

```python
# Quick test configuration (~30 minutes)
sequence_lengths = [14]           # Only train one window
optimizer_bayes.maximize(init_points=2, n_iter=5)  # Reduce hyperparameter search
```

### 2. Complete Training Configuration

```python
# Complete configuration (2-3 hours)
sequence_lengths = [7, 14, 30]    # Train all windows
optimizer_bayes.maximize(init_points=3, n_iter=10)  # Complete search
```

### 3. Production Environment Configuration

```python
# Best performance configuration
BATCH_SIZE = 256                  # Larger batch (requires 16GB+ memory)
NUM_WORKERS = 8                   # More data loading processes
sequence_lengths = [14]           # Use best performing window
```

---

## 📈 Monitor Training Progress

### Real-time Monitoring

During training, it displays:
```
Epoch [10/100], Train Loss: 0.1234, Val Loss: 0.1456, 
Time: 25.3s, Speed: 845 samples/s, GPU Mem: 3.2/4.5 GB
```

### Check Output Files

After training completion, check:
- `output/model_performance_gpu.csv` - Performance metrics
- `output/training_curves_window*_gpu.png` - Training curves
- `models/cnn_lstm_window*_optimized_gpu.pth` - Model files

---

## 🔧 Advanced Configuration

### Custom Mixed Precision Training

For finer-grained control:

```python
# In train_model function
scaler = GradScaler(
    init_scale=2.**16,    # Initial scale factor
    growth_factor=2.0,    # Growth factor
    backoff_factor=0.5,   # Backoff factor
    growth_interval=2000  # Growth interval
)
```

### Multi-GPU Training (Experimental)

If you have multiple GPUs:

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")
```

---

## 📞 Get Help

### View Detailed Documentation
```bash
# View complete README
cat README_GPU.md
```

### Check Logs
All output during training is displayed in terminal, can be redirected to file:
```bash
python CNN-LSTM-GPU.py > training.log 2>&1
```

### Debug Mode
If encountering issues, add at beginning of code:
```python
torch.autograd.set_detect_anomaly(True)  # Detect anomalous gradients
```

---

## ✅ Checklist

Confirm before use:
- [ ] CUDA installed (run `nvidia-smi`)
- [ ] PyTorch supports CUDA (`torch.cuda.is_available() == True`)
- [ ] All dependencies installed
- [ ] Data paths correctly configured
- [ ] GPU has sufficient memory (recommended 8GB+)
- [ ] Performance test has been run

Start training! 🚀

---

**Tip:** If this is your first time using the GPU version, it's strongly recommended to run `test_gpu_performance.py` for performance testing first!
