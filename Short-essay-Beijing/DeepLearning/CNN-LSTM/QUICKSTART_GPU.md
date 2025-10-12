# CNN-LSTM GPU加速版本 - 快速开始指南

## 📋 目录
1. [环境检查](#环境检查)
2. [快速运行](#快速运行)
3. [性能测试](#性能测试)
4. [常见问题](#常见问题)

---

## 🔍 环境检查

### 第一步：检查CUDA是否安装

在命令行运行：
```bash
nvidia-smi
```

如果显示GPU信息，说明CUDA已正确安装。

### 第二步：检查PyTorch CUDA支持

在Python中运行：
```python
import torch
print(torch.cuda.is_available())  # 应该返回 True
print(torch.cuda.get_device_name(0))  # 显示GPU名称
```

### 第三步：安装依赖

```bash
# 安装PyTorch (CUDA 11.8版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install pandas numpy matplotlib scikit-learn tqdm

# 可选：贝叶斯优化
pip install bayesian-optimization
```

---

## 🚀 快速运行

### 方式1：直接运行GPU版本

```bash
cd Short-Essay-Beijing/DeepLearning/CNN-LSTM
python CNN-LSTM-GPU.py
```

程序会自动：
- ✓ 检测GPU可用性
- ✓ 启用混合精度训练
- ✓ 优化数据加载
- ✓ 显示训练速度统计
- ✓ 监控GPU内存使用

### 方式2：先测试性能

建议先运行性能测试，确认GPU加速效果：

```bash
python test_gpu_performance.py
```

这会显示：
- CPU vs GPU训练速度对比
- FP32 vs FP16性能差异
- 显存使用情况
- 加速比统计

---

## 📊 性能测试

### 运行性能测试脚本

```bash
python test_gpu_performance.py
```

### 预期结果示例

```
【性能对比总结】
================================================================================

训练速度对比:
  CPU (batch=64):        150 samples/s
  GPU FP32 (batch=128):  800 samples/s  (5.33x)
  GPU FP16 (batch=128):  1200 samples/s (8.00x)

单批次时间对比:
  CPU:      426.67ms
  GPU FP32: 160.00ms  (2.67x 加速)
  GPU FP16: 106.67ms  (4.00x 加速)

显存使用 (GPU):
  FP32峰值: 3.42GB
  FP16相对节省: ~50% (估计)

推荐配置:
  ✓ GPU加速效果显著，建议使用GPU版本
  ✓ 使用混合精度可获得 1.50x 额外加速
```

### 性能指标说明

| 加速比 | 评价 | 建议 |
|--------|------|------|
| > 5x | 优秀 | 强烈推荐使用GPU |
| 3-5x | 良好 | 推荐使用GPU |
| 2-3x | 一般 | 可以使用GPU |
| < 2x | 较差 | 检查配置或考虑CPU |

---

## 🎯 实际训练

### 修改数据路径

在 `CNN-LSTM-GPU.py` 中修改以下路径：

```python
# 第111-113行
pollution_all_path = r'你的路径\Benchmark\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'你的路径\Benchmark\extra(SO2+NO2+CO+O3)'
era5_path = r'你的路径\ERA5-Beijing-CSV'
```

### 调整GPU参数（可选）

如果显存不足，可以修改批处理大小：

```python
# 第161行
BATCH_SIZE = 64  # 降低批处理大小（默认128）
```

### 开始训练

```bash
python CNN-LSTM-GPU.py
```

训练过程中会实时显示：
- 训练/验证损失
- 训练速度 (samples/s)
- GPU内存使用
- 预计剩余时间

---

## ❓ 常见问题

### Q1: 显示"CUDA不可用"

**解决方案：**
1. 检查是否安装了支持CUDA的PyTorch版本
2. 确认显卡驱动已安装
3. 运行 `nvidia-smi` 检查GPU状态

**重新安装PyTorch：**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Q2: 显存不足 (Out of Memory)

**解决方案（按优先级）：**

1. **减小批处理大小**
   ```python
   BATCH_SIZE = 64  # 或更小：32, 16
   ```

2. **减少序列长度**
   ```python
   sequence_lengths = [7, 14]  # 只训练2个窗口
   ```

3. **降低模型复杂度**（超参数搜索中）
   ```python
   hidden_size = 32  # 减小隐藏层大小
   num_filters = 16  # 减少卷积核数量
   ```

4. **减少超参数搜索**
   ```python
   # 使用更少的搜索次数
   optimizer_bayes.maximize(init_points=2, n_iter=5)
   ```

### Q3: 训练速度仍然很慢

**可能原因和解决方案：**

1. **数据加载瓶颈**
   - 增加 `NUM_WORKERS` (第162行)
   - 检查数据存储位置（SSD更快）

2. **GPU性能不足**
   - 检查GPU使用率：运行 `nvidia-smi`
   - 确认GPU不是在被其他程序占用

3. **批处理大小过小**
   - GPU适合大批次处理
   - 在显存允许的情况下增大批次

### Q4: 如何在没有GPU的机器上使用训练好的模型？

GPU训练的模型可以在CPU上加载：

```python
# 加载模型
model = torch.load('models/cnn_lstm_window14_optimized_gpu.pkl', 
                   map_location='cpu')

# 或只加载权重
model = CNNLSTMAttention(**params)
model.load_state_dict(torch.load('models/cnn_lstm_window14_optimized_gpu.pth',
                                  map_location='cpu'))
```

### Q5: 如何只训练一个窗口大小？

修改序列长度列表（第126行）：

```python
sequence_lengths = [14]  # 只训练14天窗口
```

这样可以大幅减少训练时间。

### Q6: 能否同时运行CPU和GPU版本？

可以！它们使用不同的输出文件名（GPU版本带`_gpu`后缀）：
- CPU版本：`CNN-LSTM.py` → `model_performance.csv`
- GPU版本：`CNN-LSTM-GPU.py` → `model_performance_gpu.csv`

但注意：
- 占用系统资源较多
- GPU显存会被GPU版本占用

---

## 💡 优化建议

### 1. 首次使用建议

```python
# 快速测试配置（约30分钟）
sequence_lengths = [14]           # 只训练一个窗口
optimizer_bayes.maximize(init_points=2, n_iter=5)  # 减少超参数搜索
```

### 2. 完整训练配置

```python
# 完整配置（2-3小时）
sequence_lengths = [7, 14, 30]    # 训练所有窗口
optimizer_bayes.maximize(init_points=3, n_iter=10)  # 完整搜索
```

### 3. 生产环境配置

```python
# 最佳性能配置
BATCH_SIZE = 256                  # 更大批次（需要16GB+ 显存）
NUM_WORKERS = 8                   # 更多数据加载进程
sequence_lengths = [14]           # 使用表现最好的窗口
```

---

## 📈 监控训练进度

### 实时监控

训练过程中会显示：
```
Epoch [10/100], Train Loss: 0.1234, Val Loss: 0.1456, 
Time: 25.3s, Speed: 845 samples/s, GPU Mem: 3.2/4.5 GB
```

### 检查输出文件

训练完成后检查：
- `output/model_performance_gpu.csv` - 性能指标
- `output/training_curves_window*_gpu.png` - 训练曲线
- `models/cnn_lstm_window*_optimized_gpu.pth` - 模型文件

---

## 🔧 高级配置

### 自定义混合精度训练

如果需要更细粒度的控制：

```python
# 在train_model函数中
scaler = GradScaler(
    init_scale=2.**16,    # 初始缩放因子
    growth_factor=2.0,    # 增长因子
    backoff_factor=0.5,   # 回退因子
    growth_interval=2000  # 增长间隔
)
```

### 多GPU训练（实验性）

如果有多个GPU：

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"使用 {torch.cuda.device_count()} 个GPU")
```

---

## 📞 获取帮助

### 查看详细文档
```bash
# 查看完整README
cat README_GPU.md
```

### 检查日志
训练过程中的所有输出都会显示在终端，可以重定向到文件：
```bash
python CNN-LSTM-GPU.py > training.log 2>&1
```

### 调试模式
如果遇到问题，可以在代码开头添加：
```python
torch.autograd.set_detect_anomaly(True)  # 检测异常梯度
```

---

## ✅ 检查清单

使用前确认：
- [ ] CUDA已安装（运行 `nvidia-smi`）
- [ ] PyTorch支持CUDA（`torch.cuda.is_available() == True`）
- [ ] 所有依赖已安装
- [ ] 数据路径已正确配置
- [ ] GPU有足够显存（推荐8GB+）
- [ ] 已运行性能测试

开始训练！🚀

---

**提示：** 如果这是您第一次使用GPU版本，强烈建议先运行 `test_gpu_performance.py` 进行性能测试！

