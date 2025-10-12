# CNN-LSTM PM2.5预测 - GPU加速版本

## 概述

这是CNN-LSTM模型的GPU加速版本，专为NVIDIA GPU优化，可将训练速度提升2-3倍。

## GPU加速特性

### 1. 混合精度训练 (AMP)
- 使用 `torch.cuda.amp` 自动混合精度
- FP16/FP32混合计算，减少显存占用50%
- 训练速度提升2-3倍
- 保持模型精度

### 2. 优化的数据加载
- `pin_memory=True`: 锁页内存，加速CPU→GPU传输
- `num_workers=4`: 多进程并行数据加载
- `persistent_workers=True`: 保持工作进程存活
- 更大的批处理大小 (128 vs 64)

### 3. GPU内存管理
- 自动内存监控
- 定期内存清理 (`torch.cuda.empty_cache()`)
- 显示实时GPU内存使用

### 4. 训练优化
- 梯度裁剪防止梯度爆炸
- cuDNN benchmark自动优化卷积算法
- 训练速度统计 (samples/sec)

### 5. 智能设备选择
- 自动检测GPU可用性
- 无GPU时自动回退到CPU模式
- 显示GPU型号、显存、计算能力

## 系统要求

### 必需
- Python 3.8+
- PyTorch 1.10+ (with CUDA support)
- NVIDIA GPU (推荐: GTX 1060 6GB或更高)
- CUDA 11.0+

### 推荐配置
- GPU显存: 8GB+
- CUDA 11.7+
- cuDNN 8.0+

### Python依赖
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy matplotlib scikit-learn tqdm
pip install bayesian-optimization  # 可选
```

## 使用方法

### 基本运行
```bash
python CNN-LSTM-GPU.py
```

### 检查GPU是否可用
```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
```

## 性能对比

### 训练速度 (估计)
| 配置 | CPU版本 | GPU版本 | 加速比 |
|-----|---------|---------|--------|
| 单epoch | ~120s | ~40s | 3.0x |
| 完整训练 | ~3小时 | ~1小时 | 3.0x |

### 显存使用 (估计)
- 批处理大小=128: ~4-6 GB
- 批处理大小=64: ~2-3 GB
- 混合精度可节省~50%显存

## 文件说明

### 代码文件
- `CNN-LSTM-GPU.py`: GPU加速版本主程序
- `CNN-LSTM.py`: 原始CPU版本（保留）

### 输出文件（带_gpu后缀）
- `model_performance_gpu.csv`: 模型性能对比
- `window_comparison_gpu.csv`: 窗口大小对比
- `attention_weights_window*_gpu.csv`: Attention权重
- `predictions_*_gpu.csv`: 预测结果
- `best_parameters_gpu.csv`: 最佳超参数

### 模型文件
- `cnn_lstm_window*_optimized_gpu.pth`: PyTorch模型权重
- `cnn_lstm_window*_optimized_gpu.pkl`: 完整模型对象
- `scaler_X_gpu.pkl`: 特征标准化器
- `scaler_y_gpu.pkl`: 目标标准化器

### 可视化图表（带_gpu后缀）
- `training_curves_window*_gpu.png`: 训练曲线
- `prediction_scatter_gpu.png`: 预测散点图
- `timeseries_comparison_gpu.png`: 时间序列对比
- `residuals_analysis_gpu.png`: 残差分析
- `attention_weights_gpu.png`: Attention权重
- `model_comparison_gpu.png`: 模型对比
- `error_distribution_gpu.png`: 误差分布
- `window_size_comparison_gpu.png`: 窗口对比

## 关键代码修改

### 混合精度训练
```python
scaler = GradScaler()  # 创建梯度缩放器

# 训练循环
with autocast():  # 自动混合精度
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### 优化的DataLoader
```python
train_loader = DataLoader(
    dataset, 
    batch_size=128,           # GPU使用更大批次
    shuffle=True,
    pin_memory=True,          # 锁页内存
    num_workers=4,            # 多进程加载
    persistent_workers=True   # 保持工作进程
)
```

## 故障排除

### 显存不足 (Out of Memory)
1. 减小批处理大小：`BATCH_SIZE = 64` 或 `32`
2. 减少序列长度：只训练一个窗口大小
3. 清空GPU缓存：`torch.cuda.empty_cache()`

### CUDA错误
1. 检查PyTorch和CUDA版本兼容性
2. 更新显卡驱动
3. 重新安装PyTorch with CUDA

### 速度没有提升
1. 确认使用GPU：检查设备显示
2. 检查数据加载：增加 `num_workers`
3. 确认混合精度启用

## 性能调优建议

### 1. 批处理大小
- 8GB显存: 使用128
- 6GB显存: 使用64-96
- 4GB显存: 使用32-64

### 2. 数据加载进程
- 推荐: CPU核心数的1/2到1/4
- 默认: 4个进程
- 过多进程可能导致CPU瓶颈

### 3. 超参数搜索
- 贝叶斯优化：减少搜索epoch数（30 vs 50）
- 网格搜索：减少参数组合或epoch数

## 与CPU版本的区别

| 特性 | CPU版本 | GPU版本 |
|-----|---------|---------|
| 混合精度训练 | ✗ | ✓ |
| 批处理大小 | 64 | 128 |
| pin_memory | False | True |
| num_workers | 0 | 4 |
| 梯度裁剪 | ✗ | ✓ |
| 内存监控 | ✗ | ✓ |
| 速度统计 | ✗ | ✓ |
| 文件后缀 | 无 | _gpu |

## 注意事项

1. **数据路径**: 确保修改为您的数据路径
2. **显存监控**: 注意显存使用，避免OOM
3. **并行性**: CPU和GPU版本可以同时运行（使用不同输出）
4. **模型兼容性**: GPU训练的模型可以在CPU上加载使用

## 技术支持

如果遇到问题：
1. 检查CUDA和PyTorch安装
2. 查看错误日志
3. 尝试减小批处理大小
4. 使用CPU版本作为对照

## 许可证

与原项目保持一致

## 更新日志

### v1.0 (GPU加速版)
- ✓ 混合精度训练 (AMP)
- ✓ 优化数据加载
- ✓ GPU内存管理
- ✓ 训练速度监控
- ✓ 自动CPU回退
- ✓ 完整的文档

