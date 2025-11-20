# GPU使用率低的原因分析与解决方案

## 一、问题诊断

### 1.1 当前配置分析
- **Batch Size**: 128（可能偏小）
- **模型大小**: hidden_size=64, num_layers=2（计算量不足）
- **DataLoader**: num_workers=4（可能不够）
- **序列长度**: 7-30天（相对较短）
- **混合精度**: 已启用 ✓
- **Pin Memory**: 已启用 ✓

### 1.2 主要瓶颈识别

#### 🔴 严重问题
1. **Batch Size太小**: 128对于现代GPU（如V100/A100）来说太小，无法充分利用GPU的并行计算能力
2. **模型计算量不足**: hidden_size=64太小，LSTM的计算密度低，GPU利用率低
3. **缺少数据预取**: DataLoader没有设置prefetch_factor，GPU经常等待数据
4. **CPU-GPU同步阻塞**: evaluate_model中频繁的`.cpu().numpy()`导致同步等待

#### 🟡 中等问题
5. **没有使用torch.compile**: PyTorch 2.0+的编译优化可以显著提升性能
6. **optimizer.zero_grad()效率低**: 应该使用`set_to_none=True`
7. **缺少梯度累积**: 无法模拟更大的batch size
8. **NUM_WORKERS可能不够**: 4个worker可能无法及时提供数据

#### 🟢 轻微问题
9. **序列长度较短**: 7-30天的序列可能无法充分利用GPU的并行能力
10. **没有使用torch.backends.cudnn.allow_tf32**: TensorFloat-32可以提升性能

## 二、优化方案

### 2.1 立即优化（高优先级）

#### 优化1: 增大Batch Size
```python
BATCH_SIZE = 512  # 从128增加到512（根据GPU内存调整）
```

#### 优化2: 增大模型规模
```python
basic_params = {
    'hidden_size': 256,  # 从64增加到256
    'num_layers': 3,     # 从2增加到3
    ...
}
```

#### 优化3: 优化DataLoader配置
```python
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=8,  # 增加到8
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,  # 新增：预取4个batch
    drop_last=False
)
```

#### 优化4: 移除不必要的CPU-GPU同步
```python
# 在evaluate_model中，延迟转换到CPU
predictions = []
with torch.no_grad():
    for X_batch, _ in data_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        with autocast(enabled=USE_AMP):
            outputs = model(X_batch)
        predictions.append(outputs)  # 保持在GPU上
        
# 最后一次性转换
predictions = torch.cat(predictions).cpu().numpy()
```

#### 优化5: 使用torch.compile（PyTorch 2.0+）
```python
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode='max-autotune')
```

#### 优化6: 优化optimizer.zero_grad()
```python
optimizer.zero_grad(set_to_none=True)  # 更高效
```

### 2.2 进阶优化（中优先级）

#### 优化7: 启用TensorFloat-32
```python
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
```

#### 优化8: 使用梯度累积（如果内存不足）
```python
accumulation_steps = 4  # 累积4个batch的梯度
effective_batch_size = BATCH_SIZE * accumulation_steps
```

#### 优化9: 增加序列长度
```python
SEQUENCE_LENGTHS = [30, 60, 90]  # 增加序列长度以提升计算密度
```

### 2.3 监控与调试

#### 添加GPU利用率监控
```python
import subprocess
def get_gpu_utilization():
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    return int(result.stdout.strip())
```

## 三、预期效果

### 优化前
- GPU利用率: ~20-40%
- 训练速度: ~500-1000 samples/s
- Batch处理时间: ~50-100ms/batch

### 优化后（预期）
- GPU利用率: ~80-95%
- 训练速度: ~3000-5000 samples/s（提升3-5倍）
- Batch处理时间: ~10-20ms/batch（提升5倍）

## 四、实施建议

1. **逐步优化**: 先实施高优先级优化，观察效果后再进行进阶优化
2. **监控GPU**: 使用`nvidia-smi -l 1`实时监控GPU利用率
3. **内存管理**: 增大batch size时注意GPU内存限制
4. **性能测试**: 每次优化后记录训练速度，对比效果

