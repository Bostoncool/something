# GPU优化快速参考指南

## 📊 优化前后对比

| 配置项 | 优化前 | 优化后 | 提升效果 |
|--------|--------|--------|----------|
| Batch Size | 128 | 256 | GPU利用率提升2倍 |
| NUM_WORKERS | 4 | 8 | 数据加载速度提升 |
| PREFETCH_FACTOR | 无 | 4 | 减少GPU等待时间 |
| hidden_size (基础) | 64 | 128 | 计算密度提升 |
| hidden_size (网格搜索) | [64,128] | [128,256] | 更大模型，更高利用率 |
| optimizer.zero_grad | 默认 | set_to_none=True | 性能提升5-10% |
| CPU-GPU同步 | 频繁 | 批量转换 | 减少等待时间 |
| TF32支持 | 未启用 | 已启用 | Ampere+ GPU速度提升1.2-1.5倍 |
| torch.compile | 未使用 | 已启用 | PyTorch 2.0+速度提升20-50% |

## 🎯 关键优化点

### 1. Batch Size（最重要）
```python
BATCH_SIZE = 256  # 根据GPU内存调整：128/192/256/512
```

### 2. DataLoader优化
```python
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
pin_memory = True
persistent_workers = True
```

### 3. 模型规模
```python
hidden_size = 128  # 基础模型
# 网格搜索: [128, 256]
```

### 4. 编译优化
```python
if USE_COMPILE and hasattr(torch, 'compile'):
    model = torch.compile(model, mode='max-autotune')
```

### 5. 减少同步
```python
# ❌ 错误：频繁同步
predictions.append(outputs.cpu().numpy())

# ✅ 正确：批量转换
predictions.append(outputs)  # 保持在GPU
predictions = torch.cat(predictions).cpu().numpy()  # 最后转换
```

## 🔍 监控GPU利用率

### 方法1: 命令行监控
```bash
watch -n 1 nvidia-smi
```

### 方法2: 代码中监控
训练过程中会自动显示GPU利用率（每10个epoch）

### 方法3: 使用函数
```python
gpu_util = get_gpu_utilization()
print_gpu_status("训练阶段")
```

## ⚠️ 常见问题

### Q1: GPU利用率仍然很低（<50%）
**解决方案**:
1. 增大`BATCH_SIZE`到512（如果内存允许）
2. 增大`hidden_size`到256或512
3. 检查数据加载是否成为瓶颈（观察CPU使用率）

### Q2: 内存不足（OOM）
**解决方案**:
1. 降低`BATCH_SIZE`到192或128
2. 降低`hidden_size`
3. 使用梯度累积（见进阶优化）

### Q3: torch.compile失败
**解决方案**:
- 检查PyTorch版本（需要2.0+）
- 或设置`USE_COMPILE = False`

### Q4: 训练速度没有提升
**检查清单**:
- [ ] GPU利用率是否提升（使用nvidia-smi）
- [ ] torch.compile是否生效（查看日志）
- [ ] Batch size是否足够大
- [ ] 数据加载是否成为瓶颈

## 📈 性能调优建议

### 根据GPU型号调整

#### NVIDIA A100 / H100
```python
BATCH_SIZE = 512
hidden_size = 256
NUM_WORKERS = 16
```

#### NVIDIA V100 / RTX 3090
```python
BATCH_SIZE = 256
hidden_size = 128
NUM_WORKERS = 8
```

#### NVIDIA RTX 2080 / 3080
```python
BATCH_SIZE = 128
hidden_size = 128
NUM_WORKERS = 4
```

### 根据数据大小调整

#### 大数据集（>100万样本）
- 增大`BATCH_SIZE`
- 增加`NUM_WORKERS`
- 使用更大的`prefetch_factor`

#### 小数据集（<10万样本）
- 可以适当减小`BATCH_SIZE`
- 减少`NUM_WORKERS`到4

## 🚀 进阶优化（可选）

### 梯度累积
如果GPU内存受限，使用梯度累积模拟更大的batch size：
```python
accumulation_steps = 4
effective_batch_size = BATCH_SIZE * accumulation_steps
```

### 增大序列长度
```python
SEQUENCE_LENGTHS = [30, 60, 90]  # 增加计算密度
```

### 使用多GPU（如果可用）
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## 📝 优化检查清单

运行优化后的代码前，确认：

- [x] Batch size已增大（256或更大）
- [x] NUM_WORKERS已增加（8或更多）
- [x] PREFETCH_FACTOR已设置（4）
- [x] hidden_size已增大（128或更大）
- [x] TF32已启用（Ampere+ GPU）
- [x] torch.compile已启用（PyTorch 2.0+）
- [x] optimizer.zero_grad使用set_to_none=True
- [x] CPU-GPU同步已优化（批量转换）

## 📊 预期性能提升

- **GPU利用率**: 20-40% → 70-90% ✅
- **训练速度**: 500-1000 → 2000-4000 samples/s ✅
- **Batch处理时间**: 50-100ms → 15-30ms ✅

## 🔗 相关文档

- `GPU_优化分析报告.md` - 详细的问题分析和解决方案
- `GPU优化实施总结.md` - 优化措施实施总结

