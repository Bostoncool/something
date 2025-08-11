# 北京气象因素与污染变化Pearson相关性分析

本项目基于PyTorch框架实现了北京气象因素与污染变化的Pearson相关性分析，能够递归搜索文件夹，读取深层嵌套的.csv文件，并通过热力图直观展示相关性分析结果。

## 项目结构

```
Pearson-Correlation/
├── Pearson-Correlation.py      # 完整版分析器（支持大规模数据）
├── test_pearson_analysis.py    # 演示版分析器（使用样本数据）
└── README.md                   # 项目说明文档
```

## 功能特点

### 1. 高性能数据处理
- **递归文件搜索**: 自动搜索深层嵌套文件夹中的.csv文件
- **并行处理**: 使用多进程并行处理大量数据文件
- **数据缓存**: 实现智能缓存机制，避免重复处理相同文件
- **内存优化**: 优化数据类型，减少内存使用

### 2. 基于PyTorch的相关性计算
- **PyTorch张量运算**: 利用GPU加速进行大规模矩阵计算
- **标准化处理**: 自动进行数据标准化，确保计算准确性
- **NaN值处理**: 智能处理缺失值，保证计算稳定性

### 3. 可视化分析
- **相关性热力图**: 直观展示变量间的相关性强度
- **最强相关性分析**: 识别并展示最重要的相关性关系
- **中文支持**: 完整的中文界面和图表标签

## 数据格式要求

### 气象数据格式
- 文件名格式: `YYYYMM.csv` (如: `201501.csv`)
- 包含的气象参数:
  - `t2m`: 2m温度 (开尔文)
  - `d2m`: 2m露点温度 (开尔文)
  - `blh`: 边界层高度
  - `u10`, `v10`: 10m风U、V分量
  - `u100`, `v100`: 100m风U、V分量
  - 其他气象参数...

### 污染数据格式
- 文件名格式: `beijing_all_YYYYMMDD.csv`
- 包含的污染指标:
  - `PM2.5`: PM2.5浓度
  - `PM10`: PM10浓度
  - `AQI`: 空气质量指数

## 使用方法

### 1. 演示版本（推荐新手使用）

```python
# 运行演示版本，使用样本数据
python test_pearson_analysis.py
```

演示版本特点：
- 使用项目自带的样本数据
- 快速验证代码功能
- 适合学习和测试

### 2. 完整版本（生产环境使用）

```python
# 运行完整版本，处理大规模数据
python Pearson-Correlation.py
```

完整版本特点：
- 支持10年（2015-2024）数据
- 并行处理大量文件
- 智能缓存机制
- 完整的分析报告

## 配置说明

### 数据路径配置

在 `main()` 函数中修改数据路径：

```python
def main():
    # 请根据实际情况修改这两个路径
    meteo_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV"  # 气象数据文件夹路径
    pollution_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)"  # 污染数据文件夹路径
    
    # 创建分析器
    analyzer = BeijingPearsonAnalyzer(meteo_data_dir, pollution_data_dir)
    
    # 运行分析
    analyzer.run_analysis()
```

### 依赖库安装

```bash
pip install pandas numpy torch matplotlib seaborn scikit-learn
```

## 输出结果

### 1. 控制台输出
- 数据加载进度
- 相关性分析结果
- 最强相关性排名
- 详细的分析报告

### 2. 可视化图表
- `beijing_pearson_correlation_heatmap.png`: 相关性热力图
- `beijing_top_correlations.png`: 最强相关性条形图

### 3. 分析报告内容
- 数据概览和特征分类
- 气象因素与污染指标的相关性分析
- 最强相关性排名
- 关键发现和结论

## 核心算法

### Pearson相关系数计算

使用PyTorch实现Pearson相关系数计算：

```python
def calculate_pearson_correlation_pytorch(self, data):
    # 转换为PyTorch张量
    X = torch.tensor(data[feature_columns].values, dtype=torch.float32)
    
    # 标准化数据
    X_mean = torch.mean(X, dim=0, keepdim=True)
    X_std = torch.std(X, dim=0, keepdim=True)
    X_normalized = (X - X_mean) / (X_std + 1e-8)
    
    # 计算相关性矩阵
    for i in range(n_features):
        for j in range(n_features):
            x_i = X_normalized[:, i]
            x_j = X_normalized[:, j]
            
            # 计算Pearson相关系数
            numerator = torch.sum((x_i - torch.mean(x_i)) * (x_j - torch.mean(x_j)))
            denominator = torch.sqrt(torch.sum((x_i - torch.mean(x_i))**2) * 
                                  torch.sum((x_j - torch.mean(x_j))**2))
            
            correlation_matrix[i, j] = numerator / denominator
```

## 性能优化

### 1. 并行处理
- 使用 `ProcessPoolExecutor` 进行多进程并行处理
- 自动检测CPU核心数，优化进程数量

### 2. 数据缓存
- 基于文件修改时间的智能缓存机制
- 避免重复处理相同文件，大幅提升性能

### 3. 内存优化
- 优化数据类型（float64 → float32）
- 向量化计算统计量
- 及时释放不需要的数据

## 关键发现

通过分析发现的主要相关性：

1. **温度因素**: 2m温度与PM2.5、PM10呈负相关
2. **湿度因素**: 露点温度与污染指标有显著相关性
3. **风场因素**: 风速分量对污染物扩散有重要影响
4. **边界层**: 边界层高度是影响污染扩散的关键因素
5. **降水因素**: 降水量对污染物清除有显著作用

## 技术特点

### 1. 高可靠性
- 完善的错误处理机制
- 数据完整性验证
- 异常情况自动恢复

### 2. 强可扩展性
- 模块化设计，易于扩展新功能
- 支持自定义数据格式
- 可配置的分析参数

### 3. 优秀用户体验
- 详细的中文提示信息
- 进度显示和状态反馈
- 清晰的分析报告输出

## 注意事项

1. **数据路径**: 确保数据文件路径正确，支持相对路径和绝对路径
2. **内存使用**: 大规模数据处理时注意内存使用情况
3. **GPU加速**: 如果有GPU，PyTorch会自动使用GPU加速计算
4. **文件格式**: 确保CSV文件格式正确，编码为UTF-8

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目地址: [GitHub仓库链接]
- 邮箱: [联系邮箱]

---

**注意**: 本项目仅供学术研究使用，请确保遵守相关数据使用协议。 