# 北京市多气象因素对污染变化影响的PCA分析

## 概述

本项目实现了基于多气象参数的北京市污染变化影响分析，使用主成分分析（PCA）方法探究气象因素与污染指标之间的复杂关系。

## 主要改进

### 1. 多气象参数支持

原版本只支持 `t2m`（2米温度）参数，现在支持以下17个气象参数：

| 参数代码 | 参数名称 | 物理意义 |
|---------|---------|---------|
| `t2m` | 2米温度 | 近地面温度 |
| `d2m` | 2米露点温度 | 近地面露点温度 |
| `blh` | 边界层高度 | 大气边界层厚度 |
| `cvh` | 高植被覆盖 | 高植被覆盖度 |
| `avg_tprate` | 平均总降水率 | 降水强度 |
| `u10` | 10米风速U分量 | 10米高度东西向风速 |
| `v10` | 10米风速V分量 | 10米高度南北向风速 |
| `u100` | 100米风速U分量 | 100米高度东西向风速 |
| `v100` | 100米风速V分量 | 100米高度南北向风速 |
| `lsm` | 陆海掩膜 | 地表类型标识 |
| `cvl` | 低植被覆盖 | 低植被覆盖度 |
| `mn2t` | 2米最低温度 | 日最低温度 |
| `sp` | 表面气压 | 地面气压 |
| `sd` | 雪深 | 积雪深度 |
| `str` | 表面净热辐射 | 地表净辐射 |
| `tisr` | 大气层顶入射太阳辐射 | 太阳辐射强度 |
| `tcwv` | 总水汽柱 | 大气水汽含量 |
| `tp` | 总降水 | 累计降水量 |

### 2. 数据处理优化

- **温度参数处理**：自动将开尔文温度转换为摄氏度
- **统计量计算**：为每个气象参数计算均值、标准差、最小值、最大值
- **数据质量检查**：自动过滤NaN值和异常数据
- **特征对齐**：确保气象数据和污染数据的时间对齐

### 3. 分析功能增强

#### PCA分析
- 支持多维度主成分分析
- 自动计算主成分贡献度
- 提供主成分物理意义解释

#### 相关性分析
- 多气象因素与污染指标的相关性分析
- 识别最重要的气象影响因素
- 提供相关性排序和可视化

#### 可视化改进
- 多维度相关性热力图
- 主成分贡献度分析图
- 特征重要性排序图

## 文件结构

```
PCA-Correlation/
├── PCA_Beijing_Analysis.py          # 主要分析类
├── diagnose_data_loading.py         # 数据加载诊断工具
├── test_multi_meteo_analysis.py    # 多气象参数测试
├── test_recursive_search.py         # 文件搜索测试
├── README_Multi_Meteo_PCA.md       # 本文档
├── 201501.csv                      # 示例气象数据
└── beijing_all_20150101.csv        # 示例污染数据
```

## 使用方法

### 1. 基本使用

```python
from PCA_Beijing_Analysis import BeijingPCAAnalyzer

# 创建分析器
analyzer = BeijingPCAAnalyzer(
    meteo_data_dir="path/to/meteo/data",
    pollution_data_dir="path/to/pollution/data"
)

# 运行完整分析
analyzer.run_analysis()
```

### 2. 分步使用

```python
# 1. 加载数据
analyzer.load_data()

# 2. 准备合并数据
combined_data = analyzer.prepare_combined_data()

# 3. 执行PCA分析
X_pca, feature_names, explained_variance_ratio = analyzer.perform_pca_analysis(combined_data)

# 4. 分析相关性
correlation_matrix = analyzer.analyze_correlations(combined_data)

# 5. 生成报告
analyzer.generate_analysis_report(combined_data, correlation_matrix, X_pca, feature_names, explained_variance_ratio)
```

### 3. 诊断工具

```python
# 运行数据加载诊断
python diagnose_data_loading.py

# 运行多气象参数测试
python test_multi_meteo_analysis.py
```

## 数据要求

### 气象数据格式
- 文件命名：`YYYYMM.csv`（如：`201501.csv`）
- 必需列：至少包含一个气象参数列
- 数据格式：小时级数据，包含时间戳和气象参数值

### 污染数据格式
- 文件命名：`beijing_all_YYYYMMDD.csv`
- 必需列：`type`列（标识数据类型：PM2.5、PM10、AQI）
- 数据格式：小时级污染监测数据

## 分析结果

### 1. 数据概览
- 气象参数数量：最多17个
- 污染指标数量：3个（PM2.5、PM10、AQI）
- 特征总数：最多68个（17个参数×4个统计量 + 3个污染指标×2个统计量）

### 2. 主要发现
- **多气象因素综合影响**：温度、湿度、风速等因素对污染水平有综合影响
- **边界层高度重要性**：边界层高度和大气稳定度是重要影响因素
- **扩散条件影响**：降水和风速对污染物扩散有显著作用
- **季节性变化**：不同季节的主要影响因素不同

### 3. 输出文件
- `beijing_correlation_heatmap.png`：相关性热力图
- `beijing_pca_results.png`：PCA分析结果图
- 控制台输出：详细的分析报告

## 优势对比

| 方面 | 原版本（单参数） | 新版本（多参数） |
|------|-----------------|-----------------|
| 气象参数数量 | 1个（t2m） | 17个 |
| 特征维度 | 4个 | 最多68个 |
| 分析深度 | 基础 | 全面 |
| 预测能力 | 有限 | 显著提升 |
| 物理意义 | 单一 | 丰富 |

## 注意事项

1. **数据路径**：确保数据文件夹路径正确
2. **文件格式**：严格按照命名规范命名文件
3. **数据质量**：检查数据完整性和异常值
4. **内存使用**：大量数据可能需要较多内存
5. **计算时间**：多参数分析会增加计算时间

## 故障排除

### 常见问题

1. **数据加载失败**
   - 检查文件路径是否正确
   - 确认文件命名格式
   - 验证数据文件完整性

2. **参数识别问题**
   - 检查列名是否与预期一致
   - 确认数据格式是否正确

3. **内存不足**
   - 减少数据量或分批处理
   - 增加系统内存

4. **NaN值过多**
   - 检查数据质量
   - 考虑数据预处理

### 诊断工具

使用提供的诊断工具来识别具体问题：
- `diagnose_data_loading.py`：检查数据加载问题
- `test_multi_meteo_analysis.py`：测试完整分析流程

## 更新日志

- **v2.0**：支持17个气象参数的多维度分析
- **v1.0**：基础单参数（t2m）分析

## 联系方式

如有问题或建议，请联系开发团队。 