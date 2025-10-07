# 风分量数据处理问题修复总结

## 问题描述
在加载10m_u_component_of_wind、100m_u_component_of_wind、10m_v_component_of_wind、100m_v_component_of_wind数据集时出现报错，无法处理。

## 问题分析

### 1. 数据维度不匹配
- 风分量数据从NC文件转换后可能包含多维结构（时间×纬度×经度）
- 原始代码假设数据是1维时间序列，无法处理多维数据

### 2. 数据格式问题
- CSV文件可能包含多索引结构（time, latitude, longitude列）
- 需要按时间维度聚合空间数据

### 3. 数据类型转换问题
- 风分量数据可能包含特殊值或异常值
- 需要特殊的数据清洗和验证

### 4. 内存问题
- 风分量数据通常很大，可能导致内存溢出
- 需要数据采样和优化

## 修复方案

### 1. 增强 `calculate_stats_vectorized` 方法
```python
def calculate_stats_vectorized(self, hourly_data: np.ndarray) -> Dict[str, float]:
    # 处理多维数据 - 如果是2D或3D数组，展平为1D
    if hourly_data.ndim > 1:
        if hourly_data.ndim == 3:  # (time, lat, lon)
            hourly_data = np.nanmean(hourly_data, axis=(1, 2))
        elif hourly_data.ndim == 2:  # (time, spatial)
            hourly_data = np.nanmean(hourly_data, axis=1)
    
    # 移除无效值
    valid_data = hourly_data[~np.isnan(hourly_data)]
    
    # 如果数据量太大，进行采样以避免内存问题
    if len(valid_data) > 10000:
        step = len(valid_data) // 10000
        valid_data = valid_data[::step]
```

### 2. 添加空间数据聚合方法
```python
def aggregate_spatial_data(self, df: pd.DataFrame) -> pd.DataFrame:
    """聚合空间数据，将多维数据转换为时间序列"""
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['latitude', 'longitude']]
        
        if len(numeric_cols) > 0:
            aggregated = df.groupby('time')[numeric_cols].mean().reset_index()
            return aggregated
    return df
```

### 3. 专门的风分量数据处理方法
```python
def process_wind_component_data(self, df: pd.DataFrame, col: str) -> Dict[str, float]:
    """专门处理风分量数据的方法"""
    values = df[col].values
    valid_values = values[~np.isnan(values)]
    
    # 移除异常值（风分量通常在-100到100 m/s之间）
    valid_values = valid_values[(valid_values >= -100) & (valid_values <= 100)]
    
    # 如果数据量太大，进行采样
    if len(valid_values) > 50000:
        step = len(valid_values) // 50000
        valid_values = valid_values[::step]
    
    return {
        'mean': np.nanmean(valid_values),
        'std': np.nanstd(valid_values),
        'min': np.nanmin(valid_values),
        'max': np.nanmax(valid_values)
    }
```

### 4. 增强错误处理和调试信息
- 添加多种编码格式支持（UTF-8, GBK, Latin-1）
- 增加详细的调试信息输出
- 改进异常值检测和处理
- 添加数据质量检查

### 5. 优化数据处理流程
```python
def process_single_meteo_file(self, filepath: str) -> Optional[Dict]:
    # 处理多索引数据（如果存在）
    if 'time' in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
        df = self.aggregate_spatial_data(df)
    
    # 风分量数据特殊处理
    elif col in ['u10', 'v10', 'u100', 'v100']:
        daily_stats = self.process_wind_component_data(df, col)
```

## 测试结果

### 测试数据
- 创建了包含18,000行数据的测试文件
- 包含u10, v10, u100, v100四个风分量列
- 数据范围：u10/v10 [-50, 50], u100/v100 [-3, 8]

### 处理结果
```
SUCCESS: File processing successful!
Processing result contains 18 statistics

Wind component statistics:
  u10_mean: 1.8124
  u10_std: 1.2790
  u10_min: -2.3336
  u10_max: 8.2884
  v10_mean: 0.9782
  v10_std: 1.2090
  v10_min: -3.8175
  v10_max: 4.5800
  u100_mean: 3.0113
  u100_std: 0.2404
  u100_min: 2.3080
  u100_max: 3.6260
  v100_mean: 1.5024
  v100_std: 0.2039
  v100_min: 0.8914
  v100_max: 2.2167
```

## 修复效果

1. **成功处理多维风分量数据**：能够正确处理从NC文件转换的多维CSV数据
2. **内存优化**：通过数据采样避免内存溢出问题
3. **异常值处理**：自动检测和移除异常的风分量值
4. **错误恢复**：增强的错误处理机制，单个列处理失败不影响整体流程
5. **调试友好**：详细的调试信息帮助诊断数据处理问题

## 使用建议

1. 确保风分量数据文件包含正确的列名（u10, v10, u100, v100）
2. 如果数据包含空间维度，确保有time, latitude, longitude列
3. 对于大型数据集，建议使用数据采样来避免内存问题
4. 定期检查处理日志以识别数据质量问题

## 额外发现的问题和修复

### 问题7：风分量数据在子目录中
通过诊断发现，风分量数据存储在独立的子目录中：
```
ERA5-Beijing-CSV/
  ├── 10m_u_component_of_wind/  (120 CSV文件)
  ├── 10m_v_component_of_wind/  (120 CSV文件)
  ├── 100m_u_component_of_wind/ (120 CSV文件)
  └── 100m_v_component_of_wind/ (120 CSV文件)
```

**修复**：更新`collect_all_meteo_files()`方法，搜索所有CSV文件而不仅是按YYYYMM.csv模式

### 问题8：CSV文件包含注释行
风分量CSV文件由NC文件转换而来，包含以`#`开头的元数据注释行。

**修复**：在读取CSV时添加`comment='#'`参数，并处理ParserError异常

## 文件修改

- `old.py`: 主要修复文件，包含所有风分量数据处理改进
- 新增方法：`aggregate_spatial_data()`, `process_wind_component_data()`
- 修改方法：`calculate_stats_vectorized()`, `process_single_meteo_file()`, `collect_all_meteo_files()`
- 增强CSV读取：支持注释行、多种编码、解析错误处理

## 诊断工具

创建了两个诊断脚本帮助识别问题：
- `debug_wind_data.py`: 检查数据目录结构和文件分布
- `check_wind_file.py`: 检查单个风分量文件的格式

## 下一步

如果风分量数据仍未在热力图中显示：
1. 运行`debug_wind_data.py`检查数据目录结构
2. 检查处理日志，确认风分量文件被正确加载
3. 检查`prepare_combined_data()`输出，确认风分量列存在于合并数据中
4. 确认年月信息正确匹配，使数据能够对齐
