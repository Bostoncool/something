# 北京PM2.5浓度Prophet时序预测模型

## 概述

本项目基于Facebook（现Meta）开发的Prophet时序预测模型，实现对北京PM2.5浓度的时序预测分析。Prophet是一种专门设计用于处理具有趋势、季节性和节假日效应的时序数据的预测模型。

## 主要特点

- **自动季节性处理**：自动检测和建模年季节性、周季节性和日季节性
- **趋势变化检测**：自动识别数据中的趋势变化点
- **缺失数据处理**：能够自动处理时序数据中的缺失值
- **预测置信区间**：提供预测结果的置信区间
- **高解释性**：模型结果易于理解和解释

## 数据来源

- **污染数据**：北京地区空气质量监测数据（AQI、PM2.5、PM10、SO2、NO2、CO、O3）
- **气象数据**：ERA5再分析数据（温度、湿度、风速、气压等气象变量）
- **时间范围**：2015年1月1日至2024年12月31日
- **地理范围**：北京地区（39.0°-41.25°N，115.0°-117.25°E）

## 模型配置

### Prophet参数设置
- `seasonality_mode`: 'multiplicative' - 乘法季节性模式
- `yearly_seasonality`: True - 启用年季节性
- `weekly_seasonality`: True - 启用周季节性
- `daily_seasonality`: True - 启用日季节性
- `changepoint_prior_scale`: 0.05 - 趋势变化点先验尺度
- `seasonality_prior_scale`: 10.0 - 季节性先验尺度

### 交叉验证设置
- `initial`: '730 days' - 初始训练期（2年）
- `period`: '180 days' - 预测间隔（6个月）
- `horizon`: '365 days' - 预测范围（1年）

## 文件结构

```
Prophet/
├── Prophet.py                 # 主程序文件
├── README.md                 # 说明文档
├── output/                   # 输出结果目录
│   ├── prophet_forecast.png           # 预测结果图
│   ├── prophet_components.png         # 季节性分解图
│   ├── prophet_scatter.png            # 预测vs实际散点图
│   ├── prophet_cv_rmse.png            # 交叉验证性能图
│   ├── prophet_timeseries.png         # 时间序列对比图
│   ├── prophet_forecast_results.csv   # 预测结果数据
│   ├── prophet_performance.csv        # 模型性能指标
│   ├── prophet_cross_validation.csv   # 交叉验证结果
│   └── ...
└── models/                   # 模型保存目录
    └── prophet_pm25_model.pkl        # 保存的Prophet模型
```

## 使用方法

### 环境要求
- Python 3.7+
- prophet >= 1.0
- pandas >= 1.0
- numpy >= 1.18
- matplotlib >= 3.0
- seaborn >= 0.11
- xarray >= 0.16
- netCDF4 >= 1.5

### 安装依赖
```bash
pip install prophet pandas numpy matplotlib seaborn xarray netCDF4
```

### 运行程序
```bash
python Prophet.py
```

## 输出结果

### 可视化图表
1. **预测结果图** (`prophet_forecast.png`)：完整的时序预测结果，包含历史数据和未来预测
2. **季节性分解图** (`prophet_components.png`)：展示趋势、年季节性、周季节性等组件
3. **预测vs实际散点图** (`prophet_scatter.png`)：模型预测精度可视化
4. **交叉验证性能图** (`prophet_cv_rmse.png`)：不同预测时长的RMSE性能
5. **时间序列对比图** (`prophet_timeseries.png`)：最近一年的详细预测对比

### 数据文件
1. **预测结果** (`prophet_forecast_results.csv`)：包含预测值、置信区间等完整结果
2. **模型性能** (`prophet_performance.csv`)：R²、RMSE、MAE、MAPE等指标
3. **交叉验证结果** (`prophet_cross_validation.csv`)：详细的交叉验证数据
4. **未来预测** (`prophet_future_predictions.csv`)：仅包含未来365天的预测结果

## 模型评估

### 性能指标
- **R² Score**：决定系数，衡量模型解释方差的比例
- **RMSE**：均方根误差，单位为μg/m³
- **MAE**：平均绝对误差，单位为μg/m³
- **MAPE**：平均绝对百分比误差，反映预测误差的相对大小

### 交叉验证
使用时间序列交叉验证方法，评估模型在不同时间段的预测性能。

## Prophet模型优势

1. **自动化**：自动检测季节性和趋势，无需手动特征工程
2. **鲁棒性**：对异常值和缺失数据具有较好的鲁棒性
3. **可解释性**：清晰展示趋势、季节性和其他组件的影响
4. **灵活性**：可以轻松添加节假日效应、自定义季节性等
5. **预测区间**：提供预测的不确定性区间

## 注意事项

1. **数据路径**：需要根据实际情况修改数据文件路径
2. **内存使用**：处理大量ERA5数据时需要足够的内存
3. **计算时间**：交叉验证和大规模数据处理可能需要较长时间
4. **季节性假设**：假设数据具有稳定的季节性模式

## 扩展应用

该框架可以轻松扩展到其他空气污染物或气象变量的预测：
- PM10、SO2、NO2、CO、O3等污染物
- 温度、湿度、风速等气象变量
- 不同城市的空气质量预测
- 多变量联合预测

## 参考文献

- Taylor, S. J. and Letham, B. (2018). Forecasting at scale. The American Statistician, 72(1), 37-45.
- Prophet documentation: https://facebook.github.io/prophet/
