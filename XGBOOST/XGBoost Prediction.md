# XGBoost PM2.5预测系统

## 系统功能概述

### 1. 数据预处理
- 加载CSV文件
- 处理时间特征
- 创建滚动平均等统计特征

### 2. 特征工程
- 提取时间相关特征
- 计算历史PM2.5数据的统计特征
- 处理缺失值

### 3. 模型训练
- 使用XGBoost回归器
- 实现早停机制
- 支持交叉验证

### 4. 模型评估
- 计算RMSE和R²分数
- 可视化预测结果
- 展示特征重要性

## 使用指南

### 环境配置
```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn
```

### 使用建议
1. 根据实际数据格式调整`load_and_preprocess_data`函数中的数据读取和处理逻辑

2. 优化XGBoost超参数提升模型性能：
   - max_depth
   - learning_rate
   - n_estimators
   - subsample
   - colsample_bytree

3. 可添加的增强特征：
   - 温度、湿度等气象数据
   - 空气质量指数（AQI）
   - 风向风速数据
   - 地理位置特征

4. 建议添加交叉验证以更准确评估模型性能

此实现提供基础框架，可根据具体需求进行调整和扩展。
