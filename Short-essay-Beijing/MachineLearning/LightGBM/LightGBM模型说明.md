# LightGBM PM2.5浓度预测模型

## 模型简介

LightGBM (Light Gradient Boosting Machine) 是微软开发的高效梯度提升决策树框架，具有以下优势：

### 主要特点

1. **训练速度快** - 使用基于直方图的算法，比传统GBDT快10-20倍
2. **内存消耗低** - 直方图算法减少内存占用
3. **准确率高** - 使用leaf-wise生长策略，在相同迭代次数下通常比level-wise准确
4. **支持并行学习** - 特征并行、数据并行、投票并行
5. **处理大规模数据** - 可以处理百万级别的数据和特征
6. **支持类别特征** - 直接处理类别特征，无需one-hot编码

### 与其他模型对比

| 模型 | 训练速度 | 预测精度 | 内存占用 | 可解释性 | 参数调优难度 |
|------|---------|---------|----------|----------|-------------|
| 线性回归(MLR) | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| 随机森林(RF) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| XGBoost | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **LightGBM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

## 文件说明

### 主要文件

- `LightGBM_PM25.py` - 完整版本，使用真实数据进行训练和预测
- `LightGBM_PM25_Simple.py` - 简化版本，使用模拟数据快速测试
- `LightGBM模型说明.md` - 本说明文档

### 输出文件

运行后会在以下目录生成结果：

#### output/ 目录
- `model_performance.csv` - 模型性能指标
- `feature_importance.csv` - 特征重要性
- `best_parameters.csv` - 最佳超参数
- `predictions.csv` - 预测结果
- `*.png` - 各种可视化图表

#### models/ 目录
- `lightgbm_optimized.txt` - 模型文件（文本格式）
- `lightgbm_optimized.pkl` - 模型文件（pickle格式）

## 环境配置

### 必需库

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm
```

### 可选库（用于贝叶斯优化）

```bash
pip install bayesian-optimization
```

## 数据要求

### 输入数据

1. **污染数据**
   - 路径: `C:\Users\IU\Desktop\Datebase Origin\Benchmark\`
   - 格式: CSV文件，按日期命名 (YYYYMMDD)
   - 变量: PM2.5, PM10, SO2, NO2, CO, O3

2. **气象数据**
   - 路径: `C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV\`
   - 格式: CSV文件，按年月命名 (YYYYMM.csv)
   - 变量: t2m, d2m, u10, v10, u100, v100, blh, sp, tcwv, tp等

### 特征工程

模型自动创建以下特征：

1. **风速特征**
   - 10米风速和风向
   - 100米风速和风向

2. **时间特征**
   - 年、月、日、星期几
   - 一年中的第几天、第几周
   - 季节、是否供暖季

3. **滞后特征**
   - PM2.5的1天、3天、7天滞后值
   - PM2.5的3天、7天、30天移动平均

4. **衍生特征**
   - 温度-露点差（相对湿度指标）
   - 相对湿度估算
   - 风向分类（8个方位）

## 使用方法

### 1. 快速开始（模拟数据）

```bash
python LightGBM_PM25_Simple.py
```

适用场景：
- 快速测试代码逻辑
- 了解模型工作原理
- 无需等待长时间数据加载

### 2. 完整训练（真实数据）

```bash
python LightGBM_PM25.py
```

运行时间：
- 数据加载: 5-10分钟
- 模型训练: 2-5分钟
- 总计: 约10-15分钟

## 模型参数

### 核心参数

| 参数 | 说明 | 默认值 | 调优范围 |
|------|------|--------|---------|
| `num_leaves` | 叶子节点数 | 31 | 20-100 |
| `max_depth` | 树的最大深度 | 7 | 3-12 |
| `learning_rate` | 学习率 | 0.05 | 0.01-0.1 |
| `feature_fraction` | 特征采样比例 | 0.8 | 0.5-1.0 |
| `bagging_fraction` | 样本采样比例 | 0.8 | 0.5-1.0 |
| `min_child_samples` | 叶子节点最小样本数 | 20 | 10-50 |

### 参数调优建议

1. **防止过拟合**
   - 降低 `num_leaves` 和 `max_depth`
   - 增加 `min_child_samples`
   - 降低 `learning_rate`，增加 `num_boost_round`

2. **提高精度**
   - 增加 `num_boost_round`
   - 调整 `feature_fraction` 和 `bagging_fraction`
   - 使用早停机制

3. **加快训练**
   - 减少 `num_boost_round`
   - 增加 `learning_rate`
   - 降低 `max_depth`

## 超参数优化

### 贝叶斯优化（推荐）

如果安装了 `bayesian-optimization`，代码会自动使用贝叶斯优化：

```python
# 自动搜索最佳参数
optimizer.maximize(init_points=5, n_iter=15)
```

优势：
- 智能搜索，效率高
- 需要更少的迭代次数
- 可以找到更好的参数组合

### 网格搜索（备选）

如果未安装贝叶斯优化库，使用网格搜索：

```python
param_grid = {
    'num_leaves': [31, 50, 70],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.03, 0.05, 0.07],
    'feature_fraction': [0.7, 0.8, 0.9],
}
```

## 模型评估指标

### 1. R² (决定系数)
- 范围: 0-1（越大越好）
- 含义: 模型解释的方差比例
- 优秀: > 0.8, 良好: 0.6-0.8, 一般: 0.4-0.6

### 2. RMSE (均方根误差)
- 单位: μg/m³（越小越好）
- 含义: 预测值与实际值的平均偏差
- 优秀: < 15, 良好: 15-25, 一般: 25-40

### 3. MAE (平均绝对误差)
- 单位: μg/m³（越小越好）
- 含义: 预测误差的平均绝对值
- 对异常值不敏感

### 4. MAPE (平均绝对百分比误差)
- 单位: %（越小越好）
- 含义: 相对误差的百分比
- 优秀: < 15%, 良好: 15-25%, 一般: 25-40%

## 特征重要性

### 两种重要性度量

1. **Split（分裂次数）**
   - 特征在树中被用作分裂的次数
   - 反映特征使用频率

2. **Gain（信息增益）**
   - 特征对模型性能的实际贡献
   - 更能反映特征的真实重要性（推荐）

### 预期重要特征

根据PM2.5形成机理，通常最重要的特征包括：

1. **滞后特征** - PM2.5_lag1, PM2.5_lag3, PM2.5_ma3
2. **温度** - t2m, d2m, temp_dewpoint_diff
3. **风速** - wind_speed_10m, u10, v10
4. **湿度** - tcwv, relative_humidity
5. **边界层高度** - blh
6. **时间特征** - month, season, is_heating_season

## 可视化图表

### 1. 训练过程曲线
- 展示训练集和验证集的RMSE变化
- 帮助判断是否过拟合
- 显示最佳迭代次数

### 2. 预测vs实际散点图
- 对比预测值和真实值
- 越接近对角线表示预测越准确
- 分训练集、验证集、测试集

### 3. 时间序列对比
- 展示预测曲线和实际曲线
- 直观展示模型跟踪能力
- 显示最后300天的预测效果

### 4. 残差分析
- 残差应随机分布在0附近
- 如果有明显模式，说明模型还可改进
- 帮助识别系统性偏差

### 5. 特征重要性图
- 展示Top 20重要特征
- 按Split和Gain两种方式排序
- 指导特征选择和工程

### 6. 模型性能对比
- 对比基础模型和优化模型
- 展示R²、RMSE、MAE、MAPE
- 量化优化效果

### 7. 误差分布
- 展示预测误差的分布
- 应呈正态分布
- 均值应接近0

## 模型使用

### 加载已保存的模型

```python
import lightgbm as lgb
import pickle

# 方法1: 加载文本格式
model = lgb.Booster(model_file='models/lightgbm_optimized.txt')

# 方法2: 加载pickle格式
with open('models/lightgbm_optimized.pkl', 'rb') as f:
    model = pickle.load(f)
```

### 进行预测

```python
# 准备新数据（特征必须与训练时一致）
X_new = prepare_new_data()  # 你的数据准备函数

# 预测
predictions = model.predict(X_new, num_iteration=model.best_iteration)

# 预测结果为PM2.5浓度（μg/m³）
print(f"预测的PM2.5浓度: {predictions}")
```

## 常见问题

### Q1: 为什么需要验证集？

A: 验证集用于：
- 早停机制，防止过拟合
- 超参数调优
- 监控训练过程

### Q2: 如何处理数据不足的情况？

A: 可以：
- 减少验证集比例
- 使用交叉验证
- 减少滞后特征（减少样本损失）

### Q3: 训练时间太长怎么办？

A: 可以：
- 减少 `num_boost_round`
- 增加 `learning_rate`
- 减少数据量（按月或季度采样）
- 减少超参数搜索空间

### Q4: 如何提高模型精度？

A: 可以尝试：
- 增加更多特征（如其他污染物）
- 调整超参数
- 增加训练数据
- 特征工程（交互特征、多项式特征）
- 集成多个模型

### Q5: 模型在某些时段预测不准？

A: 可能原因：
- 极端天气事件（数据分布外）
- 人为排放变化（如政策影响）
- 特殊时期（如春节、重大活动）
- 建议：添加事件标记特征

## 进阶应用

### 1. 多步预测

```python
# 预测未来7天
predictions_7day = []
X_current = X_test.iloc[-1:].copy()

for i in range(7):
    pred = model.predict(X_current)
    predictions_7day.append(pred[0])
    
    # 更新滞后特征
    X_current['PM2.5_lag1'] = pred[0]
    # ... 更新其他特征
```

### 2. 不确定性估计

```python
# 使用分位数回归
model_lower = lgb.train(params, train_data, objective='quantile', alpha=0.1)
model_upper = lgb.train(params, train_data, objective='quantile', alpha=0.9)

# 预测区间
pred_lower = model_lower.predict(X_test)
pred_upper = model_upper.predict(X_test)
```

### 3. 特征选择

```python
# 基于重要性筛选特征
importance_threshold = 1.0  # 1%
important_features = feature_importance[
    feature_importance['Importance_Gain_Norm'] > importance_threshold
]['Feature'].tolist()

# 重新训练
X_train_selected = X_train[important_features]
```

## 参考资料

### LightGBM官方文档
- GitHub: https://github.com/microsoft/LightGBM
- 文档: https://lightgbm.readthedocs.io/

### 相关论文
1. Ke et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
2. Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"

### 推荐阅读
- LightGBM调参指南
- 梯度提升决策树原理
- 时间序列特征工程

## 更新日志

### v1.0 (2024)
- 初始版本
- 支持基础LightGBM训练
- 贝叶斯优化超参数
- 完整的特征工程
- 丰富的可视化

## 联系方式

如有问题或建议，请通过以下方式联系：
- 在代码仓库提Issue
- 查看相关文档和示例

---

**祝你使用愉快！** 🚀

