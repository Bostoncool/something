# LightGBM PM2.5浓度预测

## 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm
```

可选（用于贝叶斯优化）：
```bash
pip install bayesian-optimization
```

### 2. 运行代码

#### 方式1：快速测试（推荐首次使用）

使用模拟数据，运行时间约1-2分钟：

```bash
python LightGBM_PM25_Simple.py
```

#### 方式2：完整训练

使用真实数据，运行时间约10-15分钟：

```bash
python LightGBM_PM25.py
```

### 3. 查看结果

#### 输出目录结构

```
LightGBM/
├── output/                          # 完整版输出
│   ├── model_performance.csv
│   ├── feature_importance.csv
│   ├── best_parameters.csv
│   ├── predictions.csv
│   └── *.png (各种图表)
├── output_simple/                   # 简化版输出
│   └── (与output类似的文件)
└── models/                          # 保存的模型
    ├── lightgbm_optimized.txt
    └── lightgbm_optimized.pkl
```

#### 关键图表说明

1. **training_curves.png** - 训练过程曲线
   - 查看模型是否过拟合
   - 确认早停机制是否有效

2. **prediction_scatter.png** - 预测vs实际散点图
   - 评估预测准确度
   - 点越接近对角线越好

3. **timeseries_comparison.png** - 时间序列对比
   - 查看预测趋势是否符合实际
   - 识别模型在哪些时段表现较好/较差

4. **feature_importance.png** - 特征重要性
   - 了解哪些特征对预测最重要
   - 指导后续特征工程

5. **model_comparison.png** - 模型性能对比
   - 比较基础模型和优化模型
   - 量化优化效果

## 文件说明

| 文件 | 说明 | 使用场景 |
|------|------|---------|
| `LightGBM_PM25.py` | 完整版，使用真实数据 | 正式训练和预测 |
| `LightGBM_PM25_Simple.py` | 简化版，使用模拟数据 | 快速测试、学习原理 |
| `LightGBM模型说明.md` | 详细文档 | 深入学习、问题排查 |
| `README.md` | 本文件 | 快速开始 |

## 预期结果

### 模型性能（模拟数据）

| 指标 | 基础模型 | 优化模型 |
|------|---------|---------|
| R² | 0.85-0.90 | 0.90-0.95 |
| RMSE | 10-15 μg/m³ | 8-12 μg/m³ |
| MAE | 7-10 μg/m³ | 6-9 μg/m³ |
| MAPE | 15-20% | 12-18% |

### 重要特征（预期）

1. PM2.5滞后特征（lag1, lag3, ma3）
2. 温度相关（t2m, d2m）
3. 风速（wind_speed_10m）
4. 时间特征（month, season, is_heating_season）
5. 边界层高度（blh）

## 常见问题

### Q1: 找不到数据文件

**问题**: `FileNotFoundError: 找不到数据文件`

**解决方案**:
1. 检查数据路径是否正确
2. 确认数据文件是否存在
3. 首次运行建议使用简化版（`LightGBM_PM25_Simple.py`）

### Q2: 内存不足

**问题**: `MemoryError`

**解决方案**:
1. 减少数据量（按月或季度采样）
2. 减少特征数量
3. 减少 `num_boost_round`
4. 关闭其他程序释放内存

### Q3: 训练时间太长

**问题**: 训练时间超过30分钟

**解决方案**:
1. 减少 `num_boost_round`
2. 增加 `learning_rate`
3. 简化超参数搜索空间
4. 使用更少的数据

### Q4: 模型精度不理想

**问题**: R² < 0.5 或 RMSE > 30

**解决方案**:
1. 检查数据质量（是否有异常值）
2. 增加更多特征
3. 调整超参数
4. 增加训练数据量
5. 查看特征重要性，移除不重要特征

### Q5: 运行报错

**问题**: 各种运行时错误

**解决方案**:
1. 确认所有依赖库已安装
2. 检查Python版本（建议3.8+）
3. 查看错误提示的具体信息
4. 参考 `LightGBM模型说明.md` 中的常见问题部分

## 下一步

### 改进模型

1. **特征工程**
   ```python
   # 添加更多交互特征
   df['temp_wind_interaction'] = df['t2m'] * df['wind_speed_10m']
   df['humidity_blh_interaction'] = df['relative_humidity'] * df['blh']
   ```

2. **集成学习**
   ```python
   # 集成多个模型
   from sklearn.ensemble import VotingRegressor
   ensemble = VotingRegressor([
       ('lgb', lgb_model),
       ('rf', rf_model),
       ('xgb', xgb_model)
   ])
   ```

3. **多步预测**
   ```python
   # 预测未来7天
   for i in range(7):
       pred = model.predict(X_current)
       # 更新特征...
   ```

### 深入学习

1. 阅读 `LightGBM模型说明.md` 了解更多细节
2. 学习LightGBM官方文档
3. 尝试不同的参数组合
4. 比较与其他模型（RF, XGBoost）的性能

## 技术支持

### 相关文档

- [LightGBM官方文档](https://lightgbm.readthedocs.io/)
- [LightGBM GitHub](https://github.com/microsoft/LightGBM)
- [Scikit-learn文档](https://scikit-learn.org/)

### 学习资源

- LightGBM调参指南
- 时间序列预测最佳实践
- 特征工程技巧

---

**祝你使用愉快！** 如有问题，请参考 `LightGBM模型说明.md` 或查阅官方文档。

