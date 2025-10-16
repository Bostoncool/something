# LightGBM PM2.5 Concentration Prediction Model

## Model Introduction

LightGBM (Light Gradient Boosting Machine) is an efficient gradient boosting decision tree framework developed by Microsoft with the following advantages:

### Main Features

1. **Fast Training Speed** - Uses histogram-based algorithm, 10-20x faster than traditional GBDT
2. **Low Memory Consumption** - Histogram algorithm reduces memory usage
3. **High Accuracy** - Uses leaf-wise growth strategy, usually more accurate than level-wise with same iteration count
4. **Supports Parallel Learning** - Feature parallel, data parallel, voting parallel
5. **Handles Large-scale Data** - Can handle millions of data points and features
6. **Supports Categorical Features** - Directly handles categorical features without one-hot encoding

### Comparison with Other Models

| Model | Training Speed | Prediction Accuracy | Memory Usage | Interpretability | Hyperparameter Tuning Difficulty |
|------|---------------|---------------------|--------------|------------------|----------------------------------|
| Linear Regression (MLR) | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| Random Forest (RF) | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| XGBoost | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **LightGBM** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

## File Description

### Main Files

- `LightGBM_PM25.py` - Full version, uses real data for training and prediction
- `LightGBM_PM25_Simple.py` - Simplified version, uses simulated data for quick testing
- `LightGBM_Model_Guide.md` - This documentation

### Output Files

After running, results will be generated in the following directories:

#### output/ directory
- `model_performance.csv` - Model performance metrics
- `feature_importance.csv` - Feature importance
- `best_parameters.csv` - Best hyperparameters
- `predictions.csv` - Prediction results
- `*.png` - Various visualization charts

#### models/ directory
- `lightgbm_optimized.txt` - Model file (text format)
- `lightgbm_optimized.pkl` - Model file (pickle format)

## Environment Configuration

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm
```

### Optional Libraries (for Bayesian optimization)

```bash
pip install bayesian-optimization
```

## Data Requirements

### Input Data

1. **Pollution Data**
   - Path: `C:\Users\IU\Desktop\Datebase Origin\Benchmark\`
   - Format: CSV files, named by date (YYYYMMDD)
   - Variables: PM2.5, PM10, SO2, NO2, CO, O3

2. **Meteorological Data**
   - Path: `C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV\`
   - Format: CSV files, named by year-month (YYYYMM.csv)
   - Variables: t2m, d2m, u10, v10, u100, v100, blh, sp, tcwv, tp, etc.

### Feature Engineering

The model automatically creates the following features:

1. **Wind Speed Features**
   - 10-meter wind speed and direction
   - 100-meter wind speed and direction

2. **Time Features**
   - Year, month, day, day of week
   - Day of year, week of year
   - Season, heating season indicator

3. **Lag Features**
   - PM2.5 lag values at 1, 3, 7 days
   - PM2.5 moving averages at 3, 7, 30 days

4. **Derived Features**
   - Temperature-dewpoint difference (relative humidity indicator)
   - Relative humidity estimation
   - Wind direction classification (8 directions)

## Usage

### 1. Quick Start (Simulated Data)

```bash
python LightGBM_PM25_Simple.py
```

Applicable scenarios:
- Quick code logic testing
- Understanding model working principles
- No need to wait for long data loading

### 2. Full Training (Real Data)

```bash
python LightGBM_PM25.py
```

Runtime:
- Data loading: 5-10 minutes
- Model training: 2-5 minutes
- Total: approximately 10-15 minutes

## Model Parameters

### Core Parameters

| Parameter | Description | Default Value | Tuning Range |
|-----------|-------------|---------------|--------------|
| `num_leaves` | Number of leaf nodes | 31 | 20-100 |
| `max_depth` | Maximum tree depth | 7 | 3-12 |
| `learning_rate` | Learning rate | 0.05 | 0.01-0.1 |
| `feature_fraction` | Feature sampling ratio | 0.8 | 0.5-1.0 |
| `bagging_fraction` | Sample sampling ratio | 0.8 | 0.5-1.0 |
| `min_child_samples` | Minimum samples in leaf node | 20 | 10-50 |

### Parameter Tuning Recommendations

1. **Prevent Overfitting**
   - Reduce `num_leaves` and `max_depth`
   - Increase `min_child_samples`
   - Reduce `learning_rate`, increase `num_boost_round`

2. **Improve Accuracy**
   - Increase `num_boost_round`
   - Adjust `feature_fraction` and `bagging_fraction`
   - Use early stopping mechanism

3. **Speed Up Training**
   - Reduce `num_boost_round`
   - Increase `learning_rate`
   - Reduce `max_depth`

## Hyperparameter Optimization

### Bayesian Optimization (Recommended)

If `bayesian-optimization` is installed, the code will automatically use Bayesian optimization:

```python
# Automatically search for best parameters
optimizer.maximize(init_points=5, n_iter=15)
```

Advantages:
- Intelligent search, high efficiency
- Requires fewer iterations
- Can find better parameter combinations

### Grid Search (Alternative)

If Bayesian optimization library is not installed, use grid search:

```python
param_grid = {
    'num_leaves': [31, 50, 70],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.03, 0.05, 0.07],
    'feature_fraction': [0.7, 0.8, 0.9],
}
```

## Model Evaluation Metrics

### 1. R² (Coefficient of Determination)
- Range: 0-1 (higher is better)
- Meaning: Proportion of variance explained by model
- Excellent: > 0.8, Good: 0.6-0.8, Fair: 0.4-0.6

### 2. RMSE (Root Mean Squared Error)
- Unit: μg/m³ (lower is better)
- Meaning: Average deviation between predicted and actual values
- Excellent: < 15, Good: 15-25, Fair: 25-40

### 3. MAE (Mean Absolute Error)
- Unit: μg/m³ (lower is better)
- Meaning: Average absolute value of prediction errors
- Less sensitive to outliers

### 4. MAPE (Mean Absolute Percentage Error)
- Unit: % (lower is better)
- Meaning: Percentage of relative errors
- Excellent: < 15%, Good: 15-25%, Fair: 25-40%

## Feature Importance

### Two Types of Importance Measures

1. **Split (Split Count)**
   - Number of times feature is used for splitting in trees
   - Reflects feature usage frequency

2. **Gain (Information Gain)**
   - Actual contribution of feature to model performance
   - Better reflects true feature importance (recommended)

### Expected Important Features

Based on PM2.5 formation mechanisms, usually most important features include:

1. **Lag Features** - PM2.5_lag1, PM2.5_lag3, PM2.5_ma3
2. **Temperature** - t2m, d2m, temp_dewpoint_diff
3. **Wind Speed** - wind_speed_10m, u10, v10
4. **Humidity** - tcwv, relative_humidity
5. **Boundary Layer Height** - blh
6. **Time Features** - month, season, is_heating_season

## Visualization Charts

### 1. Training Process Curve
- Shows RMSE changes for training and validation sets
- Helps determine if overfitting
- Shows best iteration count

### 2. Prediction vs Actual Scatter Plot
- Compares predicted and actual values
- Closer to diagonal line indicates more accurate prediction
- Separated by training, validation, and test sets

### 3. Time Series Comparison
- Shows prediction and actual curves
- Intuitively displays model tracking ability
- Shows prediction effect for last 300 days

### 4. Residual Analysis
- Residuals should be randomly distributed around 0
- If there's obvious pattern, model can be improved
- Helps identify systematic bias

### 5. Feature Importance Plot
- Shows Top 20 important features
- Sorted by both Split and Gain methods
- Guides feature selection and engineering

### 6. Model Performance Comparison
- Compares baseline model and optimized model
- Shows R², RMSE, MAE, MAPE
- Quantifies optimization effect

### 7. Error Distribution
- Shows distribution of prediction errors
- Should be normally distributed
- Mean should be close to 0

## Model Usage

### Loading Saved Models

```python
import lightgbm as lgb
import pickle

# Method 1: Load text format
model = lgb.Booster(model_file='models/lightgbm_optimized.txt')

# Method 2: Load pickle format
with open('models/lightgbm_optimized.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Making Predictions

```python
# Prepare new data (features must match training)
X_new = prepare_new_data()  # Your data preparation function

# Predict
predictions = model.predict(X_new, num_iteration=model.best_iteration)

# Prediction result is PM2.5 concentration (μg/m³)
print(f"Predicted PM2.5 concentration: {predictions}")
```

## Frequently Asked Questions

### Q1: Why is a validation set needed?

A: Validation set is used for:
- Early stopping mechanism to prevent overfitting
- Hyperparameter tuning
- Monitoring training process

### Q2: How to handle insufficient data?

A: You can:
- Reduce validation set proportion
- Use cross-validation
- Reduce lag features (reduce sample loss)

### Q3: What if training takes too long?

A: You can:
- Reduce `num_boost_round`
- Increase `learning_rate`
- Reduce data volume (sample by month or quarter)
- Reduce hyperparameter search space

### Q4: How to improve model accuracy?

A: Can try:
- Add more features (e.g., other pollutants)
- Adjust hyperparameters
- Increase training data
- Feature engineering (interaction features, polynomial features)
- Ensemble multiple models

### Q5: Model predictions are inaccurate in certain periods?

A: Possible reasons:
- Extreme weather events (outside data distribution)
- Changes in anthropogenic emissions (e.g., policy impacts)
- Special periods (e.g., Spring Festival, major events)
- Suggestion: Add event marker features

## Advanced Applications

### 1. Multi-step Prediction

```python
# Predict next 7 days
predictions_7day = []
X_current = X_test.iloc[-1:].copy()

for i in range(7):
    pred = model.predict(X_current)
    predictions_7day.append(pred[0])
    
    # Update lag features
    X_current['PM2.5_lag1'] = pred[0]
    # ... update other features
```

### 2. Uncertainty Estimation

```python
# Use quantile regression
model_lower = lgb.train(params, train_data, objective='quantile', alpha=0.1)
model_upper = lgb.train(params, train_data, objective='quantile', alpha=0.9)

# Prediction interval
pred_lower = model_lower.predict(X_test)
pred_upper = model_upper.predict(X_test)
```

### 3. Feature Selection

```python
# Filter features based on importance
importance_threshold = 1.0  # 1%
important_features = feature_importance[
    feature_importance['Importance_Gain_Norm'] > importance_threshold
]['Feature'].tolist()

# Retrain
X_train_selected = X_train[important_features]
```

## References

### LightGBM Official Documentation
- GitHub: https://github.com/microsoft/LightGBM
- Documentation: https://lightgbm.readthedocs.io/

### Related Papers
1. Ke et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
2. Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"

### Recommended Reading
- LightGBM Parameter Tuning Guide
- Gradient Boosting Decision Tree Principles
- Time Series Feature Engineering

## Update Log

### v1.0 (2024)
- Initial version
- Supports basic LightGBM training
- Bayesian optimization hyperparameters
- Complete feature engineering
- Rich visualizations

## Contact

For questions or suggestions, please contact via:
- Submit Issue in code repository
- Check related documentation and examples

---

**Enjoy using it!** 🚀

