# Beijing PM2.5 Concentration Prediction - Machine Learning Models

## Overview

This directory contains machine learning baseline models for predicting PM2.5 concentrations based on meteorological variables, using Multiple Linear Regression (MLR) and Generalized Additive Model (GAM) as benchmark methods.

## Model List

### 1. MLR_GAM_Baseline.py - Linear Baseline Model

This is a linear baseline model for predicting PM2.5 concentrations, including:
- **Multiple Linear Regression (MLR)**
- **Ridge Regression (L2 regularization)**
- **Generalized Additive Model (GAM)**

## Data Sources

### Meteorological Data
- Source: ERA5 reanalysis data
- Time range: 2015-01-01 to 2024-12-31
- Spatial range: Beijing area (39-41°N, 115-117°E)
- Temporal resolution: Daily averages

### Pollution Data
- Source: Beijing air quality monitoring stations
- Pollutants: PM2.5, PM10, SO2, NO2, CO, O3
- Temporal resolution: Daily averages

## Feature Selection

Based on correlation analysis results (Pearson, Spearman, Kendall correlation), select meteorological variables with strong correlation to PM2.5 concentration:

### Main Features

1. **d2m** (Dew Point Temperature) - Strongest correlation
   - Reflects air humidity, affects hygroscopicity and sedimentation of pollutants

2. **t2m** (2-meter Temperature)
   - Affects atmospheric stability and convection activity

3. **wind_speed_10m** (10-meter Wind Speed)
   - Calculated from u10 and v10
   - Affects horizontal dispersion of pollutants

4. **tcwv** (Total Column Water Vapor)
   - Reflects water vapor content in atmosphere, affects visibility and pollutant distribution

5. **tp** (Total Precipitation)
   - Precipitation has a removal effect on pollutants

6. **blh** (Boundary Layer Height)
   - Affects vertical dispersion capacity of pollutants

7. **sp** (Surface Pressure)
   - Reflects atmospheric circulation patterns

8. **str** (Surface Thermal Radiation)
   - Reflects surface temperature and radiation balance

9. **tisr** (Top Incident Solar Radiation)
   - Affects photochemical reactions and atmospheric stability

10. **Time Features** (month, season)
    - Capture seasonal variations

## Model Description

### 1. Multiple Linear Regression (MLR)

The simplest linear model, assuming a linear relationship between PM2.5 concentration and meteorological variables:

```
PM2.5 = β₀ + β₁·d2m + β₂·t2m + β₃·wind_speed + ... + ε
```

**Advantages:**
- Simple and intuitive, easy to interpret
- High computational efficiency
- Can directly see the contribution of each feature

**Disadvantages:**
- Can only capture linear relationships
- Sensitive to multicollinearity

### 2. Ridge Regression

Adds L2 regularization on top of MLR to prevent overfitting:

```
Loss = MSE + α·||β||²
```

**Advantages:**
- Alleviates multicollinearity issues
- Prevents overfitting
- All features are retained (won't become 0)

**Disadvantages:**
- Need to tune regularization parameter α
- Still can only capture linear relationships

### 3. Generalized Additive Model (GAM)

Allows non-linear relationships between features and target variable:

```
PM2.5 = β₀ + s₁(d2m) + s₂(t2m) + s₃(wind_speed) + ... + ε
```

Where s(·) is a smooth spline function

**Advantages:**
- Can capture non-linear relationships
- Maintains some interpretability
- Effect of each feature can be visualized separately

**Disadvantages:**
- Higher computational complexity
- Need to choose degrees of freedom for splines
- May overfit

## Usage

### Environment Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy pygam
```

### Running the Model

```bash
python MLR_GAM_Baseline.py
```

### Output Files

The program will generate the following files:

**Data files:**
1. `model_performance.csv` - Performance metrics for each model
2. `predictions.csv` - Prediction results on test set
3. `feature_importance.csv` - Feature importance (coefficient values)

**Visualization charts:**
1. `prediction_scatter.png` - Predicted vs actual values scatter plot
2. `prediction_timeseries.png` - Time series prediction comparison
3. `residuals_analysis.png` - Residual analysis plot
4. `feature_importance.png` - Feature importance bar chart
5. `model_comparison.png` - Model performance comparison

## Evaluation Metrics

### R² Score (Coefficient of Determination)
- Range: -∞ to 1
- Closer to 1 indicates better model fit
- 0 means model is equivalent to mean prediction
- Negative value means model is worse than mean prediction

### RMSE (Root Mean Squared Error)
```
RMSE = √(Σ(y_true - y_pred)² / n)
```
- Same unit as PM2.5 concentration (μg/m³)
- Smaller is better
- More sensitive to large errors

### MAE (Mean Absolute Error)
```
MAE = Σ|y_true - y_pred| / n
```
- Same unit as PM2.5 concentration (μg/m³)
- Smaller is better
- Treats all errors equally

## Dataset Split

- **Training Set**: First 80% of data (in chronological order)
  - Used for model training
  
- **Test Set**: Last 20% of data
  - Used to evaluate model performance
  - Simulates actual prediction scenarios

## Model Improvement Directions

### 1. Feature Engineering
- Add lag features (PM2.5 concentrations from previous days)
- Add moving average features
- Add more time features (weekends/weekdays, holidays, etc.)
- Feature interaction terms (e.g., temperature × humidity)

### 2. Non-linear Models
- Random Forest
- Gradient Boosting Trees (XGBoost, LightGBM)
- Support Vector Regression (SVR)
- Neural Networks (MLP)

### 3. Time Series Models
- ARIMA/SARIMA
- LSTM/GRU
- Transformer

### 4. Ensemble Learning
- Model fusion
- Stacking
- Blending

## References

1. Hastie, T., & Tibshirani, R. (1990). Generalized Additive Models.
2. Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression.
3. James, G., et al. (2013). An Introduction to Statistical Learning.

## Correlation Analysis Results Summary

Based on analysis in the `Preprocessing-Correlation` directory:

### Pearson Correlation (Linear correlation)
Variables most correlated with PM2.5_mean:
- d2m_max: 0.102
- d2m_mean: 0.099
- tp_min: 0.098
- tcwv_max: 0.081

### Kendall Correlation (Rank correlation, robust to outliers)
Variables most correlated with PM2.5_mean:
- d2m_min: 0.102
- d2m_max: 0.098
- d2m_mean: 0.090

### Main Findings
1. **Dew point temperature (d2m)** has the strongest correlation with PM2.5
2. **Humidity-related variables (tcwv)** have significant impact
3. **Wind speed** affects pollutant dispersion
4. **Temperature** affects atmospheric stability
5. Overall correlation is weak (<0.15), indicating PM2.5 is affected by multiple factors

## Notes

1. **Data paths**: Please modify data paths according to actual situation
2. **Computation time**: Full run may take 10-30 minutes
3. **Memory requirements**: At least 8GB RAM recommended
4. **GAM model**: If pygam is not installed, GAM model will be automatically skipped

## Contact

For questions, please refer to:
- Correlation analysis: `../Preprocessing-Correlation/`
- Deep learning models: `../DeepLearning/`

