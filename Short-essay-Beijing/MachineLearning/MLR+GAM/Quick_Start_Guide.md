# Quick Start Guide - Get Started in 5 Minutes

## Fastest Method (Recommended)

### 1. Install Dependencies (1 minute)

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy pygam
```

> 💡 If pygam installation fails, you can skip it; the program will automatically skip the GAM model

### 2. Run Simplified Script (1 minute)

```bash
cd Short-Essay-Beijing/MachineLearning
python MLR_GAM_Simple.py
```

### 3. View Results (3 minutes)

After the program completes, the following files will be generated:

**Chart Files (Important):**
1. `simple_prediction_scatter.png` - 📊 Prediction effect at a glance
2. `simple_timeseries.png` - 📈 Time series prediction curves
3. `simple_model_comparison.png` - 🏆 Model performance comparison
4. `simple_feature_importance.png` - 🔍 Feature importance ranking
5. `simple_residuals.png` - 📉 Residual analysis

**Data Files:**
- `simple_model_performance.csv` - Model score table
- `simple_coefficients.csv` - Regression coefficients

## Expected Results

After running, you will see output similar to this:

```
================================================================================
Beijing PM2.5 Concentration Prediction - Simplified Linear Baseline Model
================================================================================

1. Data Preparation
--------------------------------------------------------------------------------
Generating simulated data...
Data shape: (2000, 12)
Time range: 2015-01-01 to 2020-06-18

2. Feature and Target Separation
--------------------------------------------------------------------------------
Number of features: 11
Feature list: ['d2m', 't2m', 'wind_speed_10m', 'tcwv', 'tp', 'blh', ...]

3. Dataset Split
--------------------------------------------------------------------------------
Training set: 1600 samples (80.0%)
Test set: 400 samples (20.0%)

4. Model Training
--------------------------------------------------------------------------------
4.1 Multiple Linear Regression (MLR)
4.2 Ridge Regression
4.3 Lasso Regression
4.4 Generalized Additive Model (GAM)

5. Model Evaluation
--------------------------------------------------------------------------------
Model Performance Comparison:
 Model   Set      R²     RMSE     MAE
   MLR Train  0.8503   8.2156  6.0987
   MLR  Test  0.8467   8.4821  6.2344
 Ridge Train  0.8503   8.2157  6.0988
 Ridge  Test  0.8466   8.4828  6.2349
...

Test Set Performance Ranking:
 Model      R²     RMSE     MAE
   GAM  0.8721   7.7543  5.7821
   MLR  0.8467   8.4821  6.2344
 Ridge  0.8466   8.4828  6.2349
 Lasso  0.8421   8.6123  6.3512

Best Model: GAM
  R² Score: 0.8721
  RMSE: 7.75 μg/m³
  MAE: 5.78 μg/m³
```

## Understanding Results

### R² Score (Coefficient of Determination)
- **0.87** means the model can explain 87% of PM2.5 variation
- **Closer to 1 is better**
- With real data, R² is typically between 0.1-0.3 (because PM2.5 is affected by multiple factors)

### RMSE (Root Mean Squared Error)
- **7.75 μg/m³** indicates average prediction error
- **Lower is better**
- Can be understood as "how much the predicted value differs from actual value on average"

### MAE (Mean Absolute Error)
- **5.78 μg/m³** indicates average value of absolute errors
- **Lower is better**
- More robust than RMSE (not affected by extreme values)

## Visualization Interpretation

### 1. prediction_scatter.png
- **Diagonal line**: Ideal prediction line (predicted = actual)
- **Point distribution**: Closer to diagonal, more accurate prediction
- **Dispersion**: Reflects prediction stability

### 2. timeseries.png
- **Black solid line**: Actual PM2.5 concentration
- **Colored dashed lines**: Predictions from each model
- **Overlap**: More overlapping lines indicate more accurate predictions

### 3. model_comparison.png
- **Left chart**: Higher R² is better
- **Middle chart**: Lower RMSE is better
- **Right chart**: Lower MAE is better

### 4. feature_importance.png
- **Bar length**: Importance of feature
- **Longer**: That meteorological factor has greater impact on PM2.5

### 5. residuals.png
- **Ideal case**: Points randomly distributed on both sides of y=0
- **If there's a pattern**: Model has systematic bias

## Using Real Data

If you have real meteorological and PM2.5 data:

### Method 1: Modify Data Paths

Edit `MLR_GAM_Baseline.py`, modify lines 51-53:

```python
pollution_all_path = r'Your_Path\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'Your_Path\extra(SO2+NO2+CO+O3)'
era5_path = r'Your_Path\ERA5-Beijing-CSV'
```

Then run:
```bash
python MLR_GAM_Baseline.py
```

### Method 2: Load from CSV

If you have processed CSV files:

```python
# Replace generate_sample_data() in MLR_GAM_Simple.py
df = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)
```

CSV format should be:
```
date,d2m,t2m,wind_speed_10m,tcwv,tp,blh,sp,str,tisr,month,season,PM2.5
2015-01-01,270.5,275.3,3.2,12.5,0.0,650.2,101200,185.3,220.1,1,1,85.3
...
```

## Common Questions

### Q: Why is R² so high with simulated data?
**A:** Simulated data is generated based on meteorological variables, so correlation is strong. Real data R² is typically between 0.1-0.3.

### Q: GAM model training is slow?
**A:** GAM requires grid search for optimal parameters; you can reduce search range:
```python
gam.gridsearch(X_train_scaled, y_train.values, lam=np.logspace(-2, 2, 5))
```

### Q: Want to add more features?
**A:** Add column names to the `features` list:
```python
features = ['d2m', 't2m', 'wind_speed_10m', ..., 'your_new_feature']
```

### Q: Want to adjust training/test set ratio?
**A:** Modify the split ratio at line XX:
```python
split_idx = int(len(X) * 0.7)  # Change to 70% train, 30% test
```

## Next Steps for Learning

1. **Understand features**: Read `Machine_Learning_Models_Guide.md`
2. **View correlation**: Check `../Preprocessing-Correlation/` directory
3. **Try improvements**:
   - Add lag features (previous day's PM2.5)
   - Add moving average features
   - Try feature interactions (temperature × humidity)
4. **Advanced models**: Check `../DeepLearning/` directory

## Encountered Problems?

### Import Error
```bash
ModuleNotFoundError: No module named 'xxx'
```
**Solution:**
```bash
pip install xxx
```

### Data Path Error
```
FileNotFoundError: [Errno 2] No such file or directory
```
**Solution:** Check if data path is correct, or use simplified version (automatically generates data)

### Out of Memory
```
MemoryError
```
**Solution:**
1. Use simplified version
2. Reduce data volume
3. Process data in batches

## Checklist

- [ ] Install dependencies
- [ ] Run simplified script
- [ ] View generated charts
- [ ] Understand meaning of evaluation metrics
- [ ] Try adjusting parameters
- [ ] Use real data (optional)
- [ ] Read detailed documentation
- [ ] Try improving the model

---

**Good luck!** 🎉

For questions, please see `README.md` and `Machine_Learning_Models_Guide.md`

