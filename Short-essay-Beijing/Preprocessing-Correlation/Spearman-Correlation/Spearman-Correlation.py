# Part 1: Import necessary libraries and define paths
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import warnings
warnings.filterwarnings('ignore')

# Define paths
pollution_all_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)'
era5_path = r'C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV'

# Define date range: 2015-01-01 to 2024-12-31
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

# Pollutants to extract
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# ERA5 variables
era5_vars = [
    'd2m', 't2m', 'u10', 'v10', 'u100', 'v100', 'blh', 'cvh', 'lsm', 'cvl',
    'avg_tprate', 'mn2t', 'sd', 'str', 'sp', 'tisr', 'tcwv', 'tp'
]

# Beijing bounding box for ERA5: lat 39-41, lon 115-117, step 0.25
beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

# Function to generate all dates
def daterange(start, end):
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

# Helper function to find file for a date
def find_file(base_path, date_str, prefix):
    filename = f"{prefix}_{date_str}.csv"
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

print("Setup complete.")# Part 2: Functions to read and process pollution data
def read_pollution_day(date):
    date_str = date.strftime('%Y%m%d')
    all_file = find_file(pollution_all_path, date_str, 'beijing_all')
    extra_file = find_file(pollution_extra_path, date_str, 'beijing_extra')
    
    if not all_file or not extra_file:
        return None
    
    try:
        df_all = pd.read_csv(all_file, encoding='utf-8', on_bad_lines='skip')
        df_extra = pd.read_csv(extra_file, encoding='utf-8', on_bad_lines='skip')
        
        # Filter for hourly data only (exclude 24h averages and AQI)
        df_all = df_all[~df_all['type'].str.contains('_24h|AQI', na=False)]
        df_extra = df_extra[~df_extra['type'].str.contains('_24h', na=False)]
        
        # Combine
        df_poll = pd.concat([df_all, df_extra], ignore_index=True)
        
        # Pivot to have pollutants as columns
        df_poll = df_poll.melt(id_vars=['date', 'hour', 'type'], var_name='station', value_name='value')
        df_poll['value'] = pd.to_numeric(df_poll['value'], errors='coerce')
        
        # Drop NaN, outliers (e.g., negative values for pollutants)
        df_poll = df_poll[df_poll['value'] >= 0]
        
        # Average over stations for each pollutant, hour
        df_daily = df_poll.groupby(['date', 'type'])['value'].mean().reset_index()
        
        # Pivot to wide format
        df_daily = df_daily.pivot(index='date', columns='type', values='value')
        
        # Keep only desired pollutants
        df_daily = df_daily[pollutants]
        
        return df_daily
    except Exception as e:
        print(f"Error reading pollution for {date_str}: {e}")
        return None

# Parallel read pollution data
def read_all_pollution():
    dates = list(daterange(start_date, end_date))
    pollution_dfs = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(read_pollution_day, date) for date in dates]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                pollution_dfs.append(result)
    
    if pollution_dfs:
        df_poll_all = pd.concat(pollution_dfs)
        # Handle missing values: forward fill then mean imputation
        df_poll_all.ffill(inplace=True)
        df_poll_all.fillna(df_poll_all.mean(), inplace=True)
        return df_poll_all
    return pd.DataFrame()

df_pollution = read_all_pollution()
print(f"Pollution data shape: {df_pollution.shape}")
gc.collect()# Part 3: Functions to read and process ERA5 data
def read_era5_month(year, month):
    month_str = f"{year}{month:02d}.csv"
    file_path = find_file(era5_path, month_str, '')
    if not file_path:
        return None
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', low_memory=False)
        
        # Parse time
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])
        df['date'] = df['time'].dt.date
        
        # Filter Beijing area
        df = df[(df['latitude'].isin(beijing_lats)) & (df['longitude'].isin(beijing_lons))]
        
        # Handle expver if multiple
        if 'expver' in df.columns:
            df = df[df['expver'] == '0001']  # Assume primary
        
        # Select variables
        avail_vars = [v for v in era5_vars if v in df.columns]
        df = df[['date', 'latitude', 'longitude'] + avail_vars]
        
        # Convert to numeric, handle errors
        for col in avail_vars:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop outliers: e.g., extreme values based on physical limits
        # For example, temperature between -50 and 50 C
        if 't2m' in df:
            df['t2m'] = df['t2m'].clip(-50 + 273.15, 50 + 273.15)  # Kelvin assumed?
        # Similar for others, but simplify for now
        
        # Average over grid and hours to daily Beijing average
        df_daily = df.groupby('date')[avail_vars].mean().reset_index()
        
        df_daily.set_index('date', inplace=True)
        df_daily.index = pd.to_datetime(df_daily.index)
        
        return df_daily
    except Exception as e:
        print(f"Error reading ERA5 for {year}-{month:02d}: {e}")
        return None

# Parallel read ERA5 data, process month by month to save memory
def read_all_era5():
    era5_dfs = []
    years = range(2015, 2025)
    months = range(1, 13)
    
    with ThreadPoolExecutor(max_workers=4) as executor:  # Lower workers for memory
        futures = []
        for year in years:
            for month in months:
                if year == 2024 and month > 12: continue  # Adjust if needed
                futures.append(executor.submit(read_era5_month, year, month))
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                era5_dfs.append(result)
            gc.collect()  # Clean after each month
    
    if era5_dfs:
        df_era5_all = pd.concat(era5_dfs)
        # Handle missing
        df_era5_all.ffill(inplace=True)
        df_era5_all.fillna(df_era5_all.mean(), inplace=True)
        return df_era5_all
    return pd.DataFrame()

df_era5 = read_all_era5()
print(f"ERA5 data shape: {df_era5.shape}")
gc.collect()# Part 4: Merge data and prepare for analysis
# Ensure indices are datetime
df_pollution.index = pd.to_datetime(df_pollution.index)
df_era5.index = pd.to_datetime(df_era5.index)

# Merge on date
df_combined = df_pollution.join(df_era5, how='inner')

# Final cleaning: remove any remaining NaN or inf
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
df_combined.dropna(inplace=True)

# Compute wind speed if needed, but since u/v are components, perhaps compute magnitude
if 'u10' in df_combined and 'v10' in df_combined:
    df_combined['wind_speed_10m'] = np.sqrt(df_combined['u10']**2 + df_combined['v10']**2)
if 'u100' in df_combined and 'v100' in df_combined:
    df_combined['wind_speed_100m'] = np.sqrt(df_combined['u100']**2 + df_combined['v100']**2)

# Drop original u/v if not needed for correlation, but keep for now

print(f"Combined data shape: {df_combined.shape}")
gc.collect()# Part 5: Compute Spearman correlation
# Select all variables
all_vars = pollutants + era5_vars

# Ensure they exist
avail_vars = [v for v in all_vars if v in df_combined.columns]

# Compute correlation matrix
corr_matrix, p_matrix = spearmanr(df_combined[avail_vars])

# Convert to DataFrame
corr_df = pd.DataFrame(corr_matrix, index=avail_vars, columns=avail_vars)

print("Correlation matrix computed.")
gc.collect()# Part 6: Plot heatmap
# Nature-style: clean, minimal, blue-red cmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_df, annot=False, cmap='RdBu', vmin=-1, vmax=1, center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.75})
plt.title('Spearman Correlation Heatmap', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()  # Show one figure at a time

# Optional: second heatmap for p-values if needed, but skip for now
gc.collect()# Part 7: Analyze high correlations with PM2.5 and PM10
threshold = 0.7

# For PM2.5
if 'PM2.5' in corr_df:
    pm25_corr = corr_df['PM2.5'][abs(corr_df['PM2.5']) >= threshold].drop('PM2.5', errors='ignore')
    print("Parameters highly correlated with PM2.5 (>=0.7):")
    print(pm25_corr)

# For PM10
if 'PM10' in corr_df:
    pm10_corr = corr_df['PM10'][abs(corr_df['PM10']) >= threshold].drop('PM10', errors='ignore')
    print("Parameters highly correlated with PM10 (>=0.7):")
    print(pm10_corr)

# Plot bar for PM2.5
if not pm25_corr.empty:
    plt.figure(figsize=(8, 6))
    pm25_corr.sort_values().plot.barh(color='skyblue')
    plt.title('High Spearman Correlations with PM2.5', fontsize=14)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.tight_layout()
    plt.show()

# Plot bar for PM10
if not pm10_corr.empty:
    plt.figure(figsize=(8, 6))
    pm10_corr.sort_values().plot.barh(color='lightgreen')
    plt.title('High Spearman Correlations with PM10', fontsize=14)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.tight_layout()
    plt.show()

gc.collect()# Part 8: Cleanup
del df_pollution, df_era5, df_combined, corr_df
gc.collect()
import shutil
# Assuming no specific cache dir, but if __pycache__ exists
if os.path.exists('__pycache__'):
    shutil.rmtree('__pycache__')
print("Cleanup complete.")