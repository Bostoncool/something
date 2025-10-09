"""
支持向量回归（SVR）预测PM2.5浓度
===================================
本脚本实现基于SVR的PM2.5浓度预测模型
使用污染物数据和ERA5气象数据作为特征
"""

# Part 1: 导入必要的库
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import warnings
import glob
import multiprocessing
from pathlib import Path
warnings.filterwarnings('ignore')

# 获取CPU核心数
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)  # 保留1个核心给系统

# 尝试导入tqdm进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("提示: tqdm未安装，进度显示将使用简化版本。")
    print("      可使用 'pip install tqdm' 安装以获得更好的进度条显示。")

# 机器学习相关
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("SVR PM2.5浓度预测模型")
print("=" * 80)

# Part 2: 定义数据路径
pollution_all_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)'
era5_path = r'C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV'

# 输出路径 - 使用相对路径，在当前脚本目录下创建
script_dir = Path(__file__).parent
output_dir = script_dir / 'output'
model_dir = script_dir / 'models'
output_dir.mkdir(exist_ok=True)
model_dir.mkdir(exist_ok=True)

# 定义日期范围
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

# 定义污染物
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# 定义ERA5变量
era5_vars = [
    'd2m', 't2m', 'u10', 'v10', 'u100', 'v100', 'blh', 'cvh', 'lsm', 'cvl',
    'avg_tprate', 'mn2t', 'sd', 'str', 'sp', 'tisr', 'tcwv', 'tp'
]

# 北京边界
beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

print(f"数据时间范围: {start_date.date()} 至 {end_date.date()}")
print(f"污染物: {', '.join(pollutants)}")
print(f"气象变量数量: {len(era5_vars)}")
print(f"输出目录: {output_dir}")
print(f"模型保存目录: {model_dir}")
print(f"CPU核心数: {CPU_COUNT}, 并行工作线程: {MAX_WORKERS}")

# Part 3: 辅助函数
def daterange(start, end):
    """生成日期范围"""
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def find_file(base_path, date_str, prefix):
    """查找指定日期的文件"""
    filename = f"{prefix}_{date_str}.csv"
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Part 4: 读取污染数据
def read_pollution_day(date):
    """读取单日污染数据"""
    date_str = date.strftime('%Y%m%d')
    all_file = find_file(pollution_all_path, date_str, 'beijing_all')
    extra_file = find_file(pollution_extra_path, date_str, 'beijing_extra')
    
    if not all_file or not extra_file:
        return None
    
    try:
        df_all = pd.read_csv(all_file, encoding='utf-8', on_bad_lines='skip')
        df_extra = pd.read_csv(extra_file, encoding='utf-8', on_bad_lines='skip')
        
        # 过滤24小时平均和AQI
        df_all = df_all[~df_all['type'].str.contains('_24h|AQI', na=False)]
        df_extra = df_extra[~df_extra['type'].str.contains('_24h', na=False)]
        
        # 合并
        df_poll = pd.concat([df_all, df_extra], ignore_index=True)
        
        # 转换为长格式
        df_poll = df_poll.melt(id_vars=['date', 'hour', 'type'], 
                               var_name='station', value_name='value')
        df_poll['value'] = pd.to_numeric(df_poll['value'], errors='coerce')
        
        # 去除负值和异常值
        df_poll = df_poll[df_poll['value'] >= 0]
        
        # 按污染物和日期求平均
        df_daily = df_poll.groupby(['date', 'type'])['value'].mean().reset_index()
        
        # 转为宽格式
        df_daily = df_daily.pivot(index='date', columns='type', values='value')
        
        # 将索引转换为datetime格式 ⭐ 关键修复
        df_daily.index = pd.to_datetime(df_daily.index, format='%Y%m%d', errors='coerce')
        
        # 只保留所需污染物
        available_pollutants = [p for p in pollutants if p in df_daily.columns]
        df_daily = df_daily[available_pollutants]
        
        return df_daily
    except Exception as e:
        return None

def read_all_pollution():
    """并行读取所有污染数据"""
    print("\n正在加载污染数据...")
    print(f"使用 {MAX_WORKERS} 个并行工作线程")
    dates = list(daterange(start_date, end_date))
    pollution_dfs = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, date): date for date in dates}
        
        if TQDM_AVAILABLE:
            # 使用tqdm进度条
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="加载污染数据", unit="天"):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
        else:
            # 简化进度显示
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    pollution_dfs.append(result)
                if i % 500 == 0 or i == len(futures):
                    print(f"  已处理 {i}/{len(futures)} 天 ({i/len(futures)*100:.1f}%)")
    
    if pollution_dfs:
        print(f"  成功读取 {len(pollution_dfs)}/{len(dates)} 天的数据")
        print("  正在合并数据...")
        df_poll_all = pd.concat(pollution_dfs)
        # 前向填充然后均值填充
        df_poll_all.ffill(inplace=True)
        df_poll_all.fillna(df_poll_all.mean(), inplace=True)
        print(f"污染数据加载完成，形状: {df_poll_all.shape}")
        return df_poll_all
    return pd.DataFrame()

# Part 5: 读取ERA5数据
def read_era5_month(year, month):
    """读取单月ERA5数据 - 处理按变量分文件夹的结构"""
    month_str = f"{year}{month:02d}"
    
    # 查找所有包含该月份数据的文件（从不同变量文件夹中）
    all_files = glob.glob(os.path.join(era5_path, "**", f"*{month_str}*.csv"), recursive=True)
    
    if not all_files:
        return None
    
    # 用于存储所有变量的数据
    monthly_data = None
    loaded_vars = []
    
    for file_path in all_files:
        try:
            # 读取单个变量文件
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', 
                            low_memory=False, comment='#')
            
            if df.empty or 'time' not in df.columns:
                continue
            
            # 处理时间
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.dropna(subset=['time'])
            
            if len(df) == 0:
                continue
            
            # 过滤北京地区
            if 'latitude' in df.columns and 'longitude' in df.columns:
                df = df[(df['latitude'] >= beijing_lats.min()) & 
                       (df['latitude'] <= beijing_lats.max()) &
                       (df['longitude'] >= beijing_lons.min()) & 
                       (df['longitude'] <= beijing_lons.max())]
                
                if len(df) == 0:
                    continue
            
            # 处理expver
            if 'expver' in df.columns:
                if '0001' in df['expver'].values:
                    df = df[df['expver'] == '0001']
                else:
                    first_expver = df['expver'].iloc[0]
                    df = df[df['expver'] == first_expver]
            
            # 提取日期
            df['date'] = df['time'].dt.date
            
            # 找出这个文件包含的变量列
            avail_vars = [v for v in era5_vars if v in df.columns]
            
            if not avail_vars:
                continue
            
            # 转换为数值类型
            for col in avail_vars:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 按日期聚合（空间和时间平均）
            df_daily = df.groupby('date')[avail_vars].mean().reset_index()
            df_daily.set_index('date', inplace=True)
            df_daily.index = pd.to_datetime(df_daily.index)
            
            # 合并到monthly_data
            if monthly_data is None:
                monthly_data = df_daily
            else:
                # 使用join合并，保留所有日期
                monthly_data = monthly_data.join(df_daily, how='outer')
            
            loaded_vars.extend(avail_vars)
            
        except Exception as e:
            continue
    
    if monthly_data is not None and not monthly_data.empty:
        return monthly_data
    else:
        return None

def read_all_era5():
    """并行读取所有ERA5数据"""
    print("\n正在加载气象数据...")
    print(f"使用 {MAX_WORKERS} 个并行工作线程")
    print(f"气象数据目录: {era5_path}")
    print(f"检查目录是否存在: {os.path.exists(era5_path)}")
    
    # 首先检查目录中有哪些文件
    if os.path.exists(era5_path):
        all_csv = glob.glob(os.path.join(era5_path, "**", "*.csv"), recursive=True)
        print(f"找到 {len(all_csv)} 个CSV文件")
        if all_csv:
            print(f"示例文件: {[os.path.basename(f) for f in all_csv[:5]]}")
    
    era5_dfs = []
    years = range(2015, 2025)
    months = range(1, 13)
    
    # 准备所有任务
    month_tasks = [(year, month) for year in years for month in months 
                   if not (year == 2024 and month > 12)]
    total_months = len(month_tasks)
    print(f"尝试加载 {total_months} 个月的数据...")
    
    # 使用更多并行线程加载ERA5数据
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_era5_month, year, month): (year, month) 
                  for year, month in month_tasks}
        
        successful_reads = 0
        if TQDM_AVAILABLE:
            # 使用tqdm进度条
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="加载气象数据", unit="月"):
                result = future.result()
                if result is not None and not result.empty:
                    era5_dfs.append(result)
                    successful_reads += 1
        else:
            # 简化进度显示
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None and not result.empty:
                    era5_dfs.append(result)
                    successful_reads += 1
                if i % 20 == 0 or i == len(futures):
                    print(f"  进度: {i}/{len(futures)} 个月 (成功: {successful_reads}, {i/len(futures)*100:.1f}%)")
            
        print(f"  总计成功读取: {successful_reads}/{len(futures)} 个月")
    
    if era5_dfs:
        print("\n正在合并气象数据...")
        df_era5_all = pd.concat(era5_dfs, axis=0)
        
        # 去重（可能有重复日期）
        print("  去重处理...")
        df_era5_all = df_era5_all[~df_era5_all.index.duplicated(keep='first')]
        
        # 排序
        print("  排序处理...")
        df_era5_all.sort_index(inplace=True)
        
        print(f"合并后形状: {df_era5_all.shape}")
        print(f"时间范围: {df_era5_all.index.min()} 至 {df_era5_all.index.max()}")
        print(f"可用变量: {list(df_era5_all.columns[:10])}..." if len(df_era5_all.columns) > 10 else f"可用变量: {list(df_era5_all.columns)}")
        
        # 填充缺失值
        print("  处理缺失值...")
        initial_na = df_era5_all.isna().sum().sum()
        df_era5_all.ffill(inplace=True)
        df_era5_all.bfill(inplace=True)
        df_era5_all.fillna(df_era5_all.mean(), inplace=True)
        final_na = df_era5_all.isna().sum().sum()
        
        print(f"缺失值处理: {initial_na} -> {final_na}")
        print(f"气象数据加载完成，形状: {df_era5_all.shape}")
        
        return df_era5_all
    else:
        print("\n❌ 错误: 没有成功加载任何气象数据文件！")
        print("可能的原因:")
        print("1. 文件命名格式不匹配（期望格式: *YYYYMM*.csv）")
        print("2. 文件内容格式不正确（缺少time列）")
        print("3. 文件路径不正确")
        return pd.DataFrame()

# Part 6: 数据读取
df_pollution = read_all_pollution()
df_era5 = read_all_era5()
gc.collect()

# 检查数据加载情况
print("\n数据加载检查:")
print(f"  污染数据形状: {df_pollution.shape}")
print(f"  气象数据形状: {df_era5.shape}")

if df_pollution.empty:
    print("\n⚠️ 警告: 污染数据为空！请检查数据路径和文件。")
    import sys
    sys.exit(1)

if df_era5.empty:
    print("\n⚠️ 警告: 气象数据为空！请检查数据路径和文件。")
    import sys
    sys.exit(1)

# Part 7: 数据合并和特征工程
print("\n" + "=" * 80)
print("数据合并与特征工程")
print("=" * 80)

# 确保索引为datetime
df_pollution.index = pd.to_datetime(df_pollution.index)
df_era5.index = pd.to_datetime(df_era5.index)

print(f"  污染数据时间范围: {df_pollution.index.min()} 至 {df_pollution.index.max()}")
print(f"  气象数据时间范围: {df_era5.index.min()} 至 {df_era5.index.max()}")

# 合并数据
print("\n正在合并数据...")
df_combined = df_pollution.join(df_era5, how='inner')

if df_combined.empty:
    print("\n❌ 错误: 数据合并后为空！")
    print("   可能原因: 污染数据和气象数据的日期索引没有交集。")
    print(f"   污染数据有 {len(df_pollution)} 行")
    print(f"   气象数据有 {len(df_era5)} 行")
    print(f"   合并后有 {len(df_combined)} 行")
    import sys
    sys.exit(1)

# 特征工程
# 1. 计算风速
if 'u10' in df_combined and 'v10' in df_combined:
    df_combined['wind_speed_10m'] = np.sqrt(df_combined['u10']**2 + df_combined['v10']**2)
    df_combined['wind_dir_10m'] = np.arctan2(df_combined['v10'], df_combined['u10']) * 180 / np.pi

if 'u100' in df_combined and 'v100' in df_combined:
    df_combined['wind_speed_100m'] = np.sqrt(df_combined['u100']**2 + df_combined['v100']**2)

# 2. 温度相关特征（转换为摄氏度）
if 't2m' in df_combined:
    df_combined['t2m_celsius'] = df_combined['t2m'] - 273.15

if 'd2m' in df_combined:
    df_combined['d2m_celsius'] = df_combined['d2m'] - 273.15

# 3. 相对湿度（如果有露点温度）
if 't2m' in df_combined and 'd2m' in df_combined:
    # 简化的相对湿度计算
    df_combined['relative_humidity'] = 100 * np.exp((17.625 * df_combined['d2m_celsius']) / 
                                                      (243.04 + df_combined['d2m_celsius'])) / \
                                       np.exp((17.625 * df_combined['t2m_celsius']) / 
                                              (243.04 + df_combined['t2m_celsius']))

# 4. 时间特征
df_combined['year'] = df_combined.index.year
df_combined['month'] = df_combined.index.month
df_combined['day'] = df_combined.index.day
df_combined['dayofyear'] = df_combined.index.dayofyear
df_combined['season'] = df_combined['month'].apply(lambda x: (x % 12 + 3) // 3)
df_combined['is_winter'] = df_combined['month'].isin([12, 1, 2]).astype(int)

# 5. 周期性特征（使用正弦和余弦编码）
df_combined['month_sin'] = np.sin(2 * np.pi * df_combined['month'] / 12)
df_combined['month_cos'] = np.cos(2 * np.pi * df_combined['month'] / 12)
df_combined['day_sin'] = np.sin(2 * np.pi * df_combined['dayofyear'] / 365)
df_combined['day_cos'] = np.cos(2 * np.pi * df_combined['dayofyear'] / 365)

# 清理数据
print("\n正在清理数据...")
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)

# 删除包含NaN的行（主要是滞后特征导致的前几行）
initial_rows = len(df_combined)
df_combined.dropna(inplace=True)
final_rows = len(df_combined)
print(f"删除了 {initial_rows - final_rows} 行包含缺失值的数据")

print(f"\n合并后数据形状: {df_combined.shape}")
print(f"时间范围: {df_combined.index.min().date()} 至 {df_combined.index.max().date()}")
print(f"样本数: {len(df_combined)}")
print(f"特征数量: {len(df_combined.columns)}")
print(f"\n前5行数据:")
print(df_combined.head())

# Part 8: 准备训练数据
print("\n" + "=" * 80)
print("准备训练数据")
print("=" * 80)

# 检查PM2.5是否存在
if 'PM2.5' not in df_combined.columns:
    raise ValueError("数据中不包含PM2.5列!")

# 定义特征和目标变量
target = 'PM2.5'
# 排除其他污染物和年份
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']
feature_cols = [col for col in df_combined.columns if col not in exclude_cols]

X = df_combined[feature_cols]
y = df_combined[target]

# ============================== 数据验证 ==============================
# 检查数据是否为空
if len(X) == 0 or len(y) == 0:
    print("\n" + "=" * 80)
    print("❌ 错误: 没有可用的数据！")
    print("=" * 80)
    print("\n可能的原因:")
    print("1. 数据路径不正确，无法找到数据文件")
    print("2. 污染数据或气象数据加载失败")
    print("3. 数据合并后索引没有交集（检查日期范围是否匹配）")
    print("4. 数据清理过程中删除了所有行")
    print("\n请检查:")
    print(f"- 污染数据路径: {pollution_all_path}")
    print(f"- 气象数据路径: {era5_path}")
    print(f"- 日期范围: {start_date.date()} 至 {end_date.date()}")
    import sys
    sys.exit(1)

print(f"特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")
print(f"选择的特征数量: {len(feature_cols)}")
print(f"\nPM2.5统计信息:")
print(f"  均值: {y.mean():.2f} μg/m³")
print(f"  标准差: {y.std():.2f} μg/m³")
print(f"  最小值: {y.min():.2f} μg/m³")
print(f"  最大值: {y.max():.2f} μg/m³")
print(f"  中位数: {y.median():.2f} μg/m³")

# 数据分割（70%训练，15%验证，15%测试）
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

print(f"\n训练集大小: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"验证集大小: {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"测试集大小: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

print("\n特征已标准化完成")

# Part 9: SVR模型训练
print("\n" + "=" * 80)
print("SVR模型训练")
print("=" * 80)

# 定义不同核函数的SVR模型
models = {
    'SVR-RBF': SVR(kernel='rbf', cache_size=1000),
    'SVR-Linear': SVR(kernel='linear', cache_size=1000),
    'SVR-Poly': SVR(kernel='poly', degree=3, cache_size=1000)
}

# 网格搜索参数
param_grids = {
    'SVR-RBF': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.001, 0.01, 0.1],
        'epsilon': [0.01, 0.1, 0.2]
    },
    'SVR-Linear': {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2]
    },
    'SVR-Poly': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1],
        'epsilon': [0.1, 0.2]
    }
}

# 训练和评估每个模型
best_models = {}
results = []

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"训练模型: {name}")
    print(f"{'='*60}")
    
    # 网格搜索
    print(f"开始网格搜索...")
    grid_search = GridSearchCV(
        model, 
        param_grids[name], 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train_scaled)
    
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳CV得分 (负MSE): {grid_search.best_score_:.4f}")
    
    # 使用最佳模型
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    
    # 在验证集上评估
    y_val_pred_scaled = best_model.predict(X_val_scaled)
    y_val_pred = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
    
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n验证集性能:")
    print(f"  RMSE: {val_rmse:.4f}")
    print(f"  MAE: {val_mae:.4f}")
    print(f"  R²: {val_r2:.4f}")
    
    # 在测试集上评估
    y_test_pred_scaled = best_model.predict(X_test_scaled)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n测试集性能:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  R²: {test_r2:.4f}")
    
    # 保存结果
    results.append({
        '模型': name,
        '验证集RMSE': val_rmse,
        '验证集MAE': val_mae,
        '验证集R²': val_r2,
        '测试集RMSE': test_rmse,
        '测试集MAE': test_mae,
        '测试集R²': test_r2,
        '最佳参数': grid_search.best_params_
    })
    
    gc.collect()

# Part 10: 结果汇总
print("\n" + "=" * 80)
print("所有模型性能对比")
print("=" * 80)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# 保存结果
results_df.to_csv(output_dir / 'svr_model_comparison.csv', 
                  index=False, encoding='utf-8-sig')

# Part 11: 选择最佳模型并进行详细分析
best_model_name = results_df.loc[results_df['测试集R²'].idxmax(), '模型']
best_model = best_models[best_model_name]

print(f"\n最佳模型: {best_model_name}")
print(f"测试集R²: {results_df.loc[results_df['模型']==best_model_name, '测试集R²'].values[0]:.4f}")

# 使用最佳模型进行预测
y_train_pred_scaled = best_model.predict(X_train_scaled)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()

y_test_pred_scaled = best_model.predict(X_test_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()

# Part 12: 可视化结果
print("\n" + "=" * 80)
print("生成可视化图表")
print("=" * 80)

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# 图1: 模型性能对比
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# RMSE对比
axes[0, 0].bar(results_df['模型'], results_df['测试集RMSE'], color='steelblue', alpha=0.7)
axes[0, 0].set_title('测试集RMSE对比', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('RMSE', fontsize=10)
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# MAE对比
axes[0, 1].bar(results_df['模型'], results_df['测试集MAE'], color='coral', alpha=0.7)
axes[0, 1].set_title('测试集MAE对比', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('MAE', fontsize=10)
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# R²对比
axes[1, 0].bar(results_df['模型'], results_df['测试集R²'], color='seagreen', alpha=0.7)
axes[1, 0].set_title('测试集R²对比', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('R²', fontsize=10)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)

# 训练集 vs 测试集 R²
x_pos = np.arange(len(results_df))
width = 0.35
axes[1, 1].bar(x_pos - width/2, results_df['验证集R²'], width, label='验证集', 
               color='skyblue', alpha=0.7)
axes[1, 1].bar(x_pos + width/2, results_df['测试集R²'], width, label='测试集', 
               color='lightcoral', alpha=0.7)
axes[1, 1].set_title('验证集与测试集R²对比', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('R²', fontsize=10)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(results_df['模型'], rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 已保存: model_comparison.png")

# 图2: 最佳模型预测结果
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 训练集散点图
axes[0, 0].scatter(y_train, y_train_pred, alpha=0.3, s=10, color='blue')
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                'r--', lw=2, label='理想预测')
axes[0, 0].set_xlabel('实际PM2.5 (μg/m³)', fontsize=10)
axes[0, 0].set_ylabel('预测PM2.5 (μg/m³)', fontsize=10)
axes[0, 0].set_title(f'训练集预测结果 ({best_model_name})', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 测试集散点图
axes[0, 1].scatter(y_test, y_test_pred, alpha=0.5, s=20, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='理想预测')
axes[0, 1].set_xlabel('实际PM2.5 (μg/m³)', fontsize=10)
axes[0, 1].set_ylabel('预测PM2.5 (μg/m³)', fontsize=10)
axes[0, 1].set_title(f'测试集预测结果 (R²={test_r2:.4f})', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 残差图
residuals = y_test - y_test_pred
axes[1, 0].scatter(y_test_pred, residuals, alpha=0.5, s=20, color='purple')
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('预测PM2.5 (μg/m³)', fontsize=10)
axes[1, 0].set_ylabel('残差 (μg/m³)', fontsize=10)
axes[1, 0].set_title('残差分布图', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# 残差直方图
axes[1, 1].hist(residuals, bins=50, color='orange', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('残差 (μg/m³)', fontsize=10)
axes[1, 1].set_ylabel('频数', fontsize=10)
axes[1, 1].set_title(f'残差分布直方图 (均值={residuals.mean():.2f})', 
                     fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'prediction_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 已保存: prediction_results.png")

# 图3: 时间序列预测对比（仅测试集部分数据）
fig, ax = plt.subplots(figsize=(16, 6))

# 选择测试集前500个样本进行可视化
n_samples = min(500, len(y_test))
test_indices = y_test.index[:n_samples]

ax.plot(test_indices, y_test.iloc[:n_samples], 
        label='实际值', color='blue', linewidth=1.5, alpha=0.7)
ax.plot(test_indices, y_test_pred[:n_samples], 
        label='预测值', color='red', linewidth=1.5, alpha=0.7)
ax.fill_between(test_indices, y_test.iloc[:n_samples], y_test_pred[:n_samples],
                alpha=0.2, color='gray')

ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
ax.set_title(f'PM2.5浓度预测时间序列对比 ({best_model_name})', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / 'time_series_prediction.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 已保存: time_series_prediction.png")

# Part 13: 保存模型和标准化器
print("\n" + "=" * 80)
print("保存模型")
print("=" * 80)

# 保存最佳模型到models目录
joblib.dump(best_model, model_dir / f'{best_model_name}_best.pkl')
print(f"  ✓ 已保存最佳模型: {best_model_name}_best.pkl")

# 保存所有模型到models目录
for name, model in best_models.items():
    joblib.dump(model, model_dir / f'{name}.pkl')
print(f"  ✓ 已保存所有模型到: {model_dir}")

# 保存标准化器到models目录
joblib.dump(scaler_X, model_dir / 'scaler_X.pkl')
joblib.dump(scaler_y, model_dir / 'scaler_y.pkl')
print(f"  ✓ 已保存标准化器")

# 保存特征名称到output目录
feature_names_df = pd.DataFrame({'feature': feature_cols})
feature_names_df.to_csv(output_dir / 'feature_names.csv', 
                        index=False, encoding='utf-8-sig')
print(f"  ✓ 已保存特征名称")

# Part 14: 生成详细报告
print("\n" + "=" * 80)
print("生成详细报告")
print("=" * 80)

report_content = f"""
SVR PM2.5浓度预测模型报告
{'='*80}

1. 数据概况
   - 数据时间范围: {start_date.date()} 至 {end_date.date()}
   - 总样本数: {len(df_combined)}
   - 特征数量: {len(feature_cols)}
   - 目标变量: PM2.5浓度 (μg/m³)

2. 数据集划分
   - 训练集: {X_train.shape[0]} 样本 ({X_train.shape[0]/len(X)*100:.1f}%)
   - 验证集: {X_val.shape[0]} 样本 ({X_val.shape[0]/len(X)*100:.1f}%)
   - 测试集: {X_test.shape[0]} 样本 ({X_test.shape[0]/len(X)*100:.1f}%)

3. PM2.5统计信息
   - 均值: {y.mean():.2f} μg/m³
   - 标准差: {y.std():.2f} μg/m³
   - 最小值: {y.min():.2f} μg/m³
   - 最大值: {y.max():.2f} μg/m³
   - 中位数: {y.median():.2f} μg/m³

4. 模型性能对比
"""

for _, row in results_df.iterrows():
    report_content += f"""
   {row['模型']}:
   - 测试集RMSE: {row['测试集RMSE']:.4f}
   - 测试集MAE: {row['测试集MAE']:.4f}
   - 测试集R²: {row['测试集R²']:.4f}
   - 最佳参数: {row['最佳参数']}
"""

report_content += f"""
5. 最佳模型
   - 模型名称: {best_model_name}
   - 测试集R²: {test_r2:.4f}
   - 测试集RMSE: {test_rmse:.4f}
   - 测试集MAE: {test_mae:.4f}

6. 特征工程
   - 原始污染物特征: {', '.join(pollutants)}
   - ERA5气象特征: {', '.join([v for v in era5_vars if v in df_combined.columns])}
   - 衍生特征: 风速、风向、温度转换、相对湿度、时间特征等

7. 模型说明
   支持向量回归（SVR）是一种强大的非线性回归方法，通过核函数将特征映射到
   高维空间，能够有效拟合PM2.5与气象因素之间的复杂非线性关系。本研究对比了
   RBF、Linear和Poly三种核函数，并通过网格搜索优化超参数。

8. 文件清单
   - SVR_PM25_Prediction.py: 主程序
   - svr_model_comparison.csv: 模型对比结果
   - model_comparison.png: 模型性能对比图
   - prediction_results.png: 预测结果详细分析图
   - time_series_prediction.png: 时间序列预测对比图
   - {best_model_name}_best.pkl: 最佳模型文件
   - scaler_X.pkl, scaler_y.pkl: 标准化器
   - feature_names.csv: 特征名称列表

{'='*80}
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# 保存报告
with open(output_dir / 'model_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_content)

print(report_content)
print("\n  ✓ 已保存详细报告: model_report.txt")

# Part 15: 清理内存和总结
print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)

print("\n生成的文件:")
print("\nCSV文件 (output目录):")
print("  - svr_model_comparison.csv    模型性能对比")
print("  - feature_names.csv           特征名称列表")

print("\n图表文件 (output目录):")
print("  - model_comparison.png        模型性能对比图")
print("  - prediction_results.png      预测结果详细分析图")
print("  - time_series_prediction.png  时间序列预测对比图")

print("\n模型文件 (models目录):")
print(f"  - {best_model_name}_best.pkl  最佳模型")
print("  - SVR-RBF.pkl, SVR-Linear.pkl, SVR-Poly.pkl  所有SVR模型")
print("  - scaler_X.pkl, scaler_y.pkl  标准化器")

print("\n报告文件 (output目录):")
print("  - model_report.txt            详细分析报告")

print(f"\n输出目录: {output_dir}")
print(f"模型目录: {model_dir}")

del df_pollution, df_era5, df_combined, X_train_scaled, X_val_scaled, X_test_scaled
gc.collect()

print("\n" + "=" * 80)
print("SVR PM2.5浓度预测完成！")
print("=" * 80)

