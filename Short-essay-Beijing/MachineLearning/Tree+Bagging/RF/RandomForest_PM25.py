"""
北京PM2.5浓度预测 - 随机森林模型
使用相关性较强的气象变量（温度、湿度、风速）建立随机森林模型预测PM2.5浓度

主要特征：
- 温度相关：t2m (2米温度), d2m (露点温度)
- 湿度相关：tcwv (总柱水汽)
- 风速相关：wind_speed_10m, wind_speed_100m (从u/v分量计算)
- 其他重要气象因素：blh (边界层高度), tp (降水), sp (气压), str (辐射)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import glob
import multiprocessing
import warnings
import gc

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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 设置随机种子
np.random.seed(42)

print("=" * 80)
print("北京PM2.5浓度预测 - 随机森林模型")
print("=" * 80)

# ============================== 数据路径配置 ==============================
pollution_all_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)'
era5_path = r'C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV'

# 输出路径
output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)

# 模型保存路径
model_dir = Path('./models')
model_dir.mkdir(exist_ok=True)

# 时间范围: 2015-01-01 到 2024-12-31
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

# 北京区域范围
beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

# 污染物列表
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# ERA5 变量
era5_vars = [
    'd2m', 't2m', 'u10', 'v10', 'u100', 'v100',  # 温度、风速
    'blh', 'sp', 'tcwv',  # 边界层高度、气压、水汽
    'tp', 'avg_tprate',  # 降水
    'tisr', 'str',  # 辐射
    'cvh', 'cvl',  # 云覆盖
    'mn2t', 'sd', 'lsm'  # 其他
]

print(f"\n配置参数:")
print(f"数据时间范围: {start_date.date()} 至 {end_date.date()}")
print(f"目标变量: PM2.5浓度")
print(f"输出目录: {output_dir}")
print(f"模型保存目录: {model_dir}")
print(f"CPU核心数: {CPU_COUNT}, 并行工作线程: {MAX_WORKERS}")

# ============================== 辅助函数 ==============================
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

# ============================== 数据加载函数 ==============================
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
        
        # 过滤掉24小时平均值和AQI
        df_all = df_all[~df_all['type'].str.contains('_24h|AQI', na=False)]
        df_extra = df_extra[~df_extra['type'].str.contains('_24h', na=False)]
        
        # 合并数据
        df_poll = pd.concat([df_all, df_extra], ignore_index=True)
        
        # 转换为长格式
        df_poll = df_poll.melt(id_vars=['date', 'hour', 'type'], 
                                var_name='station', value_name='value')
        df_poll['value'] = pd.to_numeric(df_poll['value'], errors='coerce')
        
        # 移除负值和异常值
        df_poll = df_poll[df_poll['value'] >= 0]
        
        # 按日期和类型聚合（所有站点平均）
        df_daily = df_poll.groupby(['date', 'type'])['value'].mean().reset_index()
        
        # 转换为宽格式
        df_daily = df_daily.pivot(index='date', columns='type', values='value')
        
        # 将索引转换为datetime格式（关键：确保索引是datetime类型）
        df_daily.index = pd.to_datetime(df_daily.index, format='%Y%m%d', errors='coerce')
        
        # 只保留需要的污染物
        df_daily = df_daily[[col for col in pollutants if col in df_daily.columns]]
        
        return df_daily
    except Exception as e:
        # 静默错误，不打印（避免干扰进度显示）
        return None

def read_all_pollution():
    """并行读取所有污染数据"""
    print("\n" + "=" * 80)
    print("第1步: 加载污染数据")
    print("=" * 80)
    print(f"\n使用 {MAX_WORKERS} 个并行工作线程")
    
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
        print(f"\n  成功读取 {len(pollution_dfs)}/{len(dates)} 天的数据")
        print("  正在合并数据...")
        df_poll_all = pd.concat(pollution_dfs)
        # 前向填充然后均值填充
        df_poll_all.ffill(inplace=True)
        df_poll_all.fillna(df_poll_all.mean(), inplace=True)
        print(f"污染数据加载完成，形状: {df_poll_all.shape}")
        return df_poll_all
    return pd.DataFrame()

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
            
            # 处理时间（不转换为数值）
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
    print("\n" + "=" * 80)
    print("第2步: 加载气象数据")
    print("=" * 80)
    print(f"\n使用 {MAX_WORKERS} 个并行工作线程")
    print(f"气象数据目录: {era5_path}")
    
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
        
        print(f"\n  总计成功读取: {successful_reads}/{len(futures)} 个月")
    
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
        return pd.DataFrame()

# ============================== 加载并合并数据 ==============================
df_pollution = read_all_pollution()
df_era5 = read_all_era5()

# 检查数据加载情况
print("\n" + "=" * 80)
print("第3步: 数据合并和验证")
print("=" * 80)

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

# 确保索引是datetime类型
df_pollution.index = pd.to_datetime(df_pollution.index)
df_era5.index = pd.to_datetime(df_era5.index)

print(f"\n  污染数据时间范围: {df_pollution.index.min()} 至 {df_pollution.index.max()}")
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

print(f"合并后数据形状: {df_combined.shape}")
print(f"时间范围: {df_combined.index.min().date()} 至 {df_combined.index.max().date()}")

# 创建特征
print("\n正在创建特征...")

# 1. 风速特征
if 'u10' in df_combined.columns and 'v10' in df_combined.columns:
    df_combined['wind_speed_10m'] = np.sqrt(df_combined['u10']**2 + df_combined['v10']**2)
    df_combined['wind_dir_10m'] = np.arctan2(df_combined['v10'], df_combined['u10']) * 180 / np.pi
    df_combined['wind_dir_10m'] = (df_combined['wind_dir_10m'] + 360) % 360  # 转换为0-360度

if 'u100' in df_combined.columns and 'v100' in df_combined.columns:
    df_combined['wind_speed_100m'] = np.sqrt(df_combined['u100']**2 + df_combined['v100']**2)
    df_combined['wind_dir_100m'] = np.arctan2(df_combined['v100'], df_combined['u100']) * 180 / np.pi
    df_combined['wind_dir_100m'] = (df_combined['wind_dir_100m'] + 360) % 360

# 2. 时间特征
df_combined['year'] = df_combined.index.year
df_combined['month'] = df_combined.index.month
df_combined['day'] = df_combined.index.day
df_combined['day_of_year'] = df_combined.index.dayofyear
df_combined['day_of_week'] = df_combined.index.dayofweek
df_combined['week_of_year'] = df_combined.index.isocalendar().week

# 季节特征
df_combined['season'] = df_combined['month'].apply(
    lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
)

# 是否供暖季（北京11月15日-3月15日）
df_combined['is_heating_season'] = ((df_combined['month'] >= 11) | (df_combined['month'] <= 3)).astype(int)

# 3. 温度相关特征
if 't2m' in df_combined.columns and 'd2m' in df_combined.columns:
    # 温度-露点差（反映相对湿度）
    df_combined['temp_dewpoint_diff'] = df_combined['t2m'] - df_combined['d2m']

# 4. 相对湿度估算（简化公式）
if 't2m' in df_combined.columns and 'd2m' in df_combined.columns:
    # Magnus公式近似
    df_combined['relative_humidity'] = 100 * np.exp((17.625 * (df_combined['d2m'] - 273.15)) / 
                                                      (243.04 + (df_combined['d2m'] - 273.15))) / \
                                        np.exp((17.625 * (df_combined['t2m'] - 273.15)) / 
                                               (243.04 + (df_combined['t2m'] - 273.15)))
    df_combined['relative_humidity'] = df_combined['relative_humidity'].clip(0, 100)

# 清理数据
print("\n正在清理数据...")
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)

# 删除包含NaN的行
initial_rows = len(df_combined)
df_combined.dropna(inplace=True)
final_rows = len(df_combined)
print(f"删除了 {initial_rows - final_rows} 行包含缺失值的数据")

print(f"\n最终数据形状: {df_combined.shape}")
print(f"样本数: {len(df_combined)}")

# PM2.5统计
if 'PM2.5' in df_combined.columns:
    print(f"\nPM2.5统计信息:")
    print(f"  均值: {df_combined['PM2.5'].mean():.2f} μg/m³")
    print(f"  标准差: {df_combined['PM2.5'].std():.2f} μg/m³")
    print(f"  最小值: {df_combined['PM2.5'].min():.2f} μg/m³")
    print(f"  最大值: {df_combined['PM2.5'].max():.2f} μg/m³")
    print(f"  中位数: {df_combined['PM2.5'].median():.2f} μg/m³")

gc.collect()

# ============================== 特征选择 ==============================
print("\n" + "=" * 80)
print("第4步: 特征选择和数据准备")
print("=" * 80)

# 定义目标变量
target = 'PM2.5'

# 排除的列（目标变量、其他污染物、年份等）
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']

# 选择相关性较强的特征
selected_features = []

# 温度特征
if 't2m' in df_combined.columns:
    selected_features.append('t2m')
if 'd2m' in df_combined.columns:
    selected_features.append('d2m')
if 'temp_dewpoint_diff' in df_combined.columns:
    selected_features.append('temp_dewpoint_diff')

# 湿度特征
if 'tcwv' in df_combined.columns:
    selected_features.append('tcwv')
if 'relative_humidity' in df_combined.columns:
    selected_features.append('relative_humidity')

# 风速特征
if 'wind_speed_10m' in df_combined.columns:
    selected_features.append('wind_speed_10m')
if 'wind_speed_100m' in df_combined.columns:
    selected_features.append('wind_speed_100m')

# 其他重要气象因素
for feature in ['blh', 'tp', 'sp', 'str', 'tisr', 'avg_tprate']:
    if feature in df_combined.columns:
        selected_features.append(feature)

# 时间特征
for feature in ['month', 'season', 'day_of_year', 'day_of_week', 'is_heating_season']:
    if feature in df_combined.columns:
        selected_features.append(feature)

print(f"\n选择的特征数量: {len(selected_features)}")
print(f"目标变量: {target}")

# 显示特征列表
print(f"\n特征列表:")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i}. {feat}")

# 准备建模数据
X = df_combined[selected_features].copy()
y = df_combined[target].copy()

print(f"\n特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

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
    import sys
    sys.exit(1)

print(f"\nPM2.5统计信息:")
print(f"  均值: {y.mean():.2f} μg/m³")
print(f"  标准差: {y.std():.2f} μg/m³")
print(f"  最小值: {y.min():.2f} μg/m³")
print(f"  最大值: {y.max():.2f} μg/m³")
print(f"  中位数: {y.median():.2f} μg/m³")

# ============================== 数据集划分 ==============================
print("\n" + "=" * 80)
print("第5步: 数据集划分")
print("=" * 80)

# 按时间顺序划分：训练集70%，验证集15%，测试集15%
n_samples = len(X)
train_size = int(n_samples * 0.70)
val_size = int(n_samples * 0.15)

X_train = X.iloc[:train_size]
X_val = X.iloc[train_size:train_size + val_size]
X_test = X.iloc[train_size + val_size:]

y_train = y.iloc[:train_size]
y_val = y.iloc[train_size:train_size + val_size]
y_test = y.iloc[train_size + val_size:]

print(f"\n训练集: {len(X_train)} 样本 ({len(X_train)/n_samples*100:.1f}%)")
print(f"  时间范围: {X_train.index.min().date()} 至 {X_train.index.max().date()}")
print(f"  PM2.5: {y_train.mean():.2f} ± {y_train.std():.2f} μg/m³")

print(f"\n验证集: {len(X_val)} 样本 ({len(X_val)/n_samples*100:.1f}%)")
print(f"  时间范围: {X_val.index.min().date()} 至 {X_val.index.max().date()}")
print(f"  PM2.5: {y_val.mean():.2f} ± {y_val.std():.2f} μg/m³")

print(f"\n测试集: {len(X_test)} 样本 ({len(X_test)/n_samples*100:.1f}%)")
print(f"  时间范围: {X_test.index.min().date()} 至 {X_test.index.max().date()}")
print(f"  PM2.5: {y_test.mean():.2f} ± {y_test.std():.2f} μg/m³")

# ============================== 模型训练 ==============================
print("\n" + "=" * 80)
print("第6步: 随机森林模型训练")
print("=" * 80)

# 6.1 基础随机森林模型
print("\n6.1 训练基础随机森林模型...")
rf_basic = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("  开始训练...")
rf_basic.fit(X_train, y_train)
print("  ✓ 基础模型训练完成")

# 预测
y_train_pred_basic = rf_basic.predict(X_train)
y_val_pred_basic = rf_basic.predict(X_val)
y_test_pred_basic = rf_basic.predict(X_test)

# 6.2 网格搜索优化随机森林
print("\n6.2 网格搜索优化随机森林...")

# 使用较小的搜索空间进行快速测试
param_grid_small = {
    'n_estimators': [100, 200],
    'max_depth': [20, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

print(f"  参数网格: {param_grid_small}")
print(f"  总共 {int(np.prod([len(v) for v in param_grid_small.values()]))} 种参数组合")

rf_grid = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf_grid, 
    param_grid_small, 
    cv=3, 
    scoring='r2',
    verbose=1,
    n_jobs=-1
)

print("  开始网格搜索...")
grid_search.fit(X_train, y_train)
rf_optimized = grid_search.best_estimator_

print(f"\n  ✓ 网格搜索完成")
print(f"  最佳参数: {grid_search.best_params_}")
print(f"  最佳交叉验证R²: {grid_search.best_score_:.4f}")

# 预测
y_train_pred_opt = rf_optimized.predict(X_train)
y_val_pred_opt = rf_optimized.predict(X_val)
y_test_pred_opt = rf_optimized.predict(X_test)

# ============================== 模型评估 ==============================
print("\n" + "=" * 80)
print("第7步: 模型评估")
print("=" * 80)

def evaluate_model(y_true, y_pred, model_name, dataset):
    """评估模型性能"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'Model': model_name,
        'Dataset': dataset,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

# 评估所有模型
results = []
results.append(evaluate_model(y_train, y_train_pred_basic, 'RF_Basic', 'Train'))
results.append(evaluate_model(y_val, y_val_pred_basic, 'RF_Basic', 'Validation'))
results.append(evaluate_model(y_test, y_test_pred_basic, 'RF_Basic', 'Test'))
results.append(evaluate_model(y_train, y_train_pred_opt, 'RF_Optimized', 'Train'))
results.append(evaluate_model(y_val, y_val_pred_opt, 'RF_Optimized', 'Validation'))
results.append(evaluate_model(y_test, y_test_pred_opt, 'RF_Optimized', 'Test'))

results_df = pd.DataFrame(results)
print("\n模型性能对比:")
print(results_df.to_string(index=False))

# 测试集性能排名
test_results = results_df[results_df['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\n测试集性能排名:")
print(test_results.to_string(index=False))

# 性能提升
basic_test_r2 = results_df[(results_df['Model'] == 'RF_Basic') & (results_df['Dataset'] == 'Test')]['R²'].values[0]
opt_test_r2 = results_df[(results_df['Model'] == 'RF_Optimized') & (results_df['Dataset'] == 'Test')]['R²'].values[0]
basic_test_rmse = results_df[(results_df['Model'] == 'RF_Basic') & (results_df['Dataset'] == 'Test')]['RMSE'].values[0]
opt_test_rmse = results_df[(results_df['Model'] == 'RF_Optimized') & (results_df['Dataset'] == 'Test')]['RMSE'].values[0]

if opt_test_r2 > basic_test_r2:
    r2_improvement = (opt_test_r2 - basic_test_r2) / abs(basic_test_r2) * 100
    print(f"\n优化效果:")
    print(f"  R²提升: {r2_improvement:.2f}%")
    
if opt_test_rmse < basic_test_rmse:
    rmse_improvement = (basic_test_rmse - opt_test_rmse) / basic_test_rmse * 100
    print(f"  RMSE降低: {rmse_improvement:.2f}%")

# ============================== 特征重要性分析 ==============================
print("\n" + "=" * 80)
print("第8步: 特征重要性分析")
print("=" * 80)

# 获取特征重要性
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance_Basic': rf_basic.feature_importances_,
    'Importance_Optimized': rf_optimized.feature_importances_
})

# 归一化重要性（百分比）
feature_importance['Importance_Basic_Norm'] = (feature_importance['Importance_Basic'] / 
                                                feature_importance['Importance_Basic'].sum() * 100)
feature_importance['Importance_Optimized_Norm'] = (feature_importance['Importance_Optimized'] / 
                                                     feature_importance['Importance_Optimized'].sum() * 100)

feature_importance = feature_importance.sort_values('Importance_Optimized', ascending=False)

print(f"\nTop 15 重要特征 (优化模型):")
print(feature_importance.head(15)[['Feature', 'Importance_Optimized_Norm']].to_string(index=False))

# ============================== 可视化 ==============================
print("\n" + "=" * 80)
print("第9步: 生成可视化图表")
print("=" * 80)

# 9.1 预测vs实际散点图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

models_data = [
    ('RF_Basic', y_test_pred_basic, 'blue'),
    ('RF_Optimized', y_test_pred_opt, 'green')
]

for i, (name, pred, color) in enumerate(models_data):
    test_result = results_df[(results_df['Model'] == name) & 
                             (results_df['Dataset'] == 'Test')].iloc[0]
    
    axes[i].scatter(y_test, pred, alpha=0.5, s=30, color=color, edgecolors='black', linewidth=0.5)
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='理想预测线')
    axes[i].set_xlabel('实际PM2.5浓度 (μg/m³)', fontsize=12)
    axes[i].set_ylabel('预测PM2.5浓度 (μg/m³)', fontsize=12)
    axes[i].set_title(f'{name}\nR²={test_result["R²"]:.4f}, RMSE={test_result["RMSE"]:.2f}, MAE={test_result["MAE"]:.2f}', 
                      fontsize=13, fontweight='bold')
    axes[i].legend(fontsize=11)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'rf_prediction_scatter.png', dpi=300, bbox_inches='tight')
print("保存: rf_prediction_scatter.png")
plt.close()

# 9.2 时间序列对比图
fig, ax = plt.subplots(figsize=(18, 6))

plot_range = min(365, len(y_test))  # 显示最后一年的数据
plot_idx = range(len(y_test) - plot_range, len(y_test))
time_idx = y_test.index[plot_idx]

ax.plot(time_idx, y_test.iloc[plot_idx], 'k-', label='实际值', linewidth=2, alpha=0.8)
ax.plot(time_idx, y_test_pred_basic[plot_idx], '--', color='blue', 
        label='RF_Basic', linewidth=1.5, alpha=0.7)
ax.plot(time_idx, y_test_pred_opt[plot_idx], '--', color='green', 
        label='RF_Optimized', linewidth=1.5, alpha=0.7)

ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
ax.set_title('PM2.5浓度预测时间序列对比 (测试集最后一年)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / 'rf_timeseries.png', dpi=300, bbox_inches='tight')
print("保存: rf_timeseries.png")
plt.close()

# 9.3 残差分析
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, (name, pred, color) in enumerate(models_data):
    residuals = y_test - pred
    
    axes[i].scatter(pred, residuals, alpha=0.5, s=30, color=color, edgecolors='black', linewidth=0.5)
    axes[i].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[i].set_xlabel('预测值 (μg/m³)', fontsize=12)
    axes[i].set_ylabel('残差 (μg/m³)', fontsize=12)
    axes[i].set_title(f'{name} - 残差分析', fontsize=13, fontweight='bold')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'rf_residuals.png', dpi=300, bbox_inches='tight')
print("保存: rf_residuals.png")
plt.close()

# 9.4 特征重要性图
fig, ax = plt.subplots(figsize=(10, 8))

top_n = min(15, len(feature_importance))  # 显示前15个特征
top_features = feature_importance.head(top_n)

y_pos = np.arange(len(top_features))
width = 0.35

ax.barh(y_pos - width/2, top_features['Importance_Basic_Norm'], width, 
        label='RF_Basic', color='blue', alpha=0.7)
ax.barh(y_pos + width/2, top_features['Importance_Optimized_Norm'], width, 
        label='RF_Optimized', color='green', alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_features['Feature'])
ax.set_xlabel('特征重要性 (%)', fontsize=12)
ax.set_title(f'随机森林特征重要性对比 (Top {top_n})', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'rf_feature_importance.png', dpi=300, bbox_inches='tight')
print("保存: rf_feature_importance.png")
plt.close()

# 9.5 模型性能对比
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

test_df = results_df[results_df['Dataset'] == 'Test']
models = test_df['Model'].tolist()
x_pos = np.arange(len(models))

colors = ['blue', 'green']

for i, metric in enumerate(['R²', 'RMSE', 'MAE']):
    axes[i].bar(x_pos, test_df[metric], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[i].set_xticks(x_pos)
    axes[i].set_xticklabels(['Basic', 'Optimized'], rotation=0, fontsize=11)
    axes[i].set_ylabel(metric, fontsize=12)
    
    if metric == 'R²':
        axes[i].set_title(f'{metric} 对比\n(越大越好)', fontsize=12, fontweight='bold')
    else:
        axes[i].set_title(f'{metric} 对比\n(越小越好)', fontsize=12, fontweight='bold')
    
    axes[i].grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上显示数值
    for j, v in enumerate(test_df[metric]):
        axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'rf_model_comparison.png', dpi=300, bbox_inches='tight')
print("保存: rf_model_comparison.png")
plt.close()

# 9.6 预测误差分布
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, (name, pred, color) in enumerate(models_data):
    errors = y_test - pred
    
    axes[i].hist(errors, bins=50, color=color, alpha=0.7, edgecolor='black')
    axes[i].axvline(x=0, color='red', linestyle='--', linewidth=2.5, label='零误差')
    axes[i].set_xlabel('预测误差 (μg/m³)', fontsize=12)
    axes[i].set_ylabel('频数', fontsize=12)
    axes[i].set_title(f'{name} - 预测误差分布\n均值={errors.mean():.2f}, 标准差={errors.std():.2f}', 
                      fontsize=13, fontweight='bold')
    axes[i].legend(fontsize=11)
    axes[i].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'rf_error_distribution.png', dpi=300, bbox_inches='tight')
print("保存: rf_error_distribution.png")
plt.close()

# ============================== 保存结果 ==============================
print("\n" + "=" * 80)
print("第10步: 保存结果")
print("=" * 80)

# 保存模型性能
results_df.to_csv(output_dir / 'rf_model_performance.csv', index=False, encoding='utf-8-sig')
print("保存: rf_model_performance.csv")

# 保存特征重要性
feature_importance.to_csv(output_dir / 'rf_feature_importance.csv', index=False, encoding='utf-8-sig')
print("保存: rf_feature_importance.csv")

# 保存预测结果
predictions_df = pd.DataFrame({
    'Date': y_test.index,
    'Actual_PM25': y_test.values,
    'Predicted_Basic': y_test_pred_basic,
    'Predicted_Optimized': y_test_pred_opt,
    'Error_Basic': y_test.values - y_test_pred_basic,
    'Error_Optimized': y_test.values - y_test_pred_opt
})
predictions_df.to_csv(output_dir / 'rf_predictions.csv', index=False, encoding='utf-8-sig')
print("保存: rf_predictions.csv")

# 保存最佳参数
best_params_df = pd.DataFrame([grid_search.best_params_])
best_params_df.to_csv(output_dir / 'rf_best_parameters.csv', index=False, encoding='utf-8-sig')
print("保存: rf_best_parameters.csv")

# 保存模型（使用pickle）
import pickle
with open(model_dir / 'rf_optimized.pkl', 'wb') as f:
    pickle.dump(rf_optimized, f)
print("保存: rf_optimized.pkl")

# ============================== 总结报告 ==============================
print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)

print(f"\n输出目录: {output_dir}")
print(f"模型目录: {model_dir}")

print("\n生成的文件:")
print("\nCSV文件:")
print("  - rf_model_performance.csv       模型性能指标")
print("  - rf_feature_importance.csv      特征重要性")
print("  - rf_predictions.csv             预测结果")
print("  - rf_best_parameters.csv         最佳参数")

print("\n图表文件:")
print("  - rf_prediction_scatter.png      预测vs实际散点图")
print("  - rf_timeseries.png              时间序列对比")
print("  - rf_residuals.png               残差分析")
print("  - rf_feature_importance.png      特征重要性图")
print("  - rf_model_comparison.png        模型性能对比")
print("  - rf_error_distribution.png      误差分布")

print("\n模型文件:")
print("  - rf_optimized.pkl               随机森林优化模型")

# 输出最佳模型信息
best_model = test_results.iloc[0]
print(f"\n最佳模型: {best_model['Model']}")
print(f"  R² Score: {best_model['R²']:.4f}")
print(f"  RMSE: {best_model['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model['MAPE']:.2f}%")

print(f"\nTop 5 最重要特征:")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance_Optimized_Norm']:.2f}%")

print(f"\n数据集信息:")
print(f"  训练集样本数: {len(X_train)}")
print(f"  验证集样本数: {len(X_val)}")
print(f"  测试集样本数: {len(X_test)}")
print(f"  特征数量: {len(selected_features)}")

print("\n" + "=" * 80)
print("随机森林 PM2.5浓度预测完成！")
print("=" * 80)

