import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import warnings
import pickle
from pathlib import Path
import glob
import multiprocessing

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

# 机器学习库
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# XGBoost
import xgboost as xgb

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 设置随机种子
np.random.seed(42)

print("=" * 80)
print("北京PM2.5浓度预测 - XGBoost模型")
print("=" * 80)

# ============================== 第1部分: 配置和路径设置 ==============================
print("\n配置参数...")

# 数据路径
pollution_all_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)'
era5_path = r'C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV'

# 输出路径
output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)

# 模型保存路径
model_dir = Path('./models')
model_dir.mkdir(exist_ok=True)

# 日期范围
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

# 北京地理范围
beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

# 污染物列表
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# ERA5变量
era5_vars = [
    'd2m', 't2m', 'u10', 'v10', 'u100', 'v100',  # 温度、风速
    'blh', 'sp', 'tcwv',  # 边界层高度、气压、水汽
    'tp', 'avg_tprate',  # 降水
    'tisr', 'str',  # 辐射
    'cvh', 'cvl',  # 云覆盖
    'mn2t', 'sd', 'lsm'  # 其他
]

print(f"数据时间范围: {start_date.date()} 至 {end_date.date()}")
print(f"目标变量: PM2.5浓度")
print(f"输出目录: {output_dir}")
print(f"模型保存目录: {model_dir}")
print(f"CPU核心数: {CPU_COUNT}, 并行工作线程: {MAX_WORKERS}")

# ============================== 第2部分: 数据加载函数 ==============================
def daterange(start, end):
    """生成日期序列"""
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def find_file(base_path, date_str, prefix):
    """查找指定日期的文件"""
    filename = f"{prefix}_{date_str}.csv"
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

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
        
        # 过滤掉24小时平均和AQI
        df_all = df_all[~df_all['type'].str.contains('_24h|AQI', na=False)]
        df_extra = df_extra[~df_extra['type'].str.contains('_24h', na=False)]
        
        # 合并
        df_poll = pd.concat([df_all, df_extra], ignore_index=True)
        
        # 转换为长格式
        df_poll = df_poll.melt(id_vars=['date', 'hour', 'type'], 
                                var_name='station', value_name='value')
        df_poll['value'] = pd.to_numeric(df_poll['value'], errors='coerce')
        
        # 删除负值和异常值
        df_poll = df_poll[df_poll['value'] >= 0]
        
        # 按日期和类型聚合（所有站点平均）
        df_daily = df_poll.groupby(['date', 'type'])['value'].mean().reset_index()
        
        # 转换为宽格式
        df_daily = df_daily.pivot(index='date', columns='type', values='value')
        
        # 将索引转换为datetime格式
        df_daily.index = pd.to_datetime(df_daily.index, format='%Y%m%d', errors='coerce')
        
        # 只保留需要的污染物
        df_daily = df_daily[[col for col in pollutants if col in df_daily.columns]]
        
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

def read_era5_month(year, month):
    """读取单月ERA5数据 - 处理按变量分文件夹的结构"""
    month_str = f"{year}{month:02d}"
    
    # 查找所有包含该月份数据的文件（从不同变量文件夹中）
    all_files = glob.glob(os.path.join(era5_path, "**", f"*{month_str}*.csv"), recursive=True)
    
    if not all_files:
        # print(f"  警告: 未找到 {year}年{month}月 的气象数据文件")
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
            # print(f"  错误: 处理文件 {os.path.basename(file_path)} 时出错 - {e}")
            continue
    
    if monthly_data is not None and not monthly_data.empty:
        print(f"  成功读取: {year}-{month:02d}, 日数: {len(monthly_data)}, 变量数: {len(loaded_vars)}")
        return monthly_data
    else:
        # print(f"  警告: {year}年{month}月 没有成功加载任何数据")
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

# ============================== 第3部分: 特征工程 ==============================
def create_features(df):
    """创建额外特征"""
    df_copy = df.copy()
    
    # 1. 风速特征
    if 'u10' in df_copy and 'v10' in df_copy:
        df_copy['wind_speed_10m'] = np.sqrt(df_copy['u10']**2 + df_copy['v10']**2)
        df_copy['wind_dir_10m'] = np.arctan2(df_copy['v10'], df_copy['u10']) * 180 / np.pi
        df_copy['wind_dir_10m'] = (df_copy['wind_dir_10m'] + 360) % 360  # 转换为0-360度
    
    if 'u100' in df_copy and 'v100' in df_copy:
        df_copy['wind_speed_100m'] = np.sqrt(df_copy['u100']**2 + df_copy['v100']**2)
        df_copy['wind_dir_100m'] = np.arctan2(df_copy['v100'], df_copy['u100']) * 180 / np.pi
        df_copy['wind_dir_100m'] = (df_copy['wind_dir_100m'] + 360) % 360
    
    # 2. 时间特征
    df_copy['year'] = df_copy.index.year
    df_copy['month'] = df_copy.index.month
    df_copy['day'] = df_copy.index.day
    df_copy['day_of_year'] = df_copy.index.dayofyear
    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['week_of_year'] = df_copy.index.isocalendar().week
    
    # 季节特征
    df_copy['season'] = df_copy['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )
    
    # 是否供暖季（北京11月15日-3月15日）
    df_copy['is_heating_season'] = ((df_copy['month'] >= 11) | (df_copy['month'] <= 3)).astype(int)
    
    # 3. 温度相关特征
    if 't2m' in df_copy and 'd2m' in df_copy:
        # 温度-露点差（反映相对湿度）
        df_copy['temp_dewpoint_diff'] = df_copy['t2m'] - df_copy['d2m']
    
    # 4. 滞后特征（前1天、前3天、前7天的PM2.5）
    if 'PM2.5' in df_copy:
        df_copy['PM2.5_lag1'] = df_copy['PM2.5'].shift(1)
        df_copy['PM2.5_lag3'] = df_copy['PM2.5'].shift(3)
        df_copy['PM2.5_lag7'] = df_copy['PM2.5'].shift(7)
        
        # 滚动平均特征
        df_copy['PM2.5_ma3'] = df_copy['PM2.5'].rolling(window=3, min_periods=1).mean()
        df_copy['PM2.5_ma7'] = df_copy['PM2.5'].rolling(window=7, min_periods=1).mean()
        df_copy['PM2.5_ma30'] = df_copy['PM2.5'].rolling(window=30, min_periods=1).mean()
    
    # 5. 相对湿度估算（简化公式）
    if 't2m' in df_copy and 'd2m' in df_copy:
        # Magnus公式近似
        df_copy['relative_humidity'] = 100 * np.exp((17.625 * (df_copy['d2m'] - 273.15)) / 
                                                      (243.04 + (df_copy['d2m'] - 273.15))) / \
                                        np.exp((17.625 * (df_copy['t2m'] - 273.15)) / 
                                               (243.04 + (df_copy['t2m'] - 273.15)))
        df_copy['relative_humidity'] = df_copy['relative_humidity'].clip(0, 100)
    
    # 6. 风向分类（按8个方位）
    if 'wind_dir_10m' in df_copy:
        df_copy['wind_dir_category'] = pd.cut(df_copy['wind_dir_10m'], 
                                                bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                                                labels=[0, 1, 2, 3, 4, 5, 6, 7],
                                                include_lowest=True).astype(int)
    
    return df_copy

# ============================== 第4部分: 数据加载和预处理 ==============================
print("\n" + "=" * 80)
print("第1步: 数据加载和预处理")
print("=" * 80)

df_pollution = read_all_pollution()
df_era5 = read_all_era5()

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

# 确保索引是datetime类型
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

# 创建特征
print("\n正在创建特征...")
df_combined = create_features(df_combined)

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
print(f"特征数: {df_combined.shape[1]}")

# 显示部分特征列表
print(f"\n特征列表（前20个）:")
for i, col in enumerate(df_combined.columns[:20], 1):
    print(f"  {i}. {col}")
if len(df_combined.columns) > 20:
    print(f"  ... 还有 {len(df_combined.columns) - 20} 个特征")

# ============================== 第5部分: 特征选择和数据准备 ==============================
print("\n" + "=" * 80)
print("第2步: 特征选择和数据准备")
print("=" * 80)

# 定义目标变量
target = 'PM2.5'

# 排除的列（目标变量、其他污染物、年份等）
exclude_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'year']

# 选择数值型特征
numeric_features = [col for col in df_combined.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_cols]

print(f"\n选择的特征数量: {len(numeric_features)}")
print(f"目标变量: {target}")

# 准备建模数据
X = df_combined[numeric_features].copy()
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
    print("\n请检查:")
    print(f"- 污染数据路径: {pollution_all_path}")
    print(f"- 气象数据路径: {era5_path}")
    print(f"- 日期范围: {start_date.date()} 至 {end_date.date()}")
    print(f"\n污染数据形状: {df_pollution.shape}")
    print(f"气象数据形状: {df_era5.shape}")
    print(f"合并后数据形状: {df_combined.shape}")
    import sys
    sys.exit(1)

print(f"\nPM2.5统计信息:")
print(f"  均值: {y.mean():.2f} μg/m³")
print(f"  标准差: {y.std():.2f} μg/m³")
print(f"  最小值: {y.min():.2f} μg/m³")
print(f"  最大值: {y.max():.2f} μg/m³")
print(f"  中位数: {y.median():.2f} μg/m³")

# ============================== 第6部分: 数据集划分 ==============================
print("\n" + "=" * 80)
print("第3步: 数据集划分")
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

# XGBoost使用标准的numpy数组/DataFrame

# ============================== 第7部分: XGBoost基础模型 ==============================
print("\n" + "=" * 80)
print("第4步: XGBoost基础模型训练")
print("=" * 80)

# 基础参数
params_basic = {
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': MAX_WORKERS,
    'eval_metric': 'rmse',  # 在模型初始化时设置评估指标
    'early_stopping_rounds': 50  # 添加早停机制
}

print("\n基础模型参数:")
for key, value in params_basic.items():
    print(f"  {key}: {value}")

print("\n开始训练基础模型...")
model_basic = xgb.XGBRegressor(**params_basic)
evals_result_basic = {}
model_basic.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)

# 获取评估结果
evals_result_basic = model_basic.evals_result()
print(f"\n✓ 基础模型训练完成")

# 预测
y_train_pred_basic = model_basic.predict(X_train)
y_val_pred_basic = model_basic.predict(X_val)
y_test_pred_basic = model_basic.predict(X_test)

# 评估
def evaluate_model(y_true, y_pred, dataset_name):
    """评估模型性能"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'Dataset': dataset_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

results_basic = []
results_basic.append(evaluate_model(y_train, y_train_pred_basic, 'Train'))
results_basic.append(evaluate_model(y_val, y_val_pred_basic, 'Validation'))
results_basic.append(evaluate_model(y_test, y_test_pred_basic, 'Test'))

results_basic_df = pd.DataFrame(results_basic)
print("\n基础模型性能:")
print(results_basic_df.to_string(index=False))

# ============================== 第8部分: 超参数优化 ==============================
print("\n" + "=" * 80)
print("步骤 5: 超参数优化")
print("=" * 80)

optimize = input("\n是否进行超参数优化? (y/n, 默认n): ").strip().lower() == 'y'

if optimize:
    print("\n使用网格搜索进行超参数优化...")
    print("这可能需要几分钟时间...\n")
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=MAX_WORKERS
    )
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=min(4, MAX_WORKERS)
    )
    
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    model_optimized = grid_search.best_estimator_
    
    print("\n最佳超参数:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
else:
    print("\n跳过超参数优化，使用基础模型...")
    model_optimized = model_basic
    best_params = params_basic

# ============================== 第9部分: 训练优化模型 ==============================
print("\n" + "=" * 80)
print("第6步: 使用最佳参数训练优化模型")
print("=" * 80)

if optimize:
    # 使用GridSearchCV找到的最佳参数重新训练，以获取eval_set结果
    print("\n使用最佳参数重新训练模型（用于获取评估曲线）...")
    
    params_optimized = {
        'objective': 'reg:squarederror',
        'max_depth': best_params.get('max_depth', 5),
        'learning_rate': best_params.get('learning_rate', 0.05),
        'n_estimators': best_params.get('n_estimators', 200),
        'min_child_weight': best_params.get('min_child_weight', 3),
        'subsample': best_params.get('subsample', 0.8),
        'colsample_bytree': best_params.get('colsample_bytree', 0.8),
        'random_state': 42,
        'n_jobs': MAX_WORKERS,
        'eval_metric': 'rmse',
        'early_stopping_rounds': 50
    }
    
    print("\n优化模型参数:")
    for key, value in params_optimized.items():
        print(f"  {key}: {value}")
    
    # 用最佳参数重新训练
    model_optimized = xgb.XGBRegressor(**params_optimized)
    model_optimized.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )
    
    # 获取评估结果
    evals_result_opt = model_optimized.evals_result()
    print(f"\n✓ 优化模型训练完成")
else:
    print("\n使用基础模型参数")
    params_optimized = params_basic
    evals_result_opt = evals_result_basic

# 预测
y_train_pred_opt = model_optimized.predict(X_train)
y_val_pred_opt = model_optimized.predict(X_val)
y_test_pred_opt = model_optimized.predict(X_test)

# 评估
results_opt = []
results_opt.append(evaluate_model(y_train, y_train_pred_opt, 'Train'))
results_opt.append(evaluate_model(y_val, y_val_pred_opt, 'Validation'))
results_opt.append(evaluate_model(y_test, y_test_pred_opt, 'Test'))

results_opt_df = pd.DataFrame(results_opt)
print("\n优化模型性能:")
print(results_opt_df.to_string(index=False))

# ============================== 第10部分: 模型比较 ==============================
print("\n" + "=" * 80)
print("第7步: 模型性能比较")
print("=" * 80)

# 合并结果
results_basic_df['Model'] = 'XGBoost_Basic'
results_opt_df['Model'] = 'XGBoost_Optimized'
all_results = pd.concat([results_basic_df, results_opt_df])

# 重新排列列顺序
all_results = all_results[['Model', 'Dataset', 'R²', 'RMSE', 'MAE', 'MAPE']]

print("\n所有模型性能对比:")
print(all_results.to_string(index=False))

# 测试集性能对比
test_results = all_results[all_results['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\n测试集性能排名:")
print(test_results.to_string(index=False))

# 性能提升
basic_test_r2 = results_basic_df[results_basic_df['Dataset'] == 'Test']['R²'].values[0]
opt_test_r2 = results_opt_df[results_opt_df['Dataset'] == 'Test']['R²'].values[0]
basic_test_rmse = results_basic_df[results_basic_df['Dataset'] == 'Test']['RMSE'].values[0]
opt_test_rmse = results_opt_df[results_opt_df['Dataset'] == 'Test']['RMSE'].values[0]

r2_improvement = (opt_test_r2 - basic_test_r2) / basic_test_r2 * 100
rmse_improvement = (basic_test_rmse - opt_test_rmse) / basic_test_rmse * 100

print(f"\n优化效果:")
print(f"  R²提升: {r2_improvement:.2f}%")
print(f"  RMSE降低: {rmse_improvement:.2f}%")

# ============================== 第11部分: 特征重要性分析 ==============================
print("\n" + "=" * 80)
print("第8步: 特征重要性分析")
print("=" * 80)

# 获取特征名称（从训练数据中）
feature_names = X_train.columns.tolist()

# 获取特征重要性（XGBoost方法）
# weight: 特征在所有树中被用作分割的次数
# gain: 特征带来的平均增益
# cover: 特征覆盖的样本数量
importance_weight = model_optimized.get_booster().get_score(importance_type='weight')
importance_gain = model_optimized.get_booster().get_score(importance_type='gain')

# 创建特征重要性DataFrame
feature_importance_data = []
for feature in feature_names:
    # XGBoost默认用f0, f1, f2...作为特征名，需要映射
    feature_idx = f'f{feature_names.index(feature)}'
    weight = importance_weight.get(feature_idx, 0)
    gain = importance_gain.get(feature_idx, 0)
    feature_importance_data.append({
        'Feature': feature,
        'Importance_Weight': weight,
        'Importance_Gain': gain
    })

feature_importance = pd.DataFrame(feature_importance_data)

# 归一化重要性
feature_importance['Importance_Weight_Norm'] = (feature_importance['Importance_Weight'] / 
                                                 feature_importance['Importance_Weight'].sum() * 100)
feature_importance['Importance_Gain_Norm'] = (feature_importance['Importance_Gain'] / 
                                               feature_importance['Importance_Gain'].sum() * 100)

# 按增益重要性排序
feature_importance = feature_importance.sort_values('Importance_Gain', ascending=False)

print(f"\nTop 20 重要特征 (按Gain):")
print(feature_importance.head(20)[['Feature', 'Importance_Gain_Norm']].to_string(index=False))

# ============================== 第12部分: 可视化 ==============================
print("\n" + "=" * 80)
print("第9步: 生成可视化图表")
print("=" * 80)

# 12.1 训练过程曲线
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 基础模型
# 新版本XGBoost使用 'validation_0', 'validation_1' 作为键名
if 'validation_0' in evals_result_basic:
    # 新版本API
    axes[0].plot(evals_result_basic['validation_0']['rmse'], label='训练集', linewidth=2)
    axes[0].plot(evals_result_basic['validation_1']['rmse'], label='验证集', linewidth=2)
elif 'train' in evals_result_basic:
    # 旧版本API
    axes[0].plot(evals_result_basic['train']['rmse'], label='训练集', linewidth=2)
    axes[0].plot(evals_result_basic['valid']['rmse'], label='验证集', linewidth=2)

if hasattr(model_basic, 'best_iteration') and model_basic.best_iteration is not None:
    axes[0].axvline(x=model_basic.best_iteration, color='r', linestyle='--', 
                    label=f'最佳迭代({model_basic.best_iteration})', linewidth=1.5)
axes[0].set_xlabel('迭代次数', fontsize=12)
axes[0].set_ylabel('RMSE', fontsize=12)
axes[0].set_title('XGBoost基础模型 - 训练过程', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 优化模型
if 'validation_0' in evals_result_opt:
    # 新版本API
    axes[1].plot(evals_result_opt['validation_0']['rmse'], label='训练集', linewidth=2)
    axes[1].plot(evals_result_opt['validation_1']['rmse'], label='验证集', linewidth=2)
elif 'train' in evals_result_opt:
    # 旧版本API
    axes[1].plot(evals_result_opt['train']['rmse'], label='训练集', linewidth=2)
    axes[1].plot(evals_result_opt['valid']['rmse'], label='验证集', linewidth=2)

if hasattr(model_optimized, 'best_iteration') and model_optimized.best_iteration is not None:
    axes[1].axvline(x=model_optimized.best_iteration, color='r', linestyle='--',
                    label=f'最佳迭代({model_optimized.best_iteration})', linewidth=1.5)
axes[1].set_xlabel('迭代次数', fontsize=12)
axes[1].set_ylabel('RMSE', fontsize=12)
axes[1].set_title('XGBoost优化模型 - 训练过程', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
print("保存: training_curves.png")
plt.close()

# 12.2 预测vs实际值散点图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

models_data = [
    ('Basic', y_train_pred_basic, y_train, 'Train'),
    ('Basic', y_val_pred_basic, y_val, 'Val'),
    ('Basic', y_test_pred_basic, y_test, 'Test'),
    ('Optimized', y_train_pred_opt, y_train, 'Train'),
    ('Optimized', y_val_pred_opt, y_val, 'Val'),
    ('Optimized', y_test_pred_opt, y_test, 'Test')
]

for idx, (model_name, y_pred, y_true, dataset) in enumerate(models_data):
    row = idx // 3
    col = idx % 3
    
    ax = axes[row, col]
    
    # 散点图
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    
    # 理想预测线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测线')
    
    # 计算指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    ax.set_xlabel('实际PM2.5浓度 (μg/m³)', fontsize=11)
    ax.set_ylabel('预测PM2.5浓度 (μg/m³)', fontsize=11)
    ax.set_title(f'XGBoost_{model_name} - {dataset}\nR²={r2:.4f}, RMSE={rmse:.2f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
print("保存: prediction_scatter.png")
plt.close()

# 12.3 时间序列预测对比
fig, axes = plt.subplots(2, 1, figsize=(18, 10))

# 测试集 - 基础模型
plot_range = min(300, len(y_test))
plot_idx = range(len(y_test) - plot_range, len(y_test))
time_idx = y_test.index[plot_idx]

axes[0].plot(time_idx, y_test.iloc[plot_idx], 'k-', label='实际值', 
             linewidth=2, alpha=0.8)
axes[0].plot(time_idx, y_test_pred_basic[plot_idx], 'b--', label='基础模型预测', 
             linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('日期', fontsize=12)
axes[0].set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
axes[0].set_title('XGBoost基础模型 - 时间序列预测对比（测试集最后300天）', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

# 测试集 - 优化模型
axes[1].plot(time_idx, y_test.iloc[plot_idx], 'k-', label='实际值', 
             linewidth=2, alpha=0.8)
axes[1].plot(time_idx, y_test_pred_opt[plot_idx], 'g--', label='优化模型预测', 
             linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('日期', fontsize=12)
axes[1].set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
axes[1].set_title('XGBoost优化模型 - 时间序列预测对比（测试集最后300天）', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(output_dir / 'timeseries_comparison.png', dpi=300, bbox_inches='tight')
print("保存: timeseries_comparison.png")
plt.close()

# 12.4 残差分析
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (model_name, y_pred, y_true, dataset) in enumerate(models_data):
    row = idx // 3
    col = idx % 3
    
    ax = axes[row, col]
    
    residuals = y_true - y_pred
    
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('预测值 (μg/m³)', fontsize=11)
    ax.set_ylabel('残差 (μg/m³)', fontsize=11)
    ax.set_title(f'XGBoost_{model_name} - {dataset}\n残差均值={residuals.mean():.2f}, 标准差={residuals.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
print("保存: residuals_analysis.png")
plt.close()

# 12.5 特征重要性图
fig, axes = plt.subplots(1, 2, figsize=(16, 10))

top_n = 20
top_features_gain = feature_importance.head(top_n)
top_features_weight = feature_importance.sort_values('Importance_Weight', ascending=False).head(top_n)

# 按Gain排序
axes[0].barh(range(top_n), top_features_gain['Importance_Gain_Norm'], color='steelblue')
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(top_features_gain['Feature'], fontsize=10)
axes[0].set_xlabel('重要性 (%)', fontsize=12)
axes[0].set_title(f'Top {top_n} 重要特征 (按Gain)', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')
axes[0].invert_yaxis()

# 按Weight排序
axes[1].barh(range(top_n), top_features_weight['Importance_Weight_Norm'], color='coral')
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(top_features_weight['Feature'], fontsize=10)
axes[1].set_xlabel('重要性 (%)', fontsize=12)
axes[1].set_title(f'Top {top_n} 重要特征 (按Weight)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
print("保存: feature_importance.png")
plt.close()

# 12.6 模型性能对比柱状图
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

test_results = all_results[all_results['Dataset'] == 'Test']
models = test_results['Model'].tolist()
x_pos = np.arange(len(models))
colors = ['blue', 'green']

metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
for i, metric in enumerate(metrics):
    axes[i].bar(x_pos, test_results[metric], color=colors, alpha=0.7, 
                edgecolor='black', linewidth=1.5)
    axes[i].set_xticks(x_pos)
    axes[i].set_xticklabels(['Basic', 'Optimized'], fontsize=11)
    axes[i].set_ylabel(metric, fontsize=12)
    
    if metric == 'R²':
        axes[i].set_title(f'{metric} 对比\n(越大越好)', fontsize=12, fontweight='bold')
    else:
        axes[i].set_title(f'{metric} 对比\n(越小越好)', fontsize=12, fontweight='bold')
    
    axes[i].grid(True, alpha=0.3, axis='y')
    
    # 显示数值
    for j, v in enumerate(test_results[metric]):
        if metric == 'MAPE':
            axes[i].text(j, v, f'{v:.1f}%', ha='center', va='bottom', 
                         fontsize=10, fontweight='bold')
        else:
            axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom', 
                         fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
print("保存: model_comparison.png")
plt.close()

# 12.7 误差分布直方图
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

errors_basic = y_test - y_test_pred_basic
errors_opt = y_test - y_test_pred_opt

axes[0].hist(errors_basic, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='零误差')
axes[0].set_xlabel('预测误差 (μg/m³)', fontsize=12)
axes[0].set_ylabel('频数', fontsize=12)
axes[0].set_title(f'基础模型 - 预测误差分布\n均值={errors_basic.mean():.2f}, 标准差={errors_basic.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].hist(errors_opt, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='零误差')
axes[1].set_xlabel('预测误差 (μg/m³)', fontsize=12)
axes[1].set_ylabel('频数', fontsize=12)
axes[1].set_title(f'优化模型 - 预测误差分布\n均值={errors_opt.mean():.2f}, 标准差={errors_opt.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
print("保存: error_distribution.png")
plt.close()

# ============================== 第13部分: 保存结果 ==============================
print("\n" + "=" * 80)
print("第10步: 保存结果")
print("=" * 80)

# 保存模型性能
all_results.to_csv(output_dir / 'model_performance.csv', index=False, encoding='utf-8-sig')
print("保存: model_performance.csv")

# 保存特征重要性
feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False, encoding='utf-8-sig')
print("保存: feature_importance.csv")

# 保存最佳参数
best_params_df = pd.DataFrame([params_optimized])
best_params_df.to_csv(output_dir / 'best_parameters.csv', index=False, encoding='utf-8-sig')
print("保存: best_parameters.csv")

# 保存预测结果
predictions_df = pd.DataFrame({
    'Date': y_test.index,
    'Actual': y_test.values,
    'Prediction_Basic': y_test_pred_basic,
    'Prediction_Optimized': y_test_pred_opt,
    'Error_Basic': y_test.values - y_test_pred_basic,
    'Error_Optimized': y_test.values - y_test_pred_opt
})
predictions_df.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
print("保存: predictions.csv")

# 保存模型
model_optimized.save_model(str(model_dir / 'xgboost_optimized.txt'))
print("保存: xgboost_optimized.txt")

# 使用pickle保存模型（可选）
with open(model_dir / 'xgboost_optimized.pkl', 'wb') as f:
    pickle.dump(model_optimized, f)
print("保存: xgboost_optimized.pkl")

# ============================== 第14部分: 总结报告 ==============================
print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)

print("\n生成的文件:")
print("\nCSV文件:")
print("  - model_performance.csv       模型性能对比")
print("  - feature_importance.csv      特征重要性")
print("  - best_parameters.csv         最佳参数")
print("  - predictions.csv             预测结果")

print("\n图表文件:")
print("  - training_curves.png         训练过程曲线")
print("  - prediction_scatter.png      预测vs实际散点图")
print("  - timeseries_comparison.png   时间序列对比")
print("  - residuals_analysis.png      残差分析")
print("  - feature_importance.png      特征重要性图")
print("  - model_comparison.png        模型性能对比")
print("  - error_distribution.png      误差分布")

print("\n模型文件:")
print("  - xgboost_optimized.txt      XGBoost模型（文本格式）")
print("  - xgboost_optimized.pkl      XGBoost模型（pickle格式）")

# 最佳模型信息
best_model = test_results.iloc[0]
print(f"\n最佳模型: {best_model['Model']}")
print(f"  R² Score: {best_model['R²']:.4f}")
print(f"  RMSE: {best_model['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model['MAPE']:.2f}%")

print("\nTop 5 最重要特征:")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance_Gain_Norm']:.2f}%")

print("\n" + "=" * 80)
print("XGBoost PM2.5浓度预测完成！")
print("=" * 80)