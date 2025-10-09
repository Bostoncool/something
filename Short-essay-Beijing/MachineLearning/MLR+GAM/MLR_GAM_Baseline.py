"""
北京PM2.5浓度预测 - 线性基线模型
多元线性回归(MLR)和广义加性模型(GAM)

基于相关性分析结果，选择相关性较强的气象变量建立基线模型
主要气象变量：温度(t2m)、露点温度(d2m)、风速(u10/v10)、湿度(tcwv)、降水(tp)、边界层高度(blh)等
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# GAM库
try:
    from pygam import LinearGAM, s, f, te
    GAM_AVAILABLE = True
except ImportError:
    print("警告: pygam未安装，将跳过GAM模型。请使用 'pip install pygam' 安装。")
    GAM_AVAILABLE = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 设置随机种子
np.random.seed(42)

print("=" * 80)
print("北京PM2.5浓度预测 - 线性基线模型")
print("=" * 80)

# ============================== 第1部分: 定义路径和参数 ==============================
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

# 定义日期范围
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

# 北京地理范围
beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)

# 定义污染物
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

# 定义ERA5变量 (基于相关性分析结果选择重要变量)
# 完整列表，后续会筛选
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
print(f"气象特征数: {len(era5_vars)}")
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
        # print(f"  成功读取: {year}-{month:02d}, 日数: {len(monthly_data)}, 变量数: {len(set(loaded_vars))}")
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
    
    # 使用并行线程加载ERA5数据
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

# ============================== 第3部分: 数据加载和预处理 ==============================
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

# 计算风速大小（从风分量）
if 'u10' in df_combined and 'v10' in df_combined:
    df_combined['wind_speed_10m'] = np.sqrt(df_combined['u10']**2 + df_combined['v10']**2)
    df_combined['wind_dir_10m'] = np.arctan2(df_combined['v10'], df_combined['u10']) * 180 / np.pi
    
if 'u100' in df_combined and 'v100' in df_combined:
    df_combined['wind_speed_100m'] = np.sqrt(df_combined['u100']**2 + df_combined['v100']**2)

# 添加时间特征
df_combined['month'] = df_combined.index.month
df_combined['season'] = df_combined['month'].apply(
    lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
)
df_combined['day_of_year'] = df_combined.index.dayofyear

# 清理数据
df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
df_combined.dropna(inplace=True)

print(f"\n合并后数据形状: {df_combined.shape}")
print(f"时间范围: {df_combined.index.min().date()} 至 {df_combined.index.max().date()}")
print(f"样本数: {len(df_combined)}")

# ============================== 第4部分: 特征选择 ==============================
print("\n" + "=" * 80)
print("第2步: 特征选择 - 基于相关性分析结果")
print("=" * 80)

# 定义目标变量
target = 'PM2.5'

# 基于相关性分析结果选择特征
# 从Pearson和Kendall相关性分析中，与PM2.5相关性较强的特征：
# d2m (露点温度), t2m (温度), u10/v10 (风速), tcwv (水汽), tp (降水), blh (边界层高度), sp (气压), str (辐射)
selected_features = [
    'd2m',  # 露点温度 - 相关性最强
    't2m',  # 2米温度
    'wind_speed_10m',  # 10米风速（计算得到）
    'tcwv',  # 总柱水汽
    'tp',  # 降水
    'blh',  # 边界层高度
    'sp',  # 气压
    'str',  # 表面热辐射
    'tisr',  # 太阳辐射
    'month',  # 月份（时间特征）
    'season',  # 季节（时间特征）
]

# 检查特征是否存在
available_features = [f for f in selected_features if f in df_combined.columns]
missing_features = [f for f in selected_features if f not in df_combined.columns]

if missing_features:
    print(f"\n警告: 以下特征不存在，将被跳过: {missing_features}")

print(f"\n选择的特征 ({len(available_features)} 个):")
for i, feat in enumerate(available_features, 1):
    print(f"  {i}. {feat}")

# 准备建模数据
X = df_combined[available_features].copy()
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

print(f"目标变量统计:")
print(f"  均值: {y.mean():.2f} μg/m³")
print(f"  标准差: {y.std():.2f} μg/m³")
print(f"  最小值: {y.min():.2f} μg/m³")
print(f"  最大值: {y.max():.2f} μg/m³")
print(f"  中位数: {y.median():.2f} μg/m³")

# ============================== 第5部分: 数据集划分 ==============================
print("\n" + "=" * 80)
print("第3步: 数据集划分")
print("=" * 80)

# 按时间顺序划分，前80%作为训练集，后20%作为测试集
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"\n训练集: {len(X_train)} 样本 ({len(X_train)/len(X)*100:.1f}%)")
print(f"  时间范围: {X_train.index.min().date()} 至 {X_train.index.max().date()}")
print(f"  PM2.5: {y_train.mean():.2f} ± {y_train.std():.2f} μg/m³")

print(f"\n测试集: {len(X_test)} 样本 ({len(X_test)/len(X)*100:.1f}%)")
print(f"  时间范围: {X_test.index.min().date()} 至 {X_test.index.max().date()}")
print(f"  PM2.5: {y_test.mean():.2f} ± {y_test.std():.2f} μg/m³")

# ============================== 第6部分: 模型训练 - 多元线性回归 (MLR) ==============================
print("\n" + "=" * 80)
print("模型1: 多元线性回归 (Multiple Linear Regression, MLR)")
print("=" * 80)

# 创建管道：标准化 + 线性回归
mlr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# 训练模型
print("\n训练MLR模型...")
mlr_pipeline.fit(X_train, y_train)

# 预测
y_train_pred_mlr = mlr_pipeline.predict(X_train)
y_test_pred_mlr = mlr_pipeline.predict(X_test)

# 评估
train_r2_mlr = r2_score(y_train, y_train_pred_mlr)
train_rmse_mlr = np.sqrt(mean_squared_error(y_train, y_train_pred_mlr))
train_mae_mlr = mean_absolute_error(y_train, y_train_pred_mlr)

test_r2_mlr = r2_score(y_test, y_test_pred_mlr)
test_rmse_mlr = np.sqrt(mean_squared_error(y_test, y_test_pred_mlr))
test_mae_mlr = mean_absolute_error(y_test, y_test_pred_mlr)

print("\nMLR模型性能:")
print(f"  训练集:")
print(f"    R² Score: {train_r2_mlr:.4f}")
print(f"    RMSE: {train_rmse_mlr:.4f}")
print(f"    MAE: {train_mae_mlr:.4f}")
print(f"  测试集:")
print(f"    R² Score: {test_r2_mlr:.4f}")
print(f"    RMSE: {test_rmse_mlr:.4f}")
print(f"    MAE: {test_mae_mlr:.4f}")

# 获取特征系数
mlr_coef = mlr_pipeline.named_steps['regressor'].coef_
mlr_intercept = mlr_pipeline.named_steps['regressor'].intercept_

print(f"\n模型参数:")
print(f"  截距: {mlr_intercept:.4f}")
print(f"  特征系数:")
for feat, coef in zip(available_features, mlr_coef):
    print(f"    {feat}: {coef:.4f}")

# 交叉验证
print("\n进行5折交叉验证...")
cv_scores_mlr = cross_val_score(mlr_pipeline, X_train, y_train, 
                                  cv=5, scoring='r2')
print(f"  交叉验证R² Scores: {cv_scores_mlr}")
print(f"  平均R²: {cv_scores_mlr.mean():.4f} ± {cv_scores_mlr.std():.4f}")

# ============================== 第7部分: 模型训练 - Ridge回归 ==============================
print("\n" + "=" * 80)
print("模型2: Ridge回归 (L2正则化)")
print("=" * 80)

# 创建Ridge回归管道
ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=1.0))
])

# 训练
print("\n训练Ridge模型...")
ridge_pipeline.fit(X_train, y_train)

# 预测
y_train_pred_ridge = ridge_pipeline.predict(X_train)
y_test_pred_ridge = ridge_pipeline.predict(X_test)

# 评估
train_r2_ridge = r2_score(y_train, y_train_pred_ridge)
train_rmse_ridge = np.sqrt(mean_squared_error(y_train, y_train_pred_ridge))
train_mae_ridge = mean_absolute_error(y_train, y_train_pred_ridge)

test_r2_ridge = r2_score(y_test, y_test_pred_ridge)
test_rmse_ridge = np.sqrt(mean_squared_error(y_test, y_test_pred_ridge))
test_mae_ridge = mean_absolute_error(y_test, y_test_pred_ridge)

print("\nRidge模型性能:")
print(f"  训练集:")
print(f"    R² Score: {train_r2_ridge:.4f}")
print(f"    RMSE: {train_rmse_ridge:.4f}")
print(f"    MAE: {train_mae_ridge:.4f}")
print(f"  测试集:")
print(f"    R² Score: {test_r2_ridge:.4f}")
print(f"    RMSE: {test_rmse_ridge:.4f}")
print(f"    MAE: {test_mae_ridge:.4f}")

# ============================== 第8部分: 模型训练 - GAM ==============================
if GAM_AVAILABLE:
    print("\n" + "=" * 80)
    print("模型3: 广义加性模型 (Generalized Additive Model, GAM)")
    print("=" * 80)
    
    # 标准化数据（GAM也需要）
    scaler_gam = StandardScaler()
    X_train_scaled = scaler_gam.fit_transform(X_train)
    X_test_scaled = scaler_gam.transform(X_test)
    
    # 创建GAM模型
    # 为每个特征添加样条项
    print("\n训练GAM模型...")
    gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + f(9) + f(10))
    
    try:
        gam.gridsearch(X_train_scaled, y_train.values)
        
        # 预测
        y_train_pred_gam = gam.predict(X_train_scaled)
        y_test_pred_gam = gam.predict(X_test_scaled)
        
        # 评估
        train_r2_gam = r2_score(y_train, y_train_pred_gam)
        train_rmse_gam = np.sqrt(mean_squared_error(y_train, y_train_pred_gam))
        train_mae_gam = mean_absolute_error(y_train, y_train_pred_gam)
        
        test_r2_gam = r2_score(y_test, y_test_pred_gam)
        test_rmse_gam = np.sqrt(mean_squared_error(y_test, y_test_pred_gam))
        test_mae_gam = mean_absolute_error(y_test, y_test_pred_gam)
        
        print("\nGAM模型性能:")
        print(f"  训练集:")
        print(f"    R² Score: {train_r2_gam:.4f}")
        print(f"    RMSE: {train_rmse_gam:.4f}")
        print(f"    MAE: {train_mae_gam:.4f}")
        print(f"  测试集:")
        print(f"    R² Score: {test_r2_gam:.4f}")
        print(f"    RMSE: {test_rmse_gam:.4f}")
        print(f"    MAE: {test_mae_gam:.4f}")
        
        print(f"\n最优参数: λ = {gam.lam}")
        
    except Exception as e:
        print(f"\nGAM模型训练出错: {e}")
        GAM_AVAILABLE = False

# ============================== 第9部分: 模型比较 ==============================
print("\n" + "=" * 80)
print("模型比较")
print("=" * 80)

results = {
    'Model': ['MLR', 'Ridge'],
    'Train R²': [train_r2_mlr, train_r2_ridge],
    'Train RMSE': [train_rmse_mlr, train_rmse_ridge],
    'Train MAE': [train_mae_mlr, train_mae_ridge],
    'Test R²': [test_r2_mlr, test_r2_ridge],
    'Test RMSE': [test_rmse_mlr, test_rmse_ridge],
    'Test MAE': [test_mae_mlr, test_mae_ridge],
}

if GAM_AVAILABLE:
    results['Model'].append('GAM')
    results['Train R²'].append(train_r2_gam)
    results['Train RMSE'].append(train_rmse_gam)
    results['Train MAE'].append(train_mae_gam)
    results['Test R²'].append(test_r2_gam)
    results['Test RMSE'].append(test_rmse_gam)
    results['Test MAE'].append(test_mae_gam)

results_df = pd.DataFrame(results)
print("\n模型性能对比:")
print(results_df.to_string(index=False))

# ============================== 第10部分: 可视化 ==============================
print("\n" + "=" * 80)
print("生成可视化结果")
print("=" * 80)

# 1. 预测vs实际值散点图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# MLR
axes[0].scatter(y_test, y_test_pred_mlr, alpha=0.5, s=20)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='理想预测线')
axes[0].set_xlabel('实际PM2.5浓度 (μg/m³)', fontsize=12)
axes[0].set_ylabel('预测PM2.5浓度 (μg/m³)', fontsize=12)
axes[0].set_title(f'MLR模型\nR²={test_r2_mlr:.4f}, RMSE={test_rmse_mlr:.2f}', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Ridge
axes[1].scatter(y_test, y_test_pred_ridge, alpha=0.5, s=20, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='理想预测线')
axes[1].set_xlabel('实际PM2.5浓度 (μg/m³)', fontsize=12)
axes[1].set_ylabel('预测PM2.5浓度 (μg/m³)', fontsize=12)
axes[1].set_title(f'Ridge模型\nR²={test_r2_ridge:.4f}, RMSE={test_rmse_ridge:.2f}', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# GAM
if GAM_AVAILABLE:
    axes[2].scatter(y_test, y_test_pred_gam, alpha=0.5, s=20, color='orange')
    axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='理想预测线')
    axes[2].set_xlabel('实际PM2.5浓度 (μg/m³)', fontsize=12)
    axes[2].set_ylabel('预测PM2.5浓度 (μg/m³)', fontsize=12)
    axes[2].set_title(f'GAM模型\nR²={test_r2_gam:.4f}, RMSE={test_rmse_gam:.2f}', fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
else:
    axes[2].text(0.5, 0.5, 'GAM模型未训练', ha='center', va='center', fontsize=14)
    axes[2].axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
print("\n保存: prediction_scatter.png")
plt.close()

# 2. 时间序列预测对比
fig, ax = plt.subplots(figsize=(16, 6))

# 只绘制测试集的一部分（最后200个点）
plot_range = min(200, len(y_test))
plot_indices = range(len(y_test) - plot_range, len(y_test))
time_index = y_test.index[plot_indices]

ax.plot(time_index, y_test.iloc[plot_indices], 'k-', label='实际值', linewidth=2, alpha=0.7)
ax.plot(time_index, y_test_pred_mlr[plot_indices], 'b--', label='MLR预测', linewidth=1.5, alpha=0.7)
ax.plot(time_index, y_test_pred_ridge[plot_indices], 'g-.', label='Ridge预测', linewidth=1.5, alpha=0.7)
if GAM_AVAILABLE:
    ax.plot(time_index, y_test_pred_gam[plot_indices], 'r:', label='GAM预测', linewidth=1.5, alpha=0.7)

ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
ax.set_title('PM2.5浓度预测时间序列对比 (测试集最后200天)', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / 'prediction_timeseries.png', dpi=300, bbox_inches='tight')
print("保存: prediction_timeseries.png")
plt.close()

# 3. 残差分析
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# MLR残差
residuals_mlr = y_test - y_test_pred_mlr
axes[0].scatter(y_test_pred_mlr, residuals_mlr, alpha=0.5, s=20)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('预测值', fontsize=12)
axes[0].set_ylabel('残差', fontsize=12)
axes[0].set_title('MLR模型残差图', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Ridge残差
residuals_ridge = y_test - y_test_pred_ridge
axes[1].scatter(y_test_pred_ridge, residuals_ridge, alpha=0.5, s=20, color='green')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('预测值', fontsize=12)
axes[1].set_ylabel('残差', fontsize=12)
axes[1].set_title('Ridge模型残差图', fontsize=12)
axes[1].grid(True, alpha=0.3)

# GAM残差
if GAM_AVAILABLE:
    residuals_gam = y_test - y_test_pred_gam
    axes[2].scatter(y_test_pred_gam, residuals_gam, alpha=0.5, s=20, color='orange')
    axes[2].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[2].set_xlabel('预测值', fontsize=12)
    axes[2].set_ylabel('残差', fontsize=12)
    axes[2].set_title('GAM模型残差图', fontsize=12)
    axes[2].grid(True, alpha=0.3)
else:
    axes[2].axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
print("保存: residuals_analysis.png")
plt.close()

# 4. 特征重要性（基于MLR系数）
fig, ax = plt.subplots(figsize=(10, 6))

feature_importance = pd.DataFrame({
    'Feature': available_features,
    'Coefficient': np.abs(mlr_coef)
}).sort_values('Coefficient', ascending=True)

ax.barh(feature_importance['Feature'], feature_importance['Coefficient'], color='steelblue')
ax.set_xlabel('绝对系数值', fontsize=12)
ax.set_title('MLR模型特征重要性（绝对系数）', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
print("保存: feature_importance.png")
plt.close()

# 5. 模型性能对比条形图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = results_df['Model'].tolist()
x_pos = np.arange(len(models))

# R² Score
axes[0].bar(x_pos, results_df['Test R²'], color=['blue', 'green', 'orange'][:len(models)])
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(models)
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('测试集R² Score对比', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')

# RMSE
axes[1].bar(x_pos, results_df['Test RMSE'], color=['blue', 'green', 'orange'][:len(models)])
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(models)
axes[1].set_ylabel('RMSE', fontsize=12)
axes[1].set_title('测试集RMSE对比', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

# MAE
axes[2].bar(x_pos, results_df['Test MAE'], color=['blue', 'green', 'orange'][:len(models)])
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(models)
axes[2].set_ylabel('MAE', fontsize=12)
axes[2].set_title('测试集MAE对比', fontsize=12)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
print("保存: model_comparison.png")
plt.close()

# ============================== 第11部分: 保存结果 ==============================
print("\n" + "=" * 80)
print("保存结果")
print("=" * 80)

# 保存模型性能
results_df.to_csv(output_dir / 'model_performance.csv', index=False, encoding='utf-8-sig')
print("\n保存: model_performance.csv")

# 保存预测结果
predictions_df = pd.DataFrame({
    'Date': y_test.index,
    'Actual': y_test.values,
    'MLR_Prediction': y_test_pred_mlr,
    'Ridge_Prediction': y_test_pred_ridge,
})

if GAM_AVAILABLE:
    predictions_df['GAM_Prediction'] = y_test_pred_gam

predictions_df.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
print("保存: predictions.csv")

# 保存特征重要性
feature_importance_df = pd.DataFrame({
    'Feature': available_features,
    'MLR_Coefficient': mlr_coef,
    'Ridge_Coefficient': ridge_pipeline.named_steps['regressor'].coef_
})
feature_importance_df.to_csv(output_dir / 'feature_importance.csv', index=False, encoding='utf-8-sig')
print("保存: feature_importance.csv")

# 保存模型（使用pickle）
import pickle
with open(model_dir / 'mlr_model.pkl', 'wb') as f:
    pickle.dump(mlr_pipeline, f)
print("保存: mlr_model.pkl")

with open(model_dir / 'ridge_model.pkl', 'wb') as f:
    pickle.dump(ridge_pipeline, f)
print("保存: ridge_model.pkl")

if GAM_AVAILABLE:
    with open(model_dir / 'gam_model.pkl', 'wb') as f:
        pickle.dump((gam, scaler_gam), f)
    print("保存: gam_model.pkl")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)

print("\n生成的文件:")
print("\nCSV文件:")
print("  - model_performance.csv       模型性能对比")
print("  - predictions.csv             预测结果")
print("  - feature_importance.csv      特征重要性")

print("\n图表文件:")
print("  - prediction_scatter.png      预测vs实际散点图")
print("  - prediction_timeseries.png   时间序列预测对比")
print("  - residuals_analysis.png      残差分析图")
print("  - feature_importance.png      特征重要性图")
print("  - model_comparison.png        模型性能对比图")

print("\n模型文件:")
print("  - mlr_model.pkl               多元线性回归模型")
print("  - ridge_model.pkl             Ridge回归模型")
if GAM_AVAILABLE:
    print("  - gam_model.pkl               广义加性模型")

# 最佳模型信息
best_model_idx = results_df['Test R²'].idxmax()
best_model = results_df.loc[best_model_idx]
print(f"\n最佳模型: {best_model['Model']}")
print(f"  R² Score: {best_model['Test R²']:.4f}")
print(f"  RMSE: {best_model['Test RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model['Test MAE']:.2f} μg/m³")

print("\nTop 5 重要特征:")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Coefficient']:.4f}")

print("\n" + "=" * 80)
print("MLR+GAM基线模型训练完成！")
print("=" * 80)

