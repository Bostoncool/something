import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn风格
sns.set_style("whitegrid")
sns.set_palette("husl")

def process_single_file(file_path):
    """
    处理单个CSV文件，提取年月和城市污染物数据

    参数:
    file_path: CSV文件路径

    返回:
    (month_key, month_data) 元组，month_data是字典 {city: {pollutant: value}}
    month_key格式为'YYYY-MM'
    """
    try:
        # 从文件名提取年月
        file_name = Path(file_path).stem
        # 假设文件名格式为: china_cities_YYYYMMDD.csv
        # 提取YYYYMM部分，然后转换为YYYY-MM格式
        date_str = file_name.split('_')[-1][:8]  # 获取YYYYMMDD
        year_month = f"{date_str[:4]}-{date_str[4:6]}"  # 转换为YYYY-MM格式

        # 读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8-sig')

        # 提取城市列（排除元数据列）
        city_columns = [col for col in df.columns if col not in ['date', 'hour', 'type', '__file__', '__missing_cols__']]

        # 按污染物类型和城市分组计算日均值
        # 只处理PM2.5数据（不包括PM2.5_24h）
        realtime_types = ['PM2.5']
        df_realtime = df[df['type'].isin(realtime_types)].copy()

        # 按小时计算平均值（对于每天的数据）
        daily_avg = df_realtime.groupby('type')[city_columns].mean()

        # 存储每月的平均数据
        month_data = {}

        # 对于每个城市，存储各污染物的月均值
        for city in city_columns:
            month_data[city] = {}
            for pollutant in realtime_types:
                if pollutant in daily_avg.index:
                    value = daily_avg.loc[pollutant, city]
                    if not pd.isna(value):
                        month_data[city][pollutant] = value

        return (year_month, month_data)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def calculate_monthly_means(folder_path, n_processes=None):
    """
    从原始CSV文件计算每个城市的月度平均值

    参数:
    folder_path: CSV文件所在文件夹路径
    n_processes: 进程数，默认为CPU核心数

    返回:
    pandas.DataFrame: 包含城市、污染物、月份和值的DataFrame
    """
    # 存储所有月度数据的字典
    all_data = {}

    # 1. 遍历文件夹中的所有CSV文件
    csv_files = list(Path(folder_path).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    if len(csv_files) == 0:
        print(f"Error: No CSV files found in {folder_path}")
        return None

    # 确定进程数
    if n_processes is None:
        n_processes = cpu_count()
    print(f"Using {n_processes} processes for parallel processing...")

    # 使用多进程处理文件
    print("Reading and processing CSV files...")
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, csv_files),
            total=len(csv_files),
            desc="Processing CSV files",
            unit="files"
        ))

    # 合并结果 - 收集同一个月所有天的数据
    # all_data结构: {month_key: {city: {pollutant: [values...]}}}
    for result in results:
        if result is not None:
            month_key, month_data = result
            if month_key not in all_data:
                all_data[month_key] = {}
            # 对于每个城市和污染物，累积所有天的值
            for city, pollutants in month_data.items():
                if city not in all_data[month_key]:
                    all_data[month_key][city] = {}
                for pollutant, value in pollutants.items():
                    if pollutant not in all_data[month_key][city]:
                        all_data[month_key][city][pollutant] = []
                    all_data[month_key][city][pollutant].append(value)
    
    # 计算每个月的平均值
    # 将 all_data 转换为 {month_key: {city: {pollutant: avg_value}}}
    monthly_avg_data = {}
    for month_key, month_data in all_data.items():
        monthly_avg_data[month_key] = {}
        for city, pollutants in month_data.items():
            monthly_avg_data[month_key][city] = {}
            for pollutant, values in pollutants.items():
                if values:  # 确保有数据
                    avg_value = np.mean(values)
                    monthly_avg_data[month_key][city][pollutant] = avg_value
    
    # 使用计算好的月度平均值数据
    all_data = monthly_avg_data

    # 整理数据为DataFrame格式
    months = sorted(all_data.keys())
    print(f"Processed {len(months)} months of data: {months[0]} to {months[-1]}")

    # 只处理PM2.5数据（不包括PM2.5_24h）
    realtime_types = ['PM2.5']

    # 获取所有城市
    all_cities = set()
    for month_data in all_data.values():
        all_cities.update(month_data.keys())

    print(f"Found {len(all_cities)} cities")

    # 构建DataFrame
    data_list = []
    for city in all_cities:
        for pollutant in realtime_types:
            for month in months:
                if month in all_data and city in all_data[month]:
                    if pollutant in all_data[month][city]:
                        value = all_data[month][city][pollutant]
                        data_list.append({
                            'City': city,
                            'Pollutant': pollutant,
                            'Month': pd.to_datetime(month, format='%Y-%m'),
                            'Value': value
                        })

    if not data_list:
        print("No data found")
        return None

    df = pd.DataFrame(data_list)
    print(f"Created DataFrame with {len(df)} records")
    
    return df

def prepare_time_series_data(df, city, pollutant):
    """
    为STL分解准备时间序列数据

    参数:
    df: 包含所有数据的DataFrame
    city: 城市名称
    pollutant: 污染物类型

    返回:
    pandas.Series: 时间序列数据，按月份索引
    """
    # 筛选指定城市和污染物的月度数据
    mask = (df['City'] == city) & (df['Pollutant'] == pollutant)
    city_pollutant_data = df[mask].copy()

    if city_pollutant_data.empty:
        return None

    # 按月份排序
    city_pollutant_data = city_pollutant_data.sort_values('Month')

    # 创建时间序列
    ts_data = pd.Series(
        city_pollutant_data['Value'].values,
        index=city_pollutant_data['Month'],
        name=f'{city}_{pollutant}'
    )

    return ts_data

def perform_stl_decomposition(ts_data, seasonal_period=12):
    """
    执行STL分解

    参数:
    ts_data: 时间序列数据 (pandas.Series)
    seasonal_period: 季节性周期，默认为12（月度数据）

    返回:
    statsmodels.tsa.seasonal.DecomposeResult: STL分解结果
    """
    try:
        # 确保数据长度足够
        data_length = len(ts_data)
        
        # 确保数据长度至少是周期的2倍
        if data_length < 2 * seasonal_period:
            # 如果数据不足，尝试使用较小的周期
            adjusted_period = max(2, data_length // 2)
            if adjusted_period < seasonal_period:
                print(f"    Adjusting seasonal period from {seasonal_period} to {adjusted_period} (data length: {data_length})")
                seasonal_period = adjusted_period
        
        # 再次检查数据长度
        if data_length < 2 * seasonal_period:
            raise ValueError(f"Data length ({data_length}) must be at least 2 times the seasonal period ({seasonal_period})")
        
        # STL要求seasonal参数必须是奇数且>=3
        # 如果seasonal_period是偶数，调整为最接近的奇数
        if seasonal_period % 2 == 0:
            # 对于月度数据，12个月调整为13（最接近的奇数）
            seasonal_period = seasonal_period + 1
            print(f"    Adjusting seasonal period to odd number: {seasonal_period}")
        
        # 确保seasonal_period >= 3
        if seasonal_period < 3:
            seasonal_period = 3
            print(f"    Adjusting seasonal period to minimum value: {seasonal_period}")
        
        # 确保数据是连续的（没有缺失的月份）
        # 对于STL分解，我们需要连续的时间序列
        ts_data_clean = ts_data.dropna()
        if len(ts_data_clean) < len(ts_data) * 0.8:  # 如果缺失超过20%的数据
            print(f"    Warning: {len(ts_data) - len(ts_data_clean)} missing values found")
        
        # 执行STL分解
        # statsmodels的STL使用seasonal参数指定季节性周期（必须是奇数且>=3）
        stl = STL(ts_data_clean, seasonal=seasonal_period, robust=True)
        result = stl.fit()

        return result

    except Exception as e:
        print(f"    STL decomposition failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_stl_components(result, city, pollutant, save_path=None):
    """
    绘制STL分解的各个成分

    参数:
    result: STL分解结果
    city: 城市名称
    pollutant: 污染物类型
    save_path: 保存路径，如果为None则显示图像
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    # 原始数据
    axes[0].plot(result.observed, 'b-', linewidth=1.5, alpha=0.8)
    axes[0].set_title(f'{city} - {pollutant} Original Time Series', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Concentration', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # 趋势成分
    axes[1].plot(result.trend, 'r-', linewidth=1.5, alpha=0.8)
    axes[1].set_title('Trend Component', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Trend Value', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # 季节性成分
    axes[2].plot(result.seasonal, 'g-', linewidth=1.5, alpha=0.8)
    axes[2].set_title('Seasonal Component', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Seasonal Value', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    # 残差成分
    axes[3].plot(result.resid, 'orange', linewidth=1, alpha=0.7)
    axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[3].set_title('Residual Component', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('Residual Value', fontsize=12)
    axes[3].set_xlabel('Time', fontsize=12)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_seasonal_analysis(result, city, pollutant, save_path=None):
    """
    分析季节性模式的详细图表

    参数:
    result: STL分解结果
    city: 城市名称
    pollutant: 污染物类型
    save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 季节性成分箱线图（按月份）
    seasonal_df = pd.DataFrame({
        'Month': result.seasonal.index.month,
        'Seasonal_Value': result.seasonal.values
    })

    sns.boxplot(data=seasonal_df, x='Month', y='Seasonal_Value', ax=axes[0,0])
    axes[0,0].set_title(f'{city} - {pollutant} Seasonal Component by Month', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Month', fontsize=10)
    axes[0,0].set_ylabel('Seasonal Value', fontsize=10)
    axes[0,0].grid(True, alpha=0.3)

    # 季节性成分的小提琴图
    sns.violinplot(data=seasonal_df, x='Month', y='Seasonal_Value', ax=axes[0,1])
    axes[0,1].set_title('Seasonal Component Distribution Density', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Month', fontsize=10)
    axes[0,1].set_ylabel('Seasonal Value', fontsize=10)
    axes[0,1].grid(True, alpha=0.3)

    # 趋势成分的变化率
    trend_changes = result.trend.pct_change() * 100
    axes[1,0].plot(trend_changes.index, trend_changes.values, 'b-', alpha=0.7)
    axes[1,0].set_title('Trend Component Monthly Change Rate', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Time', fontsize=10)
    axes[1,0].set_ylabel('Change Rate (%)', fontsize=10)
    axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1,0].grid(True, alpha=0.3)

    # 残差分布直方图
    axes[1,1].hist(result.resid.values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1,1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[1,1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Residual Value', fontsize=10)
    axes[1,1].set_ylabel('Frequency', fontsize=10)
    axes[1,1].grid(True, alpha=0.3)

    plt.suptitle(f'{city} - {pollutant} STL Decomposition Detailed Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_stl_results(result, city, pollutant):
    """
    分析STL分解结果并返回统计信息

    参数:
    result: STL分解结果
    city: 城市名称
    pollutant: 污染物类型

    返回:
    dict: 包含分析结果的字典
    """
    analysis = {
        'city': city,
        'pollutant': pollutant,
        'trend_strength': None,
        'seasonal_strength': None,
        'residual_strength': None,
        'trend_direction': None,
        'seasonal_peak_month': None,
        'seasonal_valley_month': None,
        'data_range': None,
        'trend_range': None,
        'seasonal_range': None
    }

    try:
        # 计算各成分的强度（变异系数）
        observed_var = np.var(result.observed)
        trend_var = np.var(result.trend)
        seasonal_var = np.var(result.seasonal)
        residual_var = np.var(result.resid)

        # 趋势强度
        analysis['trend_strength'] = trend_var / observed_var if observed_var > 0 else 0

        # 季节性强度
        analysis['seasonal_strength'] = seasonal_var / (observed_var - trend_var) if (observed_var - trend_var) > 0 else 0

        # 残差强度
        analysis['residual_strength'] = residual_var / observed_var if observed_var > 0 else 0

        # 趋势方向
        trend_start = result.trend.iloc[0]
        trend_end = result.trend.iloc[-1]
        if trend_end > trend_start * 1.05:  # 上升5%以上
            analysis['trend_direction'] = 'Increasing'
        elif trend_end < trend_start * 0.95:  # 下降5%以上
            analysis['trend_direction'] = 'Decreasing'
        else:
            analysis['trend_direction'] = 'Stable'

        # 季节性峰谷月份
        seasonal_by_month = result.seasonal.groupby(result.seasonal.index.month).mean()
        analysis['seasonal_peak_month'] = seasonal_by_month.idxmax()
        analysis['seasonal_valley_month'] = seasonal_by_month.idxmin()

        # 数据范围
        analysis['data_range'] = f"{result.observed.min():.2f} - {result.observed.max():.2f}"
        analysis['trend_range'] = f"{result.trend.min():.2f} - {result.trend.max():.2f}"
        analysis['seasonal_range'] = f"{result.seasonal.min():.2f} - {result.seasonal.max():.2f}"

    except Exception as e:
        print(f"Error analyzing results: {e}")

    return analysis

def save_analysis_results(analyses, output_dir):
    """
    保存分析结果到CSV文件

    参数:
    analyses: 分析结果列表
    output_dir: 输出目录
    """
    if not analyses:
        print("No analysis results to save")
        return

    # 转换为DataFrame
    df_results = pd.DataFrame(analyses)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存结果
    output_path = os.path.join(output_dir, 'STL_Decomposition_Analysis.csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"STL decomposition analysis results saved to: {output_path}")

def main():
    """
    主函数：执行STL分解分析
    """
    print("="*60)
    print("PM2.5 Time Series STL Decomposition Analysis")
    print("="*60)

    # 设置原始CSV文件所在文件夹路径
    # 参考PRD-月度变化.py的输入路径
    input_folder = r"E:\DATA Science\大论文Result\PRD\filtered_daily"

    # 设置输出目录
    output_dir = r"E:\DATA Science\大论文Result\PRD\时间序列分解"

    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"Error: Input folder does not exist: {input_folder}")
        print("Please ensure the folder path is correct")
        return

    # 从原始CSV文件计算月度平均值
    print(f"Calculating monthly means from CSV files in: {input_folder}")
    df = calculate_monthly_means(input_folder)

    if df is None or df.empty:
        print("Unable to calculate monthly means from data files")
        return

    print(f"Successfully calculated monthly means, {len(df)} records total")
    print(f"Data time range: {df['Month'].min()} to {df['Month'].max()}")

    # 获取所有城市和污染物
    cities = df['City'].unique()
    pollutants = df['Pollutant'].unique()

    print(f"\nFound {len(cities)} cities: {list(cities)}")
    print(f"Found {len(pollutants)} pollutants: {list(pollutants)}")

    # 只分析PM2.5数据
    target_pollutant = 'PM2.5'
    if target_pollutant not in pollutants:
        print(f"Error: {target_pollutant} data not found")
        return

    print(f"\nStarting STL decomposition analysis for {target_pollutant}...")
    
    # 检查每个城市的数据情况
    print("\nChecking data availability for each city:")
    city_data_counts = {}
    for city in cities:
        city_pollutant_df = df[(df['City'] == city) & (df['Pollutant'] == target_pollutant)]
        count = len(city_pollutant_df)
        if count > 0:
            months = sorted(city_pollutant_df['Month'].unique())
            city_data_counts[city] = {
                'count': count,
                'months': months,
                'start': months[0] if months else None,
                'end': months[-1] if months else None
            }
            print(f"  {city}: {count} months ({months[0] if months else 'N/A'} to {months[-1] if months else 'N/A'})")
        else:
            print(f"  {city}: No data available")

    # 存储所有分析结果
    all_analyses = []

    # 为每个城市进行STL分解
    for city in tqdm(cities, desc=f"Analyzing {target_pollutant} data", unit="city"):
        try:
            # 准备时间序列数据
            ts_data = prepare_time_series_data(df, city, target_pollutant)

            if ts_data is None:
                print(f"  {city}: No time series data available")
                continue
                
            if len(ts_data) < 12:  # 至少需要12个月的数据（1个完整年度周期）
                print(f"  {city}: Insufficient data, skipping (only {len(ts_data)} months of data, need at least 12)")
                continue

            print(f"  {city}: Processing {len(ts_data)} months of data ({ts_data.index.min()} to {ts_data.index.max()})")
            
            # 检查数据是否有缺失值
            if ts_data.isna().any():
                print(f"  {city}: Warning - data contains NaN values, filling with forward fill")
                ts_data = ts_data.fillna(method='ffill').fillna(method='bfill')
            
            # 检查数据是否全为零或常数
            if ts_data.nunique() <= 1:
                print(f"  {city}: Skipping - data is constant (all values are the same)")
                continue

            # 执行STL分解
            stl_result = perform_stl_decomposition(ts_data)

            if stl_result is None:
                print(f"  {city}: STL decomposition failed")
                continue

            # 分析结果
            analysis = analyze_stl_results(stl_result, city, target_pollutant)
            all_analyses.append(analysis)

            # 创建输出目录
            city_output_dir = os.path.join(output_dir, city)
            os.makedirs(city_output_dir, exist_ok=True)

            # 保存STL分解图表
            stl_plot_path = os.path.join(city_output_dir, f'{city}_{target_pollutant}_STL_Decomposition.png')
            plot_stl_components(stl_result, city, target_pollutant, stl_plot_path)

            # 保存详细分析图表
            analysis_plot_path = os.path.join(city_output_dir, f'{city}_{target_pollutant}_Seasonal_Analysis.png')
            plot_seasonal_analysis(stl_result, city, target_pollutant, analysis_plot_path)

            print(f"    Analysis charts saved to: {city_output_dir}")

        except Exception as e:
            print(f"  {city} analysis failed: {e}")
            continue

    # 保存综合分析结果
    save_analysis_results(all_analyses, output_dir)

    # 打印分析摘要
    print(f"\n{'='*60}")
    print("STL Decomposition Analysis Summary")
    print(f"{'='*60}")

    if all_analyses:
        df_summary = pd.DataFrame(all_analyses)

        print(f"\nSuccessfully analyzed {len(all_analyses)} cities for {target_pollutant} data")
        print("\nCity Trend Analysis:")
        for _, row in df_summary.iterrows():
            print(f"  {row['city']}: "
                  f"Trend {row['trend_direction']} "
                  f"(Strength: {row['trend_strength']:.3f}) "
                  f"Seasonal Peak Month: {row['seasonal_peak_month']} "
                  f"Valley Month: {row['seasonal_valley_month']}")

        print("\nTrend Strength Statistics:")
        print(f"  Average: {df_summary['trend_strength'].mean():.3f}")
        print(f"  Maximum: {df_summary['trend_strength'].max():.3f} ({df_summary.loc[df_summary['trend_strength'].idxmax(), 'city']})")
        print(f"  Minimum: {df_summary['trend_strength'].min():.3f} ({df_summary.loc[df_summary['trend_strength'].idxmin(), 'city']})")

        print("\nSeasonal Strength Statistics:")
        print(f"  Average: {df_summary['seasonal_strength'].mean():.3f}")
        print(f"  Maximum: {df_summary['seasonal_strength'].max():.3f} ({df_summary.loc[df_summary['seasonal_strength'].idxmax(), 'city']})")
        print(f"  Minimum: {df_summary['seasonal_strength'].min():.3f} ({df_summary.loc[df_summary['seasonal_strength'].idxmin(), 'city']})")
    else:
        print("No successful analysis results")

    print(f"\nAnalysis results saved to directory: {output_dir}")
    print("STL Decomposition Analysis Complete!")

if __name__ == "__main__":
    main()