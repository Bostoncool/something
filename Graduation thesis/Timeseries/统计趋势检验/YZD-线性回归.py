import pandas as pd
import numpy as np
from scipy import stats
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_single_file(file_path):
    """
    处理单个CSV文件，提取年份和城市污染物数据
    
    参数:
    file_path: CSV文件路径
    
    返回:
    (year, year_data) 元组，year_data是字典 {city: {pollutant: value}}
    """
    try:
        # 从文件名提取年份
        file_name = Path(file_path).stem
        # 假设文件名格式为: china_cities_YYYYMMDD.csv
        # 提取YYYY部分
        year = file_name.split('_')[-1][:4]
        
        # 读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # 提取城市列（排除元数据列）
        city_columns = [col for col in df.columns if col not in ['date', 'hour', 'type', '__file__', '__missing_cols__']]
        
        # 按污染物类型和城市分组计算日均值
        # 筛选出PM2.5污染物数据（包括实时和24小时平均数据）
        pm25_types = ['PM2.5', 'PM2.5_24h']
        df_pm25 = df[df['type'].isin(pm25_types)].copy()
        
        # 按小时计算平均值（对于每天的数据）
        daily_avg = df_pm25.groupby('type')[city_columns].mean()
        
        # 存储每年的数据
        year_data = {}
        
        # 对于每个城市，存储PM2.5的年均值（优先使用PM2.5_24h，如果没有则使用PM2.5）
        for city in city_columns:
            year_data[city] = {}
            # 优先使用24小时平均值
            if 'PM2.5' in daily_avg.index:
                value = daily_avg.loc['PM2.5', city]
                if not pd.isna(value):
                    year_data[city]['PM2.5'] = value
            elif 'PM2.5_24h' in daily_avg.index:
                value = daily_avg.loc['PM2.5_24h', city]
                if not pd.isna(value):
                    year_data[city]['PM2.5'] = value
        
        return (year, year_data)
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def calculate_linear_regression(folder_path, n_processes=None):
    """
    计算每个城市PM2.5污染物的线性回归，用斜率判断年均增减趋势及速度
    
    参数:
    folder_path: CSV文件所在文件夹路径
    n_processes: 进程数，默认为CPU核心数
    
    返回:
    results: 字典，包含每个城市的线性回归结果
    years: 年份列表
    """
    
    # 存储所有年份数据的字典
    all_data = {}
    
    # 1. 遍历文件夹中的所有CSV文件
    csv_files = list(Path(folder_path).glob("*.csv"))
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 确定进程数
    if n_processes is None:
        n_processes = cpu_count()
    print(f"使用 {n_processes} 个进程并行处理...")
    
    # 使用多进程处理文件
    print("正在读取和处理CSV文件...")
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, csv_files),
            total=len(csv_files),
            desc="处理CSV文件",
            unit="文件"
        ))
    
    # 合并结果
    for result in results:
        if result is not None:
            year, year_data = result
            if year not in all_data:
                all_data[year] = {}
            all_data[year].update(year_data)
    
    # 整理数据，计算线性回归
    years = sorted(all_data.keys())
    print(f"处理了 {len(years)} 年的数据: {years}")
    
    # 对每个城市进行计算
    all_cities = set()
    for year_data in all_data.values():
        all_cities.update(year_data.keys())
    
    print(f"\n正在计算 {len(all_cities)} 个城市的线性回归...")
    regression_results = {}
    
    for city in tqdm(all_cities, desc="计算线性回归", unit="城市"):
        # 提取该城市各年的PM2.5数据
        city_pm25_data = {}
        for year in years:
            if city in all_data[year] and 'PM2.5' in all_data[year][city]:
                city_pm25_data[int(year)] = all_data[year][city]['PM2.5']
        
        # 至少需要3年的数据才能进行线性回归
        if len(city_pm25_data) >= 3:
            # 准备回归数据
            years_array = np.array(list(city_pm25_data.keys()))
            values_array = np.array(list(city_pm25_data.values()))
            
            # 进行线性回归
            # y = slope * x + intercept
            # 其中 x 是年份，y 是PM2.5浓度
            slope, intercept, r_value, p_value, std_err = stats.linregress(years_array, values_array)
            
            # 计算R²
            r_squared = r_value ** 2
            
            # 判断趋势
            if slope > 0:
                trend = "增加"
                trend_en = "Increasing"
            elif slope < 0:
                trend = "减少"
                trend_en = "Decreasing"
            else:
                trend = "无变化"
                trend_en = "No change"
            
            # 计算年均变化速度（斜率即为年均变化量，单位：μg/m³/年）
            annual_change_rate = slope
            
            # 计算变化百分比（相对于首年）
            if len(years_array) > 0:
                first_year_value = values_array[0]
                if first_year_value != 0:
                    percent_change_per_year = (slope / first_year_value) * 100
                else:
                    percent_change_per_year = np.nan
            else:
                percent_change_per_year = np.nan
            
            # 存储结果
            regression_results[city] = {
                'years': years_array.tolist(),
                'values': values_array.tolist(),
                'slope': slope,  # 斜率（年均变化量，μg/m³/年）
                'intercept': intercept,  # 截距
                'r_squared': r_squared,  # R²
                'p_value': p_value,  # p值（显著性检验）
                'std_err': std_err,  # 标准误差
                'trend': trend,  # 趋势（增加/减少/无变化）
                'trend_en': trend_en,
                'annual_change_rate': annual_change_rate,  # 年均变化速度（μg/m³/年）
                'percent_change_per_year': percent_change_per_year,  # 年均变化百分比（%）
                'n_years': len(years_array)  # 数据年份数
            }
        else:
            # 数据不足，记录为NaN
            regression_results[city] = {
                'years': [],
                'values': [],
                'slope': np.nan,
                'intercept': np.nan,
                'r_squared': np.nan,
                'p_value': np.nan,
                'std_err': np.nan,
                'trend': "数据不足",
                'trend_en': "Insufficient data",
                'annual_change_rate': np.nan,
                'percent_change_per_year': np.nan,
                'n_years': len(city_pm25_data)
            }
    
    return regression_results, years

def save_results_to_csv(regression_results, years, output_dir):
    """
    将线性回归结果保存为CSV文件
    
    参数:
    regression_results: 线性回归结果字典
    years: 年份列表
    output_dir: 输出目录路径
    """
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建线性回归结果表格
    print("\n正在生成线性回归结果...")
    regression_data = []
    
    for city, result in tqdm(regression_results.items(), desc="处理回归结果", unit="城市"):
        row = {
            'City': city,
            'Slope_μg_m3_per_year': result['slope'],  # 斜率（年均变化量，μg/m³/年）
            'Intercept': result['intercept'],  # 截距
            'R_squared': result['r_squared'],  # R²
            'P_value': result['p_value'],  # p值
            'Std_Error': result['std_err'],  # 标准误差
            'Trend': result['trend'],  # 趋势
            'Trend_EN': result['trend_en'],
            'Annual_Change_Rate_μg_m3_per_year': result['annual_change_rate'],  # 年均变化速度
            'Percent_Change_Per_Year': result['percent_change_per_year'],  # 年均变化百分比
            'N_Years': result['n_years']  # 数据年份数
        }
        
        # 添加各年的PM2.5浓度值
        for i, year in enumerate(result['years']):
            row[f'PM2.5_{year}'] = result['values'][i]
        
        regression_data.append(row)
    
    if regression_data:
        df_regression = pd.DataFrame(regression_data)
        regression_path = os.path.join(output_dir, 'YZD_PM25_Linear_Regression.csv')
        df_regression.to_csv(regression_path, index=False, encoding='utf-8-sig')
        print(f"线性回归结果已保存到: {regression_path}")
    
    print(f"\n所有结果已保存到目录: {output_dir}")

def print_summary(regression_results, years):
    """
    打印线性回归结果摘要
    """
    print("\n" + "="*80)
    print("PM2.5线性回归分析摘要")
    print("="*80)
    
    # 统计趋势分布
    trend_counts = {}
    for city, result in regression_results.items():
        trend = result['trend']
        if trend not in trend_counts:
            trend_counts[trend] = 0
        trend_counts[trend] += 1
    
    print(f"\n趋势分布统计:")
    for trend, count in trend_counts.items():
        print(f"  {trend}: {count} 个城市")
    
    print(f"\n详细结果:")
    print("-"*80)
    
    for city, result in regression_results.items():
        print(f"\n城市: {city}")
        print(f"  数据年份数: {result['n_years']}")
        
        if not np.isnan(result['slope']):
            print(f"  斜率（年均变化量）: {result['slope']:.4f} μg/m³/年")
            print(f"  截距: {result['intercept']:.4f}")
            print(f"  R²: {result['r_squared']:.4f}")
            print(f"  p值: {result['p_value']:.6f}")
            print(f"  标准误差: {result['std_err']:.4f}")
            print(f"  趋势: {result['trend']}")
            print(f"  年均变化速度: {result['annual_change_rate']:.4f} μg/m³/年")
            if not np.isnan(result['percent_change_per_year']):
                print(f"  年均变化百分比: {result['percent_change_per_year']:.2f}%")
            
            # 显示各年数据
            print(f"  各年PM2.5浓度:")
            for i, year in enumerate(result['years']):
                print(f"    {year}: {result['values'][i]:.2f} μg/m³")
        else:
            print(f"  数据不足，无法进行线性回归分析")

# 主程序
if __name__ == "__main__":
    # 设置文件夹路径（根据您的实际情况修改）
    folder_path = r"E:\DATA Science\大论文Result\YZD\filtered_daily"
    
    # 输出目录路径
    output_dir = r"E:\DATA Science\大论文Result\YZD\统计趋势检验"
    
    # 设置进程数（None表示使用所有CPU核心，也可以指定具体数字，如4）
    n_processes = None
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹路径不存在: {folder_path}")
        print("请确保路径正确，或者修改脚本中的folder_path变量")
    else:
        # 计算线性回归（使用多进程）
        regression_results, years = calculate_linear_regression(folder_path, n_processes=n_processes)
        
        # 打印摘要
        print_summary(regression_results, years)
        
        # 保存结果为CSV文件
        save_results_to_csv(regression_results, years, output_dir)
        
        print("\n线性回归分析完成！")
        print("\n说明:")
        print("- 斜率（Slope）> 0 表示PM2.5浓度年均增加趋势")
        print("- 斜率（Slope）< 0 表示PM2.5浓度年均减少趋势")
        print("- 斜率的绝对值表示年均变化速度（μg/m³/年）")
        print("- R² 表示回归模型的拟合优度（越接近1越好）")
        print("- p值 < 0.05 表示回归关系在统计上显著")
