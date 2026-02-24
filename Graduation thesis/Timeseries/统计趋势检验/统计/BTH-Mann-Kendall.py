import pandas as pd
import numpy as np
from scipy import stats
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.stats import norm

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

def mann_kendall_test(data):
    """
    实现Mann-Kendall趋势检验

    参数:
    data: 时间序列数据列表或数组

    返回:
    字典包含检验结果
    """
    n = len(data)
    if n < 3:
        return {
            'S': np.nan,
            'var_S': np.nan,
            'Z': np.nan,
            'p_value': np.nan,
            'trend': '数据不足',
            'trend_en': 'Insufficient data',
            'sen_slope': np.nan,
            'significant': False
        }

    # 计算S统计量
    S = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if data[j] > data[i]:
                S += 1
            elif data[j] < data[i]:
                S -= 1

    # 计算方差（考虑自相关）
    # 对于无自相关的数据，使用简化公式
    unique_values = len(set(data))
    if n == unique_values:  # 所有值都不同
        var_S = n * (n - 1) * (2 * n + 5) / 18
    else:
        # 处理重复值
        tp = np.zeros(unique_values + 1)
        for val in data:
            count = data.count(val)
            tp[count] += 1

        var_S = n * (n - 1) * (2 * n + 5) / 18
        for i in range(1, len(tp)):
            var_S -= tp[i] * (i - 1) * i * (2 * i + 5) / 18

    # 计算Z统计量
    if S > 0:
        Z = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        Z = (S + 1) / np.sqrt(var_S)
    else:
        Z = 0

    # 计算p值（双尾检验）
    p_value = 2 * (1 - norm.cdf(abs(Z)))

    # 判断趋势
    alpha = 0.05
    if p_value < alpha:
        if Z > 0:
            trend = "增加"
            trend_en = "Increasing"
        else:
            trend = "减少"
            trend_en = "Decreasing"
        significant = True
    else:
        trend = "无显著趋势"
        trend_en = "No significant trend"
        significant = False

    # 计算Sen斜率（趋势幅度）
    sen_slope = calculate_sen_slope(data)

    return {
        'S': S,
        'var_S': var_S,
        'Z': Z,
        'p_value': p_value,
        'trend': trend,
        'trend_en': trend_en,
        'sen_slope': sen_slope,  # Sen斜率（年均变化量，μg/m³/年）
        'significant': significant
    }

def calculate_sen_slope(data):
    """
    计算Sen斜率（稳健的趋势幅度估计）

    参数:
    data: 时间序列数据

    返回:
    Sen斜率
    """
    n = len(data)
    slopes = []

    for i in range(n-1):
        for j in range(i+1, n):
            slope = (data[j] - data[i]) / (j - i)
            slopes.append(slope)

    if slopes:
        return np.median(slopes)
    else:
        return np.nan

def calculate_mann_kendall(folder_path, n_processes=None):
    """
    计算每个城市PM2.5污染物的Mann-Kendall趋势检验

    参数:
    folder_path: CSV文件所在文件夹路径
    n_processes: 进程数，默认为CPU核心数

    返回:
    results: 字典，包含每个城市的Mann-Kendall检验结果
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

    # 整理数据，计算Mann-Kendall检验
    years = sorted(all_data.keys())
    print(f"处理了 {len(years)} 年的数据: {years}")

    # 对每个城市进行计算
    all_cities = set()
    for year_data in all_data.values():
        all_cities.update(year_data.keys())

    print(f"\n正在计算 {len(all_cities)} 个城市的Mann-Kendall检验...")
    mk_results = {}

    for city in tqdm(all_cities, desc="计算Mann-Kendall检验", unit="城市"):
        # 提取该城市各年的PM2.5数据
        city_pm25_data = {}
        for year in years:
            if city in all_data[year] and 'PM2.5' in all_data[year][city]:
                city_pm25_data[int(year)] = all_data[year][city]['PM2.5']

        # 至少需要3年的数据才能进行Mann-Kendall检验
        if len(city_pm25_data) >= 3:
            # 准备检验数据（按时间顺序排序）
            sorted_years = sorted(city_pm25_data.keys())
            values_array = [city_pm25_data[year] for year in sorted_years]

            # 进行Mann-Kendall检验
            mk_result = mann_kendall_test(values_array)

            # 计算年均变化百分比（相对于首年，使用Sen斜率）
            if not np.isnan(mk_result['sen_slope']) and len(values_array) > 0:
                first_year_value = values_array[0]
                if first_year_value != 0:
                    percent_change_per_year = (mk_result['sen_slope'] / first_year_value) * 100
                else:
                    percent_change_per_year = np.nan
            else:
                percent_change_per_year = np.nan

            # 存储结果
            mk_results[city] = {
                'years': sorted_years,
                'values': values_array,
                'S': mk_result['S'],  # Mann-Kendall S统计量
                'var_S': mk_result['var_S'],  # S的方差
                'Z': mk_result['Z'],  # Z统计量
                'p_value': mk_result['p_value'],  # p值
                'trend': mk_result['trend'],  # 趋势
                'trend_en': mk_result['trend_en'],
                'sen_slope': mk_result['sen_slope'],  # Sen斜率（年均变化量，μg/m³/年）
                'percent_change_per_year': percent_change_per_year,  # 年均变化百分比（%）
                'significant': mk_result['significant'],  # 是否显著
                'n_years': len(sorted_years)  # 数据年份数
            }
        else:
            # 数据不足，记录为NaN
            mk_results[city] = {
                'years': list(city_pm25_data.keys()),
                'values': list(city_pm25_data.values()),
                'S': np.nan,
                'var_S': np.nan,
                'Z': np.nan,
                'p_value': np.nan,
                'trend': "数据不足",
                'trend_en': "Insufficient data",
                'sen_slope': np.nan,
                'percent_change_per_year': np.nan,
                'significant': False,
                'n_years': len(city_pm25_data)
            }

    return mk_results, years

def save_results_to_csv(mk_results, years, output_dir):
    """
    将Mann-Kendall检验结果保存为CSV文件

    参数:
    mk_results: Mann-Kendall检验结果字典
    years: 年份列表
    output_dir: 输出目录路径
    """

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建Mann-Kendall检验结果表格
    print("\n正在生成Mann-Kendall检验结果...")
    mk_data = []

    for city, result in tqdm(mk_results.items(), desc="处理检验结果", unit="城市"):
        row = {
            'City': city,
            'S_Statistic': result['S'],  # Mann-Kendall S统计量
            'Variance_S': result['var_S'],  # S的方差
            'Z_Statistic': result['Z'],  # Z统计量
            'P_value': result['p_value'],  # p值
            'Trend': result['trend'],  # 趋势
            'Trend_EN': result['trend_en'],
            'Sen_Slope_μg_m3_per_year': result['sen_slope'],  # Sen斜率（年均变化量，μg/m³/年）
            'Percent_Change_Per_Year': result['percent_change_per_year'],  # 年均变化百分比
            'Significant': result['significant'],  # 是否显著
            'N_Years': result['n_years']  # 数据年份数
        }

        # 添加各年的PM2.5浓度值
        for i, year in enumerate(result['years']):
            row[f'PM2.5_{year}'] = result['values'][i]

        mk_data.append(row)

    if mk_data:
        df_mk = pd.DataFrame(mk_data)
        mk_path = os.path.join(output_dir, 'BTH_PM25_Mann_Kendall.csv')
        df_mk.to_csv(mk_path, index=False, encoding='utf-8-sig')
        print(f"Mann-Kendall检验结果已保存到: {mk_path}")

    print(f"\n所有结果已保存到目录: {output_dir}")

def print_summary(mk_results, years):
    """
    打印Mann-Kendall检验结果摘要
    """
    print("\n" + "="*80)
    print("PM2.5 Mann-Kendall趋势检验摘要")
    print("="*80)

    # 统计趋势分布
    trend_counts = {}
    significant_counts = {'显著': 0, '不显著': 0}

    for city, result in mk_results.items():
        trend = result['trend']
        if trend not in trend_counts:
            trend_counts[trend] = 0
        trend_counts[trend] += 1

        if result['significant']:
            significant_counts['显著'] += 1
        else:
            significant_counts['不显著'] += 1

    print(f"\n趋势分布统计:")
    for trend, count in trend_counts.items():
        print(f"  {trend}: {count} 个城市")

    print(f"\n显著性统计:")
    for sig, count in significant_counts.items():
        print(f"  {sig}: {count} 个城市")

    print(f"\n详细结果:")
    print("-"*80)

    for city, result in mk_results.items():
        print(f"\n城市: {city}")
        print(f"  数据年份数: {result['n_years']}")

        if not np.isnan(result['S']):
            print(f"  Mann-Kendall S统计量: {result['S']}")
            print(f"  Z统计量: {result['Z']:.4f}")
            print(f"  p值: {result['p_value']:.6f}")
            print(f"  趋势: {result['trend']}")
            print(f"  是否显著: {'是' if result['significant'] else '否'}")
            print(f"  Sen斜率（年均变化量）: {result['sen_slope']:.4f} μg/m³/年")
            if not np.isnan(result['percent_change_per_year']):
                print(f"  年均变化百分比: {result['percent_change_per_year']:.2f}%")

            # 显示各年数据
            print(f"  各年PM2.5浓度:")
            for i, year in enumerate(result['years']):
                print(f"    {year}: {result['values'][i]:.2f} μg/m³")
        else:
            print(f"  数据不足，无法进行Mann-Kendall趋势检验")

# 主程序
if __name__ == "__main__":
    # 设置文件夹路径（根据您的实际情况修改）
    folder_path = r"E:\DATA Science\大论文Result\BTH\filtered_daily"

    # 输出目录路径
    output_dir = r"E:\DATA Science\大论文Result\BTH\统计趋势检验"

    # 设置进程数（None表示使用所有CPU核心，也可以指定具体数字，如4）
    n_processes = None

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹路径不存在: {folder_path}")
        print("请确保路径正确，或者修改脚本中的folder_path变量")
    else:
        # 计算Mann-Kendall检验（使用多进程）
        mk_results, years = calculate_mann_kendall(folder_path, n_processes=n_processes)

        # 打印摘要
        print_summary(mk_results, years)

        # 保存结果为CSV文件
        save_results_to_csv(mk_results, years, output_dir)

        print("\nMann-Kendall趋势检验完成！")
        print("\n说明:")
        print("- S统计量 > 0 表示上升趋势，S统计量 < 0 表示下降趋势")
        print("- Z统计量的绝对值越大，趋势越显著")
        print("- p值 < 0.05 表示趋势在统计上显著")
        print("- Sen斜率表示趋势的稳健估计（年均变化量，μg/m³/年）")
        print("- 显著性检验基于双尾检验，α = 0.05")