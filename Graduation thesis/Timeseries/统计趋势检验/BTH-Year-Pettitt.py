import pandas as pd
import numpy as np
from scipy import stats
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def pettitt_test(data):
    """
    执行Pettitt检验检测时间序列中的突变点

    参数:
    data: 时间序列数据（numpy数组或列表）

    返回:
    dict: 包含检验结果的字典
        - change_point: 突变点位置（索引，从0开始）
        - change_year: 突变点年份
        - K: Pettitt统计量
        - p_value: p值
        - significant: 是否显著（p < 0.05）
        - trend_before: 突变点前趋势
        - trend_after: 突变点后趋势
    """
    n = len(data)
    if n < 4:  # 至少需要4个数据点
        return {
            'change_point': None,
            'change_year': None,
            'K': np.nan,
            'p_value': np.nan,
            'significant': False,
            'trend_before': None,
            'trend_after': None
        }

    # 计算Pettitt统计量
    U = np.zeros(n-1)
    for k in range(1, n):
        # 计算U_k = Σ_{i=1 to k} Σ_{j=k+1 to n} sign(x_i - x_j)
        # 使用更高效的计算方式
        left_sum = 0
        right_sum = 0
        for i in range(k):
            for j in range(k, n):
                left_sum += np.sign(data[i] - data[j])
        U[k-1] = left_sum

    # 找到|U|的最大值
    abs_U = np.abs(U)
    max_idx = np.argmax(abs_U)
    K = abs_U[max_idx]
    change_point = max_idx + 1  # 突变点位置（从1开始计数，对应第k年之后）

    # 计算p值（近似正态分布）
    # p值计算公式：p = 2 * exp(-6 * K^2 / (n^3 + n^2))
    if K > 0:
        p_value = 2 * np.exp(-6 * K**2 / (n**3 + n**2))
    else:
        p_value = 1.0

    # 判断显著性
    significant = p_value < 0.05

    # 分析突变点前后的趋势
    if significant and change_point > 1 and change_point < n-1:
        # 突变点前的数据
        before_data = data[:change_point]
        if len(before_data) >= 2:
            slope_before, _ = np.polyfit(range(len(before_data)), before_data, 1)
            trend_before = "增加" if slope_before > 0 else "减少" if slope_before < 0 else "稳定"
        else:
            trend_before = "数据不足"

        # 突变点后的数据
        after_data = data[change_point:]
        if len(after_data) >= 2:
            slope_after, _ = np.polyfit(range(len(after_data)), after_data, 1)
            trend_after = "增加" if slope_after > 0 else "减少" if slope_after < 0 else "稳定"
        else:
            trend_after = "数据不足"
    else:
        trend_before = None
        trend_after = None

    return {
        'change_point': change_point if significant else None,
        'change_year': None,  # 将在调用处设置
        'K': K,
        'p_value': p_value,
        'significant': significant,
        'trend_before': trend_before,
        'trend_after': trend_after
    }

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

def calculate_pettitt_test(folder_path, n_processes=None):
    """
    对每个城市PM2.5污染物数据执行Pettitt检验，检测突变点

    参数:
    folder_path: CSV文件所在文件夹路径
    n_processes: 进程数，默认为CPU核心数

    返回:
    results: 字典，包含每个城市的Pettitt检验结果
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

    # 整理数据，执行Pettitt检验
    years = sorted(all_data.keys())
    print(f"处理了 {len(years)} 年的数据: {years}")

    # 对每个城市进行计算
    all_cities = set()
    for year_data in all_data.values():
        all_cities.update(year_data.keys())

    print(f"\n正在对 {len(all_cities)} 个城市执行Pettitt检验...")
    pettitt_results = {}

    for city in tqdm(all_cities, desc="执行Pettitt检验", unit="城市"):
        # 提取该城市各年的PM2.5数据
        city_pm25_data = {}
        for year in years:
            if city in all_data[year] and 'PM2.5' in all_data[year][city]:
                city_pm25_data[int(year)] = all_data[year][city]['PM2.5']

        # 至少需要4年的数据才能进行Pettitt检验
        if len(city_pm25_data) >= 4:
            # 准备检验数据（按年份排序）
            sorted_years = sorted(city_pm25_data.keys())
            values_array = np.array([city_pm25_data[year] for year in sorted_years])

            # 执行Pettitt检验
            pettitt_result = pettitt_test(values_array)

            # 设置突变点年份
            if pettitt_result['significant'] and pettitt_result['change_point'] is not None:
                pettitt_result['change_year'] = sorted_years[pettitt_result['change_point'] - 1]

            # 存储结果
            pettitt_results[city] = {
                'years': sorted_years,
                'values': values_array.tolist(),
                'change_point': pettitt_result['change_point'],  # 突变点位置（索引）
                'change_year': pettitt_result['change_year'],  # 突变点年份
                'K_statistic': pettitt_result['K'],  # Pettitt统计量
                'p_value': pettitt_result['p_value'],  # p值
                'significant': pettitt_result['significant'],  # 是否显著
                'trend_before': pettitt_result['trend_before'],  # 突变点前趋势
                'trend_after': pettitt_result['trend_after'],  # 突变点后趋势
                'n_years': len(sorted_years)  # 数据年份数
            }
        else:
            # 数据不足，记录为NaN
            pettitt_results[city] = {
                'years': list(city_pm25_data.keys()),
                'values': list(city_pm25_data.values()),
                'change_point': None,
                'change_year': None,
                'K_statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'trend_before': None,
                'trend_after': None,
                'n_years': len(city_pm25_data)
            }

    return pettitt_results, years

def save_results_to_csv(pettitt_results, years, output_dir):
    """
    将Pettitt检验结果保存为CSV文件

    参数:
    pettitt_results: Pettitt检验结果字典
    years: 年份列表
    output_dir: 输出目录路径
    """

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建Pettitt检验结果表格
    print("\n正在生成Pettitt检验结果...")
    pettitt_data = []

    for city, result in tqdm(pettitt_results.items(), desc="处理检验结果", unit="城市"):
        row = {
            'City': city,
            'Change_Point_Index': result['change_point'],  # 突变点位置（索引）
            'Change_Year': result['change_year'],  # 突变点年份
            'K_Statistic': result['K_statistic'],  # Pettitt统计量
            'P_Value': result['p_value'],  # p值
            'Significant': result['significant'],  # 是否显著
            'Trend_Before': result['trend_before'],  # 突变点前趋势
            'Trend_After': result['trend_after'],  # 突变点后趋势
            'N_Years': result['n_years']  # 数据年份数
        }

        # 添加各年的PM2.5浓度值
        for i, year in enumerate(result['years']):
            row[f'PM2.5_{year}'] = result['values'][i]

        pettitt_data.append(row)

    if pettitt_data:
        df_pettitt = pd.DataFrame(pettitt_data)
        pettitt_path = os.path.join(output_dir, 'BTH_PM25_Pettitt_Test.csv')
        df_pettitt.to_csv(pettitt_path, index=False, encoding='utf-8-sig')
        print(f"Pettitt检验结果已保存到: {pettitt_path}")

    print(f"\n所有结果已保存到目录: {output_dir}")

def print_summary(pettitt_results, years):
    """
    打印Pettitt检验结果摘要
    """
    print("\n" + "="*80)
    print("PM2.5 Pettitt突变点检验分析摘要")
    print("="*80)

    # 统计显著性分布
    significant_count = 0
    total_count = 0
    change_detected_cities = []

    for city, result in pettitt_results.items():
        total_count += 1
        if result['significant']:
            significant_count += 1
            change_detected_cities.append(city)

    print(f"\n总城市数: {total_count}")
    print(f"检测到显著突变点的城市数: {significant_count}")
    print(f"显著比例: {significant_count/total_count*100:.1f}%")
    print(f"\n趋势变化分析:")
    trend_changes = {}
    for city, result in pettitt_results.items():
        if result['significant'] and result['trend_before'] and result['trend_after']:
            change_type = f"{result['trend_before']}→{result['trend_after']}"
            if change_type not in trend_changes:
                trend_changes[change_type] = 0
            trend_changes[change_type] += 1

    for change_type, count in trend_changes.items():
        print(f"  {change_type}: {count} 个城市")

    print(f"\n详细结果:")
    print("-"*80)

    for city, result in pettitt_results.items():
        print(f"\n城市: {city}")
        print(f"  数据年份数: {result['n_years']}")

        if result['significant']:
            print(f"  突变点位置: 第{result['change_point']}个观测点（{result['change_year']}年）")
            print(f"  Pettitt统计量 K: {result['K_statistic']:.4f}")
            print(f"  p值: {result['p_value']:.6f} (显著)")
            if result['trend_before'] and result['trend_after']:
                print(f"  趋势变化: {result['trend_before']} → {result['trend_after']}")

            # 显示各年数据
            print(f"  各年PM2.5浓度:")
            for i, year in enumerate(result['years']):
                marker = " ← 突变点" if i + 1 == result['change_point'] else ""
                print(".2f")
        else:
            print(f"  未检测到显著突变点")
            if not np.isnan(result['p_value']):
                print(f"  p值: {result['p_value']:.6f} (不显著)")

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
        # 执行Pettitt检验（使用多进程）
        pettitt_results, years = calculate_pettitt_test(folder_path, n_processes=n_processes)

        # 打印摘要
        print_summary(pettitt_results, years)

        # 保存结果为CSV文件
        save_results_to_csv(pettitt_results, years, output_dir)

        print("\nPettitt突变点检验完成！")
        print("\n说明:")
        print("- Pettitt检验用于检测时间序列中的突变点")
        print("- p值 < 0.05 表示在该置信水平下存在显著突变点")
        print("- 突变点表示污染水平发生显著变化的时间点")
        print("- 趋势变化显示突变点前后污染水平的变动方向")