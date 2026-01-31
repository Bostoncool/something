import pandas as pd
import numpy as np
from scipy import stats
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def sliding_t_test(data, window_size=None):
    """
    执行滑动T检验检测时间序列中的突变点

    参数:
    data: 时间序列数据（numpy数组或列表）
    window_size: 滑动窗口大小，默认为数据长度的1/3

    返回:
    dict: 包含检验结果的字典
        - change_point: 突变点位置（索引，从0开始）
        - change_month: 突变点月份（YYYY-MM格式）
        - t_statistic: T统计量的最大值
        - p_value: 对应的p值
        - significant: 是否显著（p < 0.05）
        - trend_before: 突变点前趋势
        - trend_after: 突变点后趋势
        - window_size: 使用的窗口大小
    """
    n = len(data)
    if n < 6:  # 至少需要6个数据点进行有意义的T检验
        return {
            'change_point': None,
            'change_month': None,
            't_statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'trend_before': None,
            'trend_after': None,
            'window_size': window_size
        }

    # 设置默认窗口大小
    if window_size is None:
        window_size = max(6, n // 4)  # 月度数据更长，使用更大的窗口，至少6，至少为数据长度的1/4

    # 确保窗口大小合理
    window_size = min(window_size, n - 6)  # 留出至少6个月用于另一半

    # 存储T检验结果
    t_statistics = []
    p_values = []
    positions = []

    # 对每个可能的位置进行滑动T检验
    for i in range(window_size, n - window_size + 1):
        # 前半部分：i-window_size 到 i
        before_data = data[i-window_size:i]
        # 后半部分：i 到 i+window_size
        after_data = data[i:i+window_size]

        # 执行独立样本T检验
        try:
            t_stat, p_val = stats.ttest_ind(before_data, after_data, equal_var=False)
            t_statistics.append(abs(t_stat))  # 使用绝对值，因为我们关心差异的大小
            p_values.append(p_val)
            positions.append(i)
        except:
            # 如果T检验失败（例如数据方差为0），跳过
            continue

    if not t_statistics:
        return {
            'change_point': None,
            'change_month': None,
            't_statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'trend_before': None,
            'trend_after': None,
            'window_size': window_size
        }

    # 找到T统计量最大的位置
    max_idx = np.argmax(t_statistics)
    change_point = positions[max_idx]
    t_stat_max = t_statistics[max_idx]
    p_value = p_values[max_idx]

    # 判断显著性
    significant = p_value < 0.05

    # 分析突变点前后的趋势
    if significant and change_point > window_size and change_point < n - window_size:
        # 突变点前的数据（使用与窗口大小相同的长度）
        before_data = data[max(0, change_point - window_size):change_point]
        if len(before_data) >= 3:  # 月度数据至少需要3个月来判断趋势
            slope_before, _ = np.polyfit(range(len(before_data)), before_data, 1)
            trend_before = "增加" if slope_before > 0 else "减少" if slope_before < 0 else "稳定"
        else:
            trend_before = "数据不足"

        # 突变点后的数据
        after_data = data[change_point:min(n, change_point + window_size)]
        if len(after_data) >= 3:
            slope_after, _ = np.polyfit(range(len(after_data)), after_data, 1)
            trend_after = "增加" if slope_after > 0 else "减少" if slope_after < 0 else "稳定"
        else:
            trend_after = "数据不足"
    else:
        trend_before = None
        trend_after = None

    return {
        'change_point': change_point if significant else None,
        'change_month': None,  # 将在调用处设置
        't_statistic': t_stat_max,
        'p_value': p_value,
        'significant': significant,
        'trend_before': trend_before,
        'trend_after': trend_after,
        'window_size': window_size
    }

def process_single_file(file_path):
    """
    处理单个CSV文件，提取月份和城市污染物数据

    参数:
    file_path: CSV文件路径

    返回:
    (month, month_data) 元组，month_data是字典 {city: {pollutant: value}}
    """
    try:
        # 从文件名提取月份
        file_name = Path(file_path).stem
        # 假设文件名格式为: china_cities_YYYYMMDD.csv
        # 提取YYYY-MM部分作为月份标识
        date_part = file_name.split('_')[-1][:8]  # YYYYMMDD
        if len(date_part) >= 6:
            year = date_part[:4]
            month = date_part[4:6]
            month_key = f"{year}-{month}"
        else:
            # 如果日期格式不正确，跳过
            return None

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

        # 存储每月的数据
        month_data = {}

        # 对于每个城市，存储PM2.5的月均值（优先使用PM2.5_24h，如果没有则使用PM2.5）
        for city in city_columns:
            month_data[city] = {}
            # 优先使用24小时平均值
            if 'PM2.5' in daily_avg.index:
                value = daily_avg.loc['PM2.5', city]
                if not pd.isna(value):
                    month_data[city]['PM2.5'] = value
            elif 'PM2.5_24h' in daily_avg.index:
                value = daily_avg.loc['PM2.5_24h', city]
                if not pd.isna(value):
                    month_data[city]['PM2.5'] = value

        return (month_key, month_data)

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def calculate_sliding_t_test(folder_path, window_size=None, n_processes=None):
    """
    对每个城市PM2.5污染物数据执行滑动T检验，检测突变点

    参数:
    folder_path: CSV文件所在文件夹路径
    window_size: 滑动窗口大小，默认为None（自动计算）
    n_processes: 进程数，默认为CPU核心数

    返回:
    results: 字典，包含每个城市的滑动T检验结果
    months: 月份列表
    """

    # 存储所有月份数据的字典
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
            month, month_data = result
            if month not in all_data:
                all_data[month] = {}
            all_data[month].update(month_data)

    # 整理数据，执行滑动T检验
    months = sorted(all_data.keys())
    print(f"处理了 {len(months)} 个月的数据: {months[:5]}...{months[-5:] if len(months) > 5 else months}")

    # 对每个城市进行计算
    all_cities = set()
    for month_data in all_data.values():
        all_cities.update(month_data.keys())

    print(f"\n正在对 {len(all_cities)} 个城市执行滑动T检验...")
    sliding_t_results = {}

    for city in tqdm(all_cities, desc="执行滑动T检验", unit="城市"):
        # 提取该城市各月的PM2.5数据
        city_pm25_data = {}
        for month in months:
            if city in all_data[month] and 'PM2.5' in all_data[month][city]:
                city_pm25_data[month] = all_data[month][city]['PM2.5']

        # 至少需要12个月的数据才能进行有意义的滑动T检验
        if len(city_pm25_data) >= 12:
            # 准备检验数据（按月份排序）
            sorted_months = sorted(city_pm25_data.keys())
            values_array = np.array([city_pm25_data[month] for month in sorted_months])

            # 执行滑动T检验
            sliding_t_result = sliding_t_test(values_array, window_size=window_size)

            # 设置突变点月份
            if sliding_t_result['significant'] and sliding_t_result['change_point'] is not None:
                sliding_t_result['change_month'] = sorted_months[sliding_t_result['change_point']]

            # 存储结果
            sliding_t_results[city] = {
                'months': sorted_months,
                'values': values_array.tolist(),
                'change_point': sliding_t_result['change_point'],  # 突变点位置（索引）
                'change_month': sliding_t_result['change_month'],  # 突变点月份
                't_statistic': sliding_t_result['t_statistic'],  # T统计量
                'p_value': sliding_t_result['p_value'],  # p值
                'significant': sliding_t_result['significant'],  # 是否显著
                'trend_before': sliding_t_result['trend_before'],  # 突变点前趋势
                'trend_after': sliding_t_result['trend_after'],  # 突变点后趋势
                'window_size': sliding_t_result['window_size'],  # 窗口大小
                'n_months': len(sorted_months)  # 数据月份数
            }
        else:
            # 数据不足，记录为NaN
            sliding_t_results[city] = {
                'months': list(city_pm25_data.keys()),
                'values': list(city_pm25_data.values()),
                'change_point': None,
                'change_month': None,
                't_statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'trend_before': None,
                'trend_after': None,
                'window_size': window_size,
                'n_months': len(city_pm25_data)
            }

    return sliding_t_results, months

def save_results_to_csv(sliding_t_results, months, output_dir):
    """
    将滑动T检验结果保存为CSV文件

    参数:
    sliding_t_results: 滑动T检验结果字典
    months: 月份列表
    output_dir: 输出目录路径
    """

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建滑动T检验结果表格
    print("\n正在生成滑动T检验结果...")
    sliding_t_data = []

    for city, result in tqdm(sliding_t_results.items(), desc="处理检验结果", unit="城市"):
        row = {
            'City': city,
            'Change_Point_Index': result['change_point'],  # 突变点位置（索引）
            'Change_Month': result['change_month'],  # 突变点月份
            'T_Statistic': result['t_statistic'],  # T统计量
            'P_Value': result['p_value'],  # p值
            'Significant': result['significant'],  # 是否显著
            'Trend_Before': result['trend_before'],  # 突变点前趋势
            'Trend_After': result['trend_after'],  # 突变点后趋势
            'Window_Size': result['window_size'],  # 窗口大小
            'N_Months': result['n_months']  # 数据月份数
        }

        # 添加各月的PM2.5浓度值
        for i, month in enumerate(result['months']):
            row[f'PM2.5_{month}'] = result['values'][i]

        sliding_t_data.append(row)

    if sliding_t_data:
        df_sliding_t = pd.DataFrame(sliding_t_data)
        sliding_t_path = os.path.join(output_dir, 'BTH_PM25_Monthly_Sliding_T_Test.csv')
        df_sliding_t.to_csv(sliding_t_path, index=False, encoding='utf-8-sig')
        print(f"月度滑动T检验结果已保存到: {sliding_t_path}")

    print(f"\n所有结果已保存到目录: {output_dir}")

def print_summary(sliding_t_results, months):
    """
    打印滑动T检验结果摘要
    """
    print("\n" + "="*80)
    print("PM2.5 月度滑动T检验突变点分析摘要")
    print("="*80)

    # 统计显著性分布
    significant_count = 0
    total_count = 0
    change_detected_cities = []

    for city, result in sliding_t_results.items():
        total_count += 1
        if result['significant']:
            significant_count += 1
            change_detected_cities.append(city)

    print(f"\n总城市数: {total_count}")
    print(f"检测到显著突变点的城市数: {significant_count}")
    print(f"显著比例: {significant_count/total_count*100:.1f}%")
    print(f"\n趋势变化分析:")
    trend_changes = {}
    for city, result in sliding_t_results.items():
        if result['significant'] and result['trend_before'] and result['trend_after']:
            change_type = f"{result['trend_before']}→{result['trend_after']}"
            if change_type not in trend_changes:
                trend_changes[change_type] = 0
            trend_changes[change_type] += 1

    for change_type, count in trend_changes.items():
        print(f"  {change_type}: {count} 个城市")

    print(f"\n详细结果:")
    print("-"*80)

    for city, result in sliding_t_results.items():
        print(f"\n城市: {city}")
        print(f"  数据月份数: {result['n_months']}")
        print(f"  滑动窗口大小: {result['window_size']}")

        if result['significant']:
            print(f"  突变点位置: 第{result['change_point']}个观测点（{result['change_month']}）")
            print(f"  T统计量: {result['t_statistic']:.4f}")
            print(f"  p值: {result['p_value']:.6f} (显著)")
            if result['trend_before'] and result['trend_after']:
                print(f"  趋势变化: {result['trend_before']} → {result['trend_after']}")

            # 显示关键月份的数据
            print(f"  突变点前后PM2.5浓度:")
            change_idx = result['change_point']
            months_list = result['months']
            values_list = result['values']

            # 显示突变点前后的几个月份
            start_idx = max(0, change_idx - 2)
            end_idx = min(len(months_list), change_idx + 3)

            for i in range(start_idx, end_idx):
                marker = " ← 突变点" if i == change_idx else ""
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

    # 设置滑动窗口大小（None表示自动计算，也可以指定具体数字，如12或24表示1-2年）
    window_size = None

    # 设置进程数（None表示使用所有CPU核心，也可以指定具体数字，如4）
    n_processes = None

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹路径不存在: {folder_path}")
        print("请确保路径正确，或者修改脚本中的folder_path变量")
    else:
        # 执行滑动T检验（使用多进程）
        sliding_t_results, months = calculate_sliding_t_test(folder_path, window_size=window_size, n_processes=n_processes)

        # 打印摘要
        print_summary(sliding_t_results, months)

        # 保存结果为CSV文件
        save_results_to_csv(sliding_t_results, months, output_dir)

        print("\n月度滑动T检验完成！")
        print("\n说明:")
        print("- 月度滑动T检验通过比较滑动窗口内前后两段数据的均值差异来检测突变点")
        print("- p值 < 0.05 表示在该置信水平下存在显著突变点")
        print("- 突变点表示污染水平发生显著变化的时间点")
        print("- 趋势变化显示突变点前后污染水平的变动方向")
        print("- 月度分析对季节性变化更敏感，需要更大的窗口来获得稳定的结果")
        print("- 建议窗口大小至少为12个月（1年）以平滑季节效应")