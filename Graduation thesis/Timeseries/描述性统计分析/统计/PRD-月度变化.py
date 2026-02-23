import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

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
        date_str = file_name.split('_')[-1][:6]  # 获取YYYYMM
        year_month = f"{date_str[:4]}-{date_str[4:6]}"  # 转换为YYYY-MM格式

        # 读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8-sig')

        # 提取城市列（排除元数据列）
        city_columns = [col for col in df.columns if col not in ['date', 'hour', 'type', '__file__', '__missing_cols__']]

        # 按污染物类型和城市分组计算日均值
        # 筛选出AQI和污染物数据（包括实时和24小时平均数据）
        realtime_types = ['AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h',
                         'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h']
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
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def calculate_city_statistics(folder_path, n_processes=None):
    """
    计算每个城市的月均值、月际变化率和累积变化百分比

    参数:
    folder_path: CSV文件所在文件夹路径
    n_processes: 进程数，默认为CPU核心数

    累积变化百分比以2018年1月为基准，计算每一月的累积变化率
    """

    # 存储所有月度数据的字典
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
            month_key, month_data = result
            if month_key not in all_data:
                all_data[month_key] = {}
            all_data[month_key].update(month_data)

    # 整理数据，计算统计指标
    months = sorted(all_data.keys())
    print(f"处理了 {len(months)} 个月的数据: {months}")

    # 定义污染物类型
    realtime_types = ['AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h',
                     'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h']

    results = {}

    # 对每个城市进行计算
    all_cities = set()
    for month_data in all_data.values():
        all_cities.update(month_data.keys())

    print(f"\n正在计算 {len(all_cities)} 个城市的统计指标...")
    for city in tqdm(all_cities, desc="计算城市统计", unit="城市"):
        results[city] = {
            'monthly_means': {},  # 各月月均值
            'monthly_changes': {},  # 月际变化率
            'monthly_cumulative_changes': {}  # 累积变化百分比
        }

        # 提取该城市各月的数据
        city_data = {}
        for month in months:
            if city in all_data[month]:
                city_data[month] = all_data[month][city]

        # 计算每个污染物的统计指标
        for pollutant in realtime_types:
            # 提取该污染物各月的平均值
            pollutant_values = {}
            for month in months:
                if month in city_data and pollutant in city_data[month]:
                    pollutant_values[month] = city_data[month][pollutant]

            if len(pollutant_values) >= 2:  # 至少需要2个月的数据
                # 计算月均值
                results[city]['monthly_means'][pollutant] = pollutant_values

                # 计算月际变化率
                monthly_changes = {}
                sorted_months = sorted(pollutant_values.keys())
                for i in range(1, len(sorted_months)):
                    month1 = sorted_months[i-1]
                    month2 = sorted_months[i]
                    if pollutant_values[month1] != 0:  # 避免除零错误
                        change_rate = ((pollutant_values[month2] - pollutant_values[month1]) /
                                      pollutant_values[month1] * 100)
                        monthly_changes[f"{month1}-{month2}"] = f"{change_rate:.2f}%"

                results[city]['monthly_changes'][pollutant] = monthly_changes

                # 计算累积变化百分比（以2018-01为基准，计算每一月的累积变化率）
                base_month = '2018-01'  # 基准月份
                monthly_cumulative_changes = {}
                sorted_months = sorted(pollutant_values.keys())

                # 确保基准月份存在
                if base_month in pollutant_values and pollutant_values[base_month] != 0:
                    base_value = pollutant_values[base_month]
                    for month in sorted_months:
                        if month != base_month:  # 基准月份自身的变化率为0，不需要计算
                            cumulative_change = ((pollutant_values[month] - base_value) / base_value * 100)
                            monthly_cumulative_changes[month] = f"{cumulative_change:.2f}%"
                    # 基准月份的变化率为0
                    monthly_cumulative_changes[base_month] = "0.00%"
                else:
                    # 如果2018-01不存在，使用第一个月份作为基准
                    first_month = min(pollutant_values.keys())
                    if pollutant_values[first_month] != 0:
                        base_value = pollutant_values[first_month]
                        for month in sorted_months:
                            if month != first_month:
                                cumulative_change = ((pollutant_values[month] - base_value) / base_value * 100)
                                monthly_cumulative_changes[month] = f"{cumulative_change:.2f}%"
                        monthly_cumulative_changes[first_month] = "0.00%"

                results[city]['monthly_cumulative_changes'][pollutant] = monthly_cumulative_changes

    return results, months

def save_results_to_csv(results, months, output_dir):
    """
    将结果保存为3个CSV文件

    1. Monthly_Means.csv - 各月月均值
    2. Monthly_Changes.csv - 月际变化率
    3. Cumulative_Changes.csv - 累积变化百分比（每月相对于2018年1月的变化率）
    """

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 1. 创建月均值表格
    print("\n正在生成月均值数据...")
    monthly_data = []
    for city, city_results in tqdm(results.items(), desc="处理月均值", unit="城市"):
        for pollutant, values in city_results['monthly_means'].items():
            row = {'City': city, 'Pollutant': pollutant}
            for month in months:
                row[month] = values.get(month, np.nan)
            monthly_data.append(row)

    if monthly_data:
        df_monthly = pd.DataFrame(monthly_data)
        monthly_path = os.path.join(output_dir, 'Monthly_Means.csv')
        df_monthly.to_csv(monthly_path, index=False, encoding='utf-8-sig')
        print(f"月均值数据已保存到: {monthly_path}")

    # 2. 创建月际变化率表格
    print("\n正在生成月际变化率数据...")
    change_data = []
    for city, city_results in tqdm(results.items(), desc="处理月际变化率", unit="城市"):
        for pollutant, changes in city_results['monthly_changes'].items():
            row = {'City': city, 'Pollutant': pollutant}
            for period, change_rate in changes.items():
                row[period] = change_rate
            change_data.append(row)

    if change_data:
        df_changes = pd.DataFrame(change_data)
        changes_path = os.path.join(output_dir, 'Monthly_Changes.csv')
        df_changes.to_csv(changes_path, index=False, encoding='utf-8-sig')
        print(f"月际变化率数据已保存到: {changes_path}")

    # 3. 创建累积变化百分比表格（每月相对于2018年1月的变化率）
    print("\n正在生成累积变化百分比数据...")
    cumulative_data = []
    for city, city_results in tqdm(results.items(), desc="处理累积变化率", unit="城市"):
        for pollutant, monthly_cumulative_changes in city_results['monthly_cumulative_changes'].items():
            row = {'City': city, 'Pollutant': pollutant}
            # 为每一月添加累积变化率列
            for month in months:
                row[month] = monthly_cumulative_changes.get(month, np.nan)
            cumulative_data.append(row)

    if cumulative_data:
        df_cumulative = pd.DataFrame(cumulative_data)
        cumulative_path = os.path.join(output_dir, 'Monthly_Cumulative_Changes.csv')
        df_cumulative.to_csv(cumulative_path, index=False, encoding='utf-8-sig')
        print(f"累积变化百分比数据已保存到: {cumulative_path}")

    print(f"\n所有结果已保存到目录: {output_dir}")

def print_summary(results, months):
    """
    打印结果摘要
    """
    print("\n" + "="*60)
    print("月度统计分析摘要")
    print("="*60)

    for city, city_results in results.items():
        print(f"\n城市: {city}")
        print("-"*40)

        # 打印月均值
        print("月均值:")
        for pollutant, values in city_results['monthly_means'].items():
            print(f"  {pollutant}: ", end="")
            for month in months[:5]:  # 只显示前5个月的数据，避免输出过长
                if month in values:
                    print(f"{month}: {values[month]:.2f}  ", end="")
            if len(months) > 5:
                print("... (更多月份数据请查看CSV文件)")
            else:
                print()

        # 打印月际变化率
        print("月际变化率:")
        for pollutant, changes in city_results['monthly_changes'].items():
            if changes:
                print(f"  {pollutant}: ", end="")
                change_items = list(changes.items())[:3]  # 只显示前3个变化期
                for period, rate in change_items:
                    print(f"{period}: {rate}  ", end="")
                if len(changes) > 3:
                    print("... (更多变化期请查看CSV文件)")
                else:
                    print()

        # 打印累积变化（每月相对于2018年1月的变化率）
        print("累积变化百分比（相对于2018年1月）:")
        for pollutant, monthly_cumulative_changes in city_results['monthly_cumulative_changes'].items():
            print(f"  {pollutant}: ", end="")
            for month in months[:5]:  # 只显示前5个月的数据
                if month in monthly_cumulative_changes:
                    print(f"{month}: {monthly_cumulative_changes[month]}  ", end="")
            if len(months) > 5:
                print("... (更多月份数据请查看CSV文件)")
            else:
                print()

# 主程序
if __name__ == "__main__":
    # 设置文件夹路径（根据您的实际情况修改）
    folder_path = r"E:\DATA Science\大论文Result\PRD\filtered_daily"

    # 输出目录路径
    output_dir = r"E:\DATA Science\大论文Result\PRD\描述性统计分析"

    # 设置进程数（None表示使用所有CPU核心，也可以指定具体数字，如4）
    n_processes = None

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹路径不存在: {folder_path}")
        print("请确保路径正确，或者修改脚本中的folder_path变量")
    else:
        # 计算统计指标（使用多进程）
        results, months = calculate_city_statistics(folder_path, n_processes=n_processes)

        # 打印摘要
        print_summary(results, months)

        # 保存结果为3个CSV文件
        save_results_to_csv(results, months, output_dir)

        print("\n月度分析完成！")