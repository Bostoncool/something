import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def get_season_from_month(month):
    """
    根据月份返回对应的季节

    参数:
    month: 月份 (1-12)

    返回:
    (season_name, season_key) 元组
    """
    month = int(month)
    if month in [3, 4, 5]:
        return ('春', 'spring')
    elif month in [6, 7, 8]:
        return ('夏', 'summer')
    elif month in [9, 10, 11]:
        return ('秋', 'autumn')
    elif month in [12, 1, 2]:
        return ('冬', 'winter')
    else:
        return ('未知', 'unknown')

def get_season_key(year, month):
    """
    根据年月生成季节键

    参数:
    year: 年份
    month: 月份

    返回:
    季节键，格式为 'YYYY-季节'
    """
    season_name, season_key = get_season_from_month(month)
    # 对于冬季（12月、1月、2月），如果月份是12月，年份加1
    if month == 12:
        return f"{int(year)+1}-{season_name}"
    else:
        return f"{year}-{season_name}"

def process_single_file(file_path):
    """
    处理单个CSV文件，提取年月和城市污染物数据

    参数:
    file_path: CSV文件路径

    返回:
    (season_key, season_data) 元组，season_data是字典 {city: {pollutant: value}}
    season_key格式为'YYYY-季节'
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

        # 将月度数据转换为季度数据
        season_key = get_season_key(date_str[:4], date_str[4:6])

        return (season_key, month_data)

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def calculate_city_statistics(folder_path, n_processes=None):
    """
    计算每个城市的季度均值、季度际变化率和累积变化百分比

    参数:
    folder_path: CSV文件所在文件夹路径
    n_processes: 进程数，默认为CPU核心数

    累积变化百分比以2018年春季为基准，计算每一季度的累积变化率
    """

    # 存储所有季度数据的字典
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

    # 合并结果，按季度聚合数据
    seasonal_data = {}
    for result in results:
        if result is not None:
            season_key, month_data = result
            if season_key not in seasonal_data:
                seasonal_data[season_key] = {}
            # 合并同一季度的数据
            for city, city_data in month_data.items():
                if city not in seasonal_data[season_key]:
                    seasonal_data[season_key][city] = {}
                for pollutant, value in city_data.items():
                    if pollutant not in seasonal_data[season_key][city]:
                        seasonal_data[season_key][city][pollutant] = []
                    seasonal_data[season_key][city][pollutant].append(value)

    # 计算季度均值
    all_data = {}
    for season, season_data in seasonal_data.items():
        all_data[season] = {}
        for city, city_data in season_data.items():
            all_data[season][city] = {}
            for pollutant, values in city_data.items():
                if values:  # 确保有数据
                    all_data[season][city][pollutant] = np.mean(values)

    # 整理数据，计算统计指标
    seasons = sorted(all_data.keys())
    print(f"处理了 {len(seasons)} 个季度的数据: {seasons}")

    # 定义污染物类型
    realtime_types = ['AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h',
                     'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h']

    results = {}

    # 对每个城市进行计算
    all_cities = set()
    for season_data in all_data.values():
        all_cities.update(season_data.keys())

    print(f"\n正在计算 {len(all_cities)} 个城市的统计指标...")
    for city in tqdm(all_cities, desc="计算城市统计", unit="城市"):
        results[city] = {
            'seasonal_means': {},  # 各季度均值
            'seasonal_changes': {},  # 季度际变化率
            'seasonal_cumulative_changes': {}  # 累积变化百分比
        }

        # 提取该城市各季度的数据
        city_data = {}
        for season in seasons:
            if city in all_data[season]:
                city_data[season] = all_data[season][city]

        # 计算每个污染物的统计指标
        for pollutant in realtime_types:
            # 提取该污染物各季度的平均值
            pollutant_values = {}
            for season in seasons:
                if season in city_data and pollutant in city_data[season]:
                    pollutant_values[season] = city_data[season][pollutant]

            if len(pollutant_values) >= 2:  # 至少需要2个季度的数据
                # 计算季度均值
                results[city]['seasonal_means'][pollutant] = pollutant_values

                # 计算季度际变化率
                seasonal_changes = {}
                sorted_seasons = sorted(pollutant_values.keys())
                for i in range(1, len(sorted_seasons)):
                    season1 = sorted_seasons[i-1]
                    season2 = sorted_seasons[i]
                    if pollutant_values[season1] != 0:  # 避免除零错误
                        change_rate = ((pollutant_values[season2] - pollutant_values[season1]) /
                                      pollutant_values[season1] * 100)
                        seasonal_changes[f"{season1}-{season2}"] = f"{change_rate:.2f}%"

                results[city]['seasonal_changes'][pollutant] = seasonal_changes

                # 计算累积变化百分比（以2018年春季为基准，计算每一季度的累积变化率）
                base_season = '2018-春'  # 基准季度
                seasonal_cumulative_changes = {}
                sorted_seasons = sorted(pollutant_values.keys())

                # 确保基准季度存在
                if base_season in pollutant_values and pollutant_values[base_season] != 0:
                    base_value = pollutant_values[base_season]
                    for season in sorted_seasons:
                        if season != base_season:  # 基准季度自身的变化率为0，不需要计算
                            cumulative_change = ((pollutant_values[season] - base_value) / base_value * 100)
                            seasonal_cumulative_changes[season] = f"{cumulative_change:.2f}%"
                    # 基准季度的变化率为0
                    seasonal_cumulative_changes[base_season] = "0.00%"
                else:
                    # 如果2018年春季不存在，使用第一个季度作为基准
                    first_season = min(pollutant_values.keys())
                    if pollutant_values[first_season] != 0:
                        base_value = pollutant_values[first_season]
                        for season in sorted_seasons:
                            if season != first_season:
                                cumulative_change = ((pollutant_values[season] - base_value) / base_value * 100)
                                seasonal_cumulative_changes[season] = f"{cumulative_change:.2f}%"
                        seasonal_cumulative_changes[first_season] = "0.00%"

                results[city]['seasonal_cumulative_changes'][pollutant] = seasonal_cumulative_changes

    return results, seasons

def save_results_to_csv(results, seasons, output_dir):
    """
    将结果保存为3个CSV文件

    1. Seasonal_Means.csv - 各季度均值
    2. Seasonal_Changes.csv - 季度际变化率
    3. Cumulative_Changes.csv - 累积变化百分比（每季度相对于2018年春季的变化率）
    """

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 1. 创建季度均值表格
    print("\n正在生成季度均值数据...")
    seasonal_data = []
    for city, city_results in tqdm(results.items(), desc="处理季度均值", unit="城市"):
        for pollutant, values in city_results['seasonal_means'].items():
            row = {'City': city, 'Pollutant': pollutant}
            for season in seasons:
                row[season] = values.get(season, np.nan)
            seasonal_data.append(row)

    if seasonal_data:
        df_seasonal = pd.DataFrame(seasonal_data)
        seasonal_path = os.path.join(output_dir, 'Seasonal_Means.csv')
        df_seasonal.to_csv(seasonal_path, index=False, encoding='utf-8-sig')
        print(f"季度均值数据已保存到: {seasonal_path}")

    # 2. 创建季度际变化率表格
    print("\n正在生成季度际变化率数据...")
    change_data = []
    for city, city_results in tqdm(results.items(), desc="处理季度际变化率", unit="城市"):
        for pollutant, changes in city_results['seasonal_changes'].items():
            row = {'City': city, 'Pollutant': pollutant}
            for period, change_rate in changes.items():
                row[period] = change_rate
            change_data.append(row)

    if change_data:
        df_changes = pd.DataFrame(change_data)
        changes_path = os.path.join(output_dir, 'Seasonal_Changes.csv')
        df_changes.to_csv(changes_path, index=False, encoding='utf-8-sig')
        print(f"季度际变化率数据已保存到: {changes_path}")

    # 3. 创建累积变化百分比表格（每季度相对于2018年春季的变化率）
    print("\n正在生成累积变化百分比数据...")
    cumulative_data = []
    for city, city_results in tqdm(results.items(), desc="处理累积变化率", unit="城市"):
        for pollutant, seasonal_cumulative_changes in city_results['seasonal_cumulative_changes'].items():
            row = {'City': city, 'Pollutant': pollutant}
            # 为每一季度添加累积变化率列
            for season in seasons:
                row[season] = seasonal_cumulative_changes.get(season, np.nan)
            cumulative_data.append(row)

    if cumulative_data:
        df_cumulative = pd.DataFrame(cumulative_data)
        cumulative_path = os.path.join(output_dir, 'Season_Cumulative_Changes.csv')
        df_cumulative.to_csv(cumulative_path, index=False, encoding='utf-8-sig')
        print(f"累积变化百分比数据已保存到: {cumulative_path}")

    print(f"\n所有结果已保存到目录: {output_dir}")

def print_summary(results, seasons):
    """
    打印结果摘要
    """
    print("\n" + "="*60)
    print("季度统计分析摘要")
    print("="*60)

    for city, city_results in results.items():
        print(f"\n城市: {city}")
        print("-"*40)

        # 打印季度均值
        print("季度均值:")
        for pollutant, values in city_results['seasonal_means'].items():
            print(f"  {pollutant}: ", end="")
            for season in seasons[:5]:  # 只显示前5个季度的数据，避免输出过长
                if season in values:
                    print(f"{season}: {values[season]:.2f}  ", end="")
            if len(seasons) > 5:
                print("... (更多季度数据请查看CSV文件)")
            else:
                print()

        # 打印季度际变化率
        print("季度际变化率:")
        for pollutant, changes in city_results['seasonal_changes'].items():
            if changes:
                print(f"  {pollutant}: ", end="")
                change_items = list(changes.items())[:3]  # 只显示前3个变化期
                for period, rate in change_items:
                    print(f"{period}: {rate}  ", end="")
                if len(changes) > 3:
                    print("... (更多变化期请查看CSV文件)")
                else:
                    print()

        # 打印累积变化（每季度相对于2018年春季的变化率）
        print("累积变化百分比（相对于2018年春季）:")
        for pollutant, seasonal_cumulative_changes in city_results['seasonal_cumulative_changes'].items():
            print(f"  {pollutant}: ", end="")
            for season in seasons[:5]:  # 只显示前5个季度的数据
                if season in seasonal_cumulative_changes:
                    print(f"{season}: {seasonal_cumulative_changes[season]}  ", end="")
            if len(seasons) > 5:
                print("... (更多季度数据请查看CSV文件)")
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
        results, seasons = calculate_city_statistics(folder_path, n_processes=n_processes)

        # 打印摘要
        print_summary(results, seasons)

        # 保存结果为3个CSV文件
        save_results_to_csv(results, seasons, output_dir)

        print("\n季度分析完成！")