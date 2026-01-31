# 计算年均值，年均变化率和累积变化百分比，统计全年空气质量优良比例，重污染天气比例
# 其中是因为颗粒物导致的天气污染，占比是多少呢？
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

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
        # 筛选出AQI和污染物数据（包括实时和24小时平均数据）
        realtime_types = ['AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h', 
                         'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h']
        df_realtime = df[df['type'].isin(realtime_types)].copy()
        
        # 按小时计算平均值（对于每天的数据）
        daily_avg = df_realtime.groupby('type')[city_columns].mean()
        
        # 存储每年的数据
        year_data = {}
        
        # 对于每个城市，存储各污染物的年均值
        for city in city_columns:
            year_data[city] = {}
            for pollutant in realtime_types:
                if pollutant in daily_avg.index:
                    value = daily_avg.loc[pollutant, city]
                    if not pd.isna(value):
                        year_data[city][pollutant] = value
        
        return (year, year_data)
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def calculate_city_statistics(folder_path, n_processes=None):
    """
    计算每个城市的年均值、年际变化率和累积变化百分比
    
    参数:
    folder_path: CSV文件所在文件夹路径
    n_processes: 进程数，默认为CPU核心数
    
    累积变化百分比以2018年为基准，计算每一年的累积变化率
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
    
    # 5. 整理数据，计算统计指标
    years = sorted(all_data.keys())
    print(f"处理了 {len(years)} 年的数据: {years}")
    
    # 定义污染物类型
    realtime_types = ['AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h', 
                     'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h']
    
    results = {}
    
    # 对每个城市进行计算
    all_cities = set()
    for year_data in all_data.values():
        all_cities.update(year_data.keys())
    
    print(f"\n正在计算 {len(all_cities)} 个城市的统计指标...")
    for city in tqdm(all_cities, desc="计算城市统计", unit="城市"):
        results[city] = {
            'annual_means': {},  # 各年年均值
            'yearly_changes': {},  # 年际变化率
            'annual_cumulative_changes': {}  # 累积变化百分比
        }
        
        # 提取该城市各年的数据
        city_data = {}
        for year in years:
            if city in all_data[year]:
                city_data[year] = all_data[year][city]
        
        # 计算每个污染物的统计指标
        for pollutant in realtime_types:
            # 提取该污染物各年的值
            pollutant_values = {}
            for year in years:
                if year in city_data and pollutant in city_data[year]:
                    pollutant_values[year] = city_data[year][pollutant]
            
            if len(pollutant_values) >= 2:  # 至少需要2年的数据
                # 计算年均值
                results[city]['annual_means'][pollutant] = pollutant_values
                
                # 计算年际变化率
                yearly_changes = {}
                sorted_years = sorted(pollutant_values.keys())
                for i in range(1, len(sorted_years)):
                    year1 = sorted_years[i-1]
                    year2 = sorted_years[i]
                    if pollutant_values[year1] != 0:  # 避免除零错误
                        change_rate = ((pollutant_values[year2] - pollutant_values[year1]) / 
                                      pollutant_values[year1] * 100)
                        yearly_changes[f"{year1}-{year2}"] = f"{change_rate:.2f}%"
                
                results[city]['yearly_changes'][pollutant] = yearly_changes
                
                # 计算累积变化百分比（以2018年为基准，计算每一年的累积变化率）
                base_year = '2018'  # 基准年份
                annual_cumulative_changes = {}
                sorted_years = sorted(pollutant_values.keys())
                
                # 确保基准年份存在
                if base_year in pollutant_values and pollutant_values[base_year] != 0:
                    base_value = pollutant_values[base_year]
                    for year in sorted_years:
                        if year != base_year:  # 基准年份自身的变化率为0，不需要计算
                            cumulative_change = ((pollutant_values[year] - base_value) / base_value * 100)
                            annual_cumulative_changes[year] = f"{cumulative_change:.2f}%"
                    # 基准年份的变化率为0
                    annual_cumulative_changes[base_year] = "0.00%"
                else:
                    # 如果2018年不存在，使用首年作为基准
                    first_year = min(pollutant_values.keys())
                    if pollutant_values[first_year] != 0:
                        base_value = pollutant_values[first_year]
                        for year in sorted_years:
                            if year != first_year:
                                cumulative_change = ((pollutant_values[year] - base_value) / base_value * 100)
                                annual_cumulative_changes[year] = f"{cumulative_change:.2f}%"
                        annual_cumulative_changes[first_year] = "0.00%"
                
                results[city]['annual_cumulative_changes'][pollutant] = annual_cumulative_changes
    
    return results, years

def save_results_to_csv(results, years, output_dir):
    """
    将结果保存为3个CSV文件
    
    1. Annual_Means.csv - 各年年均值
    2. Yearly_Changes.csv - 年际变化率
    3. Cumulative_Changes.csv - 累积变化百分比（每年相对于2018年的变化率）
    """
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 创建年均值表格
    print("\n正在生成年均值数据...")
    annual_data = []
    for city, city_results in tqdm(results.items(), desc="处理年均值", unit="城市"):
        for pollutant, values in city_results['annual_means'].items():
            row = {'City': city, 'Pollutant': pollutant}
            for year in years:
                row[year] = values.get(year, np.nan)
            annual_data.append(row)
    
    if annual_data:
        df_annual = pd.DataFrame(annual_data)
        annual_path = os.path.join(output_dir, 'Annual_Means.csv')
        df_annual.to_csv(annual_path, index=False, encoding='utf-8-sig')
        print(f"年均值数据已保存到: {annual_path}")
    
    # 2. 创建年际变化率表格
    print("\n正在生成年际变化率数据...")
    change_data = []
    for city, city_results in tqdm(results.items(), desc="处理年际变化率", unit="城市"):
        for pollutant, changes in city_results['yearly_changes'].items():
            row = {'City': city, 'Pollutant': pollutant}
            for period, change_rate in changes.items():
                row[period] = change_rate
            change_data.append(row)
    
    if change_data:
        df_changes = pd.DataFrame(change_data)
        changes_path = os.path.join(output_dir, 'Yearly_Changes.csv')
        df_changes.to_csv(changes_path, index=False, encoding='utf-8-sig')
        print(f"年际变化率数据已保存到: {changes_path}")
    
    # 3. 创建累积变化百分比表格（每年相对于2018年的变化率）
    print("\n正在生成累积变化百分比数据...")
    cumulative_data = []
    for city, city_results in tqdm(results.items(), desc="处理累积变化率", unit="城市"):
        for pollutant, annual_cumulative_changes in city_results['annual_cumulative_changes'].items():
            row = {'City': city, 'Pollutant': pollutant}
            # 为每一年添加累积变化率列
            for year in years:
                row[year] = annual_cumulative_changes.get(year, np.nan)
            cumulative_data.append(row)
    
    if cumulative_data:
        df_cumulative = pd.DataFrame(cumulative_data)
        cumulative_path = os.path.join(output_dir, 'Annual_Cumulative_Changes.csv')
        df_cumulative.to_csv(cumulative_path, index=False, encoding='utf-8-sig')
        print(f"累积变化百分比数据已保存到: {cumulative_path}")
    
    print(f"\n所有结果已保存到目录: {output_dir}")

def print_summary(results, years):
    """
    打印结果摘要
    """
    print("\n" + "="*60)
    print("统计分析摘要")
    print("="*60)
    
    for city, city_results in results.items():
        print(f"\n城市: {city}")
        print("-"*40)
        
        # 打印年均值
        print("年均值:")
        for pollutant, values in city_results['annual_means'].items():
            print(f"  {pollutant}: ", end="")
            for year in years:
                if year in values:
                    print(f"{year}: {values[year]:.2f}  ", end="")
            print()
        
        # 打印年际变化率
        print("年际变化率:")
        for pollutant, changes in city_results['yearly_changes'].items():
            if changes:
                print(f"  {pollutant}: ", end="")
                for period, rate in changes.items():
                    print(f"{period}: {rate}  ", end="")
                print()
        
        # 打印累积变化（每年相对于2018年的变化率）
        print("累积变化百分比（相对于2018年）:")
        for pollutant, annual_cumulative_changes in city_results['annual_cumulative_changes'].items():
            print(f"  {pollutant}: ", end="")
            for year in years:
                if year in annual_cumulative_changes:
                    print(f"{year}: {annual_cumulative_changes[year]}  ", end="")
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
        results, years = calculate_city_statistics(folder_path, n_processes=n_processes)
        
        # 打印摘要
        print_summary(results, years)
        
        # 保存结果为3个CSV文件
        save_results_to_csv(results, years, output_dir)
        
        print("\n分析完成！")