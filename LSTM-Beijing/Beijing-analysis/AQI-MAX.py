import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def read_all_csv_files(root_folder):
    """
    读取指定文件夹及其子文件夹中的所有CSV文件
    """
    all_data = []
    
    # 遍历所有子文件夹
    for folder_name, subfolders, filenames in os.walk(root_folder):
        # 获取当前文件夹下所有CSV文件
        csv_files = [f for f in filenames if f.endswith('.csv')]
        
        for file in csv_files:
            file_path = os.path.join(folder_name, file)
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 检查是否为空
                if df.empty:
                    print(f"警告：文件为空 - {file_path}")
                    continue
                
                # 确保有日期列
                if 'date' not in df.columns:
                    # 尝试从文件名中提取日期信息
                    match = re.search(r'(\d{8})', file)
                    if match:
                        df['date'] = match.group(1)
                    else:
                        print(f"警告：无法获取日期信息 - {file_path}")
                        continue
                
                # 将所有空值填充为0
                df = df.fillna(0)
                
                # 添加文件源信息
                df['source_file'] = os.path.basename(file_path)
                
                # 将数据添加到列表
                all_data.append(df)
                
            except Exception as e:
                print(f"错误：读取文件时出错 - {file_path}")
                print(f"错误信息：{str(e)}")
    
    # 合并所有数据框
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # 确保所有空值都被填充为0
        combined_df = combined_df.fillna(0)
        return combined_df
    else:
        print("没有找到有效的数据文件")
        return pd.DataFrame()

def process_aqi_data(df):
    """
    处理数据，统一格式，以便筛选AQI>200的数据
    """
    try:
        print("开始处理数据...")
        
        # 确保所有空值被填充为0
        df = df.fillna(0)
        
        # 检查是否已经有type列
        if 'type' not in df.columns:
            # 如果没有type列，说明需要从第一列获取type信息
            df = df.copy()
            df['type'] = df.iloc[:, 0]  # 第一列包含type信息
        
        # 筛选包含AQI的行（考虑到可能有数字前缀）
        aqi_df = df[df['type'].astype(str).str.contains('NO2', regex=True, case=False)].copy()
        print(f"AQI数据筛选后记录数: {len(aqi_df)}")
        
        if len(aqi_df) == 0:
            print("警告：未找到AQI数据")
            return pd.DataFrame()
        
        # 获取日期信息（如果需要）
        if 'date' not in aqi_df.columns:
            # 从type列或文件名中提取日期
            date_pattern = r'(\d{8})'
            aqi_df['date'] = aqi_df['type'].astype(str).str.extract(date_pattern)[0]
        
        # 将宽格式转换为长格式
        # 排除非数据列
        non_data_cols = ['type', 'date']
        station_cols = [col for col in aqi_df.columns if col not in non_data_cols]
        
        # 转换为长格式
        melted_df = pd.melt(aqi_df,
                           id_vars=['date', 'type'],
                           value_vars=station_cols,
                           var_name='station',
                           value_name='value')
        
        # 确保日期列为datetime格式
        melted_df['date'] = pd.to_datetime(melted_df['date'], format='%Y%m%d')
        
        # 添加年份、月份列
        melted_df['year'] = melted_df['date'].dt.year
        melted_df['month'] = melted_df['date'].dt.month
        
        # 按照中国四季划分季节
        conditions = [
            (melted_df['month'].isin([3, 4, 5])),
            (melted_df['month'].isin([6, 7, 8])),
            (melted_df['month'].isin([9, 10, 11])),
            (melted_df['month'].isin([12, 1, 2]))
        ]
        seasons = ['春季', '夏季', '秋季', '冬季']
        melted_df['season'] = np.select(conditions, seasons, default='未知')
        
        # 确保数值列为数值类型并替换所有空值为0
        melted_df['value'] = pd.to_numeric(melted_df['value'], errors='coerce')
        melted_df['value'] = melted_df['value'].fillna(0)
        
        # 按日期排序
        melted_df = melted_df.sort_values('date')
        
        print(f"数据处理完成，最终记录数: {len(melted_df)}")
        return melted_df
        
    except Exception as e:
        print(f"处理数据时出错：{str(e)}")
        print("错误详细信息:")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame()

def analyze_heavy_pollution(df):
    """
    分析AQI数据并筛选AQI>200的重污染天气数据
    """
    # 确保所有数值都是有效的
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['value'] = df['value'].fillna(0)
    
    # 筛选AQI>200的重污染天气
    heavy_pollution = df[df['value'] > 1200].copy()
    
    # 分析重污染天气的基本统计信息
    total_days = len(df['date'].dt.date.unique())
    heavy_days = len(heavy_pollution['date'].dt.date.unique())
    heavy_rate = heavy_days / total_days * 100 if total_days > 0 else 0
    
    print(f"总天数: {total_days}天")
    print(f"重污染天数(AQI>200): {heavy_days}天")
    print(f"重污染发生率: {heavy_rate:.2f}%")
    
    # 按年份分析
    yearly_stats = heavy_pollution.groupby(['year', 'date']).size().reset_index().groupby('year').size()
    yearly_total = df.groupby(['year', 'date']).size().reset_index().groupby('year').size()
    yearly_rate = (yearly_stats / yearly_total * 100).fillna(0)
    
    # 按月份分析
    monthly_stats = heavy_pollution.groupby(['month', 'date']).size().reset_index().groupby('month').size()
    monthly_total = df.groupby(['month', 'date']).size().reset_index().groupby('month').size()
    monthly_rate = (monthly_stats / monthly_total * 100).fillna(0)
    
    # 按季节分析
    seasonal_stats = heavy_pollution.groupby(['season', 'date']).size().reset_index().groupby('season').size()
    seasonal_total = df.groupby(['season', 'date']).size().reset_index().groupby('season').size()
    seasonal_rate = (seasonal_stats / seasonal_total * 100).fillna(0)
    
    # 可视化分析结果
    plt.figure(figsize=(12, 8))
    
    # 年度趋势
    plt.subplot(2, 2, 1)
    yearly_rate.plot(kind='bar', color='darkred')
    plt.title('重污染天气年度发生率(%)')
    plt.ylabel('发生率(%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 月度分布
    plt.subplot(2, 2, 2)
    monthly_rate.plot(kind='bar', color='orange')
    plt.title('重污染天气月度发生率(%)')
    plt.ylabel('发生率(%)')
    plt.xticks(range(12), ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 季节分布
    plt.subplot(2, 2, 3)
    seasonal_rate.plot(kind='bar', color='darkgreen')
    plt.title('重污染天气季节发生率(%)')
    plt.ylabel('发生率(%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 站点分布
    plt.subplot(2, 2, 4)
    station_counts = heavy_pollution.groupby('station').size().sort_values(ascending=False)
    station_rates = station_counts / heavy_pollution.shape[0] * 100
    station_rates.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('Set3'), textprops={'fontsize': 8})
    plt.title('各站点重污染天气分布')
    
    plt.tight_layout()
    
    # 保存图表到桌面
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    plt.savefig(os.path.join(desktop_path, '重污染天气分析.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return heavy_pollution

def save_data_to_desktop(df, filename):
    """
    将数据保存到桌面
    """
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    output_path = os.path.join(desktop_path, filename)
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"数据已保存至：{output_path}")

def main():
    # 读取数据
    root_folder = r"C:\Users\IU\Desktop\Beijing-AQI-Benchmark"
    
    if not os.path.exists(root_folder):
        print("错误：指定的文件夹不存在")
        return
    
    print(f"开始读取文件夹：{root_folder}")
    raw_data = read_all_csv_files(root_folder)
    
    if raw_data.empty:
        print("错误：未读取到有效数据")
        return
    
    print(f"数据读取完成，共读取 {raw_data.shape[0]} 条记录")
    
    # 处理数据
    processed_data = process_aqi_data(raw_data)
    
    if processed_data.empty:
        print("错误：数据处理失败")
        return
    
    print(f"数据处理完成，共处理 {processed_data.shape[0]} 条记录")
    
    # 分析重污染天气数据
    heavy_pollution = analyze_heavy_pollution(processed_data)
    
    # 将重污染天气数据保存到桌面
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_data_to_desktop(heavy_pollution, f"重污染天气数据_{timestamp}.csv")
    
    # 按年份、月份、季节汇总统计重污染天气
    yearly_summary = heavy_pollution.groupby(['year', 'station'])['value'].agg(['count', 'mean', 'min', 'max']).reset_index()
    monthly_summary = heavy_pollution.groupby(['year', 'month', 'station'])['value'].agg(['count', 'mean', 'min', 'max']).reset_index()
    seasonal_summary = heavy_pollution.groupby(['year', 'season', 'station'])['value'].agg(['count', 'mean', 'min', 'max']).reset_index()
    
    # 保存汇总数据
    save_data_to_desktop(yearly_summary, f"重污染天气年度统计_{timestamp}.csv")
    save_data_to_desktop(monthly_summary, f"重污染天气月度统计_{timestamp}.csv")
    save_data_to_desktop(seasonal_summary, f"重污染天气季节统计_{timestamp}.csv")
    
    print("分析完成！")

if __name__ == "__main__":
    main()
