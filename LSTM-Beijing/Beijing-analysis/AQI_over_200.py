import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def read_all_csv_files(root_folder): # 读取指定文件夹及其子文件夹中的所有CSV文件

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

    # 处理数据，统一格式，以便筛选数据
    # AQI > 200
    # CO_24h > 24
    # CO > 60
    # NO2_24h > 240
    # NO2 > 1200
    # O3_24h > 265
    # O3 > 400
    # PM2.5_24h > 150
    # PM10_24h > 350
    # SO2_24h > 800
    # SO2 > 800

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
        global index_name 
        index_name = 'SO2'
        index_df = df[df['type'].astype(str).str.contains(index_name, regex=True, case=False)].copy()
        print(f"{index_name}数据筛选后记录数: {len(index_df)}")
        
        if len(index_df) == 0:
            print(f"警告：未找到{index_name}数据")
            return pd.DataFrame()
        
        # 获取日期信息（如果需要）
        if 'date' not in index_df.columns:
            # 从type列或文件名中提取日期
            date_pattern = r'(\d{8})'
            index_df['date'] = index_df['type'].astype(str).str.extract(date_pattern)[0]
        
        # 将宽格式转换为长格式
        # 排除非数据列
        non_data_cols = ['type', 'date']
        station_cols = [col for col in index_df.columns if col not in non_data_cols]
        
        # 转换为长格式
        melted_df = pd.melt(index_df,
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

def gradient_image(ax, extent, direction=0, cmap_range=(0, 0.5), **kwargs):
    # 创建渐变效果
    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    X = np.array([[v @ [0, 0], v @ [0, 0]],
                  [v @ [1, 1], v @ [1, 1]]])
    a, b = cmap_range
    X = a + (b - a) / X.max() * X
    im = ax.imshow(X, extent=extent, interpolation='bicubic',
                   vmin=0, vmax=1, aspect='auto', **kwargs)
    return im

def gradient_bar(ax, x, y, width=0.5, bottom=0, color_start=(0.8, 0.2, 0.2), color_end=(1, 0.6, 0.6)):
    
    # 绘制渐变柱状图
    for left, top in zip(x, y):
        left = left - width/2
        right = left + width
        colors = [color_start, color_end]
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
        gradient_image(ax, extent=(left, right, bottom, top),
                       cmap=cmap, cmap_range=(0, 0.8))

def create_output_folder(index_name):
    """创建输出文件夹并返回路径"""
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    folder_name = f"{index_name}超标分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_folder = os.path.join(desktop_path, folder_name)
    
    # 创建文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出文件夹：{output_folder}")
    
    return output_folder

def save_data_to_folder(df, filename, output_folder):
    """将数据保存到指定文件夹"""
    output_path = os.path.join(output_folder, filename)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"数据已保存至：{output_path}")

def analyze_heavy_pollution(df):

    # 分析数据，筛选阈值
    # 确保所有数值都是有效的
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['value'] = df['value'].fillna(0)
    
    # 筛选阈值超标的数据
    index_threshold = 800
    heavy_pollution = df[df['value'] > index_threshold].copy()
    
    # 分析重污染天气的基本统计信息
    total_days = len(df['date'].dt.date.unique())
    heavy_days = len(heavy_pollution['date'].dt.date.unique())
    heavy_rate = heavy_days / total_days * 100 if total_days > 0 else 0
    
    print(f"总天数: {total_days}天")
    print(f"重污染天数({index_name} > {index_threshold}): {heavy_days}天")
    print(f"重污染发生率: {heavy_rate:.2f}%")
    
    # 按年份、月份、季节汇总统计
    yearly_summary = heavy_pollution.groupby(['year', 'station'])['value'].agg(['count', 'mean', 'min', 'max']).reset_index()
    monthly_summary = heavy_pollution.groupby(['year', 'month', 'station'])['value'].agg(['count', 'mean', 'min', 'max']).reset_index()
    seasonal_summary = heavy_pollution.groupby(['year', 'season', 'station'])['value'].agg(['count', 'mean', 'min', 'max']).reset_index()
    
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
    
    # 创建输出文件夹
    output_folder = create_output_folder(index_name)
    
    # 可视化分析结果
    plt.style.use('bmh')
    fig = plt.figure(figsize=(20, 16))
    
    # 年度趋势
    ax1 = plt.subplot(2, 2, 1)
    x = np.arange(len(yearly_rate))
    gradient_bar(ax1, x + 0.5, yearly_rate.values, width=0.7,
                color_start=(0.8, 0.2, 0.2),  # 深红色
                color_end=(1, 0.6, 0.6))      # 浅红色
    
    plt.title(f'{index_name}超标年度发生率(%)', pad=20)
    plt.ylabel('发生率(%)')
    plt.xlabel('年份')
    
    # 设置y轴范围，留出适当空间显示数值标签
    y_max = yearly_rate.max()
    plt.ylim(0, y_max * 1.15)  # 留出15%的空间显示标签
    
    # 设置x轴范围，确保柱子完整显示
    plt.xlim(0, len(yearly_rate))
    
    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(x + 0.5, yearly_rate.index, rotation=45, ha='right')
    
    # 为柱状图添加数值标签，调整位置
    for i, v in enumerate(yearly_rate.values):
        ax1.text(i + 0.5, v + (y_max * 0.02), f'{v:.1f}%',  # 将i+1改为i+0.5
                 ha='center', va='bottom', fontsize=10)
    
    # 月度分布
    ax2 = plt.subplot(2, 2, 2)
    monthly_rate.plot(kind='bar', color=sns.color_palette('YlOrRd', n_colors=12), ax=ax2)
    plt.title(f'{index_name}超标月度发生率(%)', pad=20)
    plt.ylabel('发生率(%)')
    plt.xlabel('月份')
    plt.xticks(range(12), ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月'], 
               rotation = 45, ha ='right')
    plt.grid(axis='y', linestyle='--', alpha = 0.7)
    
    # 季节分布
    ax3 = plt.subplot(2, 2, 3)
    seasonal_rate.plot(kind='bar', color=sns.color_palette('Greens', n_colors=4), ax=ax3)
    plt.title(f'{index_name}超标季节发生率(%)', pad=20)
    plt.ylabel('发生率(%)')
    plt.xlabel('季节')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)  # 季节标签不需要旋转
    
    # 站点分布
    ax4 = plt.subplot(2, 2, 4)
    # 只显示前10个站点，其他归类为"其他"
    station_counts = heavy_pollution.groupby('station').size().sort_values(ascending=False)
    top_10_stations = station_counts.head(10)
    others = pd.Series({'其他': station_counts[10:].sum()})
    station_data = pd.concat([top_10_stations, others])
    station_rates = station_data / station_data.sum() * 100
    
    colors = sns.color_palette('Set3', n_colors=len(station_rates))
    # 设置饼图的起始角度和方向
    startangle = 90
    # 绘制饼图，设置标签位置为外部
    wedges, texts, autotexts = ax4.pie(station_rates, 
                                      labels=station_rates.index,
                                      colors=colors,
                                      startangle=startangle,
                                      labeldistance = 1.2,  # 标签距离中心的距离
                                      pctdistance = 0.85,   # 百分比标签距离中心的距离
                                      autopct=lambda pct: f'{pct:.1f}%' if pct >= 2 else '',  # 只显示大于2%的标签
                                      textprops={'fontsize': 10})
    
    # 添加图例
    ax4.legend(wedges, station_rates.index,
              title="监测站点",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.title(f'各站点{index_name}超标分布（前10站点）', pad=20)
    
    # 调整布局，确保图表不重叠
    plt.tight_layout(pad=3.0)
    
    # 添加整体标题
    fig.suptitle(f'北京市{index_name}超标分析报告', fontsize=16, y=1.02)
    
    # 保存图表
    plt.savefig(os.path.join(output_folder, f'{index_name}超标分析.png'), 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # 保存数据文件
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_data_to_folder(heavy_pollution, f"超标数据_{timestamp}.csv", output_folder)
    save_data_to_folder(yearly_summary, f"年度统计_{timestamp}.csv", output_folder)
    save_data_to_folder(monthly_summary, f"月度统计_{timestamp}.csv", output_folder)
    save_data_to_folder(seasonal_summary, f"季节统计_{timestamp}.csv", output_folder)
    
    return heavy_pollution

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
    
    # 分析数据
    heavy_pollution = analyze_heavy_pollution(processed_data)
    
    print("分析完成！所有结果已保存到桌面新建的文件夹中。")

if __name__ == "__main__":
    main()
