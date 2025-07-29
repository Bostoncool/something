import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import os
import sys

# 设置中文字体
rcParams['font.family'] = 'Microsoft YaHei'  # 或者 'SimHei'、'Microsoft YaHei' 等
rcParams['axes.unicode_minus'] = False  # 显示负号

# 创建保存图表的文件夹
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# 定义污染物类型
pollutants = ["PM2.5", "PM10", "AQI"]
# pollutants = ["CO","NO2","O3","SO2"]

# 按年份循环处理数据
for year in range(2015, 2025): # 左闭右开
    year = str(year)  # 转换为字符串
    print(f"\n开始处理 {year} 年的数据...")
    
    # 创建年份对应的文件夹
    folder1 = f"-Overall_{year}"
    folder2 = f"-Station_{year}"
    
    overall_plots_folder = os.path.join(desktop_path, folder1)
    station_plots_folder = os.path.join(desktop_path, folder2)

    # 如果文件夹不存在，则创建
    if not os.path.exists(overall_plots_folder):
        os.makedirs(overall_plots_folder)
    if not os.path.exists(station_plots_folder):
        os.makedirs(station_plots_folder)

    # 设定该年份的文件夹路径
    # path_name = fr"C:\Users\IU\Desktop\Beijing-AQI-Date\beijing_{year}0101-{year}1231\all"
    # path_name = fr"E:\DATA Science\Datebase Origin\Date\beijing_{year}0101-{year}1231\extra"
    path_name = fr"E:\DATA Science\Datebase Origin\Date\beijing_{year}0101-{year}1231\all"
    folder_path = os.path.join(path_name)

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"警告：{year}年的文件夹 {folder_path} 不存在，跳过该年份")
        continue

    # 获取该年份的所有CSV文件
    file_paths = sorted(glob.glob(os.path.join(folder_path, f"beijing_extra_{year}*.csv")))

    if not file_paths:
        print(f"警告：在 {folder_path} 中没有找到任何CSV文件，跳过该年份")
        continue

    # 读取并合并该年份的所有CSV文件
    df_list = []
    for file in file_paths:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                df_list.append(df)
                print(f"成功读取文件：{file}")
            else:
                print(f"警告：文件 {file} 是空的，跳过该文件。")
        except Exception as e:
            print(f"警告：读取文件 {file} 时发生错误：{str(e)}，跳过该文件。")

    if not df_list:
        print(f"警告：{year}年没有找到任何有效的CSV文件，跳过该年份")
        continue

    # 合并该年份的数据
    df = pd.concat(df_list, ignore_index=True)
    print(f"成功合并了 {len(df_list)} 个文件，总行数：{len(df)}")

    # 选择主要污染物
    df_filtered = df[df["type"].isin(pollutants)].copy()

    if df_filtered.empty:
        print(f"警告：{year}年筛选后的数据为空，跳过该年份")
        continue

    # 解析日期格式
    df_filtered["date"] = pd.to_datetime(df_filtered["date"], format="%Y%m%d")

    # 提取所有监测站点列
    non_station_cols = ["date", "hour", "type"]
    station_cols = [col for col in df_filtered.columns if col not in non_station_cols]
    
    # 清理站点名称，确保它们是有效的文件名
    def clean_filename(name):
        # 移除或替换无效字符
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        return name.strip()
    
    # 创建列名映射字典
    column_mapping = {col: clean_filename(col) for col in station_cols}
    
    # 重命名数据框的列
    df_filtered = df_filtered.rename(columns=column_mapping)
    
    # 更新站点列名列表
    station_cols = list(column_mapping.values())
    
    # 计算每个站点的每日均值
    df_station_avg = df_filtered.groupby(["date", "type"])[station_cols].mean().reset_index()

    # 计算所有站点整体均值
    df_overall_avg = df_station_avg.groupby(["date", "type"]).mean().reset_index()

    # 绘制全站点均值的趋势图
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(f"北京{year}年主要污染物变化趋势 (全站点平均值)", fontsize=14)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, (pollutant, color) in enumerate(zip(pollutants, colors)):
        df_plot = df_overall_avg[df_overall_avg["type"] == pollutant]
        axes[idx].plot(df_plot["date"], df_plot[station_cols].mean(axis=1), 
                      marker="o", color=color, label=pollutant)
        axes[idx].set_xlabel("日期")
        axes[idx].set_ylabel(f"{pollutant} 浓度 (μg/m³)")
        axes[idx].legend()
        axes[idx].grid(True)
        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    overall_plot_path = os.path.join(overall_plots_folder, f"Beijing_{year}_Overall_Pollutants_Trend_Separate.png")
    plt.savefig(overall_plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存

    # 绘制每个站点的趋势图
    for station in station_cols:
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        fig.suptitle(f"{station} {year}年污染物变化趋势", fontsize=14)
        
        for idx, (pollutant, color) in enumerate(zip(pollutants, colors)):
            df_plot = df_station_avg[df_station_avg["type"] == pollutant]
            axes[idx].plot(df_plot["date"], df_plot[station], 
                          marker="o", color=color, label=pollutant)
            axes[idx].set_xlabel("日期")
            axes[idx].set_ylabel(f"{pollutant} 浓度 (μg/m³)")
            axes[idx].legend()
            axes[idx].grid(True)
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        station_plot_path = os.path.join(station_plots_folder, f"{station}_{year}_Pollutants_Trend_Separate.png")
        plt.savefig(station_plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，释放内存

    print(f"{year}年数据处理完成！")
    print(f"全站点均值趋势图已保存至: {overall_plots_folder}")
    print(f"各站点趋势图已保存至: {station_plots_folder}")

    # 清理内存
    del df, df_filtered, df_station_avg, df_overall_avg, df_list
    import gc
    gc.collect()

print("\n所有年份数据处理完成！")
