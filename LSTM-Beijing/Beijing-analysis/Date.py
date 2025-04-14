import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import os

# 设置中文字体
rcParams['font.family'] = 'Microsoft YaHei'  # 或者 'SimHei'、'Microsoft YaHei' 等
rcParams['axes.unicode_minus'] = False  # 显示负号

# 创建保存图表的文件夹
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
overall_plots_folder = os.path.join(desktop_path, "Beijing_Overall_Plots")
station_plots_folder = os.path.join(desktop_path, "Beijing_Station_Plots")

# 如果文件夹不存在，则创建
if not os.path.exists(overall_plots_folder):
    os.makedirs(overall_plots_folder)
if not os.path.exists(station_plots_folder):
    os.makedirs(station_plots_folder)

# 1. 设定文件夹路径
folder_path = r"C:\Users\IU\Desktop\Beijing-AQI-Date\beijing_20240101-20241231\all"  # 你的文件夹路径
file_paths = sorted(glob.glob(os.path.join(folder_path, "beijing_all_2024*.csv")))

# folder_path = r"C:\Users\IU\Desktop\Beijing-AQI-Date\beijing_20240101-20241231\extra"  # 你的文件夹路径
# file_paths = sorted(glob.glob(os.path.join(folder_path, "beijing_extra_2024*.csv")))

#两份文件只需要改改文件路径，和污染物种类，就能完成


# 打印找到的文件路径
print("找到的文件路径：", file_paths)

# 2. 读取所有 CSV 文件并合并
df_list = []
for file in file_paths:
    try:
        df = pd.read_csv(file)
        df_list.append(df)
    except pd.errors.EmptyDataError:
        print(f"文件 {file} 是空的，跳过该文件。")
    except FileNotFoundError:
        print(f"文件 {file} 未找到，请检查路径。")

# 检查 df_list 是否为空
if not df_list:
    print("没有找到任何有效的 CSV 文件，请检查文件路径和文件内容。")
else:
    df = pd.concat(df_list, ignore_index=True)

# 3. 选择主要污染物
pollutants = ["PM2.5", "PM10", "AQI"]
#pollutants = ["SO2", "NO2", "CO", "O3"]
#两份代码合一，只需要改改文件路径，和污染物种类
df_filtered = df[df["type"].isin(pollutants)].copy()

# 4. 解析日期格式
df_filtered["date"] = pd.to_datetime(df_filtered["date"], format="%Y%m%d")

# 5. 提取所有监测站点列（去掉非站点的列）
non_station_cols = ["date", "hour", "type"]
station_cols = [col for col in df_filtered.columns if col not in non_station_cols]

# 6. 计算 **每个站点** 的每日均值
df_station_avg = df_filtered.groupby(["date", "type"])[station_cols].mean().reset_index()

# 7. 计算 **所有站点整体均值**
df_overall_avg = df_station_avg.groupby(["date", "type"]).mean().reset_index()

# 8. 绘制 **全站点均值的趋势图**
plt.figure(figsize=(12, 6))
for pollutant in pollutants:
    df_plot = df_overall_avg[df_overall_avg["type"] == pollutant]
    plt.plot(df_plot["date"], df_plot[station_cols].mean(axis=1), marker="o", label=pollutant)

plt.xlabel("日期")
plt.ylabel("污染物浓度 (全站点平均值)")
plt.title("北京年度主要污染物变化趋势 (全站点平均值)")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()  # 调整布局，确保标签不被裁剪

# 保存全站点均值趋势图
overall_plot_path = os.path.join(overall_plots_folder, "Beijing_Overall_Pollutants_Trend.png")
plt.savefig(overall_plot_path, dpi=300)
# plt.show()

# 9. **绘制每个站点的趋势图**
for station in station_cols:
    plt.figure(figsize=(12, 6))
    for pollutant in pollutants:
        df_plot = df_station_avg[df_station_avg["type"] == pollutant]
        plt.plot(df_plot["date"], df_plot[station], marker="o", label=pollutant)
    
    plt.xlabel("日期")
    plt.ylabel(f"{station} 污染物浓度(μg/m³)")
    plt.title(f"{station} 污染物变化趋势")  
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()  # 调整布局，确保标签不被裁剪
    
    # 保存每个站点的趋势图
    station_plot_path = os.path.join(station_plots_folder, f"{station}_Pollutants_Trend.png")
    plt.savefig(station_plot_path, dpi=300)
    # plt.show()

print(f"全站点均值趋势图已保存至: {overall_plots_folder}")
print(f"各站点趋势图已保存至: {station_plots_folder}")
