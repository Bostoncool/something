import os
import pandas as pd
import matplotlib.pyplot as plt

def load_and_process_data(root_folder):
    """遍历所有子文件夹，读取所有CSV文件，并合并数据"""
    yearly_data = {}
    yearly_station_data = {}  # 存储每个站点的年平均值
    pollutants = ['PM2.5', 'PM10', 'AQI']  # 可扩展其他污染物
    #pollutants = ['SO2', 'NO2', 'CO', 'O3']
    #两份代码合一，只需要改改文件路径，和污染物种类
    for folder in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            for file in sorted(os.listdir(folder_path)):
                if file.endswith(".csv"):
                    file_path = os.path.join(folder_path, file)
                    try:
                        df = pd.read_csv(file_path)
                        if df.empty:
                            print(f"Warning: {file_path} is empty.")
                            continue
                        if 'date' not in df.columns:
                            print(f"Error: {file_path} does not contain 'date' column.")
                            continue
                        
                        # 将空值填充为0
                        df = df.fillna(0)
                    except pd.errors.EmptyDataError:
                        print(f"Error: {file_path} is empty or has no columns.")
                        continue
                    
                    # 确保日期格式正确
                    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
                    year = df['date'].dt.year.iloc[0] if not df['date'].isna().all() else None
                    
                    if year:
                        if year not in yearly_data:
                            yearly_data[year] = {pollutant: [] for pollutant in pollutants}
                            yearly_station_data[year] = {pollutant: {} for pollutant in pollutants}
                        
                        for pollutant in pollutants:
                            df_pollutant = df[df['type'] == pollutant]
                            if not df_pollutant.empty:
                                # 计算每个站点的年平均值并存储
                                station_means = df_pollutant.iloc[:, 3:].mean()
                                
                                # 存储每个站点的数据
                                for station in station_means.index:
                                    if station not in yearly_station_data[year][pollutant]:
                                        yearly_station_data[year][pollutant][station] = []
                                    yearly_station_data[year][pollutant][station].append(station_means[station])
                                
                                # 计算所有站点的平均值
                                overall_mean = station_means.mean()
                                yearly_data[year][pollutant].append(overall_mean)

    # 计算每个站点的年平均值
    for year in yearly_station_data:
        for pollutant in pollutants:
            for station in yearly_station_data[year][pollutant]:
                values = yearly_station_data[year][pollutant][station]
                yearly_station_data[year][pollutant][station] = sum(values) / len(values) if values else 0

    # 将每年的数据转换为DataFrame
    for year in yearly_data:
        for pollutant in pollutants:
            yearly_data[year][pollutant] = pd.Series(yearly_data[year][pollutant]).mean()

    return pd.DataFrame(yearly_data).T, yearly_station_data

def calculate_and_plot_changes(yearly_df):
    """计算年际变化率，并进行可视化"""
    change_rate = yearly_df.pct_change() * 100
    
    # 可视化
    plt.figure(figsize=(10, 5))
    for pollutant in yearly_df.columns:
        plt.plot(yearly_df.index, yearly_df[pollutant], marker='o', label=pollutant)
    
    plt.xlabel("Year")
    plt.ylabel("Average Concentration")
    plt.title("Yearly Average Pollution Levels")
    plt.legend()
    plt.grid()
    plt.show()
    
    return change_rate

def print_station_data(yearly_station_data):
    """打印每个站点的年平均值"""
    for year in sorted(yearly_station_data.keys()):
        print(f"\n年份: {year}")
        for pollutant in yearly_station_data[year]:
            print(f"  污染物: {pollutant}")
            for station, value in yearly_station_data[year][pollutant].items():
                print(f"    站点 {station}: {value:.2f}")

def save_yearly_data_to_csv(yearly_station_data, output_dir):
    """将每年的站点数据保存到CSV文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    for year in sorted(yearly_station_data.keys()):
        # 为每年创建一个数据框
        data = {}
        stations = set()
        
        # 收集所有站点
        for pollutant in yearly_station_data[year]:
            stations.update(yearly_station_data[year][pollutant].keys())
        
        # 为每个污染物创建一列
        for pollutant in yearly_station_data[year]:
            data[pollutant] = []
            
            # 添加每个站点的数据
            for station in sorted(stations):
                value = yearly_station_data[year][pollutant].get(station, 0)
                data[pollutant].append(value)
        
        # 创建数据框
        df = pd.DataFrame(data, index=sorted(stations))
        df.index.name = 'Station'
        
        # 保存到CSV
        output_file = os.path.join(output_dir, f"Beijing_AQI_{year}.csv")
        df.to_csv(output_file)
        print(f"已保存 {year} 年数据到 {output_file}")

if __name__ == "__main__":
    root_folder = r"C:\Users\IU\Desktop\Beijing-AQI-Benchmark\all"  # 替换成你的数据存放路径
    output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Beijing_AQI_Yearly")  # 桌面上的输出目录
    
    yearly_df, yearly_station_data = load_and_process_data(root_folder)
    change_rate = calculate_and_plot_changes(yearly_df)
    
    print("Yearly Change Rate (%):\n", change_rate)
    
    # 打印每个站点的年平均值
    print("\n各站点年平均值:")
    print_station_data(yearly_station_data)
    
    # 保存每年的数据到CSV文件
    save_yearly_data_to_csv(yearly_station_data, output_dir)
