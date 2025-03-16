import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def load_and_process_data(root_folder):
    """加载并处理数据，返回年度和季节性统计数据"""
    yearly_data = {}
    yearly_station_data = {}
    seasonal_data = {}
    pollutants = ['PM2.5', 'PM10', 'AQI']
    
    def get_season(month):
        if 3 <= month <= 5: return '春季'
        elif 6 <= month <= 8: return '夏季'
        elif 9 <= month <= 11: return '秋季'
        else: return '冬季'
    
    for folder in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            for file in sorted(os.listdir(folder_path)):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(folder_path, file))
                    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                    year = df['date'].dt.year.iloc[0]
                    
                    if year not in yearly_data:
                        yearly_data[year] = {p: [] for p in pollutants}
                        yearly_station_data[year] = {p: {} for p in pollutants}
                        seasonal_data[year] = {
                            season: {p: [] for p in pollutants}
                            for season in ['春季', '夏季', '秋季', '冬季']
                        }
                    
                    for pollutant in pollutants:
                        df_poll = df[df['type'] == pollutant]
                        if not df_poll.empty:
                            # 计算站点平均值
                            station_means = df_poll.iloc[:, 3:].mean()
                            overall_mean = station_means.mean()
                            yearly_data[year][pollutant].append(overall_mean)
                            
                            # 按季节分组计算
                            seasons = df_poll['date'].dt.month.map(get_season)
                            for season in seasons.unique():
                                season_data = df_poll[seasons == season].iloc[:, 3:].mean().mean()
                                seasonal_data[year][season][pollutant].append(season_data)
    
    # 计算最终平均值
    for year in yearly_data:
        for pollutant in pollutants:
            yearly_data[year][pollutant] = pd.Series(yearly_data[year][pollutant]).mean()
            for season in seasonal_data[year]:
                if seasonal_data[year][season][pollutant]:
                    seasonal_data[year][season][pollutant] = pd.Series(
                        seasonal_data[year][season][pollutant]
                    ).mean()
                else:
                    seasonal_data[year][season][pollutant] = 0
    
    return pd.DataFrame(yearly_data).T, yearly_station_data, seasonal_data

def calculate_and_plot_changes(yearly_df):
    """计算年际变化率并绘图"""
    change_rate = yearly_df.pct_change() * 100
    
    plt.figure(figsize=(10, 5))
    for pollutant in yearly_df.columns:
        plt.plot(yearly_df.index, yearly_df[pollutant], marker='o', label=pollutant)
    
    plt.xlabel("年份")
    plt.ylabel("平均浓度")
    plt.title("年度污染物浓度变化")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return change_rate

def plot_seasonal_changes(seasonal_data):
    """绘制季节变化图"""
    # 准备数据
    seasons = ['春季', '夏季', '秋季', '冬季']
    years = sorted(seasonal_data.keys())
    pollutants = ['PM2.5', 'PM10', 'AQI']
    
    # 创建子图
    fig, axes = plt.subplots(len(pollutants), 1, figsize=(12, 4*len(pollutants)))
    fig.suptitle('北京市污染物季节变化趋势', fontsize=16)
    
    for idx, pollutant in enumerate(pollutants):
        ax = axes[idx]
        data = []
        for year in years:
            year_data = []
            for season in seasons:
                year_data.append(seasonal_data[year][season][pollutant])
            data.append(year_data)
        
        # 创建热力图数据
        heatmap_data = pd.DataFrame(data, columns=seasons, index=years)
        
        # 绘制热力图
        sns.heatmap(heatmap_data, ax=ax, cmap='YlOrRd', 
                   annot=True, fmt='.1f', cbar_kws={'label': f'{pollutant} 浓度'})
        ax.set_title(f'{pollutant} 季节变化')
        ax.set_ylabel('年份')
    
    plt.tight_layout()
    plt.show()
    
    # 绘制季节变化折线图
    plt.figure(figsize=(15, 8))
    for pollutant in pollutants:
        seasonal_means = {season: [] for season in seasons}
        for year in years:
            for season in seasons:
                seasonal_means[season].append(seasonal_data[year][season][pollutant])
        
        for season in seasons:
            plt.plot(years, seasonal_means[season], 
                    marker='o', label=f'{pollutant}-{season}')
    
    plt.title('污染物季节变化趋势')
    plt.xlabel('年份')
    plt.ylabel('污染物浓度')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
    os.makedirs(output_dir, exist_ok=True)
    for year in sorted(yearly_station_data.keys()):
        data = {}
        stations = set()
        for pollutant in yearly_station_data[year]:
            stations.update(yearly_station_data[year][pollutant].keys())
        
        for pollutant in yearly_station_data[year]:
            data[pollutant] = [yearly_station_data[year][pollutant].get(station, 0) 
                              for station in sorted(stations)]
        
        df = pd.DataFrame(data, index=sorted(stations))
        df.index.name = 'Station'
        df.to_csv(os.path.join(output_dir, f"Beijing_AQI_{year}.csv"))

if __name__ == "__main__":
    root_folder = r"C:\Users\IU\Desktop\Beijing-AQI-Benchmark\all"
    output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Beijing_AQI_Yearly")
    
    yearly_df, yearly_station_data, seasonal_data = load_and_process_data(root_folder)
    
    # 绘制季节变化图
    plot_seasonal_changes(seasonal_data)
    
    # 计算年际变化率
    change_rate = calculate_and_plot_changes(yearly_df)
    print("Yearly Change Rate (%):\n", change_rate)
    
    # 打印各站点年平均值
    print("\n各站点年平均值:")
    print_station_data(yearly_station_data)
    
    # 保存年度数据到CSV
    save_yearly_data_to_csv(yearly_station_data, output_dir)
