import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from matplotlib.font_manager import FontProperties

# 设置中文字体
try:
    # 依次尝试不同的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果上述字体都不可用，尝试使用系统默认字体
    import matplotlib.font_manager as fm
    # 查找系统中的中文字体
    chinese_fonts = [f.name for f in fm.fontManager.ttflist if '黑体' in f.name or '微软雅黑' in f.name]
    if chinese_fonts:
        plt.rcParams['font.sans-serif'] = chinese_fonts
    plt.rcParams['axes.unicode_minus'] = False
    print("使用系统字体:", chinese_fonts[0] if chinese_fonts else "未找到合适的中文字体")

def load_and_process_data(root_folder):
    """加载并处理数据，返回季节性统计数据和站点季节数据"""
    seasonal_data = {}
    station_seasonal_data = {}  # 新增：存储站点季节数据
    pollutants = ['PM2.5', 'PM10', 'AQI']
    # pollutants = ['CO', 'NO2', 'SO2', 'O3']

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
                    file_path = os.path.join(folder_path, file)
                    try:
                        # 检查文件是否为空
                        if os.path.getsize(file_path) == 0:
                            print(f"警告: 文件为空 {file_path}")
                            continue
                            
                        df = pd.read_csv(file_path)
                        if df.empty or 'date' not in df.columns or 'type' not in df.columns:
                            continue
                            
                        df = df.fillna(0)
                        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
                        
                        if df['date'].isna().all():
                            continue
                            
                        year = df['date'].dt.year.iloc[0]
                        
                        if year not in seasonal_data:
                            seasonal_data[year] = {
                                season: {p: [] for p in pollutants}
                                for season in ['春季', '夏季', '秋季', '冬季']
                            }
                            station_seasonal_data[year] = {}  # 新增：初始化年份的站点数据
                        
                        for pollutant in pollutants:
                            df_poll = df[df['type'] == pollutant]
                            if not df_poll.empty and df_poll.shape[1] > 3:
                                # 获取站点列表
                                stations = df_poll.columns[3:]
                                
                                # 按季节分组计算每个站点的数据
                                seasons = df_poll['date'].dt.month.map(get_season)
                                for station in stations:
                                    if station not in station_seasonal_data[year]:
                                        station_seasonal_data[year][station] = {
                                            season: {p: [] for p in pollutants}
                                            for season in ['春季', '夏季', '秋季', '冬季']
                                        }
                                    
                                    for season in seasons.unique():
                                        season_data = df_poll[seasons == season][station].mean()
                                        station_seasonal_data[year][station][season][pollutant].append(season_data)
                                
                                # 计算总体季节平均值
                                for season in seasons.unique():
                                    season_data = df_poll[seasons == season].iloc[:, 3:].mean().mean()
                                    seasonal_data[year][season][pollutant].append(season_data)
                                    
                    except Exception as e:
                        print(f"错误: 处理文件时出错 {file_path}")
                        print(f"错误信息: {str(e)}")
                        continue
    
    # 计算最终平均值
    for year in seasonal_data:
        # 计算总体季节平均值
        for season in seasonal_data[year]:
            for pollutant in pollutants:
                if seasonal_data[year][season][pollutant]:
                    seasonal_data[year][season][pollutant] = pd.Series(
                        seasonal_data[year][season][pollutant]
                    ).mean()
                else:
                    seasonal_data[year][season][pollutant] = 0
        
        # 计算站点季节平均值
        for station in station_seasonal_data[year]:
            for season in station_seasonal_data[year][station]:
                for pollutant in pollutants:
                    values = station_seasonal_data[year][station][season][pollutant]
                    if values:
                        station_seasonal_data[year][station][season][pollutant] = sum(values) / len(values)
                    else:
                        station_seasonal_data[year][station][season][pollutant] = 0
    
    return seasonal_data, station_seasonal_data

def save_station_seasonal_data(station_seasonal_data, output_dir):
    """保存每个站点的季节数据到CSV文件"""
    output_dir = os.path.join(output_dir, "Station_Seasonal_Data")
    os.makedirs(output_dir, exist_ok=True)
    
    for year in sorted(station_seasonal_data.keys()):
        for station in sorted(station_seasonal_data[year].keys()):
            # 创建数据框
            seasons = ['春季', '夏季', '秋季', '冬季']
            pollutants = ['PM2.5', 'PM10', 'AQI']
            # pollutants = ['CO', 'NO2', 'SO2', 'O3']
            data = []
            for season in seasons:
                row = {'季节': season}
                for pollutant in pollutants:
                    row[pollutant] = station_seasonal_data[year][station][season][pollutant]
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # 保存到CSV
            filename = f"{year}-{station}.csv"
            df.to_csv(os.path.join(output_dir, filename), index=False, encoding='utf-8-sig')
            print(f"已保存文件: {filename}")

def plot_seasonal_changes(seasonal_data, output_dir):
    """为每种污染物分别绘制季节变化图"""
    # 创建图片保存目录
    img_output_dir = os.path.join(output_dir, "Season_Analysis_Plots")
    os.makedirs(img_output_dir, exist_ok=True)
    
    seasons = ['春季', '夏季', '秋季', '冬季']
    years = sorted(seasonal_data.keys())
    # pollutants = ['CO', 'NO2', 'SO2', 'O3']
    pollutants = ['PM2.5', 'PM10', 'AQI']  
    # 为每种污染物创建单独的图表
    for pollutant in pollutants:
        plt.figure(figsize=(12, 6))
        seasonal_means = {season: [] for season in seasons}
        
        # 收集该污染物的所有季节数据
        for year in years:
            for season in seasons:
                seasonal_means[season].append(seasonal_data[year][season][pollutant])
        
        # 绘制四季的变化趋势
        for season in seasons:
            plt.plot(years, seasonal_means[season], 
                    marker='o', linewidth=2, markersize=8,
                    label=f'{season}')
        
        plt.title(f'{pollutant}污染物季节变化趋势', fontsize=14, pad=15)
        plt.xlabel('年份', fontsize=12)
        plt.ylabel(f'{pollutant}浓度', fontsize=12)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置x轴刻度
        plt.xticks(years, rotation=45)
        
        # 添加网格线
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 调整布局以防止标签被切割
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(img_output_dir, f'{pollutant}_seasonal_trends.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    # 绘制热力图
    for pollutant in pollutants:
        plt.figure(figsize=(10, 6))
        data = []
        for year in years:
            year_data = []
            for season in seasons:
                year_data.append(seasonal_data[year][season][pollutant])
            data.append(year_data)
        
        # 创建热力图数据
        heatmap_data = pd.DataFrame(data, columns=seasons, index=years)
        
        # 绘制热力图
        sns.heatmap(heatmap_data, cmap='YlOrRd', 
                   annot=True, fmt='.2f', cbar_kws={'label': f'{pollutant} 浓度'})
        plt.title(f'{pollutant} 季节变化热力图')
        plt.ylabel('年份')
        
        plt.tight_layout()
        
        # 保存热力图
        plt.savefig(os.path.join(img_output_dir, f'{pollutant}_heatmap.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == "__main__":
    # root_folder = r"C:\Users\IU\Desktop\Beijing-AQI-Benchmark\extra"
    root_folder = r"C:\Users\IU\Desktop\Beijing-AQI-Benchmark\all"
    output_dir = os.path.join(os.path.expanduser("~"), "Desktop")
    
    seasonal_data, station_seasonal_data = load_and_process_data(root_folder)
    plot_seasonal_changes(seasonal_data, output_dir)
    save_station_seasonal_data(station_seasonal_data, output_dir)
