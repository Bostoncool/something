import pandas as pd
import numpy as np
import os
import glob
import multiprocessing
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_aqi_level(aqi_value):
    """根据AQI值判断空气质量等级"""
    if pd.isna(aqi_value):
        return "数据缺失"
    elif aqi_value <= 50:
        return "优"
    elif aqi_value <= 100:
        return "良"
    elif aqi_value <= 150:
        return "轻度污染"
    elif aqi_value <= 200:
        return "中度污染"
    elif aqi_value <= 300:
        return "重度污染"
    else:
        return "严重污染"

def process_single_file(file_path):
    """处理单个CSV文件，提取AQI数据"""
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # 从文件名提取日期
        file_name = os.path.basename(file_path)
        # 假设文件名格式为: china_cities_YYYYMMDD.csv
        date_str = file_name.split('_')[-1].replace('.csv', '')
        
        # 提取城市列（排除元数据列）
        city_columns = [col for col in df.columns if col not in ['date', 'hour', 'type', '__file__', '__missing_cols__']]
        
        # 筛选AQI数据（排除24小时平均数据）
        aqi_data = df[df['type'] == 'AQI'].copy()
        
        if aqi_data.empty:
            # 如果没有AQI数据，尝试使用其他名称
            if 'aqi' in df['type'].str.lower().unique():
                aqi_data = df[df['type'].str.lower() == 'aqi'].copy()
            else:
                return None
        
        # 计算每个城市的日均AQI（当天所有小时的平均值）
        daily_aqi = {}
        
        for city in city_columns:
            # 获取该城市所有小时的AQI值
            city_aqi_values = aqi_data[city].dropna()
            
            if len(city_aqi_values) > 0:
                # 计算日均值
                daily_avg = city_aqi_values.mean()
                # 判断空气质量等级
                aqi_level = get_aqi_level(daily_avg)
                
                daily_aqi[city] = {
                    'date': date_str,
                    'city': city,
                    'daily_aqi': round(daily_avg, 2),
                    'aqi_level': aqi_level
                }
        
        # 转换为DataFrame
        if daily_aqi:
            result_df = pd.DataFrame(list(daily_aqi.values()))
            return result_df
        else:
            return None
            
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def process_files_parallel(file_paths, num_processes=None):
    """多进程并行处理文件"""
    if num_processes is None:
        num_processes = cpu_count() - 1  # 留一个核心给系统
    
    print(f"使用 {num_processes} 个进程并行处理 {len(file_paths)} 个文件...")
    
    # 使用进程池并行处理
    with Pool(processes=num_processes) as pool:
        # 使用tqdm显示进度条
        results = list(tqdm(
            pool.imap(process_single_file, file_paths),
            total=len(file_paths),
            desc="处理文件中"
        ))
    
    # 过滤掉None结果
    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        # 合并所有结果
        combined_df = pd.concat(valid_results, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

def analyze_aqi_trends(aqi_df):
    """分析AQI趋势和统计"""
    if aqi_df.empty:
        print("没有有效数据进行分析")
        return None
    
    # 将日期列转换为datetime格式
    aqi_df['date'] = pd.to_datetime(aqi_df['date'], format='%Y%m%d', errors='coerce')
    aqi_df['year'] = aqi_df['date'].dt.year
    aqi_df['month'] = aqi_df['date'].dt.month
    
    # 按城市和年份统计
    city_stats = []
    
    for city in aqi_df['city'].unique():
        city_data = aqi_df[aqi_df['city'] == city].copy()
        
        # 按年份统计
        yearly_stats = []
        for year in sorted(city_data['year'].unique()):
            year_data = city_data[city_data['year'] == year]
            
            # 计算年平均值
            yearly_avg = year_data['daily_aqi'].mean()
            
            # 计算各等级天数
            level_counts = year_data['aqi_level'].value_counts()
            
            yearly_stats.append({
                'city': city,
                'year': year,
                'yearly_avg_aqi': round(yearly_avg, 2),
                '优_天数': level_counts.get('优', 0),
                '良_天数': level_counts.get('良', 0),
                '轻度污染_天数': level_counts.get('轻度污染', 0),
                '中度污染_天数': level_counts.get('中度污染', 0),
                '重度污染_天数': level_counts.get('重度污染', 0),
                '严重污染_天数': level_counts.get('严重污染', 0),
                '数据缺失_天数': level_counts.get('数据缺失', 0),
                '总天数': len(year_data)
            })
        
        # 计算年际变化
        if len(yearly_stats) > 1:
            # 按年份排序
            yearly_stats_sorted = sorted(yearly_stats, key=lambda x: x['year'])
            
            for i in range(1, len(yearly_stats_sorted)):
                prev_avg = yearly_stats_sorted[i-1]['yearly_avg_aqi']
                curr_avg = yearly_stats_sorted[i]['yearly_avg_aqi']
                
                if prev_avg > 0:  # 避免除零错误
                    change_rate = ((curr_avg - prev_avg) / prev_avg * 100)
                    yearly_stats_sorted[i]['相对于前一年变化率(%)'] = round(change_rate, 2)
                else:
                    yearly_stats_sorted[i]['相对于前一年变化率(%)'] = np.nan
            
            city_stats.extend(yearly_stats_sorted)
    
    # 创建统计DataFrame
    stats_df = pd.DataFrame(city_stats)
    
    # 重新排列列顺序
    col_order = ['city', 'year', 'yearly_avg_aqi', '相对于前一年变化率(%)', 
                 '优_天数', '良_天数', '轻度污染_天数', '中度污染_天数', 
                 '重度污染_天数', '严重污染_天数', '数据缺失_天数', '总天数']
    
    # 只保留存在的列
    existing_cols = [col for col in col_order if col in stats_df.columns]
    stats_df = stats_df[existing_cols]
    
    return stats_df

def main():
    # 设置文件夹路径
    folder_path = r"E:\DATA Science\大论文Result\YZD\filtered_daily"
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹路径不存在: {folder_path}")
        print("请确保路径正确，或者修改脚本中的folder_path变量")
        return
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"在文件夹 {folder_path} 中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 多进程处理文件
    aqi_results = process_files_parallel(csv_files)
    
    if aqi_results.empty:
        print("没有提取到有效的AQI数据")
        return
    
    print(f"\n成功处理了 {len(aqi_results['date'].unique())} 天的数据")
    print(f"包含 {len(aqi_results['city'].unique())} 个城市")
    
    # 保存原始AQI数据
    output_file1 = r"E:\DATA Science\大论文Result\YZD\daily_aqi_data.csv"
    aqi_results.to_csv(output_file1, index=False, encoding='utf-8-sig')
    print(f"\n日均AQI数据已保存到: {output_file1}")
    
    # 分析AQI趋势
    stats_results = analyze_aqi_trends(aqi_results)
    
    if stats_results is not None and not stats_results.empty:
        # 保存统计分析结果
        output_file2 = r"E:\DATA Science\大论文Result\YZD\aqi_statistics.csv"
        stats_results.to_csv(output_file2, index=False, encoding='utf-8-sig')
        print(f"AQI统计分析结果已保存到: {output_file2}")
        
        # 打印摘要统计
        print("\n" + "="*80)
        print("AQI统计分析摘要")
        print("="*80)
        
        # 按城市打印统计
        for city in stats_results['city'].unique():
            city_stats = stats_results[stats_results['city'] == city]
            print(f"\n{city} - AQI统计:")
            print("-"*40)
            print(city_stats.to_string(index=False))
    
    # 生成各城市空气质量等级分布汇总
    if not aqi_results.empty:
        # 计算各城市各等级天数占比
        level_distribution = pd.crosstab(
            aqi_results['city'], 
            aqi_results['aqi_level'],
            normalize='index'
        ).round(4) * 100
        
        # 添加平均AQI
        avg_aqi = aqi_results.groupby('city')['daily_aqi'].mean().round(2)
        level_distribution['平均AQI'] = avg_aqi
        
        # 重新排列列
        level_order = ['优', '良', '轻度污染', '中度污染', '重度污染', '严重污染', '数据缺失']
        existing_levels = [l for l in level_order if l in level_distribution.columns]
        level_distribution = level_distribution[existing_levels + ['平均AQI']]
        
        # 保存等级分布
        output_file3 = r"E:\DATA Science\大论文Result\YZD\aqi_level_distribution.csv"
        level_distribution.to_csv(output_file3, encoding='utf-8-sig')
        print(f"\n空气质量等级分布已保存到: {output_file3}")
        
        print("\n" + "="*80)
        print("各城市空气质量等级分布(%)")
        print("="*80)
        print(level_distribution.round(2).to_string())

if __name__ == "__main__":
    # 设置多进程启动方式
    multiprocessing.freeze_support()
    
    # 运行主程序
    main()