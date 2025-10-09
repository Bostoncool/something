"""
测试数据加载和合并 - 快速验证修复效果
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob

# 测试参数
start_date = datetime(2015, 1, 1)
end_date = datetime(2015, 1, 31)  # 只测试1月份

pollution_all_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)'
era5_path = r'C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV'

pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
beijing_lats = np.arange(39.0, 41.25, 0.25)
beijing_lons = np.arange(115.0, 117.25, 0.25)
era5_vars = ['d2m', 't2m', 'u10', 'v10', 'u100', 'v100', 'blh', 'cvh', 'lsm', 'cvl',
             'avg_tprate', 'mn2t', 'sd', 'str', 'sp', 'tisr', 'tcwv', 'tp']

def daterange(start, end):
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def find_file(base_path, date_str, prefix):
    filename = f"{prefix}_{date_str}.csv"
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

print("=" * 80)
print("测试数据加载和日期转换")
print("=" * 80)

# 测试1: 读取单日污染数据
print("\n测试1: 读取2015-01-01污染数据...")
date = datetime(2015, 1, 1)
date_str = date.strftime('%Y%m%d')
all_file = find_file(pollution_all_path, date_str, 'beijing_all')
extra_file = find_file(pollution_extra_path, date_str, 'beijing_extra')

if all_file and extra_file:
    df_all = pd.read_csv(all_file, encoding='utf-8', on_bad_lines='skip')
    df_extra = pd.read_csv(extra_file, encoding='utf-8', on_bad_lines='skip')
    
    df_all = df_all[~df_all['type'].str.contains('_24h|AQI', na=False)]
    df_extra = df_extra[~df_extra['type'].str.contains('_24h', na=False)]
    
    df_poll = pd.concat([df_all, df_extra], ignore_index=True)
    df_poll = df_poll.melt(id_vars=['date', 'hour', 'type'], 
                           var_name='station', value_name='value')
    df_poll['value'] = pd.to_numeric(df_poll['value'], errors='coerce')
    df_poll = df_poll[df_poll['value'] >= 0]
    
    df_daily = df_poll.groupby(['date', 'type'])['value'].mean().reset_index()
    df_daily = df_daily.pivot(index='date', columns='type', values='value')
    
    print(f"转换前的索引类型: {type(df_daily.index[0])}")
    print(f"转换前的索引值: {df_daily.index[0]}")
    
    # 关键修复：转换日期格式
    df_daily.index = pd.to_datetime(df_daily.index, format='%Y%m%d', errors='coerce')
    
    print(f"转换后的索引类型: {type(df_daily.index[0])}")
    print(f"转换后的索引值: {df_daily.index[0]}")
    print(f"数据形状: {df_daily.shape}")
    print(f"✓ 污染数据读取成功")
else:
    print("✗ 未找到污染数据文件")

# 测试2: 读取2015年1月ERA5数据
print("\n测试2: 读取2015年1月ERA5数据...")
month_str = "201501"
all_files = glob.glob(os.path.join(era5_path, "**", f"*{month_str}*.csv"), recursive=True)
print(f"找到 {len(all_files)} 个文件")

if all_files:
    monthly_data = None
    for file_path in all_files[:3]:  # 只测试前3个文件
        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', 
                            low_memory=False, comment='#')
            
            if df.empty or 'time' not in df.columns:
                continue
            
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.dropna(subset=['time'])
            
            if 'latitude' in df.columns and 'longitude' in df.columns:
                df = df[(df['latitude'] >= beijing_lats.min()) & 
                       (df['latitude'] <= beijing_lats.max()) &
                       (df['longitude'] >= beijing_lons.min()) & 
                       (df['longitude'] <= beijing_lons.max())]
            
            df['date'] = df['time'].dt.date
            avail_vars = [v for v in era5_vars if v in df.columns]
            
            if avail_vars:
                for col in avail_vars:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df_daily_era5 = df.groupby('date')[avail_vars].mean().reset_index()
                df_daily_era5.set_index('date', inplace=True)
                df_daily_era5.index = pd.to_datetime(df_daily_era5.index)
                
                if monthly_data is None:
                    monthly_data = df_daily_era5
                else:
                    monthly_data = monthly_data.join(df_daily_era5, how='outer')
                
                print(f"  ✓ 读取文件: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  ✗ 读取文件失败: {os.path.basename(file_path)}")
    
    if monthly_data is not None:
        print(f"\nERA5数据形状: {monthly_data.shape}")
        print(f"时间范围: {monthly_data.index.min()} 至 {monthly_data.index.max()}")
        print(f"✓ ERA5数据读取成功")
    else:
        print("✗ ERA5数据合并失败")
else:
    print("✗ 未找到ERA5数据文件")

# 测试3: 数据合并
print("\n测试3: 数据合并测试...")
if all_file and extra_file and all_files and monthly_data is not None:
    print(f"污染数据时间范围: {df_daily.index.min()} 至 {df_daily.index.max()}")
    print(f"气象数据时间范围: {monthly_data.index.min()} 至 {monthly_data.index.max()}")
    
    combined = df_daily.join(monthly_data, how='inner')
    print(f"\n合并后数据形状: {combined.shape}")
    
    if len(combined) > 0:
        print(f"合并后时间范围: {combined.index.min()} 至 {combined.index.max()}")
        print("✓✓✓ 数据合并成功！问题已解决！")
    else:
        print("✗ 数据合并后为空")
else:
    print("跳过合并测试")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)

