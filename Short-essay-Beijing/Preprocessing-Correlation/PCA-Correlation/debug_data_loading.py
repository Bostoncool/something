#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试数据加载问题
"""

import os
import pandas as pd
import numpy as np

def debug_data_loading():
    """调试数据加载问题"""
    
    # 设置数据文件夹路径
    meteo_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijin-CSV"
    pollution_data_dir = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)"
    
    print("="*80)
    print("数据加载调试")
    print("="*80)
    
    # 1. 检查目录
    print("\n1. 检查目录:")
    print(f"气象数据目录: {meteo_data_dir}")
    print(f"  存在: {os.path.exists(meteo_data_dir)}")
    print(f"污染数据目录: {pollution_data_dir}")
    print(f"  存在: {os.path.exists(pollution_data_dir)}")
    
    # 2. 检查气象数据文件
    print("\n2. 检查气象数据文件:")
    if os.path.exists(meteo_data_dir):
        meteo_files = []
        for root, dirs, files in os.walk(meteo_data_dir):
            for file in files:
                if file.endswith('.csv'):
                    meteo_files.append(os.path.join(root, file))
        
        print(f"找到 {len(meteo_files)} 个CSV文件")
        
        # 检查文件命名
        year_month_files = []
        for file in meteo_files:
            filename = os.path.basename(file)
            if len(filename) == 11 and filename.endswith('.csv') and filename[:8].isdigit():
                year_month_files.append(file)
        
        print(f"符合YYYYMM.csv格式的文件: {len(year_month_files)}")
        
        # 检查前几个文件
        for i, file in enumerate(year_month_files[:3]):
            print(f"\n检查文件 {i+1}: {os.path.basename(file)}")
            try:
                df = pd.read_csv(file)
                print(f"  形状: {df.shape}")
                print(f"  列名: {list(df.columns)}")
                
                # 检查气象参数
                meteo_columns = ['t2m', 'blh', 'cvh', 'avg_tprate', 'u10', 'v10', 'u100', 'v100', 
                               'lsm', 'cvl', 'mn2t', 'sp', 'sd', 'str', 'tisr', 'tcwv', 'tp']
                
                available_cols = [col for col in meteo_columns if col in df.columns]
                print(f"  包含的气象参数: {len(available_cols)} 个")
                print(f"  参数列表: {available_cols}")
                
                # 检查数据质量
                for col in available_cols[:3]:
                    if col in df.columns:
                        non_null_count = df[col].notna().sum()
                        print(f"  {col}: 非空值 {non_null_count}/{len(df)}")
                        if col in ['t2m', 'mn2t']:
                            temp_kelvin = df[col].values
                            temp_celsius = temp_kelvin - 273.15
                            print(f"    温度范围: {temp_celsius.min():.2f}°C - {temp_celsius.max():.2f}°C")
                        else:
                            print(f"    数值范围: {df[col].min():.2f} - {df[col].max():.2f}")
                
            except Exception as e:
                print(f"  读取错误: {e}")
    
    # 3. 检查污染数据文件
    print("\n3. 检查污染数据文件:")
    if os.path.exists(pollution_data_dir):
        pollution_files = []
        for root, dirs, files in os.walk(pollution_data_dir):
            for file in files:
                if file.startswith('beijing_all_') and file.endswith('.csv'):
                    pollution_files.append(os.path.join(root, file))
        
        print(f"找到 {len(pollution_files)} 个污染数据文件")
        
        # 检查前几个文件
        for i, file in enumerate(pollution_files[:3]):
            print(f"\n检查文件 {i+1}: {os.path.basename(file)}")
            try:
                df = pd.read_csv(file)
                print(f"  形状: {df.shape}")
                print(f"  列名: {list(df.columns)}")
                
                if 'type' in df.columns:
                    unique_types = df['type'].unique()
                    print(f"  数据类型: {list(unique_types)}")
                    
                    for data_type in unique_types:
                        type_data = df[df['type'] == data_type]
                        print(f"  {data_type}: {len(type_data)} 行")
                        
                        # 检查数据列
                        if len(type_data) > 0:
                            data_columns = type_data.iloc[:, 3:].columns
                            print(f"    数据列数: {len(data_columns)}")
                            
                            # 检查数据质量
                            sample_data = type_data.iloc[:, 3:].values
                            if sample_data.size > 0:
                                print(f"    数据形状: {sample_data.shape}")
                                print(f"    非空值比例: {np.sum(~np.isnan(sample_data)) / sample_data.size:.2%}")
                
            except Exception as e:
                print(f"  读取错误: {e}")
    
    # 4. 模拟数据加载过程
    print("\n4. 模拟数据加载过程:")
    
    # 模拟气象数据加载
    meteo_data = []
    if os.path.exists(meteo_data_dir):
        for year in range(2015, 2025):
            for month in range(1, 13):
                filename_pattern = f"{year}{month:02d}.csv"
                found_files = []
                
                for root, dirs, files in os.walk(meteo_data_dir):
                    for file in files:
                        if filename_pattern in file:
                            found_files.append(os.path.join(root, file))
                
                for filepath in found_files:
                    try:
                        df = pd.read_csv(filepath)
                        if not df.empty:
                            # 检查气象参数
                            meteo_columns = ['t2m', 'blh', 'cvh', 'avg_tprate', 'u10', 'v10', 'u100', 'v100', 
                                           'lsm', 'cvl', 'mn2t', 'sp', 'sd', 'str', 'tisr', 'tcwv', 'tp']
                            available_columns = [col for col in meteo_columns if col in df.columns]
                            
                            if available_columns:
                                monthly_stats = {}
                                
                                for col in available_columns:
                                    if col in df.columns:
                                        # 处理温度参数
                                        if col in ['t2m', 'mn2t']:
                                            temp_kelvin = df[col].values
                                            temp_celsius = temp_kelvin - 273.15
                                            values = temp_celsius
                                        else:
                                            values = df[col].values
                                        
                                        # 计算统计量
                                        daily_stats = calculate_daily_stats(values)
                                        
                                        monthly_stats[f'{col}_mean'] = daily_stats['mean']
                                        monthly_stats[f'{col}_std'] = daily_stats['std']
                                        monthly_stats[f'{col}_min'] = daily_stats['min']
                                        monthly_stats[f'{col}_max'] = daily_stats['max']
                                
                                monthly_stats['year'] = year
                                monthly_stats['month'] = month
                                
                                meteo_data.append(monthly_stats)
                                print(f"  ✓ 加载: {os.path.basename(filepath)} ({len(available_columns)} 参数)")
                            else:
                                print(f"  ✗ 跳过: {os.path.basename(filepath)} (无气象参数)")
                    except Exception as e:
                        print(f"  ✗ 错误: {os.path.basename(filepath)} - {e}")
    
    print(f"\n气象数据加载结果: {len(meteo_data)} 个文件")
    
    # 模拟污染数据加载
    pollution_data = []
    if os.path.exists(pollution_data_dir):
        def pollution_file_filter(filename):
            return filename.startswith('beijing_all_') and filename.endswith('.csv')
        
        found_pollution_files = []
        for root, dirs, files in os.walk(pollution_data_dir):
            for file in files:
                if pollution_file_filter(file):
                    found_pollution_files.append(os.path.join(root, file))
        
        for filepath in found_pollution_files:
            try:
                df = pd.read_csv(filepath)
                if not df.empty and 'type' in df.columns:
                    pm25_data = df[df['type'] == 'PM2.5'].iloc[:, 3:].values
                    pm10_data = df[df['type'] == 'PM10'].iloc[:, 3:].values
                    aqi_data = df[df['type'] == 'AQI'].iloc[:, 3:].values
                    
                    if len(pm25_data) > 0:
                        daily_pm25 = np.nanmean(pm25_data, axis=0) if pm25_data.ndim > 1 else pm25_data
                        daily_pm10 = np.nanmean(pm10_data, axis=0) if pm10_data.ndim > 1 else pm10_data
                        daily_aqi = np.nanmean(aqi_data, axis=0) if aqi_data.ndim > 1 else aqi_data
                        
                        pollution_data.append({
                            'pm25_mean': np.nanmean(daily_pm25),
                            'pm10_mean': np.nanmean(daily_pm10),
                            'aqi_mean': np.nanmean(daily_aqi),
                            'pm25_std': np.nanstd(daily_pm25),
                            'pm10_std': np.nanstd(daily_pm10),
                            'aqi_std': np.nanstd(daily_aqi)
                        })
                        print(f"  ✓ 加载: {os.path.basename(filepath)}")
                    else:
                        print(f"  ✗ 跳过: {os.path.basename(filepath)} (无PM2.5数据)")
                else:
                    print(f"  ✗ 跳过: {os.path.basename(filepath)} (格式错误)")
            except Exception as e:
                print(f"  ✗ 错误: {os.path.basename(filepath)} - {e}")
    
    print(f"\n污染数据加载结果: {len(pollution_data)} 个文件")
    
    # 5. 模拟数据合并
    print("\n5. 模拟数据合并:")
    if meteo_data and pollution_data:
        meteo_df = pd.DataFrame(meteo_data)
        pollution_df = pd.DataFrame(pollution_data)
        
        print(f"气象数据形状: {meteo_df.shape}")
        print(f"污染数据形状: {pollution_df.shape}")
        
        # 检查NaN值
        print(f"气象数据NaN值: {meteo_df.isna().sum().sum()}")
        print(f"污染数据NaN值: {pollution_df.isna().sum().sum()}")
        
        # 对齐数据
        min_len = min(len(meteo_df), len(pollution_df))
        meteo_df = meteo_df.head(min_len)
        pollution_df = pollution_df.head(min_len)
        
        # 合并数据
        combined_data = pd.concat([meteo_df, pollution_df], axis=1)
        print(f"合并后数据形状: {combined_data.shape}")
        
        # 检查NaN值
        nan_counts = combined_data.isna().sum()
        print(f"合并后NaN值分布:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"  {col}: {count}")
        
        # 尝试不同的清理策略
        print(f"\n尝试不同的数据清理策略:")
        
        # 策略1: 移除完全空行
        data1 = combined_data.dropna(how='all')
        print(f"策略1 (移除完全空行): {data1.shape}")
        
        # 策略2: 前向填充
        data2 = combined_data.fillna(method='ffill').fillna(method='bfill')
        print(f"策略2 (前向填充): {data2.shape}")
        
        # 策略3: 均值填充
        data3 = combined_data.copy()
        numeric_columns = data3.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data3[col].isna().any():
                mean_val = data3[col].mean()
                if not pd.isna(mean_val):
                    data3[col].fillna(mean_val, inplace=True)
        print(f"策略3 (均值填充): {data3.shape}")
        
        # 策略4: 只保留完整行
        data4 = combined_data.dropna()
        print(f"策略4 (只保留完整行): {data4.shape}")
        
        print(f"\n推荐使用策略2或策略3，因为它们保留了更多数据")
    
    else:
        print("没有足够的数据进行合并测试")

def calculate_daily_stats(hourly_data):
    """计算每日统计量"""
    daily_data = []
    for i in range(0, len(hourly_data), 24):
        day_data = hourly_data[i:i+24]
        if len(day_data) > 0:
            daily_data.append({
                'mean': np.nanmean(day_data),
                'std': np.nanstd(day_data),
                'min': np.nanmin(day_data),
                'max': np.nanmax(day_data)
            })
    
    if daily_data:
        return {
            'mean': np.nanmean([d['mean'] for d in daily_data]),
            'std': np.nanmean([d['std'] for d in daily_data]),
            'min': np.nanmin([d['min'] for d in daily_data]),
            'max': np.nanmax([d['max'] for d in daily_data])
        }
    else:
        return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}

if __name__ == "__main__":
    debug_data_loading() 