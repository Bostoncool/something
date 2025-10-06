import netCDF4 as nc
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from datetime import datetime
import glob

def convert_nc_to_csv(args):
    """
    将单个NC文件转换为CSV格式
    """
    input_file, output_dir = args
    
    try:
        print(f"正在处理: {os.path.basename(input_file)}")
        
        # 读取NC文件
        dataset = nc.Dataset(input_file, 'r')
        
        # 获取维度信息
        time_dim = dataset.dimensions['valid_time']
        lat_dim = dataset.dimensions['latitude'] 
        lon_dim = dataset.dimensions['longitude']
        
        # 读取坐标变量
        times = dataset.variables['valid_time'][:]
        latitudes = dataset.variables['latitude'][:]
        longitudes = dataset.variables['longitude'][:]
        
        # 转换时间为可读格式
        time_units = dataset.variables['valid_time'].units
        calendar = dataset.variables['valid_time'].calendar
        time_dates = nc.num2date(times, units=time_units, calendar=calendar)
        
        # 创建数据列表存储所有数据点
        data_list = []
        
        # 处理每个变量
        for var_name in dataset.variables:
            var = dataset.variables[var_name]
            
            # 跳过坐标变量
            if var_name in ['valid_time', 'latitude', 'longitude', 'number', 'expver']:
                continue
                
            print(f"  处理变量: {var_name}")
            
            # 获取变量数据
            var_data = var[:]
            
            # 处理缺失值
            if hasattr(var, '_FillValue'):
                var_data = np.ma.masked_equal(var_data, var._FillValue)
                var_data = var_data.filled(np.nan)
            
            # 单位转换：开尔文转摄氏度
            if hasattr(var, 'units'):
                units = var.units
                if units == 'K' and 'temperature' in var_name.lower():
                    print(f"    转换温度单位: K -> °C")
                    var_data = var_data - 273.15
                    units = '°C'
            
            # 重塑数据为2D表格格式
            # 假设变量维度为 (time, lat, lon) 或 (lat, lon)
            if var_data.ndim == 3:  # (time, lat, lon)
                for t_idx, time_val in enumerate(time_dates):
                    for lat_idx, lat_val in enumerate(latitudes):
                        for lon_idx, lon_val in enumerate(longitudes):
                            value = var_data[t_idx, lat_idx, lon_idx]
                            if not np.isnan(value):
                                data_list.append({
                                    'time': time_val,
                                    'latitude': lat_val,
                                    'longitude': lon_val,
                                    var_name: value,  # 直接使用变量名作为列名
                                    'units': units if hasattr(var, 'units') else 'unknown'
                                })
            elif var_data.ndim == 2:  # (lat, lon)
                for lat_idx, lat_val in enumerate(latitudes):
                    for lon_idx, lon_val in enumerate(longitudes):
                        value = var_data[lat_idx, lon_idx]
                        if not np.isnan(value):
                            data_list.append({
                                'time': 'constant',
                                'latitude': lat_val,
                                'longitude': lon_val,
                                var_name: value,  # 直接使用变量名作为列名
                                'units': units if hasattr(var, 'units') else 'unknown'
                            })
        
        # 关闭数据集
        dataset.close()
        
        if not data_list:
            print(f"警告: {os.path.basename(input_file)} 中没有有效数据")
            return False
        
        # 转换为DataFrame
        df = pd.DataFrame(data_list)
        
        # 创建输出文件名（保持原名，扩展名改为.csv）
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.csv")
        
        # 保存为CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        # 输出文件信息
        print(f"转换完成: {os.path.basename(output_file)}")
        print(f"  数据点数: {len(df)}")
        print(f"  时间范围: {df['time'].min()} 到 {df['time'].max()}")
        # 获取除了time, latitude, longitude, units之外的所有列名（即变量名）
        variable_columns = [col for col in df.columns if col not in ['time', 'latitude', 'longitude', 'units']]
        print(f"  变量: {variable_columns}")
        
        return True
        
    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {str(e)}")
        return False

def batch_convert_nc_to_csv(input_dir, output_dir, num_processes=None):
    """
    批量转换NC文件到CSV格式
    
    参数:
    input_dir: 输入文件夹路径
    output_dir: 输出文件夹路径  
    num_processes: 进程数，默认为CPU核心数
    """
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 查找所有NC文件
    nc_files = glob.glob(os.path.join(input_dir, "*.nc"))
    
    if not nc_files:
        print(f"在目录 {input_dir} 中未找到NC文件")
        return
    
    print(f"找到 {len(nc_files)} 个NC文件")
    
    # 准备多进程参数
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"使用 {num_processes} 个进程进行转换")
    
    # 创建参数列表
    args_list = [(nc_file, output_dir) for nc_file in nc_files]
    
    # 使用多进程池
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(convert_nc_to_csv, args_list)
    
    # 统计结果
    successful = sum(results)
    failed = len(nc_files) - successful
    
    print(f"\n转换完成!")
    print(f"成功: {successful} 个文件")
    print(f"失败: {failed} 个文件")
    print(f"输出目录: {output_dir}")

def main():
    """
    主函数 - 在这里设置输入输出路径
    """
    # 设置输入输出路径 - 请根据实际情况修改这些路径
    input_directory = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-NC\surface_pressure"  # 输入NC文件目录
    output_directory = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV\surface_pressure"  # 输出CSV文件目录
    
    # 进程数 (None表示自动使用CPU核心数)
    processes = None
    
    print("NC文件批量转换工具")
    print("=" * 50)
    print(f"输入目录: {input_directory}")
    print(f"输出目录: {output_directory}")
    print("=" * 50)
    
    # 执行批量转换
    batch_convert_nc_to_csv(input_directory, output_directory, processes)

if __name__ == "__main__":
    # 防止Windows多进程问题
    mp.freeze_support()
    main()