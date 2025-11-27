import os
import sys
import glob
import pandas as pd
import netCDF4 as nc
import numpy as np
from multiprocessing import Pool, cpu_count

# 设置控制台编码为UTF-8
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

def process_file(args):
    """
    处理单个NetCDF文件的函数
    参数: (nc_file, output_dir) 元组
    """
    nc_file, output_dir = args
    try:
        # 读取NetCDF
        with nc.Dataset(nc_file, 'r') as ds:
            # 获取数据
            times = ds.variables['valid_time'][:]
            lats = ds.variables['latitude'][:]
            lons = ds.variables['longitude'][:]
            
            # 转换时间
            time_units = ds.variables['valid_time'].units
            time_dates = nc.num2date(times, units=time_units)
            
            # 获取主要数据变量
            data_var = None
            var_name = None
            for var in ds.variables:
                if var not in ['valid_time', 'latitude', 'longitude', 'number', 'expver']:
                    data_var = ds.variables[var]
                    var_name = var
                    break
            
            if data_var is None:
                print(f"未找到数据变量: {nc_file}")
                return False
            
            data = data_var[:]
            
            # 处理缺失值
            if hasattr(data_var, '_FillValue'):
                data = np.ma.masked_equal(data, data_var._FillValue)
            
            # 创建数据记录
            records = []
            for t_idx, time_val in enumerate(time_dates):
                for lat_idx, lat_val in enumerate(lats):
                    for lon_idx, lon_val in enumerate(lons):
                        value = data[t_idx, lat_idx, lon_idx]
                        records.append({
                            'time': time_val,
                            'latitude': lat_val,
                            'longitude': lon_val,
                            var_name: float(value) if not (hasattr(value, 'mask') and value.mask) else np.nan
                        })
            
            # 保存CSV
            df = pd.DataFrame(records)
            output_file = os.path.join(output_dir, 
                                     os.path.basename(nc_file).replace('.nc', '.csv'))
            df.to_csv(output_file, index=False)
            print(f"转换完成: {os.path.basename(output_file)}")
            return True
            
    except Exception as e:
        print(f"错误处理 {nc_file}: {e}")
        return False

def simple_batch_convert(input_dir, output_dir, processes=None):
    """
    简化版的批量转换函数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有NetCDF文件
    nc_files = glob.glob(os.path.join(input_dir, "*.nc"))
    
    if processes is None:
        processes = min(cpu_count(), len(nc_files))
    
    # 准备参数列表 (nc_file, output_dir)
    file_args = [(nc_file, output_dir) for nc_file in nc_files]
    
    # 并行处理
    with Pool(processes=processes) as pool:
        results = pool.map(process_file, file_args)
    
    print(f"处理完成! 成功: {sum(results)}/{len(nc_files)}")

# 使用示例
if __name__ == "__main__":
    # 在这里指定您的输入和输出路径
    INPUT_DIRECTORY = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-NC\mean_total_precipitation_rate"
    OUTPUT_DIRECTORY = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV\mean_total_precipitation_rate"
    
    simple_batch_convert(INPUT_DIRECTORY, OUTPUT_DIRECTORY)