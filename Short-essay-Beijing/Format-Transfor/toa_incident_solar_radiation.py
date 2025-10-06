import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NCtoCSVConverter:
    def __init__(self, input_folder, output_folder):
        """
        初始化转换器
        
        Args:
            input_folder: 输入NC文件文件夹路径
            output_folder: 输出CSV文件文件夹路径
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.required_variables = ['valid_time', 'latitude', 'longitude', 'tisr']
        
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
    
    def kelvin_to_celsius(self, temperature_k):
        """
        将开尔文温度转换为摄氏度
        
        Args:
            temperature_k: 开尔文温度
            
        Returns:
            temperature_c: 摄氏度温度
        """
        if temperature_k is None:
            return None
        return temperature_k - 273.15
    
    def process_time_dimension(self, time_data, time_units):
        """
        处理时间维度，转换为可读格式
        
        Args:
            time_data: 时间数据
            time_units: 时间单位信息
            
        Returns:
            readable_time: 可读时间格式
        """
        try:
            # 转换秒数为可读时间格式
            if 'seconds since 1970-01-01' in time_units:
                base_time = datetime(1970, 1, 1)
                readable_times = []
                for seconds in time_data:
                    delta = pd.Timedelta(seconds=float(seconds))
                    readable_time = base_time + delta
                    readable_times.append(readable_time.strftime('%Y-%m-%d %H:%M:%S'))
                return readable_times
            else:
                # 其他时间格式处理
                return [str(t) for t in time_data]
        except Exception as e:
            print(f"时间转换错误: {e}")
            return [str(t) for t in time_data]
    
    def handle_missing_values(self, data, fill_value):
        """
        处理缺失值
        
        Args:
            data: 原始数据
            fill_value: 缺失值标记
            
        Returns:
            cleaned_data: 清理后的数据
        """
        if fill_value is not None and not np.isnan(fill_value):
            data = np.where(data == fill_value, np.nan, data)
        return data
    
    def process_single_file(self, nc_file_path):
        """
        处理单个NC文件
        
        Args:
            nc_file_path: NC文件路径
            
        Returns:
            success: 处理是否成功
        """
        try:
            print(f"正在处理文件: {os.path.basename(nc_file_path)}")
            
            # 读取NC文件
            with xr.open_dataset(nc_file_path) as ds:
                # 检查必要的变量是否存在
                missing_vars = [var for var in self.required_variables if var not in ds.variables]
                if missing_vars:
                    print(f"警告: 文件 {nc_file_path} 缺少变量: {missing_vars}")
                    return False
                
                # 提取数据
                valid_time = ds['valid_time'].values
                latitude = ds['latitude'].values
                longitude = ds['longitude'].values
                tisr_data = ds['tisr'].values
                
                # 获取属性信息
                time_units = ds['valid_time'].attrs.get('units', '')
                tisr_fill_value = ds['tisr'].attrs.get('_FillValue', np.nan)
                tisr_units = ds['tisr'].attrs.get('units', '')
                
                # 处理缺失值
                tisr_data = self.handle_missing_values(tisr_data, tisr_fill_value)
                
                # 转换时间为可读格式
                readable_times = self.process_time_dimension(valid_time, time_units)
                
                # 创建数据列表
                data_records = []
                
                # 遍历所有维度组合
                for time_idx, time_val in enumerate(readable_times):
                    for lat_idx, lat_val in enumerate(latitude):
                        for lon_idx, lon_val in enumerate(longitude):
                            record = {
                                'valid_time': time_val,
                                'latitude': lat_val,
                                'longitude': lon_val,
                                'tisr': tisr_data[time_idx, lat_idx, lon_idx],
                                'tisr_units': tisr_units
                            }
                            data_records.append(record)
                
                # 创建DataFrame
                df = pd.DataFrame(data_records)
                
                # 生成输出文件名
                base_name = os.path.splitext(os.path.basename(nc_file_path))[0]
                output_file = os.path.join(self.output_folder, f"{base_name}.csv")
                
                # 保存为CSV
                df.to_csv(output_file, index=False, encoding='utf-8')
                
                # 添加单位信息到单独的文件
                units_info = {
                    'variable': ['valid_time', 'latitude', 'longitude', 'tisr'],
                    'units': ['datetime', 'degrees_north', 'degrees_east', tisr_units],
                    'description': [
                        '验证时间',
                        '纬度',
                        '经度', 
                        'TOA入射短波（太阳）辐射'
                    ]
                }
                units_df = pd.DataFrame(units_info)
                units_file = os.path.join(self.output_folder, f"{base_name}_units_info.csv")
                units_df.to_csv(units_file, index=False, encoding='utf-8')
                
                print(f"成功转换: {os.path.basename(nc_file_path)} -> {base_name}.csv")
                print(f"数据量: {len(df)} 行")
                print(f"时间范围: {readable_times[0]} 到 {readable_times[-1]}")
                print(f"空间范围: 纬度 {latitude.min():.2f}~{latitude.max():.2f}, "
                      f"经度 {longitude.min():.2f}~{longitude.max():.2f}")
                print("-" * 50)
                
                return True
                
        except Exception as e:
            print(f"处理文件 {nc_file_path} 时出错: {e}")
            return False
    
    def batch_convert(self, num_processes=None):
        """
        批量转换NC文件到CSV格式
        
        Args:
            num_processes: 进程数，默认为CPU核心数
        """
        if num_processes is None:
            num_processes = cpu_count()
        
        # 获取所有NC文件
        nc_files = glob.glob(os.path.join(self.input_folder, "*.nc"))
        if not nc_files:
            print(f"在文件夹 {self.input_folder} 中未找到NC文件")
            return
        
        print(f"找到 {len(nc_files)} 个NC文件")
        print(f"使用 {num_processes} 个进程进行转换")
        print("开始转换...")
        
        # 使用多进程处理
        with Pool(processes=num_processes) as pool:
            results = pool.map(self.process_single_file, nc_files)
        
        # 统计结果
        successful = sum(results)
        failed = len(nc_files) - successful
        
        print("\n转换完成!")
        print(f"成功: {successful} 个文件")
        print(f"失败: {failed} 个文件")
        print(f"输出文件夹: {self.output_folder}")

def main():
    """
    主函数 - 在这里设置输入和输出文件夹路径
    """
    # 设置输入和输出文件夹路径
    input_folder = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-NC\toa_incident_solar_radiation"  # 替换为您的NC文件输入文件夹路径
    output_folder = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV\toa_incident_solar_radiation"  # 替换为您的CSV文件输出文件夹路径
    
    # 创建转换器实例
    converter = NCtoCSVConverter(input_folder, output_folder)
    
    # 开始批量转换
    converter.batch_convert(num_processes=4)  # 可以调整进程数

if __name__ == "__main__":
    main()