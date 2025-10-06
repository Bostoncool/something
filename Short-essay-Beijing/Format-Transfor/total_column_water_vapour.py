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
    def __init__(self, input_folder, output_folder, processes=None):
        """
        初始化转换器
        
        Parameters:
        -----------
        input_folder : str
            输入NC文件文件夹路径
        output_folder : str
            输出CSV文件文件夹路径
        processes : int, optional
            进程数，默认使用CPU核心数
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.processes = processes or cpu_count()
        
        # 创建输出文件夹
        os.makedirs(self.output_folder, exist_ok=True)
        
        # 关键变量映射（根据您的NC文件结构调整）
        self.key_variables = {
            'tcwv': 'tcwv',
            't2m': 'temperature_2m',  # 如果有温度变量
            'd2m': 'dewpoint_2m',     # 如果有露点温度
            'msl': 'mean_sea_level_pressure',
            'u10': 'wind_u_10m',
            'v10': 'wind_v_10m'
        }
    
    def kelvin_to_celsius(self, temp_kelvin):
        """将开尔文温度转换为摄氏度"""
        if temp_kelvin is None:
            return None
        return temp_kelvin - 273.15
    
    def process_time_dimension(self, time_data, time_units):
        """处理时间维度，转换为可读格式"""
        try:
            # 使用xarray处理时间
            if 'since' in time_units:
                time_data = xr.cftime_range(
                    start=pd.to_datetime(time_units.split('since ')[-1]),
                    periods=len(time_data),
                    freq='H'  # 根据实际情况调整频率
                )
            return time_data
        except:
            # 如果处理失败，返回原始数据
            return time_data
    
    def extract_key_variables(self, ds):
        """提取关键变量并处理"""
        data_dict = {}
        
        # 处理时间维度
        if 'valid_time' in ds.dims:
            time_data = ds['valid_time'].values
            time_units = ds['valid_time'].attrs.get('units', '')
            data_dict['time'] = self.process_time_dimension(time_data, time_units)
        
        # 处理经纬度
        if 'latitude' in ds.variables:
            data_dict['latitude'] = ds['latitude'].values
        if 'longitude' in ds.variables:
            data_dict['longitude'] = ds['longitude'].values
        
        # 提取关键变量数据
        for nc_var, csv_name in self.key_variables.items():
            if nc_var in ds.variables:
                var_data = ds[nc_var].values
                var_attrs = ds[nc_var].attrs
                
                # 处理缺失值
                fill_value = var_attrs.get('_FillValue', np.nan)
                if not np.isnan(fill_value):
                    var_data = np.where(var_data == fill_value, np.nan, var_data)
                
                # 温度单位转换
                units = var_attrs.get('units', '')
                if 'K' in units and 'kelvin' in units.lower():
                    var_data = self.kelvin_to_celsius(var_data)
                    data_dict[f'{csv_name}_celsius'] = var_data
                    data_dict[f'{csv_name}_unit'] = '°C'
                else:
                    data_dict[csv_name] = var_data
                    data_dict[f'{csv_name}_unit'] = units
        
        return data_dict
    
    def create_dataframe(self, data_dict):
        """从数据字典创建DataFrame"""
        # 获取维度信息
        time_len = len(data_dict.get('time', [1]))
        lat_len = len(data_dict.get('latitude', [1]))
        lon_len = len(data_dict.get('longitude', [1]))
        
        records = []
        
        # 展开多维数据
        for t_idx in range(time_len):
            time_val = data_dict.get('time', [None])[t_idx] if 'time' in data_dict else None
            
            for lat_idx in range(lat_len):
                lat_val = data_dict.get('latitude', [None])[lat_idx] if 'latitude' in data_dict else None
                
                for lon_idx in range(lon_len):
                    lon_val = data_dict.get('longitude', [None])[lon_idx] if 'longitude' in data_dict else None
                    
                    record = {
                        'time': time_val,
                        'latitude': lat_val,
                        'longitude': lon_val
                    }
                    
                    # 添加变量数据
                    for key, value in data_dict.items():
                        if key not in ['time', 'latitude', 'longitude'] and not key.endswith('_unit'):
                            if isinstance(value, np.ndarray) and value.ndim == 3:
                                # 三维数据 (time, lat, lon)
                                record[key] = value[t_idx, lat_idx, lon_idx] if value.size > 1 else value[0,0,0]
                            elif isinstance(value, np.ndarray) and value.ndim == 2:
                                # 二维数据 (lat, lon)
                                record[key] = value[lat_idx, lon_idx] if value.size > 1 else value[0,0]
                            elif isinstance(value, np.ndarray) and value.ndim == 1:
                                # 一维数据
                                record[key] = value[t_idx] if len(value) > 1 else value[0]
                            else:
                                record[key] = value
                    
                    records.append(record)
        
        return pd.DataFrame(records)
    
    def process_single_file(self, nc_file_path):
        """处理单个NC文件"""
        try:
            print(f"正在处理: {os.path.basename(nc_file_path)}")
            
            # 读取NC文件
            with xr.open_dataset(nc_file_path) as ds:
                # 提取关键变量
                data_dict = self.extract_key_variables(ds)
                
                # 创建DataFrame
                df = self.create_dataframe(data_dict)
                
                # 生成输出文件名
                base_name = os.path.splitext(os.path.basename(nc_file_path))[0]
                output_file = os.path.join(self.output_folder, f"{base_name}.csv")
                
                # 保存为CSV
                df.to_csv(output_file, index=False, encoding='utf-8')
                
                # 生成元数据文件
                self.create_metadata_file(ds, output_file)
                
            print(f"成功转换: {os.path.basename(nc_file_path)}")
            return True
            
        except Exception as e:
            print(f"处理文件 {nc_file_path} 时出错: {str(e)}")
            return False
    
    def create_metadata_file(self, dataset, csv_file_path):
        """创建元数据文件"""
        meta_file = csv_file_path.replace('.csv', '_metadata.txt')
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            f.write("NC文件转换元数据信息\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"原始文件维度: {dict(dataset.dims)}\n\n")
            
            f.write("变量信息:\n")
            for var_name in dataset.variables:
                var = dataset[var_name]
                f.write(f"- {var_name}:\n")
                f.write(f"  维度: {var.dims}\n")
                f.write(f"  形状: {var.shape}\n")
                if hasattr(var, 'units'):
                    f.write(f"  单位: {var.units}\n")
                if hasattr(var, 'long_name'):
                    f.write(f"  描述: {var.long_name}\n")
                f.write("\n")
            
            f.write("全局属性:\n")
            for attr_name in dataset.attrs:
                f.write(f"- {attr_name}: {dataset.attrs[attr_name]}\n")
    
    def batch_convert(self):
        """批量转换NC文件"""
        # 查找所有NC文件
        nc_pattern = os.path.join(self.input_folder, "*.nc")
        nc_files = glob.glob(nc_pattern)
        
        if not nc_files:
            print(f"在文件夹 {self.input_folder} 中未找到NC文件")
            return
        
        print(f"找到 {len(nc_files)} 个NC文件，使用 {self.processes} 个进程进行转换...")
        
        # 使用多进程处理
        with Pool(processes=self.processes) as pool:
            results = pool.map(self.process_single_file, nc_files)
        
        success_count = sum(results)
        print(f"\n转换完成! 成功: {success_count}/{len(nc_files)}")

def main():
    """主函数 - 在这里设置输入输出路径"""
    # 设置输入和输出文件夹路径
    input_folder = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-NC\total_column_water_vapour"  # 替换为您的NC文件文件夹路径
    output_folder = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV\total_column_water_vapour"  # 替换为您想要保存CSV的文件夹路径
    
    # 进程数 (None表示使用所有CPU核心)
    processes = None
    
    # 创建转换器并执行批量转换
    converter = NCtoCSVConverter(input_folder, output_folder, processes)
    converter.batch_convert()

if __name__ == "__main__":
    main()