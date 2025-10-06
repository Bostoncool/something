import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

class NetCDFToCSVConverter:
    def __init__(self, input_folder, output_folder, num_processes=None):
        """
        初始化转换器
        
        Parameters:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        num_processes: 进程数，默认为CPU核心数
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_processes = num_processes or cpu_count()
        
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
    def kelvin_to_celsius(self, temp_kelvin):
        """将开尔文温度转换为摄氏度"""
        return temp_kelvin - 273.15
    
    def process_time_dimension(self, time_data, time_units=None):
        """处理时间维度，转换为可读格式"""
        try:
            if hasattr(time_data, 'values'):
                time_values = time_data.values
            else:
                time_values = time_data
                
            # 如果时间是以秒为单位的数值，转换为datetime
            if time_units and 'seconds since' in time_units:
                time_coords = pd.to_datetime(time_values, unit='s', origin='unix')
            else:
                # 尝试直接转换为datetime
                time_coords = pd.to_datetime(time_values)
                
            return time_coords
        except:
            # 如果转换失败，返回原始值
            return time_values
    
    def process_single_file(self, nc_file):
        """处理单个NetCDF文件"""
        try:
            print(f"正在处理: {os.path.basename(nc_file)}")
            
            # 读取NetCDF文件
            ds = xr.open_dataset(nc_file)
            
            # 获取文件基本信息
            file_info = {
                'filename': os.path.basename(nc_file),
                'dimensions': dict(ds.dims),
                'variables': list(ds.variables.keys())
            }
            
            # 准备存储所有数据的列表
            all_data = []
            
            # 处理时间维度
            time_coords = None
            if 'valid_time' in ds.dims:
                time_var = ds['valid_time']
                time_units = getattr(time_var, 'units', None)
                time_coords = self.process_time_dimension(time_var, time_units)
            elif 'time' in ds.dims:
                time_var = ds['time']
                time_units = getattr(time_var, 'units', None)
                time_coords = self.process_time_dimension(time_var, time_units)
            
            # 获取经纬度坐标
            lat_coords = ds['latitude'].values if 'latitude' in ds.variables else None
            lon_coords = ds['longitude'].values if 'longitude' in ds.variables else None
            
            # 遍历所有数据变量
            for var_name in ds.variables:
                var_data = ds[var_name]
                
                # 跳过维度变量
                if var_name in ['valid_time', 'time', 'latitude', 'longitude', 'number', 'expver']:
                    continue
                
                print(f"  处理变量: {var_name}")
                
                # 检查变量维度结构
                dims = var_data.dims
                
                # 处理不同维度结构的数据
                if len(dims) == 3 and 'valid_time' in dims and 'latitude' in dims and 'longitude' in dims:
                    # 三维数据 (时间, 纬度, 经度)
                    self.process_3d_data(var_data, var_name, time_coords, lat_coords, lon_coords, all_data)
                elif len(dims) == 2 and 'latitude' in dims and 'longitude' in dims:
                    # 二维空间数据
                    self.process_2d_spatial_data(var_data, var_name, lat_coords, lon_coords, all_data)
                elif len(dims) == 1:
                    # 一维数据
                    self.process_1d_data(var_data, var_name, all_data)
                else:
                    print(f"  跳过变量 {var_name}: 不支持的维度结构 {dims}")
            
            # 关闭数据集
            ds.close()
            
            # 转换为DataFrame
            if all_data:
                df = pd.DataFrame(all_data)
                
                # 处理缺失值
                df = self.handle_missing_values(df)
                
                # 转换为宽格式：将variable列的值作为列名，value列作为数据
                if 'variable' in df.columns and 'value' in df.columns:
                    # 创建透视表，将variable列的值作为列名
                    df_wide = df.pivot_table(
                        index=['time', 'latitude', 'longitude'], 
                        columns='variable', 
                        values='value',
                        aggfunc='first'  # 如果有重复值，取第一个
                    ).reset_index()
                    
                    # 添加单位信息到列名
                    units_info = df.groupby('variable')['units'].first().to_dict()
                    for var_name, units in units_info.items():
                        if var_name in df_wide.columns and units != 'unknown':
                            df_wide = df_wide.rename(columns={var_name: f"{var_name} ({units})"})
                    
                    df = df_wide
                
                # 生成输出文件名
                output_filename = os.path.splitext(os.path.basename(nc_file))[0] + '.csv'
                output_path = os.path.join(self.output_folder, output_filename)
                
                # 保存为CSV
                df.to_csv(output_path, index=False, encoding='utf-8')
                
                # 生成元数据文件
                self.create_metadata_file(output_path, file_info, ds.attrs)
                
                print(f"✓ 完成: {output_filename} (共 {len(df)} 行)")
                return True
            else:
                print(f"⚠ 警告: {os.path.basename(nc_file)} 没有可转换的数据")
                return False
                
        except Exception as e:
            print(f"✗ 处理文件 {nc_file} 时出错: {str(e)}")
            return False
    
    def process_3d_data(self, var_data, var_name, time_coords, lat_coords, lon_coords, all_data):
        """处理三维数据 (时间, 纬度, 经度)"""
        data_values = var_data.values
        
        # 获取变量属性
        units = getattr(var_data, 'units', 'unknown')
        long_name = getattr(var_data, 'long_name', var_name)
        
        # 检查是否需要温度转换
        if units.lower() in ['k', 'kelvin']:
            print(f"    将 {var_name} 从开尔文转换为摄氏度")
            data_values = self.kelvin_to_celsius(data_values)
            units = '°C'
        
        # 处理缺失值
        fill_value = getattr(var_data, '_FillValue', np.nan)
        if not np.isnan(fill_value):
            data_values = np.where(data_values == fill_value, np.nan, data_values)
        
        # 展平数据
        for t_idx in range(len(time_coords)):
            for lat_idx in range(len(lat_coords)):
                for lon_idx in range(len(lon_coords)):
                    value = data_values[t_idx, lat_idx, lon_idx]
                    
                    # 跳过NaN值
                    if np.isnan(value):
                        continue
                    
                    record = {
                        'variable': var_name,
                        'long_name': long_name,
                        'units': units,
                        'time': time_coords[t_idx],
                        'latitude': lat_coords[lat_idx],
                        'longitude': lon_coords[lon_idx],
                        'value': value
                    }
                    all_data.append(record)
    
    def process_2d_spatial_data(self, var_data, var_name, lat_coords, lon_coords, all_data):
        """处理二维空间数据"""
        data_values = var_data.values
        
        # 获取变量属性
        units = getattr(var_data, 'units', 'unknown')
        long_name = getattr(var_data, 'long_name', var_name)
        
        # 温度转换
        if units.lower() in ['k', 'kelvin']:
            print(f"    将 {var_name} 从开尔文转换为摄氏度")
            data_values = self.kelvin_to_celsius(data_values)
            units = '°C'
        
        # 处理缺失值
        fill_value = getattr(var_data, '_FillValue', np.nan)
        if not np.isnan(fill_value):
            data_values = np.where(data_values == fill_value, np.nan, data_values)
        
        # 展平数据
        for lat_idx in range(len(lat_coords)):
            for lon_idx in range(len(lon_coords)):
                value = data_values[lat_idx, lon_idx]
                
                if np.isnan(value):
                    continue
                
                record = {
                    'variable': var_name,
                    'long_name': long_name,
                    'units': units,
                    'time': 'N/A',
                    'latitude': lat_coords[lat_idx],
                    'longitude': lon_coords[lon_idx],
                    'value': value
                }
                all_data.append(record)
    
    def process_1d_data(self, var_data, var_name, all_data):
        """处理一维数据"""
        data_values = var_data.values
        
        # 获取变量属性
        units = getattr(var_data, 'units', 'unknown')
        long_name = getattr(var_data, 'long_name', var_name)
        
        # 温度转换
        if units.lower() in ['k', 'kelvin']:
            print(f"    将 {var_name} 从开尔文转换为摄氏度")
            data_values = self.kelvin_to_celsius(data_values)
            units = '°C'
        
        # 处理缺失值
        fill_value = getattr(var_data, '_FillValue', np.nan)
        if not np.isnan(fill_value):
            data_values = np.where(data_values == fill_value, np.nan, data_values)
        
        # 处理数据
        for idx, value in enumerate(data_values):
            if np.isnan(value):
                continue
            
            record = {
                'variable': var_name,
                'long_name': long_name,
                'units': units,
                'time': f'index_{idx}',
                'latitude': 'N/A',
                'longitude': 'N/A',
                'value': value
            }
            all_data.append(record)
    
    def handle_missing_values(self, df):
        """处理缺失值"""
        # 移除所有值都为NaN的行
        df_clean = df.dropna(subset=['value'])
        return df_clean
    
    def create_metadata_file(self, csv_path, file_info, global_attrs):
        """创建元数据文件"""
        meta_filename = os.path.splitext(csv_path)[0] + '_metadata.txt'
        
        with open(meta_filename, 'w', encoding='utf-8') as f:
            f.write("NetCDF to CSV 转换元数据\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("文件信息:\n")
            f.write(f"  原始文件: {file_info['filename']}\n")
            f.write(f"  维度: {file_info['dimensions']}\n")
            f.write(f"  变量: {file_info['variables']}\n\n")
            
            f.write("全局属性:\n")
            for key, value in global_attrs.items():
                f.write(f"  {key}: {value}\n")
    
    def batch_convert(self):
        """批量转换所有NetCDF文件"""
        # 查找所有NetCDF文件
        pattern = os.path.join(self.input_folder, "*.nc")
        nc_files = glob.glob(pattern)
        
        if not nc_files:
            print(f"在文件夹 {self.input_folder} 中未找到NetCDF文件")
            return
        
        print(f"找到 {len(nc_files)} 个NetCDF文件")
        print(f"使用 {self.num_processes} 个进程进行转换")
        print("-" * 50)
        
        # 使用多进程处理
        with Pool(processes=self.num_processes) as pool:
            results = pool.map(self.process_single_file, nc_files)
        
        # 统计结果
        successful = sum(results)
        failed = len(nc_files) - successful
        
        print("-" * 50)
        print(f"转换完成! 成功: {successful}, 失败: {failed}")
        print(f"输出文件夹: {self.output_folder}")

def main():
    """主函数"""
    # 在这里设置您的输入和输出文件夹路径
    input_folder = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-NC\boundary_layer_height"  # 修改为您的输入文件夹路径
    output_folder = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV\boundary_layer_height"  # 修改为您的输出文件夹路径
    
    # 设置进程数 (None表示使用所有CPU核心)
    num_processes = None
    
    # 创建转换器并执行批量转换
    converter = NetCDFToCSVConverter(input_folder, output_folder, num_processes)
    converter.batch_convert()

if __name__ == "__main__":
    main()