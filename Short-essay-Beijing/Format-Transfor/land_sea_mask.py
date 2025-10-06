import os
import glob
import multiprocessing as mp
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
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
        self.num_processes = num_processes or mp.cpu_count()
        
        # 创建输出文件夹
        os.makedirs(self.output_folder, exist_ok=True)
    
    def find_netcdf_files(self):
        """查找所有NetCDF文件"""
        patterns = ['*.nc', '*.nc4', '*.netcdf']
        nc_files = []
        for pattern in patterns:
            nc_files.extend(glob.glob(os.path.join(self.input_folder, pattern)))
        return sorted(nc_files)
    
    def process_file(self, nc_file_path):
        """
        处理单个NetCDF文件
        
        Parameters:
        nc_file_path: NetCDF文件路径
        """
        try:
            print(f"正在处理: {os.path.basename(nc_file_path)}")
            
            # 读取NetCDF文件
            ds = xr.open_dataset(nc_file_path)
            
            # 获取输出文件名
            base_name = os.path.splitext(os.path.basename(nc_file_path))[0]
            output_file = os.path.join(self.output_folder, f"{base_name}.csv")
            
            # 处理数据
            processed_dfs = self.process_dataset(ds)
            
            # 保存为CSV
            if processed_dfs:
                # 合并所有数据框
                final_df = pd.concat(processed_dfs, ignore_index=True)
                
                # 保存CSV文件
                final_df.to_csv(output_file, index=False, encoding='utf-8')
                print(f"成功转换: {os.path.basename(output_file)}")
            else:
                print(f"警告: 文件 {nc_file_path} 没有可处理的数据变量")
                
            # 关闭数据集
            ds.close()
            
        except Exception as e:
            print(f"处理文件 {nc_file_path} 时出错: {str(e)}")
    
    def process_dataset(self, ds):
        """
        处理数据集，返回DataFrame列表
        
        Parameters:
        ds: xarray Dataset对象
        """
        dataframes = []
        
        # 检查维度结构
        dims_to_check = ['valid_time', 'time', 'latitude', 'lat', 'longitude', 'lon']
        available_dims = {dim: dim in ds.dims for dim in dims_to_check}
        
        print(f"可用维度: {available_dims}")
        
        # 识别时间、纬度、经度维度名称
        time_dim = self._identify_dimension(ds, ['valid_time', 'time'])
        lat_dim = self._identify_dimension(ds, ['latitude', 'lat'])
        lon_dim = self._identify_dimension(ds, ['longitude', 'lon'])
        
        if not all([time_dim, lat_dim, lon_dim]):
            print("警告: 缺少必要的时间、纬度或经度维度")
            return dataframes
        
        # 处理每个数据变量
        for var_name in ds.data_vars:
            var = ds[var_name]
            
            # 检查变量维度是否包含时间、纬度、经度
            if not all(dim in var.dims for dim in [time_dim, lat_dim, lon_dim]):
                continue
                
            print(f"处理变量: {var_name}")
            
            # 转换为DataFrame
            df = var.to_dataframe().reset_index()
            
            # 处理缺失值
            df = self._handle_missing_values(df, var.attrs)
            
            # 转换温度单位（开尔文到摄氏度）
            df = self._convert_temperature_units(df, var_name, var.attrs)
            
            # 转换时间格式
            df = self._convert_time_format(df, time_dim)
            
            # 添加变量信息
            df['variable_name'] = var_name
            if 'units' in var.attrs:
                df['variable_units'] = var.attrs['units']
            if 'long_name' in var.attrs:
                df['variable_description'] = var.attrs['long_name']
            
            dataframes.append(df)
        
        return dataframes
    
    def _identify_dimension(self, ds, possible_names):
        """识别维度名称"""
        for name in possible_names:
            if name in ds.dims:
                return name
        return None
    
    def _handle_missing_values(self, df, attrs):
        """处理缺失值"""
        fill_value = attrs.get('_FillValue', attrs.get('missing_value', np.nan))
        
        if not np.isnan(fill_value):
            # 替换填充值为NaN
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col not in ['latitude', 'longitude']:  # 避免替换坐标值
                    df[col] = df[col].replace(fill_value, np.nan)
        
        return df
    
    def _convert_temperature_units(self, df, var_name, attrs):
        """转换温度单位（开尔文到摄氏度）"""
        units = attrs.get('units', '').lower()
        var_name_lower = var_name.lower()
        
        # 检查是否是温度变量
        is_temperature = any(keyword in var_name_lower for keyword in 
                           ['temp', 'temperature', 't2m', 't_', 'air_temp'])
        
        if is_temperature and units in ['k', 'kelvin']:
            print(f"转换温度单位: {var_name} 从开尔文到摄氏度")
            # 找到数据列（排除坐标列）
            data_columns = [col for col in df.columns if col not in 
                          ['valid_time', 'time', 'latitude', 'lat', 'longitude', 'lon', 
                           'variable_name', 'variable_units', 'variable_description']]
            
            for col in data_columns:
                if df[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
                    df[col] = df[col] - 273.15
                    # 更新单位信息
                    if 'variable_units' in df.columns:
                        df['variable_units'] = '°C'
        
        return df
    
    def _convert_time_format(self, df, time_dim):
        """转换时间为可读格式"""
        if time_dim in df.columns:
            # 尝试转换为datetime格式
            try:
                # 处理UNIX时间戳（秒 since 1970-01-01）
                if df[time_dim].dtype in [np.int32, np.int64, np.float32, np.float64]:
                    if df[time_dim].max() > 1e9:  # 可能是UNIX时间戳
                        df['datetime'] = pd.to_datetime(df[time_dim], unit='s')
                    else:
                        df['datetime'] = pd.to_datetime(df[time_dim])
                else:
                    df['datetime'] = pd.to_datetime(df[time_dim])
                
                # 添加日期时间组件
                df['year'] = df['datetime'].dt.year
                df['month'] = df['datetime'].dt.month
                df['day'] = df['datetime'].dt.day
                df['hour'] = df['datetime'].dt.hour
                df['minute'] = df['datetime'].dt.minute
                
            except Exception as e:
                print(f"时间转换警告: {str(e)}")
                df['datetime'] = df[time_dim]
        
        return df
    
    def process_all_files(self):
        """批量处理所有文件"""
        nc_files = self.find_netcdf_files()
        
        if not nc_files:
            print(f"在文件夹 {self.input_folder} 中未找到NetCDF文件")
            return
        
        print(f"找到 {len(nc_files)} 个NetCDF文件")
        print(f"使用 {self.num_processes} 个进程进行转换")
        
        # 使用多进程处理
        with mp.Pool(processes=self.num_processes) as pool:
            pool.map(self.process_file, nc_files)
        
        print("所有文件处理完成！")

def main():
    """主函数"""
    # 配置输入输出路径
    input_folder = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-NC\land_sea_mask"  # 修改为您的输入文件夹路径
    output_folder = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV\land_sea_mask"  # 修改为您的输出文件夹路径
    
    # 创建转换器实例
    converter = NetCDFToCSVConverter(
        input_folder=input_folder,
        output_folder=output_folder,
        num_processes=4  # 可根据您的CPU核心数调整
    )
    
    # 执行转换
    converter.process_all_files()

if __name__ == "__main__":
    main()