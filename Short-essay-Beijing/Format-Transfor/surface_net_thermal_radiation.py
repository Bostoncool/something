import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
from multiprocessing import Pool, cpu_count
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
        self.num_processes = num_processes or cpu_count()
        
        # 创建输出文件夹
        os.makedirs(self.output_folder, exist_ok=True)
        
        # 关键变量映射（可根据需要修改）
        self.key_variables = {
            'str': 'surface_net_upward_longwave_flux',
            't2m': '2m_temperature', 
            'tp': 'total_precipitation',
            'u10': '10m_u_wind_component',
            'v10': '10m_v_wind_component',
            'msl': 'mean_sea_level_pressure'
        }
    
    def find_netcdf_files(self):
        """查找所有NetCDF文件"""
        patterns = ['*.nc', '*.nc4', '*.netcdf']
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(self.input_folder, pattern)))
        return sorted(files)
    
    def convert_temperature_units(self, data_array, var_name):
        """
        将开尔文温度转换为摄氏度
        
        Parameters:
        data_array: 数据数组
        var_name: 变量名
        
        Returns:
        转换后的数据数组
        """
        # 检查是否为温度变量
        temp_indicators = ['temperature', 'kelvin', 'K']
        if (any(indicator in str(data_array.attrs.get('units', '')).lower() for indicator in temp_indicators)
               or any(indicator in var_name.lower() for indicator in temp_indicators)):
            
            if data_array.attrs.get('units') in ['K', 'kelvin']:
                print(f"  转换温度单位: {var_name} 从开尔文到摄氏度")
                # 开尔文转摄氏度: °C = K - 273.15
                converted_data = data_array - 273.15
                converted_data.attrs['units'] = '°C'
                converted_data.attrs['long_name'] = data_array.attrs.get('long_name', '') + ' (Celsius)'
                return converted_data
        
        return data_array
    
    def process_single_file(self, nc_file_path):
        """
        处理单个NetCDF文件
        
        Parameters:
        nc_file_path: NetCDF文件路径
        
        Returns:
        成功返回True，失败返回False
        """
        try:
            file_name = os.path.basename(nc_file_path)
            output_file = os.path.join(self.output_folder, 
                                     file_name.replace('.nc', '.csv').replace('.nc4', '.csv'))
            
            print(f"处理文件: {file_name}")
            
            # 读取NetCDF文件
            with xr.open_dataset(nc_file_path) as ds:
                print(f"  文件维度: {dict(ds.dims)}")
                print(f"  文件变量: {list(ds.variables.keys())}")
                
                # 确定关键变量
                available_vars = list(ds.variables.keys())
                key_vars_to_process = []
                
                for var in available_vars:
                    # 跳过坐标变量
                    if var in ['time', 'latitude', 'longitude', 'lat', 'lon', 'valid_time']:
                        continue
                    
                    # 检查是否在关键变量列表中或有数据维度
                    if (var in self.key_variables or 
                        hasattr(ds[var], 'dims') and len(ds[var].dims) >= 2):
                        key_vars_to_process.append(var)
                
                print(f"  将处理的关键变量: {key_vars_to_process}")
                
                if not key_vars_to_process:
                    print(f"  警告: 在文件 {file_name} 中未找到关键变量")
                    return False
                
                # 处理每个关键变量
                all_dfs = []
                
                for var_name in key_vars_to_process:
                    print(f"  处理变量: {var_name}")
                    
                    try:
                        # 获取数据
                        data_array = ds[var_name]
                        
                        # 处理缺失值
                        if '_FillValue' in data_array.attrs:
                            fill_value = data_array.attrs['_FillValue']
                            data_array = data_array.where(data_array != fill_value)
                        
                        # 转换温度单位
                        data_array = self.convert_temperature_units(data_array, var_name)
                        
                        # 转换为DataFrame
                        df = data_array.to_dataframe(name=var_name).reset_index()
                        
                        # 处理时间维度
                        time_columns = [col for col in df.columns if 'time' in col.lower()]
                        for time_col in time_columns:
                            if np.issubdtype(df[time_col].dtype, np.datetime64):
                                df[time_col] = df[time_col].dt.strftime('%Y-%m-%d %H:%M:%S')
                                print(f"    已转换时间格式: {time_col}")
                        
                        # 不添加单位信息到列名，保持原始变量名
                        # units = data_array.attrs.get('units', '')
                        # if units:
                        #     df = df.rename(columns={var_name: f"{var_name} ({units})"})
                        
                        all_dfs.append(df)
                        
                    except Exception as e:
                        print(f"    处理变量 {var_name} 时出错: {str(e)}")
                        continue
                
                if not all_dfs:
                    print(f"  错误: 无法处理文件 {file_name} 中的任何变量")
                    return False
                
                # 合并所有DataFrame
                if len(all_dfs) == 1:
                    final_df = all_dfs[0]
                else:
                    # 基于共同的坐标列合并
                    common_columns = []
                    for col in all_dfs[0].columns:
                        if all(col in df.columns for df in all_dfs):
                            common_columns.append(col)
                    
                    final_df = all_dfs[0]
                    for df in all_dfs[1:]:
                        final_df = final_df.merge(df, on=common_columns)
                
                # 控制输出文件大小 - 采样策略
                original_rows = len(final_df)
                if original_rows > 1000000:  # 如果超过100万行，进行采样
                    sample_fraction = 1000000 / original_rows
                    final_df = final_df.sample(frac=sample_fraction, random_state=42)
                    print(f"  数据采样: {original_rows} -> {len(final_df)} 行")
                
                # 保存为CSV
                final_df.to_csv(output_file, index=False, encoding='utf-8')
                print(f"  成功保存: {output_file} (共 {len(final_df)} 行)")
                
                # 生成元数据文件
                self.generate_metadata_file(ds, output_file.replace('.csv', '_metadata.txt'))
                
                return True
                
        except Exception as e:
            print(f"处理文件 {nc_file_path} 时出错: {str(e)}")
            return False
    
    def generate_metadata_file(self, dataset, metadata_file_path):
        """生成元数据文件"""
        try:
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                f.write("NetCDF 到 CSV 转换元数据\n")
                f.write("=" * 50 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("维度信息:\n")
                for dim, size in dataset.dims.items():
                    f.write(f"  {dim}: {size}\n")
                
                f.write("\n变量信息:\n")
                for var_name in dataset.variables:
                    var = dataset.variables[var_name]
                    f.write(f"  {var_name}:\n")
                    f.write(f"    维度: {var.dims}\n")
                    f.write(f"    形状: {var.shape}\n")
                    if hasattr(var, 'units'):
                        f.write(f"    单位: {var.units}\n")
                    if hasattr(var, 'long_name'):
                        f.write(f"    描述: {var.long_name}\n")
                    f.write("\n")
                
                f.write("全局属性:\n")
                for attr_name in dataset.attrs:
                    f.write(f"  {attr_name}: {dataset.attrs[attr_name]}\n")
                    
            print(f"  元数据文件已生成: {metadata_file_path}")
            
        except Exception as e:
            print(f"生成元数据文件时出错: {str(e)}")
    
    def batch_convert(self):
        """批量转换所有NetCDF文件"""
        nc_files = self.find_netcdf_files()
        
        if not nc_files:
            print(f"在文件夹 {self.input_folder} 中未找到NetCDF文件")
            return
        
        print(f"找到 {len(nc_files)} 个NetCDF文件")
        print(f"使用 {self.num_processes} 个进程进行并行处理")
        
        # 使用多进程处理
        with Pool(processes=self.num_processes) as pool:
            results = pool.map(self.process_single_file, nc_files)
        
        # 统计结果
        successful = sum(results)
        failed = len(results) - successful
        
        print(f"\n转换完成!")
        print(f"成功: {successful} 个文件")
        print(f"失败: {failed} 个文件")
        print(f"输出文件夹: {self.output_folder}")

def main():
    """主函数"""
    # 在这里指定输入和输出文件夹路径
    INPUT_FOLDER = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-NC\surface_net_thermal_radiation"  # 替换为您的输入文件夹路径
    OUTPUT_FOLDER = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV\surface_net_thermal_radiation"  # 替换为您的输出文件夹路径
    
    # 创建转换器实例
    converter = NetCDFToCSVConverter(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        num_processes=4  # 可根据需要调整进程数
    )
    
    # 开始批量转换
    converter.batch_convert()

if __name__ == "__main__":
    main()