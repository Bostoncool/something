import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

class NCToCSVConverter:
    def __init__(self, input_folder, output_folder, num_processes=None):
        """
        初始化转换器
        
        Parameters:
        - input_folder: 输入NC文件文件夹路径
        - output_folder: 输出CSV文件文件夹路径
        - num_processes: 进程数，默认为CPU核心数
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_processes = num_processes or cpu_count()
        
        # 创建输出文件夹
        os.makedirs(self.output_folder, exist_ok=True)
        
    def process_single_file(self, nc_file_path):
        """
        处理单个NC文件
        
        Parameters:
        - nc_file_path: NC文件路径
        
        Returns:
        - 成功返回True，失败返回False
        """
        try:
            print(f"正在处理文件: {os.path.basename(nc_file_path)}")
            
            # 读取NC文件
            with xr.open_dataset(nc_file_path, engine='netcdf4') as ds:
                # 获取文件名（不含扩展名）
                base_name = os.path.splitext(os.path.basename(nc_file_path))[0]
                output_csv_path = os.path.join(self.output_folder, f"{base_name}.csv")
                
                # 处理数据 - 现在只返回一个主要的DataFrame
                main_df = self.process_dataset(ds)
                
                if main_df is not None:
                    # 保存为CSV
                    main_df.to_csv(output_csv_path, index=False, encoding='utf-8')
                    print(f"成功转换: {os.path.basename(nc_file_path)} -> {base_name}.csv")
                    return True
                else:
                    print(f"无法处理文件: {os.path.basename(nc_file_path)}")
                    return False
                
        except Exception as e:
            print(f"处理文件 {nc_file_path} 时出错: {str(e)}")
            return False
    
    def process_dataset(self, ds):
        """
        处理数据集，将所有变量合并到一个DataFrame中
        
        Parameters:
        - ds: xarray Dataset对象
        
        Returns:
        - 合并后的DataFrame
        """
        try:
            # 获取所有数据变量
            data_vars = list(ds.data_vars.keys())
            print(f"找到变量: {data_vars}")
            
            # 将整个数据集转换为DataFrame
            df = ds.to_dataframe().reset_index()
            
            # 处理缺失值
            df = self.handle_missing_values_global(df, ds)
            
            # 转换时间格式
            df = self.convert_time_format(df)
            
            # 温度单位转换
            df = self.convert_temperature_units_global(df, ds)
            
            # 添加单位信息到列名
            df = self.add_unit_info_global(df, ds)
            
            # 数据采样（如果数据量过大）
            df = self.sample_data_if_large(df)
            
            print(f"最终DataFrame形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"处理数据集时出错: {str(e)}")
            return None
    
    def handle_missing_values_global(self, df, ds):
        """
        全局处理缺失值
        
        Parameters:
        - df: DataFrame
        - ds: xarray Dataset对象
        
        Returns:
        - 处理缺失值后的DataFrame
        """
        for var_name in ds.data_vars:
            var_data = ds[var_name]
            
            # 检查是否有缺失值属性
            fill_value = var_data.attrs.get('_FillValue', None)
            missing_value = var_data.attrs.get('missing_value', None)
            
            # 替换缺失值
            if fill_value is not None and var_name in df.columns:
                df[var_name] = df[var_name].replace(fill_value, np.nan)
            
            if missing_value is not None and var_name in df.columns:
                df[var_name] = df[var_name].replace(missing_value, np.nan)
        
        return df
    
    def convert_time_format(self, df):
        """
        转换时间为可读格式
        
        Parameters:
        - df: DataFrame
        
        Returns:
        - 转换时间格式后的DataFrame
        """
        time_columns = ['valid_time', 'time', 'datetime']
        
        for time_col in time_columns:
            if time_col in df.columns:
                try:
                    # 尝试转换为datetime格式
                    if df[time_col].dtype in ['int64', 'int32', 'float64']:
                        # 假设是Unix时间戳（秒 since 1970-01-01）
                        df[f'{time_col}_datetime'] = pd.to_datetime(
                            df[time_col], unit='s', errors='coerce'
                        )
                        # 移除原始时间列（可选）
                        # df = df.drop(columns=[time_col])
                    else:
                        df[f'{time_col}_datetime'] = pd.to_datetime(
                            df[time_col], errors='coerce'
                        )
                    
                    print(f"已转换时间列: {time_col}")
                    
                except Exception as e:
                    print(f"转换时间列 {time_col} 时出错: {str(e)}")
        
        return df
    
    def convert_temperature_units_global(self, df, ds):
        """
        全局温度单位转换（开尔文转摄氏度）
        
        Parameters:
        - df: DataFrame
        - ds: xarray Dataset对象
        
        Returns:
        - 转换单位后的DataFrame
        """
        for var_name in ds.data_vars:
            var_attrs = ds[var_name].attrs
            
            # 检查变量单位
            units = var_attrs.get('units', '').lower()
            long_name = var_attrs.get('long_name', '').lower()
            
            # 识别温度变量
            temperature_indicators = ['kelvin', 'k', 'temperature']
            is_temperature = any(indicator in units.lower() for indicator in temperature_indicators)
            is_temperature = is_temperature or any(indicator in long_name for indicator in temperature_indicators)
            
            if is_temperature and 'K' in units.upper() and var_name in df.columns:
                if df[var_name].dtype in [np.float32, np.float64, np.int32, np.int64]:
                    # 开尔文转摄氏度: °C = K - 273.15
                    df[var_name] = df[var_name] - 273.15
                    print(f"已将变量 {var_name} 从开尔文转换为摄氏度")
        
        return df
    
    def add_unit_info_global(self, df, ds):
        """
        全局添加单位信息到列名
        
        Parameters:
        - df: DataFrame
        - ds: xarray Dataset对象
        
        Returns:
        - 添加单位信息后的DataFrame
        """
        column_rename = {}
        
        for var_name in ds.data_vars:
            if var_name in df.columns:
                var_attrs = ds[var_name].attrs
                units = var_attrs.get('units', '')
                
                # 直接使用原始变量名，不添加单位信息
                new_name = var_name
                
                column_rename[var_name] = new_name
        
        # 重命名列
        if column_rename:
            df = df.rename(columns=column_rename)
            print(f"已重命名列: {column_rename}")
        
        return df
    
    def sample_data_if_large(self, df, max_rows=100000):
        """
        如果数据量过大，进行采样
        
        Parameters:
        - df: DataFrame
        - max_rows: 最大行数限制
        
        Returns:
        - 采样后的DataFrame
        """
        if len(df) > max_rows:
            print(f"数据量过大 ({len(df)} 行)，进行采样至 {max_rows} 行")
            # 等间隔采样
            step = len(df) // max_rows
            df_sampled = df.iloc[::step].reset_index(drop=True)
            return df_sampled
        else:
            return df
    
    def batch_convert(self):
        """
        批量转换NC文件为CSV格式
        """
        # 获取所有NC文件
        nc_pattern = os.path.join(self.input_folder, "*.nc")
        nc_files = glob.glob(nc_pattern)
        
        if not nc_files:
            print(f"在文件夹 {self.input_folder} 中未找到NC文件")
            return
        
        print(f"找到 {len(nc_files)} 个NC文件，使用 {self.num_processes} 个进程进行处理...")
        
        # 使用多进程处理
        with Pool(processes=self.num_processes) as pool:
            results = pool.map(self.process_single_file, nc_files)
        
        # 统计处理结果
        successful = sum(results)
        failed = len(results) - successful
        
        print(f"\n处理完成!")
        print(f"成功: {successful} 个文件")
        print(f"失败: {failed} 个文件")
        print(f"输出文件夹: {self.output_folder}")

def main():
    """
    主函数 - 在这里设置输入和输出路径
    """
    # 设置输入和输出文件夹路径
    input_folder = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-NC\low_vegetation_cover"  # 修改为您的NC文件文件夹路径
    output_folder = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV\low_vegetation_cover"  # 修改为您想要保存CSV文件的文件夹路径
    
    # 创建转换器实例
    converter = NCToCSVConverter(
        input_folder=input_folder,
        output_folder=output_folder,
        num_processes=4  # 可以根据您的CPU核心数调整
    )
    
    # 执行批量转换
    converter.batch_convert()

if __name__ == "__main__":
    main()