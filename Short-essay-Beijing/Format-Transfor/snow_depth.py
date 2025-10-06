import os
import glob
import multiprocessing as mp
from datetime import datetime
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class NCToCSVConverter:
    def __init__(self, input_folder, output_folder, num_processes=None):
        """
        初始化NC到CSV转换器
        
        Parameters:
        -----------
        input_folder : str
            输入NC文件文件夹路径
        output_folder : str
            输出CSV文件文件夹路径
        num_processes : int, optional
            进程数，默认为CPU核心数
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_processes = num_processes or mp.cpu_count()
        
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 温度单位映射（开尔文转摄氏度）
        self.temp_unit_mapping = {
            'K': '°C',
            'kelvin': '°C'
        }
        
        # 关键变量保留列表（可根据需要调整）
        self.key_variables = [
            'sd',           # 雪深
            't2m',          # 2米温度
            'tp',           # 总降水
            'u10', 'v10',   # 10米风场
            'msl',          # 平均海平面气压
            'd2m',          # 2米露点温度
            'blh',          # 边界层高度
            'ssr',          # 表面净太阳辐射
            'str',          # 表面净热辐射
            'ssrd',         # 表面太阳辐射向下
            'strd',         # 表面热辐射向下
        ]
    
    def find_nc_files(self):
        """查找所有NC文件"""
        pattern = os.path.join(self.input_folder, "*.nc")
        nc_files = glob.glob(pattern)
        print(f"找到 {len(nc_files)} 个NC文件")
        return nc_files
    
    def convert_temperature_units(self, data_array, units):
        """
        温度单位转换：开尔文转摄氏度
        
        Parameters:
        -----------
        data_array : xarray.DataArray
            数据数组
        units : str
            原始单位
            
        Returns:
        --------
        converted_data : xarray.DataArray
            转换后的数据
        new_units : str
            新单位
        """
        if units.lower() in ['k', 'kelvin']:
            # 开尔文转摄氏度: °C = K - 273.15
            converted_data = data_array - 273.15
            new_units = '°C'
            print(f"温度单位已从 {units} 转换为 {new_units}")
            return converted_data, new_units
        else:
            return data_array, units
    
    def process_time_dimension(self, ds):
        """
        处理时间维度，转换为可读格式
        
        Parameters:
        -----------
        ds : xarray.Dataset
            NC数据集
            
        Returns:
        --------
        ds : xarray.Dataset
            处理后的数据集
        """
        # 检查时间相关变量
        time_vars = ['time', 'valid_time', 'forecast_time']
        time_var = None
        
        for var in time_vars:
            if var in ds.variables:
                time_var = var
                break
        
        if time_var:
            try:
                # 解码时间
                if 'units' in ds[time_var].attrs:
                    units = ds[time_var].attrs['units']
                    if 'since' in units:
                        ds = xr.decode_cf(ds)
                        print(f"时间变量 '{time_var}' 已解码")
            except Exception as e:
                print(f"时间解码失败: {e}")
        
        return ds
    
    def handle_missing_values(self, data_array):
        """
        处理缺失值
        
        Parameters:
        -----------
        data_array : xarray.DataArray
            数据数组
            
        Returns:
        --------
        cleaned_data : xarray.DataArray
            清理后的数据
        """
        # 检查是否有_FillValue属性
        if '_FillValue' in data_array.attrs:
            fill_value = data_array.attrs['_FillValue']
            # 使用NaN替换填充值
            data_array = data_array.where(data_array != fill_value)
            print(f"已处理缺失值: {fill_value}")
        
        # 检查是否有missing_value属性
        elif 'missing_value' in data_array.attrs:
            missing_value = data_array.attrs['missing_value']
            data_array = data_array.where(data_array != missing_value)
            print(f"已处理缺失值: {missing_value}")
        
        return data_array
    
    def select_key_variables(self, ds):
        """
        选择关键变量
        
        Parameters:
        -----------
        ds : xarray.Dataset
            原始数据集
            
        Returns:
        --------
        selected_ds : xarray.Dataset
            选择变量后的数据集
        """
        available_vars = list(ds.variables)
        selected_vars = []
        
        # 选择关键变量
        for var in self.key_variables:
            if var in available_vars:
                selected_vars.append(var)
        
        # 确保保留维度坐标
        for dim in ['latitude', 'longitude', 'lat', 'lon', 'time', 'valid_time']:
            if dim in available_vars and dim not in selected_vars:
                selected_vars.append(dim)
        
        print(f"选择的变量: {selected_vars}")
        return ds[selected_vars]
    
    def convert_single_file(self, nc_file_path):
        """
        转换单个NC文件为CSV
        
        Parameters:
        -----------
        nc_file_path : str
            NC文件路径
            
        Returns:
        --------
        success : bool
            转换是否成功
        """
        try:
            # 获取文件名（不含扩展名）
            file_name = os.path.splitext(os.path.basename(nc_file_path))[0]
            output_csv_path = os.path.join(self.output_folder, f"{file_name}.csv")
            
            print(f"正在处理: {file_name}")
            
            # 读取NC文件
            with xr.open_dataset(nc_file_path) as ds:
                print(f"文件维度结构: {dict(ds.dims)}")
                
                # 处理时间维度
                ds = self.process_time_dimension(ds)
                
                # 选择关键变量
                ds = self.select_key_variables(ds)
                
                # 处理每个数据变量
                for var_name in ds.data_vars:
                    var = ds[var_name]
                    
                    # 处理缺失值
                    var = self.handle_missing_values(var)
                    
                    # 检查并转换温度单位
                    if 'units' in var.attrs:
                        original_units = var.attrs['units']
                        if original_units in self.temp_unit_mapping:
                            converted_data, new_units = self.convert_temperature_units(
                                var, original_units
                            )
                            ds[var_name] = converted_data
                            ds[var_name].attrs['units'] = new_units
                            ds[var_name].attrs['original_units'] = original_units
                
                # 转换为DataFrame
                # 首先尝试将多维数据展平
                if len(ds.dims) > 1:
                    # 重置多维坐标为列
                    df = ds.to_dataframe().reset_index()
                else:
                    # 对于一维数据直接转换
                    df = ds.to_dataframe()
                
                # 清理索引名称
                df = df.reset_index(drop=True)
                
                # 控制输出文件大小 - 如果数据太大，进行采样
                max_rows = 1000000  # 最大行数限制
                if len(df) > max_rows:
                    print(f"数据过大 ({len(df)} 行)，进行采样...")
                    # 按时间均匀采样
                    if 'time' in df.columns:
                        df = df.iloc[::len(df)//max_rows + 1]
                    else:
                        df = df.sample(n=max_rows)
                    print(f"采样后数据大小: {len(df)} 行")
                
                # 保存为CSV
                df.to_csv(output_csv_path, index=False, encoding='utf-8')
                
                # 生成元数据文件
                self.generate_metadata_file(ds, file_name)
                
            print(f"成功转换: {file_name} -> {output_csv_path}")
            return True
            
        except Exception as e:
            print(f"转换失败 {nc_file_path}: {e}")
            return False
    
    def generate_metadata_file(self, ds, file_name):
        """
        生成元数据文件
        
        Parameters:
        -----------
        ds : xarray.Dataset
            数据集
        file_name : str
            文件名
        """
        metadata_path = os.path.join(self.output_folder, f"{file_name}_metadata.txt")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("=== 数据文件元信息 ===\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("维度信息:\n")
            for dim, size in ds.dims.items():
                f.write(f"  {dim}: {size}\n")
            
            f.write("\n变量信息:\n")
            for var_name in ds.variables:
                var = ds[var_name]
                f.write(f"\n变量名: {var_name}\n")
                f.write(f"  形状: {var.shape}\n")
                if hasattr(var, 'dtype'):
                    f.write(f"  数据类型: {var.dtype}\n")
                if var.attrs:
                    f.write("  属性:\n")
                    for attr, value in var.attrs.items():
                        f.write(f"    {attr}: {value}\n")
            
            f.write("\n全局属性:\n")
            for attr, value in ds.attrs.items():
                f.write(f"  {attr}: {value}\n")
    
    def batch_convert(self):
        """批量转换所有NC文件"""
        nc_files = self.find_nc_files()
        
        if not nc_files:
            print("未找到NC文件，请检查输入路径")
            return
        
        print(f"使用 {self.num_processes} 个进程进行并行处理")
        
        # 使用多进程处理
        with mp.Pool(processes=self.num_processes) as pool:
            results = list(tqdm(
                pool.imap(self.convert_single_file, nc_files),
                total=len(nc_files),
                desc="转换进度"
            ))
        
        # 统计结果
        successful = sum(results)
        failed = len(results) - successful
        
        print(f"\n转换完成!")
        print(f"成功: {successful} 个文件")
        print(f"失败: {failed} 个文件")
        print(f"输出目录: {self.output_folder}")


def main():
    """主函数"""
    # ================================
    # 在这里设置输入和输出文件夹路径
    # ================================
    
    # 输入文件夹（包含NC文件）
    INPUT_FOLDER = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-NC\snow_depth"  # 请修改为实际路径
    
    # 输出文件夹（CSV文件将保存到这里）
    OUTPUT_FOLDER = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV\snow_depth"  # 请修改为实际路径
    
    # 进程数（None表示使用所有CPU核心）
    NUM_PROCESSES = None
    
    # 检查输入文件夹是否存在
    if not os.path.exists(INPUT_FOLDER):
        print(f"错误: 输入文件夹不存在: {INPUT_FOLDER}")
        return
    
    # 创建转换器并执行批量转换
    converter = NCToCSVConverter(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        num_processes=NUM_PROCESSES
    )
    
    converter.batch_convert()


if __name__ == "__main__":
    main()