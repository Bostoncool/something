import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NCToCSVConverter:
    def __init__(self, input_folder, output_folder, processes=None):
        """
        初始化NC到CSV转换器
        
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
            processes: 进程数，默认为CPU核心数
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.processes = processes or cpu_count()
        
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
    
    def process_single_file(self, nc_file):
        """
        处理单个NC文件
        
        Args:
            nc_file: NC文件路径
            
        Returns:
            tuple: (输入文件路径, 输出文件路径, 是否成功, 错误信息)
        """
        try:
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(nc_file))[0]
            output_file = os.path.join(self.output_folder, f"{base_name}.csv")
            
            print(f"正在处理: {nc_file}")
            
            # 读取NC文件
            with xr.open_dataset(nc_file) as ds:
                # 检查维度结构
                print(f"  文件维度: {dict(ds.dims)}")
                print(f"  文件变量: {list(ds.variables.keys())}")
                
                # 处理主要数据变量（u100）
                if 'u100' in ds.variables:
                    # 创建DataFrame
                    df = self.create_dataframe_from_nc(ds)
                    
                    # 处理缺失值
                    df = self.handle_missing_values(df, ds)
                    
                    # 保存为CSV
                    self.save_to_csv(df, output_file, ds)
                    
                    print(f"  成功转换: {output_file}")
                    return (nc_file, output_file, True, None)
                else:
                    error_msg = f"  文件中未找到u100变量"
                    print(error_msg)
                    return (nc_file, None, False, error_msg)
                    
        except Exception as e:
            error_msg = f"  处理文件时出错: {str(e)}"
            print(error_msg)
            return (nc_file, None, False, error_msg)
    
    def create_dataframe_from_nc(self, ds):
        """
        从NC数据集创建DataFrame
        
        Args:
            ds: xarray数据集
            
        Returns:
            pandas.DataFrame: 转换后的数据框
        """
        # 提取坐标数据
        time_data = ds['valid_time']
        lat_data = ds['latitude']
        lon_data = ds['longitude']
        
        # 转换时间为可读格式
        if hasattr(time_data, 'values'):
            if np.issubdtype(time_data.dtype, np.integer):
                # 处理Unix时间戳
                times = pd.to_datetime(time_data.values, unit='s', origin='unix')
            else:
                times = pd.to_datetime(time_data.values)
        else:
            times = pd.to_datetime(time_data)
        
        # 创建多索引
        multi_index = pd.MultiIndex.from_product(
            [times, lat_data.values, lon_data.values],
            names=['time', 'latitude', 'longitude']
        )
        
        # 提取u100数据并展平
        u100_data = ds['u100'].values
        u100_flat = u100_data.reshape(-1)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'u100': u100_flat
        }, index=multi_index)
        
        # 重置索引，将多索引变为列
        df = df.reset_index()
        
        # 添加其他变量（如果存在）
        if 'number' in ds.variables:
            df['ensemble_number'] = ds['number'].values.item() if ds['number'].values.size == 1 else ds['number'].values[0]
        
        if 'expver' in ds.variables:
            # expver是时间维度的变量，需要适当处理
            expver_data = ds['expver'].values
            if len(expver_data) == len(times):
                df['expver'] = np.repeat(expver_data, len(lat_data) * len(lon_data))
        
        return df
    
    def handle_missing_values(self, df, ds):
        """
        处理缺失值
        
        Args:
            df: 数据框
            ds: xarray数据集
            
        Returns:
            pandas.DataFrame: 处理缺失值后的数据框
        """
        # 检查u100变量的缺失值
        if 'u100' in ds.variables and '_FillValue' in ds['u100'].attrs:
            fill_value = ds['u100'].attrs['_FillValue']
            df['u100'] = df['u100'].replace(fill_value, np.nan)
        
        # 移除所有行都是NaN的行
        df = df.dropna(how='all')
        
        return df
    
    def save_to_csv(self, df, output_file, ds):
        """
        保存DataFrame为CSV文件
        
        Args:
            df: 数据框
            output_file: 输出文件路径
            ds: xarray数据集（用于获取元数据）
        """
        # 添加单位信息作为注释
        units_info = []
        if 'u100' in ds.variables and 'units' in ds['u100'].attrs:
            units_info.append(f"u100 units: {ds['u100'].attrs['units']}")
        
        # 控制输出文件大小 - 如果数据量太大，进行采样
        max_rows = 100000  # 最大行数限制
        if len(df) > max_rows:
            print(f"  数据量较大 ({len(df)} 行)，进行采样...")
            # 每n行采样一次
            sampling_rate = len(df) // max_rows + 1
            df = df.iloc[::sampling_rate]
            print(f"  采样后数据量: {len(df)} 行")
        
        # 写入CSV文件
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入元数据注释
            if units_info:
                for info in units_info:
                    f.write(f"# {info}\n")
            f.write(f"# 数据来源: {os.path.basename(output_file)}\n")
            f.write(f"# 转换时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 总数据行数: {len(df)}\n")
            
            # 写入数据
            df.to_csv(f, index=False)
    
    def batch_convert(self):
        """
        批量转换NC文件为CSV格式
        """
        # 获取所有NC文件
        nc_files = glob.glob(os.path.join(self.input_folder, "*.nc"))
        
        if not nc_files:
            print(f"在文件夹 {self.input_folder} 中未找到NC文件")
            return
        
        print(f"找到 {len(nc_files)} 个NC文件")
        print(f"使用 {self.processes} 个进程进行并行处理")
        
        # 使用多进程处理
        with Pool(processes=self.processes) as pool:
            results = pool.map(self.process_single_file, nc_files)
        
        # 统计结果
        successful = 0
        failed = 0
        
        print("\n" + "="*50)
        print("转换结果汇总:")
        print("="*50)
        
        for input_file, output_file, success, error in results:
            if success:
                print(f"✓ 成功: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
                successful += 1
            else:
                print(f"✗ 失败: {os.path.basename(input_file)} - {error}")
                failed += 1
        
        print(f"\n总计: {successful} 个文件成功, {failed} 个文件失败")

def main():
    """
    主函数 - 在这里指定输入和输出路径
    """
    # 在这里指定您的文件夹路径
    input_folder = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-NC\100m_u_component_of_wind"  # 替换为您的输入文件夹路径
    output_folder = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV\100m_u_component_of_wind"  # 替换为您的输出文件夹路径
    
    # 进程数 (None表示使用所有CPU核心)
    processes = None
    
    # 创建转换器并执行批量转换
    converter = NCToCSVConverter(input_folder, output_folder, processes)
    converter.batch_convert()

if __name__ == "__main__":
    main()