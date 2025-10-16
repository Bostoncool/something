import os
import glob
import multiprocessing as mp
import pandas as pd
import netCDF4 as nc
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class NetCDFToCSVConverter:
    def __init__(self, input_folder, output_folder, num_processes=None):
        """
        初始化NetCDF转CSV转换器
        
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
            num_processes: 进程数，默认为CPU核心数
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_processes = num_processes or mp.cpu_count()
        
        # 创建输出文件夹
        os.makedirs(self.output_folder, exist_ok=True)
        
    def process_single_file(self, nc_file_path):
        """
        处理单个NetCDF文件
        
        Args:
            nc_file_path: NetCDF文件路径
            
        Returns:
            tuple: (输入文件路径, 输出文件路径, 状态, 错误信息)
        """
        try:
            # 获取文件名（不含扩展名）
            file_name = os.path.splitext(os.path.basename(nc_file_path))[0]
            output_file_path = os.path.join(self.output_folder, f"{file_name}.csv")
            
            # 读取NetCDF文件
            with nc.Dataset(nc_file_path, 'r') as dataset:
                # 检查必要的变量是否存在
                if 'PM2.5' not in dataset.variables:
                    return nc_file_path, output_file_path, "failed", "缺少PM2.5变量"
                
                # 获取维度信息
                lat_dim = dataset.dimensions['lat']
                lon_dim = dataset.dimensions['lon']
                
                # 获取变量数据
                lat_data = dataset.variables['lat'][:]
                lon_data = dataset.variables['lon'][:]
                pm25_data = dataset.variables['PM2.5'][:]
                
                # 获取PM2.5的属性
                pm25_attrs = dataset.variables['PM2.5'].__dict__
                fill_value = pm25_attrs.get('_FillValue', 65535)
                scale_factor = pm25_attrs.get('scale_factor', 0.1)
                add_offset = pm25_attrs.get('add_offset', 0.0)
                units = pm25_attrs.get('units', 'µg/m3')
            
            # 处理缺失值和数据缩放
            pm25_data = pm25_data.astype(np.float32)
            pm25_data[pm25_data == fill_value] = np.nan
            pm25_data = pm25_data * scale_factor + add_offset
            
            # 创建网格
            lon_grid, lat_grid = np.meshgrid(lon_data, lat_data)
            
            # 展平数据并创建DataFrame
            df_data = {
                'latitude': lat_grid.flatten(),
                'longitude': lon_grid.flatten(),
                'PM25': pm25_data.flatten()
            }
            
            df = pd.DataFrame(df_data)
            
            # 移除缺失值以减小文件大小
            df = df.dropna(subset=['PM25'])
            
            # 添加单位信息作为注释
            metadata_comment = f"# Units: PM25 = {units}\n"
            metadata_comment += f"# Data scaled with: value = raw * {scale_factor} + {add_offset}\n"
            metadata_comment += f"# Original dimensions: lat={len(lat_data)}, lon={len(lon_data)}\n"
            metadata_comment += f"# Processed data points: {len(df)}\n"
            
            # 写入CSV文件
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(metadata_comment)
                df.to_csv(f, index=False)
            
            return nc_file_path, output_file_path, "success", None
            
        except Exception as e:
            return nc_file_path, "", "failed", str(e)
    
    def batch_convert(self, file_pattern="*.nc"):
        """
        批量转换NetCDF文件为CSV格式（递归搜索所有子文件夹）
        
        Args:
            file_pattern: 文件匹配模式
        """
        # 检查输入文件夹是否存在
        if not os.path.exists(self.input_folder):
            print(f"错误：输入文件夹不存在: {self.input_folder}")
            print("请检查路径是否正确，或者U盘是否已正确连接")
            return
        
        print(f"Searching folder: {self.input_folder}")
        
        # 递归获取所有NetCDF文件（包括子文件夹）
        # 使用 os.walk 来确保递归搜索
        nc_files = []
        for root, dirs, files in os.walk(self.input_folder):
            for file in files:
                if file.endswith('.nc'):
                    nc_files.append(os.path.join(root, file))
        
        # 如果没找到文件，尝试列出目录内容进行调试
        if not nc_files:
            print(f"No {file_pattern} files found in folder {self.input_folder}")
            print("\nDebug info:")
            print(f"Folder exists: {os.path.exists(self.input_folder)}")
            print(f"Is directory: {os.path.isdir(self.input_folder)}")
            
            # 尝试列出目录内容
            try:
                dir_contents = os.listdir(self.input_folder)
                print(f"Directory contents: {dir_contents[:10]}...")  # 只显示前10个
                
                # 尝试不同的搜索模式
                print("\nTrying different search patterns:")
                patterns_to_try = ["*.nc", "**/*.nc", "**/*.NC", "*/*.nc"]
                for pattern in patterns_to_try:
                    test_files = glob.glob(os.path.join(self.input_folder, pattern), recursive=True)
                    print(f"  Pattern '{pattern}': found {len(test_files)} files")
                    if test_files:
                        print(f"    Example file: {test_files[0]}")
                
                # 检查子文件夹内容
                print("\nChecking subfolder contents:")
                for subfolder in dir_contents[:3]:  # 只检查前3个子文件夹
                    subfolder_path = os.path.join(self.input_folder, subfolder)
                    if os.path.isdir(subfolder_path):
                        try:
                            sub_contents = os.listdir(subfolder_path)
                            nc_files_in_sub = [f for f in sub_contents if f.endswith('.nc')]
                            print(f"  {subfolder}: {len(nc_files_in_sub)} .nc files")
                            if nc_files_in_sub:
                                print(f"    Example: {nc_files_in_sub[0]}")
                        except Exception as e:
                            print(f"  {subfolder}: Error - {e}")
            except Exception as e:
                print(f"Cannot access directory: {e}")
            return
        
        print(f"Found {len(nc_files)} NetCDF files")
        print(f"Using {self.num_processes} processes for conversion")
        print(f"Output folder: {self.output_folder}")
        print("-" * 50)
        
        # 使用多进程处理
        with mp.Pool(processes=self.num_processes) as pool:
            results = list(tqdm(
                pool.imap(self.process_single_file, nc_files),
                total=len(nc_files),
                desc="Conversion progress"
            ))
        
        # 统计结果
        successful = 0
        failed = 0
        
        print("\nConversion results:")
        print("-" * 50)
        for input_file, output_file, status, error_msg in results:
            if status == "success":
                print(f"[OK] {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
                successful += 1
            else:
                print(f"[FAIL] {os.path.basename(input_file)} -> failed: {error_msg}")
                failed += 1
        
        print("-" * 50)
        print(f"Success: {successful}, Failed: {failed}")

def test_path_access():
    """
    测试路径访问功能
    """
    possible_paths = [
        r"G:\2000-2023[PM2.5(NC)]",
        r"G:\Bodhi_Tree\2000-2023[PM2.5(NC)]",
        r"G:\\",
        r"G:\Bodhi_Tree"
    ]
    
    print("Testing possible paths:")
    for path in possible_paths:
        exists = os.path.exists(path)
        is_dir = os.path.isdir(path) if exists else False
        print(f"  {path}: exists={exists}, is_dir={is_dir}")
        
        if exists and is_dir:
            try:
                contents = os.listdir(path)
                print(f"    contents: {contents[:5]}...")
            except Exception as e:
                print(f"    cannot list contents: {e}")

def main():
    """
    主函数 - 在这里设置输入和输出路径
    """
    # ================================
    # 在这里设置您的文件夹路径
    # ================================
    
    # 首先测试路径访问
    test_path_access()
    print("\n" + "="*50 + "\n")
    
    # 输入文件夹路径（包含.nc文件的文件夹）
    INPUT_FOLDER = r"G:\2000-2023[PM2.5(NC)]"  # 修改为您的输入文件夹路径
    
    # 输出文件夹路径（将保存.csv文件的文件夹）
    OUTPUT_FOLDER = r"G:\2000-2023[PM2.5(CSV)]"  # 修改为您的输出文件夹路径
    
    # 进程数（默认为CPU核心数，可根据需要调整）
    NUM_PROCESSES = mp.cpu_count()  # 可以设置为具体数字，如 4
    
    # ================================
    # 执行转换
    # ================================
    
    converter = NetCDFToCSVConverter(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        num_processes=NUM_PROCESSES
    )
    
    # 开始批量转换
    converter.batch_convert()

if __name__ == "__main__":
    # 在Windows上使用多进程时需要这个保护
    mp.freeze_support()
    main()