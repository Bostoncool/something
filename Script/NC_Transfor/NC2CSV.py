import netCDF4 as nc
import pandas as pd
import numpy as np
import os
import time
import multiprocessing as mp
from tqdm import tqdm
import psutil
import gc

def nc_to_csv_single(args):
    """
    单文件NC转CSV函数（用于多进程）
    
    参数:
    args: (nc_file_path, output_csv_path) 元组
    """
    nc_file_path, output_csv_path = args
    try:
        # 检查输入文件是否存在
        if not os.path.exists(nc_file_path):
            return f"错误: 找不到输入文件: {nc_file_path}"
        
        # 打开NC文件
        dataset = nc.Dataset(nc_file_path, 'r')
        
        # 获取所有变量
        variables = dataset.variables
        data_dict = {}
        max_length = 0
        
        # 遍历所有变量并存储到字典中
        for var_name, var in variables.items():
            try:
                # 检查变量类型和属性
                if not hasattr(var, 'shape'):
                    continue
                
                # 特殊处理标量变量（形状为()的变量）
                if var.shape == ():
                    continue
                
                # 获取变量数据
                var_data = var[:]
                
                # 检查数据类型并相应处理
                if isinstance(var_data, np.ndarray):
                    if len(var_data.shape) > 1:
                        data = var_data.flatten()
                    else:
                        data = var_data
                elif isinstance(var_data, (int, float, np.number)):
                    data = np.array([var_data])
                elif isinstance(var_data, str):
                    continue
                else:
                    try:
                        data = np.array(var_data)
                        if len(data.shape) > 1:
                            data = data.flatten()
                    except Exception:
                        continue
                
                # 检查数据是否为空
                if data.size == 0:
                    continue
                
                # 确保数据是一维的
                if len(data.shape) > 1:
                    data = data.flatten()
                
                data_dict[var_name] = data
                current_length = len(data)
                max_length = max(max_length, current_length)
                
            except Exception:
                continue
        
        # 检查是否有有效数据
        if not data_dict:
            dataset.close()
            return f"错误: 没有找到可转换的变量数据: {nc_file_path}"
        
        # 在创建DataFrame之前，统一数据类型
        for key in data_dict:
            if np.issubdtype(data_dict[key].dtype, np.integer):
                data_dict[key] = data_dict[key].astype(np.float64)
        
        # 确保所有数组长度相同
        for key in data_dict:
            if len(data_dict[key]) < max_length:
                data_dict[key] = np.pad(data_dict[key], (0, max_length - len(data_dict[key])), 
                                      mode='constant', constant_values=np.nan)
        
        # 创建DataFrame
        df = pd.DataFrame(data_dict)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # 保存为CSV文件
        df.to_csv(output_csv_path, index=False)
        
        # 关闭NC文件
        dataset.close()
        
        # 清理内存
        del df, data_dict
        gc.collect()
        
        return f"成功: {os.path.basename(nc_file_path)}"
        
    except Exception as e:
        return f"错误: {os.path.basename(nc_file_path)} - {str(e)}"

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def batch_convert_parallel(input_folder, output_folder, max_workers=None):
    """
    多进程批量转换文件夹中的所有NC文件
    
    参数:
    input_folder: 输入NC文件所在文件夹
    output_folder: 输出CSV文件所在文件夹
    max_workers: 最大进程数，默认为CPU核心数
    """
    try:
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"找不到输入文件夹: {input_folder}")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # 设置进程数
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)  # 限制最大进程数为8
        
        print(f"使用 {max_workers} 个进程进行并行处理")
        print(f"初始内存使用: {get_memory_usage():.1f} MB")
        
        year_folders = [f for f in os.listdir(input_folder) 
                       if os.path.isdir(os.path.join(input_folder, f))]
        print(f"找到以下年份文件夹: {year_folders}")
        
        overall_start = time.time()
        total_files = 0
        processed_files = 0
        
        # 收集所有需要处理的文件
        all_tasks = []
        for year_folder in year_folders:
            year_input_path = os.path.join(input_folder, year_folder)
            year_output_path = os.path.join(output_folder, year_folder)
            os.makedirs(year_output_path, exist_ok=True)
            
            nc_files = [f for f in os.listdir(year_input_path) if f.endswith('.nc')]
            total_files += len(nc_files)
            
            for filename in nc_files:
                nc_path = os.path.join(year_input_path, filename)
                csv_path = os.path.join(year_output_path, filename.replace('.nc', '.csv'))
                
                # 检查输出文件是否已经存在
                if not os.path.exists(csv_path):
                    all_tasks.append((nc_path, csv_path))
        
        print(f"总共需要处理 {len(all_tasks)} 个文件")
        
        # 检查是否有文件需要处理
        if len(all_tasks) == 0:
            print("没有需要处理的文件，所有文件都已转换完成或输入文件夹为空")
            return
        
        # 使用多进程处理
        with mp.Pool(processes=max_workers) as pool:
            # 使用tqdm显示进度
            results = list(tqdm(
                pool.imap(nc_to_csv_single, all_tasks),
                total=len(all_tasks),
                desc="文件转换进度"
            ))
        
        # 统计结果
        success_count = sum(1 for result in results if result.startswith("成功"))
        error_count = len(results) - success_count
        
        overall_end = time.time()
        print(f"\n转换完成统计:")
        print(f"成功转换: {success_count} 个文件")
        print(f"转换失败: {error_count} 个文件")
        print(f"总耗时: {overall_end - overall_start:.2f} 秒")
        
        # 检查是否有处理的任务，避免除零错误
        if len(all_tasks) > 0:
            print(f"平均每个文件耗时: {(overall_end - overall_start) / len(all_tasks):.2f} 秒")
        else:
            print("没有需要处理的文件")
            
        print(f"最终内存使用: {get_memory_usage():.1f} MB")
        
        # 显示错误信息
        if error_count > 0:
            print("\n错误详情:")
            for result in results:
                if result.startswith("错误"):
                    print(f"  {result}")
        
    except Exception as e:
        print(f"批量转换过程中出现错误: {str(e)}")
        raise e

def batch_convert(input_folder, output_folder):
    """
    单进程批量转换（保留原版本作为备选）
    """
    try:
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"找不到输入文件夹: {input_folder}")
        
        os.makedirs(output_folder, exist_ok=True)
        year_folders = [f for f in os.listdir(input_folder) 
                       if os.path.isdir(os.path.join(input_folder, f))]
        print(f"找到以下年份文件夹: {year_folders}")
        
        total_years = len(year_folders)
        overall_start = time.time()
        
        # 检查是否有年份文件夹
        if not year_folders:
            print("没有找到年份文件夹，请检查输入路径")
            return
            
        # 年份文件夹进度条
        for year_folder in tqdm(year_folders, desc="年份进度"):
            try:
                year_input_path = os.path.join(input_folder, year_folder)
                year_output_path = os.path.join(output_folder, year_folder)
                print(f"\n处理年份文件夹: {year_folder}")
                os.makedirs(year_output_path, exist_ok=True)
                nc_files = [f for f in os.listdir(year_input_path) if f.endswith('.nc')]
                print(f"在 {year_folder} 中找到 {len(nc_files)} 个NC文件")
                
                # 检查是否有NC文件
                if not nc_files:
                    print(f"在 {year_folder} 中没有找到NC文件，跳过")
                    continue
                
                # 文件进度条
                for filename in tqdm(nc_files, desc=f"{year_folder}文件进度", leave=False):
                    nc_path = os.path.join(year_input_path, filename)
                    csv_path = os.path.join(year_output_path, filename.replace('.nc', '.csv'))
                    if os.path.exists(csv_path):
                        print(f"文件已存在，跳过转换: {csv_path}")
                        continue
                    nc_to_csv(nc_path, csv_path)
            except Exception as year_error:
                print(f"处理年份 {year_folder} 时出错: {str(year_error)}")
                continue
        
        overall_end = time.time()
        print(f"\n全部批量转换完成，总耗时: {overall_end - overall_start:.2f} 秒")
    except Exception as e:
        print(f"批量转换过程中出现错误: {str(e)}")
        raise e

def nc_to_csv(nc_file_path, output_csv_path):
    """
    将NC文件转换为CSV文件（单文件处理，保留原版本）
    """
    try:
        print(f"开始处理文件: {nc_file_path}")
        start_time = time.time()
        
        if not os.path.exists(nc_file_path):
            raise FileNotFoundError(f"找不到输入文件: {nc_file_path}")
        
        dataset = nc.Dataset(nc_file_path, 'r')
        print(f"成功打开NC文件，包含的变量: {list(dataset.variables.keys())}")
        
        variables = dataset.variables
        data_dict = {}
        max_length = 0
        
        # 遍历所有变量并存储到字典中（加进度条）
        for var_name, var in tqdm(variables.items(), desc="变量处理进度", leave=False):
            try:
                print(f"正在处理变量: {var_name}")
                if not hasattr(var, 'shape'):
                    print(f"跳过变量 {var_name}：没有shape属性")
                    continue
                print(f"变量 {var_name} 的形状: {var.shape}, 数据类型: {var.dtype}")
                if var.shape == ():
                    print(f"跳过标量变量 {var_name}")
                    continue
                var_data = var[:]
                if isinstance(var_data, np.ndarray):
                    if len(var_data.shape) > 1:
                        data = var_data.flatten()
                    else:
                        data = var_data
                elif isinstance(var_data, (int, float, np.number)):
                    data = np.array([var_data])
                elif isinstance(var_data, str):
                    print(f"跳过字符串变量 {var_name}")
                    continue
                else:
                    try:
                        data = np.array(var_data)
                        if len(data.shape) > 1:
                            data = data.flatten()
                    except Exception as convert_error:
                        print(f"无法转换变量 {var_name} 的数据类型: {str(convert_error)}")
                        continue
                if data.size == 0:
                    print(f"跳过空变量 {var_name}")
                    continue
                if len(data.shape) > 1:
                    data = data.flatten()
                data_dict[var_name] = data
                current_length = len(data)
                max_length = max(max_length, current_length)
                print(f"成功处理变量 {var_name}，长度: {current_length}")
            except Exception as var_error:
                print(f"处理变量 {var_name} 时出错: {str(var_error)}")
                continue
        
        if not data_dict:
            raise ValueError("没有找到可转换的变量数据")
        
        # 在创建DataFrame之前，统一数据类型
        for key in data_dict:
            if np.issubdtype(data_dict[key].dtype, np.integer):
                data_dict[key] = data_dict[key].astype(np.float64)
        
        # 确保所有数组长度相同
        for key in data_dict:
            if len(data_dict[key]) < max_length:
                data_dict[key] = np.pad(data_dict[key], (0, max_length - len(data_dict[key])), 
                                      mode='constant', constant_values=np.nan)
        
        df = pd.DataFrame(data_dict)
        print(f"成功创建DataFrame，大小: {df.shape}")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"成功保存CSV文件: {output_csv_path}")
        dataset.close()
        end_time = time.time()
        print(f"文件 {os.path.basename(nc_file_path)} 转换完成，耗时: {end_time - start_time:.2f} 秒\n")
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        # 批量转换
        input_folder = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-NC"
        output_folder = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV"
        print(f"开始批量转换\n输入文件夹: {input_folder}\n输出文件夹: {output_folder}")
        
        # 使用多进程版本（推荐）
        print("使用多进程并行处理...")
        batch_convert_parallel(input_folder, output_folder)
        
        # 如果需要使用单进程版本，取消下面这行的注释
        # batch_convert(input_folder, output_folder)
        
        print("批量转换完成！")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")