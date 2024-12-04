import netCDF4 as nc
import pandas as pd
import numpy as np
import os

def nc_to_csv(nc_file_path, output_csv_path):
    """
    将NC文件转换为CSV文件
    
    参数:
    nc_file_path: NC文件的路径
    output_csv_path: 输出CSV文件的路径
    """
    try:
        print(f"开始处理文件: {nc_file_path}")
        
        # 检查输入文件是否存在
        if not os.path.exists(nc_file_path):
            raise FileNotFoundError(f"找不到输入文件: {nc_file_path}")
        
        # 打开NC文件
        dataset = nc.Dataset(nc_file_path, 'r')
        print(f"成功打开NC文件，包含的变量: {list(dataset.variables.keys())}")
        
        # 获取所有变量
        variables = dataset.variables
        data_dict = {}
        max_length = 0
        
        # 遍历所有变量并存储到字典中
        for var_name, var in variables.items():
            try:
                print(f"正在处理变量: {var_name}, 形状: {var.shape}")
                # 将变量数据转换为一维数组
                if len(var.shape) > 1:
                    data = var[:].flatten()
                else:
                    data = var[:]
                
                data_dict[var_name] = data
                max_length = max(max_length, len(data))
            except Exception as var_error:
                print(f"处理变量 {var_name} 时出错: {str(var_error)}")
        
        # 确保所有数组长度相同
        for key in data_dict:
            if len(data_dict[key]) < max_length:
                data_dict[key] = np.pad(data_dict[key], (0, max_length - len(data_dict[key])), mode='constant', constant_values=np.nan)
        
        # 创建DataFrame
        df = pd.DataFrame(data_dict)
        print(f"成功创建DataFrame，大小: {df.shape}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # 保存为CSV文件
        df.to_csv(output_csv_path, index=False)
        print(f"成功保存CSV文件: {output_csv_path}")
        
        # 关闭NC文件
        dataset.close()
        
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")
        raise e

def batch_convert(input_folder, output_folder):
    """
    批量转换文件夹中的所有NC文件
    
    参数:
    input_folder: 输入NC文件所在文件夹
    output_folder: 输出CSV文件所在文件夹
    """
    try:
        # 检查输入文件夹是否存在
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"找不到输入文件夹: {input_folder}")
        
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取所有年份文件夹
        year_folders = [f for f in os.listdir(input_folder) 
                       if os.path.isdir(os.path.join(input_folder, f))]
        print(f"找到以下年份文件夹: {year_folders}")
        
        # 遍历输入文件夹中的年份子文件夹
        for year_folder in year_folders:
            try:
                year_input_path = os.path.join(input_folder, year_folder)
                year_output_path = os.path.join(output_folder, year_folder)
                
                print(f"\n处理年份文件夹: {year_folder}")
                
                # 确保输出年份文件夹存在
                os.makedirs(year_output_path, exist_ok=True)
                
                # 获取所有NC文件
                nc_files = [f for f in os.listdir(year_input_path) 
                          if f.endswith('.nc')]
                print(f"在 {year_folder} 中找到 {len(nc_files)} 个NC文件")
                
                # 遍历年份文件夹中的所有NC文件
                for filename in nc_files:
                    nc_path = os.path.join(year_input_path, filename)
                    csv_path = os.path.join(year_output_path, 
                                          filename.replace('.nc', '.csv'))
                    
                    # 检查输出文件是否已经存在
                    if os.path.exists(csv_path):
                        print(f"文件已存在，跳过转换: {csv_path}")
                        continue
                    
                    print(f"\n正在转换文件: {filename}")
                    nc_to_csv(nc_path, csv_path)
                    
            except Exception as year_error:
                print(f"处理年份 {year_folder} 时出错: {str(year_error)}")
                continue
                
    except Exception as e:
        print(f"批量转换过程中出现错误: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        # 批量转换
        input_folder = r"G:\PM2.5(NC)"
        output_folder = r"G:\PM2.5(CSV)"
        print(f"开始批量转换\n输入文件夹: {input_folder}\n输出文件夹: {output_folder}")
        batch_convert(input_folder, output_folder)
        print("批量转换完成！")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")