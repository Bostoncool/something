import os
from osgeo import gdal
import numpy as np

def process_tif_files(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.tif'):
            tif_path = os.path.join(folder_path, filename)
            
            # 打开tif文件
            dataset = gdal.Open(tif_path)
            
            # 读取栅格数据
            band = dataset.GetRasterBand(1)
            
            # 将栅格数据转换为NumPy数组
            array = band.ReadAsArray()
            
            # 关闭数据集
            dataset = None
            
            # 生成输出文件名（与输入文件名相同，但扩展名为.npy）
            output_filename = os.path.splitext(filename)[0] + '.npy'
            output_path = os.path.join(folder_path, output_filename)
            
            # 保存NumPy数组
            np.save(output_path, array)
            
            print(f"已处理文件: {filename}")
            print(f"数组形状: {array.shape}")
            print(f"数组数据类型: {array.dtype}")
            print(f"已保存为: {output_filename}\n")

# 指定要处理的文件夹路径
folder_to_process = 'C:\\path\\to\\your\\folder'

# 调用函数处理文件夹
process_tif_files(folder_to_process)