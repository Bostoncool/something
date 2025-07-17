import h5py
import numpy as np
import tifffile
import os
from pathlib import Path

def h5_to_tif(h5_file_path, output_dir=None, dataset_name=None):
    """
    将H5文件转换为TIF格式
    
    参数:
        h5_file_path: H5文件路径
        output_dir: 输出目录，默认与输入文件相同
        dataset_name: H5文件中的数据集名称，如果为None则尝试自动检测
    
    返回:
        保存的TIF文件路径
    """
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.dirname(h5_file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输出文件名
    base_name = Path(h5_file_path).stem
    tif_file_path = os.path.join(output_dir, f"{base_name}.tif")
    
    try:
        # 打开H5文件
        with h5py.File(h5_file_path, 'r') as h5_file:
            # 如果未指定数据集名称，则尝试查找
            if dataset_name is None:
                # 获取所有数据集
                datasets = []
                
                def collect_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        datasets.append(name)
                
                h5_file.visititems(collect_datasets)
                
                if not datasets:
                    raise ValueError("在H5文件中未找到数据集")
                
                # 使用第一个数据集
                dataset_name = datasets[0]
                print(f"使用数据集: {dataset_name}")
            
            # 读取数据
            data = h5_file[dataset_name][:]
            
            # 确保数据类型适合图像
            if data.dtype == np.float64 or data.dtype == np.float32:
                # 归一化浮点数据到0-65535范围
                if np.min(data) != np.max(data):
                    data = 65535 * (data - np.min(data)) / (np.max(data) - np.min(data))
                data = data.astype(np.uint16)
            
            # 保存为TIF文件
            tifffile.imwrite(tif_file_path, data)
            print(f"文件已保存为: {tif_file_path}")
            
            return tif_file_path
    
    except Exception as e:
        print(f"转换过程中出错: {e}")
        return None

def batch_convert(input_dir, output_dir=None, file_pattern="*.h5", dataset_name=None):
    """
    批量转换目录中的所有H5文件为TIF格式
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        file_pattern: 文件匹配模式
        dataset_name: H5文件中的数据集名称
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有符合模式的H5文件
    h5_files = list(input_path.glob(file_pattern))
    
    if not h5_files:
        print(f"在 {input_dir} 中没有找到匹配 {file_pattern} 的文件")
        return
    
    print(f"找到 {len(h5_files)} 个H5文件待转换")
    
    # 转换每个文件
    for h5_file in h5_files:
        print(f"正在转换: {h5_file}")
        h5_to_tif(str(h5_file), output_dir, dataset_name)
    
    print("批量转换完成")

# 使用示例
if __name__ == "__main__":
    # 单个文件转换
    # h5_to_tif("path/to/your/file.h5")
    
    # 批量转换
     batch_convert("path/to/your/h5/files")
