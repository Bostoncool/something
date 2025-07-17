#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import netCDF4 as nc
import argparse
from datetime import datetime

def h5_to_nc(h5_file_path, nc_file_path=None):
    """
    将H5格式文件转换为NC格式
    
    参数:
        h5_file_path: H5文件的路径
        nc_file_path: 输出NC文件的路径，如果为None则使用与输入文件相同的名称但扩展名为.nc
    
    返回:
        nc_file_path: 创建的NC文件路径
    """
    # 如果未指定输出文件路径，则自动生成
    if nc_file_path is None:
        nc_file_path = os.path.splitext(h5_file_path)[0] + '.nc'
    
    print(f"转换 {h5_file_path} 为 {nc_file_path}")
    
    # 打开H5文件
    with h5py.File(h5_file_path, 'r') as h5_file:
        # 创建NC文件
        with nc.Dataset(nc_file_path, 'w', format='NETCDF4') as nc_file:
            # 添加全局属性
            nc_file.title = f"由H5文件转换 {os.path.basename(h5_file_path)}"
            nc_file.description = "使用Python h5py和netCDF4库转换"
            nc_file.history = f"创建于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            nc_file.source = h5_file_path
            
            # 复制H5文件的属性到NC文件
            for attr_name, attr_value in h5_file.attrs.items():
                try:
                    nc_file.setncattr(attr_name, attr_value)
                except:
                    print(f"无法设置属性: {attr_name}")
            
            # 递归处理H5文件中的组和数据集
            def process_group(h5_group, nc_group):
                # 处理组中的所有数据集
                for name, h5_dataset in h5_group.items():
                    # 如果项目是数据集
                    if isinstance(h5_dataset, h5py.Dataset):
                        # 获取数据集维度
                        dims = h5_dataset.shape
                        
                        # 创建维度（如果尚不存在）
                        dim_names = []
                        for i, dim_size in enumerate(dims):
                            dim_name = f"{name}_dim{i}"
                            if dim_name not in nc_group.dimensions:
                                nc_group.createDimension(dim_name, dim_size)
                            dim_names.append(dim_name)
                        
                        # 创建变量
                        dtype = h5_dataset.dtype
                        # 处理复杂数据类型
                        if dtype.kind == 'S':  # 字符串
                            nc_var = nc_group.createVariable(name, 'S1', dim_names)
                            nc_var[:] = np.array(h5_dataset[()])
                        else:
                            try:
                                nc_var = nc_group.createVariable(name, dtype, dim_names)
                                nc_var[:] = h5_dataset[()]
                            except TypeError:
                                print(f"警告: 无法直接转换数据集 '{name}'，尝试转换为float类型")
                                nc_var = nc_group.createVariable(name, np.float64, dim_names)
                                nc_var[:] = h5_dataset[()].astype(np.float64)
                        
                        # 复制数据集属性
                        for attr_name, attr_value in h5_dataset.attrs.items():
                            try:
                                nc_var.setncattr(attr_name, attr_value)
                            except:
                                print(f"无法设置 {name} 的属性: {attr_name}")
                    
                    # 如果项目是组，创建一个新的NC组并递归处理
                    elif isinstance(h5_dataset, h5py.Group):
                        new_nc_group = nc_group.createGroup(name)
                        process_group(h5_dataset, new_nc_group)
            
            # 从根组开始处理
            process_group(h5_file, nc_file)
    
    print(f"转换完成: {nc_file_path}")
    return nc_file_path

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将H5格式文件转换为NC格式')
    parser.add_argument('input_file', help='输入H5文件的路径')
    parser.add_argument('-o', '--output_file', help='输出NC文件的路径（可选）')
    args = parser.parse_args()
    
    # 执行转换
    h5_to_nc(args.input_file, args.output_file)

if __name__ == "__main__":
    main() 