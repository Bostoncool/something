import os
import numpy as np
import pandas as pd
from osgeo import gdal, osr

def csv_to_tif(csv_path, tif_path, x_col='x', y_col='y', value_col='value'):
    """
    将CSV文件转换为TIF格式的栅格文件
    
    参数:
        csv_path: 输入CSV文件路径
        tif_path: 输出TIF文件路径
        x_col: CSV中x坐标列名 (默认'x')
        y_col: CSV中y坐标列名 (默认'y') 
        value_col: CSV中值列名 (默认'value')
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 获取唯一坐标点
    x_coords = np.sort(df[x_col].unique())
    y_coords = np.sort(df[y_col].unique())[::-1]  # 反转y坐标
    
    # 创建空数组
    rows = len(y_coords)
    cols = len(x_coords)
    data = np.zeros((rows, cols), dtype=np.float32)
    
    # 填充数组
    for _, row in df.iterrows():
        x_idx = np.where(x_coords == row[x_col])[0][0]
        y_idx = np.where(y_coords == row[y_col])[0][0]
        data[y_idx, x_idx] = row[value_col]
    
    # 创建TIF文件
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(tif_path, cols, rows, 1, gdal.GDT_Float32)
    
    # 设置地理变换
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    pixel_width = (x_max - x_min) / (cols - 1)
    pixel_height = (y_max - y_min) / (rows - 1)
    
    out_ds.SetGeoTransform((x_min, pixel_width, 0, y_max, 0, -pixel_height))
    
    # 设置空间参考 (WGS84)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_ds.SetProjection(srs.ExportToWkt())
    
    # 写入数据
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(data)
    out_band.FlushCache()
    out_band.SetNoDataValue(-9999)
    
    # 关闭文件
    out_ds = None

if __name__ == "__main__":
    # 示例用法
    input_csv = "input.csv"  # 替换为你的CSV文件路径
    output_tif = "output.tif"  # 替换为输出TIF文件路径
    
    csv_to_tif(input_csv, output_tif)