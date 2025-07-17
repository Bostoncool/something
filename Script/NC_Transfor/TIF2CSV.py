import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional

def tif_to_csv(
    input_path: str,
    output_path: str,
    include_coords: bool = False,
    band_names: Optional[list] = None
) -> None:
    """
    将TIF图像转换为CSV文件
    
    参数:
        input_path: 输入TIF文件路径
        output_path: 输出CSV文件路径
        include_coords: 是否包含像素坐标
        band_names: 波段名称列表(多波段时使用)
    """
    try:
        # 读取TIF文件
        with Image.open(input_path) as img:
            # 转换为numpy数组
            arr = np.array(img)
            
            # 处理单波段和多波段情况
            if len(arr.shape) == 2:
                arr = arr[..., np.newaxis]  # 增加一个维度
                
            height, width, bands = arr.shape
            
            # 生成坐标网格(如果需要)
            if include_coords:
                y_coords, x_coords = np.mgrid[0:height, 0:width]
                x_coords = x_coords.flatten()
                y_coords = y_coords.flatten()
            
            # 准备数据框
            data = {}
            if include_coords:
                data['x'] = x_coords
                data['y'] = y_coords
                
            # 添加波段数据
            for b in range(bands):
                band_name = f'band_{b}' if band_names is None else band_names[b]
                data[band_name] = arr[..., b].flatten()
                
            df = pd.DataFrame(data)
            
            # 保存CSV
            df.to_csv(output_path, index=False)
            print(f"成功转换: {input_path} → {output_path}")
            
    except Exception as e:
        print(f"转换失败: {str(e)}")

if __name__ == "__main__":
    # 示例用法
    input_file = "input.tif"
    output_file = "output.csv"
    
    if os.path.exists(input_file):
        tif_to_csv(
            input_file,
            output_file,
            include_coords=True,
            band_names=["red", "green", "blue"]  # 适用于RGB图像
        )
    else:
        print(f"输入文件不存在: {input_file}")
