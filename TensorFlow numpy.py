import numpy as np
from osgeo import gdal
import tensorflow as tf

def load_tif_to_tensor(image_path):
    dataset = gdal.Open(image_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray()
    tensor = tf.image.decode_image(data)
    return tensor

# 示例用法
image_path = 'path_to_your_tif_file.tif'
tensor = load_tif_to_tensor(image_path)
print(tensor.shape)  # 输出张量的形状

