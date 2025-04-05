import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import rasterio
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
from pathlib import Path
import os

def load_data(file_path):
    """加载NC或TIF格式的气象遥感数据"""
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.nc':
        # 使用xarray加载NC文件
        data = xr.open_dataset(file_path)
        # 获取主要变量数据
        var_name = list(data.data_vars)[0]
        array_data = data[var_name].values
        metadata = {
            'dims': data[var_name].dims,
            'coords': {dim: data[dim].values for dim in data[var_name].dims}
        }
        return array_data, metadata
    
    elif file_extension in ['.tif', '.tiff']:
        # 使用rasterio加载TIF文件
        with rasterio.open(file_path) as src:
            array_data = src.read()
            metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'height': src.height,
                'width': src.width
            }
        return array_data, metadata
    
    else:
        raise ValueError(f"不支持的文件格式: {file_extension}")

def apply_svd(data, n_components=None):
    """应用SVD进行降维和压缩"""
    # 保存原始形状
    original_shape = data.shape
    
    # 如果数据是3D或更高维，将其重塑为2D矩阵
    if len(original_shape) > 2:
        # 假设第一个维度是通道/波段
        n_bands = original_shape[0]
        data_2d = data.reshape(n_bands, -1)
    else:
        data_2d = data.copy()
    
    # 如果没有指定组件数，默认使用全部组件
    if n_components is None:
        # 执行完整SVD
        U, Sigma, Vt = np.linalg.svd(data_2d, full_matrices=False)
        return U, Sigma, Vt, original_shape
    else:
        # 使用截断SVD（更高效，适用于大型数据集）
        svd = TruncatedSVD(n_components=n_components)
        reduced_data = svd.fit_transform(data_2d.T).T
        
        # 获取SVD组件
        Vt = svd.components_
        Sigma = svd.singular_values_
        U = reduced_data / Sigma[:, np.newaxis]
        
        return U, Sigma, Vt, original_shape

def reconstruct_from_svd(U, Sigma, Vt, original_shape, n_components=None):
    """从SVD组件中重建数据"""
    if n_components is None:
        n_components = len(Sigma)
    
    # 使用前n_components个奇异值和奇异向量重构数据
    reconstructed_2d = (U[:, :n_components] * Sigma[:n_components][np.newaxis, :]) @ Vt[:n_components, :]
    
    # 重塑回原始维度
    if len(original_shape) > 2:
        reconstructed = reconstructed_2d.reshape(original_shape)
    else:
        reconstructed = reconstructed_2d
    
    return reconstructed

def calculate_compression_ratio(original_data, n_components, sigma):
    """计算压缩比率"""
    original_size = np.prod(original_data.shape)
    
    # 压缩后的大小是U, Sigma和Vt的元素总数
    if len(original_data.shape) > 2:
        n_bands = original_data.shape[0]
        flattened_size = n_bands * np.prod(original_data.shape[1:])
        compressed_size = n_bands * n_components + n_components + n_components * flattened_size // n_bands
    else:
        compressed_size = original_data.shape[0] * n_components + n_components + n_components * original_data.shape[1]
    
    compression_ratio = original_size / compressed_size
    
    # 计算保留的信息量(方差解释率)
    explained_variance = (sigma[:n_components]**2).sum() / (sigma**2).sum()
    
    return compression_ratio, explained_variance

def visualize_results(original, reconstructed, sigma, n_components, explained_variance):
    """可视化原始数据与重建数据的对比，以及奇异值分布"""
    plt.figure(figsize=(18, 10))
    
    # 绘制奇异值分布
    plt.subplot(2, 3, 1)
    plt.plot(sigma, 'o-')
    plt.axvline(x=n_components, color='r', linestyle='--')
    plt.title(f'奇异值分布 (使用前{n_components}个)')
    plt.xlabel('索引')
    plt.ylabel('奇异值')
    plt.grid(True)
    
    # 绘制累积解释方差
    plt.subplot(2, 3, 2)
    cum_explained_variance = np.cumsum(sigma**2) / np.sum(sigma**2)
    plt.plot(cum_explained_variance, 'o-')
    plt.axhline(y=explained_variance, color='r', linestyle='--')
    plt.axvline(x=n_components, color='r', linestyle='--')
    plt.title(f'累积解释方差 ({explained_variance:.2%})')
    plt.xlabel('组件数')
    plt.ylabel('累积解释方差')
    plt.grid(True)
    
    # 如果数据是3D的，为每个通道显示原始和重建图像
    if len(original.shape) == 3:
        # 显示第一个通道的原始数据
        plt.subplot(2, 3, 4)
        plt.imshow(original[0], cmap='viridis')
        plt.title('原始数据 (第1通道)')
        plt.colorbar()
        
        # 显示第一个通道的重建数据
        plt.subplot(2, 3, 5)
        plt.imshow(reconstructed[0], cmap='viridis')
        plt.title(f'重建数据 (第1通道, {n_components}个组件)')
        plt.colorbar()
        
        # 显示差异
        plt.subplot(2, 3, 6)
        diff = original[0] - reconstructed[0]
        plt.imshow(diff, cmap='coolwarm')
        plt.title('差异 (原始 - 重建)')
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def main(file_path, n_components=None, auto_select=False, variance_threshold=0.95):
    """主函数：加载数据，应用SVD，重建和可视化"""
    print(f"正在处理文件: {file_path}")
    
    # 加载数据
    data, metadata = load_data(file_path)
    print(f"数据形状: {data.shape}")
    
    # 应用SVD
    U, Sigma, Vt, original_shape = apply_svd(data)
    
    # 如果启用自动选择组件数
    if auto_select:
        # 基于方差阈值选择组件数
        cum_variance = np.cumsum(Sigma**2) / np.sum(Sigma**2)
        n_components = np.argmax(cum_variance >= variance_threshold) + 1
        print(f"基于{variance_threshold:.2%}方差阈值，自动选择组件数: {n_components}")
    elif n_components is None:
        # 默认使用10%的组件
        n_components = max(1, int(len(Sigma) * 0.1))
        print(f"使用默认组件数: {n_components} (总共{len(Sigma)}个)")
    
    # 重建数据
    reconstructed = reconstruct_from_svd(U, Sigma, Vt, original_shape, n_components)
    
    # 计算压缩比和解释方差
    compression_ratio, explained_variance = calculate_compression_ratio(data, n_components, Sigma)
    
    print(f"压缩比: {compression_ratio:.2f}x (原始大小的{1/compression_ratio:.2%})")
    print(f"保留信息: {explained_variance:.2%}")
    print(f"使用的组件数: {n_components} / {len(Sigma)}")
    
    # 可视化结果
    visualize_results(data, reconstructed, Sigma, n_components, explained_variance)
    
    return U, Sigma, Vt, reconstructed, metadata

if __name__ == "__main__":
    # 示例使用方法
    # 如果你有实际的NC或TIF文件，请修改此路径
    # file_path = "path/to/your/satellite_data.nc"
    
    # 生成模拟数据用于演示
    np.random.seed(42)
    # 创建10个波段，每个100x100像素的模拟卫星数据
    bands = 10
    height, width = 100, 100
    
    # 创建具有一些结构而非纯随机的数据
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    # 在不同波段中添加不同的模式
    simulated_data = np.zeros((bands, height, width))
    
    for i in range(bands):
        # 创建具有空间相关性的基本模式
        base = np.sin(x * (i+5) * np.pi) * np.cos(y * (i+3) * np.pi)
        # 添加一些噪声
        noise = np.random.normal(0, 0.1, (height, width))
        simulated_data[i] = base + noise
    
    # 保存为临时文件
    tmp_file = "simulated_satellite_data.nc"
    
    # 创建xarray数据集并保存
    ds = xr.Dataset(
        data_vars={
            "reflectance": (["band", "y", "x"], simulated_data),
        },
        coords={
            "band": np.arange(1, bands+1),
            "y": np.arange(height),
            "x": np.arange(width),
        }
    )
    ds.to_netcdf(tmp_file)
    
    # 运行SVD分析
    # 自动选择组件数，使得保留95%的方差
    U, Sigma, Vt, reconstructed, metadata = main(tmp_file, auto_select=True, variance_threshold=0.95)
    
    # 清理临时文件
    os.remove(tmp_file)
