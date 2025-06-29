import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def simple_pm25_detector(input_dir: str, output_dir: str = "output"):
    """
    简化的PM2.5区域检测器
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    
    # 查找所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"在目录 {input_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理参数
    black_threshold = 50  # 黑色像素阈值
    min_area = 100       # 最小区域面积
    
    for i, image_path in enumerate(image_files, 1):
        print(f"处理图像 {i}/{len(image_files)}: {image_path.name}")
        
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  无法读取图像: {image_path}")
                continue
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊去噪
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 检测黑色区域
            _, black_mask = cv2.threshold(blurred, black_threshold, 255, cv2.THRESH_BINARY_INV)
            
            # 形态学操作去除噪声
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
            black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤小面积区域
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    filtered_contours.append(contour)
            
            # 标注图像
            annotated = image.copy()
            total_area = 0
            
            for j, contour in enumerate(filtered_contours):
                # 计算边界框
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                total_area += area
                
                # 绘制边界框
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 添加标签
                label = f"PM2.5-{j+1}"
                cv2.putText(annotated, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 添加总体信息
            info_text = f"PM2.5 regions: {len(filtered_contours)}, Total area: {total_area:.0f}px"
            cv2.putText(annotated, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 保存结果
            annotated_path = output_path / f"annotated_{image_path.name}"
            mask_path = output_path / f"mask_{image_path.name}"
            
            cv2.imwrite(str(annotated_path), annotated)
            cv2.imwrite(str(mask_path), black_mask)
            
            print(f"  检测到 {len(filtered_contours)} 个PM2.5区域")
            print(f"  总面积: {total_area:.0f} 像素")
            print(f"  已保存: {annotated_path}")
            
        except Exception as e:
            print(f"  处理失败: {e}")
    
    print(f"\n处理完成! 结果保存在: {output_dir}")


def visualize_results(image_path: str, annotated_path: str, mask_path: str):
    """
    可视化检测结果
    
    Args:
        image_path: 原始图像路径
        annotated_path: 标注图像路径
        mask_path: 掩码图像路径
    """
    # 读取图像
    original = cv2.imread(image_path)
    annotated = cv2.imread(annotated_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 转换颜色空间用于显示
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示原始图像
    axes[0].imshow(original_rgb)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示标注图像
    axes[1].imshow(annotated_rgb)
    axes[1].set_title('PM2.5区域标注')
    axes[1].axis('off')
    
    # 显示掩码
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('PM2.5区域掩码')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 使用示例
    input_directory = "medical_images"  # 替换为您的图像目录
    output_directory = "pm25_results"
    
    # 检查输入目录是否存在
    if not Path(input_directory).exists():
        print(f"输入目录 {input_directory} 不存在，请创建目录并放入图像文件")
        print("或者修改代码中的 input_directory 变量")
    else:
        # 运行检测器
        simple_pm25_detector(input_directory, output_directory)
        
        # 如果有结果，可以选择可视化第一个结果
        output_path = Path(output_directory)
        annotated_files = list(output_path.glob("annotated_*"))
        
        if annotated_files:
            print(f"\n找到 {len(annotated_files)} 个处理结果")
            print("要可视化结果，请取消注释以下代码:")
            print(f"# visualize_results('{input_directory}/image.jpg', '{annotated_files[0]}', '{output_directory}/mask_image.jpg')") 