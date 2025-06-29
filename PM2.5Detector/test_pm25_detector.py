#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PM2.5检测器测试脚本
用于验证检测器的功能和参数调优
"""

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from medical_pm25_detector import MedicalPM25Detector
from simple_pm25_detector import simple_pm25_detector

def create_test_image(output_path: str = "test_image.jpg"):
    """
    创建测试图像，包含模拟的PM2.5黑色区域
    
    Args:
        output_path: 输出图像路径
    """
    # 创建白色背景图像
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # 添加一些模拟的PM2.5黑色区域
    # 大区域
    cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 0), -1)
    
    # 中等区域
    cv2.circle(image, (300, 100), 40, (0, 0, 0), -1)
    
    # 小区域
    cv2.ellipse(image, (500, 80), (20, 15), 0, 0, 360, (0, 0, 0), -1)
    
    # 不规则形状
    points = np.array([[100, 250], [150, 200], [200, 250], [180, 300], [120, 300]], np.int32)
    cv2.fillPoly(image, [points], (0, 0, 0))
    
    # 添加一些噪声（小的黑色点）
    for i in range(10):
        x = np.random.randint(0, 600)
        y = np.random.randint(0, 400)
        cv2.circle(image, (x, y), 2, (0, 0, 0), -1)
    
    # 保存测试图像
    cv2.imwrite(output_path, image)
    print(f"测试图像已创建: {output_path}")
    
    return output_path

def test_single_image():
    """测试单张图像处理"""
    print("=== 单张图像测试 ===")
    
    # 创建测试图像
    test_image_path = create_test_image()
    
    # 创建检测器
    detector = MedicalPM25Detector(
        black_threshold=50,
        min_area=50,
        output_dir="test_output"
    )
    
    # 处理图像
    result = detector.process_single_image(test_image_path)
    
    print(f"处理结果: {result}")
    
    return result

def test_batch_processing():
    """测试批量处理"""
    print("\n=== 批量处理测试 ===")
    
    # 创建测试目录
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # 创建多个测试图像
    for i in range(3):
        create_test_image(str(test_dir / f"test_image_{i+1}.jpg"))
    
    # 使用简化版检测器
    simple_pm25_detector(str(test_dir), "test_batch_output")
    
    print("批量处理测试完成")

def test_parameter_tuning():
    """测试参数调优"""
    print("\n=== 参数调优测试 ===")
    
    test_image_path = create_test_image("tuning_test.jpg")
    
    # 测试不同的阈值
    thresholds = [30, 50, 70]
    areas = [50, 100, 200]
    
    results = []
    
    for thresh in thresholds:
        for area in areas:
            detector = MedicalPM25Detector(
                black_threshold=thresh,
                min_area=area,
                output_dir=f"tuning_output_{thresh}_{area}"
            )
            
            result = detector.process_single_image(test_image_path)
            results.append({
                'threshold': thresh,
                'min_area': area,
                'regions': result['pm25_regions'],
                'total_area': result['total_area']
            })
            
            print(f"阈值: {thresh}, 最小面积: {area}, 检测区域数: {result['pm25_regions']}")
    
    return results

def visualize_test_results():
    """可视化测试结果"""
    print("\n=== 可视化测试结果 ===")
    
    # 检查是否有测试输出
    test_output = Path("test_output/annotated")
    if test_output.exists():
        annotated_files = list(test_output.glob("annotated_*"))
        if annotated_files:
            # 显示第一个结果
            original_path = "test_image.jpg"
            annotated_path = str(annotated_files[0])
            mask_path = str(annotated_files[0]).replace("annotated_", "mask_")
            
            if Path(mask_path).exists():
                from simple_pm25_detector import visualize_results
                visualize_results(original_path, annotated_path, mask_path)
                print("可视化结果已显示")
            else:
                print("未找到掩码文件")
        else:
            print("未找到标注文件")
    else:
        print("未找到测试输出目录")

def performance_test():
    """性能测试"""
    print("\n=== 性能测试 ===")
    
    import time
    
    # 创建大尺寸测试图像
    large_image = np.ones((1000, 1500, 3), dtype=np.uint8) * 255
    
    # 添加多个PM2.5区域
    for i in range(20):
        x = np.random.randint(0, 1500)
        y = np.random.randint(0, 1000)
        radius = np.random.randint(10, 50)
        cv2.circle(large_image, (x, y), radius, (0, 0, 0), -1)
    
    cv2.imwrite("large_test_image.jpg", large_image)
    
    # 测试处理时间
    detector = MedicalPM25Detector()
    
    start_time = time.time()
    result = detector.process_single_image("large_test_image.jpg")
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    print(f"大图像处理时间: {processing_time:.2f} 秒")
    print(f"检测到 {result['pm25_regions']} 个PM2.5区域")
    print(f"总面积: {result['total_area']} 像素")

def cleanup_test_files():
    """清理测试文件"""
    print("\n=== 清理测试文件 ===")
    
    test_files = [
        "test_image.jpg",
        "tuning_test.jpg",
        "large_test_image.jpg"
    ]
    
    test_dirs = [
        "test_images",
        "test_output",
        "test_batch_output",
        "tuning_output_30_50",
        "tuning_output_30_100",
        "tuning_output_30_200",
        "tuning_output_50_50",
        "tuning_output_50_100",
        "tuning_output_50_200",
        "tuning_output_70_50",
        "tuning_output_70_100",
        "tuning_output_70_200"
    ]
    
    # 删除测试文件
    for file_path in test_files:
        if Path(file_path).exists():
            Path(file_path).unlink()
            print(f"已删除: {file_path}")
    
    # 删除测试目录
    for dir_path in test_dirs:
        if Path(dir_path).exists():
            import shutil
            shutil.rmtree(dir_path)
            print(f"已删除目录: {dir_path}")

def main():
    """主测试函数"""
    print("PM2.5检测器测试开始...")
    
    try:
        # 运行各种测试
        test_single_image()
        test_batch_processing()
        test_parameter_tuning()
        performance_test()
        visualize_test_results()
        
        print("\n所有测试完成!")
        
        # 询问是否清理测试文件
        response = input("\n是否清理测试文件? (y/n): ")
        if response.lower() == 'y':
            cleanup_test_files()
        else:
            print("测试文件保留在磁盘上")
            
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 