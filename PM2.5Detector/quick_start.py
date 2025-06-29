#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PM2.5检测器快速启动脚本
提供简单的交互式界面来使用检测器
"""

import os
import sys
from pathlib import Path
from medical_pm25_detector import MedicalPM25Detector
from simple_pm25_detector import simple_pm25_detector

def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("🏥 医疗影像PM2.5区域检测系统")
    print("=" * 60)
    print("基于Python和OpenCV的智能检测工具")
    print("专门用于识别病理切片图中的黑色PM2.5区域")
    print("=" * 60)

def get_input_directory():
    """获取输入目录"""
    while True:
        input_dir = input("请输入图像目录路径 (或按回车使用默认路径 'medical_images'): ").strip()
        
        if not input_dir:
            input_dir = "medical_images"
        
        if Path(input_dir).exists():
            return input_dir
        else:
            print(f"❌ 目录 '{input_dir}' 不存在")
            create_dir = input("是否创建此目录? (y/n): ").strip().lower()
            if create_dir == 'y':
                Path(input_dir).mkdir(exist_ok=True)
                print(f"✅ 已创建目录: {input_dir}")
                print(f"请将图像文件放入 {input_dir} 目录中，然后重新运行程序")
                return None
            else:
                continue

def get_parameters():
    """获取处理参数"""
    print("\n📋 参数设置:")
    print("-" * 30)
    
    # 黑色像素阈值
    while True:
        try:
            threshold = input("黑色像素阈值 (0-255, 默认50): ").strip()
            if not threshold:
                threshold = 50
            else:
                threshold = int(threshold)
                if 0 <= threshold <= 255:
                    break
                else:
                    print("❌ 阈值必须在0-255之间")
        except ValueError:
            print("❌ 请输入有效的数字")
    
    # 最小区域面积
    while True:
        try:
            min_area = input("最小区域面积 (默认100): ").strip()
            if not min_area:
                min_area = 100
            else:
                min_area = int(min_area)
                if min_area > 0:
                    break
                else:
                    print("❌ 面积必须大于0")
        except ValueError:
            print("❌ 请输入有效的数字")
    
    # 输出目录
    output_dir = input("输出目录 (默认 'pm25_results'): ").strip()
    if not output_dir:
        output_dir = "pm25_results"
    
    return threshold, min_area, output_dir

def select_mode():
    """选择处理模式"""
    print("\n🔧 选择处理模式:")
    print("1. 完整版检测器 (推荐) - 包含详细统计和报告")
    print("2. 简化版检测器 - 快速处理，基本功能")
    print("3. 退出")
    
    while True:
        choice = input("请选择 (1-3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        else:
            print("❌ 请输入1、2或3")

def run_full_detector(input_dir, threshold, min_area, output_dir):
    """运行完整版检测器"""
    print(f"\n🚀 启动完整版检测器...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"黑色阈值: {threshold}")
    print(f"最小面积: {min_area}")
    
    try:
        # 创建检测器
        detector = MedicalPM25Detector(
            black_threshold=threshold,
            min_area=min_area,
            output_dir=output_dir,
            save_annotated=True,
            save_stats=True
        )
        
        # 批量处理
        results = detector.batch_process(input_dir)
        
        # 生成报告
        detector.generate_report(results["results"])
        
        # 显示结果摘要
        stats = results["statistics"]
        print(f"\n✅ 处理完成!")
        print(f"📊 处理统计:")
        print(f"   - 总图像数: {stats['total_images']}")
        print(f"   - 成功处理: {stats['processed_images']}")
        print(f"   - 处理失败: {stats['failed_images']}")
        print(f"   - 总PM2.5面积: {stats['total_pm25_area']:.0f} 像素")
        print(f"   - 平均PM2.5面积: {stats['average_pm25_area']:.0f} 像素")
        print(f"   - 处理时间: {stats['processing_time']:.2f} 秒")
        
        print(f"\n📁 结果文件:")
        print(f"   - 标注图像: {output_dir}/annotated/")
        print(f"   - 统计报告: {output_dir}/processing_report.txt")
        print(f"   - 详细数据: {output_dir}/processing_statistics.json")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return False

def run_simple_detector(input_dir, output_dir):
    """运行简化版检测器"""
    print(f"\n🚀 启动简化版检测器...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    try:
        simple_pm25_detector(input_dir, output_dir)
        print(f"\n✅ 处理完成!")
        print(f"📁 结果保存在: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return False

def show_help():
    """显示帮助信息"""
    print("\n📖 使用帮助:")
    print("-" * 30)
    print("1. 准备图像文件:")
    print("   - 支持的格式: JPG, PNG, TIFF, BMP")
    print("   - 将图像文件放入指定目录")
    print("   - 确保图像清晰，对比度良好")
    
    print("\n2. 参数说明:")
    print("   - 黑色像素阈值: 识别黑色区域的标准 (0-255)")
    print("   - 最小区域面积: 过滤小噪声区域")
    
    print("\n3. 输出结果:")
    print("   - 标注图像: 在原图上标记PM2.5区域")
    print("   - 掩码图像: 二值化的PM2.5区域")
    print("   - 统计报告: 详细的处理统计信息")
    
    print("\n4. 参数调优:")
    print("   - 检测过多区域: 增加阈值或最小面积")
    print("   - 漏检区域: 减少阈值或最小面积")
    print("   - 建议阈值范围: 30-80")
    print("   - 建议面积范围: 50-500")

def main():
    """主函数"""
    print_banner()
    
    while True:
        # 选择模式
        mode = select_mode()
        
        if mode == '3':
            print("👋 感谢使用PM2.5检测器!")
            break
        
        # 获取输入目录
        input_dir = get_input_directory()
        if input_dir is None:
            continue
        
        # 获取参数
        threshold, min_area, output_dir = get_parameters()
        
        # 运行检测器
        success = False
        if mode == '1':
            success = run_full_detector(input_dir, threshold, min_area, output_dir)
        elif mode == '2':
            success = run_simple_detector(input_dir, output_dir)
        
        if success:
            # 询问是否继续
            continue_choice = input("\n是否继续处理其他图像? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("👋 感谢使用PM2.5检测器!")
                break
        else:
            # 询问是否重试
            retry_choice = input("\n是否重试? (y/n): ").strip().lower()
            if retry_choice != 'y':
                print("👋 感谢使用PM2.5检测器!")
                break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序已中断，感谢使用!")
    except Exception as e:
        print(f"\n❌ 程序出现错误: {e}")
        print("如需帮助，请运行: python quick_start.py --help") 