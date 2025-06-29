import cv2
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import argparse
import json
from datetime import datetime

class MedicalPM25Detector:
    """
    医疗影像PM2.5区域检测器
    专门用于识别病理切片图中的黑色PM2.5区域
    """
    
    def __init__(self, 
                 black_threshold: int = 50,
                 min_area: int = 100,
                 output_dir: str = "output",
                 save_annotated: bool = True,
                 save_stats: bool = True):
        """
        初始化检测器
        
        Args:
            black_threshold: 黑色像素阈值 (0-255)
            min_area: 最小区域面积
            output_dir: 输出目录
            save_annotated: 是否保存标注后的图片
            save_stats: 是否保存统计信息
        """
        self.black_threshold = black_threshold
        self.min_area = min_area
        self.output_dir = Path(output_dir)
        self.save_annotated = save_annotated
        self.save_stats = save_stats
        
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)
        self.annotated_dir = self.output_dir / "annotated"
        self.annotated_dir.mkdir(exist_ok=True)
        
        # 统计信息
        self.stats = {
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "total_pm25_area": 0,
            "average_pm25_area": 0,
            "processing_time": 0
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 直方图均衡化增强对比度
        equalized = cv2.equalizeHist(blurred)
        
        return equalized
    
    def detect_black_regions(self, image: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        检测黑色区域
        
        Args:
            image: 预处理后的图像
            
        Returns:
            二值化图像和轮廓列表
        """
        # 创建黑色区域掩码
        # 黑色区域像素值小于阈值
        _, black_mask = cv2.threshold(image, self.black_threshold, 255, cv2.THRESH_BINARY_INV)
        
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
            if area >= self.min_area:
                filtered_contours.append(contour)
        
        return black_mask, filtered_contours
    
    def annotate_image(self, image: np.ndarray, contours: List, 
                      filename: str) -> np.ndarray:
        """
        在图像上标注检测到的PM2.5区域
        
        Args:
            image: 原始图像
            contours: 检测到的轮廓
            filename: 文件名
            
        Returns:
            标注后的图像
        """
        annotated = image.copy()
        
        # 为每个轮廓绘制边界框和标签
        for i, contour in enumerate(contours):
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # 绘制边界框
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 添加标签
            label = f"PM2.5-{i+1}"
            cv2.putText(annotated, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 在图像上显示面积信息
            area_text = f"Area: {area:.0f}px"
            cv2.putText(annotated, area_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # 添加总体信息
        total_area = sum(cv2.contourArea(c) for c in contours)
        info_text = f"Total PM2.5 regions: {len(contours)}, Total area: {total_area:.0f}px"
        cv2.putText(annotated, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated
    
    def process_single_image(self, image_path: str) -> dict:
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            处理结果字典
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 预处理
            processed = self.preprocess_image(image)
            
            # 检测黑色区域
            black_mask, contours = self.detect_black_regions(processed)
            
            # 计算统计信息
            total_area = sum(cv2.contourArea(c) for c in contours)
            
            # 标注图像
            if self.save_annotated:
                annotated = self.annotate_image(image, contours, Path(image_path).name)
                
                # 保存标注后的图像
                output_path = self.annotated_dir / f"annotated_{Path(image_path).name}"
                cv2.imwrite(str(output_path), annotated)
                
                # 保存掩码图像
                mask_path = self.annotated_dir / f"mask_{Path(image_path).name}"
                cv2.imwrite(str(mask_path), black_mask)
            
            return {
                "filename": Path(image_path).name,
                "status": "success",
                "pm25_regions": len(contours),
                "total_area": total_area,
                "average_area": total_area / len(contours) if contours else 0,
                "contours": [cv2.contourArea(c) for c in contours]
            }
            
        except Exception as e:
            return {
                "filename": Path(image_path).name,
                "status": "failed",
                "error": str(e),
                "pm25_regions": 0,
                "total_area": 0,
                "average_area": 0,
                "contours": []
            }
    
    def batch_process(self, input_dir: str, 
                     extensions: List[str] = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']) -> dict:
        """
        批量处理文件夹中的图像
        
        Args:
            input_dir: 输入目录
            extensions: 支持的图像扩展名
            
        Returns:
            批量处理结果
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"输入目录不存在: {input_dir}")
        
        # 查找所有图像文件
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(str(input_path / f"*{ext}")))
            image_files.extend(glob.glob(str(input_path / f"*{ext.upper()}")))
        
        if not image_files:
            raise ValueError(f"在目录 {input_dir} 中未找到图像文件")
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 批量处理
        results = []
        start_time = datetime.now()
        
        for i, image_path in enumerate(image_files, 1):
            print(f"处理图像 {i}/{len(image_files)}: {Path(image_path).name}")
            result = self.process_single_image(image_path)
            results.append(result)
            
            if result["status"] == "success":
                print(f"  - 检测到 {result['pm25_regions']} 个PM2.5区域")
                print(f"  - 总面积: {result['total_area']:.0f} 像素")
            else:
                print(f"  - 处理失败: {result['error']}")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # 更新统计信息
        self.stats.update({
            "total_images": len(image_files),
            "processed_images": sum(1 for r in results if r["status"] == "success"),
            "failed_images": sum(1 for r in results if r["status"] == "failed"),
            "total_pm25_area": sum(r["total_area"] for r in results if r["status"] == "success"),
            "processing_time": processing_time
        })
        
        if self.stats["processed_images"] > 0:
            self.stats["average_pm25_area"] = self.stats["total_pm25_area"] / self.stats["processed_images"]
        
        # 保存统计信息
        if self.save_stats:
            self.save_statistics(results)
        
        return {
            "statistics": self.stats,
            "results": results
        }
    
    def save_statistics(self, results: List[dict]):
        """保存统计信息到JSON文件"""
        stats_file = self.output_dir / "processing_statistics.json"
        
        # 准备保存的数据
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "black_threshold": self.black_threshold,
                "min_area": self.min_area
            },
            "statistics": self.stats,
            "detailed_results": results
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"统计信息已保存到: {stats_file}")
    
    def generate_report(self, results: List[dict]):
        """生成处理报告"""
        report_file = self.output_dir / "processing_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("医疗影像PM2.5区域检测报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总图像数: {self.stats['total_images']}\n")
            f.write(f"成功处理: {self.stats['processed_images']}\n")
            f.write(f"处理失败: {self.stats['failed_images']}\n")
            f.write(f"总PM2.5面积: {self.stats['total_pm25_area']:.0f} 像素\n")
            f.write(f"平均PM2.5面积: {self.stats['average_pm25_area']:.0f} 像素\n")
            f.write(f"处理时间: {self.stats['processing_time']:.2f} 秒\n\n")
            
            f.write("详细结果:\n")
            f.write("-" * 30 + "\n")
            
            for result in results:
                f.write(f"文件: {result['filename']}\n")
                f.write(f"状态: {result['status']}\n")
                if result['status'] == 'success':
                    f.write(f"PM2.5区域数: {result['pm25_regions']}\n")
                    f.write(f"总面积: {result['total_area']:.0f} 像素\n")
                    f.write(f"平均面积: {result['average_area']:.0f} 像素\n")
                else:
                    f.write(f"错误: {result['error']}\n")
                f.write("\n")
        
        print(f"处理报告已保存到: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="医疗影像PM2.5区域检测器")
    parser.add_argument("input_dir", help="输入图像目录路径")
    parser.add_argument("--output_dir", default="output", help="输出目录")
    parser.add_argument("--black_threshold", type=int, default=50, 
                       help="黑色像素阈值 (0-255)")
    parser.add_argument("--min_area", type=int, default=100, 
                       help="最小区域面积")
    parser.add_argument("--no_annotated", action="store_true", 
                       help="不保存标注后的图像")
    parser.add_argument("--no_stats", action="store_true", 
                       help="不保存统计信息")
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = MedicalPM25Detector(
        black_threshold=args.black_threshold,
        min_area=args.min_area,
        output_dir=args.output_dir,
        save_annotated=not args.no_annotated,
        save_stats=not args.no_stats
    )
    
    try:
        # 批量处理
        results = detector.batch_process(args.input_dir)
        
        # 生成报告
        detector.generate_report(results["results"])
        
        print("\n处理完成!")
        print(f"输出目录: {args.output_dir}")
        
    except Exception as e:
        print(f"处理失败: {e}")


if __name__ == "__main__":
    main() 