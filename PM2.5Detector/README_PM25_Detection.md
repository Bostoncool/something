# 医疗影像PM2.5区域检测系统

基于Python和OpenCV开发的医疗影像PM2.5区域自动检测系统，专门用于识别病理切片图中的黑色PM2.5区域。

## 功能特点

- 🔍 **批量处理**: 自动处理文件夹中的所有图像文件
- 🎯 **精确检测**: 专门针对黑色PM2.5区域进行识别
- 📊 **统计分析**: 提供详细的检测统计信息和报告
- 🖼️ **可视化标注**: 在图像上标记检测到的PM2.5区域
- 📁 **多格式支持**: 支持JPG、PNG、TIFF、BMP等常见图像格式
- ⚙️ **参数可调**: 可根据不同图像特点调整检测参数

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 方法一：使用完整版检测器（推荐）

```bash
python medical_pm25_detector.py 图像目录路径 [选项]
```

**参数说明:**
- `图像目录路径`: 包含医疗影像图片的文件夹路径
- `--output_dir`: 输出目录（默认: output）
- `--black_threshold`: 黑色像素阈值 0-255（默认: 50）
- `--min_area`: 最小区域面积（默认: 100）
- `--no_annotated`: 不保存标注后的图像
- `--no_stats`: 不保存统计信息

**使用示例:**
```bash
# 基本使用
python medical_pm25_detector.py medical_images

# 自定义参数
python medical_pm25_detector.py medical_images --black_threshold 60 --min_area 200 --output_dir results

# 只生成统计信息，不保存标注图像
python medical_pm25_detector.py medical_images --no_annotated
```

### 方法二：使用简化版检测器

```python
from simple_pm25_detector import simple_pm25_detector

# 处理图像
simple_pm25_detector("medical_images", "output_results")
```

### 方法三：在代码中使用

```python
from medical_pm25_detector import MedicalPM25Detector

# 创建检测器
detector = MedicalPM25Detector(
    black_threshold=50,    # 黑色像素阈值
    min_area=100,         # 最小区域面积
    output_dir="results"  # 输出目录
)

# 批量处理
results = detector.batch_process("medical_images")

# 查看统计信息
print(f"处理了 {results['statistics']['total_images']} 张图像")
print(f"检测到总PM2.5面积: {results['statistics']['total_pm25_area']} 像素")
```

## 输出结果

处理完成后，系统会在输出目录中生成以下文件：

### 1. 标注图像 (`annotated/` 目录)
- `annotated_原文件名.jpg`: 带有PM2.5区域标注的原始图像
- `mask_原文件名.jpg`: PM2.5区域的二值化掩码

### 2. 统计报告
- `processing_statistics.json`: 详细的JSON格式统计信息
- `processing_report.txt`: 人类可读的处理报告

### 3. 标注说明
- 绿色边界框: 标记检测到的PM2.5区域
- 绿色标签: PM2.5区域编号
- 蓝色文字: 每个区域的面积信息
- 红色文字: 总体统计信息

## 参数调优指南

### 黑色像素阈值 (`black_threshold`)
- **值范围**: 0-255
- **默认值**: 50
- **调优建议**:
  - 如果检测到太多非PM2.5区域，**增加**阈值
  - 如果漏检PM2.5区域，**减少**阈值
  - 建议范围: 30-80

### 最小区域面积 (`min_area`)
- **默认值**: 100
- **调优建议**:
  - 如果检测到太多噪声小区域，**增加**面积阈值
  - 如果漏检小的PM2.5区域，**减少**面积阈值
  - 建议范围: 50-500

## 图像预处理流程

1. **灰度转换**: 将彩色图像转换为灰度图
2. **高斯模糊**: 去除图像噪声
3. **直方图均衡化**: 增强图像对比度
4. **阈值分割**: 识别黑色区域
5. **形态学操作**: 去除噪声，连接断开的区域
6. **轮廓检测**: 提取PM2.5区域边界
7. **面积过滤**: 去除过小的区域

## 支持的图像格式

- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)
- BMP (.bmp)

## 性能优化建议

1. **批量处理**: 系统已优化批量处理性能
2. **内存管理**: 处理大图像时注意内存使用
3. **并行处理**: 对于大量图像，可考虑多进程处理

## 故障排除

### 常见问题

1. **无法读取图像**
   - 检查图像文件是否损坏
   - 确认图像格式是否支持
   - 检查文件路径是否正确

2. **检测结果不准确**
   - 调整 `black_threshold` 参数
   - 调整 `min_area` 参数
   - 检查图像质量和对比度

3. **处理速度慢**
   - 减少图像分辨率
   - 调整形态学操作参数
   - 使用SSD存储提高I/O性能

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 处理单张图像进行调试
detector = MedicalPM25Detector()
result = detector.process_single_image("test_image.jpg")
print(result)
```

## 技术原理

### PM2.5区域检测算法

1. **颜色空间分析**: 在灰度空间中识别黑色区域
2. **阈值分割**: 使用自适应阈值进行二值化
3. **形态学处理**: 使用开运算和闭运算优化区域形状
4. **轮廓分析**: 提取区域边界和几何特征
5. **面积筛选**: 基于面积阈值过滤噪声

### 图像增强技术

- **高斯滤波**: 减少图像噪声
- **直方图均衡化**: 增强图像对比度
- **形态学操作**: 优化区域形状和连接性

## 扩展功能

### 自定义检测器

```python
class CustomPM25Detector(MedicalPM25Detector):
    def custom_preprocessing(self, image):
        # 添加自定义预处理步骤
        return processed_image
    
    def custom_detection(self, image):
        # 添加自定义检测算法
        return contours
```

### 批量参数优化

```python
# 测试不同参数组合
thresholds = [30, 50, 70]
areas = [50, 100, 200]

for thresh in thresholds:
    for area in areas:
        detector = MedicalPM25Detector(black_threshold=thresh, min_area=area)
        results = detector.batch_process("test_images")
        print(f"Threshold: {thresh}, Area: {area}, Regions: {results['statistics']['processed_images']}")
```

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 联系方式

如有问题或建议，请通过GitHub Issues联系。 