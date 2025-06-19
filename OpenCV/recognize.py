import cv2
import numpy as np

# 加载图像（请将图片路径替换为你的图片路径）
image_path = 'your_image.jpg'  # 替换为你的图片路径
image = cv2.imread(image_path)

if image is None:
    print("无法加载图像，请检查路径是否正确！")
    exit()

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用阈值化或边缘检测来找到轮廓
# 这里使用 Canny 边缘检测
edges = cv2.Canny(gray, 100, 200)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历轮廓
for contour in contours:
    # 计算轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)
    
    # 绘制边界框
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 计算边界框的重心
    center_x = x + w // 2
    center_y = y + h // 2
    
    # 绘制重心点
    cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
    
    # 绘制以重心为圆心的不同半径的圆
    radii = [10, 20, 30, 40]  # 定义不同半径，step，像素和距离的转化
    for radius in radii:
        cv2.circle(image, (center_x, center_y), radius, (255, 0, 0), 2)

# 显示结果
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()