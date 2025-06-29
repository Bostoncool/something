import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

class CellClassifier(nn.Module):
    def __init__(self):
        super(CellClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 5)  # 假设有5种细胞类型
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def detect_lesion_area(image):
    """检测病灶区域并返回圆心和半径"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # 使用霍夫圆变换检测圆形
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=30,
        maxRadius=100
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0][0]  # 返回第一个检测到的圆
    return None

def process_image(image_path):
    """处理图像并统计细胞"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像")
    
    # 检测病灶区域
    circle = detect_lesion_area(image)
    if circle is None:
        raise ValueError("未检测到病灶区域")
    
    x, y, r = circle
    
    # 创建掩码
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    
    # 提取圆形区域
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # 初始化模型
    model = CellClassifier()
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 统计细胞
    cell_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # 假设有5种细胞类型
    
    # 在圆形区域内进行滑动窗口检测
    window_size = 224
    stride = 112
    
    for i in range(0, masked_image.shape[0] - window_size, stride):
        for j in range(0, masked_image.shape[1] - window_size, stride):
            if mask[i + window_size//2, j + window_size//2] > 0:
                window = masked_image[i:i+window_size, j:j+window_size]
                window_tensor = transform(window).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(window_tensor)
                    cell_type = torch.argmax(output).item()
                    cell_counts[cell_type] += 1
    
    # 绘制结果
    result_image = image.copy()
    cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)
    
    # 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('检测结果')
    
    # 打印统计结果
    print("\n细胞统计结果：")
    for cell_type, count in cell_counts.items():
        print(f"细胞类型 {cell_type}: {count}个")
    
    plt.show()

if __name__ == "__main__":
    # 使用示例
    image_path = "path_to_your_image.jpg"  # 替换为实际的图像路径
    try:
        process_image(image_path)
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")