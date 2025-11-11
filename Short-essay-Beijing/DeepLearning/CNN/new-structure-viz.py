"""
CNN 神经网络架构可视化脚本
使用 Graphviz 生成神经网络结构图
"""

from graphviz import Digraph

def create_cnn_architecture():
    """创建 CNN 架构的可视化图表"""
    # 创建一个新的有向图
    dot = Digraph(comment='CNN Architecture for PM2.5 Prediction')
    dot.attr(rankdir='LR')  # 从左到右布局
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
    
    # 输入层
    dot.node('input', 'Input Layer\n(batch, 1, window, num_features)', fillcolor='lightblue')
    
    # Conv2D Layer 1 + BatchNorm + ReLU + MaxPool
    dot.node('conv1', 'Conv2D Layer 1\n(batch, 32, window, num_features)', fillcolor='lightcoral')
    dot.node('bn1', 'BatchNorm + ReLU', fillcolor='lightyellow')
    dot.node('pool1', 'MaxPool2D\n(batch, 32, window/2, num_features/2)', fillcolor='lightgreen')
    
    # Conv2D Layer 2 + BatchNorm + ReLU + MaxPool
    dot.node('conv2', 'Conv2D Layer 2\n(batch, 64, window/2, num_features/2)', fillcolor='lightcoral')
    dot.node('bn2', 'BatchNorm + ReLU', fillcolor='lightyellow')
    dot.node('pool2', 'MaxPool2D\n(batch, 64, window/4, num_features/4)', fillcolor='lightgreen')
    
    # Conv2D Layer 3 + BatchNorm + ReLU + MaxPool
    dot.node('conv3', 'Conv2D Layer 3\n(batch, 128, window/4, num_features/4)', fillcolor='lightcoral')
    dot.node('bn3', 'BatchNorm + ReLU', fillcolor='lightyellow')
    dot.node('pool3', 'MaxPool2D\n(batch, 128, window/8, num_features/8)', fillcolor='lightgreen')
    
    # Flatten
    dot.node('flatten', 'Flatten\n(batch, flattened_size)', fillcolor='lavender')
    
    # Fully Connected Layer 1 + BatchNorm + ReLU + Dropout
    dot.node('fc1', 'Fully Connected Layer 1\n(batch, 256)', fillcolor='lightpink')
    dot.node('bn4', 'BatchNorm + ReLU', fillcolor='lightyellow')
    dot.node('drop1', 'Dropout', fillcolor='lightgray')
    
    # Fully Connected Layer 2 + BatchNorm + ReLU + Dropout
    dot.node('fc2', 'Fully Connected Layer 2\n(batch, 64)', fillcolor='lightpink')
    dot.node('bn5', 'BatchNorm + ReLU', fillcolor='lightyellow')
    dot.node('drop2', 'Dropout', fillcolor='lightgray')
    
    # Output Layer
    dot.node('output', 'Output Layer\n(batch, 1)', fillcolor='orange')
    
    # 连接节点
    edges = [
        ('input', 'conv1'),
        ('conv1', 'bn1'),
        ('bn1', 'pool1'),
        ('pool1', 'conv2'),
        ('conv2', 'bn2'),
        ('bn2', 'pool2'),
        ('pool2', 'conv3'),
        ('conv3', 'bn3'),
        ('bn3', 'pool3'),
        ('pool3', 'flatten'),
        ('flatten', 'fc1'),
        ('fc1', 'bn4'),
        ('bn4', 'drop1'),
        ('drop1', 'fc2'),
        ('fc2', 'bn5'),
        ('bn5', 'drop2'),
        ('drop2', 'output')
    ]
    
    for start, end in edges:
        dot.edge(start, end)
    
    return dot

def main():
    """主函数"""
    # 创建架构图
    dot = create_cnn_architecture()
    
    # 生成输出文件
    output_name = 'cnn_architecture'
    dot.render(output_name, format='png', cleanup=True)
    
    print(f"✓ 神经网络架构图已生成: {output_name}.png")
    print(f"✓ 图表包含完整的 CNN 架构：3个卷积块 + 2个全连接层")

if __name__ == '__main__':
    main()
