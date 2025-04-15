import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
 
# 解决中文显示问题的代码
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
mpl.rcParams['axes.unicode_minus'] = False
# 设置全局字体大小和图形尺寸
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = (10, 6)
 
warnings.filterwarnings('ignore')
 
"""
ax:画板
extent:调整宽高比例
direction:颜色渐变方向
cmap_range:颜色渐变的范围
**kwargs:imshow中的参数
"""
 
 
def gradient_image(ax, extent, direction=0, cmap_range=(0, 0.5), **kwargs):
    phi = direction * np.pi / 2     # 角度值
    v = np.array([np.cos(phi), np.sin(phi)])  # 求解余弦正弦
    X = np.array([[v @ [0, 0], v @ [0, 0]],
                  [v @ [1, 1], v @ [1, 1]]])  # @号代表的是矩阵运算
    a, b = cmap_range  # 代表着颜色的最小范围和最大范围
    X = a + (b - a) / X.max() * X  # 用颜色范围调整数值
    im = ax.imshow(X, extent=extent, interpolation='bicubic',
                   vmin=0, vmax=1, aspect='auto', **kwargs)
    """
    ax.imshow显示图片 
    vmin，vmax:图片的最小值和最大值，jpg（0，255）png（0，1）
    x:显示的图片数据
    aspect:自动适应大小
    interpolation:填充效果 "bicubic"渐变
    """
 
    return im
 
 
def gradient_bar(ax, x, y, width=0.5, bottom=0):
    for left, top in zip(x, y):  # zip  同时取多个数组中的数据
        left = left - width/2
        right = left + width
        # 定义渐变颜色，（起始颜色，结束颜色）
        colors = [(114 / 255, 188 / 255, 213 / 255), (1, 1, 1)]
        cmap = LinearSegmentedColormap.from_list('my_camp', colors, N=256)
        gradient_image(ax, extent=(left, right, bottom, top),
                       cmap=cmap, cmap_range=(0, 0.8))
    # cmap 柱子颜色 cmap_range 柱子颜色渐变范围(0, 1)
 
 
def matplot(data):
    xticks = data[0]
    x = range(1, len(xticks)+1)
    y = data[1]
    xlabel = data[2]
    ylabel = data[3]
    title = data[-1]
    ylim_max = max(y) * 6 / 5
 
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111)
 
    # 标示柱状图的数值
    for x0, y0 in zip(x, y):
        plt.text(x0, y0, '%s' % float(y0),
                 ha='center',
                 va='bottom',
                 size=15,
                 family="Arial")
    xmin, xmax = 0, len(xticks)+1
    ymin, ymax = 0, ylim_max
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), autoscale_on=False)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.xticks(x, xticks)
 
    plt.title(title)
    gradient_bar(ax=ax, x=x, y=y)
 
    plt.show()
 
 
# 将数据打包传入画图函数
"""
dataList: [xlist, ylist, 'XLabel', 'yLabel', 'title']
"""
data = [['3×3', '4×4', '5×5', '6×6'], [181.5, 102.1, 65.4, 45.3], 'Arrange',
        'Potential(V)', 'Number']
 
matplot(data)