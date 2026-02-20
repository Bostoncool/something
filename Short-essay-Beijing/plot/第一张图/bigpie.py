import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 设置中文字体，避免中文乱码
def setup_chinese_font():
    """设置中文字体，支持Windows、macOS和Linux系统"""
    import platform
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统字体
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    elif system == 'Darwin':  # macOS
        # macOS系统字体
        font_list = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        # Linux系统字体
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    
    # 检查字体是否可用
    available_fonts = [font.name for font in fm.fontManager.ttflist]
    chinese_font = None
    
    for font in font_list:
        if font in available_fonts:
            chinese_font = font
            break
    
    if chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font] + font_list
        print(f"Using Chinese font: {chinese_font}")
    else:
        # 备用方案：使用系统默认字体
        plt.rcParams['font.sans-serif'] = font_list + ['DejaVu Sans', 'Arial']
        print("Warning: No Chinese font found, may display as squares")
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12

# 调用字体设置函数
setup_chinese_font()

# 清除matplotlib字体缓存，确保新设置生效
try:
    fm._rebuild()
except:
    pass

# 数据准备
categories = [ '区域传输','本地排放']
values = [57, 43]

# 创建柔和的配色方案（Pastel柔和色调）
colors = ['#FFB3BA', '#FFDFBA']

# 创建图形和轴
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制饼图，不显示任何标签
wedges, texts, autotexts = ax.pie(values, 
                                  labels=None,  # 不显示外部标签
                                  colors=colors,
                                  autopct='',   # 不显示百分比
                                  startangle=90)

# 手动添加内部标签
for i, (wedge, category, value) in enumerate(zip(wedges, categories, values)):
    # 计算每个扇区的中心角度
    angle = (wedge.theta2 + wedge.theta1) / 2
    
    # 计算标签位置（在饼图内部）
    radius = 0.5  # 距离圆心的距离
    x = radius * np.cos(np.radians(angle))
    y = radius * np.sin(np.radians(angle))
    
    # 添加标签文本
    label_text = f'{category}\n{value}%'
    ax.text(x, y, label_text, 
            ha='center', va='center', 
            fontsize=25, fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

# 设置标题，不需要可以注释
# ax.set_title('产品销售分布图\nProduct Sales Distribution', 
#              fontsize=16, 
#              fontweight='bold', 
#              pad=20)

# # 添加图例，不需要可以注释
# ax.legend(wedges, 
#         [f'{category}: {value}%' for category, value in zip(categories, values)],
#         title="产品类别",
#         loc="center left",
#         bbox_to_anchor=(1, 0, 0.5, 1))

# 确保饼图是圆形
ax.axis('equal')

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 保存图形（可选）
# plt.savefig('pie_chart.png', dpi=300, bbox_inches='tight')