import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ===================== 基础配置 =====================
# 英文化，使用 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 全局样式配置
plt.rcParams['figure.figsize'] = (20, 20)  # 画布大小（正方形）
plt.rcParams['axes.grid'] = True           # 显示网格
plt.rcParams['grid.alpha'] = 0.3           # 网格透明度
plt.rcParams['xtick.labelsize'] = 18       # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18       # y轴刻度字体大小
plt.rcParams['axes.labelsize'] = 20        # 坐标轴标签字体大小
plt.rcParams['axes.titlesize'] = 22        # 子图标题字体大小
plt.rcParams['font.weight'] = 'normal'     # 字体粗细

# 区域配色方案（BTH/YRD/PRD）
region_config = {
    'BTH': {'color': '#C73E1D', 'pm_color': '#FF6B6B', 'light_color': '#FFB6C1'},
    'YRD': {'color': '#2E86AB', 'pm_color': '#4ECDC4', 'light_color': '#87CEEB'},
    'PRD': {'color': '#F18F01', 'pm_color': '#FFD166', 'light_color': '#FFD700'}
}

# ===================== 数据准备（示例数据） =====================
# 模拟三大地区的面板数据分析结果
# 实际使用时替换为你的真实数据
np.random.seed(42)
regions = ['BTH', 'YRD', 'PRD']
years = list(range(2018, 2024))

# PM2.5 annual data
pm25_data = {
    'BTH': {'mean': [45.2, 44.1, 43.5, 42.8, 41.9, 40.8], 'std': [10.2, 9.8, 10.5, 9.9, 10.1, 9.7]},
    'YRD': {'mean': [34.5, 33.2, 32.8, 31.9, 31.2, 30.5], 'std': [7.2, 7.0, 7.5, 6.8, 7.1, 6.9]},
    'PRD': {'mean': [25.8, 25.1, 24.5, 23.9, 23.5, 22.8], 'std': [4.2, 4.0, 4.5, 3.9, 4.1, 3.8]}
}

# Industrial emission data (Top5)
emission_data = {
    'BTH': [8117684, 1332399, 1128019, 550335, 470043],
    'YRD': [4248721, 1624581, 907246, 306001, 299824],
    'PRD': [4624474, 1892627, 988936, 367166, 324528]
}
emission_labels = ['CO', 'VOC', r'NO$_{x}$', r'NH$_{3}$', r'PM$_{10}$']

# Socioeconomic indicators correlation
corr_data = {
    'BTH': [0.261, 0.249, 0.122, -0.009, -0.146, -0.168],
    'YRD': [0.008, -0.019, -0.056, -0.088, -0.145, -0.178],
    'PRD': [-0.028, -0.061, -0.079, -0.083, -0.175, -0.194]
}
corr_labels = ['Gas Supply', 'Night Light', 'Road Density', 'GDP', 'Electricity', 'Population']

# ===================== 九合一图表绘制 =====================
# 创建画布（constrained_layout 与 set_box_aspect 兼容，保持子图正方形）
fig = plt.figure(figsize=(20, 20), facecolor='white', constrained_layout=True)

# 使用GridSpec创建9宫格布局（3行3列）
# height_ratios: 行高比例，width_ratios: 列宽比例
gs = GridSpec(3, 3, figure=fig, 
              height_ratios=[1, 1, 1],    # 三行等高（正方形网格）
              width_ratios=[1, 1, 1])     # 三列等宽（正方形网格）

# 遍历每个区域，绘制对应的子图
for idx, region in enumerate(regions):
    config = region_config[region]
    
    # ========== 第一列：PM2.5年度变化趋势图 ==========
    ax1 = fig.add_subplot(gs[idx, 0])  # 第idx行，第0列
    
    # 绘制均值折线
    mean_vals = pm25_data[region]['mean']
    std_vals = pm25_data[region]['std']
    
    line = ax1.plot(years, mean_vals, marker='o', linewidth=3, 
                    color=config['pm_color'], label='Annual Mean', markersize=8)
    
    # 绘制标准差填充带
    ax1.fill_between(years, 
                    np.array(mean_vals) - np.array(std_vals),
                    np.array(mean_vals) + np.array(std_vals),
                    alpha=0.3, color=config['pm_color'], label='±1 Std')
    
    # 美化配置
    ax1.set_xlabel('Year', fontsize=18)
    ax1.set_ylabel(r'PM$_{2.5}$ (μg/m³)', fontsize=18)
    ax1.set_title(f'{region} PM$_{{2.5}}$ Annual Trend', fontweight='bold', fontsize=20)
    ax1.legend(loc='upper right', fontsize=15)
    ax1.set_ylim(0, max(mean_vals) + max(std_vals) + 5)
    ax1.grid(True, alpha=0.3)
    
    # 标注数值
    for x, y in zip(years, mean_vals):
        ax1.text(x, y+1, f'{y:.1f}', ha='center', va='bottom', fontsize=15)
    ax1.set_box_aspect(1)  # 子图正方形
    
    # ========== 第二列：主要工业排放物柱状图 ==========
    ax2 = fig.add_subplot(gs[idx, 1])  # 第idx行，第1列
    
    # 绘制柱状图
    emissions = emission_data[region]
    bars = ax2.bar(range(len(emission_labels)), emissions, 
                   color=config['color'], alpha=0.7, edgecolor='white', linewidth=1)
    
    # 美化配置
    ax2.set_xticks(range(len(emission_labels)))
    ax2.set_xticklabels(emission_labels, rotation=45, ha='right', fontsize=15)
    ax2.set_ylabel('Emissions (units)', fontsize=18)
    ax2.set_title(f'{region} Major Industrial Emissions', fontweight='bold', fontsize=20)
    
    # 数值标签（简化显示）
    for bar, value in zip(bars, emissions):
        # 科学计数法显示大数
        label = f'{value/1e6:.1f}M' if value > 1e6 else f'{value/1e3:.0f}K'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(emissions)*0.01,
                label, ha='center', va='bottom', fontsize=15)
    ax2.set_box_aspect(1)  # 子图正方形
    
    # ========== 第三列：经济社会指标相关性水平条形图 ==========
    ax3 = fig.add_subplot(gs[idx, 2])  # 第idx行，第2列
    
    # 相关性数据
    corrs = corr_data[region]
    
    # 根据正负值设置颜色（正相关绿色，负相关红色）
    colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in corrs]
    
    # 绘制水平条形图
    bars = ax3.barh(range(len(corr_labels)), corrs, color=colors, alpha=0.7)
    
    # 美化配置
    ax3.set_yticks(range(len(corr_labels)))
    ax3.set_yticklabels(corr_labels, fontsize=16)
    ax3.set_xlabel(r'Correlation with PM$_{2.5}$', fontsize=18)
    ax3.set_title(f'{region} Socioeconomic Indicators vs PM$_{{2.5}}$', fontweight='bold', fontsize=20)
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    ax3.set_xlim(-0.3, 0.3)
    
    # 数值标签
    for bar, corr in zip(bars, corrs):
        x_pos = bar.get_width() + 0.01 if corr > 0 else bar.get_width() - 0.01
        ha_align = 'left' if corr > 0 else 'right'
        ax3.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'{corr:.3f}', va='center', ha=ha_align, fontsize=15)
    ax3.set_box_aspect(1)  # 子图正方形

# ===================== 整体优化 =====================


# 保存图片（高分辨率，保存到脚本所在目录）
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           'three_regions_comprehensive_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Finished!")