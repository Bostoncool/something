import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ======================== 基础设置 ========================
# 英文化，使用 Times New Roman（参考 Geodetector 可视化）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (20, 16)   # 图表整体尺寸
plt.rcParams['axes.grid'] = True            # 默认显示网格
plt.rcParams['grid.alpha'] = 0.3            # 网格透明度
plt.rcParams['xtick.labelsize'] = 18        # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18        # y轴刻度字体大小
plt.rcParams['axes.labelsize'] = 20         # 坐标轴标签字体大小
plt.rcParams['axes.titlesize'] = 22         # 子图标题字体大小
plt.rcParams['font.weight'] = 'normal'      # 字体粗细
plt.rcParams['savefig.dpi'] = 300          # 保存图片的分辨率
plt.rcParams['savefig.bbox'] = 'tight'     # 紧凑保存，去除多余空白

# ======================== 准备数据 ========================
# 从分析结果中提取的核心数据
# 1. 整体相关性数据
regions_short = ['BTH', 'PRD', 'YRD']
sig_ratios = [71.05, 55.26, 81.58]          # Significant factor ratio (%)
avg_corrs = [0.2453, 0.2840, 0.1862]       # Average correlation coefficient

# 2. 相关性强度分布数据（强正、中正、弱正、强负、中负、弱负）
strength_data = {
    'BTH': [2, 20, 2, 1, 1, 1],
    'PRD': [13, 3, 1, 0, 4, 0],
    'YRD': [6, 10, 4, 0, 7, 4]
}
strength_labels = ['Strong Positive', 'Moderate Positive', 'Weak Positive',
                   'Strong Negative', 'Moderate Negative', 'Weak Negative']
strength_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#87CEEB']

# 3. 年度显著因子比例数据
years = [2018, 2019, 2020, 2021, 2022, 2023]
bth_yearly_sig = [15.79, 18.42, 18.42, 15.79, 15.79, 13.16]
prd_yearly_sig = [0.00, 0.00, 7.89, 2.63, 0.00, 0.00]
yrd_yearly_sig = [57.89, 57.89, 60.53, 57.89, 55.26, 55.26]

# 4. 年度强相关因子数量
bth_strong = [5, 7, 6, 5, 6, 5]
prd_strong = [0, 0, 3, 1, 1, 0]
yrd_strong = [10, 11, 12, 10, 9, 9]

# 5. 关键环境因子相关性数据
factors = [r'PM$_{2.5}$ Emission', 'Temperature', 'Wind Speed', 'Cloud Cover', 'Pressure']
bth_values = [0.428, 0.764, -0.830, 0.594, -0.456]
prd_values = [0.816, 0.000, -0.391, 0.517, -0.391]
yrd_values = [0.815, -0.679, -0.614, -0.666, 0.000]

# 6. 数据质量指标
metrics = ['Total Factors', 'Significant Factors', 'Avg Sample Size', 'Data Completeness']
bth_metrics = [38, 27, 13, 100]
prd_metrics = [38, 21, 9, 95]
yrd_metrics = [38, 31, 156, 100]

# ======================== 创建图表 ========================
# 使用 GridSpec 创建 2 行 3 列布局，子图正方形
fig = plt.figure(figsize=(20, 14), facecolor='white', constrained_layout=True)
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1])
# 子图间距：hspace 控制上下两行间距，wspace 控制左右三列间距（数值越大间距越大）
fig.set_constrained_layout_pads(hspace=0.1, wspace=0.08)

# 子图1: 各区域整体相关性对比（双Y轴）
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(regions_short))
width = 0.35

# 绘制显著因子比例柱状图（左Y轴）
bars1 = ax1.bar(x - width/2, sig_ratios, width, 
                label='Significant Factor Ratio (%)', color='#2E86AB', alpha=0.8)
ax1.set_ylabel('Significant Factor Ratio (%)', fontsize=18, color='#2E86AB')
ax1.tick_params(axis='y', labelcolor='#2E86AB')

# 创建右侧Y轴，绘制平均相关系数
ax1_twin = ax1.twinx()
bars2 = ax1_twin.bar(x + width/2, avg_corrs, width, 
                     label='Average Correlation Coefficient', color='#A23B72', alpha=0.8)
ax1_twin.set_ylabel('Average Correlation Coefficient', fontsize=18, color='#A23B72')
ax1_twin.tick_params(axis='y', labelcolor='#A23B72')
ax1.set_ylim(0, 100)       # 蓝色左轴上限
ax1_twin.set_ylim(0, 0.4)  # 紫色右轴上限

# 设置标题和标签
ax1.set_xlabel('Region', fontsize=18)
ax1.set_title('Overall Correlation Comparison', fontsize=22, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(regions_short)

# 添加数值标签
for bar, value in zip(bars1, sig_ratios):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{value:.1f}%', ha='center', va='bottom', fontsize=15)
for bar, value in zip(bars2, avg_corrs):
    ax1_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                  f'{value:.3f}', ha='center', va='bottom', fontsize=15)
ax1.set_box_aspect(1)

# 子图2: 各区域相关性强度分布
ax2 = fig.add_subplot(gs[0, 1])
width = 0.15  # 每个柱子的宽度

# 绘制分组柱状图
for i, (label, color) in enumerate(zip(strength_labels, strength_colors)):
    values = [strength_data[region][i] for region in regions_short]
    ax2.bar(x + (i - 2.5) * width, values, width, 
            label=label, color=color, alpha=0.8)

ax2.set_xlabel('Region', fontsize=18)
ax2.set_ylabel('Number of Factors', fontsize=18)
ax2.set_title('Correlation Strength Distribution', fontsize=22, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(regions_short)
ax2.legend(loc='upper right', fontsize=15)
ax2.set_box_aspect(1)

# 子图3: 年度显著因子比例变化趋势
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(years, bth_yearly_sig, marker='o', linewidth=2, 
         label='BTH', color='#FF6B6B', markersize=6)
ax3.plot(years, prd_yearly_sig, marker='s', linewidth=2, 
         label='PRD', color='#4ECDC4', markersize=6)
ax3.plot(years, yrd_yearly_sig, marker='^', linewidth=2, 
         label='YRD', color='#45B7D1', markersize=6)

ax3.set_xlabel('Year', fontsize=18)
ax3.set_ylabel('Significant Factor Ratio (%)', fontsize=18)
ax3.set_title('Annual Significant Factor Ratio Trend', fontsize=22, fontweight='bold')
ax3.legend()
ax3.set_xticks(years)
ax3.set_xticklabels(years, rotation=45)
ax3.set_box_aspect(1)

# 子图4: 各区域强相关因子年度变化
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(years, bth_strong, marker='o', linewidth=2, 
         label='BTH', color='#FF6B6B', markersize=6)
ax4.plot(years, prd_strong, marker='s', linewidth=2, 
         label='PRD', color='#4ECDC4', markersize=6)
ax4.plot(years, yrd_strong, marker='^', linewidth=2, 
         label='YRD', color='#45B7D1', markersize=6)

ax4.set_xlabel('Year', fontsize=18)
ax4.set_ylabel(r'Strong Correlation Factor Count (|r|≥0.7)', fontsize=18)
ax4.set_title('Annual Strong Correlation Factor Count', fontsize=22, fontweight='bold')
ax4.legend()
ax4.set_xticks(years)
ax4.set_xticklabels(years, rotation=45)
ax4.set_box_aspect(1)

# 子图5: 各区域关键因子相关性强度对比
ax5 = fig.add_subplot(gs[1, 1])
x = np.arange(len(factors))
width = 0.25

ax5.bar(x - width, bth_values, width, label='BTH', color='#FF6B6B', alpha=0.8)
ax5.bar(x, prd_values, width, label='PRD', color='#4ECDC4', alpha=0.8)
ax5.bar(x + width, yrd_values, width, label='YRD', color='#45B7D1', alpha=0.8)

ax5.set_xlabel('Key Environmental Factor', fontsize=18)
ax5.set_ylabel('Spearman Correlation Coefficient', fontsize=18)
ax5.set_title('Key Environmental Factor Correlation Comparison', fontsize=22, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(factors, rotation=45, ha='right', fontsize=15)
ax5.legend()
ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)  # 添加0基准线
ax5.set_box_aspect(1)

# 子图6: 各区域数据质量统计
ax6 = fig.add_subplot(gs[1, 2])
x = np.arange(len(metrics))
width = 0.25

ax6.bar(x - width, bth_metrics, width, label='BTH', color='#FF6B6B', alpha=0.8)
ax6.bar(x, prd_metrics, width, label='PRD', color='#4ECDC4', alpha=0.8)
ax6.bar(x + width, yrd_metrics, width, label='YRD', color='#45B7D1', alpha=0.8)

ax6.set_xlabel('Data Quality Indicator', fontsize=18)
ax6.set_ylabel('Value', fontsize=18)
ax6.set_title('Data Quality Comparison', fontsize=22, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics, rotation=45, ha='right', fontsize=15)
ax6.legend()
ax6.set_box_aspect(1)

# ======================== 最终调整与保存 ========================
plt.savefig('three_regions_spearman_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Chart saved successfully: three_regions_spearman_analysis.png")