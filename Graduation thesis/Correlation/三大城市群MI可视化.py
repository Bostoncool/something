import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ===================== 基础配置 =====================
# 英文化，使用 Times New Roman（与 Geodetector 一致）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 全局样式配置
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 26
plt.rcParams['font.weight'] = 'normal'

# 数据路径
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_BASE = Path(r"H:\大论文Result\大论文图\三大城市群\三大城市群相关性分析图")

# 区域配色方案（与 Geodetector 一致：BTH/YRD/PRD）
region_mapping = {'bth': 'BTH', 'prd': 'PRD', 'yrd': 'YRD'}
region_colors = {
    'bth': '#C73E1D',    # BTH 红
    'yrd': '#2E86AB',    # YRD 蓝
    'prd': '#F18F01'     # PRD 橙
}

# 重新加载数据（确保数据可用）
data_dict = {
    'mi_factor': {},
    'mi_factor_by_year': {},
    'mi_pairwise_p': {}
}

# 读取各区域MI因子数据
region_output_dirs = {
    'bth': DATA_BASE / "mutual_info_bth",
    'prd': DATA_BASE / "mutual_info_prd",
    'yrd': DATA_BASE / "mutual_info_yrd",
}
for region in ['bth', 'prd', 'yrd']:
    csv_path = region_output_dirs[region] / f"{region}_mi_factor.csv"
    try:
        mi_factor = pd.read_csv(csv_path, encoding='utf-8-sig').dropna(subset=['mi'])
        data_dict['mi_factor'][region] = mi_factor
    except Exception as e:
        print(f"读取{region}数据失败: {csv_path} - {e}")

if not data_dict['mi_factor']:
    raise SystemExit(
        f"未读取到任何区域数据。请确认以下目录存在且包含 *mi_factor.csv：\n  {DATA_BASE}"
    )

# 定义因子分类函数（英文化，与 Geodetector 一致）
def classify_factor(factor_name):
    if 'met_' in factor_name:
        if 'temperature' in factor_name:
            return 'Meteorology-Temperature'
        elif 'wind' in factor_name:
            return 'Meteorology-Wind'
        elif 'pressure' in factor_name:
            return 'Meteorology-Pressure'
        elif 'precipitation' in factor_name:
            return 'Meteorology-Precipitation'
        elif 'cloud' in factor_name:
            return 'Meteorology-Cloud'
        elif 'dewpoint' in factor_name:
            return 'Meteorology-Dewpoint'
        else:
            return 'Meteorology-Other'
    elif 'ind_emis_' in factor_name:
        if 'pm25' in factor_name:
            return r'Industrial-PM$_{2.5}$'
        elif 'pm10' in factor_name:
            return r'Industrial-PM$_{10}$'
        elif 'so2' in factor_name:
            return r'Industrial-SO$_{2}$'
        elif 'nox' in factor_name:
            return r'Industrial-NO$_{x}$'
        elif 'co' in factor_name:
            return 'Industrial-CO'
        elif 'voc' in factor_name:
            return 'Industrial-VOC'
        else:
            return 'Industrial-Other'
    elif 'population' in factor_name:
        return 'Socioeconomic-Population'
    elif 'gdp' in factor_name:
        return 'Socioeconomic-GDP'
    elif 'electricity' in factor_name:
        return 'Energy-Electricity'
    elif 'gas_' in factor_name or 'lpg_' in factor_name:
        return 'Energy-Gas'
    elif 'road_' in factor_name:
        return 'Transport-Road Density'
    elif 'new_energy_vehicles' in factor_name:
        return 'Transport-NEV'
    elif 'landuse' in factor_name or 'industrial_land' in factor_name:
        return 'Land Use'
    elif 'night_light' in factor_name:
        return 'Night Light'
    elif 'fvc_mean' in factor_name:
        return 'Vegetation'
    elif 'pm25' in factor_name:
        return r'Air Quality-PM$_{2.5}$'
    else:
        return 'Other'

# ===================== 因子类别分析图表 =====================
fig, axes = plt.subplots(2, 2, figsize=(20, 20), facecolor='white', constrained_layout=True)

# 1. 各区域TOP5因子类别MI值总和对比（优化显示）
ax1 = axes[0, 0]
# 收集所有区域的因子类别统计
factor_category_stats = {}
all_categories = set()

for region in ['bth', 'prd', 'yrd']:
    if region in data_dict['mi_factor']:
        mi_data = data_dict['mi_factor'][region].copy()
        mi_data['factor_category'] = mi_data['factor'].apply(classify_factor)
        
        # 统计各类别
        category_summary = mi_data.groupby('factor_category').agg({
            'mi': 'sum'
        }).round(4)
        category_summary.columns = ['MI值总和']
        category_summary = category_summary.sort_values('MI值总和', ascending=False)
        
        factor_category_stats[region] = category_summary
        all_categories.update(category_summary.index.tolist())

# 准备绘图数据（只选所有区域中MI总和前8的类别，避免拥挤）
top_categories = []
for cat in all_categories:
    total = 0
    for region in ['bth', 'prd', 'yrd']:
        if region in factor_category_stats and cat in factor_category_stats[region].index:
            total += factor_category_stats[region].loc[cat, 'MI值总和']
    top_categories.append((cat, total))

# 取前8个类别
top_categories = sorted(top_categories, key=lambda x: x[1], reverse=True)[:8]
top_cat_names = [x[0] for x in top_categories]

x_pos = np.arange(len(top_cat_names))
width = 0.25

# 绘制柱状图
for i, region in enumerate(['bth', 'prd', 'yrd']):
    if region in factor_category_stats:
        mi_totals = []
        for cat in top_cat_names:
            if cat in factor_category_stats[region].index:
                mi_totals.append(factor_category_stats[region].loc[cat, 'MI值总和'])
            else:
                mi_totals.append(0)
        
        bars = ax1.bar(x_pos + i*width, mi_totals, width, 
                       label=region_mapping[region], color=region_colors[region], alpha=0.7,
                       edgecolor='white', linewidth=1)
        
        # 添加数值标签（只显示>0的值）
        for bar, value in zip(bars, mi_totals):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                         f'{value:.2f}', ha='center', va='bottom', fontsize=19, fontweight='bold')

ax1.set_title('Major Factor Category MI Sum by Region', fontweight='bold', fontsize=24, pad=20)
ax1.set_ylabel('MI Sum', fontsize=22)
ax1.set_xticks(x_pos + width)
ax1.set_xticklabels(top_cat_names, rotation=45, ha='right', fontsize=19)
ax1.legend(loc='upper right', fontsize=19)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 6)
ax1.set_box_aspect(1)  # 子图正方形

# 2. 各区域因子类别显著率对比（优化版）
ax2 = axes[0, 1]
# 重新计算显著率
significant_rates = {}
for region in ['bth', 'prd', 'yrd']:
    if region in data_dict['mi_factor']:
        mi_data = data_dict['mi_factor'][region].copy()
        mi_data['factor_category'] = mi_data['factor'].apply(classify_factor)
        
        # 统计各类别显著率
        cat_significant = mi_data.groupby('factor_category').agg({
            'mi_significant': ['count', 'sum']
        })
        cat_significant.columns = ['总数', '显著数']
        cat_significant['显著率'] = (cat_significant['显著数'] / cat_significant['总数']).round(4)
        
        # 只选择因子数量>=2的类别
        major_cats = cat_significant[cat_significant['总数'] >= 2].sort_values('显著率', ascending=True)
        significant_rates[region] = major_cats

# 绘制横向条形图（每个区域单独一列）
y_positions = {}
current_y = 0
cat_list = []

# 收集所有主要类别
for region in ['bth', 'prd', 'yrd']:
    if region in significant_rates:
        for cat in significant_rates[region].index:
            if cat not in cat_list:
                cat_list.append(cat)
                y_positions[cat] = current_y
                current_y += 1

# 绘制每个区域的显著率
region_x_pos = [0, 1, 2]  # 三个区域的x位置
for i, region in enumerate(['bth', 'prd', 'yrd']):
    if region in significant_rates:
        x = region_x_pos[i]
        for cat in significant_rates[region].index:
            if cat in y_positions:
                rate = significant_rates[region].loc[cat, '显著率']
                ax2.barh(y_positions[cat] + i*0.25, rate, 0.25,
                         color=region_colors[region], alpha=0.7,
                         edgecolor='white', linewidth=1)
                
                # 添加百分比标签
                if rate > 0:
                    ax2.text(x + 0.25, y_positions[cat] + i*0.25,
                             f'{rate:.0%}', ha='left', va='center', 
                             fontsize=19, fontweight='bold')

ax2.set_title('Factor Category Significance Rate (n≥2)', fontweight='bold', fontsize=24, pad=20)
ax2.set_xlabel('Significance Rate', fontsize=22)
ax2.set_xlim(0, 1.1)
ax2.set_yticks(list(y_positions.values()))
ax2.set_yticklabels(list(y_positions.keys()), fontsize=24)
ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=19)
ax2.legend([region_mapping[r] for r in ['bth', 'prd', 'yrd']], fontsize=19, loc='lower right')
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_box_aspect(1)  # 子图正方形

# 3. 各区域年度关键因子类别变化（优化版）
ax3 = axes[1, 0]
# 读取年度数据并统计
yearly_key_categories = {}
years = []

for region in ['bth', 'prd', 'yrd']:
    try:
        yearly_path = region_output_dirs[region] / f"{region}_mi_factor_by_year.csv"
        yearly_data = pd.read_csv(yearly_path, encoding='utf-8-sig').dropna(subset=['mi'])
        if not years:
            years = sorted(yearly_data['year'].unique())
        
        key_cats = []
        for year in years:
            year_data = yearly_data[yearly_data['year'] == year]
            top_factor = year_data.loc[year_data['mi'].idxmax(), 'factor']
            top_category = classify_factor(top_factor)
            key_cats.append(top_category)
        
        yearly_key_categories[region] = key_cats
    except:
        yearly_key_categories[region] = []

# 创建类别颜色映射
unique_cats = []
for region_data in yearly_key_categories.values():
    unique_cats.extend(region_data)
unique_cats = list(set(unique_cats))
cat_color_map = plt.cm.Set3(np.linspace(0, 1, len(unique_cats)))
cat_to_color = {cat: cat_color_map[i] for i, cat in enumerate(unique_cats)}

# 绘制年度关键因子类别变化（使用不同标记和颜色）
markers = ['o', 's', '^']  # 圆形、方形、三角形
for i, region in enumerate(['bth', 'prd', 'yrd']):
    if region in yearly_key_categories and yearly_key_categories[region]:
        # 为每个点设置对应的颜色
        point_colors = [cat_to_color[cat] for cat in yearly_key_categories[region]]
        
        # 绘制折线
        ax3.plot(years, [i+1]*len(years), marker=markers[i], linewidth=2.5, markersize=10,
                label=region_mapping[region], color=region_colors[region], alpha=0.8,
                markeredgecolor='white', markeredgewidth=1)
        
        # 填充标记颜色
        for j, (year, cat) in enumerate(zip(years, yearly_key_categories[region])):
            ax3.scatter(year, i+1, color=cat_to_color[cat], s=100, marker=markers[i],
                      edgecolor='white', linewidth=1, alpha=0.9)
            
            # 添加类别标签（旋转显示）
            ax3.text(year, i+1 + 0.1, cat, ha='center', va='bottom', 
                    fontsize=19, rotation=45, fontweight='bold')

ax3.set_title('Annual Key Factor Category by Region', fontweight='bold', fontsize=24, pad=20)
ax3.set_xlabel('Year', fontsize=22)
ax3.set_ylabel('Region', fontsize=22)
ax3.set_yticks([1, 2, 3])
ax3.set_yticklabels([region_mapping[r] for r in ['bth', 'prd', 'yrd']], fontsize=24)
ax3.set_xticks(years)
ax3.set_xticklabels(years, fontsize=19)
ax3.legend(loc='upper right', fontsize=19)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0.5, 3.5)
ax3.set_box_aspect(1)  # 子图正方形

# 4. 各区域MI值分布对比（修复重叠问题，使用核密度估计）
ax4 = axes[1, 1]
from scipy.stats import gaussian_kde

# 为每个区域计算核密度
for region in ['bth', 'prd', 'yrd']:
    if region in data_dict['mi_factor']:
        mi_values = data_dict['mi_factor'][region]['mi'].values
        mi_values = mi_values[~np.isnan(mi_values)]  # 确保无空值
        
        # 计算核密度
        kde = gaussian_kde(mi_values)
        x_range = np.linspace(mi_values.min(), mi_values.max(), 200)
        y_kde = kde(x_range)
        
        # 绘制密度曲线
        ax4.plot(x_range, y_kde, linewidth=3, label=region_mapping[region],
                color=region_colors[region], alpha=0.8)
        
        # 填充曲线下方区域
        ax4.fill_between(x_range, y_kde, alpha=0.2, color=region_colors[region])
        
        # 添加统计信息标注
        mean_mi = mi_values.mean()
        std_mi = mi_values.std()
        ax4.text(0.02, 0.98 - i*0.1, 
                f'{region_mapping[region]}: Mean={mean_mi:.4f}\n     Std={std_mi:.4f}',
                transform=ax4.transAxes, fontsize=19, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=region_colors[region], alpha=0.1))

ax4.set_title('MI Value KDE Distribution by Region', fontweight='bold', fontsize=24, pad=20)
ax4.set_xlabel('MI Value', fontsize=22)
ax4.set_ylabel('Kernel Density', fontsize=22)
ax4.legend(loc='upper right', fontsize=19)
ax4.grid(True, alpha=0.3)
ax4.set_box_aspect(1)  # 子图正方形

# 保存图表（与 Geodetector 一致：高分辨率，脚本所在目录）
output_path = SCRIPT_DIR / 'mi_factor_category_optimized.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Finished! mi_factor_category_optimized.png")

# ===================== 总体分析图表 =====================
fig2, axes2 = plt.subplots(2, 2, figsize=(20, 20), facecolor='white', constrained_layout=True)

# 1. MI值分布箱线图（优化）
ax1 = axes2[0, 0]
mi_data_for_box = []
labels_for_box = []

for i, region in enumerate(['bth', 'prd', 'yrd']):
    if region in data_dict['mi_factor']:
        mi_values = data_dict['mi_factor'][region]['mi'].dropna()
        mi_data_for_box.append(mi_values)
        labels_for_box.append(region_mapping[region])

box_plot = ax1.boxplot(mi_data_for_box, labels=labels_for_box, patch_artist=True, 
                       notch=True, showmeans=True, meanprops={'marker':'o', 'markerfacecolor':'red', 'markersize':6})

# 设置箱线图颜色
for patch, color in zip(box_plot['boxes'], [region_colors[r] for r in ['bth', 'prd', 'yrd']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('white')
    patch.set_linewidth(1)

ax1.set_title('MI Value Distribution by Region', fontweight='bold', fontsize=24, pad=15)
ax1.set_ylabel(r'Mutual Information (MI)', fontsize=22)
ax1.set_ylim(0, 0.6)
ax1.grid(True, alpha=0.3)

# 2. 显著因子占比柱状图（优化）
ax2 = axes2[0, 1]
regions_names = []
significant_ratios = []

for region in ['bth', 'prd', 'yrd']:
    if region in data_dict['mi_factor']:
        mi_data = data_dict['mi_factor'][region]
        ratio = mi_data['mi_significant'].sum() / len(mi_data)
        regions_names.append(region_mapping[region])
        significant_ratios.append(ratio)

bars = ax2.bar(regions_names, significant_ratios, 
               color=[region_colors[r] for r in ['bth', 'prd', 'yrd']], 
               alpha=0.7, edgecolor='white', linewidth=1)
ax2.set_title('Significant MI Factor Ratio by Region', fontweight='bold', fontsize=24, pad=15)
ax2.set_ylabel('Significant Factor Ratio', fontsize=22)
ax2.set_ylim(0, 1.05)

# 在柱子上添加数值标签
for bar, ratio in zip(bars, significant_ratios):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{ratio:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=19)

# 3. TOP5因子MI值对比（优化）
ax3 = axes2[1, 0]
x_pos = np.arange(5)
width = 0.25

for i, region in enumerate(['bth', 'prd', 'yrd']):
    if region in data_dict['mi_factor']:
        # 获取前5个最重要的因子
        top_factors = data_dict['mi_factor'][region].nlargest(5, 'mi')
        mi_values = top_factors['mi'].values
        
        # 调整x位置
        bars = ax3.bar(x_pos + i*width, mi_values, width, 
                       label=region_mapping[region], color=region_colors[region], alpha=0.7,
                       edgecolor='white', linewidth=1)
        
        # 添加数值标签
        for bar, value in zip(bars, mi_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=19, fontweight='bold')

ax3.set_title('TOP5 Key Factor MI Value by Region', fontweight='bold', fontsize=24, pad=15)
ax3.set_ylabel(r'Mutual Information (MI)', fontsize=22)
ax3.set_ylim(0, 0.6)
ax3.set_xlabel('Factor Rank', fontsize=22)
ax3.set_xticks(x_pos + width)
ax3.set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'], fontsize=19)
ax3.legend(fontsize=19)
ax3.grid(True, alpha=0.3)

# 4. MI值分布对比（改用核密度估计，解决重叠）
ax4 = axes2[1, 1]
for i, region in enumerate(['bth', 'prd', 'yrd']):
    if region in data_dict['mi_factor']:
        mi_values = data_dict['mi_factor'][region]['mi'].dropna()
        
        # 计算核密度
        kde = gaussian_kde(mi_values)
        x_range = np.linspace(0, mi_values.max()*1.1, 200)
        y_kde = kde(x_range)
        
        # 绘制密度曲线
        ax4.plot(x_range, y_kde, linewidth=2.5, label=region_mapping[region],
                color=region_colors[region], alpha=0.8)
        
        # 填充曲线下方
        ax4.fill_between(x_range, y_kde, alpha=0.2, color=region_colors[region])

ax4.set_title('MI Value KDE Distribution by Region', fontweight='bold', fontsize=24, pad=15)
ax4.set_xlabel(r'Mutual Information (MI)', fontsize=22)
ax4.set_ylabel('Kernel Density', fontsize=22)
ax4.legend(fontsize=19)
ax4.grid(True, alpha=0.3)

output_path2 = SCRIPT_DIR / 'mi_factor_overall_optimized.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Finished! mi_factor_overall_optimized.png")