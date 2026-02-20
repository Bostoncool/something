import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10

# 读取数据
lightgbm_df = pd.read_csv(r"E:\DATA Science\小论文Result\Fine_model\-LightGBM\Split2\output\feature_importance.csv")

# 定义特征分类函数
def classify_feature(feature_name):
    name_lower = feature_name.lower()
    if 'pm25' in name_lower or 'pm2.5' in name_lower:
        return 'PM2.5 Related'
    else:
        return 'Meteorological/Others'

# 分类特征
lightgbm_df['Category'] = lightgbm_df['Feature'].apply(classify_feature)
lightgbm_pm25 = lightgbm_df[lightgbm_df['Category'] == 'PM2.5 Related']
lightgbm_meteo = lightgbm_df[lightgbm_df['Category'] == 'Meteorological/Others']

# 基于Split的统计
total_split = lightgbm_df['Importance_Split_Norm'].sum()
pm25_split_total = lightgbm_pm25['Importance_Split_Norm'].sum()
meteo_split_total = lightgbm_meteo['Importance_Split_Norm'].sum()

print("=" * 60)
print("LightGBM 基于 Split 的特征重要性分析")
print("=" * 60)
print(f"总特征数: {len(lightgbm_df)}")
print(f"PM2.5相关特征: {len(lightgbm_pm25)}个 ({len(lightgbm_pm25)/len(lightgbm_df)*100:.1f}%)")
print(f"气象/其他特征: {len(lightgbm_meteo)}个 ({len(lightgbm_meteo)/len(lightgbm_df)*100:.1f}%)")
print(f"\n【Split贡献占比】")
print(f"PM2.5相关特征总Split: {pm25_split_total:.2f}%")
print(f"气象因子总Split: {meteo_split_total:.2f}%")

# TOP 10整体特征（基于Split）
print(f"\n{'='*60}")
print("LightGBM 整体TOP 10特征（基于Split）")
print(f"{'='*60}")
top10_split = lightgbm_df.nlargest(10, 'Importance_Split_Norm')[['Feature', 'Category', 'Importance_Split_Norm']]
for idx, row in top10_split.iterrows():
    category_mark = "【气象】" if row['Category'] == 'Meteorological/Others' else "【PM2.5】"
    print(f"{idx+1:2d}. {row['Feature']:25s} {row['Importance_Split_Norm']:6.2f}% {category_mark}")

# 非PM2.5特征TOP 5（基于Split）
print(f"\n{'='*60}")
print("LightGBM 非PM2.5特征 TOP 5（基于Split）")
print(f"{'='*60}")
meteo_top5_split = lightgbm_meteo.nlargest(5, 'Importance_Split_Norm')[['Feature', 'Importance_Split_Norm']]
for idx, row in meteo_top5_split.iterrows():
    print(f"{idx+1}. {row['Feature']:25s}: {row['Importance_Split_Norm']:.2f}%")

# 保存结果
lightgbm_pm25_split = lightgbm_pm25['Importance_Split_Norm'].sum()
lightgbm_meteo_split = lightgbm_meteo['Importance_Split_Norm'].sum()

# 可视化：LightGBM Split分析图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：类别占比饼图（Split）
ax1 = axes[0]
colors_pie = ['#FF9999', '#66B2FF']
explode = (0.05, 0.05)
ax1.pie([lightgbm_pm25_split, lightgbm_meteo_split], 
        labels=[f'PM2.5 Related\n{lightgbm_pm25_split:.1f}%', f'Meteorological\n{lightgbm_meteo_split:.1f}%'],
        autopct='', colors=colors_pie, explode=explode, startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title('LightGBM: Feature Category Distribution', 
              fontsize=13, fontweight='bold', pad=15)

# 右图：非PM2.5特征Split TOP 5横向条形图
ax2 = axes[1]
meteo_top5_plot = lightgbm_meteo.nlargest(5, 'Importance_Split_Norm').sort_values('Importance_Split_Norm')
colors_bar = ['#9370DB', '#FF6B6B', '#4ECDC4', '#FFA07A', '#45B7D1']
bars = ax2.barh(meteo_top5_plot['Feature'], meteo_top5_plot['Importance_Split_Norm'], color=colors_bar)
ax2.set_xlabel('Normalized Split (%)', fontsize=12, fontweight='bold')
ax2.set_title('LightGBM: Non-PM2.5 Features (Split)\nTOP 5 Contributions', 
              fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
for i, (idx, row) in enumerate(meteo_top5_plot.iterrows()):
    ax2.text(row['Importance_Split_Norm'] + 0.05, i, f"{row['Importance_Split_Norm']:.2f}%",
             va='center', ha='left', fontsize=10, fontweight='bold')

# 调整x轴范围，给数值标签留出空间
current_xlim = ax2.get_xlim()
max_value = meteo_top5_plot['Importance_Split_Norm'].max()
ax2.set_xlim(current_xlim[0], max_value * 1.15)  # 增加15%的空间

plt.tight_layout()
plt.savefig(r"Short-essay-Beijing\plot\最后两张图\lightgbm_split_analysis.svg", bbox_inches='tight')
plt.show()