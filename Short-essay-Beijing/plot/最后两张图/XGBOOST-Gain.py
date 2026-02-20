import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10

# 读取数据
xgboost_df = pd.read_csv(r"E:\DATA Science\小论文Result\Fine_model\-XGBOOST\XGBOOST\plot_feature_importance_top20__xgboost_optimized.csv")

# 定义特征分类函数
def classify_feature(feature_name):
    name_lower = feature_name.lower()
    if 'pm25' in name_lower or 'pm2.5' in name_lower:
        return 'PM2.5 Related'
    else:
        return 'Meteorological/Others'

# 分类特征
xgboost_df['Category'] = xgboost_df['Feature'].apply(classify_feature)
xgboost_pm25 = xgboost_df[xgboost_df['Category'] == 'PM2.5 Related']
xgboost_meteo = xgboost_df[xgboost_df['Category'] == 'Meteorological/Others']

# 基于Gain的统计
total_gain = xgboost_df['Importance_Gain_Norm'].sum()
pm25_gain_total = xgboost_pm25['Importance_Gain_Norm'].sum()
meteo_gain_total = xgboost_meteo['Importance_Gain_Norm'].sum()

print("=" * 60)
print("XGBoost 基于 Gain 的特征重要性分析")
print("=" * 60)
print(f"总特征数: {len(xgboost_df)}")
print(f"PM2.5相关特征: {len(xgboost_pm25)}个 ({len(xgboost_pm25)/len(xgboost_df)*100:.1f}%)")
print(f"气象/其他特征: {len(xgboost_meteo)}个 ({len(xgboost_meteo)/len(xgboost_df)*100:.1f}%)")
print(f"\n【Gain贡献占比】")
print(f"PM2.5相关特征总Gain: {pm25_gain_total:.2f}%")
print(f"气象因子总Gain: {meteo_gain_total:.2f}%")

# TOP 10整体特征（基于Gain）
print(f"\n{'='*60}")
print("XGBoost 整体TOP 10特征（基于Gain）")
print(f"{'='*60}")
top10_gain = xgboost_df.nlargest(10, 'Importance_Gain_Norm')[['Feature', 'Category', 'Importance_Gain_Norm']]
for idx, row in top10_gain.iterrows():
    category_mark = "【气象】" if row['Category'] == 'Meteorological/Others' else "【PM2.5】"
    print(f"{idx+1:2d}. {row['Feature']:25s} {row['Importance_Gain_Norm']:6.2f}% {category_mark}")

# 非PM2.5特征TOP 5（基于Gain）
print(f"\n{'='*60}")
print("XGBoost 非PM2.5特征 TOP 5（基于Gain）")
print(f"{'='*60}")
meteo_top5_gain = xgboost_meteo.nlargest(5, 'Importance_Gain_Norm')[['Feature', 'Importance_Gain_Norm']]
for idx, row in meteo_top5_gain.iterrows():
    print(f"{idx+1}. {row['Feature']:20s}: {row['Importance_Gain_Norm']:.2f}%")

# 保存结果
xgboost_pm25_gain = xgboost_pm25['Importance_Gain_Norm'].sum()
xgboost_meteo_gain = xgboost_meteo['Importance_Gain_Norm'].sum()

# 可视化：XGBoost Gain分析图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：类别占比饼图（Gain）
ax1 = axes[0]
colors_pie = ['#FF9999', '#66B2FF']
explode = (0.05, 0.05)
ax1.pie([xgboost_pm25_gain, xgboost_meteo_gain], 
        labels=[f'PM2.5 Related\n{xgboost_pm25_gain:.1f}%', f'Meteorological\n{xgboost_meteo_gain:.1f}%'],
        autopct='', colors=colors_pie, explode=explode, startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title('XGBoost: Feature Category Distribution', 
              fontsize=13, fontweight='bold', pad=15)

# 右图：非PM2.5特征Gain TOP 5横向条形图
ax2 = axes[1]
meteo_top5_plot = xgboost_meteo.nlargest(5, 'Importance_Gain_Norm').sort_values('Importance_Gain_Norm')
colors_bar = ['#9370DB', '#FF6B6B', '#4ECDC4', '#FFA07A', '#45B7D1']
bars = ax2.barh(meteo_top5_plot['Feature'], meteo_top5_plot['Importance_Gain_Norm'], color=colors_bar)
ax2.set_xlabel('Normalized Gain (%)', fontsize=12, fontweight='bold')
ax2.set_title('XGBoost: Non-PM2.5 Features (Gain)\nTOP 5 Contributions', 
              fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
for i, (idx, row) in enumerate(meteo_top5_plot.iterrows()):
    ax2.text(row['Importance_Gain_Norm'] + 0.1, i, f"{row['Importance_Gain_Norm']:.2f}%",
             va='center', ha='left', fontsize=10, fontweight='bold')

# 调整x轴范围，给数值标签留出空间
current_xlim = ax2.get_xlim()
max_value = meteo_top5_plot['Importance_Gain_Norm'].max()
ax2.set_xlim(current_xlim[0], max_value * 1.15)  # 增加15%的空间

plt.tight_layout()
plt.savefig(r"Short-essay-Beijing\plot\最后两张图\xgboost_gain_analysis.svg", bbox_inches='tight')
plt.show()