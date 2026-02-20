import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import matplotlib.font_manager as fm

# =========================
# 1. 读取 XGBoost 特征重要性结果
# =========================
path = r"H:\DATA Science\小论文Result\Fine_model\-XGBOOST\XGBOOST\plot_feature_importance__xgboost_optimized.csv"  # 改成你的实际路径
df = pd.read_csv(path)

# =========================
# 2. 剔除所有 PM2.5 相关特征
#    （包含 PM25 / PM2.5 及其衍生项，如 ma/diff/interaction 等）
# =========================
pm_pattern = re.compile(r"PM\s*2\.?5|PM25", re.IGNORECASE)
non_pm = df[~df["Feature"].astype(str).str.contains(pm_pattern)]

# =========================
# 3. 选取非 PM2.5 中 Gain_Norm 最高的 TOP5
# =========================
top5 = (
    non_pm
    .sort_values("Importance_Gain_Norm", ascending=False)
    .head(5)
    .copy()
)

# 为了画横向柱状图，反向排序
plot_df = top5.sort_values("Importance_Gain_Norm", ascending=True)

# =========================
# 4. 设置字体和样式
# =========================
# 设置Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 设置seaborn样式
sns.set_style("whitegrid")

# =========================
# 5. 绘图
# =========================
plt.figure(figsize=(10, 8))

# 使用五种不同的颜色
bar_colors = [
    '#1f77b4',  # 蓝色
    '#ff7f0e',  # 橙色
    '#2ca02c',  # 绿色
    '#d62728',  # 红色
    '#9467bd'   # 紫色
]

bars = plt.barh(plot_df["Feature"], plot_df["Importance_Gain_Norm"],
                color=bar_colors,
                edgecolor='white', linewidth=0.5, alpha=0.8)

# 添加数值标签
for bar, value in zip(bars, plot_df["Importance_Gain_Norm"]):
    plt.text(value + 0.001, bar.get_y() + bar.get_height()/2,
             f"{value:.1f}",
             ha='left', va='center', fontsize=11, fontweight='bold',
             color='#2C3E50')

plt.xlabel("Normalized Gain (%)", fontweight='bold')
plt.ylabel("Features", fontweight='bold')
plt.title("XGBoost Feature Importance (Non-PM2.5) — TOP 5 by Gain",
          fontweight='bold', pad=20)

# 美化网格线
plt.grid(True, axis='x', alpha=0.3, linestyle='--')
plt.gca().set_axisbelow(True)

plt.tight_layout()

# =========================
# 6. 保存图像
# =========================
plt.savefig(
    "xgboost_non_pm25_top5_gain.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
