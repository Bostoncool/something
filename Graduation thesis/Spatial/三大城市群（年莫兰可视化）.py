import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 准备你的数据
data = {
    "cluster": ["BTH", "BTH", "BTH", "BTH", "BTH", "BTH",
                "YRD", "YRD", "YRD", "YRD", "YRD", "YRD",
                "PRD", "PRD", "PRD", "PRD", "PRD", "PRD"],
    "period": [2018, 2019, 2020, 2021, 2022, 2023,
               2018, 2019, 2020, 2021, 2022, 2023,
               2018, 2019, 2020, 2021, 2022, 2023],
    "global_i": [0.936596, 0.945348, 0.94398, 0.94037, 0.941071, 0.91094,
                 0.931823, 0.928732, 0.929577, 0.942422, 0.925283, 0.919213,
                 0.771601, 0.727518, 0.741785, 0.781802, 0.851239, 0.798094]
}
df = pd.DataFrame(data)

# 2. 设置学术绘图风格
sns.set_style("white")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# 3. 创建画布
plt.figure(figsize=(10, 6))

# 4. 绘制散点图
# 京津冀(BTH)：深蓝色圆点
# 长三角(YRD)：蓝绿色圆点
# 珠三角(PRD)：黄色圆点
scatter = sns.scatterplot(
    data=df,
    x="period",
    y="global_i",
    hue="cluster",
    palette="viridis",
    s=150,
    edgecolor="black",
    alpha=0.9
)

# 5. 添加趋势线（每条城市群一条虚线）
for cluster in df["cluster"].unique():
    cluster_data = df[df["cluster"] == cluster]
    z = np.polyfit(cluster_data["period"], cluster_data["global_i"], 1)
    p = np.poly1d(z)
    plt.plot(cluster_data["period"], p(cluster_data["period"]), 
             linestyle="--", color=scatter.get_lines()[0].get_color() if cluster == "BTH" else 
             scatter.get_lines()[1].get_color() if cluster == "YRD" else 
             scatter.get_lines()[2].get_color(), alpha=0.7)

# 6. 图表美化（论文标准）
plt.title("Spatial Agglomeration of PM2.5 in Three Major Urban Agglomerations (2018-2023)", 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Year", fontsize=12, labelpad=10)
plt.ylabel("Global Moran's I", fontsize=12, labelpad=10)
plt.xticks(df['period'].unique(), fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(0.7, 0.98)  # 固定纵轴范围，突出差异
plt.axhline(y=0.85, color='lightgray', linestyle=':', linewidth=1)  # 参考线
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.legend(title="Urban Agglomeration", loc='lower right', fontsize=10, title_fontsize=11)
sns.despine()

# 7. 保存并显示
plt.tight_layout()
plt.savefig("PM25_Global_Moran_I_Scatter.png", dpi=300, bbox_inches='tight')  # 保存为300dpi高清图
print("图片已保存为 PM25_Global_Moran_I_Scatter.png")
plt.show()