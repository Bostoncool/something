"""显示颜色渐变条：从深蓝到红色"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 支持中文显示（Windows 常用字体）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# 颜色列表（12色：深蓝 -> 红，根据色带推测）
colors = [
    "#003366",  # 深蓝色
    "#3366CC",  # 蓝色
    "#6699CC",  # 浅蓝色
    "#99CCFF",  # 淡蓝色
    "#CCFFFF",  # 青色
    "#CCFFCC",  # 浅绿色
    "#99CC99",  # 绿色
    "#99CC66",  # 黄绿色
    "#FFFF99",  # 黄色
    "#FFCC66",  # 橙色
    "#FF6666",  # 红色
    "#CC3333",  # 深红色
]

fig, axes = plt.subplots(2, 1, figsize=(14, 4))

# 方式1：水平渐变条
ax1 = axes[0]
for i, c in enumerate(colors):
    ax1.axvspan(i, i + 1, facecolor=c, edgecolor="white", linewidth=0.5)
ax1.set_xlim(0, len(colors))
ax1.set_ylim(0, 1)
ax1.set_aspect("equal")
ax1.axis("off")
ax1.set_title("颜色渐变条（水平）")

# 方式2：色块 + 色码标注
ax2 = axes[1]
for i, c in enumerate(colors):
    rect = mpatches.Rectangle((i, 0), 0.9, 1, facecolor=c, edgecolor="gray")
    ax2.add_patch(rect)
    ax2.text(i + 0.45, -0.15, c, ha="center", va="top", fontsize=8, rotation=45)
ax2.set_xlim(-0.2, len(colors))
ax2.set_ylim(-0.6, 1.2)
ax2.set_aspect("equal")
ax2.axis("off")
ax2.set_title("色块及十六进制色码")

plt.tight_layout()
plt.savefig("color_swatch.svg", dpi=150, bbox_inches="tight")
plt.show()
