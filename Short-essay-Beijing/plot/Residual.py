import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# =========================
# 1. 读取 CSV 数据
# =========================
csv_path = r"H:\DATA Science\小论文Result\Fine_model\-MLR_GAM\CSV\output\plot_residuals__mlr.csv"
df = pd.read_csv(csv_path)

predicted = df["Predicted_PM25"].to_numpy()
residual = df["Residual"].to_numpy()

# =========================
# 2. 设置对称的 y 轴范围
# =========================
max_abs = np.nanmax(np.abs(residual))
padding = 0.05 * max_abs   # 5% 留白
ylim = max_abs + padding

# =========================
# 3. 创建正方形画布
# =========================
fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

# =========================
# 4. 颜色映射（以 0 为中心）
# =========================
norm = TwoSlopeNorm(vmin=-ylim, vcenter=0.0, vmax=ylim)

scatter = ax.scatter(
    predicted,
    residual,
    c=residual,
    cmap="coolwarm",
    norm=norm,
    s=18,
    alpha=0.85,
    edgecolors="none"
)

# =========================
# 5. y = 0 参考线
# =========================
ax.axhline(
    0,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="y = 0"
)

# =========================
# 6. 坐标轴、标题
# =========================
ax.set_title("GAM-MLR-Optimized", fontsize=18, fontweight="bold", fontfamily="Times New Roman", pad=10)
ax.set_xlabel("Predicted PM2.5 (µg/m³)", fontsize=12, fontfamily="Times New Roman")
ax.set_ylabel("Residual (µg/m³)", fontsize=12, fontfamily="Times New Roman")

ax.set_ylim(-ylim, ylim)

# =========================
# 7. 网格（科研风格）
# =========================
ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)

# =========================
# 8. 四边实线边框（关键要求）
# =========================
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_linestyle("-")

# =========================
# 9. 确保绘图区为正方形
# =========================
try:
    ax.set_box_aspect(1)   # matplotlib >= 3.3
except Exception:
    ax.set_aspect("equal", adjustable="box")

# =========================
# 10. 颜色条
# =========================
cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Residual (µg/m³)", fontsize=11, fontfamily="Times New Roman")

# =========================
# 11. 图例
# =========================
ax.legend(frameon=False, loc="upper right")

# =========================
# 12. 保存为正方形图片
# =========================
output_path = "GAM-MLR-Optimized.png"
fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

print(f"Residual plot saved to: {output_path}")
