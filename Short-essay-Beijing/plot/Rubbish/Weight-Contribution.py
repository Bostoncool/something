import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11

# Professional color palette inspired by top-tier journals (colorblind-friendly)
# Using a palette similar to Nature/Science publications
COLORS = [
    '#2E86AB',  # Professional blue
    '#A23B72',  # Deep purple
    '#F18F01',  # Warm orange
    '#06A77D',  # Teal green
    '#D90429',  # Deep red
    '#7209B7',  # Purple
    '#F77F00',  # Orange
    '#FCBF49',  # Golden yellow
    '#6A4C93',  # Lavender
    '#118AB2',  # Sky blue
    '#6C757D'   # Neutral gray for "Others"
]

# ======================
# 1. Parameter Settings (Modifiable)
# ======================
csv_path = r"H:\DATA Science\小论文Result\Fine_model\-XGBOOST\XGBOOST\plot_feature_importance__xgboost_optimized.csv"
output_dir = r"H:\DATA Science\小论文Result\Weight"  # 图片输出目录
top_n = 10                     # Maximum number of features to display

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# ======================
# 2. Read and Preprocess Data
# ======================
df = pd.read_csv(csv_path)
feature_col = df.columns[0]  # Use the first column for features
weight_col = df.columns[1]   # Use the second column for weights

# Filter: only keep data with weight > 0
df = df[df[weight_col] > 0]

# Sort by weight in descending order
df = df.sort_values(by=weight_col, ascending=False)

# Take top top_n features
df_top = df.iloc[:top_n].copy()

# Group the rest as "Others"
if len(df) > top_n:
    others_weight = df.iloc[top_n:][weight_col].sum()
    df_others = pd.DataFrame({
        feature_col: ["Others"],
        weight_col: [others_weight]
    })
    df_plot = pd.concat([df_top, df_others], ignore_index=True)
else:
    df_plot = df_top

# Calculate percentage (for more intuitive display)
df_plot["percentage"] = df_plot[weight_col] / df_plot[weight_col].sum() * 100

# ======================
# 3. Draw Donut Chart
# ======================
fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor('none')  # 透明背景
ax.set_facecolor('none')  # 透明轴背景

# Assign colors to each segment
n_segments = len(df_plot)
colors = COLORS[:n_segments]
# Use gray for "Others" if it exists
if "Others" in df_plot[feature_col].values:
    others_idx = df_plot[feature_col].tolist().index("Others")
    colors[others_idx] = COLORS[-1]  # Use gray for "Others"

wedges, texts, autotexts = ax.pie(
    df_plot[weight_col],
    labels=df_plot[feature_col],
    autopct="%.1f%%",
    startangle=90,
    pctdistance=0.85,
    colors=colors,
    textprops={'fontsize': 10, 'fontfamily': 'Times New Roman'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)

# Style the percentage text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)

# Create hollow center to form donut chart
centre_circle = plt.Circle((0, 0), 0.60, fc="white", edgecolor='lightgray', linewidth=1)
ax.add_artist(centre_circle)

ax.set_title("XGBOOST-Optimized Donut Chart (Top {} + Others)".format(top_n), 
             fontsize=14, fontweight='bold', pad=20)
ax.axis("equal")

plt.tight_layout()
# 保存图片，背景透明
output_path_donut = os.path.join(output_dir, 'XGBOOST-Optimized_donut_chart.png')
plt.savefig(output_path_donut, dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# ======================
# 4. Draw Horizontal Bar Chart
# ======================
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('none')  # 透明背景
ax.set_facecolor('none')  # 透明轴背景

# Assign colors to each bar
n_bars = len(df_plot)
bar_colors = COLORS[:n_bars]
# Use gray for "Others" if it exists
if "Others" in df_plot[feature_col].values:
    others_idx = df_plot[feature_col].tolist().index("Others")
    bar_colors[others_idx] = COLORS[-1]  # Use gray for "Others"

bars = ax.barh(
    df_plot[feature_col],
    df_plot["percentage"],
    color=bar_colors,
    edgecolor='white',
    linewidth=1.2,
    height=0.7
)

ax.invert_yaxis()  # Highest weight at the top
ax.set_xlabel("Percentage (%)", fontsize=12, fontweight='bold')
ax.set_ylabel("Feature", fontsize=12, fontweight='bold')
ax.set_title("XGBOOST-Optimized Horizontal Bar Chart (Top {} + Others)".format(top_n),
             fontsize=14, fontweight='bold', pad=15)

# Style the axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('lightgray')
ax.spines['bottom'].set_color('lightgray')
ax.grid(axis='x', linestyle='--', alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)

# Annotate values at the end of bars (as percentage)
for i, (pct, bar) in enumerate(zip(df_plot["percentage"], bars)):
    ax.text(pct + max(df_plot["percentage"]) * 0.01, i, f"{pct:.1f}%", 
            va="center", ha="left", fontsize=10, fontweight='bold',
            fontfamily='Times New Roman')

plt.tight_layout()
# 保存图片，背景透明
output_path_bar = os.path.join(output_dir, 'XGBOOST-Optimized_bar_chart.png')
plt.savefig(output_path_bar, dpi=300, bbox_inches='tight', transparent=True)
plt.show()
