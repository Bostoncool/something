import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter

# CSV file path
path = r"E:\DATA Science\小论文Result\Fine_model\-SVR\Split2\output\time_series_prediction.csv"

# Read CSV
df = pd.read_csv(path)

# Try to infer date and PM2.5 columns
date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]   
pm_actual_cols = [c for c in df.columns if 'actual' in c.lower() or 'true' in c.lower()]
pm_pred_cols = [c for c in df.columns if 'pred' in c.lower() or 'forecast' in c.lower()]

date_col = date_cols[0]
actual_col = pm_actual_cols[0]
pred_col = pm_pred_cols[0]

# Parse dates and filter year 2024
df[date_col] = pd.to_datetime(df[date_col])
df_2024 = df[df[date_col].dt.year == 2024].copy()

# Sort by date
df_2024.sort_values(date_col, inplace=True)

# Create figure: 16:9 rectangle
fig, ax = plt.subplots(figsize=(18, 6))

# Plot actual and predicted PM2.5
ax.plot(
    df_2024[date_col],
    df_2024[actual_col],
    linestyle='-',
    linewidth=1,
    color='black',
    label='Actual'
)
ax.plot(
    df_2024[date_col],
    df_2024[pred_col],
    linestyle='--',
    linewidth=1,
    color='green',
    label='Predicted'
)

# Reference line y=75 (mild pollution threshold and above)
ax.axhline(75, linestyle='--', linewidth=1.5, color='red', label=' 75 (µg/m³)')

# Reference line y=35 (great and above)
ax.axhline(35, linestyle='--', linewidth=1.5, color='blue', label=' 35 (µg/m³)')

# Title and axis labels
ax.set_title('SVR-Optimized', fontsize=24, fontweight='bold', fontfamily='Times New Roman', pad=8)
ax.set_xlabel('Date', fontsize=24, fontfamily='Times New Roman')
ax.set_ylabel('PM2.5 (µg/m³)', fontsize=24, fontfamily='Times New Roman')

# X-axis formatting: rotate labels 45 degrees and show fewer months
ax.xaxis.set_major_locator(MonthLocator(interval=2))
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontfamily='Times New Roman')
plt.setp(ax.yaxis.get_majorticklabels(), fontfamily='Times New Roman')

# Ensure all four spines are solid and visible
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)

# Legend and layout
ax.legend(prop={'family': 'Times New Roman', 'size': 20})
plt.tight_layout()

# Save figure (使用figure对象保存，避免show()的影响)
out_path = "SVR-Optimized.svg"
fig.savefig(out_path, format='svg', bbox_inches='tight', facecolor='white')
# 显示图形
plt.show()
# 关闭图形以释放内存
plt.close(fig)

print(f"Saved figure to: {out_path}")
