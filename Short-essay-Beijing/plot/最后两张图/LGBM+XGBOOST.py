# -*- coding: utf-8 -*-
"""
Feature Importance Analysis: XGBoost vs LightGBM Comparison
Font: Times New Roman
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Set Times New Roman font for all text elements
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11

# ====== 1) Read two feature importance files ======
xgb_path = r"H:\DATA Science\小论文Result\Fine_model\-XGBOOST\XGBOOST\plot_feature_importance__xgboost_optimized.csv"
lgb_path = r"H:\DATA Science\小论文Result\Fine_model\-LightGBM\Split2\output\feature_importance.csv"

xgb = pd.read_csv(xgb_path)
lgb = pd.read_csv(lgb_path)

# ====== 2) Grouping rules: PM2.5-related / Meteorology (non-PM2.5) / Non-meteorology (non-PM2.5) ======
pm_pattern = re.compile(r"PM\s*2\.?5|PM25", re.IGNORECASE)

meteo_keywords = [
    "wind", "u10", "v10", "ws", "wd", "wind_dir", "wind_speed",
    "humidity", "rh", "temp", "t2m", "dew", "dewpoint", "blh",
    "pbl", "boundary", "str", "tisr", "ssr", "rad", "radiation",
    "pressure", "psl", "mslp", "precip", "rain", "snow", "cloud",
    "tcc", "visibility", "vis", "albedo"
]
meteo_pattern = re.compile(r"(" + "|".join([re.escape(k) for k in meteo_keywords]) + r")", re.IGNORECASE)

def group_feature(name: str) -> str:
    """
    Classify features into three categories:
    - PM2.5-related
    - Meteorology (non-PM2.5)
    - Non-meteorology (non-PM2.5)
    """
    s = str(name)
    if pm_pattern.search(s):
        return "PM2.5-related"
    if meteo_pattern.search(s):
        return "Meteorology (non-PM2.5)"
    return "Non-meteorology (non-PM2.5)"

def summarize(df: pd.DataFrame, gain_norm_col="Importance_Gain_Norm") -> pd.Series:
    """
    Summarize feature importance by group categories.

    Parameters:
    - df: DataFrame containing feature importance data
    - gain_norm_col: Column name for normalized gain importance

    Returns:
    - Series with summed importance for each group category
    """
    tmp = df.copy()
    tmp["Group"] = tmp["Feature"].apply(group_feature)
    sums = tmp.groupby("Group")[gain_norm_col].sum()

    # Ensure all three categories exist
    for g in ["PM2.5-related", "Meteorology (non-PM2.5)", "Non-meteorology (non-PM2.5)"]:
        sums[g] = float(sums.get(g, 0.0))

    return sums[["PM2.5-related", "Meteorology (non-PM2.5)", "Non-meteorology (non-PM2.5)"]]

xgb_sums = summarize(xgb)
lgb_sums = summarize(lgb)

summary = pd.DataFrame(
    [xgb_sums, lgb_sums],
    index=["XGBoost", "LightGBM"]
)

# Normalize within rows to 100% (more stable approach)
summary = summary.div(summary.sum(axis=1), axis=0) * 100

# ====== 3) Plot stacked bar chart with optimized colors and broken y-axis ======
# Optimized color scheme (professional and colorblind-friendly)
colors = {
    "PM2.5-related": "#2E86AB",           # Professional blue
    "Meteorology (non-PM2.5)": "#A23B72", # Elegant purple
    "Non-meteorology (non-PM2.5)": "#F18F01"  # Warm orange
}

# Create figure with broken y-axis (reversed: upper part shows 60-100%, lower part shows 0-60%)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), 
                                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

# Prepare data for plotting
x_pos = np.arange(len(summary.index))
bar_width = 0.4  # Half of default width (default is 0.8)

label_map = {
    "PM2.5-related": "PM2.5-related",
    "Meteorology (non-PM2.5)": "Meteorology (non-PM2.5)",
    "Non-meteorology (non-PM2.5)": "Non-meteorology (non-PM2.5)"
}

# Plot upper axis (60-100%) - all segments with proper stacking
# Calculate the PM2.5 portion above 60%
pm25_above_60 = (summary["PM2.5-related"] - 60).clip(lower=0)

# Stack segments on upper axis
bottom_upper = pd.Series([60.0, 60.0], index=summary.index)

# PM2.5-related (above 60% only)
ax1.bar(x_pos, pm25_above_60, bottom=bottom_upper, width=bar_width,
        label=label_map["PM2.5-related"], color=colors["PM2.5-related"], 
        edgecolor='white', linewidth=0.5)
bottom_upper += pm25_above_60

# Meteorology (non-PM2.5)
ax1.bar(x_pos, summary["Meteorology (non-PM2.5)"], bottom=bottom_upper, width=bar_width,
        label=label_map["Meteorology (non-PM2.5)"], color=colors["Meteorology (non-PM2.5)"], 
        edgecolor='white', linewidth=0.5)
bottom_upper += summary["Meteorology (non-PM2.5)"]

# Non-meteorology (non-PM2.5)
ax1.bar(x_pos, summary["Non-meteorology (non-PM2.5)"], bottom=bottom_upper, width=bar_width,
        label=label_map["Non-meteorology (non-PM2.5)"], color=colors["Non-meteorology (non-PM2.5)"], 
        edgecolor='white', linewidth=0.5)

# Plot lower axis (0-60%) - only PM2.5-related features that are <= 60%
pm25_lower = summary["PM2.5-related"].clip(upper=60)
ax2.bar(x_pos, pm25_lower, width=bar_width, 
        color=colors["PM2.5-related"], edgecolor='white', linewidth=0.5)

# Configure upper axis (60-100%) - now ax1
ax1.set_ylim(60, 100)
ax1.set_xlim(-0.5, len(summary.index) - 0.5)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([])  # Hide x-axis labels on upper plot
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(labeltop=False, labelbottom=False, labelsize=10)
ax1.set_yticks([60, 70, 80, 90, 100])
ax1.set_yticklabels(['60', '70', '80', '90', '100'], fontfamily='Times New Roman')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Configure lower axis (0-60%) - now ax2
ax2.set_ylim(0, 60)
ax2.set_xlim(-0.5, len(summary.index) - 0.5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(summary.index, fontfamily='Times New Roman')  # Show x-axis labels on lower plot
ax2.spines['top'].set_visible(False)
ax2.tick_params(labeltop=False, labelbottom=True, labelsize=10)
ax2.set_ylabel("Normalized Gain Share (%)", fontfamily='Times New Roman', fontsize=11)
ax2.set_yticks([0, 10, 20, 30, 40, 50, 60])
ax2.set_yticklabels(['0', '10', '20', '30', '40', '50', '60'], fontfamily='Times New Roman')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add break marks (adjusted for reversed layout)
d = 0.015  # Size of diagonal lines
# Upper axis break marks (at bottom of ax1)
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1-d, 1+d), (-d, +d), **kwargs)

# Lower axis break marks (at top of ax2)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

# Add title and legend
fig.suptitle("Feature-importance decomposition (Full model, normalized Gain)", 
             fontfamily='Times New Roman', fontsize=12, y=0.98)
ax1.legend(loc='lower left', framealpha=0.9, fontsize=9, prop={'family': 'Times New Roman'})

plt.tight_layout()
plt.savefig("model_importance_decomposition_stacked_en.png", dpi=300, bbox_inches="tight")
plt.show()
