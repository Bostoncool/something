import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings('ignore')

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # Display minus sign correctly
plt.rcParams['font.size'] = 11

# Read data
df = pd.read_csv('c:\\Users\\IU\\Desktop\\something\\lower than 75.csv')

# Clean column names (remove spaces)
df.columns = df.columns.str.strip()

# Clean Model column values (remove leading spaces and hyphens)
df['Model'] = df['Model'].str.strip().str.lstrip('-')

# Convert to numeric types
df['R2'] = pd.to_numeric(df['R2'], errors='coerce')
df['RMSE'] = pd.to_numeric(df['RMSE'], errors='coerce')
df['MAE'] = pd.to_numeric(df['MAE'], errors='coerce')
df['MAPE'] = pd.to_numeric(df['MAPE'], errors='coerce')

# Gradient functions for beautiful bar charts
def gradient_image(ax, extent, direction=0, cmap_range=(0, 0.5), **kwargs):
    """Create gradient image for bar chart"""
    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    X = np.array([[v @ [0, 0], v @ [0, 0]],
                  [v @ [1, 1], v @ [1, 1]]])
    a, b = cmap_range
    X = a + (b - a) / X.max() * X
    im = ax.imshow(X, extent=extent, interpolation='bicubic',
                   vmin=0, vmax=1, aspect='auto', **kwargs)
    return im

def gradient_barh(ax, y_positions, x_values, height=0.5, left=0, colors=None):
    """Create horizontal gradient bars, supporting both positive and negative values"""
    if colors is None:
        colors = [(114 / 255, 188 / 255, 213 / 255), (1, 1, 1)]
    
    cmap = LinearSegmentedColormap.from_list('my_cmap', colors, N=256)
    
    for y_pos, x_val in zip(y_positions, x_values):
        bottom = y_pos - height / 2
        top = y_pos + height / 2
        
        # Handle negative values: draw from left to x_val (which can be negative)
        # For negative values, extent will be (x_val, left) to draw leftward
        if x_val < 0:
            # For negative values, draw from x_val to left (0)
            gradient_image(ax, extent=(x_val, left, bottom, top),
                           direction=0, cmap=cmap, cmap_range=(0, 0.8))
        else:
            # For positive values, draw from left (0) to x_val
            gradient_image(ax, extent=(left, x_val, bottom, top),
                           direction=0, cmap=cmap, cmap_range=(0, 0.8))

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Evaluation Metrics Comparison (PM2.5 < 75µg/m³)', fontsize=16, fontweight='bold', 
             fontfamily='Times New Roman')

# Define color schemes for each metric using specified hex colors
# Convert hex to RGB (0-1 range) and create gradients
color_schemes = {
    'R2': [(108/255, 163/255, 212/255), (200/255, 225/255, 245/255)],  # #6CA3D4 -> light blue
    'RMSE': [(225/255, 128/255, 126/255), (245/255, 200/255, 200/255)],  # #E1807E -> light red
    'MAE': [(97/255, 193/255, 191/255), (180/255, 235/255, 233/255)],  # #61C1BF -> light cyan
    'MAPE': [(191/255, 149/255, 193/255), (230/255, 210/255, 232/255)]  # #BF95C1 -> light purple
}

# Helper function to create gradient bar chart with labels
def create_gradient_barh(ax, df_sorted, metric, colors, y_positions, height=0.6):
    """Create horizontal gradient bar chart with value labels"""
    values = df_sorted[metric].values
    models = df_sorted['Model'].values
    
    # ✅ Key fix: Set y-axis limits to include full height of top bar
    ax.set_ylim(-0.5, len(values) - 0.5)
    
    # Create gradient bars
    gradient_barh(ax, y_positions, values, height=height, left=0, colors=colors)
    
    # Add value labels on bars
    x_max = values.max()
    x_min = values.min()
    x_range = x_max - x_min if x_max != x_min else (abs(x_max) if x_max != 0 else 1)
    
    for y_pos, val in zip(y_positions, values):
        # Format value based on metric
        if metric == 'R2':
            label_text = f'{val:.3f}'
        elif metric == 'MAPE':
            label_text = f'{val:.2f}%'
        else:
            label_text = f'{val:.2f}'
        
        # Position label at the end of bar with small offset
        # For negative values, position label to the left
        if val < 0:
            x_pos = val - x_range * 0.03  # Offset to the left for negative values
            ha = 'right'
        else:
            x_pos = val + x_range * 0.03  # Offset to the right for positive values
            ha = 'left'
        
        ax.text(x_pos, y_pos, label_text,
                ha=ha, va='center',
                fontsize=10, fontweight='bold',
                fontfamily='Times New Roman',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='none', alpha=0.85))
    
    # Adjust x-axis limits to accommodate labels (only if not manually set later)
    # For R2, this will be overridden, but for others it's fine
    if x_min >= 0:
        ax.set_xlim(left=0, right=x_max + x_range * 0.15)
    else:
        # For metrics with negative values, extend both sides
        ax.set_xlim(left=x_min - x_range * 0.15, right=x_max + x_range * 0.15)
    
    return models

# 1. R2 bar chart (descending order)
df_r2 = df.sort_values('R2', ascending=False)
y_positions_r2 = np.arange(len(df_r2))
models_r2 = create_gradient_barh(axes[0, 0], df_r2, 'R2', color_schemes['R2'], y_positions_r2)
axes[0, 0].set_yticks(y_positions_r2)
axes[0, 0].set_yticklabels(models_r2)
axes[0, 0].set_xlabel('R²', fontsize=12, fontfamily='Times New Roman')
axes[0, 0].set_title('R² Score (Descending)', fontsize=13, fontweight='bold', fontfamily='Times New Roman')
axes[0, 0].set_xlim(left=-2.5, right=1)  # Set X-axis range from -2 to 1
axes[0, 0].grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
axes[0, 0].invert_yaxis()
axes[0, 0].spines['top'].set_visible(False)
axes[0, 0].spines['right'].set_visible(False)
axes[0, 0].spines['left'].set_color('#CCCCCC')
axes[0, 0].spines['bottom'].set_color('#CCCCCC')

# 2. RMSE bar chart (ascending order)
df_rmse = df.sort_values('RMSE', ascending=True)
y_positions_rmse = np.arange(len(df_rmse))
models_rmse = create_gradient_barh(axes[0, 1], df_rmse, 'RMSE', color_schemes['RMSE'], y_positions_rmse)
axes[0, 1].set_yticks(y_positions_rmse)
axes[0, 1].set_yticklabels(models_rmse)
axes[0, 1].set_xlabel('RMSE', fontsize=12, fontfamily='Times New Roman')
axes[0, 1].set_title('RMSE (Ascending)', fontsize=13, fontweight='bold', fontfamily='Times New Roman')
axes[0, 1].grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
axes[0, 1].invert_yaxis()
axes[0, 1].spines['top'].set_visible(False)
axes[0, 1].spines['right'].set_visible(False)
axes[0, 1].spines['left'].set_color('#CCCCCC')
axes[0, 1].spines['bottom'].set_color('#CCCCCC')

# 3. MAE bar chart (ascending order)
df_mae = df.sort_values('MAE', ascending=True)
y_positions_mae = np.arange(len(df_mae))
models_mae = create_gradient_barh(axes[1, 0], df_mae, 'MAE', color_schemes['MAE'], y_positions_mae)
axes[1, 0].set_yticks(y_positions_mae)
axes[1, 0].set_yticklabels(models_mae)
axes[1, 0].set_xlabel('MAE', fontsize=12, fontfamily='Times New Roman')
axes[1, 0].set_title('MAE (Ascending)', fontsize=13, fontweight='bold', fontfamily='Times New Roman')
axes[1, 0].grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
axes[1, 0].invert_yaxis()
axes[1, 0].spines['top'].set_visible(False)
axes[1, 0].spines['right'].set_visible(False)
axes[1, 0].spines['left'].set_color('#CCCCCC')
axes[1, 0].spines['bottom'].set_color('#CCCCCC')

# 4. MAPE bar chart (ascending order)
df_mape = df.sort_values('MAPE', ascending=True)
y_positions_mape = np.arange(len(df_mape))
models_mape = create_gradient_barh(axes[1, 1], df_mape, 'MAPE', color_schemes['MAPE'], y_positions_mape)
axes[1, 1].set_yticks(y_positions_mape)
axes[1, 1].set_yticklabels(models_mape)
axes[1, 1].set_xlabel('MAPE (%)', fontsize=12, fontfamily='Times New Roman')
axes[1, 1].set_title('MAPE (Ascending)', fontsize=13, fontweight='bold', fontfamily='Times New Roman')
axes[1, 1].grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
axes[1, 1].invert_yaxis()
axes[1, 1].spines['top'].set_visible(False)
axes[1, 1].spines['right'].set_visible(False)
axes[1, 1].spines['left'].set_color('#CCCCCC')
axes[1, 1].spines['bottom'].set_color('#CCCCCC')

# Set font for all text elements (titles, labels, tick labels)
for ax in axes.flat:
    ax.title.set_fontfamily('Times New Roman')
    ax.xaxis.label.set_fontfamily('Times New Roman')
    ax.yaxis.label.set_fontfamily('Times New Roman')
    for label in ax.get_xticklabels():
        label.set_fontfamily('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

plt.tight_layout()
plt.savefig('c:\\Users\\IU\\Desktop\\something\\Short-Essay-Beijing\\Plot\\model_evaluation_bar_charts_lower than 75.png', 
            dpi=300, bbox_inches='tight')
print("Bar charts saved as: model_evaluation_bar_charts.png")
plt.show()
