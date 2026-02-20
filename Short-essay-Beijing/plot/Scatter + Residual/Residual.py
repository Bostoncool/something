# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# 用于让 colorbar 高度与主图一致
from mpl_toolkits.axes_grid1 import make_axes_locatable


def make_residual_plot(
    csv_path: str,
    pred_col: str | None = None,
    residual_col: str | None = None,
    cmap: str = "coolwarm",              # 残差图常用 coolwarm
    title: str = "Residual Plot",
    xlabel: str = "Predicted (μg/m³)",
    ylabel: str = "Residual (μg/m³)",
    fontfamily: str = "Times New Roman",
    legend_fontsize: int = 14,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 14,
    out_path: str = "residual_plot.svg",
    dpi: int = 300,
    point_size: int = 18,
    alpha: float = 0.85,
    show_grid: bool = True,
):
    """
    绘制残差图（残差 vs 预测值）
    
    参数:
        csv_path: CSV文件路径
        pred_col: 预测值列名（如果为None，自动检测）
        residual_col: 残差列名（如果为None，自动检测）
        cmap: 颜色映射，默认 "coolwarm"
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        fontfamily: 字体族
        legend_fontsize: 图例字体大小
        axis_label_fontsize: 坐标轴标签字体大小
        tick_label_fontsize: 刻度标签字体大小
        out_path: 输出文件路径
        dpi: 分辨率（SVG格式不需要，但保留兼容性）
        point_size: 散点大小
        alpha: 散点透明度
        show_grid: 是否显示网格
    """
    # 1) 读数据
    df = pd.read_csv(csv_path)
    
    if pred_col is None or residual_col is None:
        # 自动检测列名（与 Scatter-all.py 一致）
        # 优先检测 actual 和 predicted 列（从 predictions.csv）
        pm_actual_cols = [c for c in df.columns if 'actual' in c.lower() or 'true' in c.lower()]
        pm_pred_cols = [c for c in df.columns if 'pred' in c.lower() or 'forecast' in c.lower()]
        residual_cols = [c for c in df.columns if 'residual' in c.lower() or 'resid' in c.lower()]
        
        if len(pm_actual_cols) > 0 and len(pm_pred_cols) > 0:
            # 从 predictions.csv 读取，计算残差
            actual = df[pm_actual_cols[0]].to_numpy(float)
            predicted = df[pm_pred_cols[0]].to_numpy(float)
            residual = actual - predicted
        elif len(pm_pred_cols) > 0 and len(residual_cols) > 0:
            # 如果文件中有残差列，直接使用
            predicted = df[pm_pred_cols[0]].to_numpy(float)
            residual = df[residual_cols[0]].to_numpy(float)
        elif df.shape[1] >= 2:
            # 如果无法自动检测，使用默认列索引
            predicted = df.iloc[:, 0].to_numpy(float)
            if df.shape[1] >= 3:
                # 假设第1列是actual，第2列是predicted
                actual = df.iloc[:, 1].to_numpy(float)
                residual = actual - predicted
            else:
                residual = df.iloc[:, 1].to_numpy(float)
        else:
            raise ValueError("CSV 至少需要两列：预测值与残差（或实际值）。")
    else:
        predicted = df[pred_col].to_numpy(float)
        if residual_col is not None:
            residual = df[residual_col].to_numpy(float)
        else:
            # 如果只提供了 pred_col，尝试找 actual 列计算残差
            pm_actual_cols = [c for c in df.columns if 'actual' in c.lower() or 'true' in c.lower()]
            if len(pm_actual_cols) > 0:
                actual = df[pm_actual_cols[0]].to_numpy(float)
                residual = actual - predicted
            else:
                raise ValueError("无法找到实际值列来计算残差，请提供 residual_col 参数。")
    
    # 过滤无效值
    mask = np.isfinite(predicted) & np.isfinite(residual)
    predicted, residual = predicted[mask], residual[mask]
    n = predicted.size
    if n < 2:
        raise ValueError("有效数据点太少，无法绘图。")
    
    # 2) 设置对称的 y 轴范围
    max_abs = float(np.nanmax(np.abs(residual)))
    padding = 0.05 * max_abs if max_abs > 0 else 1.0
    ylim = max_abs + padding
    
    # 3) 设置统一字体
    plt.rcParams['font.family'] = 'serif' if fontfamily == 'Times New Roman' else fontfamily
    if fontfamily == 'Times New Roman':
        plt.rcParams['font.serif'] = ['Times New Roman']
    elif fontfamily not in ['serif', 'sans-serif', 'monospace']:
        plt.rcParams['font.sans-serif'] = [fontfamily]
    
    # 4) 创建正方形画布
    fig, ax = plt.subplots(figsize=(6.0, 6.0))  # 正方形画布
    
    # 5) 设置坐标轴范围，确保绘图区域为正方形
    x_min, x_max = float(predicted.min()), float(predicted.max())
    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    x_range = (x_max + x_pad) - (x_min - x_pad)
    y_range = 2 * ylim  # y轴是对称的，从 -ylim 到 ylim

    # 为了确保绘图区域为正方形，取x和y轴范围的最大值作为基准
    max_range = max(x_range, y_range)

    # 计算x轴和y轴的中心点
    x_center = (x_min + x_max) / 2
    y_center = 0.0  # y轴以0为中心

    # 设置正方形绘图区域
    half_range = max_range / 2
    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)

    # 确保绘图区域为正方形（通过设置等比例）
    ax.set_aspect("equal", adjustable="box")
    
    # 6) 四边实线边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_linestyle("-")
    
    # 7) 颜色映射（以 0 为中心）
    norm = TwoSlopeNorm(vmin=-ylim, vcenter=0.0, vmax=ylim)
    
    scatter = ax.scatter(
        predicted,
        residual,
        c=residual,
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=alpha,
        edgecolors="none"
    )
    
    # 8) y = 0 参考线
    ax.axhline(
        0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="y = 0"
    )
    
    # 9) 网格（可选）
    if show_grid:
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    
    # 10) 让 colorbar 高度严格匹配主图
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(scatter, cax=cax)
    # 移除colorbar的标签，避免与y轴标签重复
    
    # 11) 标题、轴标签、图例（统一字体）
    ax.set_title(title, fontweight="bold", fontfamily=fontfamily, fontsize=18)
    ax.set_xlabel(xlabel, fontfamily=fontfamily, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontfamily=fontfamily, fontsize=axis_label_fontsize)
    
    legend = ax.legend(
        loc="upper right",
        frameon=False,
        prop={'family': fontfamily, 'size': legend_fontsize},
        facecolor='none',
        edgecolor='none'
    )
    # 确保图例完全没有背景和边框
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('none')
    legend.get_frame().set_alpha(0)
    
    # 设置刻度标签字体和大小
    for label in ax.get_xticklabels():
        label.set_fontfamily(fontfamily)
        label.set_fontsize(tick_label_fontsize)
    for label in ax.get_yticklabels():
        label.set_fontfamily(fontfamily)
        label.set_fontsize(tick_label_fontsize)
    for label in cax.get_yticklabels():
        label.set_fontfamily(fontfamily)
        label.set_fontsize(tick_label_fontsize)
    
    fig.tight_layout()
    # SVG格式是矢量图，不需要DPI参数，但保留不影响
    # 确保输出为正方形：移除 bbox_inches='tight' 以保持完整的正方形画布
    fig.savefig(out_path, format='svg', dpi=dpi)
    plt.close(fig)  # 关闭图形以释放内存
    print("Saved:", out_path)


if __name__ == "__main__":
    # 定义9个文件路径及其对应的标题和输出文件名
    file_configs = [
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-BPNN\Split2\output\predictions.csv",
            "title": "BPNN-Optimized",
            "out_path": "BPNN-Optimized-Residual.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-CNN- LSTM-Transformer\C-L-N\predictions__transformer__test.csv",
            "title": "CNN-LSTM-Transformer",
            "out_path": "CNN-LSTM-Transformer-Residual.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-CNN-GridSearch\Split2\output\predictions.csv",
            "title": "CNN-Optimized",
            "out_path": "CNN-Optimized-Residual.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-MLR_GAM\Split2\output\predictions_optimized.csv",
            "title": "GAM-MLR-Optimized",
            "out_path": "GAM-MLR-Optimized-Residual.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-LightGBM\Split2\output\predictions.csv",
            "title": "LightGBM-Optimized",
            "out_path": "LightGBM-Optimized-Residual.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-RF\Split2\output\rf_predictions_nc.csv",
            "title": "RF-Optimized",
            "out_path": "RF-Optimized-Residual.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-SVR\Split2\output\time_series_prediction.csv",
            "title": "SVR-Optimized",
            "out_path": "SVR-Optimized-Residual.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-Transformer\Split2\output\predictions_test.csv",
            "title": "Transformer-Optimized",
            "out_path": "Transformer-Optimized-Residual.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-XGBOOST\CSV\output\xgboost_predictions__xgboost_optimized__test.csv",
            "title": "XGBOOST-Optimized",
            "out_path": "XGBOOST-Optimized-Residual.svg"
        }
    ]
    
    # 循环绘制9张图
    for i, config in enumerate(file_configs, 1):
        try:
            make_residual_plot(
                csv_path=config['path'],
                cmap="coolwarm",
                title=config['title'],
                xlabel="Predicted (μg/m³)",
                ylabel="Residual (μg/m³)",
                fontfamily="Times New Roman",
                out_path=config['out_path'],
            )
        except Exception as e:
            print(f"Failed to process: {config['title']}")
            print(f"Error: {str(e)}")
            continue
