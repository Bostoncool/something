# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 用于让 colorbar 高度与主图一致
from mpl_toolkits.axes_grid1 import make_axes_locatable

# density scatter 需要（可选）
from scipy.stats import gaussian_kde


def compute_metrics(obs: np.ndarray, pred: np.ndarray):
    r2 = r2_score(obs, pred)
    rmse = float(np.sqrt(mean_squared_error(obs, pred)))
    mae = float(mean_absolute_error(obs, pred))

    eps = 1e-12
    denom = np.where(np.abs(obs) < eps, np.nan, obs)
    mape = float(np.nanmean(np.abs((obs - pred) / denom)) * 100)

    return r2, rmse, mae, mape


def fit_regression(obs: np.ndarray, pred: np.ndarray):
    x = obs.reshape(-1, 1)
    y = pred.reshape(-1, 1)
    lr = LinearRegression().fit(x, y)
    slope = float(lr.coef_.ravel()[0])
    intercept = float(lr.intercept_.ravel()[0])
    pred_fit = lr.predict(x).ravel()
    return slope, intercept, pred_fit


def make_plot(
    csv_path: str,
    obs_col: str | None = None,
    pred_col: str | None = None,
    method: str = "hexbin",          # "hexbin" 或 "density"
    cmap: str = "jet",               # 参考图风格：jet；你也可用 "turbo" / "rainbow" / "viridis" 等
    gridsize: int = 60,              # hexbin 分辨率，越大越细
    bins: str | None = None,         # hexbin: None 或 "log"（点数跨度很大时用 "log"）
    title: str = "Your Title",
    xlabel: str = "Actual (μg/m³)",       # 自定义横坐标名称
    ylabel: str = "Predicted (μg/m³)",       # 自定义纵坐标名称
    fontfamily: str = "Times New Roman",       # 统一字体，常用: "Arial", "Times New Roman", "SimHei"(黑体), "SimSun"(宋体)
    legend_fontsize: int = 14,                 # 图例字体大小
    axis_label_fontsize: int = 16,             # X轴和Y轴标签字体大小
    tick_label_fontsize: int = 14,             # X轴和Y轴刻度标签字体大小
    out_path: str = "plot_hexbin.svg",
    dpi: int = 300,
):
    # 1) 读数据
    df = pd.read_csv(csv_path)
    if obs_col is None or pred_col is None:
        # 自动检测列名（与 Prediction 文件夹中的代码一致）
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        pm_actual_cols = [c for c in df.columns if 'actual' in c.lower() or 'true' in c.lower()]
        pm_pred_cols = [c for c in df.columns if 'pred' in c.lower() or 'forecast' in c.lower()]
        
        if len(pm_actual_cols) > 0 and len(pm_pred_cols) > 0:
            obs_col = pm_actual_cols[0]
            pred_col = pm_pred_cols[0]
            obs = df[obs_col].to_numpy(float)
            pred = df[pred_col].to_numpy(float)
        elif df.shape[1] >= 2:
            # 如果无法自动检测，使用默认列索引
            obs = df.iloc[:, 1].to_numpy(float)
            pred = df.iloc[:, 2].to_numpy(float)
        else:
            raise ValueError("CSV 至少需要两列：观测值与预测值。")
        xlab = xlabel
        ylab = ylabel
    else:
        obs = df[obs_col].to_numpy(float)
        pred = df[pred_col].to_numpy(float)
        xlab = xlabel
        ylab = ylabel

    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    n = obs.size
    if n < 2:
        raise ValueError("有效数据点太少，无法绘图。")

    # 2) 回归 + 指标
    slope, intercept, pred_fit = fit_regression(obs, pred)
    r2, rmse, mae, mape = compute_metrics(obs, pred)

    # 3) 设置统一字体
    plt.rcParams['font.family'] = 'serif' if fontfamily == 'Times New Roman' else fontfamily
    if fontfamily == 'Times New Roman':
        plt.rcParams['font.serif'] = ['Times New Roman']
    elif fontfamily not in ['serif', 'sans-serif', 'monospace']:
        plt.rcParams['font.sans-serif'] = [fontfamily]
    
    # 4) 坐标范围（正方形）
    min_val = float(min(obs.min(), pred.min()))
    max_val = float(max(obs.max(), pred.max()))
    pad = 0.05 * (max_val - min_val) if max_val > min_val else 1.0  # 增加边距从2%到5%，避免线条与边缘重叠
    lim0, lim1 = min_val - pad, max_val + pad

    # 5) 画图：正方形 + 四边框
    fig, ax = plt.subplots(figsize=(6.0, 6.0))  # 正方形
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lim0, lim1)
    ax.set_ylim(lim0, lim1)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_linestyle("-")

    # 6) 密度可视化（二选一）
    mappable = None

    if method.lower() == "hexbin":
        # hexbin: C 默认是计数；mincnt=1 避免空格子
        hb = ax.hexbin(
            obs, pred,
            gridsize=gridsize,
            cmap=cmap,
            mincnt=1,
            bins=bins,     # None 或 "log"
            linewidths=0.0
        )
        mappable = hb
        cbar_label = "Counts" if bins is None else "log(Counts)"

    elif method.lower() == "density":
        # density scatter: 用 KDE 得到每个点密度，颜色=密度
        xy = np.vstack([obs, pred])
        z = gaussian_kde(xy)(xy)

        # 让高密度点后画（更好看）
        idx = np.argsort(z)
        obs_s, pred_s, z_s = obs[idx], pred[idx], z[idx]

        sc = ax.scatter(
            obs_s, pred_s,
            c=z_s,
            s=6,                 # 点密集时建议小一点
            cmap=cmap,
            edgecolors="none",
            alpha=1.0
        )
        mappable = sc
        cbar_label = "Density (KDE)"
    else:
        raise ValueError("method 只能是 'hexbin' 或 'density'")

    # 7) y = x 虚线（蓝色） + 回归线（红色）+ 方程
    ax.plot([lim0, lim1], [lim0, lim1], "b--", linewidth=1.4, label="y = x")

    order = np.argsort(obs)
    # 根据截距的正负号格式化公式，避免出现两个加号
    if intercept >= 0:
        label_text = f"y = {slope:.4f}x + {intercept:.4f}"
    else:
        label_text = f"y = {slope:.4f}x - {abs(intercept):.4f}"
    ax.plot(
        obs[order],
        pred_fit[order],
        color="red",
        linewidth=1.6,
        label=label_text,
    )

    # 8) 让 colorbar 高度严格匹配主图（关键）
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)  # size 控制宽度，pad 控制间距
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("")  # 移除标签
    cbar.set_ticks([])  # 移除刻度标签

    # 9) 标题、轴标签、图例（统一字体）
    ax.set_title(title, fontweight="bold", fontfamily=fontfamily , fontsize=18)
    ax.set_xlabel(xlab, fontfamily=fontfamily, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylab, fontfamily=fontfamily, fontsize=axis_label_fontsize)
    legend = ax.legend(
        loc="lower right", 
        frameon=False, 
        prop={'family': fontfamily, 'size': legend_fontsize},  # 通过 legend_fontsize 参数调整字体大小
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

    # 10) 指标注释
    metrics_text = (
        f"R$^2$={r2:.3f}\n"
        f"RMSE={rmse:.2f}\n"
        f"MAE={mae:.2f}\n"
        f"MAPE={mape:.2f}%"
    )
    ax.text(
        0.02, 0.98,
        metrics_text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=16,
        fontfamily=fontfamily
    )

    fig.tight_layout()
    # SVG格式是矢量图，不需要DPI参数，但保留不影响
    # 确保输出为正方形：bbox_inches='tight' 保持内容紧凑，但画布本身是正方形
    fig.savefig(out_path, format='svg', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # 关闭图形以释放内存
    print("Saved:", out_path)


if __name__ == "__main__":
    # 定义9个文件路径及其对应的标题和输出文件名
    file_configs = [
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-BPNN\Split2\output\predictions.csv",
            "title": "BPNN-Optimized",
            "out_path": "BPNN-Optimized.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-CNN- LSTM-Transformer\C-L-N\predictions__transformer__test.csv",
            "title": "CNN-LSTM-Transformer",
            "out_path": "CNN-LSTM-Transformer.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-CNN-GridSearch\Split2\output\predictions.csv",
            "title": "CNN-Optimized",
            "out_path": "CNN-Optimized.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-MLR_GAM\Split2\output\predictions_optimized.csv",
            "title": "GAM-MLR-Optimized",
            "out_path": "GAM-MLR-Optimized.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-LightGBM\Split2\output\predictions.csv",
            "title": "LightGBM-Optimized",
            "out_path": "LightGBM-Optimized.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-RF\Split2\output\rf_predictions_nc.csv",
            "title": "RF-Optimized",
            "out_path": "RF-Optimized.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-SVR\Split2\output\time_series_prediction.csv",
            "title": "SVR-Optimized",
            "out_path": "SVR-Optimized.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-Transformer\Split2\output\predictions_test.csv",
            "title": "Transformer-Optimized",
            "out_path": "Transformer-Optimized.svg"
        },
        {
            "path": r"E:\DATA Science\小论文Result\Fine_model\-XGBOOST\CSV\output\xgboost_predictions__xgboost_optimized__test.csv",
            "title": "XGBOOST-Optimized",
            "out_path": "XGBOOST-Optimized.svg"
        }
    ]

    # 循环绘制9张图
    for i, config in enumerate(file_configs, 1):
        print(f"\n正在处理第 {i}/9 个文件: {config['title']}")
        print(f"路径: {config['path']}")
        
        try:
            make_plot(
                csv_path=config['path'],
                method="density",
                cmap="jet",
                title=config['title'],
                xlabel="Actual (μg/m³)",
                ylabel="Predicted (μg/m³)",
                fontfamily="Times New Roman",
                out_path=config['out_path'],
            )
            print(f"✓ 成功保存: {config['out_path']}")
        except Exception as e:
            print(f"✗ 处理失败: {config['title']}")
            print(f"  错误信息: {str(e)}")
            continue
    
    print("\n所有图片处理完成！")
