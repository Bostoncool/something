import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_validation_curve(
    csv_path: str,
    output_path: str = None,
    title: str = "CNN-Basic",
    fig_size: float = 6.0,
    dpi: int = 300
) -> None:    
    # 1) 读取数据
    df = pd.read_csv(csv_path)
    
    # 检查列数是否足够
    if len(df.columns) < 3:
        raise ValueError(f"CSV文件至少需要3列，当前只有{len(df.columns)}列。当前列: {list(df.columns)}")
    
    # 2) 自动识别列：第一列为序数，第二列为训练损失，第三列为验证损失
    epoch_col = df.columns[0]  # 第一列：序数
    train_col = df.columns[1]   # 第二列：训练损失
    valid_col = df.columns[2]   # 第三列：验证损失
    
    print(f"自动识别列: 序数={epoch_col}, 训练损失={train_col}, 验证损失={valid_col}")

    # 3) 计算最佳训练效果（验证损失最小）
    best_idx = df[valid_col].idxmin()
    best_epoch = int(df.loc[best_idx, epoch_col])
    best_val = float(df.loc[best_idx, valid_col])

    # 4) 确定坐标轴范围（带一点边距）
    x_min, x_max = df[epoch_col].min(), df[epoch_col].max()
    y_min = min(df[train_col].min(), df[valid_col].min())
    y_max = max(df[train_col].max(), df[valid_col].max())
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.01

    # 5) 自动生成输出路径和标题（如果未提供）
    if output_path is None:
        csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = f"{title}.png"

    # 6) 绘图（正方形）
    plt.figure(figsize=(fig_size, fig_size))

    # 训练/验证曲线
    plt.plot(df[epoch_col], df[train_col], color="blue", linewidth=1.5, label="Training Loss")
    plt.plot(df[epoch_col], df[valid_col], color="orange", linewidth=1.5, label="Validation Loss")

    # 最佳 epoch 红色虚线
    plt.axvline(
        x=best_epoch,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Best Epoch = {best_epoch} (val={best_val:.4f})"
    )

    # 标题与坐标轴标签
    plt.title(title, fontsize=18, fontweight="bold", fontfamily="Times New Roman")
    plt.xlabel("Epoch", fontsize=12, fontfamily="Times New Roman")
    plt.ylabel("Loss", fontsize=12, fontfamily="Times New Roman")

    # 应用坐标范围
    x_pad = (x_max - x_min) * 0.03   # 3% 留白
    plt.xlim(x_min, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)

    # 5) 四边实线封闭（spines全部可见 + 加粗）
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    # 6) 图例与排版
    plt.legend(frameon=True)
    plt.tight_layout()

    # 7) 保存图片
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {output_path}")
    print(f"Best Epoch: {best_epoch}, Best Validation RMSE: {best_val:.6f}")


if __name__ == "__main__":
    # 示例1: 使用默认设置（自动生成输出文件名和标题）
    csv_path = r"H:\DATA Science\小论文Result\Fine_model\-CNN-GridSearch\CSV\output\plot_training_curves__cnn_basic.csv"
    plot_training_validation_curve(
        csv_path=csv_path,
        fig_size=6.0,
        dpi=300
    )

    # 示例2: 手动指定输出路径和标题
    # plot_training_validation_curve(
    #     csv_path="your_file.csv",
    #     output_path="custom_output.png",
    #     title="Custom Title",
    #     fig_size=6.0,
    #     dpi=300
    # )