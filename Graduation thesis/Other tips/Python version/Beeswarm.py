import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# 图中英文文本统一为 Times New Roman（通过 serif 族指定）
_FONT_RC = {"font.family": "serif", "font.serif": ["Times New Roman"]}


def beeswarm_offsets(x, max_swarm=0.40, nbins=60):
    """
    根据 SHAP 值在 y 方向生成蜂群散点偏移量
    x: 一维数组，SHAP value
    max_swarm: 单个特征上下扩展的最大宽度
    nbins: 按 x 方向分箱，控制蜂群堆叠效果
    """
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return np.array([])

    # 如果该特征所有 SHAP 值都几乎相同
    if np.allclose(x.max(), x.min()):
        bins = np.zeros_like(x, dtype=int)
    else:
        edges = np.linspace(x.min(), x.max(), nbins + 1)
        bins = np.digitize(x, edges[1:-1], right=False)

    offsets = np.zeros(len(x), dtype=float)

    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        idx = idx[np.argsort(x[idx], kind="mergesort")]  # 保持稳定排序
        n = len(idx)

        if n == 1:
            offsets[idx[0]] = 0.0
            continue

        # 生成 0, +1, -1, +2, -2 ... 的堆叠模式
        pattern = [0.0]
        step = 1
        while len(pattern) < n:
            pattern.extend([step, -step])
            step += 1

        pattern = np.array(pattern[:n], dtype=float)
        denom = np.max(np.abs(pattern)) if np.max(np.abs(pattern)) != 0 else 1.0
        pattern = pattern / denom * max_swarm
        offsets[idx] = pattern

    return offsets


def plot_shap_beeswarm_from_long_csv(
    csv_path,
    top_n=20,
    title="LightGBM-BTH SHAP Beeswarm",
    figsize=(8.5, 10),
    dpi=150,
    save_path="shap_beeswarm.png"
):
    """
    从长表格式 CSV 绘制 SHAP beeswarm 图

    CSV 需要至少包含以下列：
    - feature
    - shap_value
    - feature_value
    - abs_shap
    """

    df = pd.read_csv(csv_path, low_memory=False)

    required_cols = {"feature", "shap_value", "feature_value", "abs_shap"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    # 强制转为数值，避免 mixed dtype / 异常字符串在 groupby().mean() 时触发 TypeError
    for col in ("shap_value", "feature_value", "abs_shap"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["shap_value", "feature_value", "abs_shap"])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(
            f"警告: {n_dropped} 行因数值无法解析已丢弃（共 {n_before} 行）。"
            "若数量很大，请检查 CSV 分隔符、小数点或导出格式。"
        )

    # 按 mean(|SHAP|) 排序，取最重要的前 top_n 个特征
    feature_order = (
        df.groupby("feature", as_index=False)["abs_shap"]
        .mean()
        .sort_values("abs_shap", ascending=False)["feature"]
        .head(top_n)
        .tolist()
    )

    plot_df = df[df["feature"].isin(feature_order)].copy()

    # beeswarm 图通常从上到下按重要性递减排列
    feature_order = feature_order[::-1]

    with mpl.rc_context(_FONT_RC):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # 颜色映射：低值-高值
        cmap = plt.cm.cool

        for y, feat in enumerate(feature_order):
            sub = plot_df.loc[
                plot_df["feature"] == feat,
                ["shap_value", "feature_value"]
            ].copy()

            # 按 shap_value 排序，提升绘图观感
            sub = sub.sort_values("shap_value", kind="mergesort")

            x = sub["shap_value"].to_numpy()
            fval = sub["feature_value"].to_numpy(dtype=float)

            # 对 feature_value 做稳健归一化，避免极端值影响颜色
            lo, hi = np.nanpercentile(fval, [5, 95])
            if np.isclose(lo, hi):
                lo, hi = np.nanmin(fval), np.nanmax(fval)

            if np.isclose(lo, hi):
                color_val = np.full_like(fval, 0.5, dtype=float)
            else:
                color_val = np.clip((fval - lo) / (hi - lo), 0, 1)

            y_offsets = beeswarm_offsets(x, max_swarm=0.40, nbins=60)

            ax.scatter(
                x,
                y + y_offsets,
                c=color_val,
                cmap=cmap,
                s=14,
                alpha=0.90,
                linewidths=0,
                rasterized=True
            )

        # 竖直参考线 x = 0
        ax.axvline(0, color="gray", lw=1)

        # 坐标轴与标签
        ax.set_yticks(range(len(feature_order)))
        ax.set_yticklabels(feature_order, fontsize=15)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=17)
        ax.set_title(title, fontsize=19)
        ax.tick_params(axis="x", labelsize=14)

        # 网格与边框风格
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 颜色条
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.05, aspect=35)
        cbar.set_label("Feature value", rotation=90, labelpad=15, fontsize=16)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])
        cbar.ax.tick_params(labelsize=14)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    print(f"图已保存到: {save_path}")


if __name__ == "__main__":
    # 这里替换成你的文件路径
    csv_path = r"H:\大论文Result\大论文图\机器学习结果\lightgbm_daily_pm25\pooled.csv"

    plot_shap_beeswarm_from_long_csv(
        csv_path=csv_path,
        top_n=15,
        title="LightGBM-Pooled SHAP Beeswarm",
        figsize=(8.5, 10),
        dpi=300,
        save_path="shap_beeswarm.png"
    )