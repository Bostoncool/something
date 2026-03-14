"""
Test Statistic: 按 region 统计 test_r2 最高的前 5 个模型，
输出 test_r2 / test_mae / test_rmse，并用多柱状图可视化。
依赖: pandas, matplotlib, openpyxl (pip install openpyxl)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# 路径常量
EXCEL_PATH = Path(
    r"h:\DATA Science\大论文Result\大论文图\机器学习结果\评价指标docs\合并文件.xlsx"
)
OUTPUT_DIR = Path(r"H:\DATA Science\大论文Result\大论文图\机器学习结果\评价指标docs")
OUTPUT_CSV = OUTPUT_DIR / "test_statistic_top5_by_region.csv"
FIG_DPI = 300

REGIONS = ["bth", "prd", "yrd", "pooled"]

# 列名映射：若 Excel 列名不同，在此指定映射到统一字段
COLUMN_MAP = {
    "model": "model",
    "region": "region",
    "test_r2": "test_r2",
    "test_mae": "test_mae",
    "test_rmse": "test_rmse",
}
# 常见别名（优先用上面，否则用别名）
COLUMN_ALIASES = {
    "r2": "test_r2",
    "mae": "test_mae",
    "rmse": "test_rmse",
}


def load_and_preprocess(path: Path) -> pd.DataFrame:
    """读取 Excel，列名标准化，region 转小写并只保留 bth/prd/yrd/pooled。"""
    df = pd.read_excel(path, engine="openpyxl")

    # 列名标准化：先尝试直接重命名，再尝试别名
    rename = {}
    for col in df.columns:
        c = str(col).strip().lower()
        if c in COLUMN_MAP:
            rename[col] = COLUMN_MAP[c]
        elif c in COLUMN_ALIASES:
            rename[col] = COLUMN_ALIASES[c]
    df = df.rename(columns=rename)

    required = {"model", "region", "test_r2", "test_mae", "test_rmse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel 缺少列: {missing}。当前列: {list(df.columns)}")

    df = df[list(required)].copy()
    df["region"] = df["region"].astype(str).str.strip().str.lower()
    df = df[df["region"].isin(REGIONS)].copy()

    for col in ["test_r2", "test_mae", "test_rmse"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["test_r2", "test_mae", "test_rmse"])

    return df


def get_top5_by_region(
    df: pd.DataFrame, regions: list[str], metric: str = "test_r2"
) -> dict[str, pd.DataFrame]:
    """按 region 取指定指标降序的前 5 个模型。"""
    out = {}
    for r in regions:
        sub = df[df["region"] == r].sort_values(metric, ascending=False).head(5)
        out[r] = sub.reset_index(drop=True)
    return out


def plot_grouped_bars(
    top5_by_region: dict[str, pd.DataFrame], output_dir: Path, dpi: int = FIG_DPI
) -> list[Path]:
    """每个 region 单独一张图，双 Y 轴：左侧 R²，右侧 MAE/RMSE。仅输出 SVG。"""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11

    saved = []
    w = 0.25  # 单根柱子宽度，三根并排

    for region in REGIONS:
        top5 = top5_by_region.get(region)
        if top5 is None or top5.empty:
            continue

        fig, ax_left = plt.subplots(figsize=(8, 5))
        ax_right = ax_left.twinx()

        models = top5["model"].astype(str).tolist()
        x = range(len(models))

        # 左轴：R²（柔和蓝）
        ax_left.bar(
            [xi - w for xi in x],
            top5["test_r2"],
            width=w,
            label="Test R² (left axis)",
            color="#64B5F6",
        )
        # 右轴：MAE（柔和橙）、RMSE（柔和紫）
        ax_right.bar(
            [xi for xi in x],
            top5["test_mae"],
            width=w,
            label="Test MAE (right axis)",
            color="#FFB74D",
        )
        ax_right.bar(
            [xi + w for xi in x],
            top5["test_rmse"],
            width=w,
            label="Test RMSE (right axis)",
            color="#B39DDB",
        )

        ax_left.set_xticks(x)
        ax_left.set_xticklabels(models, rotation=45, ha="right")
        ax_left.set_ylabel("Test R² (left axis)", color="black")
        ax_right.set_ylabel("Test MAE / RMSE (right axis, μg/m³)", color="black")
        ax_left.set_ylim(0, None)
        ax_right.set_ylim(0, None)
        ax_left.set_title(f"{region.upper()} — Top 5 models by Test R²")

        h1, l1 = ax_left.get_legend_handles_labels()
        h2, l2 = ax_right.get_legend_handles_labels()
        ax_left.legend(
            h1 + h2, l1 + l2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.38),
            ncol=3,
            frameon=False,
        )
        fig.subplots_adjust(bottom=0.44)

        out_path = output_dir / f"test_statistic_top5_{region}.svg"
        fig.savefig(out_path, format="svg", dpi=dpi, bbox_inches="tight")
        saved.append(out_path)
        plt.close(fig)

    return saved


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 读取与预处理
    df = load_and_preprocess(EXCEL_PATH)
    print(f"已读取 Excel，保留四区有效行数: {len(df)}")

    # 2. 各 region test_r2 前 5
    top5_by_region = get_top5_by_region(df, REGIONS, metric="test_r2")

    # 3. 输出表格（控制台 + 可选 CSV）
    all_top5 = []
    for region in REGIONS:
        top5 = top5_by_region[region]
        print(f"\n--- {region.upper()} test_r2 最高的 5 个模型 ---")
        if top5.empty:
            print("  (无数据)")
            continue
        print(top5.to_string(index=False))
        all_top5.append(top5)

    if all_top5:
        combined = pd.concat(all_top5, ignore_index=True)
        combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n已导出: {OUTPUT_CSV}")

    # 4. 多柱状图（每 region 一图，双 Y 轴，仅 SVG，DPI=300）
    saved_svgs = plot_grouped_bars(top5_by_region, OUTPUT_DIR)
    for p in saved_svgs:
        print(f"已保存: {p}")


if __name__ == "__main__":
    main()
