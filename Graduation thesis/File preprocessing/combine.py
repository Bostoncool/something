"""
合并 model_metrics_summary.csv 与 model_generalization_summary.csv，
输出为 XLSX 格式的合并文件。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from pathlib import Path

# 输入文件路径
INPUT_DIR = Path(r"h:\DATA Science\大论文Result\大论文图\机器学习结果\开发文档docs")
METRICS_FILE = INPUT_DIR / "model_metrics_summary.csv"
GENERALIZATION_FILE = INPUT_DIR / "model_generalization_summary.csv"

# 输出文件路径（与输入同目录）
OUTPUT_FILE = INPUT_DIR / "合并文件.xlsx"


def main() -> None:
    # 读取两个 CSV 文件
    df_metrics = pd.read_csv(METRICS_FILE)
    df_generalization = pd.read_csv(GENERALIZATION_FILE)

    # 按 model 和 region 合并（左连接，保留所有指标数据）
    df_merged = pd.merge(
        df_metrics,
        df_generalization,
        on=["model", "region"],
        how="outer",
        suffixes=("", "_gen"),
    )

    # 若存在重复列名，删除带 _gen 后缀的重复列
    cols_to_drop = [c for c in df_merged.columns if c.endswith("_gen")]
    df_merged = df_merged.drop(columns=cols_to_drop, errors="ignore")

    # 输出为 XLSX 格式
    df_merged.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")

    print(f"合并完成！共 {len(df_merged)} 行，{len(df_merged.columns)} 列")
    print(f"输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
