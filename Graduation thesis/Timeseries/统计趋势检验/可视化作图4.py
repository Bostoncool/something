# -*- coding: utf-8 -*-
"""仅绘制：三大城市群统计检验结果 2×2 对比图。依赖同目录 可视化作图.py 中的函数。"""
import importlib.util
import os
import sys
from pathlib import Path


def _load_base():
    path = Path(__file__).resolve().parent / "可视化作图.py"
    spec = importlib.util.spec_from_file_location("_viz_thesis_trend_base", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载同目录下的 可视化作图.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    m = _load_base()
    m.configure_thesis_fonts()
    od = m.OUTPUT_DIR

    m.safe_print("加载检验结果 CSV...")
    annual_results = {
        "annual_mann_kendall": m.read_csv_flexible(os.path.join(od, "年度_Mann_Kendall_城市群.csv")),
        "annual_pettitt": m.read_csv_flexible(os.path.join(od, "年度_Pettitt_城市群.csv")),
        "annual_sliding_t": m.read_csv_flexible(os.path.join(od, "年度_滑动T检验_城市群.csv")),
        "annual_linear_regression": m.read_csv_flexible(os.path.join(od, "年度_线性回归_城市群.csv")),
    }
    monthly_results = {
        "month_pettitt": m.read_csv_flexible(os.path.join(od, "月度_Pettitt_城市群.csv")),
        "month_sliding_t": m.read_csv_flexible(os.path.join(od, "月度_滑动T检验_城市群.csv")),
    }

    m.plot_test_comparison_2x2(
        annual_results,
        monthly_results,
        os.path.join(od, "三大城市群_统计检验结果对比.png"),
    )
    m.safe_print("完成：图4（2×2 检验对比，单文件含四子图）。")


if __name__ == "__main__":
    main()
