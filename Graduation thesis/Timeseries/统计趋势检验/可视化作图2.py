# -*- coding: utf-8 -*-
"""仅绘制：三大城市群月均 PM2.5 时序。依赖同目录 可视化作图.py 中的函数。"""
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

    m.safe_print("加载月度聚合序列...")
    monthly_group_series = m.load_series_from_csv(
        os.path.join(od, "城市群_月度PM2.5聚合序列.csv"),
        "Month",
    )

    m.plot_group_series(
        monthly_group_series,
        "",
        "Month",
        os.path.join(od, "三大城市群_月度PM2.5时序.png"),
    )
    m.safe_print("完成：图2（月度时序）。")


if __name__ == "__main__":
    main()
