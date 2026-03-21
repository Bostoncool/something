# -*- coding: utf-8 -*-
"""仅绘制：三大城市群季度 PM2.5 时序（突变点仅星标、无文字框）。依赖同目录 可视化作图.py 中的函数。"""
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

    m.safe_print("加载季度聚合序列与突变点（滑动 T，仅星标）...")
    seasonal_group_series = m.load_series_from_csv(
        os.path.join(od, "城市群_季度PM2.5聚合序列.csv"),
        "Season",
    )
    seasonal_plot_change_df = m.build_change_points_df_for_plot(
        seasonal_group_series,
        "Season",
        min_window=4,
        min_each_side=4,
    )

    m.plot_group_series(
        seasonal_group_series,
        "",
        "Season",
        os.path.join(od, "三大城市群_季度PM2.5时序.png"),
        sliding_t_df=seasonal_plot_change_df,
        annotate_change_points=True,
        change_point_annotation_text=False,
    )
    m.safe_print("完成：图3（季度时序）。")


if __name__ == "__main__":
    main()
