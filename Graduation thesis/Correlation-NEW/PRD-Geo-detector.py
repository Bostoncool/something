from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


PRD_CITIES = [
    "广州",
    "深圳",
    "佛山",
    "东莞",
    "中山",
    "惠州",
    "珠海",
    "江门",
    "肇庆",
]

PRD_CITY_TO_PROVINCE = {
    "广州": "guangdong",
    "深圳": "guangdong",
    "佛山": "guangdong",
    "东莞": "guangdong",
    "中山": "guangdong",
    "惠州": "guangdong",
    "珠海": "guangdong",
    "江门": "guangdong",
    "肇庆": "guangdong",
}

SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent

# =========================
# 内嵌路径配置（直接修改这里）
# =========================
USE_PANEL_CSV = False
PANEL_CSV_PATH = SCRIPT_DIR / "inputs" / "prd_panel.csv"
PM25_CITY_YEAR_CSV_PATH = SCRIPT_DIR / "inputs" / "prd_pm25_city_year.csv"
DATA_READ_DIR = THESIS_DIR / "Data Read"
DATA_ROOT = THESIS_DIR / "1.模型要用的"
PM25_NC_DIR = DATA_ROOT / "2018-2023[PM2.5-china-clusters]" / "PRD"
PM25_CITY_GEOJSON_DIR = DATA_ROOT / "地图数据"
DISCRETIZE_BINS = 6
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "geo_detector_prd"


def _load_bth_module():
    module_path = SCRIPT_DIR / "BTH-Geo-detector.py"
    spec = importlib.util.spec_from_file_location("geo_detector_bth_base_for_prd", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_pm25_csv_for_region(base, target_cities: list[str], region_label: str) -> Path:
    configured_csv = PM25_CITY_YEAR_CSV_PATH.expanduser().resolve()
    if not configured_csv.exists():
        nc_dir = PM25_NC_DIR.expanduser()
        city_geojson_dir = PM25_CITY_GEOJSON_DIR.expanduser()
        if nc_dir.exists() and city_geojson_dir.exists():
            print(f"[INFO] 未检测到 {region_label} PM2.5 城市年均 CSV，尝试通过 NC + GeoJSON 自动构建。")
            pm25_city_year_df = base.build_pm25_city_year_from_nc(
                nc_dir=nc_dir,
                city_geojson_dir=city_geojson_dir,
                target_cities=target_cities,
            )
            configured_csv.parent.mkdir(parents=True, exist_ok=True)
            pm25_city_year_df.to_csv(configured_csv, index=False, encoding="utf-8-sig")
            print(f"[INFO] 已生成 {region_label} PM2.5 城市年均 CSV: {configured_csv}")

    fallback_candidates = [
        configured_csv,
        (SCRIPT_DIR / f"{region_label.lower()}_pm25_city_year.csv").expanduser().resolve(),
        (SCRIPT_DIR / "inputs" / "pm25_city_year.csv").expanduser().resolve(),
    ]
    pm25_csv = next((path for path in fallback_candidates if path.exists()), None)
    if pm25_csv is None:
        raise FileNotFoundError(
            f"未找到 {region_label} PM2.5 城市年均表: {configured_csv}\n"
            f"请提供 `{region_label.lower()}_pm25_city_year.csv`，"
            "或确保 PM25_NC_DIR 与 PM25_CITY_GEOJSON_DIR 可用以自动生成。"
        )

    pm25_df = pd.read_csv(pm25_csv, encoding="utf-8-sig")
    pm25_df.columns = [str(c).strip() for c in pm25_df.columns]
    required_cols = {"city", "year", "pm25"}
    if required_cols - set(pm25_df.columns):
        raise ValueError(f"{region_label} PM2.5 文件缺少必要列: city, year, pm25 | file={pm25_csv}")

    pm25_cities = set(base.normalize_city_name(pm25_df["city"]).dropna().astype(str))
    region_city_set = set(target_cities)
    overlap_cities = sorted(region_city_set.intersection(pm25_cities))
    if not overlap_cities:
        raise ValueError(
            f"检测到 PM2.5 文件不属于 {region_label} 城市群: {pm25_csv}\n"
            f"该文件与 {region_label} 城市列表交集为 0，请提供 {region_label.lower()}_pm25_city_year.csv。"
        )
    if len(overlap_cities) < min(3, len(target_cities)):
        print(
            f"[WARN] {region_label} PM2.5 文件仅匹配到 {len(overlap_cities)} 个城市，"
            "结果可能不稳定，请检查城市覆盖度。"
        )
    return pm25_csv


def _plot_factor_heatmap_region(base, q_by_year: pd.DataFrame, output_png: Path) -> None:
    if q_by_year.empty:
        return
    pivot = q_by_year.pivot(index="factor", columns="year", values="q").sort_index()
    cmap = base.build_soft_blue_red_cmap()
    base._paper_plot_style()
    cell_size = 0.55
    fig_w = max(8, pivot.shape[1] * cell_size + 3)
    fig_h = max(4, pivot.shape[0] * cell_size + 2)
    plt.figure(figsize=(fig_w, fig_h))
    cbar_ticks = np.linspace(0, 1, 6)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0,
        vmax=1,
        center=0.5,
        linewidths=0.5,
        linecolor="#F2F2F2",
        square=True,
        annot_kws={"size": 8.5, "color": "#2F2F2F"},
        cbar_kws={"shrink": 0.9, "pad": 0.02, "ticks": cbar_ticks, "label": "q value"},
    )
    plt.title("PRD Geo-detector q Heatmap", pad=10)
    plt.xlabel("Year")
    plt.ylabel("Factor")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, facecolor="white")
    plt.close()


def _plot_interaction_heatmap_region(base, inter_df: pd.DataFrame, output_png: Path) -> None:
    if inter_df.empty:
        return
    factors = sorted(set(inter_df["factor_1"]).union(inter_df["factor_2"]))
    matrix = pd.DataFrame(np.nan, index=factors, columns=factors)
    for _, row in inter_df.iterrows():
        matrix.loc[row["factor_1"], row["factor_2"]] = row["q_interaction"]
        matrix.loc[row["factor_2"], row["factor_1"]] = row["q_interaction"]
    np.fill_diagonal(matrix.values, 1.0)
    mask_upper = np.triu(np.ones_like(matrix.values, dtype=bool), k=1)
    cmap = base.build_soft_blue_red_cmap()
    base._paper_plot_style()
    cell_size = 0.62
    fig_side = max(8, len(factors) * cell_size + 2)
    plt.figure(figsize=(fig_side, fig_side))
    annot_size = float(np.clip(9.5 - len(factors) * 0.2, 6.8, 8.8))
    cbar_ticks = np.linspace(0, 1, 6)
    sns.heatmap(
        matrix,
        mask=mask_upper,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0,
        vmax=1,
        center=0.5,
        linewidths=0.5,
        linecolor="#F2F2F2",
        square=True,
        annot_kws={"size": annot_size, "color": "#2F2F2F"},
        cbar_kws={"shrink": 0.9, "pad": 0.02, "ticks": cbar_ticks, "label": "q value"},
    )
    plt.title("PRD Interaction q Heatmap", pad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, facecolor="white")
    plt.close()


def main() -> int:
    base = _load_bth_module()

    # 覆写基线模块中的区域配置，复用全部数据构建与探测器计算逻辑
    base.BTH_CITIES = PRD_CITIES
    base.CITY_TO_PROVINCE = PRD_CITY_TO_PROVINCE
    base.SCRIPT_DIR = SCRIPT_DIR
    base.THESIS_DIR = THESIS_DIR
    base.USE_PANEL_CSV = USE_PANEL_CSV
    base.PANEL_CSV_PATH = PANEL_CSV_PATH
    base.PM25_CITY_YEAR_CSV_PATH = PM25_CITY_YEAR_CSV_PATH
    base.DATA_READ_DIR = DATA_READ_DIR
    base.PM25_NC_DIR = PM25_NC_DIR
    base.PM25_CITY_GEOJSON_DIR = PM25_CITY_GEOJSON_DIR
    base.DISCRETIZE_BINS = DISCRETIZE_BINS
    base.OUTPUT_DIR = OUTPUT_DIR

    sns.set_theme(style="white", context="paper")
    output_dir = OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if USE_PANEL_CSV:
        panel_csv = base.resolve_existing_path(
            PANEL_CSV_PATH,
            path_desc="面板文件",
            fallback_candidates=[SCRIPT_DIR / "prd_panel.csv", OUTPUT_DIR / "prd_panel_from_interfaces.csv"],
        )
        panel = pd.read_csv(panel_csv, encoding="utf-8-sig")
    else:
        pm25_csv = _resolve_pm25_csv_for_region(base, PRD_CITIES, "PRD")
        panel = base.build_panel_from_interfaces(
            data_read_dir=base.resolve_existing_path(DATA_READ_DIR, "Data Read 目录", [THESIS_DIR / "Data Read"]),
            pm25_city_year_csv=pm25_csv,
        )
        panel.to_csv(output_dir / "prd_panel_from_interfaces.csv", index=False, encoding="utf-8-sig")

    panel.columns = [str(c).strip() for c in panel.columns]
    panel["city"] = base.normalize_city_name(panel["city"])
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel["pm25"] = pd.to_numeric(panel["pm25"], errors="coerce")
    panel = panel.loc[panel["city"].isin(PRD_CITIES)].copy()
    if panel.empty or panel["pm25"].notna().sum() == 0:
        raise ValueError("PRD 面板中没有有效的城市 PM2.5 数据，请检查 prd_pm25_city_year.csv 是否正确。")

    factor_cols = [c for c in panel.columns if c not in {"city", "year", "pm25"}]
    factor_cols = [col for col in factor_cols if pd.to_numeric(panel[col], errors="coerce").notna().any()]
    for col in factor_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    overall = panel.dropna(subset=["pm25"]).copy()
    factor_q, strata_map = base.run_factor_detector(overall, "pm25", factor_cols, DISCRETIZE_BINS)
    inter_q = base.run_interaction_detector(overall["pm25"], strata_map, factor_q)

    by_year_rows = []
    for year, sub_df in overall.groupby("year", observed=True):
        if sub_df["city"].nunique() < 3:
            continue
        year_q, _ = base.run_factor_detector(sub_df, "pm25", factor_cols, DISCRETIZE_BINS)
        year_q["year"] = int(year)
        by_year_rows.append(year_q)
    q_by_year = pd.concat(by_year_rows, ignore_index=True) if by_year_rows else pd.DataFrame()

    factor_q.to_csv(output_dir / "prd_factor_detector.csv", index=False, encoding="utf-8-sig")
    inter_q.to_csv(output_dir / "prd_interaction_detector.csv", index=False, encoding="utf-8-sig")
    q_by_year.to_csv(output_dir / "prd_factor_q_by_year.csv", index=False, encoding="utf-8-sig")
    _plot_factor_heatmap_region(base, q_by_year, output_dir / "prd_factor_q_heatmap.png")
    _plot_interaction_heatmap_region(base, inter_q, output_dir / "prd_interaction_q_heatmap.png")

    print("=" * 80)
    print("[INFO] 珠三角 Geo-detector 分析完成")
    print(f"[INFO] 样本量: {len(overall)}")
    print(f"[INFO] 因子数量: {len(factor_cols)}")
    print(f"[INFO] 输出目录: {output_dir}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
