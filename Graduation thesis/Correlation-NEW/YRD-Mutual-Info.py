from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent

# =========================
# Embedded path configuration
# =========================
USE_PANEL_CSV = False
PANEL_CSV_PATH = SCRIPT_DIR / "inputs" / "yrd_panel.csv"
PM25_CITY_YEAR_CSV_PATH = SCRIPT_DIR / "inputs" / "yrd_pm25_city_year.csv"
DATA_READ_DIR = THESIS_DIR / "Data Read"
DATA_ROOT = THESIS_DIR / "1.模型要用的"
PM25_NC_DIR = DATA_ROOT / "2018-2023[PM2.5-china-clusters]" / "YRD"
PM25_CITY_GEOJSON_DIR = DATA_ROOT / "地图数据"

DISCRETIZE_BINS = 6
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "mutual_info_yrd"
GEO_SCRIPT_PATH = SCRIPT_DIR / "YRD-Geo-detector.py"


def load_module_from_path(module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discretize_series(series: pd.Series, bins: int = 6) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.nunique() <= 1:
        strata = pd.Series(np.nan, index=numeric.index, dtype="object")
        strata.loc[numeric.notna()] = "all"
        return strata

    k = int(min(max(2, bins), valid.nunique()))
    ranked = numeric.rank(method="first")
    return pd.qcut(ranked, q=k, duplicates="drop").astype("object")


def compute_entropy(probabilities: np.ndarray) -> float:
    probs = probabilities[(probabilities > 0) & np.isfinite(probabilities)]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def compute_mi(x: pd.Series, y: pd.Series, bins: int = 6) -> dict[str, float]:
    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()
    n_samples = int(len(pair))
    if n_samples < 3:
        return {
            "mi": np.nan,
            "entropy_x": np.nan,
            "entropy_y": np.nan,
            "joint_entropy": np.nan,
            "n_samples": n_samples,
        }

    x_bins = discretize_series(pair["x"], bins=bins)
    y_bins = discretize_series(pair["y"], bins=bins)
    binned = pd.DataFrame({"x_bin": x_bins, "y_bin": y_bins}).dropna()
    n_binned = int(len(binned))
    if n_binned < 3:
        return {
            "mi": np.nan,
            "entropy_x": np.nan,
            "entropy_y": np.nan,
            "joint_entropy": np.nan,
            "n_samples": n_binned,
        }

    p_x = binned["x_bin"].value_counts(normalize=True).to_numpy(dtype=float)
    p_y = binned["y_bin"].value_counts(normalize=True).to_numpy(dtype=float)
    p_xy = (
        binned.groupby(["x_bin", "y_bin"], observed=True)
        .size()
        .div(n_binned)
        .to_numpy(dtype=float)
    )

    h_x = compute_entropy(p_x)
    h_y = compute_entropy(p_y)
    h_xy = compute_entropy(p_xy)
    mi_value = float(max(0.0, h_x + h_y - h_xy))
    return {
        "mi": mi_value,
        "entropy_x": h_x,
        "entropy_y": h_y,
        "joint_entropy": h_xy,
        "n_samples": n_binned,
    }


def run_factor_mi(df: pd.DataFrame, y_col: str, factor_cols: list[str], bins: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for factor in factor_cols:
        stats = compute_mi(df[factor], df[y_col], bins=bins)
        rows.append({"factor": factor, **stats})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("mi", ascending=False).reset_index(drop=True)


def build_soft_blue_red_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "soft_blue_red",
        ["#5C88C5", "#A9C3E8", "#F7F7F7", "#E8B0B0", "#CF6F6F"],
        N=256,
    )


def set_paper_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def plot_factor_mi_heatmap(mi_by_year: pd.DataFrame, output_png: Path) -> None:
    if mi_by_year.empty:
        return

    pivot = mi_by_year.pivot(index="factor", columns="year", values="mi").sort_index()
    if pivot.empty:
        return

    vmax = float(np.nanmax(pivot.to_numpy(dtype=float)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    set_paper_plot_style()
    cmap = build_soft_blue_red_cmap()
    cell_size = 0.55
    fig_w = max(8, pivot.shape[1] * cell_size + 3)
    fig_h = max(4, pivot.shape[0] * cell_size + 2)
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        linewidths=0.5,
        linecolor="#F2F2F2",
        square=True,
        annot_kws={"size": 8.5, "color": "#2F2F2F"},
        cbar_kws={"shrink": 0.9, "pad": 0.02, "label": "Mutual Information"},
    )
    plt.title("YRD Mutual Information Heatmap", pad=10)
    plt.xlabel("Year")
    plt.ylabel("Factor")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, facecolor="white")
    plt.close()


def build_pairwise_mi_matrix(df: pd.DataFrame, variables: list[str], bins: int) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=variables, columns=variables, dtype=float)
    for i, var_i in enumerate(variables):
        matrix.loc[var_i, var_i] = compute_mi(df[var_i], df[var_i], bins=bins)["mi"]
        for j in range(i + 1, len(variables)):
            var_j = variables[j]
            mi_val = compute_mi(df[var_i], df[var_j], bins=bins)["mi"]
            matrix.loc[var_i, var_j] = mi_val
            matrix.loc[var_j, var_i] = mi_val
    return matrix


def plot_pairwise_triangle_heatmap(matrix: pd.DataFrame, output_png: Path, title: str) -> None:
    if matrix.empty:
        return
    values = matrix.to_numpy(dtype=float)
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    set_paper_plot_style()
    cmap = build_soft_blue_red_cmap()
    n_vars = matrix.shape[0]
    fig_side = max(8, n_vars * 0.62 + 2)
    annot_size = float(np.clip(9.3 - n_vars * 0.18, 6.5, 8.5))
    mask_upper = np.triu(np.ones_like(values, dtype=bool), k=1)

    plt.figure(figsize=(fig_side, fig_side))
    sns.heatmap(
        matrix,
        mask=mask_upper,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        linewidths=0.5,
        linecolor="#F2F2F2",
        square=True,
        annot_kws={"size": annot_size, "color": "#2F2F2F"},
        cbar_kws={"shrink": 0.9, "pad": 0.02, "label": "Mutual Information"},
    )
    plt.title(title, pad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, facecolor="white")
    plt.close()


def build_panel(base_module: Any, target_cities: list[str]) -> pd.DataFrame:
    if USE_PANEL_CSV:
        panel_csv = base_module.resolve_existing_path(
            PANEL_CSV_PATH,
            path_desc="panel file",
            fallback_candidates=[
                SCRIPT_DIR / "yrd_panel.csv",
                OUTPUT_DIR / "yrd_panel_from_interfaces.csv",
            ],
        )
        return pd.read_csv(panel_csv, encoding="utf-8-sig")

    discover_func = getattr(base_module, "_discover_pm25_candidates", None)
    pm25_discovered = discover_func(THESIS_DIR) if callable(discover_func) else []
    pm25_csv_config = PM25_CITY_YEAR_CSV_PATH.expanduser().resolve()
    if not pm25_csv_config.exists():
        nc_dir = PM25_NC_DIR.expanduser()
        city_geojson_dir = PM25_CITY_GEOJSON_DIR.expanduser()
        geojson_dir_configured = str(PM25_CITY_GEOJSON_DIR).strip() not in {"", ".", ".\\"}
        if geojson_dir_configured and nc_dir.exists() and city_geojson_dir.exists():
            print("[INFO] PM2.5 city-year csv not found, building from NC + GeoJSON.")
            pm25_city_year_df = base_module.build_pm25_city_year_from_nc(
                nc_dir=nc_dir,
                city_geojson_dir=city_geojson_dir,
                target_cities=target_cities,
            )
            pm25_csv_config.parent.mkdir(parents=True, exist_ok=True)
            pm25_city_year_df.to_csv(pm25_csv_config, index=False, encoding="utf-8-sig")
            print(f"[INFO] Generated PM2.5 city-year csv: {pm25_csv_config}")

    pm25_csv = base_module.resolve_existing_path(
        PM25_CITY_YEAR_CSV_PATH,
        path_desc="PM2.5 city-year file",
        fallback_candidates=[
            SCRIPT_DIR / "inputs" / "pm25_city_year.csv",
            *pm25_discovered,
        ],
    )
    data_read_dir = base_module.resolve_existing_path(
        DATA_READ_DIR,
        path_desc="Data Read directory",
        fallback_candidates=[THESIS_DIR / "Data Read"],
    )
    panel = base_module.build_panel_from_interfaces(
        data_read_dir=data_read_dir,
        pm25_city_year_csv=pm25_csv,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_DIR / "yrd_panel_from_interfaces.csv", index=False, encoding="utf-8-sig")
    return panel


def main() -> int:
    sns.set_theme(style="white", context="paper")
    if not GEO_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Reference script not found: {GEO_SCRIPT_PATH}")

    geo_module = load_module_from_path(GEO_SCRIPT_PATH)
    base_module = geo_module._load_bth_module() if hasattr(geo_module, "_load_bth_module") else geo_module
    target_cities = list(getattr(geo_module, "YRD_CITIES", getattr(base_module, "BTH_CITIES", [])))
    if hasattr(geo_module, "YRD_CITY_TO_PROVINCE"):
        base_module.CITY_TO_PROVINCE = geo_module.YRD_CITY_TO_PROVINCE
    base_module.BTH_CITIES = target_cities
    panel = build_panel(base_module, target_cities)

    panel.columns = [str(c).strip() for c in panel.columns]
    required_cols = {"city", "year", "pm25"}
    if required_cols - set(panel.columns):
        raise ValueError("Input panel must contain columns: city, year, pm25")

    panel["city"] = base_module.normalize_city_name(panel["city"])
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel["pm25"] = pd.to_numeric(panel["pm25"], errors="coerce")
    panel = panel.loc[panel["city"].isin(target_cities)].copy()

    factor_cols = [c for c in panel.columns if c not in {"city", "year", "pm25"}]
    if not factor_cols:
        raise ValueError("No factor columns were detected.")

    numeric_factors: list[str] = []
    for col in factor_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
        if panel[col].notna().sum() > 0:
            numeric_factors.append(col)
    factor_cols = numeric_factors
    if not factor_cols:
        raise ValueError("All factors are non-numeric after conversion.")

    overall = panel.dropna(subset=["pm25"]).copy()
    factor_mi = run_factor_mi(overall, y_col="pm25", factor_cols=factor_cols, bins=DISCRETIZE_BINS)

    by_year_rows: list[pd.DataFrame] = []
    for year, sub_df in overall.groupby("year", observed=True):
        if sub_df["city"].nunique() < 3:
            continue
        year_mi = run_factor_mi(sub_df, y_col="pm25", factor_cols=factor_cols, bins=DISCRETIZE_BINS)
        year_mi["year"] = int(year)
        by_year_rows.append(year_mi)
    mi_by_year = pd.concat(by_year_rows, ignore_index=True) if by_year_rows else pd.DataFrame()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    factor_mi.to_csv(OUTPUT_DIR / "yrd_mi_factor.csv", index=False, encoding="utf-8-sig")
    mi_by_year.to_csv(OUTPUT_DIR / "yrd_mi_factor_by_year.csv", index=False, encoding="utf-8-sig")
    plot_factor_mi_heatmap(mi_by_year, OUTPUT_DIR / "yrd_mi_factor_heatmap.png")
    pairwise_vars = ["pm25", *factor_cols]
    pairwise_mi = build_pairwise_mi_matrix(overall, variables=pairwise_vars, bins=DISCRETIZE_BINS)
    plot_pairwise_triangle_heatmap(
        pairwise_mi,
        OUTPUT_DIR / "yrd_mi_pairwise_triangle_heatmap.png",
        title="YRD Pairwise MI Triangle Heatmap",
    )

    print("=" * 80)
    print("[INFO] YRD Mutual Information analysis completed")
    print(f"[INFO] Samples: {len(overall)}")
    print(f"[INFO] Factors: {len(factor_cols)}")
    print(f"[INFO] Output directory: {OUTPUT_DIR.resolve()}")
    print("[INFO] Outputs:")
    print("       - yrd_panel_from_interfaces.csv")
    print("       - yrd_mi_factor.csv")
    print("       - yrd_mi_factor_by_year.csv")
    print("       - yrd_mi_factor_heatmap.png")
    print("       - yrd_mi_pairwise_triangle_heatmap.png")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
