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

try:
    from scipy.stats import t as t_dist
except Exception:  # pylint: disable=broad-except
    t_dist = None


SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent

# =========================
# Embedded path configuration
# =========================
USE_PANEL_CSV = False
PANEL_CSV_PATH = SCRIPT_DIR / "inputs" / "prd_panel.csv"
PM25_CITY_YEAR_CSV_PATH = SCRIPT_DIR / "inputs" / "prd_pm25_city_year.csv"
DATA_READ_DIR = THESIS_DIR / "Data Read"
PM25_NC_DIR = Path(r"F:\1.模型要用的\2018-2023[PM2.5-china]\Year")
PM25_CITY_GEOJSON_DIR = Path(r"F:\1.模型要用的\地图数据")

OUTPUT_DIR = SCRIPT_DIR / "outputs" / "spearman_prd"
GEO_SCRIPT_PATH = SCRIPT_DIR / "PRD-Geo-detector.py"
ALPHA = 0.05


def load_module_from_path(module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compute_spearman_stats(x: pd.Series, y: pd.Series) -> dict[str, float]:
    pair = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()
    n_samples = int(len(pair))
    if n_samples < 3:
        return {
            "spearman_r": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n_samples": n_samples,
        }

    rank_x = pair["x"].rank(method="average")
    rank_y = pair["y"].rank(method="average")
    spearman_r = float(rank_x.corr(rank_y, method="pearson"))
    if not np.isfinite(spearman_r):
        return {
            "spearman_r": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n_samples": n_samples,
        }

    spearman_r = float(np.clip(spearman_r, -1.0, 1.0))
    if np.isclose(abs(spearman_r), 1.0):
        t_stat = float(np.sign(spearman_r) * np.inf)
        p_value = 0.0 if t_dist is not None else np.nan
    else:
        denom = np.sqrt(max(1e-15, 1.0 - spearman_r**2))
        t_stat = float(spearman_r * np.sqrt(n_samples - 2) / denom)
        if t_dist is None:
            p_value = np.nan
        else:
            p_value = float(2.0 * t_dist.sf(np.abs(t_stat), df=n_samples - 2))

    return {
        "spearman_r": spearman_r,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_samples": n_samples,
    }


def run_factor_spearman(df: pd.DataFrame, y_col: str, factor_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for factor in factor_cols:
        stats = compute_spearman_stats(df[factor], df[y_col])
        rows.append({"factor": factor, **stats})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["significant_0_05"] = out["p_value"] < ALPHA
    out = out.sort_values(["spearman_r"], key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return out


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


def plot_spearman_heatmap(
    by_year: pd.DataFrame,
    value_col: str,
    output_png: Path,
    title: str,
    cbar_label: str,
    vmin: float,
    vmax: float,
    center: float | None = None,
) -> None:
    if by_year.empty:
        return

    pivot = by_year.pivot(index="factor", columns="year", values=value_col).sort_index()
    if pivot.empty:
        return

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
        vmin=vmin,
        vmax=vmax,
        center=center,
        linewidths=0.5,
        linecolor="#F2F2F2",
        square=True,
        annot_kws={"size": 8.5, "color": "#2F2F2F"},
        cbar_kws={"shrink": 0.9, "pad": 0.02, "label": cbar_label},
    )
    plt.title(title, pad=10)
    plt.xlabel("Year")
    plt.ylabel("Factor")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, facecolor="white")
    plt.close()


def build_pairwise_spearman_matrix(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=variables, columns=variables, dtype=float)
    for i, var_i in enumerate(variables):
        matrix.loc[var_i, var_i] = 1.0
        for j in range(i + 1, len(variables)):
            var_j = variables[j]
            pair = df[[var_i, var_j]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 3:
                corr_val = np.nan
            else:
                corr_val = float(pair[var_i].corr(pair[var_j], method="spearman"))
            matrix.loc[var_i, var_j] = corr_val
            matrix.loc[var_j, var_i] = corr_val
    return matrix


def plot_pairwise_triangle_spearman(matrix: pd.DataFrame, output_png: Path, title: str) -> None:
    if matrix.empty:
        return
    values = matrix.to_numpy(dtype=float)
    mask_upper = np.triu(np.ones_like(values, dtype=bool), k=1)
    set_paper_plot_style()
    cmap = build_soft_blue_red_cmap()
    n_vars = matrix.shape[0]
    fig_side = max(8, n_vars * 0.62 + 2)
    annot_size = float(np.clip(9.3 - n_vars * 0.18, 6.5, 8.5))

    plt.figure(figsize=(fig_side, fig_side))
    sns.heatmap(
        matrix,
        mask=mask_upper,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        linewidths=0.5,
        linecolor="#F2F2F2",
        square=True,
        annot_kws={"size": annot_size, "color": "#2F2F2F"},
        cbar_kws={"shrink": 0.9, "pad": 0.02, "label": "Spearman r"},
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
                SCRIPT_DIR / "prd_panel.csv",
                OUTPUT_DIR / "prd_panel_from_interfaces.csv",
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
    panel.to_csv(OUTPUT_DIR / "prd_panel_from_interfaces.csv", index=False, encoding="utf-8-sig")
    return panel


def main() -> int:
    sns.set_theme(style="white", context="paper")
    if not GEO_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Reference script not found: {GEO_SCRIPT_PATH}")

    geo_module = load_module_from_path(GEO_SCRIPT_PATH)
    base_module = geo_module._load_bth_module() if hasattr(geo_module, "_load_bth_module") else geo_module
    target_cities = list(getattr(geo_module, "PRD_CITIES", getattr(base_module, "BTH_CITIES", [])))
    if hasattr(geo_module, "PRD_CITY_TO_PROVINCE"):
        base_module.CITY_TO_PROVINCE = geo_module.PRD_CITY_TO_PROVINCE
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
    factor_spearman = run_factor_spearman(overall, y_col="pm25", factor_cols=factor_cols)

    by_year_rows: list[pd.DataFrame] = []
    for year, sub_df in overall.groupby("year", observed=True):
        if sub_df["city"].nunique() < 3:
            continue
        year_result = run_factor_spearman(sub_df, y_col="pm25", factor_cols=factor_cols)
        year_result["year"] = int(year)
        by_year_rows.append(year_result)
    spearman_by_year = pd.concat(by_year_rows, ignore_index=True) if by_year_rows else pd.DataFrame()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    factor_spearman.to_csv(OUTPUT_DIR / "prd_spearman_factor.csv", index=False, encoding="utf-8-sig")
    spearman_by_year.to_csv(OUTPUT_DIR / "prd_spearman_factor_by_year.csv", index=False, encoding="utf-8-sig")
    plot_spearman_heatmap(
        by_year=spearman_by_year,
        value_col="spearman_r",
        output_png=OUTPUT_DIR / "prd_spearman_factor_heatmap.png",
        title="PRD Spearman Correlation Heatmap",
        cbar_label="Spearman r",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
    )
    plot_spearman_heatmap(
        by_year=spearman_by_year,
        value_col="p_value",
        output_png=OUTPUT_DIR / "prd_spearman_pvalue_heatmap.png",
        title="PRD Spearman p-value Heatmap",
        cbar_label="p-value",
        vmin=0.0,
        vmax=1.0,
        center=None,
    )
    pairwise_vars = ["pm25", *factor_cols]
    pairwise_spearman = build_pairwise_spearman_matrix(overall, variables=pairwise_vars)
    plot_pairwise_triangle_spearman(
        pairwise_spearman,
        OUTPUT_DIR / "prd_spearman_pairwise_triangle_heatmap.png",
        title="PRD Pairwise Spearman Triangle Heatmap",
    )

    print("=" * 80)
    print("[INFO] PRD Spearman correlation analysis completed")
    print(f"[INFO] Samples: {len(overall)}")
    print(f"[INFO] Factors: {len(factor_cols)}")
    print(f"[INFO] Output directory: {OUTPUT_DIR.resolve()}")
    print("[INFO] Outputs:")
    print("       - prd_panel_from_interfaces.csv")
    print("       - prd_spearman_factor.csv")
    print("       - prd_spearman_factor_by_year.csv")
    print("       - prd_spearman_factor_heatmap.png")
    print("       - prd_spearman_pvalue_heatmap.png")
    print("       - prd_spearman_pairwise_triangle_heatmap.png")
    print("=" * 80)
    if t_dist is None:
        print("[WARN] scipy not installed, p_value uses NaN. Install scipy for significance testing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
