from __future__ import annotations

import argparse
import importlib.util
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from scipy.stats import f as f_dist
except Exception:  # pylint: disable=broad-except
    f_dist = None


BTH_CITIES = [
    "北京",
    "天津",
    "石家庄",
    "唐山",
    "秦皇岛",
    "邯郸",
    "邢台",
    "保定",
    "张家口",
    "承德",
    "沧州",
    "廊坊",
    "衡水",
]


def load_module_from_path(module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_city_name(city_series: pd.Series) -> pd.Series:
    return (
        city_series.astype(str)
        .str.strip()
        .str.replace("市", "", regex=False)
        .str.replace("地区", "", regex=False)
    )


def reshape_city_year_table(df: pd.DataFrame, city_col: str, value_name: str) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    year_cols = [c for c in data.columns if str(c).strip().isdigit() and len(str(c).strip()) == 4]
    if year_cols:
        long_df = data.melt(
            id_vars=[city_col],
            value_vars=year_cols,
            var_name="year",
            value_name=value_name,
        )
        long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    elif "year" in data.columns:
        long_df = data.rename(columns={city_col: "city", "year": "year"})
        long_df = long_df[["city", "year", value_name]].copy()
        long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    else:
        raise ValueError(f"{value_name} 未找到年份列，请检查源数据结构。")

    long_df = long_df.rename(columns={city_col: "city"})
    long_df["city"] = normalize_city_name(long_df["city"])
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    return long_df.dropna(subset=["city", "year"])


def build_panel_from_interfaces(data_read_dir: Path, pm25_city_year_csv: Path) -> pd.DataFrame:
    if not pm25_city_year_csv.exists():
        raise FileNotFoundError(f"未找到 PM2.5 城市年均表: {pm25_city_year_csv}")

    pm25_df = pd.read_csv(pm25_city_year_csv, encoding="utf-8-sig")
    pm25_df.columns = [str(c).strip() for c in pm25_df.columns]
    if {"city", "year", "pm25"} - set(pm25_df.columns):
        raise ValueError("PM2.5 表必须包含列: city, year, pm25")
    pm25_df["city"] = normalize_city_name(pm25_df["city"])
    pm25_df["year"] = pd.to_numeric(pm25_df["year"], errors="coerce").astype("Int64")
    pm25_df["pm25"] = pd.to_numeric(pm25_df["pm25"], errors="coerce")
    panel = pm25_df.copy()

    gdp_pop_module = load_module_from_path(data_read_dir / "GDP-population.py")
    pop_df, gdp_df = gdp_pop_module.load_and_filter_data()
    pop_long = reshape_city_year_table(pop_df, city_col="城市", value_name="population")
    gdp_long = reshape_city_year_table(gdp_df, city_col="城市", value_name="gdp")

    energy_module = load_module_from_path(data_read_dir / "Energy-Consumption.py")
    energy_df = energy_module.load_energy_data()
    energy_long = energy_df.rename(
        columns={
            "城市": "city",
            "年度": "year",
            "插值_全社会用电量万千瓦时": "electricity",
            "插值_人工煤气和天然气供气总量万立方米市辖区": "gas_supply",
            "插值_液化石油气供气总量吨市辖区": "lpg_supply",
        }
    )
    energy_long["city"] = normalize_city_name(energy_long["city"])
    energy_long["year"] = pd.to_numeric(energy_long["year"], errors="coerce").astype("Int64")
    for col in ("electricity", "gas_supply", "lpg_supply"):
        energy_long[col] = pd.to_numeric(energy_long[col], errors="coerce")
    energy_long = energy_long[["city", "year", "electricity", "gas_supply", "lpg_supply"]]

    newpower_module = load_module_from_path(data_read_dir / "Newpower.py")
    newpower_df = newpower_module.load_and_filter_newpower()
    newpower_col = next(
        (c for c in newpower_df.columns if "保有量" in str(c)),
        None,
    )
    if newpower_col is None:
        raise ValueError("新能源汽车数据中未识别到保有量列。")
    newpower_long = newpower_df.rename(columns={"城市": "city", "年份": "year", newpower_col: "new_energy_vehicles"})
    newpower_long["city"] = normalize_city_name(newpower_long["city"])
    newpower_long["year"] = pd.to_numeric(newpower_long["year"], errors="coerce").astype("Int64")
    newpower_long["new_energy_vehicles"] = pd.to_numeric(newpower_long["new_energy_vehicles"], errors="coerce")
    newpower_long = newpower_long[["city", "year", "new_energy_vehicles"]]

    road_module = load_module_from_path(data_read_dir / "Road.py")
    road_df = road_module.load_road_density_data()
    road_long = road_df.rename(columns={"市名": "city", "year": "year", "路网密度": "road_density"})
    road_long["city"] = normalize_city_name(road_long["city"])
    road_long["year"] = pd.to_numeric(road_long["year"], errors="coerce").astype("Int64")
    road_long["road_density"] = pd.to_numeric(road_long["road_density"], errors="coerce")
    road_long = road_long[["city", "year", "road_density"]]

    night_module = load_module_from_path(data_read_dir / "NightLight.py")
    night_df = night_module.load_nightlight_data()
    night_df.columns = [str(c).strip() for c in night_df.columns]
    night_city_col = next((c for c in night_df.columns if c in {"城市", "city"}), None)
    if night_city_col is None:
        raise ValueError("夜间灯光数据中未识别到城市列。")
    night_value_col = next(
        (c for c in night_df.columns if c not in {night_city_col} and ("灯光" in c or "night" in c.lower())),
        None,
    )
    if night_value_col is None:
        numeric_cols = [c for c in night_df.columns if c != night_city_col and pd.api.types.is_numeric_dtype(night_df[c])]
        if not numeric_cols:
            raise ValueError("夜间灯光数据中未识别到数值列。")
        night_value_col = numeric_cols[0]
    night_source = night_df.rename(columns={night_city_col: "城市", night_value_col: "night_light"})
    night_long = reshape_city_year_table(
        night_source,
        city_col="城市",
        value_name="night_light",
    )

    merge_tables = [pop_long, gdp_long, energy_long, newpower_long, road_long, night_long]
    for table in merge_tables:
        panel = panel.merge(table, on=["city", "year"], how="left")
    return panel


def discretize_factor(series: pd.Series, bins: int = 6) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.nunique() <= 1:
        return pd.Series(np.where(s.notna(), "all", np.nan), index=s.index, dtype="object")

    k = int(min(max(2, bins), valid.nunique()))
    try:
        import mapclassify as mc  # type: ignore

        classifier = mc.NaturalBreaks(valid.to_numpy(), k=k)
        edges = np.r_[-np.inf, classifier.bins]
        labels = [f"L{i+1}" for i in range(len(edges) - 1)]
        return pd.cut(s, bins=edges, labels=labels, include_lowest=True).astype("object")
    except Exception:  # pylint: disable=broad-except
        return pd.qcut(s.rank(method="first"), q=k, duplicates="drop").astype("object")


def compute_q_stat(y: pd.Series, strata: pd.Series) -> tuple[float, float]:
    df = pd.DataFrame({"y": pd.to_numeric(y, errors="coerce"), "h": strata}).dropna()
    n = len(df)
    l = df["h"].nunique()
    if n < 3 or l < 2:
        return np.nan, np.nan

    sigma2 = df["y"].var(ddof=1)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return 0.0, np.nan

    grouped = df.groupby("h", observed=True)["y"]
    ssw = sum(len(g) * g.var(ddof=1) if len(g) > 1 else 0.0 for _, g in grouped)
    q = float(np.clip(1.0 - ssw / (n * sigma2), 0.0, 1.0))

    if f_dist is None or n <= l or l <= 1 or q >= 1:
        return q, np.nan
    f_value = ((n - l) * q) / ((l - 1) * (1 - q + 1e-12))
    p_value = float(f_dist.sf(f_value, l - 1, n - l))
    return q, p_value


def classify_interaction(q1: float, q2: float, q12: float, atol: float = 1e-3) -> str:
    q_sum = q1 + q2
    q_max = max(q1, q2)
    q_min = min(q1, q2)
    if np.isclose(q12, q_sum, atol=atol):
        return "独立"
    if q12 > q_sum:
        return "双协同(非线性增强)"
    if q_max < q12 <= q_sum:
        return "协同(双因子增强)"
    if q_min < q12 <= q_max:
        return "单拮抗(单因子减弱)"
    return "拮抗(非线性减弱)"


def run_factor_detector(df: pd.DataFrame, y_col: str, factor_cols: list[str], bins: int) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    results = []
    strata_map: dict[str, pd.Series] = {}
    for factor in factor_cols:
        strata = discretize_factor(df[factor], bins=bins)
        q, p = compute_q_stat(df[y_col], strata)
        strata_map[factor] = strata
        results.append({"factor": factor, "q": q, "p_value": p})
    return pd.DataFrame(results).sort_values("q", ascending=False), strata_map


def run_interaction_detector(y: pd.Series, strata_map: dict[str, pd.Series], factor_q: pd.DataFrame) -> pd.DataFrame:
    q_lookup = factor_q.set_index("factor")["q"].to_dict()
    rows = []
    for f1, f2 in combinations(strata_map.keys(), 2):
        combined = strata_map[f1].astype(str) + "__" + strata_map[f2].astype(str)
        q12, p12 = compute_q_stat(y, combined)
        q1, q2 = q_lookup.get(f1, np.nan), q_lookup.get(f2, np.nan)
        rows.append(
            {
                "factor_1": f1,
                "factor_2": f2,
                "q1": q1,
                "q2": q2,
                "q_interaction": q12,
                "p_interaction": p12,
                "interaction_type": classify_interaction(q1, q2, q12),
            }
        )
    return pd.DataFrame(rows).sort_values("q_interaction", ascending=False)


def plot_factor_heatmap(q_by_year: pd.DataFrame, output_png: Path) -> None:
    if q_by_year.empty:
        return
    pivot = q_by_year.pivot(index="factor", columns="year", values="q").sort_index()
    plt.figure(figsize=(max(8, pivot.shape[1] * 1.2), max(4, pivot.shape[0] * 0.45)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, linewidths=0.3)
    plt.title("BTH Geo-detector q Heatmap")
    plt.xlabel("Year")
    plt.ylabel("Factor")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()


def plot_interaction_heatmap(inter_df: pd.DataFrame, output_png: Path) -> None:
    if inter_df.empty:
        return
    factors = sorted(set(inter_df["factor_1"]).union(inter_df["factor_2"]))
    matrix = pd.DataFrame(np.nan, index=factors, columns=factors)
    for _, row in inter_df.iterrows():
        matrix.loc[row["factor_1"], row["factor_2"]] = row["q_interaction"]
        matrix.loc[row["factor_2"], row["factor_1"]] = row["q_interaction"]
    np.fill_diagonal(matrix.values, 1.0)
    plt.figure(figsize=(max(8, len(factors) * 0.7), max(6, len(factors) * 0.7)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdPu", vmin=0, vmax=1, linewidths=0.3)
    plt.title("BTH Interaction q Heatmap")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geo-detector: 京津冀 PM2.5 因子探测与交互探测")
    parser.add_argument("--panel-csv", type=str, default="", help="已整合面板数据（city, year, pm25 + 因子列）")
    parser.add_argument("--pm25-city-year-csv", type=str, default="", help="用于自动拼接接口数据的 PM2.5 城市年均 CSV")
    parser.add_argument("--data-read-dir", type=str, default="", help="Data Read 目录路径")
    parser.add_argument("--bins", type=int, default=6, help="离散分层数（默认6）")
    parser.add_argument("--output-dir", type=str, default="./outputs/geo_detector_bth", help="输出目录")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sns.set_theme(style="white")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.panel_csv:
        panel = pd.read_csv(args.panel_csv, encoding="utf-8-sig")
    else:
        if not args.pm25_city_year_csv or not args.data_read_dir:
            raise ValueError("未提供 --panel-csv 时，必须同时提供 --pm25-city-year-csv 和 --data-read-dir。")
        panel = build_panel_from_interfaces(
            data_read_dir=Path(args.data_read_dir).expanduser().resolve(),
            pm25_city_year_csv=Path(args.pm25_city_year_csv).expanduser().resolve(),
        )
        panel.to_csv(output_dir / "bth_panel_from_interfaces.csv", index=False, encoding="utf-8-sig")

    panel.columns = [str(c).strip() for c in panel.columns]
    required_cols = {"city", "year", "pm25"}
    if required_cols - set(panel.columns):
        raise ValueError("面板数据至少包含列: city, year, pm25")

    panel["city"] = normalize_city_name(panel["city"])
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel["pm25"] = pd.to_numeric(panel["pm25"], errors="coerce")
    panel = panel.loc[panel["city"].isin(BTH_CITIES)].copy()

    factor_cols = [c for c in panel.columns if c not in {"city", "year", "pm25"}]
    if not factor_cols:
        raise ValueError("未检测到因子列，请检查输入数据。")

    numeric_factors = []
    for col in factor_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
        if panel[col].notna().sum() > 0:
            numeric_factors.append(col)
    factor_cols = numeric_factors
    if not factor_cols:
        raise ValueError("所有因子列均无法转为数值。")

    overall = panel.dropna(subset=["pm25"]).copy()
    factor_q, strata_map = run_factor_detector(overall, y_col="pm25", factor_cols=factor_cols, bins=args.bins)
    inter_q = run_interaction_detector(overall["pm25"], strata_map, factor_q)

    by_year_rows = []
    for year, sub_df in overall.groupby("year", observed=True):
        if sub_df["city"].nunique() < 3:
            continue
        year_q, _ = run_factor_detector(sub_df, y_col="pm25", factor_cols=factor_cols, bins=args.bins)
        year_q["year"] = int(year)
        by_year_rows.append(year_q)
    q_by_year = pd.concat(by_year_rows, ignore_index=True) if by_year_rows else pd.DataFrame()

    factor_q.to_csv(output_dir / "bth_factor_detector.csv", index=False, encoding="utf-8-sig")
    inter_q.to_csv(output_dir / "bth_interaction_detector.csv", index=False, encoding="utf-8-sig")
    q_by_year.to_csv(output_dir / "bth_factor_q_by_year.csv", index=False, encoding="utf-8-sig")

    plot_factor_heatmap(q_by_year, output_dir / "bth_factor_q_heatmap.png")
    plot_interaction_heatmap(inter_q, output_dir / "bth_interaction_q_heatmap.png")

    print("=" * 80)
    print("[INFO] 京津冀 Geo-detector 分析完成")
    print(f"[INFO] 样本量: {len(overall)}")
    print(f"[INFO] 因子数量: {len(factor_cols)}")
    print(f"[INFO] 输出目录: {output_dir}")
    print("[INFO] 主要输出:")
    print("       - bth_factor_detector.csv")
    print("       - bth_interaction_detector.csv")
    print("       - bth_factor_q_by_year.csv")
    print("       - bth_factor_q_heatmap.png")
    print("       - bth_interaction_q_heatmap.png")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
