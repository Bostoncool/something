from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# 基础配置
# -----------------------------
START_YEAR = 2018
END_YEAR = 2023
ABS_TOL = 1e-6

OUTPUT_DIR = r"H:\大论文Result\大论文图\三大城市群\AQI VS PM2.5"

REGION_PATHS = {
    "BTH": r"H:\大论文Result\BTH\filtered_daily",
    "YRD": r"H:\大论文Result\YRD\filtered_daily",
    "PRD": r"H:\大论文Result\PRD\filtered_daily",
}

META_COLS = {"__file__", "__missing_cols__"}
ID_COLS = {"date", "hour", "type"}

# IAQI 分级断点（统一）
IAQI_BREAKS = np.array([0, 50, 100, 150, 200, 300, 400, 500], dtype=float)

# 各污染物浓度断点（长度都为 8，对应 7 个区间）
POLLUTANT_BPS: Dict[str, np.ndarray] = {
    "PM2.5": np.array([0, 35, 75, 115, 150, 250, 350, 500], dtype=float),
    "PM10": np.array([0, 50, 150, 250, 350, 420, 500, 600], dtype=float),
    "SO2": np.array([0, 150, 500, 650, 800, 1600, 2100, 2620], dtype=float),
    "NO2": np.array([0, 100, 200, 700, 1200, 2340, 3090, 3840], dtype=float),
    "CO": np.array([0, 5, 10, 35, 60, 90, 120, 150], dtype=float),
    "O3_8H": np.array([0, 100, 160, 215, 265, 800, 1000, 1200], dtype=float),
}

NAME_MAP = {
    "AQI": "AQI",
    "PM2.5": "PM2.5",
    "PM25": "PM2.5",
    "PM10": "PM10",
    "SO2": "SO2",
    "NO2": "NO2",
    "CO": "CO",
    "O3_8H": "O3_8H",
    "O3-8H": "O3_8H",
    "O38H": "O3_8H",
}

REGION_LABELS = {
    "BTH": "京津冀城市群(BTH)",
    "YRD": "长江三角洲城市群(YRD)",
    "PRD": "珠江三角洲城市群(PRD)",
}

# 论文统一配色（ColorBrewer Set2 / 莫兰迪风格）
COLOR_PM25_PRIMARY = "#66c2a5"  # 柔和青绿
COLOR_OTHER_POLLUTED = "#8da0cb"  # 柔和灰蓝

# 柱状图横轴（城市名）刻度字号，按需改数值即可
PLOT_X_TICK_LABELSIZE = 16

# 化学式下标：与中文混排时用 Unicode 下标，勿用 mathtext 的 $...$（见 plot 函数说明）
PM25_UNICODE = "PM\u2082.\u2085"

# 部分环境下 matplotlib 未索引 C:\Windows\Fonts 下的楷体/雅黑，导致 "KaiTi" 等名称解析失败；
# 与《中文字体显示问题排查指南》一致：优先固定可解析的字体链，并在 Windows 上按需 addfont。
_WINDOWS_FONT_FILES_REGISTERED = False


def _register_windows_font_files_once() -> None:
    global _WINDOWS_FONT_FILES_REGISTERED
    if _WINDOWS_FONT_FILES_REGISTERED or os.name != "nt":
        return
    fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for fname in (
        "simkai.ttf",
        "simkai.ttc",
        "msyh.ttc",
        "msyhbd.ttc",
        "simhei.ttf",
        "simsun.ttc",
        "simsunb.ttf",
    ):
        fp = fonts_dir / fname
        if not fp.is_file():
            continue
        try:
            mpl.font_manager.fontManager.addfont(str(fp))
        except (OSError, ValueError, RuntimeError):
            continue
    _WINDOWS_FONT_FILES_REGISTERED = True


def _font_family_resolves(family: str) -> bool:
    fm = mpl.font_manager.fontManager
    try:
        path = fm.findfont(
            mpl.font_manager.FontProperties(family=family),
            fallback_to_default=False,
        )
    except ValueError:
        return False
    return bool(path and os.path.isfile(path))


def build_thesis_figure_font_family() -> list[str]:
    """英文优先 Times New Roman，中文优先 KaiTi（及常见别名），与排查指南中的雅黑/黑体/宋体链兼容。"""
    _register_windows_font_files_once()
    ordered: list[str] = []

    def append_if_new(name: str) -> None:
        if name not in ordered and _font_family_resolves(name):
            ordered.append(name)

    for name in ("Times New Roman", "Times"):
        append_if_new(name)
        if ordered:
            break

    for name in (
        "KaiTi",
        "STKaiti",
        "DFKai-SB",
        "FZKai-Z03",
        "楷体",
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ):
        append_if_new(name)

    # 已注册 simkai 但 family 名仍非上述别名时，从 ttflist 按路径兜底
    if not any(
        x in ordered
        for x in ("KaiTi", "STKaiti", "DFKai-SB", "FZKai-Z03", "楷体")
    ):
        for font in mpl.font_manager.fontManager.ttflist:
            if "simkai" in font.fname.replace("\\", "/").lower():
                if font.name not in ordered:
                    ordered.insert(1 if ordered else 0, font.name)
                break

    append_if_new("DejaVu Sans")
    return ordered if ordered else ["DejaVu Sans"]


@dataclass
class CityStats:
    valid_aqi_obs: int = 0
    polluted_total: int = 0
    judged_total: int = 0
    pm25_primary_total: int = 0

    def __iadd__(self, other: "CityStats") -> "CityStats":
        self.valid_aqi_obs += other.valid_aqi_obs
        self.polluted_total += other.polluted_total
        self.judged_total += other.judged_total
        self.pm25_primary_total += other.pm25_primary_total
        return self


def normalize_pollutant_name(raw_name: object) -> str:
    key = str(raw_name).strip().upper().replace(" ", "")
    return NAME_MAP.get(key, str(raw_name).strip())


def calc_iaqi(concentration: pd.Series, breakpoints: np.ndarray) -> pd.Series:
    values = pd.to_numeric(concentration, errors="coerce").to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)

    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return pd.Series(out, index=concentration.index)

    x = values[valid_mask]
    x = np.clip(x, breakpoints[0], breakpoints[-1])

    idx = np.searchsorted(breakpoints, x, side="right") - 1
    idx = np.clip(idx, 0, len(breakpoints) - 2)

    bp_lo = breakpoints[idx]
    bp_hi = breakpoints[idx + 1]
    iaqi_lo = IAQI_BREAKS[idx]
    iaqi_hi = IAQI_BREAKS[idx + 1]

    out[valid_mask] = (iaqi_hi - iaqi_lo) / (bp_hi - bp_lo) * (x - bp_lo) + iaqi_lo
    return pd.Series(out, index=concentration.index)


def parse_date_column(date_series: pd.Series) -> pd.Series:
    date_str = date_series.astype(str).str.strip()
    is_8digit = date_str.str.fullmatch(r"\d{8}")
    parsed = pd.Series(pd.NaT, index=date_series.index, dtype="datetime64[ns]")

    if is_8digit.any():
        parsed.loc[is_8digit] = pd.to_datetime(
            date_str.loc[is_8digit], format="%Y%m%d", errors="coerce"
        )
    if (~is_8digit).any():
        parsed.loc[~is_8digit] = pd.to_datetime(
            date_str.loc[~is_8digit], errors="coerce"
        )
    return parsed


def reshape_to_hourly_city(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.drop(columns=[c for c in META_COLS if c in df.columns], errors="ignore")
    missing = ID_COLS - set(clean_df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {sorted(missing)}")

    city_cols = [c for c in clean_df.columns if c not in ID_COLS]
    if not city_cols:
        raise ValueError("未识别到城市列")

    long_df = clean_df.melt(
        id_vars=["date", "hour", "type"],
        value_vars=city_cols,
        var_name="city",
        value_name="value",
    )
    long_df["type"] = long_df["type"].map(normalize_pollutant_name)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    return long_df.pivot_table(
        index=["date", "hour", "city"],
        columns="type",
        values="value",
        aggfunc="first",
    ).reset_index()


def analyze_file(
    file_path: str,
    start_year: int,
    end_year: int,
) -> Dict[str, CityStats]:
    raw_df = pd.read_csv(file_path, low_memory=False)
    hourly_df = reshape_to_hourly_city(raw_df)

    if "AQI" not in hourly_df.columns:
        return {}

    dt = parse_date_column(hourly_df["date"])
    hourly_df = hourly_df.loc[dt.dt.year.between(start_year, end_year)].copy()
    if hourly_df.empty:
        return {}

    hourly_df["AQI"] = pd.to_numeric(hourly_df["AQI"], errors="coerce")
    city_result: Dict[str, CityStats] = {}

    for city_name, city_df in hourly_df.groupby("city", sort=False):
        stats = CityStats()
        valid_mask = city_df["AQI"].notna()
        valid_df = city_df.loc[valid_mask].copy()
        stats.valid_aqi_obs = int(len(valid_df))
        if stats.valid_aqi_obs == 0:
            city_result[str(city_name)] = stats
            continue

        polluted_df = valid_df.loc[valid_df["AQI"] > 100].copy()
        stats.polluted_total = int(len(polluted_df))
        if stats.polluted_total == 0:
            city_result[str(city_name)] = stats
            continue

        for pollutant, breakpoints in POLLUTANT_BPS.items():
            if pollutant not in polluted_df.columns:
                polluted_df[pollutant] = np.nan
            polluted_df[f"IAQI_{pollutant}"] = calc_iaqi(
                polluted_df[pollutant], breakpoints
            )

        iaqi_cols = [f"IAQI_{pollutant}" for pollutant in POLLUTANT_BPS]
        iaqi_max = polluted_df[iaqi_cols].max(axis=1, skipna=True)
        judge_mask = iaqi_max.notna()
        stats.judged_total = int(judge_mask.sum())

        pm25_iaqi = polluted_df["IAQI_PM2.5"]
        pm25_primary_mask = (
            judge_mask
            & pm25_iaqi.notna()
            & np.isclose(pm25_iaqi, iaqi_max, rtol=0.0, atol=ABS_TOL)
        )
        stats.pm25_primary_total = int(pm25_primary_mask.sum())
        city_result[str(city_name)] = stats

    return city_result


def list_csv_files(root_dir: str) -> Iterable[Path]:
    root = Path(root_dir)
    if not root.exists():
        return []
    return root.rglob("*.csv")


def os_cpu_count() -> int:
    return os.cpu_count() or 1


def summarize_region(region_name: str, root_dir: str) -> Tuple[str, Dict[str, CityStats]]:
    files = list(list_csv_files(root_dir))
    stats_by_city: Dict[str, CityStats] = {}

    if not files:
        print(f"[{region_name}] 未找到 CSV 文件: {root_dir}")
        return region_name, stats_by_city

    worker_num = min(len(files), max(1, os_cpu_count()))
    print(f"[{region_name}] 文件数: {len(files)}，并行进程数: {worker_num}")

    with ProcessPoolExecutor(max_workers=worker_num) as executor:
        futures = {
            executor.submit(
                analyze_file,
                str(fp),
                START_YEAR,
                END_YEAR,
            ): fp
            for fp in files
        }
        for future in as_completed(futures):
            fp = futures[future]
            try:
                city_stats = future.result()
                for city_name, stats in city_stats.items():
                    if city_name not in stats_by_city:
                        stats_by_city[city_name] = CityStats()
                    stats_by_city[city_name] += stats
            except Exception as err:
                print(f"[{region_name}] 处理失败: {fp} | 错误: {err}")

    return region_name, stats_by_city


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def print_region_result(region_name: str, stats_by_city: Dict[str, CityStats]) -> None:
    valid_aqi_obs = int(sum(v.valid_aqi_obs for v in stats_by_city.values()))
    polluted_total = int(sum(v.polluted_total for v in stats_by_city.values()))
    judged_total = int(sum(v.judged_total for v in stats_by_city.values()))
    pm25_primary_total = int(sum(v.pm25_primary_total for v in stats_by_city.values()))

    polluted_ratio = ratio(polluted_total, valid_aqi_obs)
    pm25_ratio = ratio(pm25_primary_total, judged_total)
    polluted_ratio_text = "NaN" if np.isnan(polluted_ratio) else f"{polluted_ratio:.4%}"
    pm25_ratio_text = "NaN" if np.isnan(pm25_ratio) else f"{pm25_ratio:.4%}"

    print(
        f"[{region_name}] "
        f"城市数: {len(stats_by_city):,} | "
        f"有效AQI观测数: {valid_aqi_obs:,} | "
        f"污染时刻(AQI > 100): {polluted_total:,} | "
        f"AQI > 100占比: {polluted_ratio_text} | "
        f"PM2.5为首要占比: {pm25_ratio_text}"
    )


def build_city_results_df(all_results: Dict[str, Dict[str, CityStats]]) -> pd.DataFrame:
    rows = []
    for region_name, city_stats in all_results.items():
        for city_name, stats in city_stats.items():
            polluted_ratio = ratio(stats.polluted_total, stats.valid_aqi_obs)
            pm25_ratio = ratio(stats.pm25_primary_total, stats.judged_total)
            rows.append(
                {
                    "区域代码": region_name,
                    "区域": REGION_LABELS.get(region_name, region_name),
                    "城市": city_name,
                    "有效AQI观测数": stats.valid_aqi_obs,
                    "污染时刻数(AQI > 100)": stats.polluted_total,
                    "AQI > 100占所有有效AQI数据占比": (
                        polluted_ratio if not np.isnan(polluted_ratio) else None
                    ),
                    "可判断首要污染物数": stats.judged_total,
                    "PM2.5为首要污染物数": stats.pm25_primary_total,
                    "PM2.5为首要污染物占比": pm25_ratio if not np.isnan(pm25_ratio) else None,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["区域代码", "城市"]).reset_index(drop=True)


def save_results_csv(all_results: Dict[str, Dict[str, CityStats]], out_dir: Path) -> None:
    df = build_city_results_df(all_results)
    csv_path = out_dir / "三大城市群各城市AQI污染与PM2.5首要污染物统计.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 已保存: {csv_path}")


def plot_aqi_polluted_ratio_by_region(
    region_name: str, stats_by_city: Dict[str, CityStats], out_dir: Path
) -> None:
    if not stats_by_city:
        print(f"[{region_name}] 无城市数据，跳过绘图")
        return

    records = []
    for city_name, stats in stats_by_city.items():
        aqi_over_100_ratio = ratio(stats.polluted_total, stats.valid_aqi_obs)
        pm25_primary_ratio_on_all_valid = ratio(stats.pm25_primary_total, stats.valid_aqi_obs)
        records.append((city_name, aqi_over_100_ratio, pm25_primary_ratio_on_all_valid))

    records.sort(
        key=lambda item: (np.isnan(item[1]), -item[1] if not np.isnan(item[1]) else 0.0)
    )

    labels = [item[0] for item in records]
    aqi_ratios = [item[1] for item in records]
    pm25_ratios = [item[2] for item in records]

    aqi_plot = [value if not np.isnan(value) else 0.0 for value in aqi_ratios]
    pm25_plot = [value if not np.isnan(value) else 0.0 for value in pm25_ratios]
    other_polluted_plot = [max(aqi - pm25, 0.0) for aqi, pm25 in zip(aqi_plot, pm25_plot)]

    # 具体字体名列表 + 多文件回退（Agg）；Windows 上先注册系统字体文件，避免 KaiTi/SimHei 解析不到。
    figure_rc = {
        "font.family": build_thesis_figure_font_family(),
        "axes.unicode_minus": False,
    }

    with mpl.rc_context(figure_rc):
        fig_width = max(10, len(labels) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        bars_pm25 = ax.bar(
            labels,
            [value * 100 for value in pm25_plot],
            color=COLOR_PM25_PRIMARY,
            label=PM25_UNICODE + "为首要污染物",
        )
        bars_other = ax.bar(
            labels,
            [value * 100 for value in other_polluted_plot],
            bottom=[value * 100 for value in pm25_plot],
            color=COLOR_OTHER_POLLUTED,
            label="其他AQI > 100污染",
        )

        for idx, (bar_pm25, bar_other, aqi_ratio, pm25_ratio) in enumerate(
            zip(bars_pm25, bars_other, aqi_ratios, pm25_ratios)
        ):
            if np.isnan(aqi_ratio):
                continue
            total_height = (pm25_plot[idx] + other_polluted_plot[idx]) * 100
            text = f"{aqi_ratio:.1%}"
            ax.text(
                bar_pm25.get_x() + bar_pm25.get_width() / 2,
                total_height + 0.8,
                text,
                ha="center",
                va="bottom",
                fontsize=12,
            )
            if not np.isnan(pm25_ratio) and pm25_plot[idx] >= 0.01:
                ax.text(
                    bar_pm25.get_x() + bar_pm25.get_width() / 2,
                    (pm25_plot[idx] * 100) / 2,
                    f"{pm25_ratio:.1%}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#2d5a4a",
                )

        ax.set_ylabel("占所有有效AQI数据占比 (%)", fontsize=16)
        region_label = REGION_LABELS.get(region_name, region_name)
        ax.set_title(
            f"{START_YEAR}-{END_YEAR} {region_label}\n各城市AQI > 100及{PM25_UNICODE}首要占比",
            fontsize=17,
        )
        ax.legend(loc="upper right", fontsize=16)
        max_height_pct = max(aqi_plot) * 100 if aqi_plot else 0.0
        if max_height_pct <= 0:
            y_upper = 5.0
        else:
            # 随数据上限伸缩，避免小占比区域（如珠三角）被强制拉到 10% 显得柱子过矮
            y_upper = max_height_pct * 1.18 + max(1.0, max_height_pct * 0.05)
        ax.set_ylim(0, y_upper)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", rotation=60, labelsize=PLOT_X_TICK_LABELSIZE)
        ax.tick_params(axis="y", labelsize=16)
        plt.tight_layout()

        region_tag = region_name.lower()
        png_path = out_dir / f"{region_tag}_各城市AQI大于100占比.png"
        svg_path = out_dir / f"{region_tag}_各城市AQI大于100占比.svg"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        plt.close()

    print(f"统计图已保存: {png_path}")
    print(f"统计图已保存: {svg_path}")


def main() -> None:
    print(f"统计年份: {START_YEAR}-{END_YEAR}")
    print("开始并行处理...\n")

    all_results: Dict[str, Dict[str, CityStats]] = {}
    for region_name, region_dir in REGION_PATHS.items():
        name, stats_by_city = summarize_region(region_name, region_dir)
        all_results[name] = stats_by_city
        print_region_result(name, stats_by_city)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_results_csv(all_results, out_dir)
    for region_name, stats_by_city in all_results.items():
        plot_aqi_polluted_ratio_by_region(region_name, stats_by_city, out_dir)


if __name__ == "__main__":
    main()
