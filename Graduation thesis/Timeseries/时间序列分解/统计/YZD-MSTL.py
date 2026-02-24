import os
import importlib
import unicodedata
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from statsmodels.tsa.seasonal import MSTL
from tqdm import tqdm

warnings.filterwarnings("ignore")

def _load_lazy_pinyin():
    try:
        pypinyin_module = importlib.import_module("pypinyin")
        return pypinyin_module.lazy_pinyin
    except Exception:
        return None


lazy_pinyin = _load_lazy_pinyin()

def configure_plot_fonts():
    """
    自动配置可用中文字体，避免图片中文显示为方块。
    """
    chinese_font_chain = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]
    available_font_names = {font.name for font in font_manager.fontManager.ttflist}
    available_in_chain = [name for name in chinese_font_chain if name in available_font_names]
    if not available_in_chain:
        print("Warning: No common Chinese font found. Chinese text may not render correctly.")

    sns.set_theme(style="whitegrid", palette="husl")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = chinese_font_chain
    plt.rcParams["axes.unicode_minus"] = False
    if available_in_chain:
        print(f"Using plotting fonts: {', '.join(available_in_chain)}")


configure_plot_fonts()


CITY_PINYIN_MAP = {
    "北京": "beijing",
    "天津": "tianjin",
    "石家庄": "shijiazhuang",
    "唐山": "tangshan",
    "秦皇岛": "qinhuangdao",
    "邯郸": "handan",
    "邢台": "xingtai",
    "保定": "baoding",
    "张家口": "zhangjiakou",
    "承德": "chengde",
    "沧州": "cangzhou",
    "廊坊": "langfang",
    "衡水": "hengshui",
    "上海": "shanghai",
    "南京": "nanjing",
    "无锡": "wuxi",
    "常州": "changzhou",
    "苏州": "suzhou",
    "南通": "nantong",
    "盐城": "yancheng",
    "扬州": "yangzhou",
    "镇江": "zhenjiang",
    "泰州": "taizhou",
    "杭州": "hangzhou",
    "宁波": "ningbo",
    "温州": "wenzhou",
    "嘉兴": "jiaxing",
    "湖州": "huzhou",
    "绍兴": "shaoxing",
    "金华": "jinhua",
    "舟山": "zhoushan",
    "台州": "taizhou_zj",
    "合肥": "hefei",
    "芜湖": "wuhu",
    "马鞍山": "maanshan",
    "铜陵": "tongling",
    "安庆": "anqing",
    "滁州": "chuzhou",
    "池州": "chizhou",
    "宣城": "xuancheng",
    "广州": "guangzhou",
    "深圳": "shenzhen",
    "珠海": "zhuhai",
    "佛山": "foshan",
    "江门": "jiangmen",
    "肇庆": "zhaoqing",
    "惠州": "huizhou",
    "东莞": "dongguan",
    "中山": "zhongshan",
}

ASCII_TO_CITY_MAP = {value: key for key, value in CITY_PINYIN_MAP.items()}


def city_display_name(city_name):
    """
    将拼音城市名恢复为中文城市名（若可映射）。
    """
    return ASCII_TO_CITY_MAP.get(str(city_name), str(city_name))


def city_to_ascii_name(city_name):
    """
    将城市名统一为ASCII拼音，避免中文渲染异常。
    """
    normalized = unicodedata.normalize("NFKC", str(city_name))
    no_control_chars = "".join(
        ch for ch in normalized if not unicodedata.category(ch).startswith("C")
    )
    compact = "".join(no_control_chars.split())
    compact = compact if compact else str(city_name).strip()

    mapped = CITY_PINYIN_MAP.get(compact)
    if mapped:
        return mapped

    if lazy_pinyin is not None:
        pinyin_tokens = lazy_pinyin(compact, errors="ignore")
        pinyin_joined = "_".join(token.strip().lower() for token in pinyin_tokens if token.strip())
        if pinyin_joined:
            return pinyin_joined

    fallback = []
    for ch in compact:
        if ch.isascii() and ch.isalnum():
            fallback.append(ch.lower())
        else:
            fallback.append("_")
    fallback_name = "_".join(part for part in "".join(fallback).split("_") if part)
    return fallback_name if fallback_name else "unknown_city"


def process_single_file(file_path):
    """
    处理单个CSV文件，提取年月和城市污染物数据

    参数:
    file_path: CSV文件路径

    返回:
    (month_key, month_data) 元组，month_data是字典 {city: {pollutant: value}}
    month_key格式为'YYYY-MM'
    """
    try:
        file_name = Path(file_path).stem
        date_str = file_name.split("_")[-1][:8]
        year_month = f"{date_str[:4]}-{date_str[4:6]}"

        df = pd.read_csv(file_path, encoding="utf-8-sig")

        raw_city_columns = [
            col
            for col in df.columns
            if col not in ["date", "hour", "type", "__file__", "__missing_cols__"]
        ]

        realtime_types = ["PM2.5"]
        df_realtime = df[df["type"].isin(realtime_types)].copy()
        daily_avg = df_realtime.groupby("type")[raw_city_columns].mean()

        month_data = {}
        for raw_city in raw_city_columns:
            city = city_to_ascii_name(raw_city)
            month_data[city] = {}
            for pollutant in realtime_types:
                if pollutant in daily_avg.index:
                    value = daily_avg.loc[pollutant, raw_city]
                    if not pd.isna(value):
                        month_data[city][pollutant] = value

        return year_month, month_data

    except Exception as exc:
        print(f"Error processing file {file_path}: {exc}")
        return None


def calculate_monthly_means(folder_path, n_processes=None):
    """
    从原始CSV文件计算每个城市的月度平均值
    """
    all_data = {}

    csv_files = list(Path(folder_path).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    if len(csv_files) == 0:
        print(f"Error: No CSV files found in {folder_path}")
        return None

    if n_processes is None:
        n_processes = cpu_count()
    print(f"Using {n_processes} processes for parallel processing...")

    print("Reading and processing CSV files...")
    with Pool(processes=n_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_file, csv_files),
                total=len(csv_files),
                desc="Processing CSV files",
                unit="files",
            )
        )

    for result in results:
        if result is None:
            continue

        month_key, month_data = result
        if month_key not in all_data:
            all_data[month_key] = {}

        for city, pollutants in month_data.items():
            if city not in all_data[month_key]:
                all_data[month_key][city] = {}
            for pollutant, value in pollutants.items():
                all_data[month_key][city].setdefault(pollutant, []).append(value)

    monthly_avg_data = {}
    for month_key, month_data in all_data.items():
        monthly_avg_data[month_key] = {}
        for city, pollutants in month_data.items():
            monthly_avg_data[month_key][city] = {}
            for pollutant, values in pollutants.items():
                if values:
                    monthly_avg_data[month_key][city][pollutant] = np.mean(values)

    all_data = monthly_avg_data
    months = sorted(all_data.keys())
    print(f"Processed {len(months)} months of data: {months[0]} to {months[-1]}")

    realtime_types = ["PM2.5"]
    all_cities = set()
    for month_data in all_data.values():
        all_cities.update(month_data.keys())
    print(f"Found {len(all_cities)} cities")

    data_list = []
    for city in all_cities:
        for pollutant in realtime_types:
            for month in months:
                if (
                    month in all_data
                    and city in all_data[month]
                    and pollutant in all_data[month][city]
                ):
                    data_list.append(
                        {
                            "City": city,
                            "Pollutant": pollutant,
                            "Month": pd.to_datetime(month, format="%Y-%m"),
                            "Value": all_data[month][city][pollutant],
                        }
                    )

    if not data_list:
        print("No data found")
        return None

    df = pd.DataFrame(data_list)
    print(f"Created DataFrame with {len(df)} records")
    return df


def prepare_time_series_data(df, city, pollutant):
    """
    为MSTL分解准备时间序列数据
    """
    mask = (df["City"] == city) & (df["Pollutant"] == pollutant)
    city_pollutant_data = df.loc[mask].copy()
    if city_pollutant_data.empty:
        return None

    city_pollutant_data = city_pollutant_data.sort_values("Month")
    ts_data = pd.Series(
        city_pollutant_data["Value"].values,
        index=city_pollutant_data["Month"],
        name=f"{city}_{pollutant}",
    )
    return ts_data


def _validate_periods(data_length, seasonal_periods):
    valid_periods = []
    for period in seasonal_periods:
        if period < 2:
            continue
        if data_length >= (2 * period):
            valid_periods.append(int(period))
        else:
            print(
                f"    Skip period {period}: data length {data_length} is less than 2x period"
            )
    return tuple(sorted(set(valid_periods)))


def perform_mstl_decomposition(ts_data, seasonal_periods=(12,)):
    """
    执行MSTL分解

    参数:
    ts_data: 时间序列数据
    seasonal_periods: 多季节周期，例如:
        - 月度数据: (12,)
        - 日度数据: (7, 365)
    """
    try:
        ts_data_clean = ts_data.dropna()
        if len(ts_data_clean) < len(ts_data) * 0.8:
            print(f"    Warning: {len(ts_data) - len(ts_data_clean)} missing values found")

        data_length = len(ts_data_clean)
        periods = _validate_periods(data_length, seasonal_periods)
        if not periods:
            raise ValueError(
                f"No valid periods for data length {data_length}. "
                f"Input periods: {seasonal_periods}"
            )

        mstl = MSTL(
            ts_data_clean,
            periods=periods,
            stl_kwargs={"robust": True},
        )
        result = mstl.fit()
        return result

    except Exception as exc:
        print(f"    MSTL decomposition failed: {exc}")
        return None


def _get_seasonal_df(result):
    seasonal = result.seasonal
    if isinstance(seasonal, pd.Series):
        return seasonal.to_frame(name="seasonal_1")
    return seasonal


def plot_mstl_components(result, city, pollutant, save_path=None):
    """
    绘制MSTL分解各成分（支持多季节成分）
    """
    city_label = city_display_name(city)
    seasonal_df = _get_seasonal_df(result)
    n_seasonal = seasonal_df.shape[1]

    n_rows = 3 + n_seasonal
    fig, axes = plt.subplots(n_rows, 1, figsize=(15, 3.2 * n_rows), sharex=True)
    axes = np.atleast_1d(axes)

    axes[0].plot(result.observed, "b-", linewidth=1.5, alpha=0.8)
    axes[0].set_title(
        f"{city_label} - {pollutant} 原始时间序列", fontsize=13, fontweight="bold"
    )
    axes[0].set_ylabel("浓度")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result.trend, "r-", linewidth=1.5, alpha=0.8)
    axes[1].set_title("趋势项", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Trend")
    axes[1].grid(True, alpha=0.3)

    for idx, column in enumerate(seasonal_df.columns, start=2):
        axes[idx].plot(seasonal_df[column], "g-", linewidth=1.2, alpha=0.85)
        axes[idx].set_title(
            f"季节项（{column}）", fontsize=12, fontweight="bold"
        )
        axes[idx].set_ylabel("季节")
        axes[idx].grid(True, alpha=0.3)

    resid_axis = 2 + n_seasonal
    axes[resid_axis].plot(result.resid, color="orange", linewidth=1, alpha=0.7)
    axes[resid_axis].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[resid_axis].set_title("残差项", fontsize=12, fontweight="bold")
    axes[resid_axis].set_ylabel("残差")
    axes[resid_axis].set_xlabel("时间")
    axes[resid_axis].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_seasonal_analysis(result, city, pollutant, save_path=None):
    """
    季节性详细分析图（基于总季节项）
    """
    city_label = city_display_name(city)
    seasonal_df = _get_seasonal_df(result)
    seasonal_total = seasonal_df.sum(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    seasonal_plot_df = pd.DataFrame(
        {
            "月份": seasonal_total.index.month,
            "季节值": seasonal_total.values,
        }
    )

    sns.boxplot(data=seasonal_plot_df, x="月份", y="季节值", ax=axes[0, 0])
    axes[0, 0].set_title(
        f"{city_label} - {pollutant} 总季节项月度分布", fontsize=12, fontweight="bold"
    )
    axes[0, 0].set_xlabel("月份")
    axes[0, 0].set_ylabel("季节值")
    axes[0, 0].grid(True, alpha=0.3)

    sns.violinplot(data=seasonal_plot_df, x="月份", y="季节值", ax=axes[0, 1])
    axes[0, 1].set_title("总季节项分布密度", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("月份")
    axes[0, 1].set_ylabel("季节值")
    axes[0, 1].grid(True, alpha=0.3)

    trend_changes = result.trend.pct_change() * 100
    axes[1, 0].plot(trend_changes.index, trend_changes.values, "b-", alpha=0.7)
    axes[1, 0].set_title("趋势项月变化率", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("时间")
    axes[1, 0].set_ylabel("变化率(%)")
    axes[1, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(result.resid.values, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    axes[1, 1].axvline(x=0, color="red", linestyle="--", alpha=0.7)
    axes[1, 1].set_title("残差分布", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("残差值")
    axes[1, 1].set_ylabel("频次")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"{city_label} - {pollutant} MSTL分解详细分析", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def analyze_mstl_results(result, city, pollutant):
    """
    分析MSTL分解结果并返回统计信息
    """
    analysis = {
        "city": city,
        "pollutant": pollutant,
        "trend_strength": None,
        "seasonal_strength": None,
        "residual_strength": None,
        "trend_direction": None,
        "seasonal_peak_month": None,
        "seasonal_valley_month": None,
        "data_range": None,
        "trend_range": None,
        "seasonal_range": None,
    }

    try:
        seasonal_df = _get_seasonal_df(result)
        seasonal_total = seasonal_df.sum(axis=1)

        observed_var = np.var(result.observed)
        trend_var = np.var(result.trend)
        seasonal_var = np.var(seasonal_total)
        residual_var = np.var(result.resid)

        analysis["trend_strength"] = trend_var / observed_var if observed_var > 0 else 0
        analysis["seasonal_strength"] = (
            seasonal_var / (observed_var - trend_var) if (observed_var - trend_var) > 0 else 0
        )
        analysis["residual_strength"] = residual_var / observed_var if observed_var > 0 else 0

        trend_start = result.trend.iloc[0]
        trend_end = result.trend.iloc[-1]
        if trend_end > trend_start * 1.05:
            analysis["trend_direction"] = "Increasing"
        elif trend_end < trend_start * 0.95:
            analysis["trend_direction"] = "Decreasing"
        else:
            analysis["trend_direction"] = "Stable"

        seasonal_by_month = seasonal_total.groupby(seasonal_total.index.month).mean()
        analysis["seasonal_peak_month"] = seasonal_by_month.idxmax()
        analysis["seasonal_valley_month"] = seasonal_by_month.idxmin()

        analysis["data_range"] = f"{result.observed.min():.2f} - {result.observed.max():.2f}"
        analysis["trend_range"] = f"{result.trend.min():.2f} - {result.trend.max():.2f}"
        analysis["seasonal_range"] = f"{seasonal_total.min():.2f} - {seasonal_total.max():.2f}"

    except Exception as exc:
        print(f"Error analyzing results: {exc}")

    return analysis


def save_analysis_results(analyses, output_dir):
    if not analyses:
        print("No analysis results to save")
        return

    df_results = pd.DataFrame(analyses)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "MSTL_Decomposition_Analysis.csv")
    df_results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"MSTL decomposition analysis results saved to: {output_path}")


def main():
    print("=" * 60)
    print("PM2.5 Time Series MSTL Decomposition Analysis")
    print("=" * 60)

    input_folder = r"H:\DATA Science\大论文Result\YZD\filtered_daily"
    output_dir = r"H:\DATA Science\大论文Result\YZD\MSTL时间序列分解"

    # 当前脚本处理月度序列，推荐(12,)
    # 若后续改为日度序列，可改为(7, 365)
    seasonal_periods = (7,365)

    if not os.path.exists(input_folder):
        print(f"Error: Input folder does not exist: {input_folder}")
        print("Please ensure the folder path is correct")
        return

    print(f"Calculating monthly means from CSV files in: {input_folder}")
    df = calculate_monthly_means(input_folder)
    if df is None or df.empty:
        print("Unable to calculate monthly means from data files")
        return

    print(f"Successfully calculated monthly means, {len(df)} records total")
    print(f"Data time range: {df['Month'].min()} to {df['Month'].max()}")

    cities = df["City"].unique()
    pollutants = df["Pollutant"].unique()

    print(f"\nFound {len(cities)} cities: {list(cities)}")
    print(f"Found {len(pollutants)} pollutants: {list(pollutants)}")

    target_pollutant = "PM2.5"
    if target_pollutant not in pollutants:
        print(f"Error: {target_pollutant} data not found")
        return

    print(f"\nStarting MSTL decomposition analysis for {target_pollutant}...")
    print("\nChecking data availability for each city:")

    for city in cities:
        city_pollutant_df = df[(df["City"] == city) & (df["Pollutant"] == target_pollutant)]
        count = len(city_pollutant_df)
        if count > 0:
            months = sorted(city_pollutant_df["Month"].unique())
            print(f"  {city}: {count} months ({months[0]} to {months[-1]})")
        else:
            print(f"  {city}: No data available")

    all_analyses = []

    for city in tqdm(cities, desc=f"Analyzing {target_pollutant} data", unit="city"):
        try:
            ts_data = prepare_time_series_data(df, city, target_pollutant)
            if ts_data is None:
                print(f"  {city}: No time series data available")
                continue

            min_len = min(seasonal_periods) * 2
            if len(ts_data) < min_len:
                print(
                    f"  {city}: Insufficient data, skipping "
                    f"(only {len(ts_data)} points, need at least {min_len})"
                )
                continue

            print(
                f"  {city}: Processing {len(ts_data)} months of data "
                f"({ts_data.index.min()} to {ts_data.index.max()})"
            )

            if ts_data.isna().any():
                print(f"  {city}: Warning - data contains NaN values, filling with forward/backward fill")
                ts_data = ts_data.ffill().bfill()

            if ts_data.nunique() <= 1:
                print(f"  {city}: Skipping - data is constant")
                continue

            mstl_result = perform_mstl_decomposition(ts_data, seasonal_periods=seasonal_periods)
            if mstl_result is None:
                print(f"  {city}: MSTL decomposition failed")
                continue

            analysis = analyze_mstl_results(mstl_result, city, target_pollutant)
            all_analyses.append(analysis)

            city_label = city_display_name(city)
            city_output_dir = os.path.join(output_dir, city_label)
            os.makedirs(city_output_dir, exist_ok=True)

            mstl_plot_path = os.path.join(
                city_output_dir, f"{city_label}_{target_pollutant}_MSTL分解.png"
            )
            plot_mstl_components(mstl_result, city, target_pollutant, mstl_plot_path)

            analysis_plot_path = os.path.join(
                city_output_dir, f"{city_label}_{target_pollutant}_季节分析.png"
            )
            plot_seasonal_analysis(mstl_result, city, target_pollutant, analysis_plot_path)

            print(f"    Analysis charts saved to: {city_output_dir}")

        except Exception as exc:
            print(f"  {city} analysis failed: {exc}")
            continue

    save_analysis_results(all_analyses, output_dir)

    print(f"\n{'=' * 60}")
    print("MSTL Decomposition Analysis Summary")
    print(f"{'=' * 60}")

    if all_analyses:
        df_summary = pd.DataFrame(all_analyses)
        print(f"\nSuccessfully analyzed {len(all_analyses)} cities for {target_pollutant} data")

        print("\nCity Trend Analysis:")
        for _, row in df_summary.iterrows():
            print(
                f"  {row['city']}: "
                f"Trend {row['trend_direction']} "
                f"(Strength: {row['trend_strength']:.3f}) "
                f"Seasonal Peak Month: {row['seasonal_peak_month']} "
                f"Valley Month: {row['seasonal_valley_month']}"
            )

        print("\nTrend Strength Statistics:")
        print(f"  Average: {df_summary['trend_strength'].mean():.3f}")
        print(
            f"  Maximum: {df_summary['trend_strength'].max():.3f} "
            f"({df_summary.loc[df_summary['trend_strength'].idxmax(), 'city']})"
        )
        print(
            f"  Minimum: {df_summary['trend_strength'].min():.3f} "
            f"({df_summary.loc[df_summary['trend_strength'].idxmin(), 'city']})"
        )

        print("\nSeasonal Strength Statistics:")
        print(f"  Average: {df_summary['seasonal_strength'].mean():.3f}")
        print(
            f"  Maximum: {df_summary['seasonal_strength'].max():.3f} "
            f"({df_summary.loc[df_summary['seasonal_strength'].idxmax(), 'city']})"
        )
        print(
            f"  Minimum: {df_summary['seasonal_strength'].min():.3f} "
            f"({df_summary.loc[df_summary['seasonal_strength'].idxmin(), 'city']})"
        )
    else:
        print("No successful analysis results")

    print(f"\nAnalysis results saved to directory: {output_dir}")
    print("MSTL Decomposition Analysis Complete!")


if __name__ == "__main__":
    main()
