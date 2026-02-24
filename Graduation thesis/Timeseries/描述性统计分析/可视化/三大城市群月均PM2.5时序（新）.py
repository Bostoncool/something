import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

CITY_TO_CLUSTER = {
    # 京津冀
    "北京": "京津冀城市群(BTH)",
    "天津": "京津冀城市群(BTH)",
    "石家庄": "京津冀城市群(BTH)",
    "唐山": "京津冀城市群(BTH)",
    "秦皇岛": "京津冀城市群(BTH)",
    "邯郸": "京津冀城市群(BTH)",
    "邢台": "京津冀城市群(BTH)",
    "保定": "京津冀城市群(BTH)",
    "张家口": "京津冀城市群(BTH)",
    "承德": "京津冀城市群(BTH)",
    "沧州": "京津冀城市群(BTH)",
    "廊坊": "京津冀城市群(BTH)",
    "衡水": "京津冀城市群(BTH)",
    # 长三角
    "上海": "长江三角洲城市群(YZD)",
    "南京": "长江三角洲城市群(YZD)",
    "无锡": "长江三角洲城市群(YZD)",
    "常州": "长江三角洲城市群(YZD)",
    "苏州": "长江三角洲城市群(YZD)",
    "南通": "长江三角洲城市群(YZD)",
    "盐城": "长江三角洲城市群(YZD)",
    "扬州": "长江三角洲城市群(YZD)",
    "镇江": "长江三角洲城市群(YZD)",
    "泰州": "长江三角洲城市群(YZD)",
    "杭州": "长江三角洲城市群(YZD)",
    "宁波": "长江三角洲城市群(YZD)",
    "温州": "长江三角洲城市群(YZD)",
    "嘉兴": "长江三角洲城市群(YZD)",
    "湖州": "长江三角洲城市群(YZD)",
    "绍兴": "长江三角洲城市群(YZD)",
    "金华": "长江三角洲城市群(YZD)",
    "舟山": "长江三角洲城市群(YZD)",
    "合肥": "长江三角洲城市群(YZD)",
    "芜湖": "长江三角洲城市群(YZD)",
    "马鞍山": "长江三角洲城市群(YZD)",
    "铜陵": "长江三角洲城市群(YZD)",
    "安庆": "长江三角洲城市群(YZD)",
    "滁州": "长江三角洲城市群(YZD)",
    "池州": "长江三角洲城市群(YZD)",
    "宣城": "长江三角洲城市群(YZD)",
    "台州": "长江三角洲城市群(YZD)",
    # 珠三角
    "广州": "珠江三角洲城市群(PRD)",
    "深圳": "珠江三角洲城市群(PRD)",
    "珠海": "珠江三角洲城市群(PRD)",
    "佛山": "珠江三角洲城市群(PRD)",
    "江门": "珠江三角洲城市群(PRD)",
    "东莞": "珠江三角洲城市群(PRD)",
    "中山": "珠江三角洲城市群(PRD)",
    "肇庆": "珠江三角洲城市群(PRD)",
    "惠州": "珠江三角洲城市群(PRD)",
}


def safe_print(message: str) -> None:
    """在 Windows 终端编码不支持中文时，降级输出。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


def find_column_by_alias(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """按候选别名（忽略大小写和空格）寻找列名。"""
    normalized = {str(col).strip().lower().replace(" ", ""): col for col in df.columns}
    for alias in aliases:
        key = alias.strip().lower().replace(" ", "")
        if key in normalized:
            return normalized[key]
    return None


def detect_pm25_column(df: pd.DataFrame) -> str:
    """识别 PM2.5 列，优先使用常见别名，再用关键字兜底。"""
    aliases = [
        "PM2.5",
        "pm2.5",
        "PM25",
        "pm25",
        "月均PM2.5",
        "月均PM2.5浓度",
        "PM2_5",
    ]
    pm_col = find_column_by_alias(df, aliases)
    if pm_col is not None:
        return pm_col

    for col in df.columns:
        lowered = str(col).strip().lower().replace(" ", "")
        if "pm2.5" in lowered or "pm25" in lowered:
            return col

    raise ValueError("无法识别 PM2.5 列，请检查数据表头。")


def parse_month_column(df: pd.DataFrame) -> pd.Series:
    """
    解析月份列并返回 datetime（月起始）。
    支持：
    1) 单列年月（如 2018-01 / 2018/01 / 201801）
    2) 年列 + 月列
    """
    month_aliases = ["年月", "月份", "日期", "month", "Month", "time", "时间", "统计时间"]
    month_col = find_column_by_alias(df, month_aliases)
    if month_col is not None:
        month_dt = pd.to_datetime(df[month_col], errors="coerce")
        if month_dt.notna().sum() > 0:
            return month_dt.dt.to_period("M").dt.to_timestamp()

    year_col = find_column_by_alias(df, ["年", "year", "Year"])
    mon_col = find_column_by_alias(df, ["月", "month", "Month"])
    if year_col is not None and mon_col is not None:
        year_num = pd.to_numeric(df[year_col], errors="coerce")
        mon_num = pd.to_numeric(df[mon_col], errors="coerce")
        month_dt = pd.to_datetime(
            dict(year=year_num, month=mon_num, day=1),
            errors="coerce",
        )
        return month_dt

    raise ValueError("无法识别月份信息列，请检查是否存在 年月/月份 或 年+月 列。")


def parse_month_from_columns(columns: list[str]) -> dict[str, pd.Timestamp]:
    """从宽表列名中筛选并标准化月份列（YYYYMM / YYYY-MM / YYYY/MM）。"""
    month_map: dict[str, pd.Timestamp] = {}
    for col in columns:
        text = str(col).strip()
        dt = pd.to_datetime(text, format="%Y%m", errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(text, format="%Y-%m", errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(text, format="%Y/%m", errors="coerce")
        if not pd.isna(dt):
            month_map[text] = pd.Timestamp(dt).to_period("M").to_timestamp()
    return month_map


def build_cluster_monthly_from_wide_city_table(df: pd.DataFrame, city_col: str) -> pd.DataFrame:
    """处理“城市在行、月份在列”的宽表，聚合得到城市群月均时序。"""
    month_map = parse_month_from_columns(df.columns.tolist())
    month_cols = list(month_map.keys())
    if not month_cols:
        raise ValueError("未识别到月份列（如 201801/2018-01）。")

    work_df = df[[city_col] + month_cols].copy()
    work_df[city_col] = work_df[city_col].astype(str).str.strip()
    work_df["cluster"] = work_df[city_col].map(CITY_TO_CLUSTER)

    unknown_city = sorted(work_df.loc[work_df["cluster"].isna(), city_col].unique().tolist())
    if unknown_city:
        raise ValueError(f"存在未映射城市，需补充城市群映射：{unknown_city}")

    long_df = work_df.melt(
        id_vars=["cluster"],
        value_vars=month_cols,
        var_name="month",
        value_name="pm25",
    )
    long_df["month"] = long_df["month"].map(month_map)
    long_df["pm25"] = pd.to_numeric(long_df["pm25"], errors="coerce")
    long_df = long_df.loc[long_df["month"].notna() & long_df["pm25"].notna()].copy()

    cluster_monthly = (
        long_df.groupby(["cluster", "month"], as_index=False)["pm25"]
        .mean()
        .sort_values(["cluster", "month"])
    )
    return cluster_monthly.pivot(index="month", columns="cluster", values="pm25").sort_index()


def build_cluster_monthly_series(csv_path: str) -> pd.DataFrame:
    """读取城市级月均 PM2.5 数据，聚合为城市群月均时序。"""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="gbk")

    df.columns = [str(col).strip() for col in df.columns]
    city_col = find_column_by_alias(df, ["城市", "city", "City"])
    if city_col is not None:
        return build_cluster_monthly_from_wide_city_table(df, city_col=city_col)

    cluster_col = find_column_by_alias(
        df,
        ["城市群", "城市群名称", "区域", "cluster", "Cluster", "城市群（市）", "城市群(市)"],
    )
    if cluster_col is None:
        raise ValueError("无法识别城市或城市群列，请检查数据表头。")

    pm_col = detect_pm25_column(df)
    month_dt = parse_month_column(df)

    long_df = pd.DataFrame(
        {
            "cluster": df[cluster_col].astype(str).str.strip(),
            "month": month_dt,
            "pm25": pd.to_numeric(df[pm_col], errors="coerce"),
        }
    )
    long_df = long_df.loc[
        long_df["cluster"].ne("")
        & long_df["month"].notna()
        & long_df["pm25"].notna()
    ].copy()
    if long_df.empty:
        raise ValueError("清洗后没有可用于绘图的数据，请检查原始数据内容。")

    cluster_monthly = (
        long_df.groupby(["cluster", "month"], as_index=False)["pm25"]
        .mean()
        .sort_values(["cluster", "month"])
    )
    return cluster_monthly.pivot(index="month", columns="cluster", values="pm25").sort_index()


def choose_color(cluster_name: str) -> str:
    """根据城市群名称匹配固定颜色，未匹配时返回默认色。"""
    name = cluster_name.replace(" ", "")
    if "京津冀" in name or "bth" in name.lower():
        return "#d62828"
    if "长江三角洲" in name or "长三角" in name or "yzd" in name.lower():
        return "#1d3557"
    if "珠江三角洲" in name or "珠三角" in name or "prd" in name.lower():
        return "#2a9d8f"
    return "#6c757d"


def plot_cluster_trends(cluster_monthly_df: pd.DataFrame, output_dir: Path) -> Path:
    """绘制三大城市群（或数据中存在的城市群）月均 PM2.5 时序变化图。"""
    fig, ax = plt.subplots(figsize=(12.0, 6.0), dpi=150)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    for col in cluster_monthly_df.columns:
        series = cluster_monthly_df[col].dropna()
        if series.empty:
            continue
        ax.plot(
            series.index,
            series.values,
            label=col,
            color=choose_color(str(col)),
            linewidth=2.0,
            marker="o",
            markersize=2.8,
            alpha=0.95,
        )

    ax.set_title("三大城市群月均PM2.5时序变化（2018-2023）", fontsize=14, pad=10)
    ax.set_xlabel("时间", fontsize=12)
    ax.set_ylabel("PM2.5/(μg/m³)", fontsize=12, fontfamily="Times New Roman")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", width=1.0, length=5, labelsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    if len(ax.lines) > 0:
        ax.legend(frameon=False, fontsize=10)

    output_path = output_dir / "三大城市群月均PM2.5时序变化.svg"
    fig.savefig(output_path, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    csv_path = r"H:\DATA Science\大论文Result\三大城市群（市）月均PM2.5浓度\合并数据_2018-2023.csv"

    mpl.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    mpl.rcParams["axes.unicode_minus"] = False

    output_dir = Path(r"H:\DATA Science\大论文Result\大论文图\三大城市群\PM2.5_月均_时序图")
    cluster_monthly_df = build_cluster_monthly_series(csv_path)
    figure_path = plot_cluster_trends(cluster_monthly_df, output_dir=output_dir)

    output_data_path = output_dir / "三大城市群_月均PM2.5_聚合结果.csv"
    cluster_monthly_df.to_csv(output_data_path, encoding="utf-8-sig")

    safe_print(f"图片已保存到: {figure_path}")
    safe_print(f"聚合数据已保存到: {output_data_path}")


if __name__ == "__main__":
    main()
