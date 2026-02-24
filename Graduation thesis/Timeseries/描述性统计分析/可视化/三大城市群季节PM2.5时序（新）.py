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

# 三大城市群季节划分（按气候带差异设定）
# - BTH: 典型温带四季分明
# - YZD: 夏季更长，秋季相对偏短
# - PRD: 亚热带季风，长夏、短冬
SEASON_MONTHS_BY_CLUSTER = {
    "京津冀城市群(BTH)": {
        "春季": [3, 4, 5],
        "夏季": [6, 7, 8],
        "秋季": [9, 10, 11],
        "冬季": [12, 1, 2],
    },
    "长江三角洲城市群(YZD)": {
        "春季": [3, 4, 5],
        "夏季": [6, 7, 8, 9],
        "秋季": [10, 11],
        "冬季": [12, 1, 2],
    },
    "珠江三角洲城市群(PRD)": {
        "春季": [3, 4, 5],
        "夏季": [6, 7, 8, 9, 10],
        "秋季": [11, 12],
        "冬季": [1, 2],
    },
}

SEASON_ORDER = ["春季", "夏季", "秋季", "冬季"]
SEASON_COLORS = {
    "春季": "#2a9d8f",
    "夏季": "#1d3557",
    "秋季": "#f4a261",
    "冬季": "#d62828",
}


def safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


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
    """处理“第一列城市、第一行时间轴(月份列)”的宽表。"""
    month_map = parse_month_from_columns(df.columns.tolist())
    month_cols = list(month_map.keys())
    if not month_cols:
        raise ValueError("未识别到月份列（示例格式：201801/2018-01/2018/01）。")

    work_df = df[[city_col] + month_cols].copy()
    work_df[city_col] = work_df[city_col].astype(str).str.strip()
    work_df["cluster"] = work_df[city_col].map(CITY_TO_CLUSTER)

    unknown_city = sorted(work_df.loc[work_df["cluster"].isna(), city_col].unique().tolist())
    if unknown_city:
        raise ValueError(f"存在未映射城市，请补充映射：{unknown_city}")

    long_df = work_df.melt(
        id_vars=["cluster"],
        value_vars=month_cols,
        var_name="month_col",
        value_name="pm25",
    )
    long_df["month"] = long_df["month_col"].map(month_map)
    long_df["pm25"] = pd.to_numeric(long_df["pm25"], errors="coerce")
    long_df = long_df.loc[long_df["month"].notna() & long_df["pm25"].notna(), ["cluster", "month", "pm25"]]

    cluster_monthly = (
        long_df.groupby(["cluster", "month"], as_index=False)["pm25"]
        .mean()
        .sort_values(["cluster", "month"])
    )
    return cluster_monthly


def read_city_wide_table(csv_path: str) -> pd.DataFrame:
    """读取宽表数据，优先 utf-8-sig，失败回退 gbk。"""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="gbk")

    df.columns = [str(col).strip() for col in df.columns]
    city_col = df.columns[0]
    if "城市" not in city_col and city_col.lower() not in ("city",):
        safe_print(f"提示：默认将第一列 `{city_col}` 作为城市列处理。")
    return df


def assign_season(cluster: str, month: int) -> str | None:
    season_map = SEASON_MONTHS_BY_CLUSTER.get(cluster, {})
    for season_name, months in season_map.items():
        if month in months:
            return season_name
    return None


def build_cluster_seasonal_yearly(cluster_monthly: pd.DataFrame) -> pd.DataFrame:
    """将月均序列聚合为“年-季节”均值（按城市群差异季节定义）。"""
    work_df = cluster_monthly.copy()
    work_df["year"] = work_df["month"].dt.year
    work_df["mon"] = work_df["month"].dt.month

    work_df["season"] = [
        assign_season(cluster, month)
        for cluster, month in zip(work_df["cluster"], work_df["mon"])
    ]
    work_df = work_df.loc[work_df["season"].notna()].copy()

    # 冬季跨年：12 月归到下一年冬季，以保持一个冬季的完整性。
    work_df["season_year"] = work_df["year"]
    is_winter_december = (work_df["season"] == "冬季") & (work_df["mon"] == 12)
    work_df.loc[is_winter_december, "season_year"] = work_df.loc[is_winter_december, "season_year"] + 1

    grouped = (
        work_df.groupby(["cluster", "season_year", "season"], as_index=False)
        .agg(
            pm25=("pm25", "mean"),
            month_count=("mon", "nunique"),
        )
        .sort_values(["cluster", "season_year"])
    )

    # 仅保留完整季节，避免边界年份（如 2024 冬季仅含 2023-12）被画出来。
    expected_months = {
        (cluster, season): len(months)
        for cluster, season_map in SEASON_MONTHS_BY_CLUSTER.items()
        for season, months in season_map.items()
    }
    grouped["expected_count"] = grouped.apply(
        lambda row: expected_months.get((row["cluster"], row["season"])),
        axis=1,
    )
    seasonal = grouped.loc[grouped["month_count"] == grouped["expected_count"], ["cluster", "season_year", "season", "pm25"]]
    return seasonal


def plot_cluster_seasonal_curves(seasonal_df: pd.DataFrame, output_dir: Path) -> Path:
    clusters = [
        "京津冀城市群(BTH)",
        "长江三角洲城市群(YZD)",
        "珠江三角洲城市群(PRD)",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), dpi=150, sharey=True)
    fig.patch.set_alpha(0.0)

    for ax, cluster in zip(axes, clusters):
        sub_df = seasonal_df.loc[seasonal_df["cluster"] == cluster].copy()
        if sub_df.empty:
            ax.set_title(f"{cluster}\n(无可用数据)", fontsize=11)
            ax.axis("off")
            continue

        for season in SEASON_ORDER:
            season_df = sub_df.loc[sub_df["season"] == season].sort_values("season_year")
            if season_df.empty:
                continue
            ax.plot(
                season_df["season_year"],
                season_df["pm25"],
                label=season,
                color=SEASON_COLORS[season],
                linewidth=2.0,
                marker="o",
                markersize=3.2,
                alpha=0.95,
            )

        ax.set_title(cluster, fontsize=12, pad=8)
        ax.set_xlabel("年份", fontsize=11)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", width=1.0, length=4, labelsize=9)
        if ax.lines:
            ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")

    axes[0].set_ylabel("PM2.5/(μg/m³)", fontsize=11, fontfamily="Times New Roman")
    fig.suptitle("三大城市群分季节 PM2.5 变化曲线（2018-2023）", fontsize=14, y=1.02)
    fig.tight_layout()

    output_path = output_dir / "三大城市群分季节PM2.5变化曲线.svg"
    fig.savefig(output_path, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    csv_path = r"H:\DATA Science\大论文Result\三大城市群（市）月均PM2.5浓度\合并数据_2018-2023.csv"

    mpl.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    mpl.rcParams["axes.unicode_minus"] = False

    output_dir = Path(r"H:\DATA Science\大论文Result\大论文图\三大城市群\PM2.5_季节_时序图")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = read_city_wide_table(csv_path)
    cluster_monthly = build_cluster_monthly_from_wide_city_table(raw_df, city_col=raw_df.columns[0])
    seasonal_df = build_cluster_seasonal_yearly(cluster_monthly)

    figure_path = plot_cluster_seasonal_curves(seasonal_df, output_dir=output_dir)
    seasonal_data_path = output_dir / "三大城市群_分季节PM2.5_年均结果.csv"
    seasonal_df.to_csv(seasonal_data_path, index=False, encoding="utf-8-sig")

    safe_print(f"图片已保存到: {figure_path}")
    safe_print(f"分季节数据已保存到: {seasonal_data_path}")


if __name__ == "__main__":
    main()
