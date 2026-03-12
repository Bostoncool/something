"""
城市数据筛选脚本
功能：
1. 遍历「城市_20230101-20231231」文件夹中的 CSV 文件
2. 仅保留指定城市的数据列
3. 将筛选后的数据合并输出
4. 统计并罗列表格中无数据的城市
5. 统计并罗列缺失数据的年份和具体日期
"""

import re
import sys
import io
from datetime import date, timedelta

# 解决 Windows 控制台中文输出编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
from pathlib import Path

# 输入输出路径
# 2018-2023 六个年份/文件夹
INPUT_DIRS = [Path(fr"c:\Users\IU\Desktop\城市_{year}0101-{year}1231") for year in range(2018, 2024)]
OUTPUT_DIRS = [Path(fr"c:\Users\IU\Desktop\城市_{year}0101-{year}1231_筛选后") for year in range(2018, 2024)]
OUTPUT_FILES = [
    output_dir / f"china_cities_{year}_筛选.csv"
    for year, output_dir in zip(range(2018, 2024), OUTPUT_DIRS)
]

# 目标城市名单（京津冀、长三角、珠三角、粤港澳大湾区）
CITY_NAMES = [
    "石家庄", "唐山", "秦皇岛", "邯郸", "邢台", "保定", "张家口", "承德", "沧州", "廊坊", "衡水",
    "南京", "无锡", "南通", "盐城", "扬州", "镇江", "常州", "苏州", "泰州",
    "杭州", "宁波", "嘉兴", "湖州", "绍兴", "金华", "舟山", "台州", "温州",
    "合肥", "芜湖", "马鞍山", "铜陵", "安庆", "滁州", "池州", "宣城",
    "广州", "深圳", "佛山", "东莞", "中山", "惠州", "珠海", "江门", "肇庆",
    "北京", "天津", "上海",
]


def get_expected_dates(year: int) -> set[str]:
    """获取某年应有的全部日期，格式 YYYYMMDD"""
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    days = (end - start).days + 1
    return {(start + timedelta(days=i)).strftime("%Y%m%d") for i in range(days)}


def parse_date_from_filename(fp: Path) -> str | None:
    """从文件名 china_cities_YYYYMMDD.csv 解析出日期"""
    m = re.search(r"(\d{8})\.csv$", fp.name)
    return m.group(1) if m else None


def get_csv_files(input_dir: Path) -> tuple[list[Path], dict[str, Path]]:
    """
    获取目录下所有 CSV 文件。
    返回: (文件列表, 日期->文件路径 映射)
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    files = sorted(input_dir.glob("*.csv"))
    date_to_file = {}
    for fp in files:
        d = parse_date_from_filename(fp)
        if d:
            date_to_file[d] = fp
    return files, date_to_file


def extract_city_columns(df: pd.DataFrame, city_names: list[str]) -> tuple[pd.DataFrame, set[str]]:
    """
    从 DataFrame 中提取指定城市列。
    返回: (筛选后的 DataFrame, 未找到的城市集合)
    """
    all_columns = set(df.columns)
    # 固定列：date, hour, type
    fixed_cols = ["date", "hour", "type"]
    # 找出数据中实际存在的目标城市列
    found_cities = []
    missing_cities = set()
    for city in city_names:
        if city in all_columns:
            found_cities.append(city)
        else:
            missing_cities.add(city)
    cols_to_keep = [c for c in fixed_cols if c in df.columns] + found_cities
    return df[cols_to_keep].copy(), missing_cities


def main():
    for year, input_dir, output_dir, output_file in zip(
        range(2018, 2024), INPUT_DIRS, OUTPUT_DIRS, OUTPUT_FILES
    ):
        if not input_dir.exists():
            print(f"{year} 年: 输入目录不存在 {input_dir}，跳过。")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        csv_files, date_to_file = get_csv_files(input_dir)
        if not csv_files:
            print(f"{year} 年: 未找到任何 CSV 文件，跳过。")
            continue

        # 1. 统计缺失的日期文件（应有但未找到的日期）
        expected = get_expected_dates(year)
        actual_dates = set(date_to_file.keys())
        missing_dates = sorted(expected - actual_dates)

        print(f"{year} 年: 找到 {len(csv_files)} 个 CSV 文件")

        all_missing_cities = set()  # 完全无列的城市
        city_no_data_dates: dict[str, list[str]] = {}  # 城市 -> 无数据的日期列表
        dfs = []

        for i, fp in enumerate(csv_files):
            file_date = parse_date_from_filename(fp)
            try:
                df = pd.read_csv(fp, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(fp, encoding="gbk")

            filtered_df, missing = extract_city_columns(df, CITY_NAMES)
            all_missing_cities.update(missing)

            # 2. 检查每个城市列是否全天无有效数据（全 NaN 或空）
            for city in filtered_df.columns:
                if city in ("date", "hour", "type"):
                    continue
                col = filtered_df[city]
                # 全为空：NaN 或空字符串
                is_empty = (col.isna() | (col.astype(str).str.strip().isin(["", "nan"]))).all()
                if is_empty:
                    if city not in city_no_data_dates:
                        city_no_data_dates[city] = []
                    if file_date:
                        city_no_data_dates[city].append(file_date)

            dfs.append(filtered_df)

            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1}/{len(csv_files)} 个文件")

        # 合并所有数据
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"  筛选完成: 共 {len(merged)} 行")

        # 3. 输出缺失报告
        print("\n  ---- 缺失数据报告 ----")

        if missing_dates:
            print(f"  【缺失日期文件】{year} 年以下 {len(missing_dates)} 天无对应 CSV 文件:")
            for d in missing_dates:
                print(f"    - {d[:4]}-{d[4:6]}-{d[6:8]}")
        else:
            print(f"  【缺失日期文件】{year} 年无缺失，所有日期均有文件。")

        if all_missing_cities:
            print(f"  【无数据列城市】以下城市在表格中完全无对应列: {sorted(all_missing_cities)}")

        if city_no_data_dates:
            print(f"  【某日无数据】以下城市在部分日期无有效数据:")
            for city in sorted(city_no_data_dates.keys()):
                dates = sorted(city_no_data_dates[city])
                print(f"    - {city}: {len(dates)} 天无数据")
                for d in dates[:10]:  # 最多展示 10 个
                    print(f"        {d[:4]}-{d[4:6]}-{d[6:8]}")
                if len(dates) > 10:
                    print(f"        ... 等共 {len(dates)} 天")
        elif not all_missing_cities:
            print("  【某日无数据】所有有列的城市在各日期均有有效数据。")

        print(f"\n  输出: {output_file}\n")


if __name__ == "__main__":
    main()
