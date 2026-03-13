"""
读取 GDP 和人口数据，筛选指定城市列表
"""
import sys

import pandas as pd

# 解决 Windows 控制台中文输出编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 数据文件路径
from pathlib import Path
_DATA_ROOT = Path(__file__).resolve().parent.parent / "1.模型要用的"
POPULATION_PATH = _DATA_ROOT / "2018-2023[GDP+人口]" / "人口（万人）.csv"
GDP_PATH = _DATA_ROOT / "2018-2023[GDP+人口]" / "GDP（亿元）.csv"

# 目标城市列表
TARGET_CITIES = [
    "石家庄", "唐山", "秦皇岛", "邯郸", "邢台", "保定", "张家口", "承德", "沧州", "廊坊", "衡水",
    "南京", "无锡", "南通", "盐城", "扬州", "镇江", "常州", "苏州", "泰州",
    "杭州", "宁波", "嘉兴", "湖州", "绍兴", "金华", "舟山", "台州", "温州",
    "合肥", "芜湖", "马鞍山", "铜陵", "安庆", "滁州", "池州", "宣城",
    "广州", "深圳", "佛山", "东莞", "中山", "惠州", "珠海", "江门", "肇庆",
    "北京", "天津", "上海",
]


def load_and_filter_data():
    """读取两份数据并筛选指定城市"""
    # 读取人口数据
    df_pop = pd.read_csv(POPULATION_PATH, encoding="utf-8")
    df_pop_filtered = df_pop.loc[df_pop["城市"].isin(TARGET_CITIES)].copy()

    # 读取 GDP 数据
    df_gdp = pd.read_csv(GDP_PATH, encoding="utf-8")
    df_gdp_filtered = df_gdp.loc[df_gdp["城市"].isin(TARGET_CITIES)].copy()

    return df_pop_filtered, df_gdp_filtered


if __name__ == "__main__":
    df_population, df_gdp = load_and_filter_data()

    print("人口数据（万人）:")
    print(df_population)
    print(f"\n共 {len(df_population)} 个城市")

    print("\n" + "=" * 50)
    print("GDP 数据（亿元）:")
    print(df_gdp)
    print(f"\n共 {len(df_gdp)} 个城市")
 