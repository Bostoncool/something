"""
Excel 数据筛选脚本
功能：
1. 仅保留 2018-2023 年的数据
2. 仅保留指定城市的数据
"""

import pandas as pd

# 输入输出路径
INPUT_PATH = r"f:\1.模型要用的\要用的.xlsx"
OUTPUT_PATH = r"f:\1.模型要用的\要用的_筛选后.xlsx"

# 目标城市名单（用户提供的城市名，需加上"市"后缀以匹配数据）
CITY_NAMES = [
    "石家庄", "唐山", "秦皇岛", "邯郸", "邢台", "保定", "张家口", "承德", "沧州", "廊坊", "衡水",
    "南京", "无锡", "南通", "盐城", "扬州", "镇江", "常州", "苏州", "泰州",
    "杭州", "宁波", "嘉兴", "湖州", "绍兴", "金华", "舟山", "台州", "温州",
    "合肥", "芜湖", "马鞍山", "铜陵", "安庆", "滁州", "池州", "宣城",
    "广州", "深圳", "佛山", "东莞", "中山", "惠州", "珠海", "江门", "肇庆",
    "北京", "天津", "上海",
]

# 数据中城市列可能带"市"后缀，构建匹配集合（同时支持带市和不带市）
CITY_SET = set()
for city in CITY_NAMES:
    CITY_SET.add(city)
    CITY_SET.add(city + "市")


def main():
    df = pd.read_excel(INPUT_PATH)

    print(f"Original rows: {len(df)}")

    # 1. 筛选 2018-2023 年
    df = df.loc[(df["年度"] >= 2018) & (df["年度"] <= 2023)]
    print(f"After year filter (2018-2023): {len(df)}")

    # 2. 筛选指定城市（城市列可能为"北京市"或"北京"等形式）
    df = df.loc[df["城市"].isin(CITY_SET)]
    print(f"After city filter: {len(df)}")

    df.to_excel(OUTPUT_PATH, index=False)
    print("Done. Output saved.")


if __name__ == "__main__":
    main()
