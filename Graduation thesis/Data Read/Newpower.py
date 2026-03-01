"""
新能源汽车保有量数据批量读取与筛选
从各省市数据中提取指定城市的新能源汽车保有量
"""

import sys
import pandas as pd
from pathlib import Path

# 解决 Windows 控制台中文输出编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 数据文件夹路径
DATA_DIR = Path(r"F:\1.模型要用的\2017-2023[新能源汽车保有量数据(各省市)]")

# 目标城市列表（用户提供的城市名，数据中格式为"城市名+市"）
TARGET_CITIES = [
    "石家庄", "唐山", "秦皇岛", "邯郸", "邢台", "保定", "张家口", "承德", "沧州", "廊坊", "衡水",
    "南京", "无锡", "南通", "盐城", "扬州", "镇江", "常州", "苏州", "泰州",
    "杭州", "宁波", "嘉兴", "湖州", "绍兴", "金华", "舟山", "台州", "温州",
    "合肥", "芜湖", "马鞍山", "铜陵", "安庆", "滁州", "池州", "宣城",
    "广州", "深圳", "佛山", "东莞", "中山", "惠州", "珠海", "江门", "肇庆",
    "北京", "天津", "上海",
]

# 构建匹配用的城市名集合（数据中城市列格式为"XX市"）
TARGET_CITY_NAMES = {f"{city}市" for city in TARGET_CITIES}


def load_and_filter_newpower(data_dir: Path = None) -> pd.DataFrame:
    """
    批量读取CSV文件并筛选指定城市的新能源汽车保有量数据。

    Parameters
    ----------
    data_dir : Path, optional
        数据文件夹路径，默认为 DATA_DIR

    Returns
    -------
    pd.DataFrame
        合并后的筛选数据，列：年份、省份、城市、保有量
    """
    data_dir = data_dir or DATA_DIR

    # 匹配各省市新能源汽车保有量文件（兼容文件名中可能存在的空格）
    csv_files = sorted(data_dir.glob("*新能源汽车保有量*公安部口径*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"未在 {data_dir} 中找到匹配的CSV文件")

    dfs = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="gbk")

        # 筛选目标城市（城市列格式为"XX市"）
        mask = df["城市"].astype(str).isin(TARGET_CITY_NAMES)
        df_filtered = df.loc[mask].copy()
        dfs.append(df_filtered)

    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values(["年份", "省份", "城市"]).reset_index(drop=True)

    return result


def main():
    """主函数：加载数据并输出统计信息"""
    df = load_and_filter_newpower()
    print(f"共读取 {len(df)} 条记录")
    print(f"年份范围: {df['年份'].min()} - {df['年份'].max()}")
    print(f"涉及城市数: {df['城市'].nunique()}")
    print("\n前10行数据预览:")
    print(df.head(10))
    return df


if __name__ == "__main__":
    df = main()
