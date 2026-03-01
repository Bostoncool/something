"""
读取 2018-2023 年全国县路网密度数据（筛选.xlsx）
"""
import sys

import pandas as pd

# 解决 Windows 控制台中文输出编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 路网密度数据文件路径
ROAD_PATH = r"F:\1.模型要用的\2018-2023[全国县路网密度]\2013-2023[全国县路网密度]_2018-2023筛选.xlsx"


def load_road_density_data(sheet_name: int | str = 0) -> pd.DataFrame:
    """
    读取全国县路网密度 Excel 数据。

    Parameters
    ----------
    sheet_name : int or str, default 0
        工作表名称或索引，0 表示第一个工作表。

    Returns
    -------
    pd.DataFrame
        路网密度数据，包含列：市名、year、县.面积、路网总长度、省名、市代码、省代码、路网密度
    """
    df = pd.read_excel(ROAD_PATH, sheet_name=sheet_name, engine="openpyxl")
    return df


if __name__ == "__main__":
    df = load_road_density_data()

    print("路网密度数据预览:")
    print(df.head(10))
    print(f"\n数据形状: {df.shape}")
    print(f"\n列名: {list(df.columns)}")
    print(f"\n年份范围: {df['year'].min()} - {df['year'].max()}")
    print(f"\n数据类型:\n{df.dtypes}")
