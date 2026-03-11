"""
读取 2018-2023 年夜间灯光数据（筛选.xlsx）
"""
import sys

import pandas as pd

# 解决 Windows 控制台中文输出编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 夜间灯光数据文件路径
NIGHTLIGHT_PATH = r"F:\1.模型要用的\2018-2023[全国省、市、县域夜间灯光数据]\要用的_2018-2023筛选.xlsx"


def load_nightlight_data(sheet_name: int | str = 0) -> pd.DataFrame:
    """
    读取夜间灯光 Excel 数据。

    Parameters
    ----------
    sheet_name : int or str, default 0
        工作表名称或索引，0 表示第一个工作表。

    Returns
    -------
    pd.DataFrame
        夜间灯光数据
    """
    df = pd.read_excel(NIGHTLIGHT_PATH, sheet_name=sheet_name, engine="openpyxl")
    return df


if __name__ == "__main__":
    df = load_nightlight_data()

    print("夜间灯光数据预览:")
    print(df.head(10))
    print(f"\n数据形状: {df.shape}")
    print(f"\n列名: {list(df.columns)}")
    print(f"\n数据类型:\n{df.dtypes}")
