"""
读取能源消耗数据（2018-2023 筛选后）
数据来源：要用的_筛选后.xlsx
包含：全社会用电量、人工煤气和天然气供气总量、液化石油气供气总量
"""
import sys

import pandas as pd

# 解决 Windows 控制台中文输出编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 数据文件路径
from pathlib import Path
_DATA_ROOT = Path(__file__).resolve().parent.parent / "1.模型要用的"
ENERGY_PATH = _DATA_ROOT / "2018-2023[能源消耗]" / "要用的_筛选后.xlsx"

# 列名（便于引用）
COL_ID = "ID"
COL_CITY = "城市"
COL_YEAR = "年度"
COL_ELECTRICITY = "插值_全社会用电量万千瓦时"
COL_GAS = "插值_人工煤气和天然气供气总量万立方米市辖区"
COL_LPG = "插值_液化石油气供气总量吨市辖区"


def load_energy_data(path: str | None = None) -> pd.DataFrame:
    """
    读取能源消耗 Excel 数据

    Parameters
    ----------
    path : str, optional
        自定义文件路径，默认使用 ENERGY_PATH

    Returns
    -------
    pd.DataFrame
        能源消耗数据，列包括：ID、城市、年度、用电量、天然气供气量、液化石油气供气量
    """
    file_path = path or ENERGY_PATH
    df = pd.read_excel(file_path)
    return df


def load_energy_by_cities(cities: list[str], path: str | None = None) -> pd.DataFrame:
    """
    读取能源消耗数据并筛选指定城市

    Parameters
    ----------
    cities : list[str]
        目标城市列表（支持带"市"或不带"市"）
    path : str, optional
        自定义文件路径

    Returns
    -------
    pd.DataFrame
        筛选后的能源消耗数据
    """
    df = load_energy_data(path)
    city_set = set()
    for c in cities:
        city_set.add(c)
        city_set.add(c + "市")
    return df.loc[df[COL_CITY].isin(city_set)].copy()


if __name__ == "__main__":
    df = load_energy_data()

    print("能源消耗数据预览:")
    print(df.head(10))
    print(f"\n数据形状: {df.shape}")
    print(f"\n列名: {df.columns.tolist()}")
    print(f"\n年度范围: {df[COL_YEAR].min()} - {df[COL_YEAR].max()}")
    print(f"\n城市数量: {df[COL_CITY].nunique()}")
