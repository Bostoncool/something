"""
文件夹文件名统计脚本
将指定文件夹中所有文件名导出为 CSV 文件
"""

import os
import csv
from pathlib import Path


def export_filenames_to_csv(folder_path: str, output_csv: str = "filenames.csv") -> None:
    """
    将文件夹中所有文件名导出到 CSV 文件。

    Parameters
    ----------
    folder_path : str
        目标文件夹路径
    output_csv : str, optional
        输出 CSV 文件名，默认为 filenames.csv
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
    if not folder.is_dir():
        raise NotADirectoryError(f"路径不是文件夹: {folder_path}")

    filenames = sorted([f.name for f in folder.iterdir()])

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["序号", "文件名"])
        writer.writerows(enumerate(filenames, start=1))

    print(f"共导出 {len(filenames)} 个文件名到 {output_csv}")


if __name__ == "__main__":
    folder_path = r"F:\1.模型要用的\2018-2023[工业用地]\2018"
    output_file = "filenames_2018.csv"

    export_filenames_to_csv(folder_path, output_file)
