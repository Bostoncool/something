"""
遍历指定文件夹内所有文件和子文件夹，将名称、路径、后缀等信息输出为CSV文件到桌面。
"""

import os
from pathlib import Path

import pandas as pd


def traverse_and_export(
    root_dir: str,
    output_path: str = r"C:\Users\IU\Desktop\file_list.csv",
) -> None:
    """
    遍历文件夹内所有文件和子文件夹，输出到CSV。

    Parameters
    ----------
    root_dir : str
        要遍历的根目录路径
    output_path : str
        输出CSV文件的保存路径，默认为桌面
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root_dir}")

    records = []
    for item in root.rglob("*"):
        rel_path = item.relative_to(root)
        is_dir = item.is_dir()
        name = item.name
        suffix = item.suffix if item.is_file() else ""

        records.append(
            {
                "类型": "文件夹" if is_dir else "文件",
                "名称": name,
                "完整路径": str(item.resolve()),
                "相对路径": str(rel_path),
                "后缀": suffix,
            }
        )

    # 根目录本身也加入
    records.insert(
        0,
        {
            "类型": "文件夹",
            "名称": root.name,
            "完整路径": str(root.resolve()),
            "相对路径": ".",
            "后缀": "",
        },
    )

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已导出 {len(records)} 条记录到: {output_path}")


if __name__ == "__main__":
    # 要遍历的文件夹路径
    target_dir = r"F:\1.模型要用的\2018-2024[中国环境统计年鉴]"

    # 输出CSV路径（桌面）
    output_file = r"C:\Users\IU\Desktop\file_list.csv"

    traverse_and_export(target_dir, output_file)
