"""
根据 CSV 文件中的文件地址列表，将提及的每个数据文件复制到指定目标文件夹。
原文件保持不变。
"""

import os
import shutil
import pandas as pd

# 配置路径
CSV_PATH = r"C:\Users\IU\Desktop\大气相关数据文件筛选结果_2018-2024.csv"
DEST_FOLDER = r"C:\Users\IU\Desktop\New folder"


def copy_files_from_list(csv_path: str, dest_folder: str) -> dict:
    """
    读取 CSV 中的完整路径列，将每个文件复制到目标文件夹。

    Parameters
    ----------
    csv_path : str
        记录文件地址的 CSV 文件路径
    dest_folder : str
        目标文件夹路径，复制后的文件将存放于此

    Returns
    -------
    dict
        包含 success_count, fail_count, failed_paths 的统计信息
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    full_path_col = "完整路径"

    if full_path_col not in df.columns:
        raise ValueError(f"CSV 中未找到列 '{full_path_col}'，可用列: {list(df.columns)}")

    paths = df[full_path_col].dropna().astype(str).str.strip()
    paths = paths[paths.str.len() > 0].unique()

    os.makedirs(dest_folder, exist_ok=True)

    success_count = 0
    failed_paths = []
    used_names = set()

    for src_path in paths:
        if not os.path.isfile(src_path):
            failed_paths.append((src_path, "文件不存在"))
            continue

        base_name = os.path.basename(src_path)
        dest_path = os.path.join(dest_folder, base_name)

        # 处理目标文件夹中重名文件
        if dest_path in used_names or os.path.exists(dest_path):
            stem, ext = os.path.splitext(base_name)
            counter = 1
            while True:
                new_name = f"{stem}_{counter}{ext}"
                dest_path = os.path.join(dest_folder, new_name)
                if not os.path.exists(dest_path):
                    break
                counter += 1

        try:
            shutil.copy2(src_path, dest_path)
            used_names.add(dest_path)
            success_count += 1
            print(f"已复制: {base_name}")
        except Exception as e:
            failed_paths.append((src_path, str(e)))
            print(f"复制失败: {src_path} - {e}")

    return {
        "success_count": success_count,
        "fail_count": len(failed_paths),
        "failed_paths": failed_paths,
    }


if __name__ == "__main__":
    print(f"读取地址列表: {CSV_PATH}")
    print(f"目标文件夹: {DEST_FOLDER}")
    print("-" * 50)

    result = copy_files_from_list(CSV_PATH, DEST_FOLDER)

    print("-" * 50)
    print(f"复制完成: 成功 {result['success_count']} 个, 失败 {result['fail_count']} 个")
    if result["failed_paths"]:
        print("\n失败列表:")
        for path, reason in result["failed_paths"]:
            print(f"  - {path}: {reason}")
