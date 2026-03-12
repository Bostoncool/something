# -*- coding: utf-8 -*-
"""
根据 CSV 中的区站号筛选 china_isd_lite_2018 文件夹：
只保留 CSV 中出现的区站号对应的文件，其余删除。
"""
import pandas as pd
from pathlib import Path

CSV_PATH = r"c:\Users\IU\Desktop\城市,区站号,站名.csv"
FOLDER_PATHS = [
    fr"c:\Users\IU\Desktop\china_isd_lite_{year}"
    for year in range(2018, 2024)
]


def main():
    # 读取 CSV，获取所有区站号（5 位）
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    valid_station_ids = set(df["区站号"].astype(str).str.zfill(5))
    print(f"CSV valid station IDs: {len(valid_station_ids)}")

    for folder_path in FOLDER_PATHS:
        folder = Path(folder_path)
        if not folder.is_dir():
            print(f"Folder not found: {folder_path}")
            continue

        to_delete = []
        to_keep = []

        for f in folder.iterdir():
            if not f.is_file():
                continue
            # filename format: XXXXXX-99999-2018, use first 5 chars of first segment as station id
            name = f.name
            parts = name.split("-")
            if len(parts) < 1:
                continue
            prefix = parts[0]
            station_id_5 = prefix[:5] if len(prefix) >= 5 else prefix.zfill(5)
            if station_id_5 in valid_station_ids:
                to_keep.append(name)
            else:
                to_delete.append(f)

        print(f"[{folder.name}] Total files: {len(to_keep) + len(to_delete)}, keep: {len(to_keep)}, delete: {len(to_delete)}")

        for f in to_delete:
            f.unlink()
            print(f"Deleted: {f.name}")

    print("Done.")


if __name__ == "__main__":
    main()
