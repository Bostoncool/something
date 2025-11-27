#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多进程并行：nc → csv（摄氏度）
用法：直接 python batch_nc2csv_parallel.py
"""
import pathlib, multiprocessing as mp, xarray as xr, pandas as pd

# ---------- 路径写死 ----------
IN_DIR  = pathlib.Path(r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-NC\2m_dewpoint_temperature")
OUT_DIR = pathlib.Path(r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV\2m_dewpoint_temperature")
N_WORKERS = mp.cpu_count()          # 想用多少核就改这里
# ------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

def process_one(nc_file: pathlib.Path) -> str:
    """单文件转换，返回结果字符串"""
    csv_file = OUT_DIR / (nc_file.stem + ".csv")
    try:
        ds = xr.open_dataset(nc_file, decode_cf=True)
        da = ds["d2m"]

        # 展平 → DataFrame
        df = da.to_series().reset_index()
        df["time"] = df["valid_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df = df.drop(columns=["valid_time"])

        # K → °C
        df["d2m"] = df["d2m"].apply(
            lambda x: "" if pd.isna(x) else f"{x - 273.15:.3f}"
        )

        # expver
        exp_df = ds["expver"].to_series().reset_index()
        exp_df["time"] = pd.to_datetime(exp_df["valid_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        exp_df = exp_df.drop(columns=["valid_time"]).rename(columns={"expver": "expver"})
        df = df.merge(exp_df, on="time", how="left")

        df = df[["time", "latitude", "longitude", "expver", "d2m"]]
        df = df.sort_values(["time", "latitude", "longitude"])
        df.to_csv(csv_file, index=False, float_format="%.3f")
        return f"✔ {nc_file.name}"
    except Exception as e:
        return f"✘ {nc_file.name}  {e}"

def main():
    nc_list = list(IN_DIR.glob("*.nc"))
    if not nc_list:
        print(f"在 {IN_DIR.resolve()} 内未找到 .nc 文件")
        return

    print(f"共 {len(nc_list)} 个文件，启用 {N_WORKERS} 进程并行转换...")
    with mp.Pool(N_WORKERS) as pool:
        for msg in pool.imap_unordered(process_one, nc_list):
            print(msg)
    print("全部完成！")

if __name__ == "__main__":
    main()