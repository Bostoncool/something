# 待做：月均、年均、趋势线、同比/环比、13城之间相关性、重污染天数统计(AQI>150/200)

# -*- coding: utf-8 -*-
"""
多进程批量读取 daily csv（文件名=日期），抽取指定城市列，输出：
1) 每个文件对应的过滤后CSV（避免内存爆）
2) 全量汇总统计CSV（按 date + type 聚合）

依赖：pandas, tqdm
可选：pyarrow（如果你想输出parquet）
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found. Installing tqdm for progress bars...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm


# =========================
# 1) 配置区：改这里就行
# =========================

# 你的数据根目录（包含 “城市_20180101-20181231” 这类文件夹）
INPUT_ROOT = Path(r"C:\Users\IU\Desktop\Datebase Origin\China AQI")   # TODO: 改成你的真实路径

# 输出目录
OUTPUT_ROOT = Path(r"C:\Users\IU\Desktop\Datebase Origin\RZD")  # TODO: 改成你的真实路径

# 需要抽取的城市
CITIES = [
    "广州", "深圳", "佛山", "东莞", "中山",
    "惠州", "珠海", "江门", "肇庆"
]

# CSV 里固定字段（你的样例是 date/hour/type）
BASE_COLS = ["date", "hour", "type"]

# 文件名中的日期（如 china_cities_20230101.csv / 20230101.csv 等）
DATE_RE = re.compile(r"(20\d{6})")


# =========================
# 2) 工具函数
# =========================

def list_csv_files(root: Path) -> List[Path]:
    """递归列出所有 csv 文件"""
    return sorted([p for p in root.rglob("*.csv") if p.is_file()])


def safe_read_filtered_csv(csv_path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """
    读取单个CSV并只保留需要列。
    返回：(df 或 None, 错误信息/空字符串)
    """
    try:
        # 先读表头，确定哪些列存在（避免 usecols 因缺列报错）
        header = pd.read_csv(csv_path, nrows=0)
        available_cols = set(header.columns)

        wanted = [c for c in (BASE_COLS + CITIES) if c in available_cols]
        missing = [c for c in (BASE_COLS + CITIES) if c not in available_cols]

        if not set(BASE_COLS).issubset(available_cols):
            # 如果 date/hour/type 都不全，仍可尝试从文件名补 date
            pass

        df = pd.read_csv(csv_path, usecols=wanted)

        # 若没有 date 列，尝试从文件名里解析
        if "date" not in df.columns:
            m = DATE_RE.search(csv_path.name)
            if m:
                df["date"] = int(m.group(1))
            else:
                df["date"] = pd.NA

        # 补齐缺失的城市列（方便后面 concat/聚合）
        for c in CITIES:
            if c not in df.columns:
                df[c] = pd.NA

        # 确保列顺序统一
        cols_order = [c for c in BASE_COLS if c in df.columns] + CITIES
        # 若 hour/type 不存在也没关系：只要 date + 城市即可分析
        df = df[cols_order]

        # 额外加一个来源文件名，方便追溯
        df["__file__"] = csv_path.name
        if missing:
            df["__missing_cols__"] = ",".join(missing)
        else:
            df["__missing_cols__"] = ""

        return df, ""

    except Exception as e:
        return None, f"{csv_path}: {repr(e)}"


def process_one_file(args: Tuple[Path, Path]) -> Tuple[int, str]:
    """
    worker：读一个文件 -> 输出过滤文件
    返回：(是否成功1/0, 错误信息)
    """
    csv_path, out_dir = args
    df, err = safe_read_filtered_csv(csv_path)
    if df is None:
        return 0, err

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / csv_path.name
    # 输出过滤后的 CSV（utf-8-sig 兼容 Excel）
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return 1, ""


# =========================
# 3) 主流程
# =========================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    filtered_dir = OUTPUT_ROOT / "filtered_daily"
    logs_dir = OUTPUT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list_csv_files(INPUT_ROOT)
    if not csv_files:
        raise SystemExit(f"没找到任何CSV：{INPUT_ROOT}")

    print(f"Found {len(csv_files)} CSV files, starting multiprocess extraction of {len(CITIES)} city columns...")

    # 每个进程输出到同一个 filtered_dir（文件名不同，不会冲突）
    tasks = [(p, filtered_dir) for p in csv_files]

    n_proc = max(1, min(cpu_count(), 8))  # 进程数你也可以改大/改小
    ok = 0
    errors: List[str] = []

    print(f"Processing {len(tasks)} files with {n_proc} processes...")
    with Pool(processes=n_proc) as pool:
        with tqdm(total=len(tasks), desc="Extracting city data", unit="file") as pbar:
            for success, err in pool.imap_unordered(process_one_file, tasks, chunksize=20):
                ok += success
                if err:
                    errors.append(err)
                pbar.update(1)

    print(f"Extraction completed: {ok} / {len(csv_files)} successful")

    if errors:
        err_path = logs_dir / "errors.txt"
        err_path.write_text("\n".join(errors), encoding="utf-8")
        print(f"{len(errors)} files failed, errors written to: {err_path}")

    # =========================
    # 4) 生成汇总（可选，但建议）
    # =========================
    print("Starting summary generation (aggregated by date + type, ignoring hour)...")

    filtered_files = sorted(filtered_dir.glob("*.csv"))
    if not filtered_files:
        raise SystemExit("filtered_daily directory is empty, cannot summarize.")


    # 逐文件读入并聚合，减少内存压力
    agg_list = []
    print(f"Aggregating data from {len(filtered_files)} filtered files...")

    with tqdm(total=len(filtered_files), desc="Aggregating data", unit="file") as pbar:
        for fp in filtered_files:
            df = pd.read_csv(fp)

            # 若没有 type 列（极少数情况），用 NA 占位
            if "type" not in df.columns:
                df["type"] = "NA"

            # 确保城市列为数值类型（将非数值转换为 NaN）
            for city in CITIES:
                if city in df.columns:
                    df[city] = pd.to_numeric(df[city], errors='coerce')

            # 聚合：按 date + type，对每个城市计算均值/最大/最小（你可按需改）
            g = df.groupby(["date", "type"], as_index=False)[CITIES].agg(["mean", "max", "min"])
            # 变平列名：北京_mean 之类
            g.columns = ["_".join([c for c in col if c]).strip("_") for col in g.columns.to_flat_index()]
            agg_list.append(g)
            pbar.update(1)

    print("Combining aggregated results...")
    agg_all = pd.concat(agg_list, ignore_index=True)
    print(f"Combined {len(agg_list)} aggregations into dataframe with {len(agg_all)} rows")

    # 同一天同type可能来自多个文件（正常不会，但保险起见再聚一次）
    print("Ensuring all columns are numeric...")
    for col in agg_all.columns:
        if col not in ["date", "type"]:
            agg_all[col] = pd.to_numeric(agg_all[col], errors='coerce')

    # 对 mean/max/min 再聚合的方式：
    # - mean：再取均值
    # - max：再取最大
    # - min：再取最小
    # 这里简单按列名后缀处理
    def col_reduce(colname: str) -> str:
        if colname.endswith("_mean"):
            return "mean"
        if colname.endswith("_max"):
            return "max"
        if colname.endswith("_min"):
            return "min"
        return "mean"

    metric_cols = [c for c in agg_all.columns if c not in ["date", "type"]]
    reducers = {}
    for c in metric_cols:
        reducers[c] = col_reduce(c)

    print(f"Performing final aggregation on {len(agg_all)} rows...")
    agg_final = agg_all.groupby(["date", "type"], as_index=False).agg(reducers)
    print(f"Final result: {len(agg_final)} aggregated records")

    out_summary = OUTPUT_ROOT / "summary_by_date_type.csv"
    agg_final.to_csv(out_summary, index=False, encoding="utf-8-sig")
    print(f"Summary output to: {out_summary}")
    print("All completed. You can now do further analysis based on filtered_daily or summary_by_date_type.csv.")


if __name__ == "__main__":
    main()
