"""
遍历 2018-2023 工业排放清单文件夹，多进程读取所有 CSV 数据。
数据格式：每份 CSV 前方为年度数据，后面按月份排列。
"""
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# 解决 Windows 控制台中文输出编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 数据根目录
_DATA_ROOT = Path(__file__).resolve().parent.parent / "1.模型要用的"
DATA_ROOT = _DATA_ROOT / "2018-2023[工业排放清单]"
YEARS = (2018, 2019, 2020, 2021, 2022, 2023)

# 尝试的编码顺序（常见中文 CSV 编码）
ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb2312")


def _read_single_csv(csv_path: Path) -> tuple[Path, pd.DataFrame | None, str | None]:
    """
    读取单个 CSV 文件，支持多种编码。

    Parameters
    ----------
    csv_path : Path
        CSV 文件路径

    Returns
    -------
    tuple[Path, pd.DataFrame | None, str | None]
        (文件路径, 数据框或None, 错误信息或None)
    """
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            return (csv_path, df, None)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return (csv_path, None, "无法使用常见编码解析")


def collect_csv_paths(root: Path) -> list[Path]:
    """收集根目录下各年度文件夹内所有 CSV 文件路径。"""
    paths = []
    for year in YEARS:
        year_dir = root / str(year)
        if not year_dir.is_dir():
            continue
        for csv_file in year_dir.glob("*.csv"):
            paths.append(csv_file)
    return paths


def load_all_industrial_emission(
    root: Path | str = DATA_ROOT,
    max_workers: int | None = None,
) -> dict[Path, pd.DataFrame]:
    """
    多进程遍历并读取 2018-2023 工业排放清单中所有 CSV 文件。

    Parameters
    ----------
    root : Path | str
        数据根目录，默认 DATA_ROOT
    max_workers : int | None
        最大进程数，None 表示使用 CPU 核心数 - 1

    Returns
    -------
    dict[Path, pd.DataFrame]
        键为 CSV 路径，值为对应的 DataFrame
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")

    csv_paths = collect_csv_paths(root)
    if not csv_paths:
        return {}

    # 默认进程数：CPU 核心数 - 1，至少 1
    if max_workers is None:
        import os
        max_workers = max(1, os.cpu_count() or 4 - 1)

    result: dict[Path, pd.DataFrame] = {}
    errors: list[tuple[Path, str]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_read_single_csv, p): p for p in csv_paths}
        for future in as_completed(futures):
            path, df, err = future.result()
            if df is not None:
                result[path] = df
            else:
                errors.append((path, err or "未知错误"))

    if errors:
        print(f"读取失败 {len(errors)} 个文件:")
        for p, e in errors[:10]:
            print(f"  - {p.name}: {e}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个")

    return result


def load_all_as_dataframe(
    root: Path | str = DATA_ROOT,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """
    多进程读取所有 CSV 并合并为单个 DataFrame（按路径添加来源列）。

    Returns
    -------
    pd.DataFrame
        合并后的数据，含 'source_file' 和 'year' 列标识来源
    """
    data = load_all_industrial_emission(root, max_workers)
    if not data:
        return pd.DataFrame()

    dfs = []
    for path, df in data.items():
        df = df.copy()
        df["source_file"] = path.name
        df["year"] = int(path.parent.name)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    print("正在收集 CSV 文件路径...")
    root = DATA_ROOT
    paths = collect_csv_paths(root)
    print(f"共发现 {len(paths)} 个 CSV 文件")

    print("\n多进程读取中...")
    data = load_all_industrial_emission(root)

    print(f"\n成功读取 {len(data)} 个文件")
    for path, df in list(data.items())[:3]:
        print(f"  - {path.relative_to(root)}: {df.shape[0]} 行 x {df.shape[1]} 列")
