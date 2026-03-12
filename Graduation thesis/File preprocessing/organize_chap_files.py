"""
CHAP 数据文件分类整理脚本

将符合命名规则的文件按类型分类到对应文件夹：
1. CHAP_污染物名称_Y1K_YYYY_V4.*  -> Year/
2. CHAP_污染物名称_M1K_YYYYMM_V4.* -> Month/YYYY/
3. CHAP_污染物名称_D1K_YYYYMMDD_V4.* -> Day/YYYY/

支持任意污染物名称（如 PM2.5、PM10、NO2、SO2、O3、CO 等）及多种文件扩展名。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

# ========== 可在此处直接修改源文件夹路径 ==========
# 留空或 None 时，使用命令行参数或当前目录
SOURCE_DIR_DEFAULT: Path | None = Path(r"G:\基本用不到\SO2")
# =================================================

# 文件命名模式（支持任意污染物名称，V4 可选）
YEAR_PATTERN = re.compile(
    r"^CHAP_.+_Y1K_(\d{4})(?:_V\d+)?\.[a-zA-Z0-9]+$", re.IGNORECASE
)
MONTH_PATTERN = re.compile(
    r"^CHAP_.+_M1K_(\d{6})(?:_V\d+)?\.[a-zA-Z0-9]+$", re.IGNORECASE
)
DAY_PATTERN = re.compile(
    r"^CHAP_.+_D1K_(\d{8})(?:_V\d+)?\.[a-zA-Z0-9]+$", re.IGNORECASE
)


def get_target_dirs(source_dir: Path) -> tuple[Path, Path, Path]:
    """获取 Year、Month、Day 三个主目录路径。"""
    return (
        source_dir / "Year",
        source_dir / "Month",
        source_dir / "Day",
    )


def ensure_dirs(year_dir: Path, month_dir: Path, day_dir: Path) -> None:
    """确保主目录存在。"""
    year_dir.mkdir(parents=True, exist_ok=True)
    month_dir.mkdir(parents=True, exist_ok=True)
    day_dir.mkdir(parents=True, exist_ok=True)


def get_or_create_year_subdir(parent: Path, year: str) -> Path:
    """获取或创建年份子目录。"""
    subdir = parent / year
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def collect_files_to_process(
    source_dir: Path,
    year_dir: Path,
    month_dir: Path,
    day_dir: Path,
) -> list[tuple[Path, str]]:
    """
    递归扫描源目录，收集待处理文件。
    返回 [(文件路径, 相对路径字符串), ...]，跳过已在目标目录中的文件。
    """
    year_resolved = str(year_dir.resolve())
    month_resolved = str(month_dir.resolve())
    day_resolved = str(day_dir.resolve())

    files_to_process: list[tuple[Path, str]] = []

    for root, _, files in os.walk(source_dir, topdown=True):
        root_path = Path(root).resolve()
        root_s = str(root_path)

        # 跳过目标目录，避免重复处理
        if root_s.startswith(year_resolved) or root_s.startswith(month_resolved) or root_s.startswith(day_resolved):
            continue

        for name in files:
            file_path = root_path / name
            if not file_path.is_file():
                continue
            rel = os.path.relpath(file_path, source_dir)
            files_to_process.append((file_path, rel))

    return files_to_process


def resolve_target_path(target_dir: Path, filename: str) -> Path:
    """若目标文件已存在，则生成带数字后缀的新文件名，避免覆盖。"""
    target = target_dir / filename
    if not target.exists():
        return target

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while (target_dir / f"{stem}_{counter}{suffix}").exists():
        counter += 1
    return target_dir / f"{stem}_{counter}{suffix}"


def organize_files(
    source_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, int]:
    """
    对源目录中的 CHAP 文件进行分类整理。

    Args:
        source_dir: 源目录路径
        dry_run: 若为 True，仅打印操作，不实际移动文件
        verbose: 是否打印每条移动记录

    Returns:
        统计字典：year_count, month_count, day_count, other_count
    """
    year_dir, month_dir, day_dir = get_target_dirs(source_dir)
    ensure_dirs(year_dir, month_dir, day_dir)

    files = collect_files_to_process(source_dir, year_dir, month_dir, day_dir)

    stats = {"year_count": 0, "month_count": 0, "day_count": 0, "other_count": 0}

    for file_path, rel_path in files:
        filename = file_path.name

        # Year: CHAP_*_Y1K_YYYY_V4.*
        year_match = YEAR_PATTERN.match(filename)
        if year_match:
            target = resolve_target_path(year_dir, filename)
            if not dry_run:
                shutil.move(str(file_path), str(target))
            stats["year_count"] += 1
            if verbose:
                print(f"Year: {rel_path} -> Year/")
            continue

        # Month: CHAP_*_M1K_YYYYMM_V4.*
        month_match = MONTH_PATTERN.match(filename)
        if month_match:
            yyyymm = month_match.group(1)
            year = yyyymm[:4]
            subdir = get_or_create_year_subdir(month_dir, year)
            target = resolve_target_path(subdir, filename)
            if not dry_run:
                shutil.move(str(file_path), str(target))
            stats["month_count"] += 1
            if verbose:
                print(f"Month: {rel_path} -> Month/{year}/")
            continue

        # Day: CHAP_*_D1K_YYYYMMDD_V4.*
        day_match = DAY_PATTERN.match(filename)
        if day_match:
            yyyymmdd = day_match.group(1)
            year = yyyymmdd[:4]
            subdir = get_or_create_year_subdir(day_dir, year)
            target = resolve_target_path(subdir, filename)
            if not dry_run:
                shutil.move(str(file_path), str(target))
            stats["day_count"] += 1
            if verbose:
                print(f"Day: {rel_path} -> Day/{year}/")
            continue

        stats["other_count"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 CHAP 数据文件按 Year/Month/Day 分类整理到对应文件夹"
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        nargs="?",
        default=SOURCE_DIR_DEFAULT if SOURCE_DIR_DEFAULT is not None else Path.cwd(),
        help="源数据目录路径（默认：代码中 SOURCE_DIR_DEFAULT 或当前目录）",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="仅预览操作，不实际移动文件",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="减少输出，仅打印统计信息",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    if not source_dir.exists():
        print(f"错误：源目录不存在: {source_dir}")
        return
    if not source_dir.is_dir():
        print(f"错误：路径不是目录: {source_dir}")
        return

    if args.dry_run:
        print("【预览模式】不会实际移动文件\n")

    stats = organize_files(
        source_dir,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    print("\n=== 处理统计 ===")
    print(f"Year 格式文件: {stats['year_count']}")
    print(f"Month 格式文件: {stats['month_count']}")
    print(f"Day 格式文件: {stats['day_count']}")
    print(f"未识别文件: {stats['other_count']}")
    total = sum(stats.values())
    print(f"总文件数: {total}")

    if args.dry_run and total > 0:
        print("\n使用时不带 -n 参数将执行实际移动。")


if __name__ == "__main__":
    main()
