# -*- coding: utf-8 -*-
from __future__ import annotations

"""
统计数据缺失情况：根据文件名称，统计数据缺失的具体天数

参考 YZD-抽取城市.py 的读取方式，分析数据完整性
"""

import sys
import io

# 设置控制台编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
from datetime import datetime, timedelta

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

# 数据根目录（包含 CSV 文件的目录）
# TODO: 改成你的真实数据路径
INPUT_ROOT = Path(r"E:\DATA Science\大论文Result\PRD")  # 相对路径示例，需要根据实际情况调整

# 需要检查的城市列表
CITIES = [
    "广州", "深圳", "佛山", "东莞", "中山",
    "惠州", "珠海", "江门", "肇庆"
]

# CSV 里固定字段
BASE_COLS = ["date", "hour", "type"]

# 文件名中的日期正则表达式（如 china_cities_20230101.csv / 20230101.csv 等）
DATE_RE = re.compile(r"(20\d{6})")

# 数据时间范围（根据你的数据调整）
START_DATE = "2018-01-01"  # 开始日期
END_DATE = "2023-12-31"    # 结束日期


# =========================
# 2) 工具函数
# =========================

def list_csv_files(root: Path) -> List[Path]:
    """递归列出所有 csv 文件"""
    return sorted([p for p in root.rglob("*.csv") if p.is_file()])


def extract_date_from_filename(filename: str) -> Optional[str]:
    """从文件名中提取日期"""
    m = DATE_RE.search(filename)
    if m:
        date_str = m.group(1)
        # 转换为 YYYY-MM-DD 格式
        try:
            dt = datetime.strptime(date_str, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None


def get_expected_date_range(start_date: str, end_date: str) -> List[str]:
    """生成期望的日期范围"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = []

    current = start
    while current <= end:
        date_list.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return date_list


def analyze_file_completeness(csv_path: Path) -> Dict:
    """
    分析单个文件的完整性
    返回：包含文件信息和缺失情况的字典
    """
    result = {
        'filename': csv_path.name,
        'date': extract_date_from_filename(csv_path.name),
        'file_exists': True,
        'can_read': False,
        'missing_base_cols': [],
        'missing_cities': [],
        'total_rows': 0,
        'complete_rows': 0,
        'error': None
    }

    try:
        # 读取表头
        header = pd.read_csv(csv_path, nrows=0)
        available_cols = set(header.columns)
        result['can_read'] = True

        # 检查基础列
        result['missing_base_cols'] = [col for col in BASE_COLS if col not in available_cols]

        # 检查城市列
        result['missing_cities'] = [city for city in CITIES if city not in available_cols]

        # 读取完整数据进行更详细的分析
        df = pd.read_csv(csv_path)
        result['total_rows'] = len(df)

        # 检查哪些行是完整的（所有城市列都有数据）
        city_cols_present = [city for city in CITIES if city in df.columns]
        if city_cols_present:
            # 计算完整行数（非空值的城市列数等于存在的城市列数）
            complete_mask = df[city_cols_present].notna().all(axis=1)
            result['complete_rows'] = complete_mask.sum()

    except Exception as e:
        result['error'] = str(e)

    return result


def generate_missing_report(all_dates: List[str], file_analysis: List[Dict]) -> Dict:
    """生成缺失报告"""
    # 创建日期到文件分析的映射
    date_to_analysis = {analysis['date']: analysis for analysis in file_analysis if analysis['date']}

    # 找出完全缺失的文件（没有对应日期的文件）
    existing_dates = set(date_to_analysis.keys())
    missing_dates = [date for date in all_dates if date not in existing_dates]

    # 分析有文件但数据不完整的日期
    incomplete_dates = []
    for date, analysis in date_to_analysis.items():
        if analysis['error'] or not analysis['can_read'] or analysis['missing_base_cols'] or analysis['missing_cities']:
            incomplete_dates.append({
                'date': date,
                'issues': []
            })
            if analysis['error']:
                incomplete_dates[-1]['issues'].append(f"读取错误: {analysis['error']}")
            if not analysis['can_read']:
                incomplete_dates[-1]['issues'].append("无法读取文件")
            if analysis['missing_base_cols']:
                incomplete_dates[-1]['issues'].append(f"缺少基础列: {', '.join(analysis['missing_base_cols'])}")
            if analysis['missing_cities']:
                incomplete_dates[-1]['issues'].append(f"缺少城市列: {', '.join(analysis['missing_cities'])}")

    # 统计城市级别的缺失情况
    city_missing_stats = {}
    for city in CITIES:
        missing_count = 0
        for analysis in file_analysis:
            if analysis['can_read'] and city in analysis['missing_cities']:
                missing_count += 1
        city_missing_stats[city] = missing_count

    return {
        'total_expected_days': len(all_dates),
        'existing_files': len(existing_dates),
        'missing_files': len(missing_dates),
        'incomplete_files': len(incomplete_dates),
        'missing_dates': missing_dates,
        'incomplete_dates': incomplete_dates,
        'city_missing_stats': city_missing_stats,
        'file_analysis': file_analysis
    }


# =========================
# 3) 主流程
# =========================

def main():
    print("开始统计数据缺失情况...")
    print(f"数据目录：{INPUT_ROOT.absolute()}")

    # 检查目录是否存在
    if not INPUT_ROOT.exists():
        print(f"警告：目录不存在 - {INPUT_ROOT.absolute()}")
        print("请修改 INPUT_ROOT 配置为你的实际数据目录路径")
        print("\n示例：")
        print("INPUT_ROOT = Path(r\"C:\\Users\\YourName\\Desktop\\China_AQI_Data\")")
        return

    # 获取所有CSV文件
    csv_files = list_csv_files(INPUT_ROOT)
    if not csv_files:
        print(f"没找到任何CSV文件在：{INPUT_ROOT.absolute()}")
        print("请确保数据目录包含CSV文件")
        print("\n预期文件格式：")
        print("- 文件名包含日期，如：china_cities_20230101.csv 或 20230101.csv")
        print("- 文件内容包含列：date, hour, type, 城市名...")
        return

    print(f"发现 {len(csv_files)} 个CSV文件")

    # 生成期望的日期范围
    expected_dates = get_expected_date_range(START_DATE, END_DATE)
    print(f"期望的日期范围：{START_DATE} 到 {END_DATE}（共 {len(expected_dates)} 天）")

    # 分析每个文件的完整性
    file_analysis = []
    print("分析文件完整性...")

    with tqdm(total=len(csv_files), desc="分析文件", unit="file") as pbar:
        for csv_path in csv_files:
            analysis = analyze_file_completeness(csv_path)
            file_analysis.append(analysis)
            pbar.update(1)

    # 生成缺失报告
    report = generate_missing_report(expected_dates, file_analysis)

    # 输出统计结果
    print("\n" + "="*50)
    print("数据缺失统计报告")
    print("="*50)
    print(f"期望总天数：{report['total_expected_days']}")
    print(f"实际存在文件数：{report['existing_files']}")
    print(f"完全缺失文件的天数：{report['missing_files']}")
    print(f"数据不完整的文件数：{report['incomplete_files']}")

    # 输出缺失的日期
    if report['missing_dates']:
        print(f"\n完全缺失文件的日期（共 {len(report['missing_dates'])} 天）：")
        # 按年份分组显示
        from collections import defaultdict
        missing_by_year = defaultdict(list)
        for date in report['missing_dates']:
            year = date[:4]
            missing_by_year[year].append(date)

        for year in sorted(missing_by_year.keys()):
            dates = missing_by_year[year]
            print(f"  {year}年：{len(dates)} 天 - {dates[0]} 到 {dates[-1]}")
            # 如果日期不连续，显示具体日期
            if len(dates) <= 10:
                print(f"    具体日期：{', '.join(dates)}")
    else:
        print("\n没有完全缺失的文件日期")

    # 输出数据不完整的日期
    if report['incomplete_dates']:
        print(f"\n数据不完整的日期（共 {len(report['incomplete_dates'])} 天）：")
        for item in report['incomplete_dates'][:10]:  # 只显示前10个
            print(f"  {item['date']}: {'; '.join(item['issues'])}")
        if len(report['incomplete_dates']) > 10:
            print(f"  ... 还有 {len(report['incomplete_dates']) - 10} 个日期")
    else:
        print("\n所有存在文件的日期数据都是完整的")

    # 输出城市级别的缺失统计
    print(f"\n城市数据缺失统计（在有文件的日期中）：")
    sorted_cities = sorted(report['city_missing_stats'].items(), key=lambda x: x[1], reverse=True)
    for city, missing_count in sorted_cities:
        if missing_count > 0:
            total_files = len([a for a in file_analysis if a['can_read']])
            percentage = (missing_count / total_files * 100) if total_files > 0 else 0
            print(f"  {city}: {missing_count} 个文件缺失 ({percentage:.1f}%)")
    # 保存详细报告到CSV
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 保存缺失日期列表
    if report['missing_dates']:
        missing_dates_df = pd.DataFrame({'missing_date': report['missing_dates']})
        missing_dates_df.to_csv(output_dir / "missing_dates.csv", index=False, encoding="utf-8-sig")
        print(f"\n缺失日期详情已保存到：{output_dir / 'missing_dates.csv'}")

    # 保存文件分析详情
    analysis_df = pd.DataFrame(file_analysis)
    analysis_df.to_csv(output_dir / "file_analysis.csv", index=False, encoding="utf-8-sig")
    print(f"文件分析详情已保存到：{output_dir / 'file_analysis.csv'}")

    # 保存城市缺失统计
    city_stats_df = pd.DataFrame(list(report['city_missing_stats'].items()),
                                columns=['city', 'missing_files_count'])
    city_stats_df.to_csv(output_dir / "city_missing_stats.csv", index=False, encoding="utf-8-sig")
    print(f"城市缺失统计已保存到：{output_dir / 'city_missing_stats.csv'}")

    print("\n统计完成！")


if __name__ == "__main__":
    main()