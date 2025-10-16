"""
CSV文件按年份分类整理脚本

功能：
- 读取目录中的PM2.5 CSV文件
- 根据文件名中的年份信息创建对应文件夹
- 将文件移动到对应年份的文件夹中

文件名格式：
- CHAP_PM2.5_D1K_YYYYMMDD_V4.csv (日数据)
- CHAP_PM2.5_M1K_YYYYMM_V4.csv (月数据)
- CHAP_PM2.5_Y1K_YYYY_V4.csv (年数据)
"""

import os
import shutil
import re
from pathlib import Path


def extract_year_from_filename(filename):
    """
    从文件名中提取年份
    
    参数:
        filename: 文件名字符串
        
    返回:
        year: 年份字符串 (YYYY)，如果无法提取则返回 None
    """
    # 移除文件扩展名
    name_without_ext = filename.replace('.csv', '')
    
    # 匹配不同的文件名模式
    patterns = [
        r'CHAP_PM2\.5_D1K_(\d{4})\d{4}_V4',  # 日数据：YYYYMMDD
        r'CHAP_PM2\.5_M1K_(\d{4})\d{2}_V4',  # 月数据：YYYYMM
        r'CHAP_PM2\.5_Y1K_(\d{4})_V4'        # 年数据：YYYY
    ]
    
    for pattern in patterns:
        match = re.match(pattern, name_without_ext)
        if match:
            return match.group(1)
    
    return None


def organize_files_by_year(source_dir='.', dry_run=False):
    """
    将CSV文件按年份组织到对应的文件夹中
    
    参数:
        source_dir: 源目录路径，默认为当前目录
        dry_run: 如果为True，只打印操作而不实际移动文件
    """
    source_path = Path(source_dir)
    
    # 获取所有符合模式的CSV文件
    csv_files = [f for f in source_path.glob('CHAP_PM2.5_*.csv')]
    
    if not csv_files:
        print("未找到符合命名规则的CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个文件\n")
    
    # 统计信息
    moved_count = 0
    failed_count = 0
    year_folders = {}
    
    for csv_file in csv_files:
        filename = csv_file.name
        year = extract_year_from_filename(filename)
        
        if year is None:
            print(f"⚠ 无法从文件名提取年份: {filename}")
            failed_count += 1
            continue
        
        # 创建年份文件夹路径
        year_folder = source_path / year
        
        # 记录每个年份的文件数量
        if year not in year_folders:
            year_folders[year] = []
        year_folders[year].append(filename)
        
        # 目标文件路径
        target_file = year_folder / filename
        
        if dry_run:
            print(f"[模拟] {filename} -> {year}/{filename}")
        else:
            # 创建年份文件夹（如果不存在）
            if not year_folder.exists():
                year_folder.mkdir(parents=True, exist_ok=True)
                print(f"✓ 创建文件夹: {year}/")
            
            # 移动文件
            try:
                shutil.move(str(csv_file), str(target_file))
                print(f"✓ 移动: {filename} -> {year}/")
                moved_count += 1
            except Exception as e:
                print(f"✗ 移动失败: {filename} - {str(e)}")
                failed_count += 1
    
    # 输出统计信息
    print("\n" + "="*60)
    print("处理完成统计:")
    print("="*60)
    
    if not dry_run:
        print(f"成功移动: {moved_count} 个文件")
        print(f"失败: {failed_count} 个文件")
    else:
        print(f"将移动: {len(csv_files) - failed_count} 个文件")
        print(f"无法识别: {failed_count} 个文件")
    
    print(f"\n按年份分布:")
    for year in sorted(year_folders.keys()):
        print(f"  {year}: {len(year_folders[year])} 个文件")


def main():
    """主函数"""
    import sys
    
    print("="*60)
    print("CSV文件年份整理工具")
    print("="*60)
    print()
    
    # 获取目标目录
    if len(sys.argv) > 1:
        # 如果命令行提供了路径参数
        target_dir = Path(sys.argv[1])
    else:
        # 交互式输入路径
        print("请输入要整理的文件夹路径：")
        print("(直接按回车使用当前脚本所在目录)")
        user_input = input("路径: ").strip()
        
        if user_input:
            target_dir = Path(user_input)
        else:
            target_dir = Path(__file__).parent
    
    # 验证目录是否存在
    if not target_dir.exists():
        print(f"\n✗ 错误：目录不存在: {target_dir}")
        return
    
    if not target_dir.is_dir():
        print(f"\n✗ 错误：路径不是一个目录: {target_dir}")
        return
    
    print(f"\n目标目录: {target_dir.absolute()}\n")
    
    # 首先进行预览（模拟运行）
    print("【预览模式】查看将要执行的操作...\n")
    organize_files_by_year(target_dir, dry_run=True)
    
    print("\n" + "="*60)
    response = input("\n是否确认执行以上操作？(y/n): ").strip().lower()
    
    if response == 'y':
        print("\n开始移动文件...\n")
        organize_files_by_year(target_dir, dry_run=False)
        print("\n✓ 所有操作已完成！")
    else:
        print("\n操作已取消")


if __name__ == "__main__":
    main()

