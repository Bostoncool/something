import os
import shutil
import re

def organize_pm25_files(source_dir):
    """
    组织PM2.5数据文件到对应的文件夹结构
    
    Args:
        source_dir (str): 源文件目录路径
    """
    
    # 定义文件格式模式
    patterns = {
        'Year': r'CHAP_PM2\.5_Y1K_(\d{4})_V4\.nc',
        'Month': r'CHAP_PM2\.5_M1K_(\d{6})_V4\.nc',
        'Day': r'CHAP_PM2\.5_D1K_(\d{8})_V4\.nc'
    }
    
    # 创建主文件夹
    year_dir = os.path.join(source_dir, 'Year')
    month_dir = os.path.join(source_dir, 'Month')
    
    # 创建Year和Month主目录
    os.makedirs(year_dir, exist_ok=True)
    os.makedirs(month_dir, exist_ok=True)
    
    # 为Month文件夹创建年份子文件夹 (2000-2023)
    year_folders = {}
    for year in range(2000, 2024):
        year_folder = os.path.join(month_dir, str(year))
        os.makedirs(year_folder, exist_ok=True)
        year_folders[str(year)] = year_folder
    
    # 使用字典来存储文件路径，避免重复搜索
    files_dict = {}
    
    # 递归扫描源目录中的所有文件（包括子目录）
    print("Scanning files recursively...")
    for root, dirs, files in os.walk(source_dir):
        # 跳过我们创建的目标目录，避免重复处理
        if root.startswith(year_dir) or root.startswith(month_dir):
            continue
            
        for filename in files:
            file_path = os.path.join(root, filename)
            # 使用相对路径作为键，避免不同目录下同名文件冲突
            relative_path = os.path.relpath(file_path, source_dir)
            files_dict[relative_path] = file_path
    
    print(f"Found {len(files_dict)} files to process")
    
    # 统计计数器
    year_count = 0
    month_count = 0
    day_count = 0
    other_count = 0
    
    # 处理文件
    for relative_path, file_path in files_dict.items():
        filename = os.path.basename(file_path)
        
        # 检查Year格式文件
        year_match = re.match(patterns['Year'], filename)
        if year_match:
            target_path = os.path.join(year_dir, filename)
            # 如果目标文件已存在，添加后缀避免覆盖
            if os.path.exists(target_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(os.path.join(year_dir, f"{base}_{counter}{ext}")):
                    counter += 1
                target_path = os.path.join(year_dir, f"{base}_{counter}{ext}")
            
            shutil.move(file_path, target_path)
            year_count += 1
            print(f"Moved Year file: {relative_path} -> Year/")
            continue
        
        # 检查Month格式文件
        month_match = re.match(patterns['Month'], filename)
        if month_match:
            date_str = month_match.group(1)
            year = date_str[:4]
            
            # 检查年份是否在2000-2023范围内
            if year in year_folders:
                target_dir = year_folders[year]
                target_path = os.path.join(target_dir, filename)
                
                # 如果目标文件已存在，添加后缀避免覆盖
                if os.path.exists(target_path):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(os.path.join(target_dir, f"{base}_{counter}{ext}")):
                        counter += 1
                    target_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
                
                shutil.move(file_path, target_path)
                month_count += 1
                print(f"Moved Month file: {relative_path} -> Month/{year}/")
            else:
                print(f"Warning: Year {year} out of range for file {filename}")
            continue
        
        # 检查Day格式文件（根据要求不需要移动，但可以记录）
        day_match = re.match(patterns['Day'], filename)
        if day_match:
            day_count += 1
            # 可以取消注释下面的行来查看Day文件
            # print(f"Day format file kept in place: {relative_path}")
            continue
        
        # 其他文件
        other_count += 1
        # 可以取消注释下面的行来查看其他文件
        # print(f"Unrecognized file format: {relative_path}")
    
    # 输出统计信息
    print("\n=== Processing Summary ===")
    print(f"Year format files moved: {year_count}")
    print(f"Month format files moved: {month_count}")
    print(f"Day format files found: {day_count}")
    print(f"Other files: {other_count}")
    print(f"Total files processed: {len(files_dict)}")

def cleanup_empty_dirs(source_dir):
    """
    清理移动文件后可能产生的空目录
    
    Args:
        source_dir (str): 源文件目录路径
    """
    print("Cleaning up empty directories...")
    empty_dirs = []
    
    # 从最深层开始遍历，避免删除非空目录
    for root, dirs, files in os.walk(source_dir, topdown=False):
        # 跳过我们创建的目标目录
        if root.startswith(os.path.join(source_dir, 'Year')) or root.startswith(os.path.join(source_dir, 'Month')):
            continue
            
        # 如果目录为空，则记录并删除
        if not dirs and not files:
            empty_dirs.append(root)
            try:
                os.rmdir(root)
                print(f"Removed empty directory: {root}")
            except OSError as e:
                print(f"Error removing directory {root}: {e}")
    
    print(f"Cleaned up {len(empty_dirs)} empty directories")

def main():
    # 文件路径
    source_directory = r"E:\2000-2023[PM2.5-china]"
    
    # 检查源目录是否存在
    if not os.path.exists(source_directory):
        print(f"Error: Source directory {source_directory} does not exist!")
        return
    
    print("Starting file organization...")
    organize_pm25_files(source_directory)
    
    # 可选：清理空目录
    cleanup_empty_dirs(source_directory)
    
    print("File organization completed!")

if __name__ == "__main__":
    main()