import os
import shutil
from pathlib import Path

def move_files_from_folders(source_root, target_folder):
    """
    将source_root中所有子文件夹内的文件移动到target_folder中
    
    参数:
    source_root: 包含多个未压缩文件夹的根目录
    target_folder: 目标文件夹，所有文件将移动到这里
    """
    
    # 确保目标文件夹存在
    target_path = Path(target_folder)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 计数器
    files_moved = 0
    folders_processed = 0
    
    # 遍历源根目录
    for root, dirs, files in os.walk(source_root):
        # 跳过根目录本身（只处理子文件夹）
        if root == source_root:
            continue
            
        # 处理当前文件夹中的文件
        for file in files:
            source_file = Path(root) / file
            target_file = target_path / file
            
            # 处理文件名冲突
            counter = 1
            while target_file.exists():
                # 如果文件已存在，添加后缀
                name_parts = file.split('.')
                if len(name_parts) > 1:
                    # 有扩展名的情况
                    new_name = f"{'.'.join(name_parts[:-1])}_{counter}.{name_parts[-1]}"
                else:
                    # 无扩展名的情况
                    new_name = f"{file}_{counter}"
                target_file = target_path / new_name
                counter += 1
            
            try:
                # 移动文件
                shutil.move(str(source_file), str(target_file))
                files_moved += 1
                print(f"已移动: {source_file} -> {target_file}")
            except Exception as e:
                print(f"移动失败 {source_file}: {e}")
        
        folders_processed += 1
    
    print(f"\n处理完成!")
    print(f"已处理文件夹数: {folders_processed}")
    print(f"已移动文件数: {files_moved}")
    print(f"所有文件已移动到: {target_path}")

def move_files_with_options(source_root, target_folder, 
                            keep_structure=False, delete_empty_folders=False):
    """
    更多选项的版本
    
    参数:
    source_root: 源根目录
    target_folder: 目标文件夹
    keep_structure: 是否保持文件夹结构（默认False）
    delete_empty_folders: 移动后是否删除空文件夹（默认False）
    """
    
    target_path = Path(target_folder)
    target_path.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_root)
    files_moved = 0
    
    if keep_structure:
        # 保持文件夹结构
        for file_path in source_path.rglob('*'):
            if file_path.is_file():
                # 计算相对路径
                relative_path = file_path.relative_to(source_path)
                target_file = target_path / relative_path
                
                # 创建必要的父目录
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # 处理文件名冲突
                counter = 1
                original_target = target_file
                while target_file.exists():
                    # 添加后缀避免冲突
                    name_parts = original_target.name.split('.')
                    if len(name_parts) > 1:
                        new_name = f"{'.'.join(name_parts[:-1])}_{counter}.{name_parts[-1]}"
                    else:
                        new_name = f"{original_target.name}_{counter}"
                    target_file = original_target.parent / new_name
                    counter += 1
                
                try:
                    shutil.move(str(file_path), str(target_file))
                    files_moved += 1
                    print(f"已移动: {file_path} -> {target_file}")
                except Exception as e:
                    print(f"移动失败 {file_path}: {e}")
    else:
        # 平铺所有文件
        for root, dirs, files in os.walk(source_root):
            for file in files:
                source_file = Path(root) / file
                target_file = target_path / file
                
                # 处理文件名冲突
                counter = 1
                while target_file.exists():
                    name_parts = file.split('.')
                    if len(name_parts) > 1:
                        new_name = f"{'.'.join(name_parts[:-1])}_{counter}.{name_parts[-1]}"
                    else:
                        new_name = f"{file}_{counter}"
                    target_file = target_path / new_name
                    counter += 1
                
                try:
                    shutil.move(str(source_file), str(target_file))
                    files_moved += 1
                    print(f"已移动: {source_file} -> {target_file}")
                except Exception as e:
                    print(f"移动失败 {source_file}: {e}")
    
    # 可选：删除空文件夹
    if delete_empty_folders:
        delete_empty_folders_func(source_path)
    
    print(f"\n处理完成! 已移动 {files_moved} 个文件到 {target_path}")

def delete_empty_folders_func(folder_path):
    """删除空文件夹"""
    deleted_count = 0
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            try:
                if not any(dir_path.iterdir()):  # 文件夹为空
                    dir_path.rmdir()
                    deleted_count += 1
                    print(f"已删除空文件夹: {dir_path}")
            except Exception as e:
                print(f"无法删除文件夹 {dir_path}: {e}")
    print(f"已删除 {deleted_count} 个空文件夹")

def main():
    # 使用方法示例
    source_directory = r"C:\Users\IU\Desktop\大论文图\4.珠三角\DEM"
    target_directory = r"C:\Users\IU\Desktop\大论文图\4.珠三角\DEM\合并"
    
    # 检查路径是否存在
    if not os.path.exists(source_directory):
        print(f"错误: 源目录不存在 - {source_directory}")
        return
    
    # 确认操作
    print(f"\n将要操作:")
    print(f"源目录: {source_directory}")
    print(f"目标目录: {target_directory}")
    confirm = input("\n确认执行? (y/n): ").lower()
    
    if confirm == 'y':
        # 使用简单版本
        # move_files_from_folders(source_directory, target_directory)
        
        # 或者使用带选项的版本
        move_files_with_options(
            source_directory, 
            target_directory,
            keep_structure=False,  # 不保持文件夹结构
            delete_empty_folders=True  # 删除空文件夹
        )

if __name__ == "__main__":
    main()