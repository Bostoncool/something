import os
import hashlib

def calculate_md5(file_path):
    """计算文件的MD5值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_files_in_directory(directory):
    """获取文件夹中的所有文件路径"""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def find_duplicate_files(dir1, dir2):
    """找出两个文件夹中的重复文件"""
    files_dir1 = get_files_in_directory(dir1)
    files_dir2 = get_files_in_directory(dir2)
    
    # 用一个字典存储文件的MD5值和文件路径
    md5_dict = {}

    # 遍历第一个文件夹中的文件，存储MD5值
    for file_path in files_dir1:
        file_md5 = calculate_md5(file_path)
        md5_dict[file_md5] = md5_dict.get(file_md5, []) + [file_path]
    
    # 遍历第二个文件夹中的文件，检查是否有重复的MD5值
    duplicates = []
    for file_path in files_dir2:
        file_md5 = calculate_md5(file_path)
        if file_md5 in md5_dict:
            duplicates.append((file_path, md5_dict[file_md5]))
    
    return duplicates

def delete_duplicates_in_dir2(duplicates):
    """删除dir2中的重复文件，保留dir1中的文件"""
    deleted_count = 0
    deleted_files = []
    
    for dup in duplicates:
        file_path_dir2 = dup[0]  # dir2中的文件路径
        try:
            if os.path.exists(file_path_dir2):
                os.remove(file_path_dir2)
                deleted_count += 1
                deleted_files.append(file_path_dir2)
                print(f"已删除: {file_path_dir2}")
            else:
                print(f"文件不存在，跳过: {file_path_dir2}")
        except Exception as e:
            print(f"删除文件失败 {file_path_dir2}: {str(e)}")
    
    return deleted_count, deleted_files

def main():
    dir1 = r"D:\11408计算机考研（计网+操作系统+组成+数据结构）"
    dir2 = r"D:\11408计算机考研（计网+操作系统+组成+数据结构）"
    duplicates = find_duplicate_files(dir1, dir2)

    if duplicates:
        print(f"找到 {len(duplicates)} 个重复的文件:")
        for dup in duplicates:
            print(f"dir2中的文件: {dup[0]}")
            print(f"  与dir1中的文件重复: {dup[1]}")
        
        print("\n开始删除dir2中的重复文件...")
        deleted_count, deleted_files = delete_duplicates_in_dir2(duplicates)
        print(f"\n删除完成！共删除 {deleted_count} 个重复文件。")
    else:
        print("没有找到重复的文件。")

if __name__ == "__main__":
    main()
