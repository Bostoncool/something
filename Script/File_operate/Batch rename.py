import os

def rename_files_in_folder(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    # 只保留文件，排除子文件夹
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
    
    # 第一阶段：将所有文件重命名为临时名称，避免文件名冲突
    temp_files = []
    for i, file in enumerate(files):
        old_file_path = os.path.join(folder_path, file)
        temp_name = f"__temp_{i}__{file}"
        temp_file_path = os.path.join(folder_path, temp_name)
        os.rename(old_file_path, temp_file_path)
        temp_files.append((temp_file_path, file))
    
    # 第二阶段：将临时名称重命名为最终名称
    for i, (temp_file_path, original_file) in enumerate(temp_files, start=1):
        # 获取原始文件的扩展名
        file_extension = os.path.splitext(original_file)[1]
        # 创建新的文件名
        new_name = f"{i}{file_extension}"
        new_file_path = os.path.join(folder_path, new_name)
        
        # 修改文件名
        os.rename(temp_file_path, new_file_path)
        print(f"文件 '{original_file}' 已被重命名为 '{new_name}'")

# 示例：调用函数并传入文件夹路径
folder_path = r'D:\11408计算机考研（计网+操作系统+组成+数据结构）'
rename_files_in_folder(folder_path)
