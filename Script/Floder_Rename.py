import os

def rename_folders(directory_path):
    """
    递归遍历指定目录及其子目录中的所有文件夹，重命名包含'Add-on--'的文件夹
    
    Args:
        directory_path (str): 要处理的目录路径
    """
    # 检查目录是否存在
    if not os.path.exists(directory_path):
        print(f'错误：目录 "{directory_path}" 不存在')
        return
    
    # 检查是否是目录
    if not os.path.isdir(directory_path):
        print(f'错误："{directory_path}" 不是一个有效的目录')
        return
    
    # 使用os.walk递归遍历所有子目录
    for root, dirs, files in os.walk(directory_path):
        for dir_name in dirs:
            # 构建完整的文件路径
            item_path = os.path.join(root, dir_name)
            
            # 检查文件夹名称中是否包含'Add-on--'
            if '&' in dir_name:
                # 构建新的文件夹名称
                new_name = dir_name.replace('和', ' and ')
                new_path = os.path.join(root, new_name)
                
                try:
                    # 重命名文件夹
                    os.rename(item_path, new_path)
                    print(f'已将文件夹 "{dir_name}" 重命名为 "{new_name}"')
                except Exception as e:
                    print(f'重命名 "{dir_name}" 时发生错误: {str(e)}')

if __name__ == '__main__':
    # 在这里指定要处理的目录路径
    target_directory = r"D:\AI算法工程师就业班\07、Machine Learning-无监督学习\Chapter2：EM算法和GMM高斯混合模型"  # 示例路径，请根据实际情况修改
    rename_folders(target_directory)
    print('重命名操作完成！')
