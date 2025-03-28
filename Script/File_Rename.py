import os
import re

def batch_rename_files(folder_path, old_keyword, new_keyword):
    """
    批量重命名文件夹中的文件，将文件名中的特定关键词替换为新关键词
    
    参数:
        folder_path (str): 目标文件夹路径
        old_keyword (str): 需要替换的关键词
        new_keyword (str): 替换后的新关键词
    """
    # 验证文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 {folder_path} 不存在")
        return
    
    # 获取文件夹中所有文件
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # 如果没有文件则退出
    if not files:
        print("文件夹中没有文件")
        return
    
    renamed_count = 0
    
    for filename in files:
        # 如果文件名包含关键词
        if old_keyword in filename:
            # 构建新文件名
            new_filename = filename.replace(old_keyword, new_keyword)
            
            # 构建完整路径
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            
            # 重命名文件
            try:
                os.rename(old_path, new_path)
                print(f"重命名: {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"重命名 {filename} 失败: {str(e)}")
    
    print(f"\n完成! 共重命名了 {renamed_count} 个文件")

if __name__ == "__main__":
    # 示例用法
    folder = r"C:\Users\IU\Desktop\Moderately Polluted\AQI-200"
    old_kw = "AQI_200+"
    new_kw = "AQI_200+ "
    
    batch_rename_files(folder, old_kw, new_kw)
