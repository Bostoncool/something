import os
import shutil

def remove_nested_folders(base_path):
    """
    遍历指定路径，删除多余的同名文件夹，同时保留文件。
    """
    # 获取当前路径下的所有文件和文件夹
    items = os.listdir(base_path)
    
    for item in items:
        item_path = os.path.join(base_path, item)
        
        # 如果是文件夹
        if os.path.isdir(item_path):
            # 获取文件夹内的内容
            nested_items = os.listdir(item_path)
            
            # 如果文件夹内只有一个同名文件夹
            if len(nested_items) == 1 and nested_items[0] == item:
                nested_path = os.path.join(item_path, nested_items[0])
                
                # 将内部文件移动到外层文件夹
                for file in os.listdir(nested_path):
                    file_path = os.path.join(nested_path, file)
                    shutil.move(file_path, item_path)
                
                # 删除多余的同名文件夹
                os.rmdir(nested_path)
                print(f"Removed nested folder: {nested_path}")
            
            # 递归处理当前文件夹
            remove_nested_folders(item_path)
        else:
            # 如果是文件，直接跳过
            continue

# 指定需要处理的路径
base_path = r"H:\DATA Science\Datebase Origin"
remove_nested_folders(base_path)

print("Processing complete.")

