import os

# 在某个路径下批量生成指定名称文件夹

def create_folders(base_path):
    for year in range(1, 11):
        folder_name = str(year)
        folder_path = os.path.join(base_path, folder_name)
        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created folder: {folder_path}")
        except Exception as e:
            print(f"Failed to create folder {folder_path}: {e}")

# 使用你想要的路径替换 'your/base/path'
create_folders('E:\DATA Science\Result\XGBOOST')
