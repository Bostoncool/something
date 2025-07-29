import os
import subprocess
from pathlib import Path

def extract_archives(source_dir):
    """
    遍历目录中的压缩文件并解压到同名文件夹
    批量解压一个文件夹内的所有压缩文件
    Parameters:
        source_dir: str, 需要处理的目录路径
    """
    # 7-Zip程序的默认安装路径
    seven_zip_path = r"C:\Program Files\7-Zip\7z.exe"
    
    # 检查7-Zip是否已安装
    if not os.path.exists(seven_zip_path):
        raise FileNotFoundError("未找到7-Zip程序，请确认是否已安装或路径是否正确")
    
    # 支持的压缩文件扩展名
    archive_extensions = ('.zip', '.7z', '.rar')
    
    # 遍历目录
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(archive_extensions):
                archive_path = os.path.join(root, file)
                # 创建解压目标文件夹（使用压缩文件的名称，不包含扩展名）
                extract_folder = os.path.join(root, Path(file).stem)
                
                # 如果目标文件夹不存在，创建它
                if not os.path.exists(extract_folder):
                    os.makedirs(extract_folder)
                
                try:
                    # 构建7z解压命令
                    cmd = [
                        seven_zip_path,
                        'x',  # 解压命令
                        f'{archive_path}',  # 压缩文件路径
                        f'-o{extract_folder}',  # 输出目录
                        '-y'  # 自动回答yes
                    ]
                    
                    # 执行解压命令
                    process = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if process.returncode == 0:
                        print(f"成功解压: {file} -> {extract_folder}")
                    else:
                        print(f"解压失败: {file}")
                        print(f"错误信息: {process.stderr}")
                        
                except Exception as e:
                    print(f"处理 {file} 时发生错误: {str(e)}")

if __name__ == "__main__":
    # 指定要处理的目录路径
    source_directory = r"D:\ArcGIS\矢量地图（练习时下载这个）全国省级、地市级、县市级行政区划shp"  # 替换为你的目录路径
    
    try:
        extract_archives(source_directory)
        print("所有文件处理完成！")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
