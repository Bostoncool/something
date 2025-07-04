import os
import shutil
import zipfile

def extract_ppt_images(pptx_file, target_dir):
    """
    从PPTX文件中提取图片
    
    Args:
        pptx_file: PPTX文件路径
        target_dir: 目标输出目录
    """
    # 验证输入文件是否存在
    if not os.path.exists(pptx_file):
        print(f"错误：文件不存在 - {pptx_file}")
        return False
    
    # 创建临时文件夹和目标文件夹
    temp_dir = os.path.join(os.path.dirname(pptx_file), "temp_" + os.path.basename(pptx_file))
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 1. 创建副本并修改为ZIP
        zip_file = os.path.join(temp_dir, "temp.zip")
        shutil.copy2(pptx_file, zip_file)
        
        # 2. 解压ZIP
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # 3. 查找并复制图片到目标文件夹
        media_dir = os.path.join(temp_dir, "ppt", "media")
        if os.path.exists(media_dir):
            image_count = 0
            for filename in os.listdir(media_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff')):
                    source_path = os.path.join(media_dir, filename)
                    target_path = os.path.join(target_dir, filename)
                    shutil.copy2(source_path, target_path)
                    image_count += 1
            print(f"  成功提取 {image_count} 张图片")
            return True
        else:
            print(f"  警告：在 {pptx_file} 中未找到媒体文件夹")
            return False
    
    except Exception as e:
        print(f"  错误处理 {pptx_file}: {str(e)}")
        return False
    
    finally:
        # 清理临时文件夹
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"  警告：清理临时文件夹失败: {str(e)}")

def batch_extract_images(pptx_dir, output_dir):
    """
    批量处理目录中的所有PPTX文件
    
    Args:
        pptx_dir: 包含PPTX文件的目录
        output_dir: 输出目录
    """
    # 验证输入目录是否存在
    if not os.path.exists(pptx_dir):
        print(f"错误：输入目录不存在 - {pptx_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计信息
    total_files = 0
    processed_files = 0
    failed_files = 0
    
    # 遍历目录中的所有pptx文件
    for root, dirs, files in os.walk(pptx_dir):
        for file in files:
            if file.lower().endswith('.pptx'):
                total_files += 1
                file_path = os.path.join(root, file)
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                target_subdir = os.path.join(output_dir, base_name)
                
                print(f"处理 {file}...")
                if extract_ppt_images(file_path, target_subdir):
                    processed_files += 1
                else:
                    failed_files += 1
    
    # 输出统计信息
    print(f"\n处理完成！")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {processed_files}")
    print(f"处理失败: {failed_files}")

def main():
    """主函数"""
    print("PPTX图片提取工具")
    print("=" * 50)
    
    # 获取用户输入
    print("请输入包含PPTX文件的目录路径（或按回车使用默认路径）:")
    pptx_directory = input().strip()
    
    if not pptx_directory:
        # 使用当前目录作为默认路径
        pptx_directory = os.getcwd()
        print(f"使用当前目录: {pptx_directory}")
    
    print("请输入输出目录路径（或按回车使用默认路径）:")
    output_directory = input().strip()
    
    if not output_directory:
        # 使用当前目录下的output文件夹作为默认输出路径
        output_directory = os.path.join(os.getcwd(), "extracted_images")
        print(f"使用默认输出目录: {output_directory}")
    
    # 验证输入目录
    if not os.path.exists(pptx_directory):
        print(f"错误：指定的目录不存在 - {pptx_directory}")
        return
    
    # 开始处理
    print(f"\n开始处理...")
    print(f"输入目录: {pptx_directory}")
    print(f"输出目录: {output_directory}")
    print("-" * 50)
    
    batch_extract_images(pptx_directory, output_directory)

if __name__ == "__main__":
    main()