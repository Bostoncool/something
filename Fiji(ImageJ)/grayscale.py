from ij import IJ, ImagePlus
from ij.io import DirectoryChooser
import os

def batch_process_images(folder_path, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹中的所有图像文件
    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png', '.bmp'))]

    # 处理每个图像文件
    for file_name in file_list:
        try:
            file_path = os.path.join(folder_path, file_name)
            # 打开图像
            imp = IJ.openImage(file_path)
            if imp is not None:
                # 转换为灰度（使用正确的命令名称）
                IJ.run(imp, "8-bit", "")
                # 保存处理后的图像
                output_path = os.path.join(output_folder, file_name)
                IJ.saveAs(imp, "Tiff", output_path)
                print("Processed and saved: " + output_path)
            else:
                print("Failed to open image: " + file_path)
        except Exception as e:
            print("Error processing %s: %s" % (file_name, str(e)))

def main():
    try:
        # 选择输入文件夹
        input_folder = DirectoryChooser("Choose input folder").getDirectory()
        if not input_folder:
            print("No input folder selected. Exiting.")
            return

        # 选择输出文件夹
        output_folder = DirectoryChooser("Choose output folder").getDirectory()
        if not output_folder:
            print("No output folder selected. Exiting.")
            return

        # 批量处理图像
        batch_process_images(input_folder, output_folder)
    except Exception as e:
        print("An error occurred: %s" % str(e))

if __name__ == "__main__":
    main()