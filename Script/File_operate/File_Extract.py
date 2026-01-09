import os
import subprocess
import sys

# ================== 用户配置区域 ==================
BASE_DIR = r"D:\New"                 # 压缩文件所在目录
PASSWORD = "SMH520125"               # 解压密码
SEVEN_ZIP_PATH = r"C:\Program Files\7-Zip\7z.exe"  # 7z.exe 路径
SUPPORTED_EXTS = (".zip", ".rar", ".7z")
# =================================================


def check_7zip():
    """检查 7z 是否存在"""
    if not os.path.exists(SEVEN_ZIP_PATH):
        print(f"[ERROR] 未找到 7z.exe：{SEVEN_ZIP_PATH}")
        print("请确认 7-Zip 是否已安装，或修改 SEVEN_ZIP_PATH")
        sys.exit(1)


def extract_archive(archive_path):
    """
    使用 7z 解压单个压缩文件
    解压目录名 = 压缩文件名（不含扩展名）
    """
    base_name = os.path.splitext(os.path.basename(archive_path))[0]
    output_dir = os.path.join(os.path.dirname(archive_path), base_name)

    # 若目录已存在，避免覆盖
    if os.path.exists(output_dir):
        print(f"[SKIP] 目录已存在，跳过：{output_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        SEVEN_ZIP_PATH,
        "x",                       # x = 保留目录结构
        archive_path,
        f"-p{PASSWORD}",           # 解压密码
        "-y",                      # 全部自动确认
        f"-o{output_dir}"          # 输出目录
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        if result.returncode == 0:
            print(f"[OK] 解压成功：{archive_path}")
        else:
            print(f"[FAIL] 解压失败：{archive_path}")
            print(result.stderr.strip())

    except Exception as e:
        print(f"[ERROR] 解压异常：{archive_path}")
        print(e)


def batch_extract():
    """遍历目录，批量解压"""
    for root, _, files in os.walk(BASE_DIR):
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTS):
                full_path = os.path.join(root, file)
                extract_archive(full_path)


if __name__ == "__main__":
    check_7zip()
    print("====== 开始批量解压 ======")
    batch_extract()
    print("====== 解压完成 ======")
