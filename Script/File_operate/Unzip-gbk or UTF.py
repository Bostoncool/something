"""
使用 7-Zip 对压缩文件进行 GBK 和 UTF-8 两种编码方式的解压。
解压后在同一文件夹下生成两份解压结果，文件夹命名与压缩文件同名（分别加 _gbk、_utf8 后缀）。
"""

import os
import subprocess
from pathlib import Path

# ================== 用户配置区域 ==================
SEVEN_ZIP_PATH = r"C:\Program Files\7-Zip\7z.exe"  # 7z.exe 路径
SUPPORTED_EXTS = (".zip", ".7z", ".rar")
# GBK 编码：代码页 936
# UTF-8 编码：代码页 65001

# 直接在此处填写要解压的路径（文件或文件夹均可）
TARGET_PATH = r'F:\1.模型要用的\2018-2023[全国省、市、县域夜间灯光数据]\2023年数据\省份层级的数据'
PASSWORD = None  # 若压缩包有密码，在此填写
RECURSIVE = False  # 若 TARGET_PATH 为文件夹，True=递归子目录，False=仅当前目录
# =================================================


def check_7zip() -> bool:
    """检查 7-Zip 是否已安装"""
    global SEVEN_ZIP_PATH
    if not os.path.exists(SEVEN_ZIP_PATH):
        alt_path = r"C:\Program Files (x86)\7-Zip\7z.exe"
        if os.path.exists(alt_path):
            SEVEN_ZIP_PATH = alt_path
            return True
        print(f"[ERROR] 未找到 7-Zip：{SEVEN_ZIP_PATH}")
        print("请确认 7-Zip 是否已安装，或修改 SEVEN_ZIP_PATH")
        return False
    return True


def unzip_dual_encoding(archive_path: str, password: str | None = None) -> None:
    """
    对单个压缩文件进行 GBK 和 UTF-8 两种方式的解压。
    在同一文件夹下生成 {basename}_gbk 和 {basename}_utf8 两个解压目录。

    Parameters:
        archive_path: 压缩文件完整路径
        password: 可选，解压密码
    """
    archive_path = os.path.abspath(archive_path)
    if not os.path.isfile(archive_path):
        print(f"[SKIP] 文件不存在：{archive_path}")
        return

    base_name = Path(archive_path).stem
    parent_dir = os.path.dirname(archive_path)

    # 两个输出目录：与压缩文件同文件夹，命名保持一致（加编码后缀）
    out_gbk = os.path.join(parent_dir, f"{base_name}_gbk")
    out_utf8 = os.path.join(parent_dir, f"{base_name}_utf8")

    pwd_args = [f"-p{password}"] if password else []

    configs = [
        ("GBK", out_gbk, "-mcp=936"),
        ("UTF-8", out_utf8, "-mcp=65001"),
    ]

    for enc_name, out_dir, mcp_arg in configs:
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            SEVEN_ZIP_PATH,
            "x",
            archive_path,
            f"-o{out_dir}",
            mcp_arg,
            "-y",
            *pwd_args,
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )

            if result.returncode == 0:
                print(f"[OK] {enc_name} 解压成功：{archive_path} -> {out_dir}")
            else:
                print(f"[FAIL] {enc_name} 解压失败：{archive_path}")
                if result.stderr.strip():
                    print(f"  错误：{result.stderr.strip()[:200]}")

        except Exception as e:
            print(f"[ERROR] {enc_name} 解压异常：{archive_path}")
            print(f"  {e}")


def batch_unzip_dual_encoding(
    source_path: str,
    password: str | None = None,
    recursive: bool = False,
) -> None:
    """
    批量对压缩文件进行 GBK 和 UTF-8 双编码解压。

    Parameters:
        source_path: 单个压缩文件路径，或包含压缩文件的目录
        password: 可选，解压密码（若压缩包有密码）
        recursive: 若为目录，是否递归处理子目录中的压缩文件
    """
    source_path = os.path.abspath(source_path)

    if os.path.isfile(source_path):
        if source_path.lower().endswith(SUPPORTED_EXTS):
            unzip_dual_encoding(source_path, password)
        else:
            print(f"[SKIP] 不支持的格式：{source_path}")
        return

    if not os.path.isdir(source_path):
        print(f"[ERROR] 路径不存在：{source_path}")
        return

    walk = os.walk(source_path) if recursive else [next(os.walk(source_path))]
    count = 0

    for root, _, files in walk:
        for f in files:
            if f.lower().endswith(SUPPORTED_EXTS):
                full_path = os.path.join(root, f)
                unzip_dual_encoding(full_path, password)
                count += 1

    print(f"\n共处理 {count} 个压缩文件。")


if __name__ == "__main__":
    if not check_7zip():
        exit(1)

    # 使用上方 TARGET_PATH 配置的路径执行解压
    if os.path.isfile(TARGET_PATH):
        unzip_dual_encoding(TARGET_PATH, PASSWORD)
    else:
        batch_unzip_dual_encoding(TARGET_PATH, PASSWORD, RECURSIVE)
