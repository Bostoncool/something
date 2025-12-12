import os
import zipfile
from pathlib import Path


ROOT_PATH = Path(r'G:\2000-2023[PM2.5-china]\Day') 


def zip_first_level_dirs(root_dir: Path):
    root_dir = root_dir.resolve()
    if not root_dir.is_dir():
        print(f"❌ 路径不存在：{root_dir}")
        input("按回车退出...")
        return

    for item in root_dir.iterdir():
        if not item.is_dir() or item.name.startswith('.'):
            continue

        zip_path = item.with_suffix('.zip')
        if zip_path.exists():
            print(f"⚠️  已存在，跳过：{zip_path.name}")
            continue

        print(f"📦 正在压缩：{item.name}")
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in item.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(item)
                        zf.write(file_path, arcname)
            print(f"✅ 完成：{zip_path.name}")
        except Exception as e:
            print(f"❌ 压缩失败：{item} -> {e}")

    print("全部处理完毕！")
    input("按回车退出...")

if __name__ == '__main__':
    zip_first_level_dirs(ROOT_PATH)