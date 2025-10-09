# -*- coding: utf-8 -*-
import os
import glob

root = r'C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV'

# 获取所有子文件夹
folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
print(f'子文件夹数量: {len(folders)}')
print(f'\n所有文件夹:')
for f in sorted(folders):
    print(f'  - {f}')

# 检查一个月份的所有文件
print(f'\n\n查找2015年1月的所有文件:')
files_201501 = glob.glob(os.path.join(root, '**', '*201501*.csv'), recursive=True)
print(f'找到 {len(files_201501)} 个文件')
for f in sorted(files_201501)[:10]:
    print(f'  - {os.path.basename(os.path.dirname(f))} / {os.path.basename(f)}')

