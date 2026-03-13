"""
批量执行 Correlation 文件夹中的 9 个相关性分析脚本。
"""
import subprocess
import sys
import os

# 定义要执行的 9 个脚本名称（按区域和方法顺序排列）
script_list = [
    "BTH-Spearman.py",
    "BTH-Mutual-Info.py",
    "BTH-Geo-detector.py",
    "YRD-Spearman.py",
    "YRD-Mutual-Info.py",
    "YRD-Geo-detector.py",
    "PRD-Spearman.py",
    "PRD-Mutual-Info.py",
    "PRD-Geo-detector.py",
]

# 获取当前脚本所在目录（确保能找到其他脚本）
current_dir = os.path.dirname(os.path.abspath(__file__))

# 遍历并执行每个脚本
for idx, script in enumerate(script_list, 1):
    script_path = os.path.join(current_dir, script)

    # 检查脚本文件是否存在
    if not os.path.exists(script_path):
        print(f"❌ 第{idx}个脚本 {script} 不存在，跳过！")
        continue

    try:
        print(f"\n🚀 开始执行第{idx}个脚本：{script}")
        # 执行脚本（stdout和stderr实时输出）
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=current_dir,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print(f"✅ 第{idx}个脚本 {script} 执行完成！")

    except subprocess.CalledProcessError as e:
        print(f"❌ 第{idx}个脚本 {script} 执行失败，错误码：{e.returncode}")
        # 可选：如果某个脚本失败就停止全部执行
        # sys.exit(1)

print("\n🎉 所有脚本执行流程结束！")
