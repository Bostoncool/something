"""
并行执行 Correlation 文件夹中的 9 个相关性分析脚本。
使用 multiprocessing 多进程实现，充分利用多核 CPU。
"""
import subprocess
import sys
import os
import time
from multiprocessing import Process

# 定义要并行执行的 9 个脚本
SCRIPT_LIST = [
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


def run_script(script_name: str) -> None:
    """执行单个 Python 脚本。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)

    if not os.path.exists(script_path):
        print(f"[ERROR] 脚本 {script_name} 不存在，跳过执行！")
        return

    try:
        print(f"[START] 并行执行：{script_name} (进程ID: {os.getpid()})")
        subprocess.run(
            [sys.executable, script_path],
            cwd=current_dir,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print(f"[DONE] 脚本 {script_name} 执行完成！")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 脚本 {script_name} 执行失败，错误码：{e.returncode}")
    except Exception as e:
        print(f"[ERROR] 脚本 {script_name} 执行异常：{e}")


if __name__ == "__main__":
    print("=============== 开始并行执行所有脚本 ===============")
    start_time = time.perf_counter()

    processes = []
    for script in SCRIPT_LIST:
        p = Process(target=run_script, args=(script,))
        processes.append(p)
        p.start()
        time.sleep(0.1)  # 轻微延迟，避免日志输出混乱

    for p in processes:
        p.join()

    end_time = time.perf_counter()
    print(f"\n=============== 所有脚本并行执行完成 ===============")
    print(f"总耗时：{end_time - start_time:.2f} 秒")
