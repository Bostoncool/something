"""
并行性能测试脚本
用于验证并行优化的效果
"""

import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# 获取CPU信息
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

print("=" * 80)
print("并行性能测试")
print("=" * 80)
print(f"\nCPU核心数: {CPU_COUNT}")
print(f"并行工作线程: {MAX_WORKERS}")

# 模拟数据处理任务
def process_task(task_id):
    """模拟一个耗时任务（计算密集型）"""
    # 模拟复杂计算
    result = 0
    for i in range(1000000):
        result += np.sin(i) * np.cos(i)
    return task_id, result

# 测试1: 串行处理
print("\n" + "-" * 80)
print("测试1: 串行处理（1个线程）")
print("-" * 80)
tasks = list(range(20))
start_time = time.time()

serial_results = []
for task_id in tasks:
    result = process_task(task_id)
    serial_results.append(result)

serial_time = time.time() - start_time
print(f"完成时间: {serial_time:.2f} 秒")
print(f"每任务平均: {serial_time/len(tasks):.3f} 秒")

# 测试2: 并行处理（4线程）
print("\n" + "-" * 80)
print("测试2: 并行处理（4个线程）")
print("-" * 80)
start_time = time.time()

parallel_results_4 = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_task, task_id) for task_id in tasks]
    for future in as_completed(futures):
        result = future.result()
        parallel_results_4.append(result)

parallel_time_4 = time.time() - start_time
print(f"完成时间: {parallel_time_4:.2f} 秒")
print(f"每任务平均: {parallel_time_4/len(tasks):.3f} 秒")
print(f"加速比: {serial_time/parallel_time_4:.2f}x")

# 测试3: 并行处理（最大线程）
print("\n" + "-" * 80)
print(f"测试3: 并行处理（{MAX_WORKERS}个线程）")
print("-" * 80)
start_time = time.time()

parallel_results_max = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_task, task_id) for task_id in tasks]
    for future in as_completed(futures):
        result = future.result()
        parallel_results_max.append(result)

parallel_time_max = time.time() - start_time
print(f"完成时间: {parallel_time_max:.2f} 秒")
print(f"每任务平均: {parallel_time_max/len(tasks):.3f} 秒")
print(f"加速比: {serial_time/parallel_time_max:.2f}x")

# 性能总结
print("\n" + "=" * 80)
print("性能对比总结")
print("=" * 80)
print(f"\n{'方案':<20} {'时间(秒)':<15} {'加速比':<10}")
print("-" * 45)
print(f"{'串行(1线程)':<20} {serial_time:<15.2f} {1.0:<10.2f}")
print(f"{'并行(4线程)':<20} {parallel_time_4:<15.2f} {serial_time/parallel_time_4:<10.2f}")
print(f"{'并行({MAX_WORKERS}线程)':<20} {parallel_time_max:<15.2f} {serial_time/parallel_time_max:<10.2f}")

# 效率分析
print("\n" + "=" * 80)
print("并行效率分析")
print("=" * 80)
ideal_speedup_4 = 4.0
ideal_speedup_max = MAX_WORKERS
actual_speedup_4 = serial_time / parallel_time_4
actual_speedup_max = serial_time / parallel_time_max

efficiency_4 = (actual_speedup_4 / ideal_speedup_4) * 100
efficiency_max = (actual_speedup_max / ideal_speedup_max) * 100

print(f"\n4线程并行效率: {efficiency_4:.1f}% (理想100%)")
print(f"{MAX_WORKERS}线程并行效率: {efficiency_max:.1f}% (理想100%)")

if efficiency_max > 80:
    print("\n✅ 并行效率优秀！系统充分利用了多核CPU。")
elif efficiency_max > 60:
    print("\n✓ 并行效率良好，有一定的线程开销。")
else:
    print("\n⚠️ 并行效率较低，可能受到GIL或其他因素影响。")

# 建议
print("\n" + "=" * 80)
print("优化建议")
print("=" * 80)
print(f"\n根据测试结果:")
print(f"- 您的系统有 {CPU_COUNT} 个CPU核心")
print(f"- 推荐使用 {MAX_WORKERS} 个工作线程")
print(f"- 预期性能提升: {serial_time/parallel_time_max:.1f}倍")

if CPU_COUNT >= 8:
    print(f"\n💡 提示: 您有 {CPU_COUNT} 核CPU，非常适合大规模并行处理！")
    print("   - 数据加载将显著加速")
    print("   - 网格搜索将大幅提速")
    print("   - 建议启用tqdm进度条: pip install tqdm")
elif CPU_COUNT >= 4:
    print(f"\n💡 提示: 您有 {CPU_COUNT} 核CPU，并行处理效果显著。")
    print("   - 推荐使用并行数据加载")
    print("   - 网格搜索将明显加速")
else:
    print(f"\n💡 提示: 您有 {CPU_COUNT} 核CPU，并行处理仍有帮助。")
    print("   - 建议使用4个工作线程")
    print("   - 注意内存使用")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)

