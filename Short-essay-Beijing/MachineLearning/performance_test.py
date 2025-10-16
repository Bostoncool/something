"""
Parallel Performance Test Script
Used to verify the effect of parallel optimization
"""

import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Get CPU information
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

print("=" * 80)
print("Parallel Performance Test")
print("=" * 80)
print(f"\nCPU core count: {CPU_COUNT}")
print(f"Parallel worker threads: {MAX_WORKERS}")

# Simulate data processing task
def process_task(task_id):
    """Simulate a time-consuming task (compute-intensive)"""
    # Simulate complex computation
    result = 0
    for i in range(1000000):
        result += np.sin(i) * np.cos(i)
    return task_id, result

# Test 1: Serial processing
print("\n" + "-" * 80)
print("Test 1: Serial processing (1 thread)")
print("-" * 80)
tasks = list(range(20))
start_time = time.time()

serial_results = []
for task_id in tasks:
    result = process_task(task_id)
    serial_results.append(result)

serial_time = time.time() - start_time
print(f"Completion time: {serial_time:.2f} seconds")
print(f"Average per task: {serial_time/len(tasks):.3f} seconds")

# Test 2: Parallel processing (4 threads)
print("\n" + "-" * 80)
print("Test 2: Parallel processing (4 threads)")
print("-" * 80)
start_time = time.time()

parallel_results_4 = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_task, task_id) for task_id in tasks]
    for future in as_completed(futures):
        result = future.result()
        parallel_results_4.append(result)

parallel_time_4 = time.time() - start_time
print(f"Completion time: {parallel_time_4:.2f} seconds")
print(f"Average per task: {parallel_time_4/len(tasks):.3f} seconds")
print(f"Speedup: {serial_time/parallel_time_4:.2f}x")

# Test 3: Parallel processing (maximum threads)
print("\n" + "-" * 80)
print(f"Test 3: Parallel processing ({MAX_WORKERS} threads)")
print("-" * 80)
start_time = time.time()

parallel_results_max = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_task, task_id) for task_id in tasks]
    for future in as_completed(futures):
        result = future.result()
        parallel_results_max.append(result)

parallel_time_max = time.time() - start_time
print(f"Completion time: {parallel_time_max:.2f} seconds")
print(f"Average per task: {parallel_time_max/len(tasks):.3f} seconds")
print(f"Speedup: {serial_time/parallel_time_max:.2f}x")

# Performance summary
print("\n" + "=" * 80)
print("Performance Comparison Summary")
print("=" * 80)
print(f"\n{'Method':<20} {'Time(s)':<15} {'Speedup':<10}")
print("-" * 45)
print(f"{'Serial (1 thread)':<20} {serial_time:<15.2f} {1.0:<10.2f}")
print(f"{'Parallel (4 threads)':<20} {parallel_time_4:<15.2f} {serial_time/parallel_time_4:<10.2f}")
print(f"{'Parallel (' + str(MAX_WORKERS) + ' threads)':<20} {parallel_time_max:<15.2f} {serial_time/parallel_time_max:<10.2f}")

# Efficiency analysis
print("\n" + "=" * 80)
print("Parallel Efficiency Analysis")
print("=" * 80)
ideal_speedup_4 = 4.0
ideal_speedup_max = MAX_WORKERS
actual_speedup_4 = serial_time / parallel_time_4
actual_speedup_max = serial_time / parallel_time_max

efficiency_4 = (actual_speedup_4 / ideal_speedup_4) * 100
efficiency_max = (actual_speedup_max / ideal_speedup_max) * 100

print(f"\n4-thread parallel efficiency: {efficiency_4:.1f}% (ideal 100%)")
print(f"{MAX_WORKERS}-thread parallel efficiency: {efficiency_max:.1f}% (ideal 100%)")

if efficiency_max > 80:
    print("\n✅ Excellent parallel efficiency! System fully utilizes multi-core CPU.")
elif efficiency_max > 60:
    print("\n✓ Good parallel efficiency, with some thread overhead.")
else:
    print("\n⚠️ Low parallel efficiency, may be affected by GIL or other factors.")

# Recommendations
print("\n" + "=" * 80)
print("Optimization Recommendations")
print("=" * 80)
print(f"\nBased on test results:")
print(f"- Your system has {CPU_COUNT} CPU cores")
print(f"- Recommended to use {MAX_WORKERS} worker threads")
print(f"- Expected performance improvement: {serial_time/parallel_time_max:.1f}x")

if CPU_COUNT >= 8:
    print(f"\n💡 Note: You have {CPU_COUNT} CPU cores, excellent for large-scale parallel processing!")
    print("   - Data loading will be significantly faster")
    print("   - Grid search will be much faster")
    print("   - Recommended to enable tqdm progress bar: pip install tqdm")
elif CPU_COUNT >= 4:
    print(f"\n💡 Note: You have {CPU_COUNT} CPU cores, parallel processing is significantly effective.")
    print("   - Recommended to use parallel data loading")
    print("   - Grid search will be noticeably faster")
else:
    print(f"\n💡 Note: You have {CPU_COUNT} CPU cores, parallel processing still helps.")
    print("   - Recommended to use 4 worker threads")
    print("   - Pay attention to memory usage")

print("\n" + "=" * 80)
print("Test completed!")
print("=" * 80)

