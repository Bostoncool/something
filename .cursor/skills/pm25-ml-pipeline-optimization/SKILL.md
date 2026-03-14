---
name: pm25-ml-pipeline-optimization
description: >-
  Optimizes PM2.5 NetCDF reading and ML pipeline performance. Covers grid-to-city
  mapping cache, chunked batch processing, parquet caching, and development
  efficiency guidelines. Use when working with PM2.5 NC files, RF training,
  daily_ml_pipeline, GeoPandas spatial joins, or when optimizing data loading
  for thousands of files.
---

# PM2.5 ML 管道性能优化与开发省时指南

本技能整合 PM2.5 NetCDF 读取优化经验与通用开发效率原则，用于加速数据科学/机器学习流水线开发。

---

## 一、开发省时原则（优先遵循）

1. **资源全用上**：显卡、多线程、串行/并行结合，能并行的尽量并行。
2. **数据读取加速**：数千个资源应在 10 秒内完成读取，减少调度往返。
3. **缓存优先**：数据种类繁多时，使用缓存（如 parquet），命中缓存后直接进入模型训练。
4. **需求说详细**：修改或迭代代码时，需求描述越详细，改动越快、越精准。
5. **阶梯调参**：按显存和吞吐逐步放大 `batch-size` / `eval-batch-size` / `workers`，提高 GPU 利用率。
6. **避免资源抢占**：避免线程死锁、多线程切换开销导致整体变慢。
7. **进度可见**：使用 `tqdm` 显示进度和预估时间。
8. **预遍历路径**：批处理前先遍历所有文件，将路径存入临时字典，运行时直接寻址，避免每次执行完一个再重新遍历。

---

## 二、PM2.5 NetCDF 读取优化要点

### 2.1 核心瓶颈（避免重复）

| 瓶颈 | 问题 | 应对 |
|------|------|------|
| 重复空间连接 | 每个 NC 文件都执行 `gpd.sjoin`，6000+ 次相同计算 | 网格→城市映射缓存 |
| 全量城市参与 | 单文件只对应 BTH/YRD/PRD 之一，却加载全部城市 | 按区域使用城市子集 |
| 细粒度任务 | 一文件一 Future，调度与 IPC 开销大 | 批量 chunk（8~32 个/组） |
| NetCDF 解码 | 默认 `decode_times=True` 等，单文件打开慢 | `decode_times=False` 快速路径 |

### 2.2 优化策略速查

- **网格→城市映射缓存**：同一区域、同一网格结构只计算一次 `sjoin`，后续用 `numpy.bincount` 聚合。
- **区域推断**：从路径识别 BTH/YRD/PRD，只与对应区域城市做匹配。
- **Chunk 任务**：多文件打包为一个任务，减少 Future 数量和 pickle 反序列化次数。
- **Parquet 缓存**：首次读取后写入 parquet，二次运行直接读缓存，跳过 NC 读取。

### 2.3 涉及文件

- `daily_ml_pipeline.py`：网格缓存、区域子集、chunk 任务、快速解码、性能统计
- `cluster_training_utils.py`：合并 `prepare_stats` 时支持新性能字段
- `RF.py`：`run_info.json` 中增加 PM2.5 细粒度指标

---

## 三、性能可观测性

在 `load_pm25_daily_from_nc` 中统计并透传：

- `open_dataset_time`、`spatial_aggregate_time`
- `cache_build_hits` / `cache_build_misses`
- `files_per_second`、`processed_files`、`failed_files`
- `pm25_seconds`、`cache_hit_pm25`

写入 `run_info.json`，便于复盘和调优。

---

## 四、tqdm 使用要点

- 用 `desc` 添加描述性文本。
- 嵌套循环时内层用 `leave=False`。
- 循环内打印用 `tqdm.write()`，避免破坏进度条。
- 大批量时适当调大 `mininterval` / `maxinterval` 减少刷新开销。

---

## 五、后续调优建议

- 全量 6000+ 文件时关注 `run_info.json` 中的 `pm25_seconds`、`pm25_files_per_second`、`pm25_grid_cache_hits/misses`。
- 若磁盘 I/O 成瓶颈：调整 `--pm25-workers`、增大 chunk 大小，或使用 SSD。
- 首次运行后尽量利用 parquet 缓存，避免重复冷启动。

---

## 六、详细参考

完整 PM2.5 优化背景、问题分析、实现细节与验证结果见 [reference.md](reference.md)。
