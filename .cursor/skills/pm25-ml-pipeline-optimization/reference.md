# PM2.5 读取性能优化详细参考

## 一、问题背景

使用 `RF.py` 训练时，读取 6000+ 个 PM2.5 NC 文件约需 **20 分钟**，速度过慢。

---

## 二、架构与瓶颈分析

### 2.1 架构说明

- `RF.py` 为入口脚本，PM2.5 读取逻辑在 `daily_ml_pipeline.py` 的 `load_pm25_daily_from_nc` 调用链。
- 数据目录：6573 个日尺度 NC 文件，按 BTH/YRD/PRD 三套城市群组织，单文件约 0.5MB–2MB。

### 2.2 主要瓶颈

#### （1）每个文件重复执行 GeoPandas 空间连接（最大瓶颈）

- 每个 NC 文件读取后调用 `_extract_city_mean_from_2d_field`（位于 `Correlation/BTH-Geo-detector.py`，YRD/PRD 复用）。
- 该函数对每个文件执行：
  - 将网格点转为 GeoDataFrame
  - 使用 `gpd.sjoin` 做空间连接，判断每个网格点属于哪个城市
  - 按城市聚合求 PM2.5 均值
- **问题**：BTH/YRD/PRD 三套数据的网格结构相同，但代码对 6000+ 个文件都重复执行相同的 `sjoin`，相当于把同一套「网格→城市」映射计算了 6000+ 次。

#### （2）每次使用全部城市多边形参与计算

- 每个 worker 加载 BTH+YRD+PRD 全部城市的 GeoDataFrame。
- 单个 NC 文件实际只对应一个区域（如 BTH），却与全部城市多边形做空间索引和匹配。
- **问题**：空间索引规模大，`sjoin` 和 R-tree 构建开销不必要地增加。

#### （3）多进程调度与 IPC 开销大

- 采用「一个文件一个任务」的细粒度提交方式。
- 6000+ 个 Future 需要调度，每个任务返回一个小 DataFrame。
- **问题**：主进程需接收 6000+ 次 pickle 反序列化结果，调度和 IPC 开销随任务数线性增长。

#### （4）NetCDF 解码成本偏高

- `xr.open_dataset` 默认 `decode_times=True` 和 `decode_cf=True`，对每个文件做时间解码和 CF 元数据解析。
- 文件名已包含日期（如 `CHAP_PM2.5_D1K_20180101_V4.nc`），很多场景下可跳过这些解码。
- **问题**：单文件打开时间偏长，累积后影响整体吞吐。

---

## 三、优化方案实现细节

### 3.1 网格→城市映射缓存

**思路**：同一区域、同一网格结构只需计算一次「网格点→城市」映射。

**实现**：
- 在 worker 内维护全局缓存 `_PM25_GRID_CITY_CACHE`。
- 缓存 key：`(区域名, lat 签名, lon 签名, 城市边界, 城市数)`。
- 首次遇到某网格时执行一次 `gpd.sjoin`，缓存 `point_index -> city_index` 映射。
- 后续同网格文件直接用 `numpy.bincount` 做 sum/count 聚合，跳过 `sjoin`。
- 若缓存构建失败，自动回退到原有 `_extract_city_mean_from_2d_field`。

**效果**：6000+ 次 `sjoin` 降为约 3 次（BTH/YRD/PRD 各一次）。

### 3.2 按区域使用城市子集

**实现**：
- 新增 `_infer_region_from_path`，从路径中识别 BTH/YRD/PRD。
- worker 初始化时预构建 `region_city_gdf`，按区域拆分城市 GeoDataFrame。
- `_extract_city_mean_with_grid_cache` 根据 `region_name` 选择对应区域的城市子集。

### 3.3 批量任务（chunk）降低调度与 IPC 开销

**实现**：
- 新增 `_build_chunked_paths`，将文件列表按 8~32 个一组打包为 chunk。
- 新增 `_read_pm25_nc_chunk_worker`，每个任务处理一个 chunk，内部循环读取多个文件。
- `iter_bounded_executor_results` 改为按 chunk 提交任务。

**效果**：6000+ 个 Future 降为数百个。

### 3.4 NetCDF 快速解码路径

**实现**：
- `read_nc_with_fallback` 增加 `prefer_fast_decode` 参数。
- 优先尝试 `decode_times=False`、`decode_cf=False`，失败则回退到完整解码。

### 3.5 Parquet 缓存

- 首次读取后将 PM2.5 结果写入 parquet 缓存。
- 二次同参数运行时直接读 parquet，跳过 NC 读取和空间聚合。
- **效果**：热启动时耗时从分钟级降至秒级。

---

## 四、验证结果（2023 年 BTH 子目录，365 个文件）

| 指标 | 冷启动 | 热启动（命中 parquet 缓存） |
|------|--------|----------------------------|
| pm25_seconds | ~5 s | ~0.01 s |
| files_per_second | ~75 | - |
| cache_hit_pm25 | False | True |
| cache_hits / cache_misses | 349 / 16 | - |

- `rows_equal`: True  
- `keys_equal`: True  
- `max_abs_diff`: 0.0  
- 无重复键，城市数、日期范围正确。
