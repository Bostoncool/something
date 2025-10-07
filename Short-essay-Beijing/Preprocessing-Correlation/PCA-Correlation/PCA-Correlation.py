# =======================
# 环境、路径、并行读取工具
# =======================
import os
import re
import gc
import math
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ---- 路径（请按需修改）----
ROOT_POLL_ALL  = r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)"
ROOT_POLL_EXTRA= r"C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)"
ROOT_ERA5      = r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV"

# 日期范围
DATE_START = "2015-01-01"
DATE_END   = "2024-12-31"

# ERA5 北京范围（按你给的经纬度）
LON_MIN, LON_MAX = 115.25, 117.50  # 经度
LAT_MIN, LAT_MAX = 39.43, 41.05    # 纬度

# Matplotlib/中文/期刊风格
mpl.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "Arial"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 140
mpl.rcParams["savefig.dpi"] = 140
mpl.rcParams["font.size"] = 11
sns.set_context("talk")
# Nature风格倾向：留白、简洁、网格弱化
sns.set_style("white")
NATURE_CMAP = sns.color_palette("rocket", as_cmap=True)  # 简洁高对比但不过饱和

def _try_read_csv(path, **kwargs):
    """解决Unicode中英编码兼容的健壮读取器：优先utf-8，其次gbk，再尝试latin1。"""
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err

def list_files_by_regex(folder, pattern=r".*\.csv$"):
    """列出目录下匹配正则的CSV文件（递归搜索子目录）。"""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([str(p) for p in folder.rglob("*.csv") if re.match(pattern, p.name)])

def read_csv_parallel(file_list, parse_dates=None, usecols=None, dtype=None):
    """并行读取一批CSV，自动忽略损坏文件。"""
    dfs = []
    if not file_list:
        return dfs
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        futures = []
        for fp in file_list:
            futures.append(ex.submit(_try_read_csv, fp, parse_dates=parse_dates, usecols=usecols, dtype=dtype))
        for fut in as_completed(futures):
            try:
                df = fut.result()
                if df is not None and len(df) > 0:
                    dfs.append(df)
            except Exception:
                # 某些坏文件直接跳过
                pass
    return dfs

def robust_numeric(df):
    """将可转为数值的列转为float，其余保持原状。"""
    for c in df.columns:
        if df[c].dtype == object:
            # 尝试转数值
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def clip_outliers(df, cols, z=4.0):
    """按Z分数裁剪极值/异常，提升稳定性。"""
    for c in cols:
        if c in df.columns:
            series = pd.to_numeric(df[c], errors="coerce")
            mu, sd = series.mean(), series.std()
            if sd and not math.isclose(sd, 0.0):
                lo, hi = mu - z*sd, mu + z*sd
                df[c] = series.clip(lower=lo, upper=hi)
            else:
                df[c] = series
    return df

def assert_single_date_index(df, date_col="date"):
    """确保日期为DatetimeIndex（日尺度），并去重聚合。"""
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        # 兜底：尝试从可能的列名中识别
        for cand in ["time", "Time", "date", "Date", "日期"]:
            if cand in df.columns:
                df[cand] = pd.to_datetime(df[cand])
                df = df.set_index(cand)
                break
    df = df.sort_index()
    # 统一到逐日
    df = df.groupby(pd.Grouper(freq="1D")).mean(numeric_only=True)
    return df.loc[DATE_START:DATE_END]
# =======================
# 读取与清洗：污染数据
# =======================
# 文件名样式：
#   beijing_all_YYYYMMDD.csv  （含 PM2.5 / PM10 / 也可能混入AQI）
#   beijing_extra_YYYYMMDD.csv（含 SO2 / NO2 / CO / O3）
# 切记：排除任何 AQI 相关字段
AIM_POLL_COLS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]  # 只保留这6类

# 列出文件
files_all   = list_files_by_regex(ROOT_POLL_ALL,   r"^beijing_all_\d{8}\.csv$")
files_extra = list_files_by_regex(ROOT_POLL_EXTRA, r"^beijing_extra_\d{8}\.csv$")

print(f"Found {len(files_all)} pollution all files")
print(f"Found {len(files_extra)} pollution extra files")
if files_all:
    print(f"First pollution all file: {files_all[0]}")
if files_extra:
    print(f"First pollution extra file: {files_extra[0]}")

dfs_poll_all   = read_csv_parallel(files_all,   parse_dates=True)
dfs_poll_extra = read_csv_parallel(files_extra, parse_dates=True)

print(f"Loaded {len(dfs_poll_all)} pollution all dataframes")
print(f"Loaded {len(dfs_poll_extra)} pollution extra dataframes")

def clean_pollution_block(dfs):
    if not dfs:
        print("No pollution dataframes to process")
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    print(f"Concatenated pollution data shape: {df.shape}")
    print(f"Concatenated pollution columns count: {len(df.columns)}")
    df = robust_numeric(df)

    # 检查数据格式 - 如果是长格式（有type列），需要转换
    if 'type' in df.columns:
        print("Detected long format data, converting to wide format...")
        
        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        # 只保留需要的污染物类型
        pollution_types = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        df = df[df['type'].isin(pollution_types)]
        
        # 获取站点列（除了date, hour, type之外的所有列）
        station_cols = [c for c in df.columns if c not in ['date', 'hour', 'type']]
        
        # 按日期和污染物类型分组，计算所有站点的平均值
        df_daily = df.groupby(['date', 'type'])[station_cols].mean().reset_index()
        
        # 简化的转换方法：直接按日期和类型分组，计算所有站点的平均值
        result_df = pd.DataFrame()
        for poll_type in pollution_types:
            poll_data = df[df['type'] == poll_type]
            if not poll_data.empty:
                # 计算所有站点的平均值
                daily_avg = poll_data.groupby('date')[station_cols].mean().mean(axis=1)
                result_df[poll_type] = daily_avg
                print(f"Processed {poll_type}: {len(daily_avg)} days")
        
        print(f"Result dataframe shape: {result_df.shape}")
        print(f"Result dataframe columns: {result_df.columns.tolist()}")
        df = result_df
        print(f"After conversion, df shape: {df.shape}, columns: {df.columns.tolist()}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
    else:
        # 原有的宽格式处理逻辑
        date_col = None
        for cand in ["date", "日期", "time", "Time"]:
            if cand in df.columns:
                date_col = cand
                break
        if date_col is None:
            date_col = df.columns[0]

        # 去除 AQI 相关列
        drop_like = [c for c in df.columns if re.search(r"\bAQI\b", c, flags=re.I)]
        df = df.drop(columns=drop_like, errors="ignore")

        # 更灵活的列名匹配
        pollution_cols = []
        for c in df.columns:
            c_clean = c.upper().replace("-", "").replace("_", "").replace(".", "").replace(" ", "")
            if c_clean in ["PM25", "PM10", "SO2", "NO2", "CO", "O3"]:
                pollution_cols.append(c)
        
        print(f"Found pollution columns: {pollution_cols}")
        
        keep_cols = [date_col] + pollution_cols
        if len(keep_cols) > 1:
            df = df[keep_cols]
        else:
            print("Warning: No pollution columns found, keeping all columns")
            df = df[[date_col]] if date_col else df

        # 将显然不合理的负值设为缺失
        for c in df.columns:
            if c != date_col:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                df.loc[df[c] < 0, c] = np.nan

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

    # 逐日频率 - 跳过这个步骤，因为数据已经是日频率的
    print(f"Before daily grouping: {df.shape}")
    # df = df.groupby(pd.Grouper(freq="1D")).mean(numeric_only=True)
    # print(f"After daily grouping: {df.shape}")
    
    # 裁剪极值，减少噪声
    df = clip_outliers(df, [c for c in df.columns if c in AIM_POLL_COLS], z=4.0)
    # 只保留目标列（缺哪个算哪个）
    available_cols = [c for c in AIM_POLL_COLS if c in df.columns]
    if available_cols:
        df = df[available_cols]
    
    # 日期过滤
    try:
        df = df.loc[DATE_START:DATE_END]
        print(f"After date filtering: {df.shape}")
    except Exception as e:
        print(f"Date filtering failed: {e}")
        print(f"Available date range: {df.index.min()} to {df.index.max()}")
    
    print(f"Final pollution data shape: {df.shape}, columns: {df.columns.tolist()}")
    return df

poll_all   = clean_pollution_block(dfs_poll_all)
poll_extra = clean_pollution_block(dfs_poll_extra)

# 合并为6项污染物（逐日）
print(f"poll_all shape: {poll_all.shape}, columns: {poll_all.columns.tolist()}")
print(f"poll_extra shape: {poll_extra.shape}, columns: {poll_extra.columns.tolist()}")

pollution_daily = poll_all.join(poll_extra, how="outer")
print(f"After joining pollution data: {pollution_daily.shape}")
print(f"Available columns: {pollution_daily.columns.tolist()}")

pollution_daily = pollution_daily[[c for c in AIM_POLL_COLS if c in pollution_daily.columns]]
print(f"After filtering to target columns: {pollution_daily.shape}")

# 适度插值填补（最多连续3天）
pollution_daily = pollution_daily.sort_index().interpolate(limit=3).loc[DATE_START:DATE_END]
print(f"After interpolation and date filtering: {pollution_daily.shape}")

# =======================
# 读取与清洗：ERA5 数据
# =======================
# ERA5 文件名：YYYYMM.csv，含经纬度与逐日（或逐时）值
era5_files = list_files_by_regex(ROOT_ERA5, r"^\d{6}\.csv$")
dfs_era5 = read_csv_parallel(era5_files, parse_dates=True)

ERA5_KEEP = {
    "d2m":"2m_dewpoint_temperature",
    "t2m":"2m_temperature",
    "u10":"10m_u_component_of_wind",
    "v10":"10m_v_component_of_wind",
    "u100":"100m_u_component_of_wind",
    "v100":"100m_v_component_of_wind",
    "blh":"boundary_layer_height",
    "cvh":"high_vegetation_cover",
    "lsm":"land_sea_mask",
    "cvl":"low_vegetation_cover",
    "avg_tprate":"mean_total_precipitation_rate",
    "mn2t":"minimum_2m_temperature_since_previous_post_processing",
    "sd":"snow_depth",
    "str":"surface_net_thermal_radiation",
    "sp":"surface_pressure",
    "tisr":"toa_incident_solar_radiation",
    "tcwv":"total_column_water_vapour",
    "tp":"total_precipitation"
}

def clean_era5_block(dfs):
    if not dfs:
        return pd.DataFrame()
    
    # 限制处理的文件数量以避免内存问题
    max_files = 12  # 最多处理12个月的数据
    if len(dfs) > max_files:
        print(f"Warning: Too many ERA5 files ({len(dfs)}), using only first {max_files}")
        dfs = dfs[:max_files]
    
    df = pd.concat(dfs, ignore_index=True)
    df = robust_numeric(df)
    
    # 如果数据量太大，进行采样
    if len(df) > 100000:  # 如果超过10万行，进行采样
        sample_size = min(50000, len(df))
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled ERA5 data to {len(df)} rows")

    # 猜测日期/时间、经纬度列名
    time_col = None
    for cand in ["time","date","Date","datetime","valid_time","时间","Time"]:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        time_col = df.columns[0]

    # 经纬度可能名为 latitude/longitude / lat/lon / 经度/纬度
    lat_col = None
    lon_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ["lat","latitude","纬度","y"]:
            lat_col = c
        if lc in ["lon","longitude","经度","x"]:
            lon_col = c
    # 若仍缺，则尝试匹配
    if lat_col is None:
        for c in df.columns:
            if re.search("lat", c, re.I):
                lat_col = c; break
    if lon_col is None:
        for c in df.columns:
            if re.search("lon|lng", c, re.I):
                lon_col = c; break

    # 时间转datetime
    df[time_col] = pd.to_datetime(df[time_col])

    # 只取北京范围网格
    if lat_col and lon_col:
        df = df[(df[lon_col] >= LON_MIN) & (df[lon_col] <= LON_MAX) &
                (df[lat_col] >= LAT_MIN) & (df[lat_col] <= LAT_MAX)]

    # 变量列：保留目标18项（兼容简称与全名）
    keep_map = {}
    for short, longn in ERA5_KEEP.items():
        if short in df.columns:
            keep_map[short] = short
        elif longn in df.columns:
            keep_map[longn] = short
        else:
            # 尝试模糊匹配
            for c in df.columns:
                if re.sub(r"[^a-z0-9]", "", c.lower()) == re.sub(r"[^a-z0-9]", "", longn.lower()):
                    keep_map[c] = short
                    break
    keep_cols = [time_col] + ([lat_col] if lat_col else []) + ([lon_col] if lon_col else []) + list(keep_map.keys())
    df = df[keep_cols]

    # 列重命名为统一短名
    df = df.rename(columns={**{k:v for k,v in keep_map.items()}, time_col:"date"})

    # 逐日 + 区域平均
    df["date"] = pd.to_datetime(df["date"])
    group_cols = ["date"]
    if lat_col: group_cols.append(lat_col)
    if lon_col: group_cols.append(lon_col)
    # 先对每个网格逐日均值
    daily = df.groupby([pd.to_datetime(df["date"]).dt.date, lat_col, lon_col]).mean(numeric_only=True).reset_index()
    daily = daily.rename(columns={"level_0":"date"}) if "level_0" in daily.columns else daily
    daily["date"] = pd.to_datetime(daily["date"])
    # 对北京范围网格做区域平均
    era5_daily = daily.groupby("date").mean(numeric_only=True).sort_values("date")
    era5_daily.index = era5_daily.index.set_names("date")
    era5_daily = era5_daily.loc[DATE_START:DATE_END]

    # 清除不合理值（如tp、tprate不为负；lsm在0-1）
    for c in era5_daily.columns:
        if c in ["tp", "avg_tprate", "sd", "tcwv", "blh", "str", "tisr"]:
            era5_daily.loc[era5_daily[c] < 0, c] = np.nan
        if c == "lsm":
            era5_daily[c] = era5_daily[c].clip(0, 1)
        # 极值裁剪
        era5_daily = clip_outliers(era5_daily, [c], z=4.0)

    # 适度插值（最多连续3天）
    era5_daily = era5_daily.sort_index().interpolate(limit=3).loc[DATE_START:DATE_END]
    # 只保留18个短名次序
    final_cols = [k for k in ERA5_KEEP.keys() if k in era5_daily.columns]
    era5_daily = era5_daily[final_cols]
    return era5_daily

era5_daily = clean_era5_block(dfs_era5)

# =======================
# 合并污染 + ERA5
# =======================
print("Pollution data shape:", pollution_daily.shape)
print("Pollution data columns:", pollution_daily.columns.tolist())
print("Pollution data date range:", pollution_daily.index.min(), "to", pollution_daily.index.max())

print("ERA5 data shape:", era5_daily.shape)
print("ERA5 data columns:", era5_daily.columns.tolist())
print("ERA5 data date range:", era5_daily.index.min(), "to", era5_daily.index.max())

merged = pollution_daily.join(era5_daily, how="inner").sort_index()
print("Merged shape:", merged.shape)

# 如果合并后为空，尝试使用outer join
if merged.empty:
    print("Inner join result is empty, trying outer join...")
    merged = pollution_daily.join(era5_daily, how="outer").sort_index()
    print("Outer join shape:", merged.shape)

# 丢弃全空列
merged = merged.dropna(axis=1, how="all")
print("After dropping all-null columns:", merged.shape)

# 进一步删除缺失太多的列（>30%缺失）
valid_cols = [c for c in merged.columns if merged[c].isna().mean() <= 0.30]
merged = merged[valid_cols]
print("After dropping high-missing columns:", merged.shape)

# 末次插值/填补
merged = merged.interpolate(limit=5).dropna(how="any")
print("Final merged shape:", merged.shape)
# =======================
# 相关性 + 热力图（单图单出）
# =======================
if merged.empty:
    print("Error: Merged data is empty, cannot perform correlation analysis and PCA.")
    print("Please check if data file paths and formats are correct.")
    exit(1)

corr = merged.corr(method="pearson")

# ---- 整体相关性热力图（一次只出这一张图）----
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, cmap=NATURE_CMAP, annot=False, linewidths=0.4,
    cbar_kws={"label": "Pearson r"}, vmin=-1, vmax=1, square=True
)
plt.title("Beijing Pollution vs ERA5 Meteorological Variables Correlation Heatmap (Daily, {} to {})".format(DATE_START, DATE_END), pad=12)
plt.tight_layout()
plt.show()

# =======================
# 与 PM2.5 / PM10 相关性 ≥ 0.7 的变量
# =======================
def top_corr_with(target, threshold=0.7):
    if target not in corr.columns:
        return pd.DataFrame(columns=["var","r"]).set_index("var")
    s = corr[target].drop(labels=[target]).dropna()
    s = s[abs(s) >= threshold].sort_values(key=lambda x: -abs(x))
    out = pd.DataFrame({"var": s.index, "r": s.values}).set_index("var")
    return out

strong_pm25 = top_corr_with("PM2.5", threshold=0.7)
strong_pm10 = top_corr_with("PM10",  threshold=0.7)

print("\nVariables with |r| >= 0.7 for PM2.5:")
print(strong_pm25.to_string() if len(strong_pm25) else "(None)")
print("\nVariables with |r| >= 0.7 for PM10:")
print(strong_pm10.to_string() if len(strong_pm10) else "(None)")
# =======================
# PCA：对所有数值列（包含污染+气象）
# =======================
X = merged.copy()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X = X[numeric_cols]

if X.empty or len(numeric_cols) == 0:
    print("Error: No numeric columns available for PCA analysis.")
    exit(1)

print(f"Number of numeric columns for PCA: {len(numeric_cols)}")
print(f"Data shape: {X.shape}")

# 标准化
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# 选择前K个主成分（解释≥80% 或最多10个）
pca = PCA(n_components=min(10, Xz.shape[1]))
Z = pca.fit_transform(Xz)

expl = pca.explained_variance_ratio_
cum_expl = np.cumsum(expl)

# ---- 方差解释率柱状图（单图单出）----
plt.figure(figsize=(8,5))
plt.bar(range(1, len(expl)+1), expl, label="Individual PC")
plt.plot(range(1, len(cum_expl)+1), cum_expl, marker="o", label="Cumulative")
plt.xlabel("Principal Component Number")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Explained Variance Ratio")
plt.legend(frameon=False)
sns.despine()
plt.tight_layout()
plt.show()

# ---- 前两主成分散点（单图单出）----
plt.figure(figsize=(7,6))
plt.scatter(Z[:,0], Z[:,1], s=8, alpha=0.6)
plt.xlabel(f"PC1 ({expl[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({expl[1]*100:.1f}%)")
plt.title("PCA Scatter Plot (Daily)")
sns.despine()
plt.tight_layout()
plt.show()

# ---- 变量载荷（前两主成分），用于解释主成分物理意义（单图单出）----
loadings = pd.DataFrame(pca.components_[:2, :], columns=numeric_cols, index=["PC1","PC2"]).T
print(f"Loadings shape: {loadings.shape}")
print(f"Loadings columns: {loadings.columns.tolist()}")
print(f"Loadings index: {loadings.index.tolist()[:10]}...")  # 显示前10个变量名

# 只展示载荷强的变量，避免杂乱
topk = 15
imp = (loadings.abs().max(axis=1)).sort_values(ascending=False).head(topk).index
print(f"Top {topk} important variables: {imp.tolist()}")
lod_sub = loadings.loc[imp]
print(f"lod_sub shape: {lod_sub.shape}")
print(f"lod_sub data:\n{lod_sub.head()}")

# 检查数据是否有效
if lod_sub.empty:
    print("ERROR: lod_sub is empty!")
    # 使用所有变量
    lod_sub = loadings
    print(f"Using all variables instead. Shape: {lod_sub.shape}")

# 创建更清晰的载荷图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

try:
    # 确保数据不为空且有数值
    if not lod_sub.empty and not lod_sub.isna().all().all():
        lod_sub_sorted = lod_sub.sort_values("PC1")
        print(f"Plotting data shape: {lod_sub_sorted.shape}")
        print(f"PC1 range: {lod_sub_sorted['PC1'].min():.3f} to {lod_sub_sorted['PC1'].max():.3f}")
        print(f"PC2 range: {lod_sub_sorted['PC2'].min():.3f} to {lod_sub_sorted['PC2'].max():.3f}")
        
        # PC1 载荷图
        colors1 = ['red' if x < 0 else 'blue' for x in lod_sub_sorted['PC1']]
        bars1 = ax1.barh(range(len(lod_sub_sorted)), lod_sub_sorted['PC1'], color=colors1, alpha=0.7)
        ax1.set_yticks(range(len(lod_sub_sorted)))
        ax1.set_yticklabels(lod_sub_sorted.index)
        ax1.set_xlabel('PC1 Loading')
        ax1.set_title('PC1 Variable Loadings')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # PC2 载荷图
        colors2 = ['red' if x < 0 else 'blue' for x in lod_sub_sorted['PC2']]
        bars2 = ax2.barh(range(len(lod_sub_sorted)), lod_sub_sorted['PC2'], color=colors2, alpha=0.7)
        ax2.set_yticks(range(len(lod_sub_sorted)))
        ax2.set_yticklabels(lod_sub_sorted.index)
        ax2.set_xlabel('PC2 Loading')
        ax2.set_title('PC2 Variable Loadings')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars1, lod_sub_sorted['PC1'])):
            ax1.text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.3f}', 
                    va='center', ha='left' if val >= 0 else 'right', fontsize=9)
        
        for i, (bar, val) in enumerate(zip(bars2, lod_sub_sorted['PC2'])):
            ax2.text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.3f}', 
                    va='center', ha='left' if val >= 0 else 'right', fontsize=9)
        
        plt.suptitle("Variable Loadings for First Two Principal Components", fontsize=14, y=0.98)
        
    else:
        print("ERROR: No valid data to plot!")
        ax1.text(0.5, 0.5, "No valid data to plot", ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("Variable Loadings - No Data")
        
except Exception as e:
    print(f"ERROR in plotting: {e}")
    ax1.text(0.5, 0.5, f"Plotting error: {e}", ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title("Variable Loadings - Error")

plt.tight_layout()
plt.show()

# =======================
# 结束：释放缓存/内存
# =======================
del dfs_poll_all, dfs_poll_extra, dfs_era5
del poll_all, poll_extra, pollution_daily, era5_daily
del merged, X, Xz, Z, pca, scaler, corr, loadings, lod_sub
gc.collect()
