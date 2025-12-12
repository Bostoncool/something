# ---------------- 0. Environment ----------------
import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from numba import guvectorize, float32
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import blended_transform_factory
import pyproj

# ---------- 1. High-speed I/O: One-time Concatenation ----------
def load_pm25_fast(file_dir: str) -> tuple[xr.DataArray, xr.DataArray]:
    """Returns PM2.5 DataArray and land mask for 2000-2023 (scaled & missing values handled)"""
    import re
    import os
    
    files = sorted(Path(file_dir).glob('*.nc'))
    if not files:
        raise ValueError(f"No .nc files found in {file_dir}")
    
    print(f">>> Found {len(files)} files, starting validation...")
    
    # Validate and read files one by one
    valid_datasets = []
    valid_years = []
    
    def preprocess_func(ds):
        """Preprocessing function: handle PM2.5 data"""
        if 'PM2.5' in ds.variables:
            pm25 = ds['PM2.5'].where(ds['PM2.5'] != 65535) * 0.1
            return ds.assign(PM2_5=pm25).drop_vars('PM2.5')
        elif 'PM2_5' in ds.variables:
            pm25 = ds['PM2_5'].where(ds['PM2_5'] != 65535) * 0.1
            return ds.assign(PM2_5=pm25)
        else:
            raise ValueError("PM2.5 variable not found")
    
    for f in files:
        try:
            # Extract year
            year_match = re.search(r'\d{4}', f.stem)
            if year_match:
                year = pd.to_datetime(year_match.group(), format='%Y')
            else:
                print(f"Warning: Cannot extract year from filename: {f.name}, skipping")
                continue
            
            # Try to open file
            ds_single = None
            try:
                # Open file (load directly to memory without chunks for safety)
                ds_single = xr.open_dataset(
                    str(f),
                    decode_times=False,
                    engine='netcdf4'  # Explicitly specify engine
                )
                # Apply preprocessing
                ds_processed = preprocess_func(ds_single)
                # Load to memory immediately to avoid keeping file handles
                ds_processed = ds_processed.load()
                # Close file
                ds_single.close()
                ds_single = None
                
                valid_datasets.append(ds_processed)
                valid_years.append(year)
                print(f"  ✓ {f.name} ({year.year})")
            except Exception as e:
                if ds_single is not None:
                    try:
                        ds_single.close()
                    except:
                        pass
                print(f"  ✗ {f.name} read failed: {e}")
                # Only print detailed errors in debug mode
                if 'DEBUG' in os.environ:
                    import traceback
                    traceback.print_exc()
                continue
                
        except Exception as e:
            print(f"  ✗ {f.name} processing failed: {e}")
            continue
    
    if not valid_datasets:
        raise ValueError("No files successfully read!")
    
    print(f">>> Successfully read {len(valid_datasets)} files, starting merge...")
    
    # Merge datasets
    try:
        # Use concat instead of open_mfdataset for safety
        ds_combined = xr.concat(valid_datasets, dim='time')
        ds_combined = ds_combined.rename({'PM2_5': 'PM2.5'})
        
        # Set time coordinates
        ds_combined = ds_combined.assign_coords(time=pd.to_datetime(valid_years))
        
        # Clean up intermediate data
        for ds in valid_datasets:
            ds.close()

        pm25_data = ds_combined['PM2.5']

        # 创建陆地掩膜
        print('>>> Creating land mask to filter ocean regions...')
        land_mask = create_land_mask(pm25_data)

        return pm25_data, land_mask
    except Exception as e:
        # Clean up resources
        for ds in valid_datasets:
            try:
                ds.close()
            except:
                pass
        raise RuntimeError(f"Dataset merge failed: {e}") from e


# ---------- 1.5. Land Mask Creation ----------
def create_land_mask(data: xr.DataArray, min_valid_ratio: float = 0.1) -> xr.DataArray:
    """
    创建陆地掩膜：识别至少有一定比例时间点有有效数据的空间位置

    参数:
        data: PM2.5 DataArray (time, lat, lon)
        min_valid_ratio: 最小有效数据比例（默认0.1，即至少10%的时间有数据）

    返回:
        land_mask: 布尔掩膜，True表示陆地区域
    """
    # 计算每个空间点在时间维度上的有效数据比例
    valid_ratio = data.notnull().sum(dim='time') / data.sizes['time']

    # 创建掩膜：有效数据比例 >= min_valid_ratio 的位置为陆地
    land_mask = valid_ratio >= min_valid_ratio

    # 统计信息
    total_points = land_mask.size
    land_points = land_mask.sum().item()
    ocean_points = total_points - land_points
    reduction_ratio = (1 - land_points / total_points) * 100

    print(f'>>> Land mask created:')
    print(f'    Total grid points: {total_points}')
    print(f'    Land points: {land_points} ({land_points/total_points*100:.1f}%)')
    print(f'    Ocean points: {ocean_points} ({ocean_points/total_points*100:.1f}%)')
    print(f'    Computation reduction: {reduction_ratio:.1f}%')

    return land_mask


# ---------- 2. Numba Vectorized Regression ----------
@guvectorize([(float32[:], float32[:], float32[:])],
             '(n),(n)->()', nopython=True, target='cpu')
def _slope(y, x, out):
    """Least squares slope for each sequence"""
    n = y.size
    if n < 2:
        out[0] = np.nan
        return
    sx, sy, sxx, sxy = 0., 0., 0., 0.
    for i in range(n):
        if not (np.isnan(y[i]) or np.isnan(x[i])):
            sx += x[i]; sy += y[i]; sxx += x[i]*x[i]; sxy += x[i]*y[i]
    denom = n*sxx - sx*sx
    if abs(denom) < 1e-10:
        out[0] = np.nan
    else:
        out[0] = (n*sxy - sx*sy) / denom


def fast_trend(data: xr.DataArray, dim: str = 'time') -> xr.DataArray:
    """Calculate trend slope with error handling"""
    try:
        years = np.arange(data.sizes[dim], dtype='float32')
        result = xr.apply_ufunc(
            _slope, data, years,
            input_core_dims=[[dim], [dim]],
            output_core_dims=[[]],
            dask='parallelized',
            output_dtypes=['float32'],
            dask_gufunc_kwargs={'allow_rechunk': True}
        )
        return result
    except Exception as e:
        print(f"Warning: fast_trend calculation error, using fallback method: {e}")
        # Fallback method: use numpy's polyfit
        years = np.arange(data.sizes[dim])
        def _slope_fallback(ts):
            valid = ~np.isnan(ts)
            if valid.sum() < 2:
                return np.nan
            return np.polyfit(years[valid], ts[valid], 1)[0]
        return xr.apply_ufunc(_slope_fallback, data,
                             input_core_dims=[[dim]],
                             output_core_dims=[[]],
                             vectorize=True,
                             dask='parallelized',
                             output_dtypes=['float32'])


# ---------- 3. Numba Vectorized Mann-Kendall ----------
@guvectorize([(float32[:], float32[:])], '(n)->()', nopython=True, target='cpu')
def _mk(z, out):
    """Mann-Kendall Z value"""
    n = z.size
    if n < 2:
        out[0] = np.nan
        return
    s = 0.
    valid_pairs = 0
    for i in range(n - 1):
        if np.isnan(z[i]):
            continue
        for j in range(i + 1, n):
            if np.isnan(z[j]):
                continue
            s += np.sign(z[j] - z[i])
            valid_pairs += 1
    if valid_pairs == 0:
        out[0] = np.nan
        return
    var = (n * (n - 1) * (2*n + 5)) / 18.
    if var <= 0:
        out[0] = np.nan
        return
    if s > 0:
        out[0] = (s - 1) / np.sqrt(var)
    elif s < 0:
        out[0] = (s + 1) / np.sqrt(var)
    else:
        out[0] = 0.


def fast_mk(data: xr.DataArray, dim: str = 'time') -> xr.DataArray:
    """Calculate Mann-Kendall statistic with error handling"""
    try:
        result = xr.apply_ufunc(
            _mk, data,
            input_core_dims=[[dim]],
            output_core_dims=[[]],
            dask='parallelized',
            output_dtypes=['float32'],
            dask_gufunc_kwargs={'allow_rechunk': True}
        )
        return result
    except Exception as e:
        print(f"Warning: fast_mk calculation error, using fallback method: {e}")
        # Fallback method: use scipy.stats
        from scipy.stats import kendalltau
        def _mk_fallback(ts):
            valid = ~np.isnan(ts)
            if valid.sum() < 2:
                return np.nan
            years = np.arange(len(ts))[valid]
            tau, p_value = kendalltau(years, ts[valid])
            if tau == 0:
                return 0.0
            # Convert to Z value
            n = valid.sum()
            var = (n * (n - 1) * (2*n + 5)) / 18.
            if var <= 0:
                return np.nan
            s = tau * n * (n - 1) / 2
            if s > 0:
                return (s - 1) / np.sqrt(var)
            elif s < 0:
                return (s + 1) / np.sqrt(var)
            else:
                return 0.0
        return xr.apply_ufunc(_mk_fallback, data,
                             input_core_dims=[[dim]],
                             output_core_dims=[[]],
                             vectorize=True,
                             dask='parallelized',
                             output_dtypes=['float32'])


# ---------- 4. Statistics ----------
def basic_statistics(data, land_mask=None):
    """Calculate basic statistics, optionally masked to land only"""
    if land_mask is not None:
        data = data.where(land_mask)

    return {
        'annual_mean': data.mean(dim=['lat', 'lon'], skipna=True),
        'annual_max': data.max(dim=['lat', 'lon'], skipna=True),
        'annual_min': data.min(dim=['lat', 'lon'], skipna=True),
        'spatial_std': data.std(dim=['lat', 'lon'], skipna=True),
        'national_mean_concentration': data.mean(dim=['lat', 'lon'], skipna=True).mean(skipna=True)
    }


# ---------- 5. Time Series ----------
def time_series_analysis(data):
    national_mean = data.mean(dim=['lat', 'lon'])
    annual_change = national_mean.diff('time') / national_mean.shift(time=1) * 100
    cumulative = (national_mean - national_mean[0]) / national_mean[0] * 100
    ma5 = national_mean.rolling(time=5, center=True).mean()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(national_mean.time.dt.year, national_mean, 'o-')
    axes[0, 0].set_title('National Annual Mean PM2.5')
    axes[0, 1].bar(annual_change.time.dt.year[1:], annual_change[1:])
    axes[0, 1].set_title('Annual Change Rate %')
    axes[1, 0].plot(cumulative.time.dt.year, cumulative, 's-')
    axes[1, 0].set_title('Cumulative Change Rate (Relative to 2000) %')
    axes[1, 1].plot(national_mean.time.dt.year, national_mean, alpha=.5)
    axes[1, 1].plot(ma5.time.dt.year, ma5, 'r-', lw=2)
    axes[1, 1].set_title('5-Year Moving Average')
    # fig.tight_layout()
    return fig


# ---------- 5.5. 绘图辅助函数：比例尺和指北针 ----------
def add_scalebar(ax, length_km=500, location=(0.1, 0.08), linewidth=3):
    """
    在 Cartopy Axes 添加比例尺
    
    参数:
        ax: cartopy Axes
        length_km: 比例尺长度（公里）
        location: (x,y) 以轴域坐标为单位 (0~1)
        linewidth: 比例尺线条宽度
    """
    # 取主图中心纬度，用于换算 km→经度距离
    lon0 = (ax.get_extent(crs=ccrs.PlateCarree())[0] +
            ax.get_extent(crs=ccrs.PlateCarree())[1]) / 2
    lat0 = (ax.get_extent(crs=ccrs.PlateCarree())[2] +
            ax.get_extent(crs=ccrs.PlateCarree())[3]) / 2
    
    geod = pyproj.Geod(ellps='WGS84')
    
    # 计算 length_km 对应的经度差
    lon1, lat1, _ = geod.fwd(lon0, lat0, 90, length_km * 1000)
    dlon = lon1 - lon0
    
    # 转换到轴域坐标
    x0, y0 = location
    x1 = x0 + 0.15  # 比例尺长度可调（相对坐标）
    
    transform = blended_transform_factory(ax.transAxes, ax.transAxes)
    ax.plot([x0, x1], [y0, y0], transform=transform, color='k', linewidth=linewidth)
    ax.text(x0, y0 - 0.02, "0 km", transform=transform, ha='center', va='top', fontsize=9)
    ax.text(x1, y0 - 0.02, f"{length_km} km", transform=transform, ha='center', va='top', fontsize=9)


def add_north_arrow(ax, size=0.1, loc_x=0.95, loc_y=0.90):
    """
    在 Cartopy Axes 添加指北针
    
    参数:
        ax: cartopy Axes
        size: 箭头大小
        loc_x: x位置（轴域坐标，0~1）
        loc_y: y位置（轴域坐标，0~1）
    """
    ax.annotate(
        'N',
        xy=(loc_x, loc_y),
        xytext=(loc_x, loc_y - size),
        xycoords='axes fraction',
        ha='center',
        va='center',
        fontsize=18,
        fontweight='bold',
        arrowprops=dict(arrowstyle="-|>", color='black', lw=2)
    )


# ---------- 6. Spatial ----------
def spatial_distribution_analysis(data, land_mask=None):
    """Spatial distribution analysis with land mask optimization"""
    try:
        # 如果提供了掩膜，只对陆地区域进行计算
        if land_mask is not None:
            print('>>> Applying land mask to reduce computation...')
            data_masked = data.where(land_mask)
        else:
            data_masked = data

        # 计算空间统计（海洋区域自动为NaN）
        spatial_mean = data_masked.mean(dim='time', skipna=True)
        spatial_std = data_masked.std(dim='time', skipna=True)

        print('>>> Calculating trend slope (land only)...')
        # 对于趋势计算，只对有效点进行计算
        slope = fast_trend(data_masked)

        print('>>> Calculating Mann-Kendall statistic (land only)...')
        mk_z = fast_mk(data_masked)

        # Load to memory
        spatial_mean = spatial_mean.compute()
        spatial_std = spatial_std.compute()
        slope = slope.compute()
        mk_z = mk_z.compute()

        # 应用掩膜确保海洋区域为NaN
        if land_mask is not None:
            spatial_mean = spatial_mean.where(land_mask)
            spatial_std = spatial_std.where(land_mask)
            slope = slope.where(land_mask)
            mk_z = mk_z.where(land_mask)

    except Exception as e:
        print(f"Error: Spatial statistics calculation failed: {e}")
        raise

    # 中国最合适的 Lambert Conformal 投影
    proj = ccrs.LambertConformal(
        central_longitude=105,
        central_latitude=35,
        standard_parallels=(25, 47)
    )

    # 中国标准范围
    china_extent = [73, 135, 18, 54]

    # 导入省界（自然地理数据）
    provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none'
    )

    # 确保output文件夹存在
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    for src, title, vmin, vmax, cmap, filename in zip(
            [spatial_mean, spatial_std, slope, mk_z],
            ['2000-2023 Annual Mean PM2.5', 'Spatial Variability (Std Dev)', 
             'Trend Slope µg/m³/year', 'Mann-Kendall Z'],
            [0, 0, -2, -2],
            [100, None, 2, 2],
            ['hot_r', 'viridis', 'RdBu_r', 'RdBu_r'],
            ['PM25_mean.png', 'PM25_std.png', 'PM25_trend.png', 'PM25_mk.png']):

        fig = plt.figure(figsize=(9, 6))  # Single plot size

        ax = fig.add_subplot(1, 1, 1, projection=proj)

        # 主图：数据绘制
        im = src.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            add_colorbar=False
        )

        # 添加海岸线、国界、省界
        ax.coastlines(resolution='50m', linewidth=0.4)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.4)
        ax.add_feature(provinces, edgecolor='gray', linewidth=0.4)

        # 设置中国地图范围
        ax.set_extent(china_extent, crs=ccrs.PlateCarree())

        # 美化坐标轴：经纬度格式化
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color='gray',
            alpha=0.5,
            linestyle='--',
            x_inline=False,   # 关键：关闭经度内嵌，让标签显示在边框上
            y_inline=False    # 关键：关闭纬度内嵌，让标签显示在边框上
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True        # 只在底边显示经度
        gl.left_labels = True          # 只在左侧显示纬度
        gl.xlocator = FixedLocator(range(75, 136, 10))
        gl.ylocator = FixedLocator(range(20, 55, 5))
        gl.xformatter = LongitudeFormatter(number_format='.0f')
        gl.yformatter = LatitudeFormatter(number_format='.0f')
        
        # 移除 xarray 自动添加的轴标签
        ax.set_xlabel('')
        ax.set_ylabel('')

        ax.set_title(title)

        # 水平色标
        plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8)

        # 添加南海诸岛插图 —— 新位置：图的左下角，不遮挡台湾
        bbox = ax.get_position()

        inset_w = 0.10   # 小一些
        inset_h = 0.16

        # 左下角更安全的位置
        inset_ax = fig.add_axes([
            bbox.x0 + 0.01,     # 靠左
            bbox.y0 + 0.01,     # 靠下
            inset_w,
            inset_h
        ], projection=ccrs.PlateCarree())

        # 设置南海诸岛范围
        inset_ax.set_extent([105, 125, 3, 25], crs=ccrs.PlateCarree())

        # 绘制南海诸岛 - 使用高分辨率海岸线
        inset_ax.coastlines(resolution='10m', linewidth=0.6, color='black')
        inset_ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.4)
        inset_ax.add_feature(provinces, edgecolor='gray', linewidth=0.25)

        src.plot(
            ax=inset_ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=vmin, vmax=vmax,
            add_colorbar=False
        )

        # 去刻度提高美观
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        
        # 移除 xarray 自动添加的轴标签
        inset_ax.set_xlabel('')
        inset_ax.set_ylabel('')

        inset_ax.set_title('South China Sea', fontsize=7)

        # 添加指北针（主图右上角）
        add_north_arrow(ax)
        
        # 添加比例尺（主图左下角）
        add_scalebar(ax, length_km=500)

        # fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=600, bbox_inches='tight')
        plt.close(fig)

    # No return value, figures saved separately


# ---------- 7. Regional Comparison ----------
def regional_analysis(data):
    """Regional comparison analysis with data validity checks"""
    # First check the lat/lon range of the data
    lat_decreasing = False  # Default: assume increasing
    if 'lat' in data.coords and 'lon' in data.coords:
        lat_min, lat_max = float(data.lat.min()), float(data.lat.max())
        lon_min, lon_max = float(data.lon.min()), float(data.lon.max())
        print(f'>>> Data lat/lon range: Latitude [{lat_min:.2f}, {lat_max:.2f}], Longitude [{lon_min:.2f}, {lon_max:.2f}]')
        
        # Check latitude order (increasing or decreasing)
        lat_first = float(data.lat[0])
        lat_last = float(data.lat[-1])
        lat_decreasing = lat_first > lat_last
        if lat_decreasing:
            print(f'>>> Latitude is decreasing (from {lat_first:.2f} to {lat_last:.2f}), adjusting slice order')
        else:
            print(f'>>> Latitude is increasing (from {lat_first:.2f} to {lat_last:.2f})')
    
    regions = {
        'Beijing-Tianjin-Hebei': (36, 42, 114, 120),
        'Yangtze River Delta': (28, 34, 116, 124),
        'Pearl River Delta': (21.5, 24.5, 112, 115.5),
        'Sichuan Basin': (28, 32.5, 103, 108)
    }
    
    regional_data = {}
    for name, (l1, l2, l3, l4) in regions.items():
        try:
            # Select regional data - adjust lat slice order if latitude is decreasing
            if lat_decreasing:
                region_subset = data.sel(lat=slice(l2, l1), lon=slice(l3, l4))
            else:
                region_subset = data.sel(lat=slice(l1, l2), lon=slice(l3, l4))
            
            # Debug: print region subset info
            if region_subset.size > 0:
                print(f'>>> Region: {name}, subset shape: {region_subset.shape}, '
                      f'Lat range: [{float(region_subset.lat.min()):.2f}, {float(region_subset.lat.max()):.2f}], '
                      f'Lon range: [{float(region_subset.lon.min()):.2f}, {float(region_subset.lon.max()):.2f}]')
            
            # Check if there is valid data
            if region_subset.size == 0:
                print(f'Warning: {name} region has no data points')
                regional_data[name] = None
                continue
            
            # Calculate regional mean, skip NaN values
            region_mean = region_subset.mean(dim=['lat', 'lon'], skipna=True)
            
            # Check if there are valid values
            if region_mean.notnull().sum() == 0:
                print(f'Warning: {name} region all time points are NaN')
                regional_data[name] = None
            else:
                regional_data[name] = region_mean
                # Ensure computation is complete (if dask array)
                if hasattr(region_mean.data, 'compute'):
                    region_mean = region_mean.compute()
                    regional_data[name] = region_mean
        except Exception as e:
            print(f'Warning: {name} region selection failed: {e}')
            regional_data[name] = None

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for idx, (name, ser) in enumerate(regional_data.items()):
        ax = axes[idx // 2, idx % 2]
        if ser is not None and ser.notnull().sum() > 0:
            ax.plot(ser.time.dt.year, ser, 'o-')
            ax.set_title(f'{name} Annual Mean PM2.5')
        else:
            ax.text(0.5, 0.5, f'{name}\nNo Valid Data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name} Annual Mean PM2.5')
        ax.grid(True)
    # fig.tight_layout()

    # Build statistics table, only include valid data
    reg_stats_dict = {}
    for name, ser in regional_data.items():
        if ser is not None and ser.notnull().sum() > 0:
            try:
                mean_val = float(ser.mean(skipna=True))
                min_val = float(ser.min(skipna=True))
                max_val = float(ser.max(skipna=True))
                trend_val = float(fast_trend(ser).item())
                
                reg_stats_dict[name] = {
                    'mean': mean_val,
                    'range': f"{min_val:.1f}-{max_val:.1f}",
                    'trend': trend_val
                }
            except Exception as e:
                print(f'Warning: {name} statistics calculation failed: {e}')
                reg_stats_dict[name] = {
                    'mean': np.nan,
                    'range': 'nan-nan',
                    'trend': np.nan
                }
        else:
            reg_stats_dict[name] = {
                'mean': np.nan,
                'range': 'nan-nan',
                'trend': np.nan
            }
    
    reg_stats = pd.DataFrame(reg_stats_dict).T
    print('Key Regional Statistics\n', reg_stats)
    return fig, reg_stats


# ---------- 8. Pollution Level ----------
def pollution_level_analysis(data):
    levels = {'Excellent': (0, 35), 'Good': (35, 75), 'Light': (75, 115),
              'Moderate': (115, 150), 'Heavy': (150, 250), 'Severe': (250, 1e4)}
    total = data[0].count().item()
    level_df = pd.DataFrame({
        name: ((data > l) & (data <= h)).sum(dim=['lat', 'lon']) / total * 100
        for name, (l, h) in levels.items()
    })
    level_df.index = data.time.dt.year

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].stackplot(level_df.index, level_df.T, labels=level_df.columns)
    axes[0].set_title('Level Area Proportion')
    axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    heavy = ((data > 150).sum(dim=['lat', 'lon']) / total * 100)
    axes[1].plot(data.time.dt.year, heavy, 'ro-')
    axes[1].set_title('Heavy Pollution Area Proportion')
    good = ((data > 0) & (data <= 75)).sum(dim=['lat', 'lon']) / total * 100
    axes[2].plot(data.time.dt.year, good, 'go-')
    axes[2].set_title('Good/Excellent Area Proportion')
    # fig.tight_layout()
    return fig


# ---------- 9. Report ----------
def generate_summary_report(data, reg_stats):
    """Generate comprehensive report, handle NaN values"""
    nat = data.mean(dim=['lat', 'lon'], skipna=True)
    
    # Ensure computation is complete (if dask array)
    if hasattr(nat.data, 'compute'):
        nat = nat.compute()
    
    print('=' * 60)
    print('China PM2.5 Spatiotemporal Variation Comprehensive Analysis Report (2000-2023)')
    print('=' * 60)
    
    # Calculate national statistics, skip NaN
    nat_mean = float(nat.mean(skipna=True))
    nat_min = float(nat.min(skipna=True))
    nat_max = float(nat.max(skipna=True))
    
    print(f'National Average: {nat_mean:.2f} µg/m³')
    print(f'Concentration Range: {nat_min:.1f} - {nat_max:.1f} µg/m³')
    
    # Calculate trend
    try:
        tr = fast_trend(nat)
        if hasattr(tr.data, 'compute'):
            tr = tr.compute()
        tr_val = float(tr.item())
        print(f'Overall Trend: {tr_val:.3f} µg/m³/year')
    except Exception as e:
        print(f'Overall Trend: Calculation failed ({e})')
    
    print('Key Regions:')
    for name, row in reg_stats.iterrows():
        mean_val = row['mean']
        trend_val = row['trend']
        if pd.notna(mean_val) and pd.notna(trend_val):
            print(f'  {name}: Mean {mean_val:.1f}  Trend {trend_val:.3f}')
        else:
            print(f'  {name}: No Valid Data')
    
    # Calculate data completeness
    completeness = float((data.notnull().sum() / data.size * 100).item())
    print(f'Data Completeness: {completeness:.1f}%')


# ---------------- 10. Main Workflow ----------------
if __name__ == '__main__':
    import os
    import warnings
    warnings.filterwarnings('ignore')
    
    # Set HDF5 environment variables to avoid memory issues
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
    
    # Set matplotlib backend to avoid GUI-related issues
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    try:
        file_dir = '/root/Year'  # <-- Change to your path
        print('>>> Loading data...')
        pm25, land_mask = load_pm25_fast(file_dir)  # 现在返回数据和掩膜
        print(f'>>> Data shape: {pm25.shape}')
        print(f'>>> Time range: {pm25.time.min().values} to {pm25.time.max().values}')
        
        # Data diagnostic information
        if 'lat' in pm25.coords and 'lon' in pm25.coords:
            print(f'>>> Lat/lon range: Latitude [{float(pm25.lat.min()):.2f}, {float(pm25.lat.max()):.2f}], '
                  f'Longitude [{float(pm25.lon.min()):.2f}, {float(pm25.lon.max()):.2f}]')
        
        # Check data validity (land only)
        valid_ratio = float((pm25.where(land_mask).notnull().sum() /
                            (pm25.where(land_mask).size) * 100).item())
        print(f'>>> Data completeness (land only): {valid_ratio:.1f}%')

        # Check data value range
        if valid_ratio > 0:
            pm25_land = pm25.where(land_mask)
            data_min = float(pm25_land.min(skipna=True).item())
            data_max = float(pm25_land.max(skipna=True).item())
            data_mean = float(pm25_land.mean(skipna=True).item())
            print(f'>>> Data value range (land): {data_min:.2f} - {data_max:.2f} µg/m³')
            print(f'>>> Data mean value (land): {data_mean:.2f} µg/m³')
        
        # 传递掩膜到各个分析函数
        stats = basic_statistics(pm25, land_mask)

        # 确保output文件夹存在
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        
        print('>>> Generating time series plot...')
        time_fig = time_series_analysis(pm25.where(land_mask) if land_mask is not None else pm25)
        time_fig.savefig(output_dir / 'PM25_time_series.png', dpi=300, bbox_inches='tight')
        plt.close(time_fig)

        print('>>> Generating spatial distribution plot...')
        spatial_distribution_analysis(pm25, land_mask)

        print('>>> Regional comparison...')
        regional_fig, reg_stats = regional_analysis(pm25.where(land_mask) if land_mask is not None else pm25)
        regional_fig.savefig(output_dir / 'PM25_regional.png', dpi=300, bbox_inches='tight')
        plt.close(regional_fig)

        print('>>> Pollution level...')
        pollution_fig = pollution_level_analysis(pm25.where(land_mask) if land_mask is not None else pm25)
        pollution_fig.savefig(output_dir / 'PM25_level.png', dpi=300, bbox_inches='tight')
        plt.close(pollution_fig)

        print('>>> Comprehensive report...')
        generate_summary_report(pm25.where(land_mask) if land_mask is not None else pm25, reg_stats)

        print('>>> All completed! Images saved to output directory.')
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        raise