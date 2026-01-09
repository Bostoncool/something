import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import geopandas as gpd
import frykit.plot as fplt
from functools import lru_cache
from pathlib import Path

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = Path(__file__).parent.absolute()

# 配置字体：统一使用 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Column names in source Excel files (Chinese)
REQUIRED_METEO_COLUMNS = ["站名", "Latitude", "Longitude"]
REQUIRED_POLL_COLUMNS = ["监测点", "经度", "纬度"]


@lru_cache(maxsize=1)
def read_beijing_boundary():
    """Read Beijing 16-district GeoJSON and unify to WGS84 coordinates."""
    geojson_path = SCRIPT_DIR / "110100.geojson"
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def plot_beijing_districts(ax):
    """Color districts for better spatial recognition."""
    gdf = read_beijing_boundary().sort_values("name")
    cmap = plt.get_cmap("tab20")

    for idx, row in enumerate(gdf.itertuples()):
        color = cmap(idx % cmap.N)
        ax.add_geometries(
            [row.geometry],
            crs=ccrs.PlateCarree(),
            facecolor=color,
            edgecolor='dimgray',
            linewidth=0.6,
            alpha=0.45
        )


def _read_with_fallback(path, required_columns, skiprows_candidates=(None, 1)):
    """Try different skiprows values to ensure correct header parsing."""
    last_missing = None
    for skip in skiprows_candidates:
        df = pd.read_excel(path, skiprows=skip)
        missing = [col for col in required_columns if col not in df.columns]
        if not missing:
            return df
        last_missing = missing
    raise KeyError(f"{path} missing required columns: {last_missing}")


# ----------------  Read Excel  ----------------
def read_coords():
    """Return two DataFrames: meteorological stations and air quality stations (decimal degrees)."""
    meteo_path = SCRIPT_DIR / "北京气象站点.xlsx"
    poll_path = SCRIPT_DIR / "北京市污染监测站点列表-2021.01.23起.xlsx"
    meteo = _read_with_fallback(str(meteo_path), REQUIRED_METEO_COLUMNS)
    poll = _read_with_fallback(str(poll_path), REQUIRED_POLL_COLUMNS)

    # Meteorological stations already have Latitude/Longitude columns
    meteo = meteo[REQUIRED_METEO_COLUMNS].dropna()
    meteo = meteo.rename(columns={"站名": "Station"})

    # Air quality stations have longitude/latitude columns in Chinese
    poll = poll[REQUIRED_POLL_COLUMNS].dropna()
    poll = poll.rename(columns={"监测点": "Station", "经度": "Longitude", "纬度": "Latitude"})

    return meteo, poll

# ----------------  Basemap function  ----------------
def beijing_lcc_map():
    proj = ccrs.LambertConformal(
        central_longitude=116.4,
        central_latitude=40.0,
        standard_parallels=(37, 43)
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection=proj)

    # Display extent: Beijing municipality
    ax.set_extent([115.3, 117.6, 39.4, 41.2], crs=ccrs.PlateCarree())

    # Add coastline, borders (Natural Earth)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.3)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.2)

    # Key: Beijing 16 districts precise boundaries (Beijing Planning Commission open data)
    plot_beijing_districts(ax)

    # Gridlines with external tick labels (longitude placed below frame)
    lon_ticks = np.arange(115.5, 117.7, 0.5)
    lat_ticks = np.arange(39.5, 41.3, 0.3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2,
                      color='gray', alpha=0.5, linestyle='--',
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xlocator = mticker.FixedLocator(lon_ticks)
    gl.ylocator = mticker.FixedLocator(lat_ticks)
    gl.xformatter = LongitudeFormatter(number_format=".1f")
    gl.yformatter = LatitudeFormatter(number_format=".1f")
    gl.xlabel_style = {'size': 8, 'family': 'Times New Roman'}
    gl.ylabel_style = {'size': 8, 'family': 'Times New Roman'}

    return fig, ax

# ----------------  Main function  ----------------
def main():
    meteo, poll = read_coords()
    fig, ax = beijing_lcc_map()

    # Plot stations
    ax.scatter(meteo.Longitude, meteo.Latitude,
               s=35, c='blue', marker='o',
               transform=ccrs.PlateCarree(),
               label=f'Meteorological Stations ({len(meteo)})')

    ax.scatter(poll.Longitude, poll.Latitude,
               s=35, c='red', marker='^',
               transform=ccrs.PlateCarree(),
               label=f'Air Quality Stations ({len(poll)})')

    # 添加比例尺（位置在右下角，长度 50 km，适合北京区域）
    scale_bar = fplt.add_scale_bar(ax, 0.7, 0.1, length=50)
    scale_bar.set_xticks([0, 25, 50])  # 设置刻度
    scale_bar.set_xlabel('km', fontfamily='Times New Roman')
    scale_bar.tick_params(labelsize=10)
    for label in scale_bar.get_xticklabels():
        label.set_fontfamily('Times New Roman')

    plt.legend(loc='upper left', prop={'family': 'Times New Roman'})
    plt.title('Beijing Map\n(Lambert Conformal Conic Projection)', fontsize=16, fontweight='bold', fontfamily='Times New Roman', pad=8)
    plt.tight_layout()
    plt.savefig('beijing_stations_lcc.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()