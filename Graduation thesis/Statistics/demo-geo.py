# -*- coding: utf-8 -*-
"""
Plot sub-regions given by lat/lon box
author: you
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

# ---------------- 1. 定义四个区域的经纬度范围 ----------------
regions = {
    'Beijing-Tianjin-Hebei': {
        'lat_min': 36.00, 'lat_max': 41.99,
        'lon_min': 114.01, 'lon_max': 120.00,
        'color': 'firebrick'
    },
    'Yangtze River Delta': {
        'lat_min': 28.00, 'lat_max': 33.99,
        'lon_min': 116.01, 'lon_max': 124.00,
        'color': 'royalblue'
    },
    'Pearl River Delta': {
        'lat_min': 21.50, 'lat_max': 24.49,
        'lon_min': 112.01, 'lon_max': 115.50,
        'color': 'forestgreen'
    },
    'Sichuan Basin': {
        'lat_min': 28.00, 'lat_max': 32.49,
        'lon_min': 103.01, 'lon_max': 108.00,
        'color': 'darkorange'
    }
}

# ---------------- 2. 画图 ----------------
plt.figure(figsize=(10, 8))
# 选用亚东半球投影，中心经度 105°E
proj = ccrs.LambertAzimuthalEqualArea(central_longitude=105, central_latitude=35)
ax = plt.axes(projection=proj)
ax.set_extent([73, 135, 18, 54], crs=ccrs.PlateCarree())

# 加海岸线、省界、国界线
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5)
ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle=':', linewidth=0.4)

# 加陆地/海洋底色
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='whitesmoke')
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='aliceblue')

# 画矩形框
for name, info in regions.items():
    lon_w, lon_e = info['lon_min'], info['lon_max']
    lat_s, lat_n = info['lat_min'], info['lat_max']
    width = lon_e - lon_w
    height = lat_n - lat_s
    rect = Rectangle(
        (lon_w, lat_s), width, height,
        linewidth=1, edgecolor=info['color'], facecolor='none',
        transform=ccrs.PlateCarree(), label=name
    )
    ax.add_patch(rect)

# 图例
ax.legend(loc='upper right', frameon=True, title='Regions')

# 加网格线
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7,
                  linestyle='--', x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False

plt.title('China sub-regions', pad=15)
plt.tight_layout()
plt.show()