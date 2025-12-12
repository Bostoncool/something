import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ------------------------------------------------------------------------------
# 1. 投影参数（LCC 适合中纬度国家）
# ------------------------------------------------------------------------------
proj = ccrs.LambertConformal(
        central_longitude=105,      # 中国中心经度
        central_latitude=35,        # 中国中心纬度
        standard_parallels=(25, 47) # 两条标准纬线
)

# ------------------------------------------------------------------------------
# 2. 主图
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection=proj)
ax.set_extent([73, 135, 15, 55], crs=ccrs.PlateCarree())  # 中国大致范围

# 加海岸线、河流、湖泊（可选）
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.4)
ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=0.3, alpha=0.6)
ax.add_feature(cfeature.LAKES.with_scale('50m'), linewidth=0.3, alpha=0.6)

# ------------------------------------------------------------------------------
# 3. 读取 Natural Earth 1:50m 省界数据
#     第一次运行会自动下载到 ~/cartopy_data 目录
# ------------------------------------------------------------------------------
shp_provinces = shpreader.natural_earth(resolution='50m',
                                        category='cultural',
                                        name='admin_1_states_provinces')

# 4. 省界——为每个省份分配不同颜色
provinces = shpreader.Reader(shp_provinces).records()
china_provs = [p for p in provinces if p.attributes['adm0_a3'] == 'CHN']

# 使用 tab20 和 tab20b 配色方案，提供40种不同颜色
cmap1 = mpl.colormaps['tab20']
cmap2 = mpl.colormaps['tab20b']
colors = [cmap1(i) for i in range(20)] + [cmap2(i) for i in range(20)]

# 为每个省份单独绘制，分配不同颜色
for i, prov in enumerate(china_provs):
    color = colors[i % len(colors)]  # 循环使用颜色
    ax.add_geometries([prov.geometry], crs=ccrs.PlateCarree(),
                      facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.7)

# 6. 网格线 & 标签
gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5,
                  linestyle='--', x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# ------------------------------------------------------------------------------
# 7. 标题 & 保存
# ------------------------------------------------------------------------------
ax.set_title('China Provincial Boundaries (Lambert Conformal Conic)', pad=15)
plt.tight_layout()
plt.savefig('china_lcc.png', dpi=300)
plt.show()