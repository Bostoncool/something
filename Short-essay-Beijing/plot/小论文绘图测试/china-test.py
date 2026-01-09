import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import frykit.plot as fplt

# 1. 创建地图，使用 Lambert Conformal Conic 投影
# 参数说明：central_longitude=105 (中央经线), standard_parallels=(25, 47) (标准纬线)
proj = ccrs.LambertConformal(central_longitude=105, standard_parallels=(25, 47))
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': proj})

# 2. 设置地图范围（中国区域）
ax.set_extent([70, 140, 0, 60], crs=ccrs.PlateCarree())

# 3. 绘制国界
fplt.add_cn_border(ax, linewidth=1.2, ec='black')

# 4. 绘制省界（所有省份，默认无填充）
fplt.add_cn_province(ax, linewidth=0.8, ec='gray', fc='none')

# 5. 单独为北京市填充浅绿色
# 注意：frykit 的省名使用中文全称
fplt.add_cn_province(ax, ['北京市'], fc='lightgreen', ec='gray', linewidth=0.8)

# 6. 绘制南海九段线（南海小图）
# 注意：frykit 使用 add_cn_line 方法绘制九段线
fplt.add_cn_line(ax, linewidth=1.2, ec='black')

# 7. 添加指北针（位置在右上角，基于 Axes 坐标系）
fplt.add_compass(ax, 0.95, 0.9, size=20)

# 8. 添加比例尺（长度 1000 km，位置在左下角）
scale_bar = fplt.add_scale_bar(ax, 0.15, 0.1, length=1000)
scale_bar.set_xticks([0, 500, 1000])  # 设置刻度
scale_bar.set_xlabel('km', fontfamily='Times New Roman')
scale_bar.tick_params(labelsize=10)
for label in scale_bar.get_xticklabels():
    label.set_fontfamily('Times New Roman')

# 9. 添加经纬度网格
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.5,
    linestyle='--'
)
gl.top_labels = False    # 关闭顶部标签
gl.right_labels = False  # 关闭右侧标签
gl.xlabel_style = {'size': 10, 'fontfamily': 'Times New Roman'}
gl.ylabel_style = {'size': 10, 'fontfamily': 'Times New Roman'}

# 10. 显示地图
plt.title('China Map  (Lambert Conformal Conic Projection)', fontsize=16, fontweight='bold', fontfamily='Times New Roman', pad=8)
plt.tight_layout(pad=1.0)  # 减小边距，使图片更紧凑
plt.savefig('china-test.png', dpi=300, bbox_inches='tight')  # bbox_inches='tight' 进一步减少空白边距
plt.show()