library(readr)
library(ggplot2)
library(dplyr)
library(sf)
library(ggspatial)
library(tidyverse)
library(purrr)

# 设置LAEA投影（局部等面积投影），中心点设置为珠三角地区中心（经度113.5°，纬度23°）
laea_crs <- st_crs("+proj=laea +lat_0=23 +lon_0=113.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

# 读取并转换投影的辅助函数
read_and_transform <- function(file_path, crs = laea_crs) {
  st_read(file_path) %>% st_transform(crs = crs)
}

# 读取广东省分区数据
guangdong_shp <- read_and_transform("C:/Users/IU/Desktop/大论文图/4.珠三角/广东省 (市).geojson")

# 珠三角地区城市配置
zhusanjiao_config <- list(
  files = c(
    "C:/Users/IU/Desktop/大论文图/4.珠三角/具体城市/广州市.geojson",
    "C:/Users/IU/Desktop/大论文图/4.珠三角/具体城市/深圳市.geojson",
    "C:/Users/IU/Desktop/大论文图/4.珠三角/具体城市/佛山市.geojson",
    "C:/Users/IU/Desktop/大论文图/4.珠三角/具体城市/东莞市.geojson",
    "C:/Users/IU/Desktop/大论文图/4.珠三角/具体城市/中山市.geojson",
    "C:/Users/IU/Desktop/大论文图/4.珠三角/具体城市/惠州市.geojson",
    "C:/Users/IU/Desktop/大论文图/4.珠三角/具体城市/珠海市.geojson",
    "C:/Users/IU/Desktop/大论文图/4.珠三角/具体城市/江门市.geojson",
    "C:/Users/IU/Desktop/大论文图/4.珠三角/具体城市/肇庆市.geojson"
  ),
  color = "lightcoral"
)

# 读取珠三角地区所有城市数据
zhusanjiao_shapes <- map(zhusanjiao_config$files, read_and_transform)

# 为每个城市数据添加区域名称列，并只保留geometry和region列
zhusanjiao_shapes_with_region <- map(zhusanjiao_shapes, function(shp) {
  # 只保留geometry列，添加region列
  shp_simple <- st_sf(region = "珠三角地区", geometry = st_geometry(shp))
  return(shp_simple)
})

# 合并珠三角地区所有城市数据
zhusanjiao_data <- do.call(rbind, zhusanjiao_shapes_with_region)

# 获取珠三角地区的边界框，用于设置地图范围
zhusanjiao_bbox <- st_bbox(zhusanjiao_data)

# 创建经纬度网格线（graticule）
# 获取地图边界框（在WGS84坐标系中）
zhusanjiao_bbox_wgs84 <- st_bbox(st_transform(zhusanjiao_data, crs = st_crs(4326)))

# 创建经纬度网格线
graticule <- st_graticule(
  st_transform(zhusanjiao_data, crs = st_crs(4326)),
  lat = seq(floor(zhusanjiao_bbox_wgs84$ymin), ceiling(zhusanjiao_bbox_wgs84$ymax), by = 2),
  lon = seq(floor(zhusanjiao_bbox_wgs84$xmin), ceiling(zhusanjiao_bbox_wgs84$xmax), by = 2)
) %>% st_transform(crs = laea_crs)

# 创建珠三角放大地图
zhusanjiao_map <- ggplot() +
  # 添加经纬度网格线（先绘制，作为底层）
  geom_sf(data = graticule, color = "gray80", linewidth = 0.3, linetype = "dashed") +
  # 添加经纬度标签
  geom_sf_text(
    data = graticule %>% filter(!is.na(degree_label)),
    aes(label = degree_label),
    color = "black",
    size = 3,
    inherit.aes = FALSE,
    nudge_x = 0,
    nudge_y = 0
  ) +
  # 添加广东省边界
  geom_sf(data = guangdong_shp, fill = "lightgray", color = "black", linewidth = 0.5, alpha = 0.3) +
  # 添加珠三角城市，使用指定的颜色
  geom_sf(data = zhusanjiao_data, fill = zhusanjiao_config$color, color = "gray30", linewidth = 0.4)

# 添加主题、比例尺、指北针和坐标
zhusanjiao_final_map <- zhusanjiao_map +
  # 添加比例尺（右下角）
  annotation_scale(location = "br", width_hint = 0.3, pad_x = unit(0.3, "cm"), pad_y = unit(0.3, "cm")) +
  # 添加指北针（右上角）
  annotation_north_arrow(
    location = "tr",
    which_north = "true",
    style = north_arrow_fancy_orienteering,
    height = unit(1.5, "cm"),
    width = unit(1.5, "cm"),
    pad_x = unit(0.3, "cm"),
    pad_y = unit(0.3, "cm")
  ) +
  theme_minimal() +
  theme(
    panel.grid = element_line(color = "gray90", linewidth = 0.3),
    panel.grid.minor = element_line(color = "gray95", linewidth = 0.2),
    axis.text = element_text(size = 12, color = "black"),
    axis.ticks = element_line(color = "black", linewidth = 0.5),
    axis.title = element_text(size = 14, face = "bold"),
    panel.background = element_rect(fill = NA, color = NA),
    plot.title = element_text(hjust = 0.5, size = 18, face = "bold")
  ) +
  labs(
    title = "珠江三角洲地区地图",
    x = "经度 (°E)",
    y = "纬度 (°N)"
  ) +
  coord_sf(
    crs = laea_crs,
    expand = FALSE,
    xlim = c(zhusanjiao_bbox$xmin, zhusanjiao_bbox$xmax),
    ylim = c(zhusanjiao_bbox$ymin, zhusanjiao_bbox$ymax),
    default_crs = st_crs(4326)
  )

print(zhusanjiao_final_map)

# 保存为高分辨率SVG图片（正方形输出）
ggsave(
  filename = "珠三角地区放大图.svg",
  plot = zhusanjiao_final_map,
  device = "svg",
  width = 10,      # 正方形宽度（英寸）
  height = 10,     # 正方形高度（英寸）
  units = "in",    # 单位：英寸
  dpi = 300        # 虽然SVG是矢量格式，但此参数用于某些转换
)