library(readr)
library(ggplot2)
library(dplyr)
library(sf)
library(ggspatial)
library(tidyverse)
library(purrr)

# 设置LAEA投影（局部等面积投影），中心点设置为长三角地区中心（经度120°，纬度31°）
laea_crs <- st_crs("+proj=laea +lat_0=31 +lon_0=120 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

# 读取并转换投影的辅助函数
read_and_transform <- function(file_path, crs = laea_crs) {
  st_read(file_path) %>% st_transform(crs = crs)
}

# 定义长三角地区分区边界文件配置 - 用于绘制省界轮廓
province_config <- list(
  files = c(
    "C:/Users/IU/Desktop/大论文图/3.长三角/上海市 (划分).geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/安徽省 (市).geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/江苏省 (市).geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/浙江省 (市).geojson"
  )
)

# 定义长三角地区的具体市区文件配置 - 包含所有具体的市区行政区划
changjiang_config <- list(
  files = c(
    "C:/Users/IU/Desktop/大论文图/3.长三角/上海市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/南京市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/无锡市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/南通市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/盐城市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/扬州市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/镇江市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/常州市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/苏州市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/泰州市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/杭州市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/宁波市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/嘉兴市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/湖州市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/绍兴市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/金华市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/舟山市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/台州市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/温州市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/合肥市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/芜湖市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/马鞍山市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/铜陵市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/安庆市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/滁州市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/池州市.geojson",
    "C:/Users/IU/Desktop/大论文图/3.长三角/具体城市/宣城市.geojson"
  ),
  color = "turquoise"
)

# 读取长三角地区分区边界数据
province_shapes <- map(province_config$files, read_and_transform)

# 为分区数据添加省界标识
province_shapes_with_boundary <- map(province_shapes, function(shp) {
  shp_simple <- st_sf(boundary = "省界", geometry = st_geometry(shp))
  return(shp_simple)
})

# 合并分区边界数据
province_boundary_data <- do.call(rbind, province_shapes_with_boundary)

# 读取长三角分区数据
changjiang_shapes <- map(changjiang_config$files, read_and_transform)

# 为每个市区数据添加区域名称列，只保留geometry和region列
# 确保只保留具体的市区级别的行政区划数据
changjiang_shapes_with_region <- map(changjiang_shapes, function(shp) {
  # 只保留geometry列，添加region列
  # 筛选出市区级别的行政区划（去除县级及以下）
  if("行政区划" %in% colnames(shp) || "admin_level" %in% colnames(shp)) {
    # 如果有行政级别信息，只保留市级及以上的区划
    shp_filtered <- shp
  } else {
    # 如果没有级别信息，直接使用
    shp_filtered <- shp
  }
  shp_simple <- st_sf(region = "长三角地区", geometry = st_geometry(shp_filtered))
  return(shp_simple)
})

# 合并长三角所有分区数据
changjiang_data <- do.call(rbind, changjiang_shapes_with_region)

# 获取长三角地区的边界框，用于设置合适的显示范围
changjiang_bbox <- st_bbox(changjiang_data)

# 创建经纬度网格线（graticule）
# 获取地图边界框（在WGS84坐标系中）
changjiang_bbox_wgs84 <- st_bbox(changjiang_data %>% st_transform(crs = st_crs(4326)))

# 创建经纬度网格线，针对长三角地区的经纬度范围
graticule <- st_graticule(
  changjiang_data %>% st_transform(crs = st_crs(4326)),
  lat = seq(floor(changjiang_bbox_wgs84$ymin), ceiling(changjiang_bbox_wgs84$ymax), by = 2),
  lon = seq(floor(changjiang_bbox_wgs84$xmin), ceiling(changjiang_bbox_wgs84$xmax), by = 2)
) %>% st_transform(crs = laea_crs)

# 创建长三角地区放大地图
changjiang_map <- ggplot() +
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
  # 添加分区边界轮廓（省界）
  geom_sf(data = province_boundary_data, fill = NA, color = "black", linewidth = 0.8, alpha = 0.8) +
  # 添加长三角市区图层 - 绘制所有具体的市区行政区划
  geom_sf(data = changjiang_data, fill = changjiang_config$color, color = "gray", linewidth = 0.3)

# 添加主题、图例、比例尺、指北针和坐标
changjiang_final_map <- changjiang_map +
  # 设置填充颜色
  scale_fill_manual(
    name = "研究区域",
    values = c("长三角地区" = changjiang_config$color),
    breaks = c("长三角地区")
  ) +
  # 添加比例尺（左下角）
  annotation_scale(location = "bl", width_hint = 0.3, pad_x = unit(0.3, "cm"), pad_y = unit(0.3, "cm")) +
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
    axis.text = element_text(size = 16, color = "black"),
    axis.ticks = element_line(color = "black", linewidth = 0.5),
    axis.title = element_text(size = 18, face = "bold"),
    panel.background = element_rect(fill = NA, color = NA),
    plot.title = element_text(hjust = 0.5, size = 22, face = "bold"),
    legend.position = c(0.02, 0.15),
    legend.justification = c(0, 0),
    legend.title = element_text(size = 28, face = "bold"),
    legend.text = element_text(size = 26),
    legend.background = element_blank(),
    legend.spacing.y = unit(1.5, "cm"),
    legend.key.height = unit(0.5, "cm"),
    legend.margin = margin(t = 15, r = 15, b = 15, l = 15)
  ) +
  labs(
    title = "长三角地区分区图",
    x = "经度 (°E)",
    y = "纬度 (°N)"
  ) +
  coord_sf(
    crs = laea_crs,
    expand = FALSE,
    xlim = c(changjiang_bbox$xmin, changjiang_bbox$xmax),
    ylim = c(changjiang_bbox$ymin, changjiang_bbox$ymax),
    default_crs = st_crs(4326)
  )

print(changjiang_final_map)

# 保存为正方形高分辨率SVG图片
ggsave(
  filename = "长三角地区放大图.svg",
  plot = changjiang_final_map,
  device = "svg",
  width = 12,      # 宽度（英寸）
  height = 12,     # 高度（英寸）- 设置为正方形
  units = "in",    # 单位：英寸
  dpi = 300        # 虽然SVG是矢量格式，但此参数用于某些转换
)