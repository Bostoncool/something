library(readr)
library(ggplot2)
library(dplyr)
library(sf)
library(ggspatial)
library(tidyverse)
library(purrr)

# 设置LAEA投影（局部等面积投影），中心点设置为中国中心（经度105°，纬度35°）
laea_crs <- st_crs("+proj=laea +lat_0=35 +lon_0=105 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

# 读取并转换投影的辅助函数
read_and_transform <- function(file_path, crs = laea_crs) {
  st_read(file_path) %>% st_transform(crs = crs)
}

# 读取中国行政区划数据
china_shp <- read_and_transform("C:/Users/IU/Desktop/大论文图/1.总图/中华人民共和国.geojson")
china_province_shp <- read_and_transform("C:/Users/IU/Desktop/大论文图/1.总图/中国（省）.geojson")

# 定义各区域的城市配置（路径和颜色）
region_config <- list(
  # 京津冀地区
  jingjinji = list(
    files = c(
      "C:/Users/IU/Desktop/大论文图/2.京津冀/北京市.geojson",
      "C:/Users/IU/Desktop/大论文图/2.京津冀/天津市.geojson",
      "C:/Users/IU/Desktop/大论文图/2.京津冀/河北省 (市).geojson"
    ),
    color = "lightyellow"
  ),
  # 长三角地区
  changsanjiao = list(
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
  ),
  # 珠三角地区
  zhusanjiao = list(
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
)

# 批量读取并转换各区域数据，合并为单个sf对象并添加区域名称列
all_regions_list <- map(names(region_config), function(region_name) {
  region <- region_config[[region_name]]
  # 读取该区域所有城市数据
  region_shapes <- map(region$files, read_and_transform)
  # 为每个城市数据添加区域名称列，并只保留geometry和region列
  region_shapes_with_region <- map(region_shapes, function(shp) {
    region_name_chinese <- case_when(
      region_name == "jingjinji" ~ "京津冀地区",
      region_name == "changsanjiao" ~ "长三角地区",
      region_name == "zhusanjiao" ~ "珠三角地区"
    )
    # 只保留geometry列，添加region列
    shp_simple <- st_sf(region = region_name_chinese, geometry = st_geometry(shp))
    return(shp_simple)
  })
  # 合并该区域所有城市数据
  region_combined <- do.call(rbind, region_shapes_with_region)
  return(region_combined)
})
# 合并所有区域数据
all_regions_data <- do.call(rbind, all_regions_list)

# 创建经纬度网格线（graticule）
# 获取地图边界框（在WGS84坐标系中）
china_bbox <- st_bbox(china_shp %>% st_transform(crs = st_crs(4326)))
# 创建经纬度网格线
graticule <- st_graticule(
  china_shp %>% st_transform(crs = st_crs(4326)),
  lat = seq(floor(china_bbox$ymin), ceiling(china_bbox$ymax), by = 10),
  lon = seq(floor(china_bbox$xmin), ceiling(china_bbox$xmax), by = 10)
) %>% st_transform(crs = laea_crs)

# 获取地图边界框（在投影坐标系中），用于定位经纬度标签
map_bbox <- st_bbox(china_shp)

# 创建基础地图
base_map <- ggplot() +
  # 添加经纬度网格线（先绘制，作为底层）
  geom_sf(data = graticule, color = "gray80", linewidth = 0.3, linetype = "dashed") +
  # 添加经纬度标签（使用graticule对象中的标签信息）
  # 只显示有degree_label的行（即标签点）
  geom_sf_text(
    data = graticule %>% filter(!is.na(degree_label)),
    aes(label = degree_label),
    color = "black",
    size = 3,
    inherit.aes = FALSE,
    # 调整标签位置，避免重叠
    nudge_x = 0,
    nudge_y = 0
  ) +
  geom_sf(data = china_shp, fill = "white", color = "black", linewidth = 0.8) +
  geom_sf(data = china_province_shp, fill = "lightgray", color = "gray", linewidth = 0.3, alpha = 0.5) +
  # 添加各区域的城市图层，使用fill映射到region以生成图例
  geom_sf(data = all_regions_data, aes(fill = region), color = "gray", linewidth = 0.3)

# 添加主题、图例、比例尺、指北针和坐标
china_province_map <- base_map +
  # 手动设置填充颜色和图例
  scale_fill_manual(
    name = "研究区域",
    values = c(
      "京津冀地区" = "lightyellow",
      "长三角地区" = "turquoise",
      "珠三角地区" = "lightcoral"
    ),
    breaks = c("京津冀地区", "长三角地区", "珠三角地区")
  ) +
  # 添加比例尺（左下角，稍微靠右以避免与图例重叠）
  annotation_scale(location = "bl", width_hint = 0.3, pad_x = unit(0.3, "cm"), pad_y = unit(0.3, "cm")) +
  # 添加指北针（左上角）
  annotation_north_arrow(
    location = "tl",
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
    legend.spacing.y = unit(1.5, "cm"),      # 图例项之间的垂直间距
    legend.key.height = unit(0.5, "cm"),     # 图例键（颜色块）的高度
    legend.margin = margin(t = 15, r = 15, b = 15, l = 15)  # 图例的边距
  ) +
  labs(
    title = "中国地图（含省级边界）",
    x = "经度 (°E)",
    y = "纬度 (°N)"
  ) +
  coord_sf(
    crs = laea_crs,
    expand = FALSE,
    # 设置经纬度网格线的位置和标签
    # default_crs 指定数据源的坐标系（WGS84经纬度）
    default_crs = st_crs(4326)
  )

print(china_province_map)

# 保存为高分辨率SVG图片
ggsave(
  filename = "中国地图_研究区域.svg",
  plot = china_province_map,
  device = "svg",
  width = 12,      # 宽度（英寸）
  height = 12,      # 高度（英寸）
  units = "in",    # 单位：英寸
  dpi = 300        # 虽然SVG是矢量格式，但此参数用于某些转换
)
