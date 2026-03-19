library(ggplot2)
library(dplyr)
library(sf)
library(ggspatial)
library(tidyverse)
library(scales)

# 设置工作目录
setwd("C:/Users/IU/Desktop/大论文图")

# 设置LAEA投影（局部等面积投影），中心点设置为广州地区（珠三角中心）
laea_crs <- st_crs("+proj=laea +lat_0=23.1 +lon_0=113.2 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

# 读取并转换投影的辅助函数
read_and_transform <- function(file_path, crs = laea_crs) {
  st_read(file_path) %>% st_transform(crs = crs)
}

# 读取GDP数据（珠三角地区：广东省）
guangdong_gdp <- read.csv("五大城市群的具体GDP/广东省各区GDP（2023-2024）.csv", stringsAsFactors = FALSE)

# 检查数据结构
cat("=== 广东省GDP数据结构 ===\n")
print(head(guangdong_gdp))
cat("列名:", paste(colnames(guangdong_gdp), collapse = ", "), "\n")

# 清理和整理GDP数据
# 筛选2024年level=2的数据，并只保留珠三角9个城市的数据
# 首先检查列名
cat("=== 检查列名 ===\n")
print(colnames(guangdong_gdp))

# 使用列的位置来选择数据（更安全的方法）
guangdong_gdp_clean <- guangdong_gdp %>%
  filter(年份 == 2024 & level == 2) %>%
  select(
    区域名称 = 3,  # 县市区列
    城市 = 2,      # 市列
    GDP_2024 = 6    # GDP(亿)列
  ) %>%
  mutate(
    省份 = "广东省",
    GDP_2024 = as.numeric(GDP_2024)
  ) %>%
  # 只保留珠三角9个城市的数据
  filter(城市 %in% c("广州市", "深圳市", "佛山市", "东莞市", "中山市", "惠州市", "珠海市", "江门市", "肇庆市"))

# 使用清理后的GDP数据
gdp_data <- guangdong_gdp_clean %>%
  filter(!is.na(GDP_2024) & GDP_2024 > 0)

# 读取地理数据（珠三角9个城市的区级/市级数据）
# 注意：东莞市和中山市是直筒子市，没有区县划分，使用level == "city"
guangzhou_geo <- read_and_transform("4.珠三角/具体城市/广州市 (区).geojson") %>%
  filter(level == "district")

shenzhen_geo <- read_and_transform("4.珠三角/具体城市/深圳市 (区).geojson") %>%
  filter(level == "district")

foshan_geo <- read_and_transform("4.珠三角/具体城市/佛山市 (区).geojson") %>%
  filter(level == "district")

# 东莞市是直筒子市，使用city级别
dongguan_geo <- read_and_transform("4.珠三角/具体城市/东莞市 (区).geojson") %>%
  filter(level == "city")

# 中山市是直筒子市，使用city级别
zhongshan_geo <- read_and_transform("4.珠三角/具体城市/中山市 (区).geojson") %>%
  filter(level == "city")

huizhou_geo <- read_and_transform("4.珠三角/具体城市/惠州市 (区).geojson") %>%
  filter(level == "district")

zhuhai_geo <- read_and_transform("4.珠三角/具体城市/珠海市 (区).geojson") %>%
  filter(level == "district")

jiangmen_geo <- read_and_transform("4.珠三角/具体城市/江门市 (区).geojson") %>%
  filter(level == "district")

zhaoqing_geo <- read_and_transform("4.珠三角/具体城市/肇庆市 (区).geojson") %>%
  filter(level == "district")

# 检查地理数据的列名（以广州为例）
cat("=== 广州地理数据列名 ===\n")
print(colnames(guangzhou_geo))
cat("广州地理数据前几行:\n")
print(head(guangzhou_geo[, c("name", "level")]))

# 合并所有城市的地理数据
geo_data <- bind_rows(
  guangzhou_geo, shenzhen_geo, foshan_geo, dongguan_geo, zhongshan_geo,
  huizhou_geo, zhuhai_geo, jiangmen_geo, zhaoqing_geo
)

# 数据合并
# 地理数据的name字段对应GDP数据的区域名称
merged_data <- geo_data %>%
  left_join(gdp_data, by = c("name" = "区域名称"))

# 创建GDP分级标签
breaks <- c(0, 500, 1000, 2000, 5000, Inf)
labels <- c("0-500", "500-1000", "1000-2000", "2000-5000", "5000+")

merged_data <- merged_data %>%
  mutate(GDP_category = cut(GDP_2024, breaks = breaks, labels = labels, include.lowest = TRUE))

# 获取地图边界框
map_bbox <- st_bbox(merged_data)

# 创建经纬度网格线
graticule <- st_graticule(
  merged_data %>% st_transform(crs = st_crs(4326)),
  lat = seq(21, 25, by = 1),
  lon = seq(110, 116, by = 2)
) %>% st_transform(crs = laea_crs)

# 创建GDP地图
gdp_map <- ggplot() +
  # 添加经纬度网格线
  geom_sf(data = graticule, color = "gray80", linewidth = 0.3, linetype = "dashed") +
  # 添加地理边界
  geom_sf(data = merged_data, aes(fill = GDP_2024), color = "black", linewidth = 0.3)

# 添加主题和样式
gdp_map_styled <- gdp_map +
  # GDP颜色渐变
  scale_fill_gradientn(
    name = "GDP (亿元)",
    colors = c("#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"),
    values = scales::rescale(c(0, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000)),
    na.value = "gray90",
    labels = comma
  ) +
  # 添加比例尺
  annotation_scale(
    location = "br",
    width_hint = 0.3,
    pad_x = unit(0.5, "cm"),
    pad_y = unit(0.5, "cm")
  ) +
  # 添加指北针
  annotation_north_arrow(
    location = "tl",
    which_north = "true",
    style = north_arrow_fancy_orienteering,
    height = unit(1.5, "cm"),
    width = unit(1.5, "cm"),
    pad_x = unit(0.5, "cm"),
    pad_y = unit(0.5, "cm")
  ) +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    axis.text = element_text(size = 12, color = "black"),
    axis.ticks = element_line(color = "black", linewidth = 0.5),
    axis.title = element_text(size = 14, face = "bold"),
    panel.background = element_rect(fill = NA, color = NA),
    plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
    legend.position = c(0.95, 0.05),
    legend.justification = c(1, 0),
    legend.title = element_text(size = 14, face = "bold"),
    legend.text = element_text(size = 12),
    legend.background = element_blank(),
    legend.key.width = unit(1.5, "cm"),
    legend.key.height = unit(0.8, "cm")
  ) +
  labs(
    title = "珠三角地区2024年GDP分布图",
    x = "经度 (°E)",
    y = "纬度 (°N)"
  ) +
  coord_sf(
    crs = laea_crs,
    expand = FALSE,
    default_crs = st_crs(4326)
  )

# 显示地图
print(gdp_map_styled)

# 保存为高分辨率图片
ggsave(
  filename = "珠三角_GDP_2024.svg",
  plot = gdp_map_styled,
  device = "svg",
  width = 16,
  height = 12,
  units = "in",
  dpi = 300
)

# 同时保存为PNG格式
ggsave(
  filename = "珠三角_GDP_2024.png",
  plot = gdp_map_styled,
  device = "png",
  width = 16,
  height = 12,
  units = "in",
  dpi = 300
)

# 检查缺失GDP数据的区域
missing_gdp_regions <- merged_data %>%
  filter(is.na(GDP_2024)) %>%
  mutate(省份 = "广东省") %>%
  select(name, 省份) %>%
  arrange(省份, name)

# 输出数据统计信息
cat("\n=== 数据统计信息 ===\n")
cat("总区域数量:", nrow(merged_data), "\n")
cat("有GDP数据的区域数量:", sum(!is.na(merged_data$GDP_2024)), "\n")
cat("缺失GDP数据的区域数量:", nrow(missing_gdp_regions), "\n")

if (nrow(missing_gdp_regions) > 0) {
  cat("\n=== 缺失GDP数据的区域列表 ===\n")
  for (i in seq_len(nrow(missing_gdp_regions))) {
    cat(sprintf("%s: %s\n", missing_gdp_regions$省份[i], missing_gdp_regions$name[i]))
  }

  # 将缺失数据区域保存到文件
  write.csv(missing_gdp_regions,
            file = "珠三角_缺失GDP数据区域.csv",
            row.names = FALSE,
            fileEncoding = "UTF-8")
  cat("\n缺失GDP数据的区域已保存到: 珠三角_缺失GDP数据区域.csv\n")
} else {
  cat("所有区域都有对应的GDP数据！\n")
}

cat("\n=== GDP数据统计 ===\n")
if (sum(!is.na(merged_data$GDP_2024)) > 0) {
  cat("GDP数据范围: ", min(merged_data$GDP_2024, na.rm = TRUE), " - ", max(merged_data$GDP_2024, na.rm = TRUE), " 亿元\n")
  cat("GDP数据平均值: ", round(mean(merged_data$GDP_2024, na.rm = TRUE), 2), " 亿元\n")
  cat("GDP数据中位数: ", round(median(merged_data$GDP_2024, na.rm = TRUE), 2), " 亿元\n")
} else {
  cat("没有有效的GDP数据\n")
}