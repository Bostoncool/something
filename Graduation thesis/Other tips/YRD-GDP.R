library(ggplot2)
library(dplyr)
library(sf)
library(ggspatial)
library(tidyverse)
library(scales)

# 设置工作目录
setwd("C:/Users/IU/Desktop/大论文图")

# 设置LAEA投影（局部等面积投影），中心点设置为南京地区（长三角中心）
laea_crs <- st_crs("+proj=laea +lat_0=32.0 +lon_0=118.8 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

# 读取并转换投影的辅助函数
read_and_transform <- function(file_path, crs = laea_crs) {
  st_read(file_path) %>% st_transform(crs = crs)
}

# 读取GDP数据（长三角地区：上海、江苏、浙江、安徽）
shanghai_gdp <- read.csv("五大城市群的具体GDP/上海市各区GDP（2023-2024）.csv", stringsAsFactors = FALSE)
jiangsu_gdp <- read.csv("五大城市群的具体GDP/江苏省各区GDP（2023-2024）.csv", stringsAsFactors = FALSE)
zhejiang_gdp <- read.csv("五大城市群的具体GDP/浙江省各区GDP（2023-2024）.csv", stringsAsFactors = FALSE)
anhui_gdp <- read.csv("五大城市群的具体GDP/安徽省各区GDP（2023-2024）.csv", stringsAsFactors = FALSE)

# 检查数据结构
cat("=== 上海GDP数据结构 ===\n")
print(head(shanghai_gdp))
cat("列名:", paste(colnames(shanghai_gdp), collapse = ", "), "\n")

cat("\n=== 江苏GDP数据结构 ===\n")
print(head(jiangsu_gdp))
cat("列名:", paste(colnames(jiangsu_gdp), collapse = ", "), "\n")

cat("\n=== 浙江GDP数据结构 ===\n")
print(head(zhejiang_gdp))
cat("列名:", paste(colnames(zhejiang_gdp), collapse = ", "), "\n")

cat("\n=== 安徽GDP数据结构 ===\n")
print(head(anhui_gdp))
cat("列名:", paste(colnames(anhui_gdp), collapse = ", "), "\n")

# 清理和整理GDP数据
# 筛选2024年level=2的数据
shanghai_gdp_clean <- shanghai_gdp %>%
  filter(年份 == 2024 & level == 2) %>%
  select(区域名称 = 县市区, GDP_2024 = `GDP.亿.`) %>%
  mutate(省份 = "上海市", GDP_2024 = as.numeric(GDP_2024))

jiangsu_gdp_clean <- jiangsu_gdp %>%
  filter(年份 == 2024 & level == 2) %>%
  select(区域名称 = 县市区, GDP_2024 = `GDP.亿.`) %>%
  mutate(省份 = "江苏省", GDP_2024 = as.numeric(GDP_2024))

zhejiang_gdp_clean <- zhejiang_gdp %>%
  filter(年份 == 2024 & level == 2) %>%
  select(区域名称 = 县市区, GDP_2024 = `GDP.亿.`) %>%
  mutate(省份 = "浙江省", GDP_2024 = as.numeric(GDP_2024))

anhui_gdp_clean <- anhui_gdp %>%
  filter(年份 == 2024 & level == 2) %>%
  select(区域名称 = 县市区, GDP_2024 = `GDP.亿.`) %>%
  mutate(省份 = "安徽省", GDP_2024 = as.numeric(GDP_2024))

# 合并所有GDP数据
gdp_data <- bind_rows(shanghai_gdp_clean, jiangsu_gdp_clean, zhejiang_gdp_clean, anhui_gdp_clean) %>%
  filter(!is.na(GDP_2024) & GDP_2024 > 0)

# 读取地理数据（由于分区太多，使用南京市作为样本案例）
# 注意：实际应用中需要为每个城市准备对应的地理数据文件
# 读取长三角典型城市的区级地理数据
nanjing_geo    <- read_and_transform("3.长三角/具体城市/南京市 (区).geojson")
wuxi_geo       <- read_and_transform("3.长三角/具体城市/无锡市 (区).geojson")
nantong_geo    <- read_and_transform("3.长三角/具体城市/南通市 (区).geojson")
yancheng_geo   <- read_and_transform("3.长三角/具体城市/盐城市 (区).geojson")
yangzhou_geo   <- read_and_transform("3.长三角/具体城市/扬州市 (区).geojson")
zhenjiang_geo  <- read_and_transform("3.长三角/具体城市/镇江市 (区).geojson")
changzhou_geo  <- read_and_transform("3.长三角/具体城市/常州市 (区).geojson")
suzhou_geo     <- read_and_transform("3.长三角/具体城市/苏州市 (区).geojson")
taizhou_geo    <- read_and_transform("3.长三角/具体城市/泰州市 (区).geojson")

hangzhou_geo   <- read_and_transform("3.长三角/具体城市/杭州市 (区).geojson")
ningbo_geo     <- read_and_transform("3.长三角/具体城市/宁波市 (区).geojson")
jiaxing_geo    <- read_and_transform("3.长三角/具体城市/嘉兴市 (区).geojson")
huzhou_geo     <- read_and_transform("3.长三角/具体城市/湖州市 (区).geojson")
shaoxing_geo   <- read_and_transform("3.长三角/具体城市/绍兴市 (区).geojson")
jinhua_geo     <- read_and_transform("3.长三角/具体城市/金华市 (区).geojson")
zhoushan_geo   <- read_and_transform("3.长三角/具体城市/舟山市 (区).geojson")
taizhou_zj_geo <- read_and_transform("3.长三角/具体城市/台州市 (区).geojson")
wenzhou_geo    <- read_and_transform("3.长三角/具体城市/温州市 (区).geojson")

hefei_geo      <- read_and_transform("3.长三角/具体城市/合肥市 (区).geojson")
wuhu_geo       <- read_and_transform("3.长三角/具体城市/芜湖市 (区).geojson")
maanshan_geo   <- read_and_transform("3.长三角/具体城市/马鞍山市 (区).geojson")
tongling_geo   <- read_and_transform("3.长三角/具体城市/铜陵市 (区).geojson")
anqing_geo     <- read_and_transform("3.长三角/具体城市/安庆市 (区).geojson")
chuzhou_geo    <- read_and_transform("3.长三角/具体城市/滁州市 (区).geojson")
chizhou_geo    <- read_and_transform("3.长三角/具体城市/池州市 (区).geojson")
xuancheng_geo  <- read_and_transform("3.长三角/具体城市/宣城市 (区).geojson")

shanghai_geo    <- read_and_transform("3.长三角/上海市 (划分).geojson") %>%
  filter(level == "district")

# 合并所有城市的地理数据
# 注意：确保所有地理数据都有相同的列结构
geo_data <- bind_rows(
  nanjing_geo, wuxi_geo, nantong_geo, yancheng_geo, yangzhou_geo, zhenjiang_geo,
  changzhou_geo, suzhou_geo, taizhou_geo, hangzhou_geo, ningbo_geo, jiaxing_geo,
  huzhou_geo, shaoxing_geo, jinhua_geo, zhoushan_geo, taizhou_zj_geo, wenzhou_geo,
  hefei_geo, wuhu_geo, maanshan_geo, tongling_geo, anqing_geo, chuzhou_geo,
  chizhou_geo, xuancheng_geo, shanghai_geo
)

# 检查地理数据的列名
cat("=== 合并后地理数据列名 ===\n")
print(colnames(geo_data))
cat("合并后地理数据行数:", nrow(geo_data), "\n")
cat("合并后地理数据前几行:\n")
print(head(geo_data[, c("name", "level")]))

# 数据合并
# 地理数据的name字段对应GDP数据的区域名称
# 由于包含多个城市的地理数据，需要更精确的匹配
merged_data <- geo_data %>%
  left_join(gdp_data, by = c("name" = "区域名称"))

# 检查匹配情况
cat("\n=== 数据合并情况 ===\n")
cat("地理数据区域数量:", nrow(geo_data), "\n")
cat("GDP数据区域数量:", nrow(gdp_data), "\n")
cat("成功匹配的区域数量:", sum(!is.na(merged_data$GDP_2024)), "\n")

# 如果匹配率低，尝试更复杂的名称映射
if (sum(!is.na(merged_data$GDP_2024)) < nrow(geo_data) * 0.5) {
  cat("直接匹配成功率较低，尝试名称映射...\n")

  # 创建更复杂的名称映射
  merged_data <- geo_data %>%
    mutate(name_clean = case_when(
      # 南京市特殊处理
      name == "鼓楼区" & level == "district" ~ "南京鼓楼区",
      # 其他城市的处理可以在这里添加
      TRUE ~ name
    )) %>%
    left_join(gdp_data, by = c("name_clean" = "区域名称")) %>%
    select(-name_clean)

  cat("名称映射后匹配数量:", sum(!is.na(merged_data$GDP_2024)), "\n")
}

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
  lat = seq(28, 35, by = 2),
  lon = seq(114, 123, by = 2)
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
    title = "长三角地区2024年GDP分布图",
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
  filename = "长三角_GDP_2024.svg",
  plot = gdp_map_styled,
  device = "svg",
  width = 20,
  height = 16,
  units = "in",
  dpi = 300
)

# 同时保存为PNG格式
ggsave(
  filename = "长三角_GDP_2024.png",
  plot = gdp_map_styled,
  device = "png",
  width = 20,
  height = 16,
  units = "in",
  dpi = 300
)

# 检查缺失GDP数据的区域
missing_gdp_regions <- merged_data %>%
  filter(is.na(GDP_2024)) %>%
  select(name, level) %>%
  arrange(name)

# 输出数据统计信息
cat("\n=== 数据统计信息 ===\n")
cat("总区域数量:", nrow(merged_data), "\n")
cat("有GDP数据的区域数量:", sum(!is.na(merged_data$GDP_2024)), "\n")
cat("缺失GDP数据的区域数量:", nrow(missing_gdp_regions), "\n")

if (nrow(missing_gdp_regions) > 0) {
  cat("\n=== 缺失GDP数据的区域列表 ===\n")
  for (i in 1:nrow(missing_gdp_regions)) {
    cat(sprintf("%s\n", missing_gdp_regions$name[i]))
  }

  # 将缺失数据区域保存到文件
  write.csv(missing_gdp_regions,
            file = "长三角_缺失GDP数据区域.csv",
            row.names = FALSE,
            fileEncoding = "UTF-8")
  cat("\n缺失GDP数据的区域已保存到: 长三角_缺失GDP数据区域.csv\n")
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

# 输出长三角地区GDP数据概览
cat("\n=== 长三角地区GDP数据概览 ===\n")
gdp_summary <- gdp_data %>%
  group_by(省份) %>%
  summarise(
    区域数量 = n(),
    平均GDP = round(mean(GDP_2024, na.rm = TRUE), 2),
    最大GDP = round(max(GDP_2024, na.rm = TRUE), 2),
    最小GDP = round(min(GDP_2024, na.rm = TRUE), 2)
  )
print(gdp_summary)

cat("\n=== 脚本执行完成 ===\n")
cat("生成的文件：\n")
cat("- 长三角_GDP_2024.svg\n")
cat("- 长三角_GDP_2024.png\n")
cat("- 长三角_缺失GDP数据区域.csv（如果有缺失数据）\n")