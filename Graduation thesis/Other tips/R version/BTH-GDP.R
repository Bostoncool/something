library(ggplot2)
library(dplyr)
library(sf)
library(ggspatial)
library(tidyverse)
library(scales)

# 设置工作目录
setwd("C:/Users/IU/Desktop/大论文图")

# 设置LAEA投影（局部等面积投影），中心点设置为北京地区
laea_crs <- st_crs("+proj=laea +lat_0=39.9 +lon_0=116.4 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

# 读取并转换投影的辅助函数
read_and_transform <- function(file_path, crs = laea_crs) {
  st_read(file_path) %>% st_transform(crs = crs)
}

# 读取GDP数据（直接读取为CSV格式）
beijing_gdp <- read.csv("五大城市群的具体GDP/北京市各区GDP（2023-2024）.csv", stringsAsFactors = FALSE)
tianjin_gdp <- read.csv("五大城市群的具体GDP/天津市各区GDP（2023-2024）.csv", stringsAsFactors = FALSE)
hebei_gdp <- read.csv("五大城市群的具体GDP/河北省各区GDP（2023-2024）.csv", stringsAsFactors = FALSE)

# 检查数据结构
cat("=== 北京GDP数据结构 ===\n")
print(head(beijing_gdp))
cat("列名:", paste(colnames(beijing_gdp), collapse = ", "), "\n")

cat("\n=== 天津GDP数据结构 ===\n")
print(head(tianjin_gdp))
cat("列名:", paste(colnames(tianjin_gdp), collapse = ", "), "\n")

cat("\n=== 河北GDP数据结构 ===\n")
print(head(hebei_gdp))
cat("列名:", paste(colnames(hebei_gdp), collapse = ", "), "\n")

# 清理和整理GDP数据
# 筛选2024年level=2的数据
beijing_gdp_clean <- beijing_gdp %>%
  filter(年份 == 2024 & level == 2) %>%
  select(区域名称 = 县市区, GDP_2024 = `GDP.亿.`) %>%
  mutate(省份 = "北京市", GDP_2024 = as.numeric(GDP_2024))

tianjin_gdp_clean <- tianjin_gdp %>%
  filter(年份 == 2024 & level == 2) %>%
  select(区域名称 = 县市区, GDP_2024 = `GDP.亿.`) %>%
  mutate(省份 = "天津市", GDP_2024 = as.numeric(GDP_2024))

hebei_gdp_clean <- hebei_gdp %>%
  filter(年份 == 2024 & level == 2) %>%
  select(区域名称 = 县市区, GDP_2024 = `GDP.亿.`) %>%
  mutate(省份 = "河北省", GDP_2024 = as.numeric(GDP_2024))

# 合并所有GDP数据
gdp_data <- bind_rows(beijing_gdp_clean, tianjin_gdp_clean, hebei_gdp_clean) %>%
  filter(!is.na(GDP_2024) & GDP_2024 > 0)

# 读取地理数据并筛选level=2的区域
beijing_geo <- read_and_transform("2.京津冀/北京市 (划分).geojson") %>%
  filter(level == "district")

tianjin_geo <- read_and_transform("2.京津冀/天津市 (划分).geojson") %>%
  filter(level == "district")

hebei_geo <- read_and_transform("2.京津冀/河北省 (区).geojson") %>%
  filter(level == "district")

# 检查地理数据的列名
cat("=== 北京地理数据列名 ===\n")
print(colnames(beijing_geo))
cat("北京地理数据前几行:\n")
print(head(beijing_geo[, c("name", "level")]))

cat("\n=== 天津地理数据列名 ===\n")
print(colnames(tianjin_geo))
cat("天津地理数据前几行:\n")
print(head(tianjin_geo[, c("name", "level")]))

cat("\n=== 河北地理数据列名 ===\n")
print(colnames(hebei_geo))
cat("河北地理数据前几行:\n")
print(head(hebei_geo[, c("name", "level")]))

# 合并地理数据
geo_data <- bind_rows(beijing_geo, tianjin_geo, hebei_geo)

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
  lat = seq(35, 45, by = 2),
  lon = seq(110, 125, by = 2)
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
    title = "京津冀地区2024年GDP分布图",
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
  filename = "京津冀_GDP_2024.svg",
  plot = gdp_map_styled,
  device = "svg",
  width = 16,
  height = 12,
  units = "in",
  dpi = 300
)

# 同时保存为PNG格式
ggsave(
  filename = "京津冀_GDP_2024.png",
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
  mutate(省份 = case_when(
    name %in% c("东城区", "西城区", "朝阳区", "丰台区", "石景山区", "海淀区", "门头沟区",
                "房山区", "通州区", "顺义区", "昌平区", "大兴区", "怀柔区", "平谷区",
                "密云区", "延庆区") ~ "北京市",
    name %in% c("和平区", "河东区", "河西区", "南开区", "河北区", "红桥区", "东丽区",
                "西青区", "津南区", "北辰区", "武清区", "宝坻区", "滨海新区", "宁河区",
                "静海区", "蓟州区") ~ "天津市",
    TRUE ~ "河北省"
  )) %>%
  select(name, 省份) %>%
  arrange(省份, name)

# 输出数据统计信息
cat("\n=== 数据统计信息 ===\n")
cat("总区域数量:", nrow(merged_data), "\n")
cat("有GDP数据的区域数量:", sum(!is.na(merged_data$GDP_2024)), "\n")
cat("缺失GDP数据的区域数量:", nrow(missing_gdp_regions), "\n")

if (nrow(missing_gdp_regions) > 0) {
  cat("\n=== 缺失GDP数据的区域列表 ===\n")
  for (i in 1:nrow(missing_gdp_regions)) {
    cat(sprintf("%s: %s\n", missing_gdp_regions$省份[i], missing_gdp_regions$name[i]))
  }

  # 将缺失数据区域保存到文件
  write.csv(missing_gdp_regions,
            file = "京津冀_缺失GDP数据区域.csv",
            row.names = FALSE,
            fileEncoding = "UTF-8")
  cat("\n缺失GDP数据的区域已保存到: 京津冀_缺失GDP数据区域.csv\n")
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