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

# 读取人口密度数据（直接读取为CSV格式）
beijing_people <- read.csv("五大城市群的具体人口与占地面积/北京市各区人口与占地面积（2024）.csv", stringsAsFactors = FALSE)
tianjin_people <- read.csv("五大城市群的具体人口与占地面积/天津市各区人口与占地面积（2024）.csv", stringsAsFactors = FALSE)
hebei_people <- read.csv("五大城市群的具体人口与占地面积/河北省各区人口与占地面积（2024）.csv", stringsAsFactors = FALSE)

# 检查数据结构
cat("=== 北京人口数据结构 ===\n")
print(head(beijing_people))
cat("列名:", paste(colnames(beijing_people), collapse = ", "), "\n")

cat("\n=== 天津人口数据结构 ===\n")
print(head(tianjin_people))
cat("列名:", paste(colnames(tianjin_people), collapse = ", "), "\n")

cat("\n=== 河北人口数据结构 ===\n")
print(head(hebei_people))
cat("列名:", paste(colnames(hebei_people), collapse = ", "), "\n")

# 清理和整理人口数据
# 筛选区县级别的数据（排除省级汇总数据）
beijing_people_clean <- beijing_people %>%
  filter(column.1 != "-") %>%  # 排除北京市汇总行
  mutate(
    区域名称 = column.2,
    人口_万人 = as.numeric(gsub("万$", "", column.3)),  # 移除"万"单位并转换为数值
    面积_km2 = as.numeric(gsub("km²$", "", column.4)),  # 移除"km²"单位并转换为数值
    省份 = "北京市"
  ) %>%
  select(区域名称, 人口_万人, 面积_km2, 省份) %>%
  filter(!is.na(面积_km2) & 面积_km2 > 0) %>%  # 过滤掉面积数据缺失或为0的行
  mutate(人口密度_人每平方公里 = (人口_万人 * 10000) / 面积_km2)  # 计算人口密度

tianjin_people_clean <- tianjin_people %>%
  filter(column.1 != "-") %>%  # 排除天津市汇总行
  mutate(
    区域名称 = column.2,
    人口_万人 = as.numeric(gsub("万$", "", column.3)),  # 移除"万"单位并转换为数值
    面积_km2 = as.numeric(gsub("km²$", "", column.4)),  # 移除"km²"单位并转换为数值
    省份 = "天津市"
  ) %>%
  select(区域名称, 人口_万人, 面积_km2, 省份) %>%
  filter(!is.na(面积_km2) & 面积_km2 > 0) %>%  # 过滤掉面积数据缺失或为0的行
  mutate(人口密度_人每平方公里 = (人口_万人 * 10000) / 面积_km2)  # 计算人口密度

hebei_people_clean <- hebei_people %>%
  filter(column.1 != "-") %>%  # 排除河北省汇总行
  mutate(
    区域名称 = column.2,
    人口_万人 = as.numeric(gsub("万$", "", column.3)),  # 移除"万"单位并转换为数值
    面积_km2 = as.numeric(gsub("km²$", "", column.4)),  # 移除"km²"单位并转换为数值
    省份 = "河北省"
  ) %>%
  select(区域名称, 人口_万人, 面积_km2, 省份) %>%
  filter(!is.na(面积_km2) & 面积_km2 > 0) %>%  # 过滤掉面积数据缺失或为0的行
  mutate(人口密度_人每平方公里 = (人口_万人 * 10000) / 面积_km2)  # 计算人口密度

# 合并所有人口密度数据
people_data <- bind_rows(beijing_people_clean, tianjin_people_clean, hebei_people_clean) %>%
  filter(!is.na(人口密度_人每平方公里) & 人口密度_人每平方公里 > 0 & !is.infinite(人口密度_人每平方公里))

# 读取地理数据并筛选相应级别的区域
beijing_geo <- read_and_transform("2.京津冀/北京市 (划分).geojson") %>%
  filter(level == "district")

tianjin_geo <- read_and_transform("2.京津冀/天津市 (划分).geojson") %>%
  filter(level == "district")

hebei_geo <- read_and_transform("2.京津冀/河北省 (市).geojson") %>%
  filter(level == "city")

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
# 地理数据的name字段对应人口数据的区域名称
merged_data <- geo_data %>%
  left_join(people_data, by = c("name" = "区域名称"))

# 创建人口密度分级标签
breaks <- c(0, 500, 1000, 2000, 5000, 10000, Inf)
labels <- c("0-500", "500-1000", "1000-2000", "2000-5000", "5000-10000", "10000+")

merged_data <- merged_data %>%
  mutate(人口密度类别 = cut(人口密度_人每平方公里, breaks = breaks, labels = labels, include.lowest = TRUE))

# 获取地图边界框
map_bbox <- st_bbox(merged_data)

# 创建经纬度网格线
graticule <- st_graticule(
  merged_data %>% st_transform(crs = st_crs(4326)),
  lat = seq(35, 45, by = 2),
  lon = seq(110, 125, by = 2)
) %>% st_transform(crs = laea_crs)

# 创建人口密度地图
people_density_map <- ggplot() +
  # 添加经纬度网格线
  geom_sf(data = graticule, color = "gray80", linewidth = 0.3, linetype = "dashed") +
  # 添加地理边界
  geom_sf(data = merged_data, aes(fill = 人口密度_人每平方公里), color = "black", linewidth = 0.3)

# 添加主题和样式
people_density_map_styled <- people_density_map +
  # 人口密度颜色渐变（从浅红色到深红色，表示人口密度从低到高）
  scale_fill_gradientn(
    name = "人口密度\n(人/平方公里)",
    colors = c("#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"),
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
    title = "京津冀地区2024年人口密度分布图",
    x = "经度 (°E)",
    y = "纬度 (°N)"
  ) +
  coord_sf(
    crs = laea_crs,
    expand = FALSE,
    default_crs = st_crs(4326)
  )

# 显示地图
print(people_density_map_styled)

# 保存为高分辨率图片
ggsave(
  filename = "京津冀_人口密度_2024.svg",
  plot = people_density_map_styled,
  device = "svg",
  width = 16,
  height = 12,
  units = "in",
  dpi = 300
)

# 同时保存为PNG格式
ggsave(
  filename = "京津冀_人口密度_2024.png",
  plot = people_density_map_styled,
  device = "png",
  width = 16,
  height = 12,
  units = "in",
  dpi = 300
)

# 检查缺失人口数据的区域
missing_people_regions <- merged_data %>%
  filter(is.na(人口密度_人每平方公里)) %>%
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
cat("有人口数据的区域数量:", sum(!is.na(merged_data$人口密度_人每平方公里)), "\n")
cat("缺失人口数据的区域数量:", nrow(missing_people_regions), "\n")

if (nrow(missing_people_regions) > 0) {
  cat("\n=== 缺失人口数据的区域列表 ===\n")
  for (i in 1:nrow(missing_people_regions)) {
    cat(sprintf("%s: %s\n", missing_people_regions$省份[i], missing_people_regions$name[i]))
  }

  # 将缺失数据区域保存到文件
  write.csv(missing_people_regions,
            file = "京津冀_缺失人口数据区域.csv",
            row.names = FALSE,
            fileEncoding = "UTF-8")
  cat("\n缺失人口数据的区域已保存到: 京津冀_缺失人口数据区域.csv\n")
} else {
  cat("所有区域都有对应的人口数据！\n")
}

cat("\n=== 人口密度数据统计 ===\n")
if (sum(!is.na(merged_data$人口密度_人每平方公里)) > 0) {
  cat("人口密度范围: ", round(min(merged_data$人口密度_人每平方公里, na.rm = TRUE), 2), " - ", round(max(merged_data$人口密度_人每平方公里, na.rm = TRUE), 2), " 人/平方公里\n")
  cat("人口密度平均值: ", round(mean(merged_data$人口密度_人每平方公里, na.rm = TRUE), 2), " 人/平方公里\n")
  cat("人口密度中位数: ", round(median(merged_data$人口密度_人每平方公里, na.rm = TRUE), 2), " 人/平方公里\n")
} else {
  cat("没有有效的人口密度数据\n")
}