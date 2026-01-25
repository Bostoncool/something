library(tidyverse)
library(sf)
china_shp <- "中华人民共和国.json"
china <- sf::read_sf(china_shp)
data <- read.table("data.txt",header = T,sep = '\t')
# 确保匹配列的名称相同（china 数据框中省份名称列为 'name'）
# 添加数据列到 GeoJSON 数据中
china <- china %>%
  left_join(data, by = "name")
# 检查结果
print(china)
library(ggspatial) #添加比例尺和指北针的R包
# 可视化绘制地图，展示人口数据
ggplot(data = china) +
  geom_sf(aes(fill = investment), color = "black") +  
  coord_sf(crs = "+proj=laea +lat_0=40 +lon_0=104")+ #局部等面积投影（LAEA）坐标系
  scale_fill_viridis_c(option = "magma", direction = -1, na.value = "grey90") + # 使用 magma 配色方案
  annotation_scale(location = "bl") +
  annotation_north_arrow(location = "tl", which_north = "false",
                         style = north_arrow_fancy_orienteering)+
  labs(title = '2024年12月全国各省房地产开发投资情况（亿元）', # 图标题
       fill = '金额（亿元）')+ # 图例标题
  theme(
    panel.background = element_rect(fill = "white", color = "black"), # 白色背景
    panel.grid.major = element_blank(), # 去除网格线
    panel.grid.minor = element_blank(), 
    legend.position = "right", # 图例放在右侧
    legend.title = element_text(size = 14, face = "bold"), # 图例标题加粗
    legend.text = element_text(size = 12), # 图例文字大小
    axis.text = element_blank(), # 去除坐标轴文字
    axis.ticks = element_blank(), # 去除坐标轴刻度
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5) # 标题居中加粗
  )
