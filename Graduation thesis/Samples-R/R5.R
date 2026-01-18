# 加载所需的包
pacman::p_load(sf, raster, ggplot2, tidyverse)

# 读取矢量边界
shp <- sf::read_sf("E:/data/wh/shp/bj.shp")

# 读取栅格数据
ndvi <- raster("E:/test/NDVI/2000year_mean.tif")

# 将栅格数据转换为数据框
df <- as.data.frame(ndvi, xy=T)
colnames(df) = c("x", "y", "LandCover") # 数据转换为dataframe

# 创建地图
p1 <- ggplot(df %>% na.omit()) +
  geom_raster(aes(x, y, fill = LandCover)) +
  scale_fill_gradientn(colours = rainbow(10)) +
  labs(x = NULL, y = NULL) +
  geom_sf(size = .2, fill = "transparent", color = "grey", data = shp) + # 绘制矢量边界
  theme_bw() +
  theme(
    panel.grid.major = element_blank(), # 去除始图网格
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    legend.title = element_blank(),
    axis.ticks.length = unit(-0.1, "cm"),
    axis.text.x = element_text(margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm")),
    axis.text.y = element_text(margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm"))
  ) # 旋转坐标轴label的方向

# 打印地图
p1