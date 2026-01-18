pacman::p_load(tidyverse, sf, raster, ggspatial, ggplot2,cowplot,stars)
crs_84 <- st_crs("EPSG:4326")  ## WGS 84 大地坐标
crs_al <- st_crs("+proj=aea +lat_1=25 +lat_2=47 +lon_0=105") ## Albers Equal Area Conic投影
ndvi <- raster("E:/test/NDVI/2001year_mean.tif")
df=as.data.frame(ndvi,xy=T)
colnames(df)=c("x","y","LandCover")#数据转化为dataframe
china_all <-
  sf::st_read("https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json") %>%
  st_transform(crs_84)
shp2 <- sf::read_sf("E:/data/wh/shp/bj.shp")
p1 <-
  ggplot() +
  geom_sf(size = .2, fill = "transparent", color = "black", data = china_all) +
  geom_sf(data=shp2,fill="NA",size=0.2,color="red")+
  theme_void()
p2=ggplot(df %>% na.omit())+
  geom_raster(aes(x,y,fill=LandCover))+
  scale_fill_gradientn(colours =rainbow(10))+
  labs(x=NULL,y=NULL)+
  geom_sf(data=shp2,fill="NA",size=0.2,color="red")+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background=element_blank(),
        legend.title=element_blank(),
        legend.position="non"
  )
ggdraw()+draw_plot(p1,x=-0.35,scale=0.3)+draw_plot(p2,x=0.135,scale=0.7)+
  geom_segment(aes(x=0.19,y=0.47,xend=0.57,yend=0.295),size=0.5)+ ## 设置两根线的起点和终点
  geom_segment(aes(x=0.19,y=0.47,xend=0.5,yend=0.73),size=0.5) ## 设置两根线的起点和终点
ggsave("E:/test/map.jpg",dpi=300)#保存图片