library(tmap)
library(sf)
library(see)
library(dplyr)
library(viridis)
library(ggplot2)
library(geoviz)
#DEM 颜色
library(RColorBrewer)
crs_84 <- st_crs("EPSG:4326")  ## WGS 84 大地坐标
colormap <-  colorRampPalette(rev(brewer.pal(9,'RdYlGn')),1)(32)
# scales::show_col(colormap)
china_all <-
  sf::st_read("https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json") %>%
  st_transform(crs_84)
shp <- sf::read_sf("E:/data/qk/qzgy/sqzgy.shp")
shp2 <- sf::read_sf("E:/data/qk/qzgy/qkpoint.shp")
shp3 <- sf::read_sf("E:/data/qk/qzgy/qkcountry.shp")%>%
  st_transform(crs_84)
lat=32
long=90
square_km=1000
dem<-mapbox_dem(lat,long,square_km,
                api_key="pk.eyJ1IjoiYmVueXNmIiwiYSI6ImNrczBtdWE0ajBwNjcydnBqMjRyZDdsOXkifQ.sUcMdooE7b9uQqzfrnWdSQ")
guilin<- raster::mask(dem,shp) %>% # 将地图与DEM数据结合
  crop(.,extent(shp))
df_guilin<- as.data.frame(as(guilin,"Raster"),xy=T) #格式转换
ggplot() + 
  geom_sf(data = shp,fill="NA") + 
  geom_tile(data=df_guilin,aes(x=x,y=y,fill=layer),show.legend = F)+
  scale_fill_gradientn(colours =colormap,na.value="transparent",name="DEM",breaks = c(1000,2000,3000,4000,5000,6000,7000))+
  labs(x=NULL,y=NULL)+
  geom_sf(data = shp3,fill = "NA",color="grey66") +
  geom_sf(data = shp2,aes(fill = Id),color = "darkred",alpha = 0.6,size =1.5) +
  annotation_scale(location = "bl",text_face = "bold",pad_x = unit(1, "cm"),pad_y = unit(0.5, "cm")) +
  annotation_north_arrow(location="tr",style=north_arrow_fancy_orienteering)+  # 添加指北针
  theme_minimal()+
  theme_bw()+
  annotate('point',x=75,y=29,color='darkred',size=1.5,alpha = 0.6)+
  annotate('text',x=77,y=29,color='black',label='site',size=4)+
  annotate('rect',xmin=74,xmax = 76,ymin = 27.4,ymax = 27.8,fill='NA',color="grey66")+
  annotate('text',x=78.5,y=27.7,color='black',label='study area',size=4)+
  guides(fill = guide_colorbar(ticks.colour = "black",frame.colour = "black"))+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background=element_blank(),
        legend.position= "bottom",
        legend.background = element_blank(),
        legend.key.width = unit(30.8, "mm"),
        legend.key.height = unit(4, "mm")) # 左边边缘距离)
ggsave("E:/data/qk/picture/common.jpg",width = 8, height = 5,dpi=300)#保存图片
