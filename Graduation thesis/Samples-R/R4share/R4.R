
library(geoviz)
library(tidyverse)
library(sf)
library(terra)
library(rasterVis)
library(ggspatial)
library(rgdal)
library(raster)
library(cowplot)
library(RColorBrewer)
library(patchwork)
library(ggpubr)
library(reshape2)
library(ggprism)
library(ggalt)
shp <- sf::read_sf("E:/data/qk/qkk/qkk2/point.shp")#读入矢量点
shp2 <- as.data.frame(st_coordinates(shp))
ndvi1 <- raster('E:/data/qk/qkk/EVI18.tif')#读入栅格数据
df1=as.data.frame(ndvi1,xy=T) 
colnames(df1)=c("x","y","EVI")#数据转化为dataframe
ndvi2 <- raster('E:/data/qk/qkk/EVI19.tif')#读入栅格数据
df2=as.data.frame(ndvi2,xy=T) 
colnames(df2)=c("x","y","EVI")#数据转化为dataframe
ndvi3 <- raster('E:/data/qk/qkk/EVI20.tif')#读入栅格数据
df3=as.data.frame(ndvi3,xy=T) 
colnames(df3)=c("x","y","EVI")#数据转化为dataframe
ndvi4 <- raster('E:/data/qk/qkk/EVI21.tif')#读入栅格数据
df4=as.data.frame(ndvi4,xy=T) 
colnames(df4)=c("x","y","EVI")#数据转化为dataframe
colormap <-  colorRampPalette(rev(brewer.pal(9,'RdYlGn')),1)(32)
s18 <- extract(ndvi1, shp)
s18 <- as.data.frame(s18)
s19 <- extract(ndvi2, shp)
s19 <- as.data.frame(s19)
s20 <- extract(ndvi3, shp)
s20 <- as.data.frame(s20)
s21 <- extract(ndvi4, shp)
s21 <- as.data.frame(s21)
s <- cbind(s18,s19,s20,s21)
s <- t(s)
colnames(s)=c("p1","p2","p3","p4","p5","p6","p7","p8","p9")
data=melt(s)
data$x<-rep(c(2018,2019,2020,2021))
colnames(data)=c("id","group","y","x")

p1 <- ggplot(df1  %>% na.omit() ) +
  geom_raster(aes(x,y,fill=EVI))+
  scale_fill_gradientn(colours =colormap,na.value="transparent")+
  geom_point(data=shp2,aes(x=X,y=Y),colour="red")+ #添加点数据
  geom_text(aes(x=X+0.001,y=Y,label=rownames(shp2)),size =4,family="myFont",fontface="plain",data = shp2) + #添加点标签
  ggtitle('2018')+ #添加图标题
  labs(x=NULL,y=NULL)+ #坐标轴标签
  theme_bw()+ #去掉灰色背景
  scale_x_continuous(n.breaks = 2)+
  scale_y_continuous(n.breaks = 2)+ #控制刻度数
  guides(fill = guide_colorbar(ticks.colour = "darkgrey",frame.colour = "grey"))+ #颜色条轮廓颜色
  theme(panel.grid.major=element_blank(), #去掉背景网格
        panel.grid.minor=element_blank(),
        panel.background=element_blank(),
        # legend.position= "bottom",
        legend.background = element_blank(),
        legend.key.width = unit(3, "mm"), #设置图例宽度
        legend.key.height = unit(6, "mm")) # 设置图例高度
p2 <- ggplot(df2  %>% na.omit() ) +
  geom_raster(aes(x,y,fill=EVI))+
  scale_fill_gradientn(colours =colormap,na.value="transparent")+
  ggtitle('2019')+
  labs(x=NULL,y=NULL)+
  theme_bw()+
  scale_x_continuous(n.breaks = 4)+
  scale_y_continuous(n.breaks = 4)+ #控制刻度数
  guides(fill = guide_colorbar(ticks.colour = "darkgrey",frame.colour = "grey"))+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background=element_blank(),
        legend.background = element_blank(),
        legend.key.width = unit(3, "mm"),
        legend.key.height = unit(6, "mm"))
p3 <- ggplot(df3  %>% na.omit() ) +
  geom_raster(aes(x,y,fill=EVI))+
  scale_fill_gradientn(colours =colormap,na.value="transparent")+
  ggtitle('2020')+
  labs(x=NULL,y=NULL)+
  # geom_sf(size = .2, fill = "transparent", color = "black", data = shp)+ #绘制矢量边界
  theme_bw()+
  scale_x_continuous(n.breaks = 4)+
  scale_y_continuous(n.breaks = 4)+ #控制刻度数
  guides(fill = guide_colorbar(ticks.colour = "darkgrey",frame.colour = "grey"))+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background=element_blank(),
        legend.background = element_blank(),
        legend.key.width = unit(3, "mm"),
        legend.key.height = unit(6, "mm"))
p4 <- ggplot(df4  %>% na.omit() ) +
  geom_raster(aes(x,y,fill=EVI))+
  scale_fill_gradientn(colours =colormap,na.value="transparent")+
  ggtitle('2021')+
  labs(x=NULL,y=NULL)+
  # geom_sf(size = .2, fill = "transparent", color = "black", data = shp)+ #绘制矢量边界
  theme_bw()+
  scale_x_continuous(n.breaks = 4)+
  scale_y_continuous(n.breaks = 4)+ #控制刻度数
  guides(fill = guide_colorbar(ticks.colour = "darkgrey",frame.colour = "grey"))+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background=element_blank(),
        legend.background = element_blank(),
        legend.key.width = unit(3, "mm"),
        legend.key.height = unit(6, "mm"))
p5<-ggplot(data, aes(x, y, group=group, color=group, shape=group,linetype=group))+
  labs(x=NULL,y=NULL)+
  geom_point(size=2)+
  geom_line(size=.1)+
  theme_bw()+
  theme(panel.grid=element_blank(),
        legend.title = element_blank(),
        legend.key.width = unit(3, "mm"),
        legend.key.height = unit(3, "mm"))
  # geom_xspline(spline_shape = -0.3,size=.1)

p11 <- p1 + p2 + p3 + p4
p11/p5+plot_layout(heights = c(3, 1))
# ggarrange(p1,p2,p3,p4,nrow=2,ncol=2,common.legend = T,legend = "bottom")
ggsave("E:/data/qk/qkk/qkk2.jpg",dpi=300)#保存图片