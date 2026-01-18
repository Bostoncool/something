
pacman::p_load(sf, raster,ggplot2)#所需要的包
shp <- sf::read_sf("E:/data/wh/shp/bj.shp")#读入矢量边界
ndvi <- raster('E:/test/NDVI/2001year_mean.tif')#读入栅格数据
df=as.data.frame(ndvi,xy=T) 
colnames(df)=c("x","y","LandCover")#数据转化为dataframe
ggplot(df  %>% na.omit() ) +
  geom_raster(aes(x,y,fill=LandCover))+
  scale_fill_gradientn(colours =rainbow(10))+
  labs(x=NULL,y=NULL)+
  geom_sf(size = .2, fill = "transparent", color = "black", data = shp)+ #绘制矢量边界
  annotation_scale(location = "bl") + #设置比例尺
  annotation_north_arrow(location="tl",
                         style = north_arrow_nautical(
                           fill = c("grey40","white"),
                           line_col = "grey20"))+  # 添加指北针
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),# 去除地图网格
        panel.background = element_blank(),
        legend.title = element_blank())+
  theme(axis.ticks.length=unit(-0.1, "cm"), 
      axis.text.x = element_text(margin=unit(c(0.5,0.5,0.5,0.5), "cm")), 
      axis.text.y = element_text(margin=unit(c(0.5,0.5,0.5,0.5), "cm")) )+
    theme(axis.text.x = element_text(angle=0,hjust=1),   # 旋转坐标轴label的方向
          text = element_text(size = 12, face = "bold", family="serif"),  
          panel.spacing = unit(0,"lines") ) #分面间隔设为零
ggsave("E:/test/ndvi.jpg",dpi=300)#保存图片