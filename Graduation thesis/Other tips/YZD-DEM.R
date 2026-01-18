# 长三角地区高程数据简单可视化版本
# YZD-DEM-simple.R

# 加载必要包
library(terra)
library(ggplot2)

# 文件路径 - 请根据实际情况修改
# 江苏
nanjing_file <- "F:\\4.分城市的数据\\南京市.tif"
wuxi_file <- "F:\\4.分城市的数据\\无锡市.tif"
nantong_file <- "F:\\4.分城市的数据\\南通市.tif"
yancheng_file <- "F:\\4.分城市的数据\\盐城市.tif"
yangzhou_file <- "F:\\4.分城市的数据\\扬州市.tif"
zhenjiang_file <- "F:\\4.分城市的数据\\镇江市.tif"
changzhou_file <- "F:\\4.分城市的数据\\常州市.tif"
suzhou_file <- "F:\\4.分城市的数据\\苏州市.tif"
taizhou_js_file <- "F:\\4.分城市的数据\\泰州市.tif"

# 浙江
hangzhou_file <- "F:\\4.分城市的数据\\杭州市.tif"
ningbo_file <- "F:\\4.分城市的数据\\宁波市.tif"
jiaxing_file <- "F:\\4.分城市的数据\\嘉兴市.tif"
huzhou_file <- "F:\\4.分城市的数据\\湖州市.tif"
shaoxing_file <- "F:\\4.分城市的数据\\绍兴市.tif"
jinhua_file <- "F:\\4.分城市的数据\\金华市.tif"
zhoushan_file <- "F:\\4.分城市的数据\\舟山市.tif"
taizhou_zj_file <- "F:\\4.分城市的数据\\台州市.tif"
wenzhou_file <- "F:\\4.分城市的数据\\温州市.tif"

# 安徽
hefei_file <- "F:\\4.分城市的数据\\合肥市.tif"
wuhu_file <- "F:\\4.分城市的数据\\芜湖市.tif"
maanshan_file <- "F:\\4.分城市的数据\\马鞍山市.tif"
tongling_file <- "F:\\4.分城市的数据\\铜陵市.tif"
anqing_file <- "F:\\4.分城市的数据\\安庆市.tif"
chuzhou_file <- "F:\\4.分城市的数据\\滁州市.tif"
chizhou_file <- "F:\\4.分城市的数据\\池州市.tif"
xuancheng_file <- "F:\\4.分城市的数据\\宣城市.tif"

# 上海
shanghai_file <- "F:\\4.分城市的数据\\上海市.tif"

# 读取数据
cat("读取长三角地区高程数据...\n")

dem_list <- list()

# 江苏城市
jiangsu_files <- list(
  Nanjing = nanjing_file,
  Wuxi = wuxi_file,
  Nantong = nantong_file,
  Yancheng = yancheng_file,
  Yangzhou = yangzhou_file,
  Zhenjiang = zhenjiang_file,
  Changzhou = changzhou_file,
  Suzhou = suzhou_file,
  Taizhou_JS = taizhou_js_file
)

# 浙江城市
zhejiang_files <- list(
  Hangzhou = hangzhou_file,
  Ningbo = ningbo_file,
  Jiaxing = jiaxing_file,
  Huzhou = huzhou_file,
  Shaoxing = shaoxing_file,
  Jinhua = jinhua_file,
  Zhoushan = zhoushan_file,
  Taizhou_ZJ = taizhou_zj_file,
  Wenzhou = wenzhou_file
)

# 安徽城市
anhui_files <- list(
  Hefei = hefei_file,
  Wuhu = wuhu_file,
  Maanshan = maanshan_file,
  Tongling = tongling_file,
  Anqing = anqing_file,
  Chuzhou = chuzhou_file,
  Chizhou = chizhou_file,
  Xuancheng = xuancheng_file
)

# 上海
shanghai_files <- list(Shanghai = shanghai_file)

# 合并所有文件
files <- c(jiangsu_files, zhejiang_files, anhui_files, shanghai_files)

for (name in names(files)) {
  file_path <- files[[name]]
  if (file.exists(file_path)) {
    cat(sprintf("读取%s...\n", name))
    dem_list[[name]] <- rast(file_path)
  } else {
    cat(sprintf("%s文件不存在: %s\n", name, file_path))
  }
}

if (length(dem_list) == 0) {
  stop("没有找到任何数据文件!")
}

# 将所有数据转换为数据框并合并
cat("合并数据...\n")
all_data <- data.frame()

for (name in names(dem_list)) {
  df <- as.data.frame(dem_list[[name]], xy = TRUE)
  colnames(df) <- c("x", "y", "elevation")
  df <- na.omit(df)
  df$region <- name
  all_data <- rbind(all_data, df)
  cat(sprintf("%s: %d 个数据点\n", name, nrow(df)))
}

# 创建统一的可视化
cat("创建可视化...\n")

p <- ggplot(all_data, aes(x = x, y = y, fill = elevation)) +
  geom_tile() +
  scale_fill_gradientn(
    colors = c("blue", "cyan", "green", "yellow", "orange", "red", "darkred"),
    name = "高程 (m)"
  ) +
  coord_equal() +
  theme_minimal() +
  labs(
    title = "长三角地区高程数据",
    x = "经度",
    y = "纬度"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    legend.position = c(0.95, 0.05),  # 右下角位置
    legend.justification = c(1, 0),   # 右下角对齐
    legend.background = element_rect(fill = "white", color = "black", linewidth = 0.5),
    legend.margin = margin(5, 5, 5, 5)
  )

# 显示图像
print(p)

# 保存图像
ggsave("长三角地区_DEM_统一图.png", p, width = 12, height = 8, dpi = 300)
cat("图像已保存为: 长三角地区_DEM_统一图.png\n")

# 显示统计信息
cat("\n统计信息:\n")
summary_stats <- aggregate(elevation ~ region, all_data,
                          function(x) c(min = min(x), max = max(x),
                                       mean = mean(x), median = median(x)))
print(summary_stats)