# 京津冀地区高程数据简单可视化版本
# BTH-DEM-simple.R

# 加载必要包
library(terra)
library(ggplot2)

# 文件路径 - 请根据实际情况修改
beijing_file <- "F:\\4.分城市的数据\\北京市.tif"
tianjin_file <- "F:\\4.分城市的数据\\天津市.tif"
hebei_file <- "F:\\3.分省份的数据\\河北省\\dem地形.tif"

# 读取数据
cat("读取高程数据...\n")

dem_list <- list()
files <- list(Beijing = beijing_file, Tianjin = tianjin_file, Hebei = hebei_file)

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
    title = "京津冀地区高程数据",
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
ggsave("京津冀地区_DEM_统一图.png", p, width = 12, height = 8, dpi = 300)
cat("图像已保存为: 京津冀地区_DEM_统一图.png\n")

# 显示统计信息
cat("\n统计信息:\n")
summary_stats <- aggregate(elevation ~ region, all_data,
                          function(x) c(min = min(x), max = max(x),
                                       mean = mean(x), median = median(x)))
print(summary_stats)