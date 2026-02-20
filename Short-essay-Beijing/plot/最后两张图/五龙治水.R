# 安装并加载所需的R包
required_packages <- c("ggplot2", "dplyr", "readr", "showtext", "scales", "gridExtra", "grid", "rlang", "cowplot")

# 设置用户库路径以避免权限问题
user_lib <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
}
.libPaths(c(user_lib, .libPaths()))

# 检查并安装缺失的包（优先使用用户库）
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    tryCatch({
      cat(sprintf("安装包 %s 到用户库...\n", pkg))
      install.packages(pkg, lib = user_lib, dependencies = TRUE, repos = "https://cloud.r-project.org", type = "binary")
      library(pkg, character.only = TRUE, lib.loc = user_lib)
    }, error = function(e) {
      cat(sprintf("用户库安装失败，尝试系统库: %s\n", e$message))
      install.packages(pkg, dependencies = TRUE, repos = "https://cloud.r-project.org")
      library(pkg, character.only = TRUE)
    })
  }
}

# 安装并加载所有必需的包
for (pkg in required_packages) {
  install_if_missing(pkg)
}

# 修复ffi_list2问题的解决方案
tryCatch({
  cat("正在检查和修复包兼容性...\n")

  # 方法1：强制重新安装冲突的包
  cat("重新安装关键包...\n")
  install.packages(c("rlang", "ggplot2", "scales", "dplyr"), repos = "https://cloud.r-project.org")

  # 方法2：如果仍有问题，尝试降级rlang到稳定版本
  if (!exists("list2", where = asNamespace("rlang"))) {
    cat("检测到rlang版本问题，尝试修复...\n")
    install.packages("rlang", repos = "https://cloud.r-project.org")
  }

  cat("包修复完成！\n")
}, error = function(e) {
  cat("自动修复失败，请手动运行以下命令:\n")
  cat("install.packages(c('rlang', 'ggplot2', 'scales', 'dplyr'), repos = 'https://cloud.r-project.org')\n")
  cat("然后重启R会话重新运行脚本。\n")
})

showtext_auto()

# 读取数据
tryCatch({
  data <- read_csv("C:/Users/Bodhi_Tree/Desktop/something/Short-essay-Beijing/plot/最后两张图/九龙拉棺.csv")
}, error = function(e) {
  stop("无法读取数据文件，请检查文件路径是否正确: ", e$message)
})

# 数据预处理：提取浓度最大值并排序
data <- data %>%
  mutate(Concentration_Max = as.numeric(gsub(".*-(\\d+)$", "\\1", Concentration_Range))) %>%
  arrange(Concentration_Max)

# 筛选保留的模型（移除GAM-MLR-Optimized、SVR-Linear、CNN-LSTM-Transformer、Transformer）
# 先清理模型名称中的前缀"-"
data <- data %>% mutate(Model = gsub("^-", "", Model))
models_to_keep <- c("BPNN", "CNN-GridSearch", "LightGBM", "RF", "XGBOOST")
data <- data %>% filter(Model %in% models_to_keep)

# 设置图形设备
options(bitmapType = "cairo")

# === 定义自定义Y轴变换（用于R²图表）===
custom_trans <- trans_new(
  name = "custom",
  transform = function(y) {
    ifelse(y <= 0, y / 20, y / 2)  # 负值压缩20倍，正值压缩2倍
  },
  inverse = function(y) {
    ifelse(y <= 0, y * 20, y * 2)  # 反变换
  },
  breaks = function(y_range) {
    neg_breaks <- seq(floor(min(y_range) / 2) * 2, 0, by = 2)
    pos_breaks <- seq(0, min(ceiling(max(y_range) * 5) / 5, 1), by = 0.2)
    unique(c(neg_breaks, pos_breaks))
  }
)

# 创建MAE图表
p_mae <- ggplot(data, aes(x = Concentration_Max, y = MAE, color = Model, group = Model)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_y_continuous(name = "MAE") +
  labs(
    title = "MAE Values Trend",
    x = "Concentration Range",
    color = "Model"
  ) +
  scale_color_brewer(palette = "Set1") +
  scale_x_continuous(breaks = unique(data$Concentration_Max)) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold", family = "serif"),
    axis.title = element_text(size = 16, family = "serif", face = "bold"),
    axis.text = element_text(size = 10, family = "serif"),
    legend.title = element_text(size = 18, family = "serif", face = "bold"),
    legend.text = element_text(size = 16, family = "serif"),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    panel.grid = element_line(linewidth = 0.3, color = "gray90"),
    text = element_text(family = "serif")
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE))

# 创建MAPE图表
p_mape <- ggplot(data, aes(x = Concentration_Max, y = MAPE, color = Model, group = Model)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_y_continuous(name = "MAPE") +
  labs(
    title = "MAPE Values Trend",
    x = "Concentration Range",
    color = "Model"
  ) +
  scale_color_brewer(palette = "Set1") +
  scale_x_continuous(breaks = unique(data$Concentration_Max)) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold", family = "serif"),
    axis.title = element_text(size = 16, family = "serif", face = "bold"),
    axis.text = element_text(size = 10, family = "serif"),
    legend.title = element_text(size = 18, family = "serif", face = "bold"),
    legend.text = element_text(size = 16, family = "serif"),
    legend.position = "none",
    panel.grid = element_line(linewidth = 0.3, color = "gray90"),
    text = element_text(family = "serif")
  )

# 创建R²图表
p_r2 <- ggplot(data, aes(x = Concentration_Max, y = R2, color = Model, group = Model)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_y_continuous(
    trans = custom_trans,
    name = expression(R^2)
  ) +
  labs(
    title = expression(R^2~"Values Trend"),
    x = "Concentration Range",
    color = "Model"
  ) +
  scale_color_brewer(palette = "Set1") +
  scale_x_continuous(breaks = unique(data$Concentration_Max)) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold", family = "serif"),
    axis.title = element_text(size = 16, family = "serif", face = "bold"),
    axis.text = element_text(size = 10, family = "serif"),
    legend.title = element_text(size = 18, family = "serif", face = "bold"),
    legend.text = element_text(size = 16, family = "serif"),
    legend.position = "none",
    panel.grid = element_line(linewidth = 0.3, color = "gray90"),
    text = element_text(family = "serif")
  )

# 创建RMSE图表
p_rmse <- ggplot(data, aes(x = Concentration_Max, y = RMSE, color = Model, group = Model)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_y_continuous(name = "RMSE") +
  labs(
    title = "RMSE Values Trend",
    x = "Concentration Range",
    color = "Model"
  ) +
  scale_color_brewer(palette = "Set1") +
  scale_x_continuous(breaks = unique(data$Concentration_Max)) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold", family = "serif"),
    axis.title = element_text(size = 16, family = "serif", face = "bold"),
    axis.text = element_text(size = 10, family = "serif"),
    legend.title = element_text(size = 18, family = "serif", face = "bold"),
    legend.text = element_text(size = 16, family = "serif"),
    legend.position = "none",
    panel.grid = element_line(linewidth = 0.3, color = "gray90"),
    text = element_text(family = "serif")
  )

# === 保存高清图片 ===

# 第一步：创建四个子图（移除图例）
p_mae_no_legend <- p_mae + theme(legend.position = "none")
p_mape_no_legend <- p_mape + theme(legend.position = "none")
p_r2_no_legend <- p_r2 + theme(legend.position = "none")
p_rmse_no_legend <- p_rmse + theme(legend.position = "none")

# 第二步：排列四个子图为2x2网格
plots_grid <- arrangeGrob(p_mae_no_legend, p_mape_no_legend,
                         p_r2_no_legend, p_rmse_no_legend,
                         ncol = 2, nrow = 2)

# 第三步：提取图例
legend_plot <- get_legend(p_mae)

# 第四步：组合网格和图例
# heights参数控制垂直间距：c(网格高度, 图例高度)
# 增加第二个数值可以增加图例和图片的距离
combined_plot <- grid.arrange(
  plots_grid,
  legend_plot,
  heights = c(15, 0.5),  # 图例紧贴图片底部
  top = textGrob("Model Performance Metrics Trends (5 Models)",
                 gp = gpar(fontsize = 16, fontface = "bold", fontfamily = "serif"))
)

# 保存组合图
ggsave("五龙治水_combined_metrics.svg", plot = combined_plot, width = 16, height = 14)

# 方法2：分别保存每个指标的图表
ggsave("五龙治水_MAE.svg", plot = p_mae, width = 10, height = 7)
ggsave("五龙治水_MAPE.svg", plot = p_mape, width = 10, height = 7)
ggsave("五龙治水_R2.svg", plot = p_r2, width = 10, height = 7)
ggsave("五龙治水_RMSE.svg", plot = p_rmse, width = 10, height = 7)

# 显示组合图
print(combined_plot)

# 打印数据摘要信息
cat("数据摘要:\n")
cat("保留的模型:", paste(models_to_keep, collapse = ", "), "\n")
cat("浓度范围:", paste(unique(data$Concentration_Range), collapse = ", "), "\n")
cat("总记录数:", nrow(data), "\n")