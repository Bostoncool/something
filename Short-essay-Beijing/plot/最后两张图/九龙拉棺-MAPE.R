library(ggplot2)
library(dplyr)
library(readr)
library(showtext)
showtext_auto()

data = read_csv("C:/Users/IU/Desktop/something/九龙拉棺.csv")
data <- data %>%
  mutate(Concentration_Max = as.numeric(gsub(".*-(\\d+)$", "\\1", Concentration_Range))) %>%
  arrange(Concentration_Max)

# === 创建高质量的图表 ===
# 设置图形设备（在RStudio中运行一次即可）
options(bitmapType = "cairo")  # 启用抗锯齿

# 绘制图表
p <- ggplot(data, aes(x = Concentration_Max, y = MAPE, color = Model, group = Model)) +
  geom_line(linewidth = 1.2) +  # 稍粗线条，减少锯齿感
  geom_point(size = 3) +        # 更大点，更清晰
  
  scale_y_continuous(
    name = "MAPE",
    expand = expansion(mult = 0.05)  # 减少边缘空白
  ) +
  
  labs(
    title = "The trend of MAPE values for different models",
    x = "Concentration Range",
    color = "Model"
  ) +
  
  scale_color_brewer(palette = "Set1") +
  scale_x_continuous(breaks = unique(data$Concentration_Max)) +
  
  # 更换字体和美化主题
  theme_bw()  +  # 使用白色背景主题，线条更清晰
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold", family = "serif"),
    plot.subtitle = element_text(hjust = 0.5, size = 12, family = "serif"),
    axis.title = element_text(size = 12, family = "serif"),
    axis.text = element_text(size = 10, family = "serif"),
    legend.title = element_text(size = 11, family = "serif"),
    legend.text = element_text(size = 10, family = "serif"),
    legend.position = "right",
    panel.grid = element_line(size = 0.3, color = "gray90"),  # 更细的网格线
    text = element_text(family = "serif")
  )

# === 保存高清图片 ===
# 方法1：高DPI位图（推荐）
# ggsave("plot_high_res.png", 
#       plot = p, 
#       width = 12, height = 7, 
#       dpi = 600, 
#       antialias = "cleartype",
#       bg = "white")

# 方法2：矢量图（绝对清晰）
ggsave("plot_vector_MAPE.svg", plot = p, width = 10, height = 7)

# 方法3：在RStudio中直接显示
print(p)