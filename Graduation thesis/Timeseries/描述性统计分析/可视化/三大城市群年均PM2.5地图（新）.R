library(readr)
library(dplyr)
library(sf)
library(ggplot2)
library(ggspatial)
library(purrr)
library(stringr)
library(tidyr)

if (getRversion() >= "2.15.1") {
  utils::globalVariables(c("x_norm", "city", "geometry"))
}

# =========================
# 1) 基础配置
# =========================
data_file <- "H:/DATA Science/大论文Result/三大城市群（市）年度PM2.5浓度.csv"
years_target <- 2018:2023
output_root <- "H:/DATA Science/大论文Result/大论文图/三大城市群/PM2.5_年均_逐年地图"

dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

regions <- list(
  list(
    name = "京津冀",
    laea_crs = st_crs("+proj=laea +lat_0=39 +lon_0=116 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"),
    geojson_files = c(
      "H:/DATA Science/大论文Result/大论文图/2.京津冀/北京市.geojson",
      "H:/DATA Science/大论文Result/大论文图/2.京津冀/天津市.geojson",
      "H:/DATA Science/大论文Result/大论文图/2.京津冀/河北省 (市).geojson"
    )
  ),
  list(
    name = "长三角",
    laea_crs = st_crs("+proj=laea +lat_0=31.2 +lon_0=120.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"),
    geojson_files = c(
      "H:/DATA Science/大论文Result/大论文图/3.长三角/上海市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/南京市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/无锡市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/常州市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/苏州市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/南通市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/扬州市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/镇江市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/泰州市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/盐城市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/合肥市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/芜湖市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/马鞍山市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/铜陵市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/安庆市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/池州市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/滁州市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/宣城市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/杭州市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/宁波市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/温州市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/嘉兴市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/湖州市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/绍兴市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/金华市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/舟山市.geojson",
      "H:/DATA Science/大论文Result/大论文图/3.长三角/具体城市/台州市.geojson"
    )
  ),
  list(
    name = "珠三角",
    laea_crs = st_crs("+proj=laea +lat_0=23.4 +lon_0=113.3 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"),
    geojson_files = c(
      "H:/DATA Science/大论文Result/大论文图/4.珠三角/具体城市/广州市.geojson",
      "H:/DATA Science/大论文Result/大论文图/4.珠三角/具体城市/深圳市.geojson",
      "H:/DATA Science/大论文Result/大论文图/4.珠三角/具体城市/佛山市.geojson",
      "H:/DATA Science/大论文Result/大论文图/4.珠三角/具体城市/东莞市.geojson",
      "H:/DATA Science/大论文Result/大论文图/4.珠三角/具体城市/中山市.geojson",
      "H:/DATA Science/大论文Result/大论文图/4.珠三角/具体城市/珠海市.geojson",
      "H:/DATA Science/大论文Result/大论文图/4.珠三角/具体城市/惠州市.geojson",
      "H:/DATA Science/大论文Result/大论文图/4.珠三角/具体城市/江门市.geojson",
      "H:/DATA Science/大论文Result/大论文图/4.珠三角/具体城市/肇庆市.geojson"
    )
  )
)

# =========================
# 2) 工具函数
# =========================
detect_city_column <- function(sf_obj) {
  df_names <- names(sf_obj)
  candidates <- c(
    "name", "NAME", "Name",
    "city", "CITY",
    "市", "地名", "行政区划",
    "NAME_CHN", "CN_NAME", "NAME_2", "NAME_1"
  )
  idx <- match(tolower(candidates), tolower(df_names))
  if (all(is.na(idx))) {
    geom_col <- attr(sf_obj, "sf_column")
    if (is.null(geom_col)) {
      geom_col <- "geometry"
    }
    non_geom <- setdiff(df_names, geom_col)
    if (length(non_geom) > 0) {
      return(non_geom[1])
    }
    return(NA_character_)
  }
  df_names[idx[which(!is.na(idx))[1]]]
}

normalize_city_name <- function(x) {
  city_norm <- x %>%
    as.character() %>%
    str_replace_all("\\s+", "") %>%
    str_replace_all("(市|地区|盟|自治州)$", "")

  city_norm[city_norm == "沧"] <- "沧州"

  city_dict <- c(
    "北京", "天津", "沧州", "石家庄", "唐山", "秦皇岛", "邯郸", "邢台", "保定",
    "张家口", "承德", "廊坊", "衡水", "上海", "南京", "无锡", "常州", "苏州",
    "南通", "扬州", "镇江", "泰州", "盐城", "合肥", "芜湖", "马鞍山", "铜陵",
    "安庆", "池州", "滁州", "宣城", "杭州", "宁波", "温州", "嘉兴", "湖州",
    "绍兴", "金华", "舟山", "台州", "广州", "深圳", "佛山", "东莞", "中山",
    "珠海", "惠州", "江门", "肇庆"
  )

  for (city_name in city_dict) {
    city_norm[str_detect(city_norm, city_name)] <- city_name
  }

  city_norm
}

build_region_geometry <- function(region_cfg) {
  missing_geojson <- region_cfg$geojson_files[!file.exists(region_cfg$geojson_files)]
  if (length(missing_geojson) > 0) {
    stop(
      "以下行政区划文件不存在，请检查路径：\n",
      paste(missing_geojson, collapse = "\n")
    )
  }

  region_shapes <- map(
    region_cfg$geojson_files,
    ~ st_read(.x, quiet = TRUE) %>% st_transform(crs = region_cfg$laea_crs)
  )

  region_boundary <- map(region_shapes, function(shp) {
    st_sf(boundary = "行政区界", geometry = st_geometry(shp))
  }) %>% bind_rows()

  region_data <- map(region_shapes, function(shp) {
    city_col <- detect_city_column(shp)
    if (is.na(city_col)) {
      st_sf(city = NA_character_, geometry = st_geometry(shp))
    } else {
      st_sf(city = normalize_city_name(shp[[city_col]]), geometry = st_geometry(shp))
    }
  }) %>%
    bind_rows() %>%
    filter(!is.na(.data$city)) %>%
    group_by(.data$city) %>%
    summarise(geometry = st_union(.data$geometry), .groups = "drop")

  region_bbox_wgs84 <- st_bbox(region_data %>% st_transform(crs = st_crs(4326)))
  graticule <- st_graticule(
    region_data %>% st_transform(crs = st_crs(4326)),
    lat = seq(floor(region_bbox_wgs84$ymin), ceiling(region_bbox_wgs84$ymax), by = 2),
    lon = seq(floor(region_bbox_wgs84$xmin), ceiling(region_bbox_wgs84$xmax), by = 2)
  ) %>% st_transform(crs = region_cfg$laea_crs)

  list(
    region_data = region_data,
    region_boundary = region_boundary,
    graticule = graticule
  )
}

# =========================
# 3) 读取年度数据（宽表转长表）
# =========================
if (!file.exists(data_file)) {
  stop("数据文件不存在，请检查路径：", data_file)
}

pm25_yearly <- read_csv(data_file, show_col_types = FALSE) %>%
  rename(city = 1) %>%
  mutate(city = normalize_city_name(city)) %>%
  pivot_longer(
    cols = -city,
    names_to = "year",
    values_to = "pm25_mean"
  ) %>%
  mutate(
    year = suppressWarnings(as.integer(year)),
    pm25_mean = as.numeric(pm25_mean)
  ) %>%
  filter(!is.na(year), year %in% years_target)

# =========================
# 4) 按城市群、按年份输出SVG
# =========================
overall_start <- Sys.time()

for (region_cfg in regions) {
  message("开始处理城市群：", region_cfg$name)

  geo <- build_region_geometry(region_cfg)
  region_output_dir <- file.path(output_root, region_cfg$name)
  dir.create(region_output_dir, recursive = TRUE, showWarnings = FALSE)

  for (yr in years_target) {
    year_start <- Sys.time()

    data_year <- pm25_yearly %>%
      filter(year == yr) %>%
      select(city, pm25_mean)

    data_join <- geo$region_data %>%
      left_join(data_year, by = "city")

    p <- ggplot() +
      geom_sf(
        data = geo$graticule,
        color = "gray80",
        linewidth = 0.3,
        linetype = "dashed"
      ) +
      geom_sf_text(
        data = geo$graticule %>% filter(!is.na(degree_label)),
        aes(label = degree_label),
        color = "black",
        size = 3,
        inherit.aes = FALSE
      ) +
      geom_sf(data = geo$region_data, fill = NA, color = "gray70", linewidth = 0.3) +
      geom_sf(data = geo$region_boundary, fill = NA, color = "black", linewidth = 0.6) +
      geom_sf(
        data = data_join,
        aes(fill = pm25_mean),
        color = "gray60",
        linewidth = 0.3
      ) +
      scale_fill_gradientn(
        colours = c(
          "#F6F5E8",
          "#FFECB3",
          "#d5f9ad",
          "#42B3D5",
          "#FF9966",
          "#eb6c6c"
        ),
        values = c(0.00, 0.20, 0.45, 0.65, 0.85, 1.00),
        name = NULL,
        na.value = "white"
      ) +
      annotation_scale(
        location = "br",
        width_hint = 0.3,
        pad_x = unit(0.3, "cm"),
        pad_y = unit(0.3, "cm")
      ) +
      annotation_north_arrow(
        location = "tr",
        which_north = "true",
        style = north_arrow_fancy_orienteering,
        height = unit(1.5, "cm"),
        width = unit(1.5, "cm"),
        pad_x = unit(0.3, "cm"),
        pad_y = unit(0.3, "cm")
      ) +
      theme_minimal() +
      theme(
        panel.grid = element_line(color = "gray90", linewidth = 0.3),
        panel.grid.minor = element_line(color = "gray95", linewidth = 0.2),
        axis.text = element_text(size = 14, color = "black"),
        axis.ticks = element_line(color = "black", linewidth = 0.5),
        axis.title = element_text(size = 16, face = "bold"),
        plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
        legend.title = element_blank(),
        legend.text = element_text(size = 12),
        legend.position = c(0.94, 0.12),
        legend.justification = c(1, 0),
        legend.background = element_rect(fill = alpha("white", 0.75), color = NA)
      ) +
      labs(
        title = paste0(region_cfg$name, "PM2.5年均浓度分布（", yr, "）"),
        x = "经度 (°E)",
        y = "纬度 (°N)"
      ) +
      coord_sf(
        crs = region_cfg$laea_crs,
        expand = FALSE,
        lims_method = "geometry_bbox",
        default_crs = NULL
      )

    out_file <- file.path(
      region_output_dir,
      paste0(region_cfg$name, "_PM2.5_年均_", yr, ".svg")
    )

    ggsave(
      filename = out_file,
      plot = p,
      device = "svg",
      width = 12,
      height = 12,
      units = "in",
      dpi = 300
    )

    message(
      "完成：", out_file,
      " | 耗时：",
      round(as.numeric(difftime(Sys.time(), year_start, units = "secs")), 1),
      "s"
    )
  }
}

message(
  "全部完成时间：",
  format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  " | 总耗时：",
  round(as.numeric(difftime(Sys.time(), overall_start, units = "mins")), 2),
  " min"
)
