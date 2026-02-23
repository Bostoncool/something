library(readr)
library(dplyr)
library(sf)
library(ggplot2)
library(ggspatial)
library(purrr)
library(future.apply)
library(lubridate)
library(stringr)
library(tidyr)

# =========================
# 1) 基础配置
# =========================
geojson_files <- c(
  "E:/DATA Science/大论文Result/大论文图/2.京津冀/北京市.geojson",
  "E:/DATA Science/大论文Result/大论文图/2.京津冀/天津市.geojson",
  "E:/DATA Science/大论文Result/大论文图/2.京津冀/河北省 (市).geojson"
)

pollution_dir <- "E:/DATA Science/大论文Result/BTH/filtered_daily"
output_dir <- "E:/DATA Science/大论文Result/大论文图/2.京津冀/PM2.5_年均_逐年"
years_target <- 2018:2023

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

debug_mode <- FALSE
debug_year <- 2018
debug_dir <- file.path(output_dir, "debug")
if (debug_mode) {
  dir.create(debug_dir, recursive = TRUE, showWarnings = FALSE)
}

missing_geojson <- geojson_files[!file.exists(geojson_files)]
if (length(missing_geojson) > 0) {
  stop("以下行政区划文件不存在，请检查路径：\n", paste(missing_geojson, collapse = "\n"))
}

if (!dir.exists(pollution_dir)) {
  stop("污染数据目录不存在，请检查路径：", pollution_dir)
}

# LAEA投影（京津冀中心）
laea_crs <- st_crs("+proj=laea +lat_0=39 +lon_0=116 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

# =========================
# 2) 行政区划读取
# =========================
read_and_transform <- function(file_path, crs = laea_crs) {
  st_read(file_path, quiet = TRUE) %>% st_transform(crs = crs)
}

region_shapes <- map(geojson_files, read_and_transform)

# 提取城市名称列
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
    # 兜底：选第一个非geometry列
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
  x_norm <- x %>%
    as.character() %>%
    str_replace_all("\\s+", "") %>%
    str_replace_all("(市|地区|盟|自治州|州)$", "")

  case_when(
    str_detect(x_norm, "北京") ~ "北京",
    str_detect(x_norm, "天津") ~ "天津",
    x_norm == "沧" ~ "沧州",
    str_detect(x_norm, "沧州") ~ "沧州",
    TRUE ~ x_norm
  )
}

# 省界/市界轮廓（只保留边界线）
region_boundary <- map(region_shapes, function(shp) {
  st_sf(boundary = "行政区界", geometry = st_geometry(shp))
}) %>% bind_rows()

# 合并所有行政区划面数据，用于bbox和叠加
region_data <- map(region_shapes, function(shp) {
  city_col <- detect_city_column(shp)
  if (is.na(city_col)) {
    st_sf(city = NA_character_, geometry = st_geometry(shp))
  } else {
    st_sf(city = normalize_city_name(shp[[city_col]]), geometry = st_geometry(shp))
  }
}) %>% bind_rows() %>%
  filter(!is.na(city)) %>%
  group_by(city) %>%
  summarise(geometry = st_union(geometry), .groups = "drop")

if (debug_mode) {
  region_debug_tbl <- map2_dfr(region_shapes, geojson_files, function(shp, path) {
    city_col <- detect_city_column(shp)
    raw_vals <- if (!is.na(city_col)) as.character(shp[[city_col]]) else character()
    tibble(
      file = path,
      city_col = ifelse(is.na(city_col), "", city_col),
      sample_raw = paste(head(unique(raw_vals), 10), collapse = " | "),
      sample_norm = paste(head(unique(normalize_city_name(raw_vals)), 10), collapse = " | ")
    )
  })
  write_csv(region_debug_tbl, file.path(debug_dir, "debug_region_fields_2018.csv"))
}

region_bbox <- st_bbox(region_data)

# 经纬度网格（WGS84生成后投影到LAEA）
region_bbox_wgs84 <- st_bbox(region_data %>% st_transform(crs = st_crs(4326)))
graticule <- st_graticule(
  region_data %>% st_transform(crs = st_crs(4326)),
  lat = seq(floor(region_bbox_wgs84$ymin), ceiling(region_bbox_wgs84$ymax), by = 2),
  lon = seq(floor(region_bbox_wgs84$xmin), ceiling(region_bbox_wgs84$xmax), by = 2)
) %>% st_transform(crs = laea_crs)

# =========================
# 3) 批量多进程读取CSV
# =========================
csv_files <- list.files(pollution_dir, pattern = "\\.csv$", full.names = TRUE)
if (length(csv_files) == 0) {
  stop("未在目录中找到CSV文件：", pollution_dir)
}

plan(multisession)

detect_column <- function(df_names, candidates) {
  idx <- match(tolower(candidates), tolower(df_names))
  if (all(is.na(idx))) {
    return(NA_character_)
  }
  df_names[idx[which(!is.na(idx))[1]]]
}

parse_datetime_safe <- function(x) {
  parsed <- suppressWarnings(parse_date_time(
    x,
    orders = c("Ymd HMS", "Ymd HM", "Ymd", "Y-m-d H:M:S", "Y-m-d H:M", "Y-m-d")
  ))
  parsed
}

empty_pollution_tbl <- function() {
  tibble(
    lon = numeric(),
    lat = numeric(),
    pm25 = numeric(),
    year = integer()
  )
}

extract_year_from_filename <- function(file_path) {
  year_str <- str_extract(basename(file_path), "20\\d{2}")
  if (is.na(year_str)) {
    return(NA_integer_)
  }
  as.integer(year_str)
}

read_single_csv <- function(file_path) {
  df_raw <- tryCatch(
    read_csv(file_path, show_col_types = FALSE, guess_max = 10000),
    error = function(e) NULL
  )
  if (is.null(df_raw)) {
    return(empty_pollution_tbl())
  }

  names_raw <- names(df_raw)
  lon_col <- detect_column(names_raw, c("lon", "longitude", "lng", "x", "经度"))
  lat_col <- detect_column(names_raw, c("lat", "latitude", "y", "纬度"))
  pm_col <- detect_column(names_raw, c("pm2.5", "pm25", "pm_2_5", "pm2_5", "pm2_5_ugm3", "pm25_ugm3", "PM2.5", "PM25"))
  date_col <- detect_column(names_raw, c("date", "datetime", "time", "timestamp", "日期"))
  year_col <- detect_column(names_raw, c("year", "年份"))

  # 模式A：经纬度+PM2.5（点数据）
  if (!is.na(lon_col) && !is.na(lat_col) && !is.na(pm_col)) {
    df <- df_raw %>%
      transmute(
        lon = as.numeric(.data[[lon_col]]),
        lat = as.numeric(.data[[lat_col]]),
        pm25 = as.numeric(.data[[pm_col]]),
        date_raw = if (!is.na(date_col)) .data[[date_col]] else NA_character_,
        year_raw = if (!is.na(year_col)) as.integer(.data[[year_col]]) else NA_integer_
      )

    df <- df %>%
      mutate(
        date = ifelse(is.na(date_raw), NA, as.character(date_raw)),
        date = parse_datetime_safe(date),
        year = ifelse(!is.na(year_raw), year_raw, year(date))
      ) %>%
      select(lon, lat, pm25, year)

    if (all(is.na(df$year))) {
      year_from_name <- str_extract(basename(file_path), "20\\d{2}")
      if (!is.na(year_from_name)) {
        df <- df %>% mutate(year = as.integer(year_from_name))
      }
    }

    return(df)
  }

  # 模式B：城市宽表（列为城市，type区分污染物）
  if ("type" %in% names_raw && "date" %in% names_raw) {
    city_cols <- setdiff(
      names_raw,
      c("date", "hour", "type", "__file__", "__missing_cols__")
    )
    if (length(city_cols) == 0) {
      return(tibble(city = character(), pm25 = numeric(), year = integer()))
    }

    df_long <- df_raw %>%
      filter(.data$type == "PM2.5") %>%
      pivot_longer(
        cols = all_of(city_cols),
        names_to = "city",
        values_to = "pm25"
      ) %>%
      mutate(
        pm25 = as.numeric(pm25),
        date = as.character(.data$date),
        year = suppressWarnings(as.integer(substr(date, 1, 4)))
      ) %>%
      select(city, pm25, year)

    if (all(is.na(df_long$year))) {
      year_from_name <- str_extract(basename(file_path), "20\\d{2}")
      if (!is.na(year_from_name)) {
        df_long <- df_long %>% mutate(year = as.integer(year_from_name))
      }
    }

    return(df_long)
  }

  empty_pollution_tbl()
}

# =========================
# 4) 逐年绘图输出（SVG）
# =========================
overall_start <- Sys.time()
for (yr in years_target) {
  year_start <- Sys.time()
  file_years <- vapply(csv_files, extract_year_from_filename, integer(1))
  files_year <- csv_files[file_years == yr]
  if (length(files_year) == 0) {
    files_year <- csv_files
  }
  message("处理年份：", yr, " | 文件数：", length(files_year))

  pollution_daily <- future_lapply(files_year, read_single_csv) %>% bind_rows()

  if (all(c("lon", "lat") %in% names(pollution_daily))) {
    pollution_daily <- pollution_daily %>%
      filter(!is.na(lon), !is.na(lat), !is.na(pm25), !is.na(year)) %>%
      filter(year == yr)
    mode <- "point"
  } else if ("city" %in% names(pollution_daily)) {
    pollution_daily <- pollution_daily %>%
      filter(!is.na(city), !is.na(pm25), !is.na(year)) %>%
      filter(year == yr)
    mode <- "city"
  } else {
    stop("污染数据未解析出 lon/lat 或 city 字段，请检查CSV字段名或数据结构。")
  }

  if (nrow(pollution_daily) == 0) {
    message("年份 ", yr, " 数据为空，跳过绘图。")
    gc()
    next
  }

  if (mode == "point") {
    pollution_yearly <- pollution_daily %>%
      group_by(lon, lat) %>%
      summarise(pm25_mean = mean(pm25, na.rm = TRUE), .groups = "drop")

    data_year <- st_as_sf(
      pollution_yearly,
      coords = c("lon", "lat"),
      crs = 4326,
      remove = FALSE
    ) %>% st_transform(crs = laea_crs)
  } else {
    pollution_yearly <- pollution_daily %>%
      group_by(city) %>%
      summarise(pm25_mean = mean(pm25, na.rm = TRUE), .groups = "drop")

    data_year <- pollution_yearly %>%
      mutate(city = normalize_city_name(city))
  }

  if (debug_mode && yr == debug_year && mode == "city") {
    region_cities <- sort(unique(region_data$city))
    data_cities <- sort(unique(data_year$city))

    unmatched_region <- setdiff(region_cities, data_cities)
    unmatched_data <- setdiff(data_cities, region_cities)
    matched <- intersect(region_cities, data_cities)

    summary_tbl <- tibble(
      year = yr,
      region_city_n = length(region_cities),
      data_city_n = length(data_cities),
      matched_n = length(matched),
      region_unmatched_n = length(unmatched_region),
      data_unmatched_n = length(unmatched_data),
      match_rate = ifelse(length(region_cities) == 0, NA_real_, length(matched) / length(region_cities))
    )

    write_csv(summary_tbl, file.path(debug_dir, "debug_match_summary_2018.csv"))
    write_csv(tibble(city = unmatched_region), file.path(debug_dir, "debug_unmatched_region_2018.csv"))
    write_csv(tibble(city = unmatched_data), file.path(debug_dir, "debug_unmatched_data_2018.csv"))
  }

  p <- ggplot() +
    geom_sf(data = graticule, color = "gray80", linewidth = 0.3, linetype = "dashed") +
    geom_sf_text(
      data = graticule %>% filter(!is.na(degree_label)),
      aes(label = degree_label),
      color = "black",
      size = 3,
      inherit.aes = FALSE
    ) +
    geom_sf(data = region_data, fill = NA, color = "gray70", linewidth = 0.3) +
    geom_sf(data = region_boundary, fill = NA, color = "black", linewidth = 0.6)

  if (mode == "point") {
    p <- p +
      geom_sf(
        data = data_year,
        aes(color = pm25_mean),
        size = 0.6,
        alpha = 0.9
      ) +
      scale_color_viridis_c(name = "PM2.5 年均浓度")
  } else {
    data_join <- region_data %>%
      left_join(data_year, by = "city")
    p <- p +
      geom_sf(
        data = data_join,
        aes(fill = pm25_mean),
        color = "gray60",
        linewidth = 0.3
      ) +
      scale_fill_viridis_c(name = "PM2.5 年均浓度", na.value = "white")
  }

  p <- p +
    annotation_scale(location = "br", width_hint = 0.3, pad_x = unit(0.3, "cm"), pad_y = unit(0.3, "cm")) +
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
      legend.title = element_text(size = 14, face = "bold"),
      legend.text = element_text(size = 12)
    ) +
    labs(
      title = paste0("京津冀PM2.5年均浓度分布（", yr, "）"),
      x = "经度 (°E)",
      y = "纬度 (°N)"
    ) +
    coord_sf(
      crs = laea_crs,
      expand = FALSE,
      lims_method = "geometry_bbox",
      default_crs = NULL
    )

  out_file <- file.path(output_dir, paste0("京津冀_PM2.5_年均_", yr, ".svg"))
  ggsave(
    filename = out_file,
    plot = p,
    device = "svg",
    width = 12,
    height = 12,
    units = "in",
    dpi = 300
  )
  message("输出文件：", out_file)
  message("完成时间：", format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
          " | 耗时：", round(as.numeric(difftime(Sys.time(), year_start, units = "secs")), 1), "s")

  rm(pollution_daily, pollution_yearly, data_year)
  gc()
}
message("全部完成时间：", format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        " | 总耗时：", round(as.numeric(difftime(Sys.time(), overall_start, units = "mins")), 2), " min")
