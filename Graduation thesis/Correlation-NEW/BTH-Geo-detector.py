from __future__ import annotations

import importlib.util
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
try:
    import rasterio
    from rasterio.mask import mask as rio_mask
except Exception:  # pylint: disable=broad-except
    rasterio = None
    rio_mask = None
try:
    import geopandas as gpd
except Exception:  # pylint: disable=broad-except
    gpd = None
try:
    import xarray as xr
except Exception:  # pylint: disable=broad-except
    xr = None

try:
    from scipy.stats import f as f_dist
except Exception:  # pylint: disable=broad-except
    f_dist = None


BTH_CITIES = [
    "北京",
    "天津",
    "石家庄",
    "唐山",
    "秦皇岛",
    "邯郸",
    "邢台",
    "保定",
    "张家口",
    "承德",
    "沧州",
    "廊坊",
    "衡水",
]

CITY_TO_PROVINCE = {
    "北京": "beijing",
    "天津": "tianjin",
    "石家庄": "hebei",
    "唐山": "hebei",
    "秦皇岛": "hebei",
    "邯郸": "hebei",
    "邢台": "hebei",
    "保定": "hebei",
    "张家口": "hebei",
    "承德": "hebei",
    "沧州": "hebei",
    "廊坊": "hebei",
    "衡水": "hebei",
}

PROVINCE_NAME_ALIASES = {
    "beijing": ["北京", "北京市", "beijing"],
    "tianjin": ["天津", "天津市", "tianjin"],
    "hebei": ["河北", "河北省", "hebei"],
    "shanghai": ["上海", "上海市", "shanghai"],
    "jiangsu": ["江苏", "江苏省", "jiangsu"],
    "zhejiang": ["浙江", "浙江省", "zhejiang"],
    "anhui": ["安徽", "安徽省", "anhui"],
    "guangdong": ["广东", "广东省", "guangdong"],
}

# 三大城市群 英文(UC_NM_MN) -> 中文，用于工业用地等数据源
BTH_EN_TO_ZH = {
    # BTH 京津冀 13 城
    "beijing": "北京",
    "tianjin": "天津",
    "shijiazhuang": "石家庄",
    "tangshan": "唐山",
    "qinhuangdao": "秦皇岛",
    "handan": "邯郸",
    "xingtai": "邢台",
    "baoding": "保定",
    "zhangjiakou": "张家口",
    "chengde": "承德",
    "cangzhou": "沧州",
    "langfang": "廊坊",
    "hengshui": "衡水",
    "xian": "西安",
    "xi'an": "西安",
    # PRD 珠三角 9 城
    "guangzhou": "广州",
    "shenzhen": "深圳",
    "foshan": "佛山",
    "dongguan": "东莞",
    "zhongshan": "中山",
    "huizhou": "惠州",
    "zhuhai": "珠海",
    "jiangmen": "江门",
    "zhaoqing": "肇庆",
    # YRD 长三角 27 城
    "nanjing": "南京",
    "wuxi": "无锡",
    "nantong": "南通",
    "yancheng": "盐城",
    "yangzhou": "扬州",
    "zhenjiang": "镇江",
    "changzhou": "常州",
    "suzhou": "苏州",
    "taizhou": "泰州",
    "hangzhou": "杭州",
    "ningbo": "宁波",
    "jiaxing": "嘉兴",
    "huzhou": "湖州",
    "shaoxing": "绍兴",
    "jinhua": "金华",
    "zhoushan": "舟山",
    "taizhou_zj": "台州",
    "wenzhou": "温州",
    "hefei": "合肥",
    "wuhu": "芜湖",
    "maanshan": "马鞍山",
    "tongling": "铜陵",
    "anqing": "安庆",
    "chuzhou": "滁州",
    "chizhou": "池州",
    "xuancheng": "宣城",
    "shanghai": "上海",
}

# 目录锚点：优先使用脚本所在项目的相对路径，避免机器间迁移失效
SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent

# =========================
# 内嵌路径配置（直接修改这里）
# =========================
# True: 使用已整合面板表；False: 使用 Data Read 接口自动拼接
USE_PANEL_CSV = False

# 方式 A：已整合面板数据（必须包含 city, year, pm25 + 因子列）
PANEL_CSV_PATH = SCRIPT_DIR / "inputs" / "bth_panel.csv"

# 方式 B：接口自动拼接（需提供 PM2.5 城市年均）
PM25_CITY_YEAR_CSV_PATH = SCRIPT_DIR / "inputs" / "bth_pm25_city_year.csv"
DATA_READ_DIR = THESIS_DIR / "Data Read"

# 可选：地图读取模式（当未提供 PM2.5 城市年均 CSV 时自动启用）
# 需要：全国 PM2.5 年均 NC 目录 + 城市边界 GeoJSON 目录（每个城市一个 .geojson）
DATA_ROOT = THESIS_DIR / "1.模型要用的"
PM25_NC_DIR = DATA_ROOT / "2018-2023[PM2.5-china-clusters]" / "BTH"
PM25_CITY_GEOJSON_DIR = DATA_ROOT / "地图数据"

# 公共参数
DISCRETIZE_BINS = 6
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "geo_detector_bth"


def load_module_from_path(module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    # dataclass 等装饰器会通过 sys.modules 回查模块对象，这里需先注册再执行。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_city_name(city_series: pd.Series) -> pd.Series:
    return (
        city_series.astype(str)
        .str.strip()
        .str.replace("市", "", regex=False)
        .str.replace("地区", "", regex=False)
    )


def normalize_city_text(city_name: str) -> str:
    return (
        str(city_name).strip().replace("市", "").replace("地区", "")
    )


def _map_english_city_to_chinese(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """将英文城市名映射为中文（用于工业用地等数据源）。"""

    def map_one(val: Any) -> Any:
        if pd.isna(val) or not isinstance(val, str):
            return val
        key = str(val).strip().lower().replace(" ", "").replace("-", "").replace("'", "")
        return mapping.get(key, val)

    return series.map(map_one)


def _discover_pm25_candidates(start_dir: Path) -> list[Path]:
    patterns = ("*pm25*city*year*.csv", "*PM25*city*year*.csv", "*pm2.5*city*year*.csv")
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(start_dir.rglob(pattern))
    unique_matches = sorted({path.resolve() for path in matches if path.is_file()})
    return unique_matches


def resolve_existing_path(path_value: Path, path_desc: str, fallback_candidates: list[Path] | None = None) -> Path:
    path_resolved = path_value.expanduser().resolve()
    if path_resolved.exists():
        return path_resolved

    for candidate in fallback_candidates or []:
        candidate_resolved = candidate.expanduser().resolve()
        if candidate_resolved.exists():
            print(f"[WARN] {path_desc} 不存在，自动改用: {candidate_resolved}")
            return candidate_resolved

    raise FileNotFoundError(f"未找到 {path_desc}: {path_resolved}")


def _standardize_nc_coords(ds: Any) -> Any:
    rename_map: dict[str, str] = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)
    try:
        ds = xr.decode_cf(ds)
    except Exception:  # pylint: disable=broad-except
        pass
    return ds


def _choose_pm25_var(ds: Any) -> str:
    var_names = list(ds.data_vars)
    if not var_names:
        raise ValueError("NC 文件中未找到数据变量。")
    for name in var_names:
        if re.search(r"pm\s*2\.?5|pm25", name, flags=re.IGNORECASE):
            return name
    for name in var_names:
        dims = set(ds[name].dims)
        if {"latitude", "longitude"}.issubset(dims):
            return name
    return var_names[0]


def _extract_2d_pm25(da: Any) -> Any:
    data = da
    while data.ndim > 2:
        data = data.mean(dim=data.dims[0], skipna=True)
    if data.ndim != 2:
        raise ValueError(f"PM2.5 变量维度异常: {da.dims}")
    if "latitude" not in data.dims or "longitude" not in data.dims:
        raise ValueError(f"PM2.5 变量缺少经纬度维度: {data.dims}")
    return data


def _extract_2d_spatial_field(da: Any) -> Any:
    data = da
    while data.ndim > 2:
        data = data.mean(dim=data.dims[0], skipna=True)
    if data.ndim != 2:
        raise ValueError(f"空间变量维度异常: {da.dims}")
    if "latitude" not in data.dims or "longitude" not in data.dims:
        raise ValueError(f"空间变量缺少经纬度维度: {data.dims}")
    return data


def _choose_spatial_var(ds: Any, preferred_names: list[str] | None = None) -> str:
    preferred_names = preferred_names or []
    var_names = list(ds.data_vars)
    if not var_names:
        raise ValueError("NC 文件中未找到数据变量。")
    for name in preferred_names:
        if name in ds.data_vars:
            return name
    for name in var_names:
        dims = set(ds[name].dims)
        if {"latitude", "longitude"}.issubset(dims):
            return name
    return var_names[0]


def _discover_pm25_nc_files(nc_dir: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for nc_file in sorted(nc_dir.rglob("*.nc")):
        if not nc_file.is_file():
            continue
        year_match = re.search(r"(20\d{2})", nc_file.name)
        if not year_match:
            continue
        files.append((int(year_match.group(1)), nc_file))
    return files


def _load_city_geojson(city_geojson_dir: Path, target_cities: list[str]) -> Any:
    geo_files = sorted(city_geojson_dir.rglob("*.geojson"))
    if not geo_files:
        raise FileNotFoundError(f"未找到城市 GeoJSON 文件: {city_geojson_dir}")

    target_norm = {normalize_city_text(name) for name in target_cities}
    frames: list[Any] = []
    for geo_file in geo_files:
        gdf = gpd.read_file(geo_file)
        if gdf.empty:
            continue
        if "name" in gdf.columns:
            gdf["city"] = gdf["name"]
        else:
            gdf["city"] = geo_file.stem
        gdf["city_norm"] = gdf["city"].map(normalize_city_text)
        gdf = gdf[gdf["city_norm"].isin(target_norm)].copy()
        if not gdf.empty:
            frames.append(gdf[["city", "city_norm", "geometry"]])

    if not frames:
        raise ValueError("GeoJSON 中未匹配到目标城市。请检查文件名或 name 字段。")

    merged = pd.concat(frames, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=frames[0].crs)
    merged = merged.drop_duplicates(subset=["city_norm"], keep="first")
    return merged.to_crs("EPSG:4326")


def _extract_pm25_by_city_from_nc(nc_file: Path, city_gdf_wgs84: Any) -> pd.DataFrame:
    da2d = None
    open_errors: list[str] = []
    # Windows + 中文路径下，netcdf4 可能误报 FileNotFoundError，优先 h5netcdf 提高兼容性。
    for engine in ("h5netcdf", "netcdf4"):
        try:
            with xr.open_dataset(nc_file, engine=engine, decode_times=True) as ds:
                ds = _standardize_nc_coords(ds)
                pm_var = _choose_pm25_var(ds)
                da2d = _extract_2d_pm25(ds[pm_var]).load()
            break
        except Exception as exc:  # pylint: disable=broad-except
            open_errors.append(f"{engine}: {exc!r}")
            continue

    if da2d is None:
        error_preview = "; ".join(open_errors[:2]) if open_errors else "unknown"
        raise OSError(f"无法读取 NC 文件: {nc_file}\n尝试引擎失败: {error_preview}")

    lon = da2d["longitude"].to_numpy()
    lat = da2d["latitude"].to_numpy()
    values = da2d.to_numpy().astype(float, copy=False)

    lat_is_desc = lat[0] > lat[-1]
    lon_min, lat_min, lon_max, lat_max = city_gdf_wgs84.total_bounds
    if lat_is_desc:
        lat_idx = np.where((lat <= lat_max) & (lat >= lat_min))[0]
    else:
        lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(f"NC 与城市范围无重叠: {nc_file}")

    lat_sub = lat[lat_idx]
    lon_sub = lon[lon_idx]
    val_sub = values[np.ix_(lat_idx, lon_idx)]

    points = gpd.GeoDataFrame(
        {"pm25": val_sub.ravel()},
        geometry=gpd.points_from_xy(np.tile(lon_sub, len(lat_sub)), np.repeat(lat_sub, len(lon_sub))),
        crs="EPSG:4326",
    ).dropna(subset=["pm25"])

    joined = gpd.sjoin(
        points,
        city_gdf_wgs84[["city", "city_norm", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        raise ValueError(f"NC 像元未匹配到城市边界: {nc_file}")

    return (
        joined.groupby(["city_norm", "city"], as_index=False)
        .agg(pm25=("pm25", "mean"))
        .dropna(subset=["pm25"])
    )


def _extract_city_mean_from_2d_field(da2d: Any, city_gdf_wgs84: Any, value_col: str) -> pd.DataFrame:
    lon = da2d["longitude"].to_numpy()
    lat = da2d["latitude"].to_numpy()
    values = da2d.to_numpy().astype(float, copy=False)

    # 兼容 0~360 经度体系，转换到 -180~180 以匹配城市边界坐标系。
    lon_float = np.asarray(lon, dtype=float)
    if lon_float.ndim == 1 and lon_float.size > 0:
        lon_min_raw = float(np.nanmin(lon_float))
        lon_max_raw = float(np.nanmax(lon_float))
        if lon_min_raw >= 0.0 and lon_max_raw > 180.0:
            lon_wrapped = ((lon_float + 180.0) % 360.0) - 180.0
            lon_order = np.argsort(lon_wrapped)
            lon = lon_wrapped[lon_order]
            values = values[:, lon_order]

    lat_is_desc = lat[0] > lat[-1]
    lon_min, lat_min, lon_max, lat_max = city_gdf_wgs84.total_bounds
    if lat_is_desc:
        lat_idx = np.where((lat <= lat_max) & (lat >= lat_min))[0]
    else:
        lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError("NC 与目标城市范围无重叠。")

    lat_sub = lat[lat_idx]
    lon_sub = lon[lon_idx]
    val_sub = values[np.ix_(lat_idx, lon_idx)]
    points = gpd.GeoDataFrame(
        {value_col: val_sub.ravel()},
        geometry=gpd.points_from_xy(np.tile(lon_sub, len(lat_sub)), np.repeat(lat_sub, len(lon_sub))),
        crs="EPSG:4326",
    ).dropna(subset=[value_col])

    joined = gpd.sjoin(
        points,
        city_gdf_wgs84[["city", "city_norm", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        raise ValueError("NC 像元未匹配到任何目标城市边界。")

    return (
        joined.groupby(["city_norm", "city"], as_index=False)
        .agg(**{value_col: (value_col, "mean")})
        .dropna(subset=[value_col])
    )


def build_pm25_city_year_from_nc(nc_dir: Path, city_geojson_dir: Path, target_cities: list[str]) -> pd.DataFrame:
    if xr is None or gpd is None:
        raise ImportError("地图读取模式需要安装 xarray 与 geopandas。")
    if not nc_dir.exists():
        raise FileNotFoundError(f"未找到 PM2.5 NC 目录: {nc_dir}")
    if not city_geojson_dir.exists():
        raise FileNotFoundError(f"未找到城市 GeoJSON 目录: {city_geojson_dir}")

    city_gdf_wgs84 = _load_city_geojson(city_geojson_dir, target_cities)
    year_files = _discover_pm25_nc_files(nc_dir)
    if not year_files:
        raise ValueError(f"未在目录中识别到 PM2.5 NC 文件: {nc_dir}")

    rows: list[pd.DataFrame] = []
    for idx, (year_val, nc_file) in enumerate(year_files, start=1):
        if not nc_file.exists():
            print(f"[WARN] 跳过缺失 NC 文件: {nc_file}")
            continue
        try:
            city_pm = _extract_pm25_by_city_from_nc(nc_file, city_gdf_wgs84)
        except FileNotFoundError:
            print(f"[WARN] 跳过不可读取 NC 文件: {nc_file}")
            continue
        city_pm["city"] = normalize_city_name(city_pm["city"])
        city_pm["year"] = int(year_val)
        rows.append(city_pm[["city", "year", "pm25"]].copy())
        if idx <= 5 or idx % 200 == 0 or idx == len(year_files):
            print(
                f"[INFO] PM2.5 NC 已抽取 {idx}/{len(year_files)} | "
                f"year={year_val} | file={nc_file.name} | cities={len(city_pm)}"
            )

    if not rows:
        raise FileNotFoundError(
            "地图读取模式未成功读取任何 NC 文件。"
            f"\n请检查目录是否包含可访问的 .nc 文件: {nc_dir}"
        )

    pm25_city_year = pd.concat(rows, ignore_index=True)
    pm25_city_year = (
        pm25_city_year.groupby(["city", "year"], as_index=False)["pm25"]
        .mean()
        .sort_values(["year", "city"], kind="mergesort")
        .reset_index(drop=True)
    )
    pm25_city_year = pm25_city_year[pm25_city_year["city"].isin(target_cities)].copy()
    if pm25_city_year.empty:
        raise ValueError("地图读取模式未生成任何目标城市的 PM2.5 年均数据。")
    return pm25_city_year


def reshape_city_year_table(df: pd.DataFrame, city_col: str, value_name: str) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    year_alias = next(
        (c for c in data.columns if re.search(r"^(year|年份|年度)$", str(c), flags=re.IGNORECASE)),
        None,
    )
    if year_alias and year_alias != "year":
        data = data.rename(columns={year_alias: "year"})
    year_cols = [c for c in data.columns if re.search(r"20\d{2}", str(c).strip())]
    if year_cols:
        long_df = data.melt(
            id_vars=[city_col],
            value_vars=year_cols,
            var_name="year",
            value_name=value_name,
        )
        year_extracted = long_df["year"].astype(str).str.extract(r"(20\d{2})", expand=False)
        long_df["year"] = pd.to_numeric(year_extracted, errors="coerce").astype("Int64")
    elif "year" in data.columns:
        long_df = data.rename(columns={city_col: "city", "year": "year"})
        long_df = long_df[["city", "year", value_name]].copy()
        long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    else:
        raise ValueError(f"{value_name} 未找到年份列，请检查源数据结构。")

    long_df = long_df.rename(columns={city_col: "city"})
    long_df["city"] = normalize_city_name(long_df["city"])
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    return long_df.dropna(subset=["city", "year"])


def _match_col(columns: list[str], patterns: list[str], exclude: list[str] | None = None) -> str | None:
    exclude = exclude or []
    for pattern in patterns:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        for col in columns:
            col_text = str(col)
            if regex.search(col_text) and not any(re.search(e, col_text, flags=re.IGNORECASE) for e in exclude):
                return col
    return None


def _sanitize_factor_name(name: str) -> str:
    text = re.sub(r"[^\w]+", "_", str(name).strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _normalize_province_token(text: str) -> str:
    normalized = str(text).strip().lower()
    normalized = re.sub(r"[()\[\]{}（）【】\s_\-]+", "", normalized)
    for suffix in ("省", "市", "特别行政区", "自治区", "壮族", "回族", "维吾尔"):
        normalized = normalized.replace(suffix, "")
    return normalized


def _normalize_province_key(province_key: str) -> str:
    province_token = _normalize_province_token(province_key)
    for canonical_key, aliases in PROVINCE_NAME_ALIASES.items():
        alias_tokens = {_normalize_province_token(alias) for alias in [canonical_key, *aliases]}
        if province_token in alias_tokens:
            return canonical_key
    return province_token


def _match_province_columns(columns: list[str], target_province_keys: list[str]) -> dict[str, str]:
    normalized_cols = {col: _normalize_province_token(col) for col in columns}
    province_cols: dict[str, str] = {}

    for province_key in target_province_keys:
        aliases = PROVINCE_NAME_ALIASES.get(province_key, [])
        alias_tokens = {_normalize_province_token(alias) for alias in [province_key, *aliases] if str(alias).strip()}
        if not alias_tokens:
            continue

        matched = next((col for col, token in normalized_cols.items() if token in alias_tokens), None)
        if matched is None:
            matched = next(
                (
                    col for col, token in normalized_cols.items()
                    if any(alias and (alias in token or token in alias) for alias in alias_tokens)
                ),
                None,
            )
        if matched is not None:
            province_cols[province_key] = matched

    return province_cols


def _extract_pollutant_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem
    if "_" in stem:
        return stem.split("_", 1)[1].strip()
    return stem


def _fill_city_year_linear(df: pd.DataFrame, value_col: str, all_years: list[int]) -> pd.DataFrame:
    if df.empty:
        return df
    base = df.copy()
    base["year"] = pd.to_numeric(base["year"], errors="coerce").astype("Int64")
    base[value_col] = pd.to_numeric(base[value_col], errors="coerce")
    cities = sorted(base["city"].dropna().astype(str).unique().tolist())
    if not cities or not all_years:
        return base

    full_idx = pd.MultiIndex.from_product([cities, sorted(all_years)], names=["city", "year"])
    aligned = (
        base.drop_duplicates(subset=["city", "year"], keep="last")
        .set_index(["city", "year"])
        .reindex(full_idx)
        .reset_index()
    )
    aligned[value_col] = (
        aligned.groupby("city", observed=True)[value_col]
        .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
    )
    aligned["year"] = pd.to_numeric(aligned["year"], errors="coerce").astype("Int64")
    return aligned[["city", "year", value_col]]


def load_industrial_land_factor(data_read_dir: Path) -> pd.DataFrame:
    module = load_module_from_path(data_read_dir / "Industryland.py")
    if not hasattr(module, "load_industrial_land_with_city_info"):
        return pd.DataFrame()
    try:
        land_df = module.load_industrial_land_with_city_info()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] 工业用地接口读取失败，已跳过: {exc}")
        return pd.DataFrame()
    if land_df is None or land_df.empty:
        return pd.DataFrame()

    land_df.columns = [str(c).strip() for c in land_df.columns]
    # 优先 UC_NM_MN（城市英文名），排除 CTR_MN_NM（其为 "China" 国家级）
    city_col = _match_col(
        list(land_df.columns),
        [r"^city$", r"城市", r"市名", r"UC_NM_MN"],
        exclude=[r"CTR_MN_NM"],
    )
    if city_col is None or "year" not in land_df.columns:
        return pd.DataFrame()

    out = land_df.rename(columns={city_col: "city"}).copy()
    out["city"] = normalize_city_name(out["city"])
    out["city"] = _map_english_city_to_chinese(out["city"], BTH_EN_TO_ZH)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")

    out["industrial_land_pixels"] = pd.to_numeric(out.get("industrial_pixel_count"), errors="coerce")
    valid_pixels = pd.to_numeric(out.get("valid_pixel_count"), errors="coerce")
    out["industrial_land_ratio"] = out["industrial_land_pixels"] / valid_pixels.replace(0, np.nan)
    out = out[["city", "year", "industrial_land_pixels", "industrial_land_ratio"]]
    return out.dropna(subset=["city", "year"]).drop_duplicates(subset=["city", "year"], keep="first")


def load_landuse_factor(data_read_dir: Path) -> pd.DataFrame:
    module = load_module_from_path(data_read_dir / "LandUse.py")
    if not all(hasattr(module, attr) for attr in ("find_tif_files", "read_one_tif", "to_dataframe")):
        return pd.DataFrame()
    input_dir = Path(getattr(module, "DEFAULT_INPUT_DIR", ""))
    if not input_dir.exists():
        return pd.DataFrame()
    try:
        tif_files = module.find_tif_files(input_dir)
        if not tif_files:
            return pd.DataFrame()
        results = [module.read_one_tif(str(path)) for path in tif_files]
        landuse_df = module.to_dataframe(results)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] 土地利用接口读取失败，已跳过: {exc}")
        return pd.DataFrame()
    if landuse_df.empty:
        return pd.DataFrame()

    landuse_df.columns = [str(c).strip() for c in landuse_df.columns]
    if {"year", "province"} - set(landuse_df.columns):
        return pd.DataFrame()

    tmp = landuse_df.copy()
    tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce").astype("Int64")
    tmp["province"] = tmp["province"].map(_normalize_province_key)
    tmp["valid_ratio"] = pd.to_numeric(tmp.get("valid_pixel_count"), errors="coerce") / pd.to_numeric(
        tmp.get("pixel_count"), errors="coerce"
    ).replace(0, np.nan)
    tmp["landuse_unique_class_count"] = pd.to_numeric(tmp.get("unique_class_count"), errors="coerce")
    year_province = (
        tmp.groupby(["year", "province"], as_index=False)
        .agg(
            landuse_unique_class_count=("landuse_unique_class_count", "mean"),
            landuse_valid_ratio=("valid_ratio", "mean"),
        )
    )

    city_map = pd.DataFrame({"city": BTH_CITIES})
    city_map["province"] = city_map["city"].map(CITY_TO_PROVINCE).map(_normalize_province_key)
    missing_provinces = sorted(set(city_map["province"].dropna()) - set(year_province["province"].dropna()))
    if missing_provinces:
        print(f"[WARN] 土地利用缺少省份数据，相关城市将为空值: {missing_provinces}")
    out = city_map.merge(year_province, on="province", how="left")
    return out[["city", "year", "landuse_unique_class_count", "landuse_valid_ratio"]].dropna(
        subset=["year"], how="any"
    )


def load_fvc_factor(data_read_dir: Path) -> pd.DataFrame:
    module = load_module_from_path(data_read_dir / "FVC.py")
    if gpd is None or rasterio is None or rio_mask is None:
        print("[WARN] 未安装 geopandas/rasterio，跳过 FVC 城市裁切。")
        return pd.DataFrame()
    if not hasattr(module, "find_fvc_tif_files"):
        return pd.DataFrame()

    city_geojson_dir = PM25_CITY_GEOJSON_DIR.expanduser()
    if not city_geojson_dir.exists():
        print(f"[WARN] 未找到城市边界目录，跳过 FVC 城市裁切: {city_geojson_dir}")
        return pd.DataFrame()
    city_gdf_wgs84 = _load_city_geojson(city_geojson_dir, BTH_CITIES)

    try:
        fvc_dir = Path(getattr(module, "FVC_PATH", ""))
        if not fvc_dir.exists():
            return pd.DataFrame()
        year_files = module.find_fvc_tif_files(fvc_dir)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] FVC 接口读取失败，已跳过: {exc}")
        return pd.DataFrame()
    if not year_files:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for year, tif_path in year_files:
        try:
            with rasterio.open(tif_path) as src:
                city_gdf = city_gdf_wgs84.to_crs(src.crs) if str(src.crs) != "EPSG:4326" else city_gdf_wgs84
                for _, city_row in city_gdf.iterrows():
                    geom = city_row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    masked, _ = rio_mask(src, [geom.__geo_interface__], crop=True, filled=False)
                    city_band = masked[0]
                    city_values = city_band.compressed() if hasattr(city_band, "compressed") else city_band.ravel()
                    if city_values.size == 0:
                        fvc_mean = np.nan
                    else:
                        fvc_mean = float(np.nanmean(city_values.astype(float)))
                    rows.append(
                        {
                            "city": normalize_city_text(city_row["city"]),
                            "year": int(year),
                            "fvc_mean": fvc_mean,
                        }
                    )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] FVC 城市裁切失败，已跳过: {tif_path.name} | {exc}")
            continue
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["city"] = normalize_city_name(out["city"])
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["fvc_mean"] = pd.to_numeric(out["fvc_mean"], errors="coerce")
    return out.dropna(subset=["city", "year"]).drop_duplicates(subset=["city", "year"], keep="first")


def _infer_era5_factor_name_from_path(nc_file: Path, alias_map: dict[str, list[str]]) -> str:
    """
    从文件路径向上回溯，优先识别真实 ERA5 变量目录名（如 total_precipitation）。
    若仅命中变量别名（如 tp），返回对应规范名。
    """
    canonical_names = set(alias_map.keys())
    alias_to_canonical: dict[str, str] = {}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            alias_to_canonical[str(alias).strip()] = canonical

    for parent in nc_file.parents:
        node = parent.name.strip()
        if node in canonical_names:
            return node
        if node in alias_to_canonical:
            return alias_to_canonical[node]
    return nc_file.parent.name.strip()


def _has_cluster_token_in_path(nc_file: Path, cluster_token: str) -> bool:
    target = cluster_token.strip().upper()
    return any(parent.name.strip().upper() == target for parent in nc_file.parents)


def load_meteorology_factor(data_read_dir: Path) -> pd.DataFrame:
    module = load_module_from_path(data_read_dir / "Meteorology.py")
    if xr is None or gpd is None:
        print("[WARN] 未安装 xarray 或 geopandas，跳过气象城市裁切。")
        return pd.DataFrame()

    nc_root = Path(getattr(module, "DEFAULT_INPUT_DIR", ""))
    if not nc_root.exists():
        return pd.DataFrame()

    city_geojson_dir = PM25_CITY_GEOJSON_DIR.expanduser()
    if not city_geojson_dir.exists():
        print(f"[WARN] 未找到城市边界目录，跳过气象城市裁切: {city_geojson_dir}")
        return pd.DataFrame()
    city_gdf_wgs84 = _load_city_geojson(city_geojson_dir, BTH_CITIES)

    cluster_token = "BTH"
    all_nc_files = sorted(path for path in nc_root.rglob("*.nc") if path.is_file())
    nc_files = [path for path in all_nc_files if _has_cluster_token_in_path(path, cluster_token)]
    if not nc_files and all_nc_files:
        print(
            f"[WARN] 未在 ERA5 路径中识别到城市群目录 '{cluster_token}'，"
            "将回退读取全部 NC 文件。"
        )
        nc_files = all_nc_files
    if not nc_files:
        return pd.DataFrame()

    alias_map = {
        "2m_dewpoint_temperature": ["d2m"],
        "2m_temperature": ["t2m"],
        "10m_u_component_of_wind": ["u10"],
        "10m_v_component_of_wind": ["v10"],
        "mean_sea_level_pressure": ["msl"],
        "total_cloud_cover": ["tcc"],
        "total_precipitation": ["tp"],
    }
    unit_conversions = getattr(module, "UNIT_CONVERSIONS", {})

    rows: list[pd.DataFrame] = []
    for nc_file in nc_files:
        year_match = re.search(r"(20\d{2})", nc_file.name)
        if not year_match:
            continue
        year_val = int(year_match.group(1))
        folder_name = _infer_era5_factor_name_from_path(nc_file, alias_map)
        preferred = [folder_name, *alias_map.get(folder_name, [])]
        da2d = None
        open_errors: list[str] = []
        for engine in ("h5netcdf", "netcdf4"):
            try:
                with xr.open_dataset(nc_file, engine=engine, decode_times=True) as ds:
                    ds = _standardize_nc_coords(ds)
                    var_name = _choose_spatial_var(ds, preferred_names=preferred)
                    da2d = _extract_2d_spatial_field(ds[var_name]).load()
                    convert_fn = unit_conversions.get(folder_name) or unit_conversions.get(var_name)
                    if callable(convert_fn):
                        da2d = convert_fn(da2d)
                break
            except Exception as exc:  # pylint: disable=broad-except
                open_errors.append(f"{engine}: {exc!r}")
                continue
        if da2d is None:
            print(f"[WARN] 气象 NC 读取失败，已跳过: {nc_file.name} | {'; '.join(open_errors[:2])}")
            continue

        factor_col = f"met_{_sanitize_factor_name(folder_name)}"
        try:
            city_factor = _extract_city_mean_from_2d_field(da2d, city_gdf_wgs84, factor_col)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] 气象城市裁切失败，已跳过: {nc_file.name} | {exc}")
            continue
        city_factor["city"] = normalize_city_name(city_factor["city"])
        city_factor["year"] = int(year_val)
        rows.append(city_factor[["city", "year", factor_col]].copy())

    if not rows:
        return pd.DataFrame()
    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df.drop_duplicates(subset=["city", "year", *[c for c in long_df.columns if c not in {"city", "year"}]], keep="first")
    wide_df = (
        long_df.groupby(["city", "year"], as_index=False)
        .agg({c: "mean" for c in long_df.columns if c not in {"city", "year"}})
    )
    return wide_df


def load_industrial_emission_factor(data_read_dir: Path) -> tuple[pd.DataFrame, bool]:
    module = load_module_from_path(data_read_dir / "Industrial Emission.py")
    if not all(hasattr(module, attr) for attr in ("collect_csv_paths", "_read_single_csv")):
        return pd.DataFrame(), False
    try:
        root = Path(getattr(module, "DATA_ROOT", ""))
        if not root.exists():
            return pd.DataFrame(), False
        csv_paths = module.collect_csv_paths(root)
        if not csv_paths:
            return pd.DataFrame(), False
        dfs: list[pd.DataFrame] = []
        for csv_path in csv_paths:
            _, one_df, err = module._read_single_csv(csv_path)
            if one_df is None:
                if err:
                    print(f"[WARN] 工业排放文件读取失败: {csv_path.name} | {err}")
                continue
            one_df = one_df.copy()
            one_df["source_file"] = csv_path.name
            one_df["year"] = int(csv_path.parent.name)
            dfs.append(one_df)
        raw_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] 工业排放接口读取失败，已跳过: {exc}")
        return pd.DataFrame(), False
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), False

    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    raw_df["pollutant"] = raw_df["source_file"].map(_extract_pollutant_from_filename)
    raw_df = raw_df[~raw_df["pollutant"].str.contains("definition", case=False, na=False)].copy()
    if raw_df.empty:
        return pd.DataFrame(), False
    if "year" not in raw_df.columns:
        return pd.DataFrame(), False
    raw_df["year"] = pd.to_numeric(raw_df["year"], errors="coerce").astype("Int64")
    month_col = _match_col(list(raw_df.columns), [r"^month$", r"月份", r"月"])
    if month_col is not None:
        month_text = raw_df[month_col].astype(str).str.strip().str.lower()
        month_num = pd.to_numeric(raw_df[month_col], errors="coerce")
        annual_mask = month_text.str.contains(r"annual|全年|年度|年总") | month_num.isin([0])
        if annual_mask.any():
            raw_df = raw_df[annual_mask].copy()

    city_col = _match_col(
        list(raw_df.columns),
        [r"^city$", r"城市", r"市名", r"地级市", r"地区", r"region"],
        exclude=[r"编码", r"代码", r"code"],
    )

    if city_col is not None:
        out = raw_df.rename(columns={city_col: "city"}).copy()
        out["city"] = normalize_city_name(out["city"])
        out["city"] = _map_english_city_to_chinese(out["city"], BTH_EN_TO_ZH)
        value_cols = [
            c for c in raw_df.columns
            if c not in {"year", "source_file", "pollutant", month_col, city_col}
            and pd.api.types.is_numeric_dtype(raw_df[c])
        ]
        if not value_cols:
            return pd.DataFrame(), True
        value_col = value_cols[0]
        city_pollutant = (
            out.groupby(["city", "year", "pollutant"], as_index=False)[value_col]
            .sum(min_count=1)
            .rename(columns={value_col: "emission"})
        )
        city_pollutant["factor"] = city_pollutant["pollutant"].map(lambda x: f"ind_emis_{_sanitize_factor_name(x)}")
        city_wide = (
            city_pollutant.pivot_table(index=["city", "year"], columns="factor", values="emission", aggfunc="mean")
            .reset_index()
        )
        city_wide.columns.name = None
        return city_wide, True

    target_province_keys = sorted(
        {
            _normalize_province_key(province_key)
            for province_key in CITY_TO_PROVINCE.values()
            if str(province_key).strip()
        }
    )
    province_cols = _match_province_columns(list(raw_df.columns), target_province_keys)
    if not province_cols:
        return pd.DataFrame(), False
    missing_provinces = sorted(set(target_province_keys) - set(province_cols.keys()))
    if missing_provinces:
        print(f"[WARN] 工业排放缺少省份列，相关城市将为空值: {missing_provinces}")

    city_rows: list[pd.DataFrame] = []
    city_map_df = pd.DataFrame({"city": BTH_CITIES})
    city_map_df["province_key"] = city_map_df["city"].map(CITY_TO_PROVINCE).map(_normalize_province_key)
    city_map_df = city_map_df[city_map_df["province_key"].isin(province_cols.keys())].copy()

    for _, row in city_map_df.iterrows():
        province_key = row["province_key"]
        province_col = province_cols[province_key]
        city_name = row["city"]
        tmp = raw_df[["year", "pollutant", province_col]].copy()
        tmp = tmp.rename(columns={province_col: "emission"})
        tmp["city"] = city_name
        city_rows.append(tmp)
    if not city_rows:
        return pd.DataFrame(), False

    city_emission = pd.concat(city_rows, ignore_index=True)
    city_emission["emission"] = pd.to_numeric(city_emission["emission"], errors="coerce")
    city_pollutant = (
        city_emission.groupby(["city", "year", "pollutant"], as_index=False)["emission"]
        .sum(min_count=1)
    )
    city_pollutant["factor"] = city_pollutant["pollutant"].map(lambda x: f"ind_emis_{_sanitize_factor_name(x)}")
    city_wide = (
        city_pollutant.pivot_table(index=["city", "year"], columns="factor", values="emission", aggfunc="mean")
        .reset_index()
    )
    city_wide.columns.name = None
    return city_wide, True


def build_panel_from_interfaces(data_read_dir: Path, pm25_city_year_csv: Path) -> pd.DataFrame:
    if not pm25_city_year_csv.exists():
        raise FileNotFoundError(f"未找到 PM2.5 城市年均表: {pm25_city_year_csv}")

    pm25_df = pd.read_csv(pm25_city_year_csv, encoding="utf-8-sig")
    pm25_df.columns = [str(c).strip() for c in pm25_df.columns]
    if {"city", "year", "pm25"} - set(pm25_df.columns):
        raise ValueError("PM2.5 表必须包含列: city, year, pm25")
    pm25_df["city"] = normalize_city_name(pm25_df["city"])
    pm25_df["year"] = pd.to_numeric(pm25_df["year"], errors="coerce").astype("Int64")
    pm25_df["pm25"] = pd.to_numeric(pm25_df["pm25"], errors="coerce")
    panel = pm25_df.copy()

    gdp_pop_module = load_module_from_path(data_read_dir / "GDP-population.py")
    pop_df, gdp_df = gdp_pop_module.load_and_filter_data()
    pop_long = reshape_city_year_table(pop_df, city_col="城市", value_name="population")
    gdp_long = reshape_city_year_table(gdp_df, city_col="城市", value_name="gdp")

    energy_module = load_module_from_path(data_read_dir / "Energy-Consumption.py")
    energy_df = energy_module.load_energy_data()
    energy_long = energy_df.rename(
        columns={
            "城市": "city",
            "年度": "year",
            "插值_全社会用电量万千瓦时": "electricity",
            "插值_人工煤气和天然气供气总量万立方米市辖区": "gas_supply",
            "插值_液化石油气供气总量吨市辖区": "lpg_supply",
        }
    )
    energy_long["city"] = normalize_city_name(energy_long["city"])
    energy_long["year"] = pd.to_numeric(energy_long["year"], errors="coerce").astype("Int64")
    for col in ("electricity", "gas_supply", "lpg_supply"):
        energy_long[col] = pd.to_numeric(energy_long[col], errors="coerce")
    energy_long = energy_long[["city", "year", "electricity", "gas_supply", "lpg_supply"]]

    newpower_module = load_module_from_path(data_read_dir / "Newpower.py")
    newpower_df = newpower_module.load_and_filter_newpower()
    newpower_col = next(
        (c for c in newpower_df.columns if "保有量" in str(c)),
        None,
    )
    if newpower_col is None:
        raise ValueError("新能源汽车数据中未识别到保有量列。")
    newpower_long = newpower_df.rename(columns={"城市": "city", "年份": "year", newpower_col: "new_energy_vehicles"})
    newpower_long["city"] = normalize_city_name(newpower_long["city"])
    newpower_long["year"] = pd.to_numeric(newpower_long["year"], errors="coerce").astype("Int64")
    newpower_long["new_energy_vehicles"] = pd.to_numeric(newpower_long["new_energy_vehicles"], errors="coerce")
    newpower_long = newpower_long[["city", "year", "new_energy_vehicles"]]
    panel_years = sorted(pd.to_numeric(panel["year"], errors="coerce").dropna().astype(int).unique().tolist())
    before_missing = int(newpower_long["new_energy_vehicles"].isna().sum())
    newpower_long = _fill_city_year_linear(newpower_long, value_col="new_energy_vehicles", all_years=panel_years)
    after_missing = int(newpower_long["new_energy_vehicles"].isna().sum())
    if after_missing < before_missing:
        print(f"[INFO] 新能源汽车缺失已按城市年度线性插值补齐: {before_missing - after_missing} 条")

    road_module = load_module_from_path(data_read_dir / "Road.py")
    road_df = road_module.load_road_density_data()
    road_long = road_df.rename(columns={"市名": "city", "year": "year", "路网密度": "road_density"})
    road_long["city"] = normalize_city_name(road_long["city"])
    road_long["year"] = pd.to_numeric(road_long["year"], errors="coerce").astype("Int64")
    road_long["road_density"] = pd.to_numeric(road_long["road_density"], errors="coerce")
    road_long = road_long[["city", "year", "road_density"]]

    night_module = load_module_from_path(data_read_dir / "NightLight.py")
    night_df = night_module.load_nightlight_data()
    night_df.columns = [str(c).strip() for c in night_df.columns]
    night_city_col = next((c for c in night_df.columns if c in {"城市", "city"}), None)
    if night_city_col is None:
        city_pattern = re.compile(r"(city|城市|地市|地级市|市名|地区|region)", flags=re.IGNORECASE)
        city_candidates = [
            c
            for c in night_df.columns
            if city_pattern.search(c) and not re.search(r"年|year|代码|编码|code", c, flags=re.IGNORECASE)
        ]
        if city_candidates:
            night_city_col = city_candidates[0]
    if night_city_col is None:
        raise ValueError("夜间灯光数据中未识别到城市列。")
    night_value_col = next(
        (c for c in night_df.columns if c not in {night_city_col} and ("灯光" in c or "night" in c.lower())),
        None,
    )
    if night_value_col is None:
        numeric_cols = [c for c in night_df.columns if c != night_city_col and pd.api.types.is_numeric_dtype(night_df[c])]
        if not numeric_cols:
            raise ValueError("夜间灯光数据中未识别到数值列。")
        night_value_col = numeric_cols[0]
    night_source = night_df.rename(columns={night_city_col: "城市", night_value_col: "night_light"})
    night_long = reshape_city_year_table(
        night_source,
        city_col="城市",
        value_name="night_light",
    )

    merge_tables = [pop_long, gdp_long, energy_long, newpower_long, road_long, night_long]
    for table in merge_tables:
        panel = panel.merge(table, on=["city", "year"], how="left")

    # 额外因子：尽量自动接入，缺失或读取失败不影响主流程
    extra_city_year_tables: list[pd.DataFrame] = []
    extra_year_tables: list[pd.DataFrame] = []

    try:
        ind_land = load_industrial_land_factor(data_read_dir)
        if not ind_land.empty:
            extra_city_year_tables.append(ind_land)
            print(f"[INFO] 已接入工业用地因子: {set(ind_land.columns) - {'city', 'year'}}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] 工业用地因子接入失败，已跳过: {exc}")

    try:
        landuse = load_landuse_factor(data_read_dir)
        if not landuse.empty:
            extra_city_year_tables.append(landuse)
            print(f"[INFO] 已接入土地利用因子: {set(landuse.columns) - {'city', 'year'}}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] 土地利用因子接入失败，已跳过: {exc}")

    try:
        emis_df, emis_has_city = load_industrial_emission_factor(data_read_dir)
        if not emis_df.empty:
            if emis_has_city:
                extra_city_year_tables.append(emis_df)
                print(f"[INFO] 已接入工业排放城市因子: {set(emis_df.columns) - {'city', 'year'}}")
            else:
                extra_year_tables.append(emis_df)
                print(f"[INFO] 已接入工业排放年度因子: {set(emis_df.columns) - {'year'}}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] 工业排放因子接入失败，已跳过: {exc}")

    try:
        fvc_df = load_fvc_factor(data_read_dir)
        if not fvc_df.empty:
            if {"city", "year"}.issubset(set(fvc_df.columns)):
                extra_city_year_tables.append(fvc_df)
                print(f"[INFO] 已接入 FVC 城市因子: {set(fvc_df.columns) - {'city', 'year'}}")
            else:
                extra_year_tables.append(fvc_df)
                print(f"[INFO] 已接入 FVC 年度因子: {set(fvc_df.columns) - {'year'}}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] FVC 因子接入失败，已跳过: {exc}")

    try:
        met_df = load_meteorology_factor(data_read_dir)
        if not met_df.empty:
            if {"city", "year"}.issubset(set(met_df.columns)):
                extra_city_year_tables.append(met_df)
                print(f"[INFO] 已接入气象城市因子: {set(met_df.columns) - {'city', 'year'}}")
            else:
                extra_year_tables.append(met_df)
                print(f"[INFO] 已接入气象年度因子: {set(met_df.columns) - {'year'}}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] 气象因子接入失败，已跳过: {exc}")

    for table in extra_city_year_tables:
        panel = panel.merge(table, on=["city", "year"], how="left")
    for table in extra_year_tables:
        panel = panel.merge(table, on=["year"], how="left")

    return panel


def discretize_factor(series: pd.Series, bins: int = 6) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.nunique() <= 1:
        strata = pd.Series(np.nan, index=s.index, dtype="object")
        strata.loc[s.notna()] = "all"
        return strata

    k = int(min(max(2, bins), valid.nunique()))
    try:
        import mapclassify as mc  # type: ignore

        classifier = mc.NaturalBreaks(valid.to_numpy(), k=k)
        edges = np.r_[-np.inf, classifier.bins]
        labels = [f"L{i+1}" for i in range(len(edges) - 1)]
        return pd.cut(s, bins=edges, labels=labels, include_lowest=True).astype("object")
    except Exception:  # pylint: disable=broad-except
        return pd.qcut(s.rank(method="first"), q=k, duplicates="drop").astype("object")


def compute_q_stat(y: pd.Series, strata: pd.Series) -> tuple[float, float]:
    df = pd.DataFrame({"y": pd.to_numeric(y, errors="coerce"), "h": strata}).dropna()
    n = len(df)
    l = df["h"].nunique()
    if n < 3 or l < 2:
        return np.nan, np.nan

    sigma2 = df["y"].var(ddof=1)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return 0.0, np.nan

    grouped = df.groupby("h", observed=True)["y"]
    ssw = sum(len(g) * g.var(ddof=1) if len(g) > 1 else 0.0 for _, g in grouped)
    q = float(np.clip(1.0 - ssw / (n * sigma2), 0.0, 1.0))

    if f_dist is None or n <= l or l <= 1 or q >= 1:
        return q, np.nan
    f_value = ((n - l) * q) / ((l - 1) * (1 - q + 1e-12))
    p_value = float(f_dist.sf(f_value, l - 1, n - l))
    return q, p_value


def classify_interaction(q1: float, q2: float, q12: float, atol: float = 1e-3) -> str:
    q_sum = q1 + q2
    q_max = max(q1, q2)
    q_min = min(q1, q2)
    if np.isclose(q12, q_sum, atol=atol):
        return "独立"
    if q12 > q_sum:
        return "双协同(非线性增强)"
    if q_max < q12 <= q_sum:
        return "协同(双因子增强)"
    if q_min < q12 <= q_max:
        return "单拮抗(单因子减弱)"
    return "拮抗(非线性减弱)"


def run_factor_detector(df: pd.DataFrame, y_col: str, factor_cols: list[str], bins: int) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    results = []
    strata_map: dict[str, pd.Series] = {}
    for factor in factor_cols:
        strata = discretize_factor(df[factor], bins=bins)
        q, p = compute_q_stat(df[y_col], strata)
        strata_map[factor] = strata
        results.append({"factor": factor, "q": q, "p_value": p})
    return pd.DataFrame(results).sort_values("q", ascending=False), strata_map


def run_interaction_detector(y: pd.Series, strata_map: dict[str, pd.Series], factor_q: pd.DataFrame) -> pd.DataFrame:
    q_lookup = factor_q.set_index("factor")["q"].to_dict()
    rows = []
    for f1, f2 in combinations(strata_map.keys(), 2):
        combined = strata_map[f1].astype(str) + "__" + strata_map[f2].astype(str)
        q12, p12 = compute_q_stat(y, combined)
        q1, q2 = q_lookup.get(f1, np.nan), q_lookup.get(f2, np.nan)
        rows.append(
            {
                "factor_1": f1,
                "factor_2": f2,
                "q1": q1,
                "q2": q2,
                "q_interaction": q12,
                "p_interaction": p12,
                "interaction_type": classify_interaction(q1, q2, q12),
            }
        )
    return pd.DataFrame(rows).sort_values("q_interaction", ascending=False)


def build_soft_blue_red_cmap() -> LinearSegmentedColormap:
    # 低饱和蓝-白-红，兼顾印刷友好与数值层次感（低值冷色，高值暖色）
    return LinearSegmentedColormap.from_list(
        "soft_blue_red",
        [
            "#5C88C5",
            "#A9C3E8",
            "#F7F7F7",
            "#E8B0B0",
            "#CF6F6F",
        ],
        N=256,
    )


def _paper_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def plot_factor_heatmap(q_by_year: pd.DataFrame, output_png: Path) -> None:
    if q_by_year.empty:
        return
    pivot = q_by_year.pivot(index="factor", columns="year", values="q").sort_index()
    cmap = build_soft_blue_red_cmap()
    _paper_plot_style()
    cell_size = 0.55
    fig_w = max(8, pivot.shape[1] * cell_size + 3)
    fig_h = max(4, pivot.shape[0] * cell_size + 2)
    plt.figure(figsize=(fig_w, fig_h))
    cbar_ticks = np.linspace(0, 1, 6)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0,
        vmax=1,
        center=0.5,
        linewidths=0.5,
        linecolor="#F2F2F2",
        square=True,
        annot_kws={"size": 8.5, "color": "#2F2F2F"},
        cbar_kws={"shrink": 0.9, "pad": 0.02, "ticks": cbar_ticks, "label": "q value"},
    )
    plt.title("BTH Geo-detector q Heatmap", pad=10)
    plt.xlabel("Year")
    plt.ylabel("Factor")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, facecolor="white")
    plt.close()


def plot_interaction_heatmap(inter_df: pd.DataFrame, output_png: Path) -> None:
    if inter_df.empty:
        return
    factors = sorted(set(inter_df["factor_1"]).union(inter_df["factor_2"]))
    matrix = pd.DataFrame(np.nan, index=factors, columns=factors)
    for _, row in inter_df.iterrows():
        matrix.loc[row["factor_1"], row["factor_2"]] = row["q_interaction"]
        matrix.loc[row["factor_2"], row["factor_1"]] = row["q_interaction"]
    np.fill_diagonal(matrix.values, 1.0)
    mask_upper = np.triu(np.ones_like(matrix.values, dtype=bool), k=1)
    cmap = build_soft_blue_red_cmap()
    _paper_plot_style()
    cell_size = 0.62
    fig_side = max(8, len(factors) * cell_size + 2)
    plt.figure(figsize=(fig_side, fig_side))
    annot_size = float(np.clip(9.5 - len(factors) * 0.2, 6.8, 8.8))
    cbar_ticks = np.linspace(0, 1, 6)
    sns.heatmap(
        matrix,
        mask=mask_upper,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0,
        vmax=1,
        center=0.5,
        linewidths=0.5,
        linecolor="#F2F2F2",
        square=True,
        annot_kws={"size": annot_size, "color": "#2F2F2F"},
        cbar_kws={"shrink": 0.9, "pad": 0.02, "ticks": cbar_ticks, "label": "q value"},
    )
    plt.title("BTH Interaction q Heatmap", pad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, facecolor="white")
    plt.close()


def main() -> int:
    sns.set_theme(style="white", context="paper")

    output_dir = OUTPUT_DIR.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if USE_PANEL_CSV:
        panel_csv = resolve_existing_path(
            PANEL_CSV_PATH,
            path_desc="面板文件",
            fallback_candidates=[
                SCRIPT_DIR / "bth_panel.csv",
                OUTPUT_DIR / "bth_panel_from_interfaces.csv",
            ],
        )
        panel = pd.read_csv(panel_csv, encoding="utf-8-sig")
    else:
        pm25_discovered = _discover_pm25_candidates(THESIS_DIR)
        pm25_csv_config = PM25_CITY_YEAR_CSV_PATH.expanduser().resolve()
        if not pm25_csv_config.exists():
            nc_dir = PM25_NC_DIR.expanduser()
            city_geojson_dir = PM25_CITY_GEOJSON_DIR.expanduser()
            geojson_dir_configured = str(PM25_CITY_GEOJSON_DIR).strip() not in {"", ".", ".\\"}
            if geojson_dir_configured and nc_dir.exists() and city_geojson_dir.exists():
                print("[INFO] 未检测到 PM2.5 城市年均 CSV，启用地图读取模式（NC + GeoJSON）自动构建。")
                pm25_city_year_df = build_pm25_city_year_from_nc(
                    nc_dir=nc_dir,
                    city_geojson_dir=city_geojson_dir,
                    target_cities=BTH_CITIES,
                )
                pm25_csv_config.parent.mkdir(parents=True, exist_ok=True)
                pm25_city_year_df.to_csv(pm25_csv_config, index=False, encoding="utf-8-sig")
                print(f"[INFO] 已生成 PM2.5 城市年均 CSV: {pm25_csv_config}")

        try:
            pm25_csv = resolve_existing_path(
                PM25_CITY_YEAR_CSV_PATH,
                path_desc="PM2.5 城市年均表",
                fallback_candidates=[
                    SCRIPT_DIR / "inputs" / "pm25_city_year.csv",
                    *pm25_discovered,
                ],
            )
        except FileNotFoundError as exc:
            discovered_preview = "\n".join(f"  - {item}" for item in pm25_discovered[:10]) or "  (未搜索到候选文件)"
            raise FileNotFoundError(
                f"{exc}\n"
                "请将 PM2.5 城市年均 CSV 放在以下任一路径：\n"
                f"  - {PM25_CITY_YEAR_CSV_PATH.expanduser().resolve()}\n"
                f"  - {SCRIPT_DIR / 'inputs' / 'pm25_city_year.csv'}（旧命名，兼容）\n"
                "或将文件命名为 *pm25*city*year*.csv 并放在 `Graduation thesis` 目录下。\n"
                "当前自动搜索到的候选文件：\n"
                f"{discovered_preview}"
            ) from exc
        data_read_dir = resolve_existing_path(
            DATA_READ_DIR,
            path_desc="Data Read 目录",
            fallback_candidates=[THESIS_DIR / "Data Read"],
        )
        panel = build_panel_from_interfaces(
            data_read_dir=data_read_dir,
            pm25_city_year_csv=pm25_csv,
        )
        panel.to_csv(output_dir / "bth_panel_from_interfaces.csv", index=False, encoding="utf-8-sig")

    panel.columns = [str(c).strip() for c in panel.columns]
    required_cols = {"city", "year", "pm25"}
    if required_cols - set(panel.columns):
        raise ValueError("面板数据至少包含列: city, year, pm25")

    panel["city"] = normalize_city_name(panel["city"])
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel["pm25"] = pd.to_numeric(panel["pm25"], errors="coerce")
    panel = panel.loc[panel["city"].isin(BTH_CITIES)].copy()

    factor_cols = [c for c in panel.columns if c not in {"city", "year", "pm25"}]
    if not factor_cols:
        raise ValueError("未检测到因子列，请检查输入数据。")

    numeric_factors = []
    for col in factor_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
        if panel[col].notna().sum() > 0:
            numeric_factors.append(col)
    factor_cols = numeric_factors
    if not factor_cols:
        raise ValueError("所有因子列均无法转为数值。")

    overall = panel.dropna(subset=["pm25"]).copy()
    factor_q, strata_map = run_factor_detector(
        overall,
        y_col="pm25",
        factor_cols=factor_cols,
        bins=DISCRETIZE_BINS,
    )
    inter_q = run_interaction_detector(overall["pm25"], strata_map, factor_q)

    by_year_rows = []
    for year, sub_df in overall.groupby("year", observed=True):
        if sub_df["city"].nunique() < 3:
            continue
        year_q, _ = run_factor_detector(
            sub_df,
            y_col="pm25",
            factor_cols=factor_cols,
            bins=DISCRETIZE_BINS,
        )
        year_q["year"] = int(year)
        by_year_rows.append(year_q)
    q_by_year = pd.concat(by_year_rows, ignore_index=True) if by_year_rows else pd.DataFrame()

    factor_q.to_csv(output_dir / "bth_factor_detector.csv", index=False, encoding="utf-8-sig")
    inter_q.to_csv(output_dir / "bth_interaction_detector.csv", index=False, encoding="utf-8-sig")
    q_by_year.to_csv(output_dir / "bth_factor_q_by_year.csv", index=False, encoding="utf-8-sig")

    plot_factor_heatmap(q_by_year, output_dir / "bth_factor_q_heatmap.png")
    plot_interaction_heatmap(inter_q, output_dir / "bth_interaction_q_heatmap.png")

    print("=" * 80)
    print("[INFO] 京津冀 Geo-detector 分析完成")
    print(f"[INFO] 样本量: {len(overall)}")
    print(f"[INFO] 因子数量: {len(factor_cols)}")
    print(f"[INFO] 输出目录: {output_dir}")
    print("[INFO] 主要输出:")
    print("       - bth_factor_detector.csv")
    print("       - bth_interaction_detector.csv")
    print("       - bth_factor_q_by_year.csv")
    print("       - bth_factor_q_heatmap.png")
    print("       - bth_interaction_q_heatmap.png")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
