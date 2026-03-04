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
PM25_NC_DIR = Path(r"F:\1.模型要用的\2018-2023[PM2.5-china]\Year")
PM25_CITY_GEOJSON_DIR = Path(r"F:\1.模型要用的\地图数据")

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


def _discover_yearly_nc_files(nc_dir: Path) -> list[tuple[int, Path]]:
    year_file_map: dict[int, Path] = {}
    for nc_file in sorted(nc_dir.glob("*.nc")):
        if not nc_file.is_file():
            continue
        year_match = re.search(r"(20\d{2})", nc_file.name)
        if not year_match:
            continue
        year_val = int(year_match.group(1))
        if year_val not in year_file_map:
            year_file_map[year_val] = nc_file
    return sorted(year_file_map.items(), key=lambda item: item[0])


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


def build_pm25_city_year_from_nc(nc_dir: Path, city_geojson_dir: Path, target_cities: list[str]) -> pd.DataFrame:
    if xr is None or gpd is None:
        raise ImportError("地图读取模式需要安装 xarray 与 geopandas。")
    if not nc_dir.exists():
        raise FileNotFoundError(f"未找到 PM2.5 NC 目录: {nc_dir}")
    if not city_geojson_dir.exists():
        raise FileNotFoundError(f"未找到城市 GeoJSON 目录: {city_geojson_dir}")

    city_gdf_wgs84 = _load_city_geojson(city_geojson_dir, target_cities)
    year_files = _discover_yearly_nc_files(nc_dir)
    if not year_files:
        raise ValueError(f"未在目录中识别到年度 NC 文件: {nc_dir}")

    rows: list[pd.DataFrame] = []
    for year_val, nc_file in year_files:
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
        print(f"[INFO] PM2.5 NC 已抽取 | year={year_val} | file={nc_file.name} | cities={len(city_pm)}")

    if not rows:
        raise FileNotFoundError(
            "地图读取模式未成功读取任何 NC 文件。"
            f"\n请检查目录是否包含可访问的 .nc 文件: {nc_dir}"
        )

    pm25_city_year = pd.concat(rows, ignore_index=True)
    pm25_city_year = pm25_city_year.drop_duplicates(subset=["city", "year"], keep="first")
    pm25_city_year = pm25_city_year[pm25_city_year["city"].isin(target_cities)].copy()
    if pm25_city_year.empty:
        raise ValueError("地图读取模式未生成任何目标城市的 PM2.5 年均数据。")
    return pm25_city_year.sort_values(["year", "city"], kind="mergesort").reset_index(drop=True)


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
    city_col = _match_col(
        list(land_df.columns),
        [r"^city$", r"城市", r"市名", r"CTR_MN_NM", r"UC_NM_MN"],
    )
    if city_col is None or "year" not in land_df.columns:
        return pd.DataFrame()

    out = land_df.rename(columns={city_col: "city"}).copy()
    out["city"] = normalize_city_name(out["city"])
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
    tmp["province"] = tmp["province"].astype(str).str.strip().str.lower()
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
    city_map["province"] = city_map["city"].map(CITY_TO_PROVINCE)
    out = city_map.merge(year_province, on="province", how="left")
    return out[["city", "year", "landuse_unique_class_count", "landuse_valid_ratio"]].dropna(
        subset=["year"], how="any"
    )


def load_fvc_factor(data_read_dir: Path) -> pd.DataFrame:
    module = load_module_from_path(data_read_dir / "FVC.py")
    if not hasattr(module, "load_fvc_data"):
        return pd.DataFrame()
    try:
        fvc_data = module.load_fvc_data()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] FVC 接口读取失败，已跳过: {exc}")
        return pd.DataFrame()

    rows: list[dict[str, float | int]] = []
    for year, (arr, _) in fvc_data.items():
        try:
            fvc_mean = float(np.ma.mean(arr))
        except Exception:  # pylint: disable=broad-except
            fvc_mean = float(np.nanmean(np.asarray(arr, dtype=float)))
        rows.append({"year": int(year), "fvc_mean": fvc_mean})
    return pd.DataFrame(rows)


def load_meteorology_factor(data_read_dir: Path) -> pd.DataFrame:
    module = load_module_from_path(data_read_dir / "Meteorology.py")
    wide_df = pd.DataFrame()

    default_wide_csv = Path(getattr(module, "DEFAULT_OUTPUT_WIDE_CSV", ""))
    if str(default_wide_csv).strip() and default_wide_csv.exists():
        try:
            wide_df = pd.read_csv(default_wide_csv, encoding="utf-8-sig")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] 气象宽表 CSV 读取失败，尝试实时构建: {exc}")

    if wide_df.empty and all(hasattr(module, attr) for attr in ("find_nc_files", "build_yearly_records", "to_dataframe", "build_wide_table")):
        input_dir = Path(getattr(module, "DEFAULT_INPUT_DIR", ""))
        if input_dir.exists():
            try:
                nc_files = module.find_nc_files(input_dir)
                records = []
                for nc_file in nc_files:
                    records.extend(module.build_yearly_records(str(nc_file)))
                long_df = module.to_dataframe(records)
                wide_df = module.build_wide_table(long_df)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] 气象接口读取失败，已跳过: {exc}")
                return pd.DataFrame()

    if wide_df.empty:
        return pd.DataFrame()

    wide_df.columns = [str(c).strip() for c in wide_df.columns]
    if "year" not in wide_df.columns:
        return pd.DataFrame()
    wide_df["year"] = pd.to_numeric(wide_df["year"], errors="coerce").astype("Int64")

    preferred_cols = []
    for col in wide_df.columns:
        if col == "year":
            continue
        col_text = col.lower()
        if any(key in col_text for key in ("temperature", "t2m", "precipitation", "tp", "wind_speed", "msl", "pressure", "tcc")):
            preferred_cols.append(col)
    if not preferred_cols:
        preferred_cols = [c for c in wide_df.columns if c != "year" and pd.api.types.is_numeric_dtype(wide_df[c])]
    preferred_cols = preferred_cols[:8]
    keep_cols = ["year", *preferred_cols]
    out = wide_df[keep_cols].copy()
    rename_map = {col: f"met_{col}" for col in preferred_cols}
    return out.rename(columns=rename_map)


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
    if "year" not in raw_df.columns:
        return pd.DataFrame(), False
    raw_df["year"] = pd.to_numeric(raw_df["year"], errors="coerce").astype("Int64")
    city_col = _match_col(
        list(raw_df.columns),
        [r"^city$", r"城市", r"市名", r"地级市", r"地区", r"region"],
        exclude=[r"编码", r"代码", r"code"],
    )
    emission_cols = [
        col
        for col in raw_df.columns
        if re.search(r"排放|emission|so2|nox|voc|co2|pm2?\.?5|pm10", col, flags=re.IGNORECASE)
    ]
    numeric_cols = [col for col in emission_cols if pd.api.types.is_numeric_dtype(raw_df[col])]
    if not numeric_cols:
        exclude_cols = {"year", "source_file", city_col}
        numeric_cols = [c for c in raw_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(raw_df[c])]
    if not numeric_cols:
        return pd.DataFrame(), city_col is not None
    numeric_cols = numeric_cols[:6]

    if city_col is not None:
        out = raw_df.rename(columns={city_col: "city"}).copy()
        out["city"] = normalize_city_name(out["city"])
        grouped = out.groupby(["city", "year"], as_index=False)[numeric_cols].mean()
        rename_map = {col: f"ind_emis_{col}" for col in numeric_cols}
        grouped = grouped.rename(columns=rename_map)
        return grouped, True

    grouped_year = raw_df.groupby("year", as_index=False)[numeric_cols].mean()
    rename_map = {col: f"ind_emis_{col}" for col in numeric_cols}
    grouped_year = grouped_year.rename(columns=rename_map)
    return grouped_year, False


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
            extra_year_tables.append(fvc_df)
            print(f"[INFO] 已接入 FVC 年度因子: {set(fvc_df.columns) - {'year'}}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] FVC 因子接入失败，已跳过: {exc}")

    try:
        met_df = load_meteorology_factor(data_read_dir)
        if not met_df.empty:
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


def plot_factor_heatmap(q_by_year: pd.DataFrame, output_png: Path) -> None:
    if q_by_year.empty:
        return
    pivot = q_by_year.pivot(index="factor", columns="year", values="q").sort_index()
    plt.figure(figsize=(max(8, pivot.shape[1] * 1.2), max(4, pivot.shape[0] * 0.45)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, linewidths=0.3)
    plt.title("BTH Geo-detector q Heatmap")
    plt.xlabel("Year")
    plt.ylabel("Factor")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
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
    plt.figure(figsize=(max(8, len(factors) * 0.7), max(6, len(factors) * 0.7)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdPu", vmin=0, vmax=1, linewidths=0.3)
    plt.title("BTH Interaction q Heatmap")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()


def main() -> int:
    sns.set_theme(style="white")

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
