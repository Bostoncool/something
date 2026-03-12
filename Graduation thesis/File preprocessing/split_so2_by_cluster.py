from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely import contains_xy
from shapely.geometry.base import BaseGeometry
from shapely.wkb import dumps as wkb_dumps
from shapely.wkb import loads as wkb_loads

try:
    from tqdm.auto import tqdm
except Exception:  # pylint: disable=broad-except
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = Path(r"G:\基本用不到\SO2\Day")
DEFAULT_MAP_ROOT = Path(r"F:\1.模型要用的\地图数据")
DEFAULT_OUTPUT_ROOT = Path(r"F:\1.模型要用的\[2018-2023]SO2-clusters")
DEFAULT_SUMMARY_FILE = "run_summary.json"
DEFAULT_WRITE_ENGINE = "h5netcdf"

DATE_PATTERN = re.compile(r"CHAP_SO2_D1K_(\d{8})(?:_V\d+)?\.nc$", re.IGNORECASE)
SO2_VAR_CANDIDATES = ("SO2", "so2")

CLUSTER_DIRS = {
    "BTH": "BTH具体城市",
    "YRD": "YRD具体城市",
    "PRD": "PRD具体城市",
}

_WORKER_CONTEXT: dict[str, Any] = {}


@dataclass(frozen=True)
class ClusterProcessResult:
    cluster: str
    status: str
    output_path: str
    error: str = ""


@dataclass(frozen=True)
class FileProcessResult:
    input_file: str
    date: str
    success_count: int
    skipped_count: int
    failed_count: int
    details: list[ClusterProcessResult]


def parse_date_from_filename(file_path: Path) -> datetime | None:
    match = DATE_PATTERN.search(file_path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d")
    except ValueError:
        return None


def find_nc_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.nc") if path.is_file())


def _sanitize_attr_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_sanitize_attr_value(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_sanitize_attr_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_attr_value(item) for item in value)
    return value


def _sanitize_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {key: _sanitize_attr_value(value) for key, value in attrs.items()}


def _choose_so2_var(dataset: xr.Dataset) -> str:
    for var_name in SO2_VAR_CANDIDATES:
        if var_name in dataset.data_vars:
            return var_name
    if dataset.data_vars:
        return next(iter(dataset.data_vars))
    raise ValueError("NC 中未找到数据变量。")


def _infer_lat_lon_dims(data_array: xr.DataArray) -> tuple[str, str]:
    lat_dim = next((name for name in ("latitude", "lat", "y") if name in data_array.dims), None)
    lon_dim = next((name for name in ("longitude", "lon", "x") if name in data_array.dims), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError(f"变量缺少经纬度维度，dims={data_array.dims}")
    return lat_dim, lon_dim


def _clip_dataarray_by_polygon_mask(data_array: xr.DataArray, geometry: BaseGeometry) -> xr.DataArray | None:
    lat_dim, lon_dim = _infer_lat_lon_dims(data_array)
    lat_vals = np.asarray(data_array[lat_dim].values)
    lon_vals = np.asarray(data_array[lon_dim].values)
    if lat_vals.ndim != 1 or lon_vals.ndim != 1:
        raise ValueError("仅支持一维经纬度坐标。")

    lon_min, lat_min, lon_max, lat_max = geometry.bounds
    lon_range_mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)
    lat_range_mask = (lat_vals >= lat_min) & (lat_vals <= lat_max)
    if not lon_range_mask.any() or not lat_range_mask.any():
        return None

    lon_idx = np.where(lon_range_mask)[0]
    lat_idx = np.where(lat_range_mask)[0]
    subset = data_array.isel({lon_dim: lon_idx, lat_dim: lat_idx})
    lon_sub = np.asarray(subset[lon_dim].values)
    lat_sub = np.asarray(subset[lat_dim].values)

    lon_grid, lat_grid = np.meshgrid(lon_sub, lat_sub)
    mask_np = contains_xy(geometry, lon_grid, lat_grid)
    if not np.any(mask_np):
        return None

    mask_da = xr.DataArray(
        mask_np,
        dims=(lat_dim, lon_dim),
        coords={lat_dim: subset[lat_dim], lon_dim: subset[lon_dim]},
    )
    clipped = subset.where(mask_da)

    row_keep = mask_np.any(axis=1)
    col_keep = mask_np.any(axis=0)
    if not row_keep.any() or not col_keep.any():
        return None
    clipped = clipped.isel({lat_dim: np.where(row_keep)[0], lon_dim: np.where(col_keep)[0]})
    return clipped


def _open_dataset_with_fallback(nc_file: Path) -> xr.Dataset:
    errors: list[str] = []
    for engine in ("h5netcdf", "netcdf4"):
        try:
            return xr.open_dataset(nc_file, engine=engine, decode_times=True)
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{engine}: {exc}")
    raise RuntimeError(f"读取失败: {nc_file.name} | {' | '.join(errors[:2])}")


def _build_output_path(output_root: Path, cluster: str, date_dt: datetime, file_name: str) -> Path:
    return output_root / cluster / date_dt.strftime("%Y") / date_dt.strftime("%m") / date_dt.strftime("%d") / file_name


def _build_cluster_geometries(map_root: Path) -> dict[str, BaseGeometry]:
    cluster_geometries: dict[str, BaseGeometry] = {}
    for cluster, folder_name in CLUSTER_DIRS.items():
        cluster_dir = map_root / folder_name
        if not cluster_dir.exists():
            raise FileNotFoundError(f"{cluster} 边界目录不存在: {cluster_dir}")
        geojson_files = sorted(path for path in cluster_dir.glob("*.geojson") if path.is_file())
        if not geojson_files:
            raise FileNotFoundError(f"{cluster} 未找到 GeoJSON 文件: {cluster_dir}")

        gdf_list: list[gpd.GeoDataFrame] = []
        for geojson_file in geojson_files:
            one_gdf = gpd.read_file(geojson_file)
            if one_gdf.empty:
                continue
            if one_gdf.crs is None:
                one_gdf = one_gdf.set_crs("EPSG:4326")
            else:
                one_gdf = one_gdf.to_crs("EPSG:4326")
            gdf_list.append(one_gdf[["geometry"]].copy())
        if not gdf_list:
            raise ValueError(f"{cluster} 边界文件全部为空: {cluster_dir}")

        merged = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs="EPSG:4326")
        if hasattr(merged.geometry, "union_all"):
            cluster_geometries[cluster] = merged.geometry.union_all()
        else:
            cluster_geometries[cluster] = merged.unary_union
    return cluster_geometries


def _init_worker(cluster_wkb_map: dict[str, bytes], output_root_str: str, overwrite: bool, dry_run: bool) -> None:
    global _WORKER_CONTEXT  # pylint: disable=global-statement
    _WORKER_CONTEXT = {
        "cluster_geometries": {name: wkb_loads(blob) for name, blob in cluster_wkb_map.items()},
        "output_root": Path(output_root_str),
        "overwrite": bool(overwrite),
        "dry_run": bool(dry_run),
    }


def _process_single_file(file_path_str: str) -> FileProcessResult:
    file_path = Path(file_path_str)
    date_dt = parse_date_from_filename(file_path)
    if date_dt is None:
        detail = ClusterProcessResult(cluster="ALL", status="failed", output_path="", error="文件名无法解析日期")
        return FileProcessResult(
            input_file=str(file_path),
            date="",
            success_count=0,
            skipped_count=0,
            failed_count=1,
            details=[detail],
        )

    output_root: Path = _WORKER_CONTEXT["output_root"]
    overwrite: bool = _WORKER_CONTEXT["overwrite"]
    dry_run: bool = _WORKER_CONTEXT["dry_run"]
    cluster_geometries: dict[str, BaseGeometry] = _WORKER_CONTEXT["cluster_geometries"]

    if dry_run:
        planned = [
            ClusterProcessResult(
                cluster=cluster,
                status="dry_run",
                output_path=str(_build_output_path(output_root, cluster, date_dt, file_path.name)),
            )
            for cluster in cluster_geometries
        ]
        return FileProcessResult(
            input_file=str(file_path),
            date=date_dt.strftime("%Y-%m-%d"),
            success_count=0,
            skipped_count=len(planned),
            failed_count=0,
            details=planned,
        )

    details: list[ClusterProcessResult] = []
    success_count = 0
    skipped_count = 0
    failed_count = 0

    try:
        with _open_dataset_with_fallback(file_path) as ds:
            var_name = _choose_so2_var(ds)
            source_var = ds[var_name]
            for cluster, geometry in cluster_geometries.items():
                output_path = _build_output_path(output_root, cluster, date_dt, file_path.name)
                if output_path.exists() and not overwrite:
                    details.append(
                        ClusterProcessResult(
                            cluster=cluster,
                            status="skip_exists",
                            output_path=str(output_path),
                        )
                    )
                    skipped_count += 1
                    continue

                try:
                    clipped_var = _clip_dataarray_by_polygon_mask(source_var, geometry)
                    if clipped_var is None:
                        details.append(
                            ClusterProcessResult(
                                cluster=cluster,
                                status="skip_empty",
                                output_path=str(output_path),
                            )
                        )
                        skipped_count += 1
                        continue

                    out_ds = clipped_var.to_dataset(name=var_name)
                    out_ds.attrs = _sanitize_attrs(dict(ds.attrs))
                    out_ds[var_name].attrs = _sanitize_attrs(dict(source_var.attrs))
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    out_ds.to_netcdf(output_path, engine=DEFAULT_WRITE_ENGINE)
                    details.append(
                        ClusterProcessResult(
                            cluster=cluster,
                            status="ok",
                            output_path=str(output_path),
                        )
                    )
                    success_count += 1
                except Exception as exc:  # pylint: disable=broad-except
                    details.append(
                        ClusterProcessResult(
                            cluster=cluster,
                            status="failed",
                            output_path=str(output_path),
                            error=f"{type(exc).__name__}: {exc} | write_engine={DEFAULT_WRITE_ENGINE}",
                        )
                    )
                    failed_count += 1
    except Exception as exc:  # pylint: disable=broad-except
        for cluster in cluster_geometries:
            details.append(
                ClusterProcessResult(
                    cluster=cluster,
                    status="failed",
                    output_path=str(_build_output_path(output_root, cluster, date_dt, file_path.name)),
                    error=str(exc),
                )
            )
            failed_count += 1

    return FileProcessResult(
        input_file=str(file_path),
        date=date_dt.strftime("%Y-%m-%d"),
        success_count=success_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        details=details,
    )


def _cluster_size_bytes(output_root: Path, cluster: str) -> int:
    cluster_dir = output_root / cluster
    if not cluster_dir.exists():
        return 0
    total = 0
    for nc_file in cluster_dir.rglob("*.nc"):
        if nc_file.is_file():
            total += nc_file.stat().st_size
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将全国 SO2 日尺度 NC 数据按 BTH/YRD/PRD 三大城市群进行掩膜裁剪并分目录保存（保留原始数据）。"
    )
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR), help=f"输入 NC 根目录（默认: {DEFAULT_INPUT_DIR}）")
    parser.add_argument("--map-root", type=str, default=str(DEFAULT_MAP_ROOT), help=f"城市边界根目录（默认: {DEFAULT_MAP_ROOT}）")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"输出根目录（默认: {DEFAULT_OUTPUT_ROOT}）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 2),
        help="并行进程数（默认: CPU核数-2）",
    )
    parser.add_argument("--chunksize", type=int, default=8, help="并行任务 chunksize（默认: 8）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出文件")
    parser.add_argument("--start-date", type=str, default="", help="起始日期 YYYYMMDD（可选）")
    parser.add_argument("--end-date", type=str, default="", help="结束日期 YYYYMMDD（可选）")
    parser.add_argument("--dry-run", action="store_true", help="仅统计计划输出，不写入文件")
    parser.add_argument("--summary-file", type=str, default=DEFAULT_SUMMARY_FILE, help="汇总 JSON 文件名")
    return parser


def _parse_date_arg(value: str, arg_name: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{arg_name} 格式错误，应为 YYYYMMDD，当前值: {value}") from exc


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    map_root = Path(args.map_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    workers = max(1, int(args.workers))
    chunksize = max(1, int(args.chunksize))

    try:
        start_date = _parse_date_arg(args.start_date, "--start-date")
        end_date = _parse_date_arg(args.end_date, "--end-date")
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1
    if start_date and end_date and start_date > end_date:
        print("[ERROR] --start-date 不得晚于 --end-date")
        return 1

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] 输入目录不存在或不是目录: {input_dir}")
        return 1
    if not map_root.exists() or not map_root.is_dir():
        print(f"[ERROR] 边界目录不存在或不是目录: {map_root}")
        return 1

    nc_files = find_nc_files(input_dir)
    if not nc_files:
        print(f"[ERROR] 在输入目录未找到 NC 文件: {input_dir}")
        return 2

    if start_date or end_date:
        selected: list[Path] = []
        for one_file in nc_files:
            date_dt = parse_date_from_filename(one_file)
            if date_dt is None:
                continue
            if start_date and date_dt < start_date:
                continue
            if end_date and date_dt > end_date:
                continue
            selected.append(one_file)
        nc_files = selected
        if not nc_files:
            print("[ERROR] 日期过滤后无可处理文件。")
            return 2

    print("=" * 88)
    print(f"[INFO] 输入目录: {input_dir}")
    print(f"[INFO] 边界目录: {map_root}")
    print(f"[INFO] 输出目录: {output_root}")
    print(f"[INFO] 文件数量: {len(nc_files)}")
    print(f"[INFO] 并行进程: {workers}")
    print(f"[INFO] chunksize: {chunksize}")
    print(f"[INFO] dry_run: {bool(args.dry_run)}")
    print("=" * 88)

    stage_begin = time.perf_counter()
    try:
        cluster_geometries = _build_cluster_geometries(map_root)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] 构建城市群边界失败: {exc}")
        return 3

    for cluster, geom in cluster_geometries.items():
        lon_min, lat_min, lon_max, lat_max = geom.bounds
        print(f"[INFO] {cluster} bounds: lon[{lon_min:.4f}, {lon_max:.4f}] lat[{lat_min:.4f}, {lat_max:.4f}]")

    cluster_wkb_map = {name: wkb_dumps(geom) for name, geom in cluster_geometries.items()}
    file_args = [str(path) for path in nc_files]
    all_results: list[FileProcessResult] = []

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(cluster_wkb_map, str(output_root), bool(args.overwrite), bool(args.dry_run)),
    ) as executor:
        future_map = {
            executor.submit(_process_single_file, one_file): one_file
            for one_file in file_args
        }
        progress = tqdm(total=len(future_map), desc="切割 SO2 NC", dynamic_ncols=True) if tqdm is not None else None
        for future in as_completed(future_map):
            try:
                result = future.result()
            except Exception as exc:  # pylint: disable=broad-except
                fail_path = future_map[future]
                result = FileProcessResult(
                    input_file=fail_path,
                    date="",
                    success_count=0,
                    skipped_count=0,
                    failed_count=3,
                    details=[
                        ClusterProcessResult(cluster="BTH", status="failed", output_path="", error=str(exc)),
                        ClusterProcessResult(cluster="YRD", status="failed", output_path="", error=str(exc)),
                        ClusterProcessResult(cluster="PRD", status="failed", output_path="", error=str(exc)),
                    ],
                )
            all_results.append(result)
            if progress is not None:
                progress.update(1)
        if progress is not None:
            progress.close()

    elapsed = time.perf_counter() - stage_begin
    success_total = int(sum(item.success_count for item in all_results))
    skipped_total = int(sum(item.skipped_count for item in all_results))
    failed_total = int(sum(item.failed_count for item in all_results))

    by_cluster: dict[str, dict[str, int]] = {}
    for cluster in CLUSTER_DIRS:
        statuses = {"ok": 0, "skip_exists": 0, "skip_empty": 0, "dry_run": 0, "failed": 0}
        for file_result in all_results:
            for detail in file_result.details:
                if detail.cluster != cluster:
                    continue
                statuses[detail.status] = statuses.get(detail.status, 0) + 1
        by_cluster[cluster] = statuses

    cluster_sizes = {cluster: _cluster_size_bytes(output_root, cluster) for cluster in CLUSTER_DIRS}
    summary = {
        "input_dir": str(input_dir),
        "map_root": str(map_root),
        "output_root": str(output_root),
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "workers": workers,
        "chunksize": chunksize,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "total_input_files": len(nc_files),
        "success_total": success_total,
        "skipped_total": skipped_total,
        "failed_total": failed_total,
        "elapsed_seconds": round(elapsed, 3),
        "by_cluster": by_cluster,
        "cluster_size_bytes": cluster_sizes,
        "cluster_size_mb": {key: round(val / (1024 * 1024), 2) for key, val in cluster_sizes.items()},
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_details": [asdict(item) for item in all_results],
    }

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / args.summary_file
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] 已写入汇总: {summary_path}")
    else:
        print("[INFO] dry-run 模式，不写入 NC 与 summary 文件。")

    print("=" * 88)
    print(f"[INFO] 成功: {success_total} | 跳过: {skipped_total} | 失败: {failed_total}")
    print(f"[INFO] 耗时: {elapsed:.2f}s")
    for cluster, size_mb in summary["cluster_size_mb"].items():
        print(f"[INFO] {cluster} 输出体积: {size_mb:.2f} MB")
    print("=" * 88)
    return 0 if failed_total == 0 else 4


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
