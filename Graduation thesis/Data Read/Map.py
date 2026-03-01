from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from tqdm import tqdm


DEFAULT_INPUT_DIR = Path(r"F:\1.模型要用的\地图数据")
DEFAULT_OUTPUT_CSV = Path(r"F:\1.模型要用的\city_map_read_summary.csv")
SUPPORTED_EXTENSIONS = {".geojson", ".json", ".shp", ".gpkg", ".kml"}


@dataclass(frozen=True)
class MapReadResult:
    file_path: str
    cluster: str
    city_name: str
    status: str
    feature_count: int
    geometry_types: str
    crs: str
    error: str = ""


def discover_map_files(root_dir: Path) -> list[Path]:
    """递归扫描地图文件。"""
    return sorted(
        file_path
        for file_path in root_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def infer_cluster_from_path(file_path: Path) -> str:
    """从父目录识别城市群标识。"""
    parent_name = file_path.parent.name
    if "BTH" in parent_name:
        return "BTH"
    if "YRD" in parent_name:
        return "YRD"
    if "PRD" in parent_name:
        return "PRD"
    return parent_name


def extract_city_name(file_path: Path) -> str:
    """从文件名提取城市名（不含扩展名）。"""
    return file_path.stem.strip()


def read_one_map(file_path_str: str) -> MapReadResult:
    """读取单个地图文件并返回统计信息。"""
    file_path = Path(file_path_str)
    city_name = extract_city_name(file_path)
    cluster = infer_cluster_from_path(file_path)

    try:
        gdf = gpd.read_file(file_path)
        geometry_types = ",".join(
            sorted({str(geom_type) for geom_type in gdf.geometry.geom_type.dropna().unique()})
        )
        crs_text = str(gdf.crs) if gdf.crs is not None else ""

        return MapReadResult(
            file_path=str(file_path),
            cluster=cluster,
            city_name=city_name,
            status="ok",
            feature_count=int(len(gdf)),
            geometry_types=geometry_types,
            crs=crs_text,
            error="",
        )
    except Exception as exc:  # pylint: disable=broad-except
        return MapReadResult(
            file_path=str(file_path),
            cluster=cluster,
            city_name=city_name,
            status="failed",
            feature_count=0,
            geometry_types="",
            crs="",
            error=str(exc),
        )


def to_dataframe(results: list[MapReadResult]) -> pd.DataFrame:
    df = pd.DataFrame([result.__dict__ for result in results])
    if df.empty:
        return df
    return df.sort_values(["cluster", "city_name"], kind="mergesort").reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="并行读取三大城市群地图文件并提取城市名")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help=f"地图文件根目录（默认: {DEFAULT_INPUT_DIR}）",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"输出汇总 CSV 路径（默认: {DEFAULT_OUTPUT_CSV}）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(32, (os.cpu_count() or 1) * 4),
        help="线程池并行数（默认: min(32, CPU*4)）",
    )
    return parser


def main() -> int:
    # 避免 Windows 终端默认编码导致中文路径/城市名打印报错
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_csv = Path(args.output_csv).expanduser()
    max_workers = max(1, int(args.max_workers))

    if not input_dir.exists():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        return 1
    if not input_dir.is_dir():
        print(f"[ERROR] 输入路径不是目录: {input_dir}")
        return 1

    map_files = discover_map_files(input_dir)
    if not map_files:
        print(f"[WARN] 在目录中未发现地图文件: {input_dir}")
        return 2

    print("=" * 80)
    print(f"[INFO] 输入目录: {input_dir}")
    print(f"[INFO] 文件数量: {len(map_files)}")
    print(f"[INFO] 并行线程数: {max_workers}")
    print("=" * 80)

    results: list[MapReadResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(read_one_map, str(file_path)) for file_path in map_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="读取进度", unit="file"):
            results.append(future.result())

    df = to_dataframe(results)
    if df.empty:
        print("[ERROR] 没有可输出结果。")
        return 3

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    ok_count = int((df["status"] == "ok").sum())
    failed_count = int((df["status"] == "failed").sum())
    city_names = sorted(df.loc[df["status"] == "ok", "city_name"].dropna().unique().tolist())

    print("=" * 80)
    print(f"[INFO] 输出文件: {output_csv}")
    print(f"[INFO] 成功: {ok_count} | 失败: {failed_count}")
    print(f"[INFO] 城市数量: {len(city_names)}")
    print(f"[INFO] 城市列表: {city_names}")
    if failed_count > 0:
        print("[INFO] 失败原因可在 CSV 的 error 列查看。")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
