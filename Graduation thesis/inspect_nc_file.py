"""
NetCDF 文件结构核查工具
---------------------------------
用法示例：
    python inspect_nc_file.py "E:\\DATA Science\\ERA5-Beijing-NC\\2m_dewpoint_temperature\\201501.nc"
    python inspect_nc_file.py "E:\\DATA Science\\ERA5-Beijing-NC" --recursive --output report.csv

功能：
1. 读取单个或多个 .nc 文件并输出变量、维度、坐标等信息。
2. 检查关键字段是否缺失（time、latitude、longitude）以及 ERA5 常用变量。
3. 生成结构化 CSV 报告（可选）。
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import xarray as xr


DEFAULT_TARGET: Path = Path(
    r"E:\DATA Science\ERA5-Beijing-NC"
)
EXPECTED_COORDS: Sequence[str] = ("time", "latitude", "longitude")
EXPECTED_VARS: Sequence[str] = (
    "d2m",
    "t2m",
    "u10",
    "v10",
    "u100",
    "v100",
    "blh",
    "sp",
    "tcwv",
    "tp",
    "avg_tprate",
    "tisr",
    "str",
    "cvh",
    "cvl",
    "mn2t",
    "sd",
    "lsm",
)
COORD_ALIASES = {
    "valid_time": "time",
    "lat": "latitude",
    "lon": "longitude",
}


@dataclass
class InspectionResult:
    path: str
    status: str
    engine: Optional[str] = None
    dimensions: str = ""
    coordinates: str = ""
    variables: str = ""
    missing_coords: str = ""
    missing_vars: str = ""
    dtype_summary: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["notes"] = "; ".join(self.notes)
        return data


def configure_logger(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def summarize_dimensions(dims: Mapping[str, int]) -> str:
    return ", ".join(f"{name}={size}" for name, size in dims.items())


def summarize_variables(variables: Iterable[str]) -> str:
    return ", ".join(sorted(variables))


def summarize_dtypes(dataset: xr.Dataset) -> str:
    details = [
        f"{var}:{str(dataset[var].dtype)}"
        for var in sorted(dataset.data_vars)
    ]
    return ", ".join(details)


def inspect_nc_file(
    file_path: Path,
    expected_coords: Sequence[str] = EXPECTED_COORDS,
    expected_vars: Sequence[str] = EXPECTED_VARS,
    engine: str = "netcdf4",
) -> InspectionResult:
    result = InspectionResult(path=str(file_path), status="ok", engine=engine)

    if not file_path.exists():
        result.status = "missing_file"
        result.notes.append("文件不存在")
        return result

    if not file_path.is_file():
        result.status = "not_a_file"
        result.notes.append("路径不是文件")
        return result

    try:
        dataset = xr.open_dataset(file_path, engine=engine)
    except Exception as exc:  # pylint: disable=broad-except
        result.status = "open_failed"
        result.notes.append(f"无法打开文件: {exc}")
        return result

    try:
        coord_names = set(dataset.coords)
        dim_names = set(dataset.dims)
        var_names = set(dataset.data_vars)

        result.dimensions = summarize_dimensions(dataset.sizes)
        result.coordinates = summarize_variables(coord_names)
        result.variables = summarize_variables(var_names)
        result.dtype_summary = summarize_dtypes(dataset)

        missing_coord_aliases = []
        for coord in expected_coords:
            if coord not in coord_names:
                aliases = [alias for alias, canonical in COORD_ALIASES.items() if canonical == coord]
                alias_present = any(alias in coord_names for alias in aliases)
                if alias_present:
                    result.notes.append(f"坐标 {coord} 缺失，但存在别名 {aliases}")
                else:
                    missing_coord_aliases.append(coord)

        if missing_coord_aliases:
            result.status = "missing_coord"
            result.missing_coords = ", ".join(missing_coord_aliases)

        missing_variables = sorted(set(expected_vars) - var_names)
        if missing_variables:
            result.missing_vars = ", ".join(missing_variables)
            result.notes.append("部分 ERA5 变量缺失")

        redundant_dims = dim_names - set(expected_coords) - set(COORD_ALIASES)
        if redundant_dims:
            result.notes.append(f"额外维度: {sorted(redundant_dims)}")

        if "number" in dataset.dims:
            result.notes.append("检测到集合维度 number，可能需要先平均处理")

        for alias, canonical in COORD_ALIASES.items():
            if alias in coord_names and canonical not in coord_names:
                result.notes.append(f"建议重命名坐标 {alias} -> {canonical}")

    finally:
        dataset.close()

    return result


def collect_files(target: Path, recursive: bool) -> List[Path]:
    if target.is_file():
        return [target]
    searcher = target.rglob if recursive else target.glob
    pattern = "*.nc" if recursive else "*.nc"
    return sorted(searcher(pattern))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NetCDF 文件结构核查工具")
    parser.add_argument(
        "target",
        nargs="?",
        default=str(DEFAULT_TARGET),
        help=f"NetCDF 文件或目录路径（默认：{DEFAULT_TARGET}）",
    )
    parser.add_argument(
        "--engine",
        default="netcdf4",
        help="xarray 读取引擎（默认：netcdf4）",
    )
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="当 target 为目录时，递归遍历子目录（默认开启）",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="禁用递归遍历，仅检查当前目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="可选：将结果写入 CSV 报告",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出调试信息",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logger(verbose=args.verbose)

    target_path = Path(args.target).expanduser()
    logging.info("开始核查：%s", target_path)

    if not target_path.exists():
        logging.error("指定路径不存在：%s", target_path)
        return 1

    files = collect_files(target_path, recursive=args.recursive)
    if not files:
        logging.warning("未找到任何 .nc 文件")
        return 2

    results: List[InspectionResult] = []
    for file_path in files:
        logging.debug("检查文件：%s", file_path)
        result = inspect_nc_file(Path(file_path), engine=args.engine)
        results.append(result)

        logging.info(
            "[%s] %s | dims: %s | coords: %s",
            result.status,
            result.path,
            result.dimensions or "-",
            result.coordinates or "-",
        )
        if result.missing_coords:
            logging.warning("缺失坐标：%s", result.missing_coords)
        if result.missing_vars:
            logging.warning("缺失变量：%s", result.missing_vars)
        if result.notes:
            logging.info("备注：%s", "; ".join(result.notes))

    if args.output:
        output_path = Path(args.output).expanduser()
        df = pd.DataFrame([res.to_dict() for res in results])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logging.info("报告已保存：%s", output_path)

    logging.info("核查完成，共处理 %d 个文件", len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional

import xarray as xr


REQUIRED_COORDS = ("time", "latitude", "longitude")
ALT_COORD_MAP: Dict[str, str] = {
    "valid_time": "time",
    "lat": "latitude",
    "lon": "longitude",
}
OPTIONAL_DIMS = ("number", "ensemble")
OPTIONAL_COORDS = ("expver", "surface")
ERA5_VARS = [
    "d2m", "t2m", "u10", "v10", "u100", "v100",
    "blh", "sp", "tcwv", "tp", "avg_tprate",
    "tisr", "str", "cvh", "cvl", "mn2t", "sd", "lsm",
]


def inspect_nc_file(path: Path, expected_vars: Optional[Iterable[str]] = None) -> None:
    if not path.exists():
        print(f"❌ 文件不存在：{path}")
        return
    if not path.is_file():
        print(f"❌ 不是文件：{path}")
        return

    print(f"✅ 正在检查 NetCDF 文件：{path}")
    ds = xr.open_dataset(path, engine="netcdf4")
    print("\n--- Dataset 概要 ---")
    print(ds)

    print("\n--- 坐标/维度 ---")
    print(f"coords: {list(ds.coords)}")
    print(f"dims  : {dict(ds.sizes)}")

    missing_coords = [coord for coord in REQUIRED_COORDS if coord not in ds.coords]
    if missing_coords:
        print(f"\n⚠️ 缺失关键坐标：{missing_coords}")
        for alt, canonical in ALT_COORD_MAP.items():
            if alt in ds.coords and canonical not in ds.coords:
                print(f"  • 检测到可替代坐标 `{alt}` → 建议重命名为 `{canonical}`")
    else:
        print("\n✅ 关键坐标齐全")

    extra_coords = [coord for coord in OPTIONAL_COORDS if coord in ds.variables]
    if extra_coords:
        print(f"\nℹ️ 检测到可删除的辅助坐标变量：{extra_coords}")

    ensemble_dims = [dim for dim in OPTIONAL_DIMS if dim in ds.dims]
    if ensemble_dims:
        print(f"\nℹ️ 数据包含集合维度 {ensemble_dims}，后续可按需取均值或选择单成员")

    vars_available = [v for v in ds.data_vars]
    print("\n--- 数据变量 ---")
    for name in vars_available:
        print(f"  • {name}")

    if expected_vars:
        missing_vars = [v for v in expected_vars if v not in vars_available]
        if missing_vars:
            print(f"\n⚠️ 缺失预期变量：{missing_vars}")
        else:
            print("\n✅ 预期变量均存在")

    print("\n--- 建议操作 ---")
    if missing_coords or extra_coords or ensemble_dims:
        print(" 1) 可以在加载后执行 `.rename()`、`.drop_vars()` 或 `.mean(dim=...)` 处理。")
    else:
        print(" 1) 该文件结构符合预期，可直接进入数据处理流程。")
    print(" 2) 若需批量检查，可在外层循环遍历目录并调用 `inspect_nc_file`。")
    ds.close()


def main():
    parser = argparse.ArgumentParser(description="核查 NetCDF 文件格式与关键要素。")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_TARGET),
        help=f"NetCDF 文件路径或包含 .nc 的目录（默认：{DEFAULT_TARGET}）",
    )
    parser.add_argument("--check-era5-vars", action="store_true", help="额外校验 ERA5 常用变量")
    args = parser.parse_args()
    target = Path(args.path)

    if target.is_dir():
        files = sorted(target.glob("*.nc"))
        if not files:
            print(f"⚠️ 指定目录下未找到 .nc 文件：{target}")
            return
        for file in files:
            print("\n" + "=" * 80)
            inspect_nc_file(
                file,
                expected_vars=ERA5_VARS if args.check_era5_vars else None,
            )
    else:
        inspect_nc_file(
            target,
            expected_vars=ERA5_VARS if args.check_era5_vars else None,
        )


if __name__ == "__main__":
    main()