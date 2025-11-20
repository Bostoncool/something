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
        result.notes.append("File does not exist")
        return result

    if not file_path.is_file():
        result.status = "not_a_file"
        result.notes.append("Path is not a file")
        return result

    try:
        dataset = xr.open_dataset(file_path, engine=engine)
    except Exception as exc:  # pylint: disable=broad-except
        result.status = "open_failed"
        result.notes.append(f"Failed to open file: {exc}")
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
                    result.notes.append(f"Coordinate {coord} missing, but alias {aliases} exists")
                else:
                    missing_coord_aliases.append(coord)

        if missing_coord_aliases:
            result.status = "missing_coord"
            result.missing_coords = ", ".join(missing_coord_aliases)

        missing_variables = sorted(set(expected_vars) - var_names)
        if missing_variables:
            result.missing_vars = ", ".join(missing_variables)
            result.notes.append("Some ERA5 variables are missing")

        redundant_dims = dim_names - set(expected_coords) - set(COORD_ALIASES)
        if redundant_dims:
            result.notes.append(f"Extra dimensions: {sorted(redundant_dims)}")

        if "number" in dataset.dims:
            result.notes.append("Detected ensemble dimension 'number', may need averaging first")

        for alias, canonical in COORD_ALIASES.items():
            if alias in coord_names and canonical not in coord_names:
                result.notes.append(f"Suggest renaming coordinate {alias} -> {canonical}")

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
    parser = argparse.ArgumentParser(description="NetCDF File Structure Inspection Tool")
    parser.add_argument(
        "target",
        nargs="?",
        default=str(DEFAULT_TARGET),
        help=f"NetCDF file or directory path (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--engine",
        default="netcdf4",
        help="xarray reading engine (default: netcdf4)",
    )
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="When target is a directory, recursively traverse subdirectories (enabled by default)",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Disable recursive traversal, only check current directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional: write results to CSV report",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Output debug information",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logger(verbose=args.verbose)

    target_path = Path(args.target).expanduser()
    logging.info("Starting inspection: %s", target_path)

    if not target_path.exists():
        logging.error("Specified path does not exist: %s", target_path)
        return 1

    files = collect_files(target_path, recursive=args.recursive)
    if not files:
        logging.warning("No .nc files found")
        return 2

    results: List[InspectionResult] = []
    for file_path in files:
        logging.debug("Checking file: %s", file_path)
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
            logging.warning("Missing coordinates: %s", result.missing_coords)
        if result.missing_vars:
            logging.warning("Missing variables: %s", result.missing_vars)
        if result.notes:
            logging.info("Notes: %s", "; ".join(result.notes))

    if args.output:
        output_path = Path(args.output).expanduser()
        df = pd.DataFrame([res.to_dict() for res in results])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logging.info("Report saved: %s", output_path)

    logging.info("Inspection completed, processed %d files", len(results))
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
        print(f"❌ File does not exist: {path}")
        return
    if not path.is_file():
        print(f"❌ Not a file: {path}")
        return

    print(f"✅ Checking NetCDF file: {path}")
    ds = xr.open_dataset(path, engine="netcdf4")
    print("\n--- Dataset Summary ---")
    print(ds)

    print("\n--- Coordinates/Dimensions ---")
    print(f"coords: {list(ds.coords)}")
    print(f"dims  : {dict(ds.sizes)}")

    missing_coords = [coord for coord in REQUIRED_COORDS if coord not in ds.coords]
    if missing_coords:
        print(f"\n⚠️ Missing key coordinates: {missing_coords}")
        for alt, canonical in ALT_COORD_MAP.items():
            if alt in ds.coords and canonical not in ds.coords:
                print(f"  • Detected alternative coordinate `{alt}` → suggest renaming to `{canonical}`")
    else:
        print("\n✅ All key coordinates present")

    extra_coords = [coord for coord in OPTIONAL_COORDS if coord in ds.variables]
    if extra_coords:
        print(f"\nℹ️ Detected auxiliary coordinate variables that can be removed: {extra_coords}")

    ensemble_dims = [dim for dim in OPTIONAL_DIMS if dim in ds.dims]
    if ensemble_dims:
        print(f"\nℹ️ Data contains ensemble dimensions {ensemble_dims}, may need averaging or single member selection")

    vars_available = [v for v in ds.data_vars]
    print("\n--- Data Variables ---")
    for name in vars_available:
        print(f"  • {name}")

    if expected_vars:
        missing_vars = [v for v in expected_vars if v not in vars_available]
        if missing_vars:
            print(f"\n⚠️ Missing expected variables: {missing_vars}")
        else:
            print("\n✅ All expected variables present")

    print("\n--- Suggested Actions ---")
    if missing_coords or extra_coords or ensemble_dims:
        print(" 1) Can execute `.rename()`, `.drop_vars()`, or `.mean(dim=...)` after loading.")
    else:
        print(" 1) File structure meets expectations, can proceed directly to data processing.")
    print(" 2) For batch inspection, iterate through directories in outer loop and call `inspect_nc_file`.")
    ds.close()


def main():
    parser = argparse.ArgumentParser(description="Inspect NetCDF file format and key elements.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_TARGET),
        help=f"NetCDF file path or directory containing .nc files (default: {DEFAULT_TARGET})",
    )
    parser.add_argument("--check-era5-vars", action="store_true", help="Additionally validate common ERA5 variables")
    args = parser.parse_args()
    target = Path(args.path)

    if target.is_dir():
        files = sorted(target.glob("*.nc"))
        if not files:
            print(f"⚠️ No .nc files found in specified directory: {target}")
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