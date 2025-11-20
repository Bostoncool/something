from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import xarray as xr


def _resolve_dim_order(data: xr.DataArray, sample_dim: str | None,
                       target_dims: Sequence[str] | None) -> xr.DataArray:
    dims = list(data.dims)

    if target_dims is not None:
        missing = set(target_dims) - set(dims)
        if missing:
            raise ValueError(f"Target dimensions {missing} not found in variable {dims}")
        return data.transpose(*target_dims)

    if sample_dim is None:
        return data

    if sample_dim not in dims:
        raise ValueError(f"Sample dimension {sample_dim} not found in variable {dims}")

    ordered_dims = [sample_dim] + [dim for dim in dims if dim != sample_dim]
    return data.transpose(*ordered_dims)


def _coerce_dtype(array: np.ndarray, dtype: str) -> np.ndarray:
    if dtype not in {"float32", "float64"}:
        raise ValueError("Only float32 or float64 output types are supported")
    return array.astype(dtype, copy=False)


def _replace_missing(array: np.ndarray, fill_value: float) -> np.ndarray:
    return np.nan_to_num(array, nan=fill_value)


def load_nc_tensor(nc_file: Path, var_name: str, *, sample_dim: str | None,
                   target_dims: Sequence[str] | None, dtype: str,
                   fill_value: float, squeeze: bool) -> torch.Tensor:
    dataset = xr.open_dataset(nc_file)

    if var_name not in dataset.data_vars:
        available = ", ".join(dataset.data_vars)
        raise KeyError(f"Variable {var_name} not found, available variables: {available}")

    data = dataset[var_name]

    # Decode CF metadata (e.g., scale_factor, add_offset)
    data = xr.decode_cf(data.to_dataset(name=var_name))[var_name]

    data = data.squeeze() if squeeze else data
    data = _resolve_dim_order(data, sample_dim, target_dims)

    np_array = data.to_numpy()
    np_array = _replace_missing(np_array, fill_value)
    np_array = _coerce_dtype(np_array, dtype)

    tensor = torch.from_numpy(np_array)
    return tensor.contiguous()


def _format_stats(tensor: torch.Tensor) -> str:
    with torch.no_grad():
        finite_tensor = tensor[torch.isfinite(tensor)]
        if finite_tensor.numel() == 0:
            return "Tensor has no finite values"

        stats = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "min": float(finite_tensor.min()),
            "max": float(finite_tensor.max()),
            "mean": float(finite_tensor.mean()),
            "std": float(finite_tensor.std()),
        }

    return ", ".join(f"{key}={value}" for key, value in stats.items())


def parse_dims(dims: str | None) -> Sequence[str] | None:
    if dims is None:
        return None
    cleaned = [dim.strip() for dim in dims.split(",") if dim.strip()]
    return cleaned or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read NetCDF files and output PyTorch tensors"
    )
    parser.add_argument("--nc-file", type=Path, required=True, help="NetCDF file path")
    parser.add_argument("--var-name", type=str, required=True, help="Target variable name")
    parser.add_argument("--sample-dim", type=str, default=None,
                        help="Place specified dimension first, commonly used for time dimension")
    parser.add_argument("--dims", type=str, default=None,
                        help="Explicitly specify dimension order, comma-separated")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float64"], help="Output tensor data type")
    parser.add_argument("--fill-value", type=float, default=0.0,
                        help="Fill value for missing values")
    parser.add_argument("--squeeze", action="store_true",
                        help="Remove dimensions of length 1")
    parser.add_argument("--stats", action="store_true",
                        help="Print tensor statistics")
    parser.add_argument("--output", type=Path, default=None,
                        help="If specified, use torch.save() to output *.pt file")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    tensor = load_nc_tensor(
        nc_file=args.nc_file,
        var_name=args.var_name,
        sample_dim=args.sample_dim,
        target_dims=parse_dims(args.dims),
        dtype=args.dtype,
        fill_value=args.fill_value,
        squeeze=args.squeeze,
    )

    if args.stats:
        print(_format_stats(tensor))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, args.output)
        print(f"Tensor saved to {args.output}")


if __name__ == "__main__":
    main()

