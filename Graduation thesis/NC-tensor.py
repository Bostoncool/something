"""将 NetCDF 变量转换为可供 PyTorch 训练的张量.

使用示例：
    python NC-tensor.py --nc-file path/to/file.nc --var-name temperature \
        --sample-dim time --output tensor.pt --stats

依赖：xarray、netCDF4、torch、numpy
"""

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
            raise ValueError(f"目标维度 {missing} 不存在于变量 {dims} 中")
        return data.transpose(*target_dims)

    if sample_dim is None:
        return data

    if sample_dim not in dims:
        raise ValueError(f"样本维度 {sample_dim} 不存在于变量 {dims} 中")

    ordered_dims = [sample_dim] + [dim for dim in dims if dim != sample_dim]
    return data.transpose(*ordered_dims)


def _coerce_dtype(array: np.ndarray, dtype: str) -> np.ndarray:
    if dtype not in {"float32", "float64"}:
        raise ValueError("仅支持 float32 或 float64 输出类型")
    return array.astype(dtype, copy=False)


def _replace_missing(array: np.ndarray, fill_value: float) -> np.ndarray:
    return np.nan_to_num(array, nan=fill_value)


def load_nc_tensor(nc_file: Path, var_name: str, *, sample_dim: str | None,
                   target_dims: Sequence[str] | None, dtype: str,
                   fill_value: float, squeeze: bool) -> torch.Tensor:
    dataset = xr.open_dataset(nc_file)

    if var_name not in dataset.data_vars:
        available = ", ".join(dataset.data_vars)
        raise KeyError(f"变量 {var_name} 不存在，当前可用变量：{available}")

    data = dataset[var_name]

    # 解码 CF 元数据（如 scale_factor、add_offset）
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
            return "张量无有限值"

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
        description="读取 NetCDF 文件并输出 PyTorch 张量"
    )
    parser.add_argument("--nc-file", type=Path, required=True, help="NetCDF 文件路径")
    parser.add_argument("--var-name", type=str, required=True, help="目标变量名")
    parser.add_argument("--sample-dim", type=str, default=None,
                        help="将指定维度置于首位，常用于时间维度")
    parser.add_argument("--dims", type=str, default=None,
                        help="显式指定维度顺序，逗号分隔")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float64"], help="输出张量数据类型")
    parser.add_argument("--fill-value", type=float, default=0.0,
                        help="缺失值填充值")
    parser.add_argument("--squeeze", action="store_true",
                        help="移除长度为 1 的维度")
    parser.add_argument("--stats", action="store_true",
                        help="打印张量统计信息")
    parser.add_argument("--output", type=Path, default=None,
                        help="若指定，则使用 torch.save() 输出 *.pt 文件")
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
        print(f"张量已保存至 {args.output}")


if __name__ == "__main__":
    main()

