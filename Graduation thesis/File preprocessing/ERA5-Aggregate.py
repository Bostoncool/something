from __future__ import annotations

import gc
import re
import shutil
import tempfile
from pathlib import Path

import xarray as xr


SOURCE_ROOT = Path(r"F:\1.模型要用的\2018-2023[ERA5_PM2.5]")
OUTPUT_ROOT = Path(r"F:\1.模型要用的\Year")
YEAR_RANGE = range(2018, 2024)  # 2018-2023


def infer_time_dim(data_array: xr.DataArray) -> str:
    """推断时间维度名称，优先匹配包含 time 的维度名。"""
    dims = list(data_array.dims)
    lowered = {dim: dim.lower() for dim in dims}

    # 常见时间维名称优先
    for dim in dims:
        dim_lower = lowered[dim]
        if "time" in dim_lower:
            return dim

    # 兜底：从空间维以外的第一维作为时间维
    for dim in dims:
        dim_lower = lowered[dim]
        if dim_lower not in {"latitude", "longitude", "lat", "lon"}:
            return dim

    raise ValueError(f"无法识别时间维度，变量维度为: {dims}")


def parse_year_month(file_name: str) -> tuple[int, int]:
    """从文件名解析年月，格式示例: xxx_201801.nc"""
    match = re.search(r"_(\d{4})(\d{2})\.nc$", file_name)
    if not match:
        raise ValueError(f"文件名不符合预期格式: {file_name}")
    return int(match.group(1)), int(match.group(2))


def compute_month_sum_count(
    file_path: Path,
) -> tuple[xr.DataArray, xr.DataArray, str, dict]:
    """
    计算单月文件在时间维上的 sum/count（用于最终年均）。

    读取策略：
    1) 原始路径尝试 netcdf4 / h5netcdf
    2) 若失败，将文件复制到 ASCII 临时路径后再尝试 netcdf4 / h5netcdf
    """
    attempts: list[tuple[Path, str, tempfile.TemporaryDirectory | None]] = []
    attempts.append((file_path, "netcdf4", None))
    attempts.append((file_path, "h5netcdf", None))

    # 复制到临时目录（纯 ASCII 路径）作为兜底
    tmp_dir = tempfile.TemporaryDirectory(prefix="era5_nc_")
    tmp_file = Path(tmp_dir.name) / file_path.name
    shutil.copy2(file_path, tmp_file)
    attempts.append((tmp_file, "netcdf4", tmp_dir))
    attempts.append((tmp_file, "h5netcdf", tmp_dir))

    last_error = None
    used_tmp_dir = False

    for candidate_path, engine, candidate_tmp_dir in attempts:
        ds = None
        try:
            ds = xr.open_dataset(candidate_path, engine=engine)
            current_var_name = list(ds.data_vars)[0]
            data_array = ds[current_var_name]
            time_dim = infer_time_dim(data_array)

            month_sum = data_array.sum(dim=time_dim, skipna=True)
            month_count = data_array.count(dim=time_dim)
            template_attrs = dict(data_array.isel({time_dim: 0}, drop=True).attrs)

            if candidate_tmp_dir is not None:
                used_tmp_dir = True
            return month_sum, month_count, current_var_name, template_attrs
        except Exception as exc:
            last_error = exc
        finally:
            if ds is not None:
                ds.close()
            gc.collect()

    # 所有尝试都失败后清理临时目录并抛错
    if not used_tmp_dir:
        tmp_dir.cleanup()
    else:
        tmp_dir.cleanup()

    raise RuntimeError(f"读取失败: {file_path}") from last_error


def aggregate_one_variable(variable_dir: Path, output_root: Path) -> None:
    """聚合单个气象变量目录，按年输出年均数据。"""
    nc_files = sorted(variable_dir.glob("*.nc"))
    if not nc_files:
        print(f"[跳过] 未找到 nc 文件: {variable_dir}")
        return

    # 按年组织文件
    files_by_year: dict[int, list[Path]] = {year: [] for year in YEAR_RANGE}
    for file_path in nc_files:
        try:
            year, month = parse_year_month(file_path.name)
        except ValueError:
            print(f"[警告] 跳过异常命名文件: {file_path.name}")
            continue

        if year in files_by_year and 1 <= month <= 12:
            files_by_year[year].append(file_path)

    # 输出目录：Year/<变量名>/
    var_output_dir = output_root / variable_dir.name
    var_output_dir.mkdir(parents=True, exist_ok=True)

    for year in YEAR_RANGE:
        year_files = sorted(files_by_year[year], key=lambda p: p.name)
        if not year_files:
            print(f"[警告] {variable_dir.name} {year} 年无月文件，跳过该年。")
            continue
        if len(year_files) != 12:
            print(
                f"[警告] {variable_dir.name} {year} 年文件数为 {len(year_files)}，"
                "将使用可读取月份计算年均。"
            )

        annual_sum = None
        annual_count = None
        var_name = None
        template_attrs = {}
        success_months = 0
        failed_months: list[str] = []

        print(f"[开始] 变量={variable_dir.name}, 年份={year}")

        # 一次只读取一个月，避免内存峰值过高
        for month_file in year_files:
            try:
                month_sum, month_count, current_var_name, month_attrs = (
                    compute_month_sum_count(month_file)
                )
                if annual_sum is None:
                    annual_sum = month_sum
                    annual_count = month_count
                    var_name = current_var_name
                    template_attrs = month_attrs
                else:
                    annual_sum = annual_sum + month_sum
                    annual_count = annual_count + month_count
                success_months += 1
            except Exception as exc:
                print(
                    f"[错误] 变量={variable_dir.name}, 年份={year}, 文件={month_file.name} "
                    f"读取失败，跳过该月。原因: {exc}"
                )
                failed_months.append(month_file.name)
                continue
            finally:
                gc.collect()

        if annual_sum is None or annual_count is None or var_name is None:
            print(f"[警告] {variable_dir.name} {year} 年无有效数据，跳过。")
            continue

        annual_mean = annual_sum / annual_count.where(annual_count != 0)
        annual_mean = annual_mean.rename(var_name)
        annual_mean.attrs.update(template_attrs)
        annual_mean.attrs["aggregation"] = "annual mean from hourly data"
        annual_mean.attrs["year"] = year
        annual_mean.attrs["months_used"] = success_months

        out_ds = annual_mean.to_dataset()
        out_ds.attrs["source_variable_dir"] = variable_dir.name
        out_ds.attrs["aggregation_method"] = "sum_over_time / count_over_time"
        out_ds.attrs["months_used"] = success_months
        out_ds.attrs["failed_month_count"] = len(failed_months)
        out_ds.attrs["failed_month_files"] = ",".join(failed_months)

        output_file = var_output_dir / f"{variable_dir.name}_{year}_annual_mean.nc"
        out_ds.to_netcdf(output_file, engine="h5netcdf")
        out_ds.close()

        # 处理完一年后显式释放内存
        del annual_sum, annual_count, annual_mean, out_ds, template_attrs
        gc.collect()

        print(
            f"[完成] {output_file} (有效月份: {success_months}/"
            f"{len(year_files)}, 失败月份: {len(failed_months)})"
        )


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"输入目录不存在: {SOURCE_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    variable_dirs = sorted([p for p in SOURCE_ROOT.iterdir() if p.is_dir()])
    if not variable_dirs:
        print(f"[结束] 未找到变量子目录: {SOURCE_ROOT}")
        return

    print(f"[信息] 变量目录数量: {len(variable_dirs)}")
    for variable_dir in variable_dirs:
        aggregate_one_variable(variable_dir, OUTPUT_ROOT)
        gc.collect()

    print("[结束] 全部变量处理完成。")


if __name__ == "__main__":
    main()
