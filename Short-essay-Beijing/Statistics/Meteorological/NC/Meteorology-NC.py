import glob
import hashlib
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from netCDF4 import Dataset  # noqa: F401

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (16, 8)


class DataCache:
    """基于文件修改时间的简单缓存."""

    def __init__(self, cache_dir: str = "cache_meteorology_nc") -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, filepath: str) -> str:
        stat = os.stat(filepath)
        payload = f"{filepath}_{stat.st_mtime}".encode("utf-8")
        return hashlib.md5(payload).hexdigest()

    def load(self, filepath: str) -> Optional[Dict[str, float]]:
        cache_file = os.path.join(self.cache_dir, f"{self._cache_key(filepath)}.pkl")
        if not os.path.exists(cache_file):
            return None
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            return None

    def save(self, filepath: str, data: Dict[str, float]) -> None:
        cache_file = os.path.join(self.cache_dir, f"{self._cache_key(filepath)}.pkl")
        pd.to_pickle(data, cache_file)


class MeteorologicalAnalyzerNC:
    """基于 ERA5 NetCDF 的北京气象数据分析器."""

    def __init__(self, data_dir: str = ".", max_workers: Optional[int] = None) -> None:
        self.data_dir = data_dir
        self.max_workers = (
            max_workers
            if max_workers
            else min(max(4, (os.cpu_count() or 4) - 1), 12)
        )
        self.print_lock = Lock()
        self.cache = DataCache()

        self.meteo_data = pd.DataFrame()

        self.beijing_lats = np.arange(39.0, 41.25, 0.25)
        self.beijing_lons = np.arange(115.0, 117.25, 0.25)

        self.era5_vars = [
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
        ]

        self.param_names_en = {
            "t2m": "2m Temperature",
            "d2m": "2m Dewpoint Temperature",
            "blh": "Boundary Layer Height",
            "cvh": "High Vegetation Cover",
            "avg_tprate": "Mean Precipitation Rate",
            "u10": "10m U Wind Component",
            "v10": "10m V Wind Component",
            "u100": "100m U Wind Component",
            "v100": "100m V Wind Component",
            "lsm": "Land Sea Mask",
            "cvl": "Low Vegetation Cover",
            "mn2t": "Minimum 2m Temperature",
            "sp": "Surface Pressure",
            "sd": "Snow Depth",
            "str": "Surface Net Thermal Radiation",
            "tisr": "TOA Incident Solar Radiation",
            "tcwv": "Total Column Water Vapour",
            "tp": "Total Precipitation",
        }

    # -------------------------------------------------------------------------
    # 数据发现
    # -------------------------------------------------------------------------
    def find_meteorological_files(self) -> List[str]:
        pattern = os.path.join(self.data_dir, "**", "*.nc")
        files = glob.glob(pattern, recursive=True)
        files.sort()

        with self.print_lock:
            print(f"搜索目录: {self.data_dir}")
            print(f"发现 {len(files)} 个 NetCDF 文件")
            if files:
                print(f"示例文件: {os.path.basename(files[0])}")
        return files

    # -------------------------------------------------------------------------
    # NC 文件处理
    # -------------------------------------------------------------------------
    def _rename_common_coords(self, dataset: xr.Dataset) -> xr.Dataset:
        rename_map: Dict[str, str] = {}
        for time_key in (
            "valid_time",
            "forecast_time",
            "verification_time",
            "time1",
            "time2",
        ):
            if time_key in dataset.coords and "time" not in dataset.coords:
                rename_map[time_key] = "time"
        if "lat" in dataset.coords and "latitude" not in dataset.coords:
            rename_map["lat"] = "latitude"
        if "lon" in dataset.coords and "longitude" not in dataset.coords:
            rename_map["lon"] = "longitude"
        return dataset.rename(rename_map) if rename_map else dataset

    def _select_beijing_region(self, dataset: xr.Dataset) -> xr.Dataset:
        if "latitude" not in dataset.coords or "longitude" not in dataset.coords:
            return dataset

        latitude_values = dataset["latitude"].values
        if latitude_values[0] > latitude_values[-1]:
            lat_slice = slice(self.beijing_lats.max(), self.beijing_lats.min())
        else:
            lat_slice = slice(self.beijing_lats.min(), self.beijing_lats.max())

        dataset = dataset.sel(
            latitude=lat_slice,
            longitude=slice(self.beijing_lons.min(), self.beijing_lons.max()),
        )
        if "latitude" in dataset.dims and "longitude" in dataset.dims:
            dataset = dataset.mean(dim=["latitude", "longitude"], skipna=True)
        return dataset

    def _process_single_file(self, filepath: str) -> Optional[Dict[str, float]]:
        cached = self.cache.load(filepath)
        if cached:
            return cached

        try:
            with xr.open_dataset(
                filepath, engine="netcdf4", decode_times=True, cache=False
            ) as dataset:
                dataset = self._rename_common_coords(dataset)
                try:
                    dataset = xr.decode_cf(dataset)
                except Exception:
                    pass

                for extra in ("expver", "surface"):
                    if extra in dataset.variables:
                        dataset = dataset.drop_vars(extra)

                if "number" in dataset.dims:
                    dataset = dataset.mean(dim="number", skipna=True)

                available_vars = [
                    var for var in self.era5_vars if var in dataset.data_vars
                ]
                if not available_vars:
                    with self.print_lock:
                        print(
                            f"[WARN] {os.path.basename(filepath)} 缺少目标变量，跳过"
                        )
                    return None

                dataset = self._select_beijing_region(dataset)

                if "time" not in dataset.coords:
                    with self.print_lock:
                        print(
                            f"[WARN] {os.path.basename(filepath)} 缺少时间维度，跳过"
                        )
                    return None

                dataset = dataset.sortby("time")
                dataset = dataset.resample(time="1D").mean(keep_attrs=False)
                dataset = dataset.dropna("time", how="all")
                if dataset.sizes.get("time", 0) == 0:
                    with self.print_lock:
                        print(
                            f"[WARN] {os.path.basename(filepath)} 无有效时间步，跳过"
                        )
                    return None

                dataset.load()

                stats: Dict[str, float] = {}
                time_index = pd.to_datetime(dataset["time"].values)
                if len(time_index) > 0:
                    first_time = time_index[0]
                    stats["year"] = int(first_time.year)
                    stats["month"] = int(first_time.month)
                    stats["days"] = int(len(time_index))
                else:
                    stats["days"] = int(dataset.sizes.get("time", 0))

                if "year" not in stats or "month" not in stats:
                    match = re.search(r"(\d{4})(\d{2})", os.path.basename(filepath))
                    if match:
                        stats["year"] = int(match.group(1))
                        stats["month"] = int(match.group(2))

                year = stats.get("year")
                month = stats.get("month")
                if year is not None and month is not None:
                    stats["year_month"] = f"{int(year):04d}{int(month):02d}"
                else:
                    stats["year_month"] = os.path.basename(filepath)

                for var in available_vars:
                    try:
                        values = np.asarray(dataset[var].values, dtype=np.float32)
                        values = values[np.isfinite(values)]
                        if values.size == 0:
                            continue

                        if var in {"t2m", "d2m", "mn2t"} and np.nanmax(values) > 100:
                            values = values - 273.15

                        stats[f"{var}_mean"] = float(np.nanmean(values))
                        stats[f"{var}_std"] = float(np.nanstd(values))
                        stats[f"{var}_min"] = float(np.nanmin(values))
                        stats[f"{var}_max"] = float(np.nanmax(values))
                        stats[f"{var}_median"] = float(np.nanmedian(values))
                    except Exception as err:
                        with self.print_lock:
                            print(
                                f"[ERROR] {os.path.basename(filepath)} 变量 {var} 处理失败: {err}"
                            )
                        continue

                stats["source_file"] = os.path.basename(filepath)
                self.cache.save(filepath, stats)
                with self.print_lock:
                    print(
                        f"  [+] {os.path.basename(filepath)} -> 变量: {len(available_vars)}, "
                        f"时间步: {stats.get('days', 0)}"
                    )
                return stats
        except Exception as exc:
            with self.print_lock:
                print(
                    f"[ERROR] 打开 {os.path.basename(filepath)} 失败: "
                    f"{type(exc).__name__}: {exc}"
                )
            return None

    # -------------------------------------------------------------------------
    # 数据加载主流程
    # -------------------------------------------------------------------------
    def load_all_data(self) -> None:
        with self.print_lock:
            print("\n开始使用并行方式加载气象数据（NC）...")
            print(f"线程数: {self.max_workers}")

        files = self.find_meteorological_files()
        if not files:
            with self.print_lock:
                print("错误: 未找到任何 NetCDF 文件")
            self.meteo_data = pd.DataFrame()
            return

        all_stats: List[Dict[str, float]] = []
        total = len(files)
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_single_file, path): path for path in files}
            for future in as_completed(futures):
                filepath = futures[future]
                completed += 1
                result = future.result()
                if result:
                    all_stats.append(result)

                if completed % 20 == 0 or completed == total:
                    with self.print_lock:
                        print(f"进度: [{completed}/{total}] 已处理 {os.path.basename(filepath)}")

        if not all_stats:
            with self.print_lock:
                print("错误: 数据加载失败，未得到有效统计结果")
            self.meteo_data = pd.DataFrame()
            return

        aggregated: Dict[Tuple[int, int], Dict[str, float]] = {}
        leftovers: List[Dict[str, float]] = []

        for item in all_stats:
            year = item.get("year")
            month = item.get("month")
            if year is None or month is None or pd.isna(year) or pd.isna(month):
                leftovers.append(item)
                continue

            year_int = int(year)
            month_int = int(month)
            key = (year_int, month_int)

            bucket = aggregated.setdefault(
                key,
                {
                    "year": year_int,
                    "month": month_int,
                    "year_month": item.get(
                        "year_month", f"{year_int:04d}{month_int:02d}"
                    ),
                    "days": int(item.get("days", 0)),
                    "source_files": [],
                },
            )

            bucket["days"] = max(bucket.get("days", 0), int(item.get("days", 0)))

            source_file = item.get("source_file")
            if source_file:
                sources = bucket.setdefault("source_files", [])
                sources.append(source_file)

            for key_name, value in item.items():
                if key_name in {
                    "source_file",
                    "year",
                    "month",
                    "year_month",
                    "days",
                }:
                    continue
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue
                if key_name not in bucket or pd.isna(bucket[key_name]):
                    bucket[key_name] = value

        if aggregated:
            aggregated_list: List[Dict[str, float]] = []
            for key in sorted(aggregated.keys()):
                row = aggregated[key]
                if "source_files" in row:
                    unique_sources = []
                    for src in row["source_files"]:
                        if src not in unique_sources:
                            unique_sources.append(src)
                    row["source_files"] = "|".join(unique_sources)
                aggregated_list.append(row)
            aggregated_list.extend(leftovers)
            self.meteo_data = pd.DataFrame(aggregated_list)
        else:
            self.meteo_data = pd.DataFrame(all_stats)
        self._finalize_dataset()

    def _finalize_dataset(self) -> None:
        numeric_cols = self.meteo_data.select_dtypes(include=[np.number]).columns
        self.meteo_data[numeric_cols] = self.meteo_data[numeric_cols].replace(
            [np.inf, -np.inf], np.nan
        )
        self.meteo_data = self.meteo_data.sort_values(["year", "month"]).reset_index(
            drop=True
        )

        if {"year", "month"}.issubset(self.meteo_data.columns):
            self.meteo_data["date"] = pd.to_datetime(
                self.meteo_data["year"].astype(int).astype(str)
                + "-"
                + self.meteo_data["month"].astype(int).astype(str).str.zfill(2)
                + "-01"
            )

        if {"u10_mean", "v10_mean"}.issubset(self.meteo_data.columns):
            self.meteo_data["wind_speed_10m"] = np.sqrt(
                self.meteo_data["u10_mean"] ** 2 + self.meteo_data["v10_mean"] ** 2
            )
        if {"u100_mean", "v100_mean"}.issubset(self.meteo_data.columns):
            self.meteo_data["wind_speed_100m"] = np.sqrt(
                self.meteo_data["u100_mean"] ** 2 + self.meteo_data["v100_mean"] ** 2
            )

        with self.print_lock:
            print("\n气象数据加载完成!")
            print(f"总记录数: {len(self.meteo_data)}")
            if "date" in self.meteo_data.columns:
                print(
                    f"时间范围: {self.meteo_data['date'].min()} -> "
                    f"{self.meteo_data['date'].max()}"
                )

    # -------------------------------------------------------------------------
    # 可视化与统计
    # -------------------------------------------------------------------------
    def plot_temperature_timeseries(self, save_path: str = "temperature_timeseries_nc.png") -> None:
        if self.meteo_data.empty:
            print("错误: 无数据可用于温度序列图绘制")
            return

        print("\n开始绘制温度时间序列图...")

        temp_params = ["t2m", "d2m", "mn2t"]
        available_temps = [p for p in temp_params if f"{p}_mean" in self.meteo_data.columns]

        if not available_temps:
            print("警告: 无可用温度变量")
            return

        fig, ax = plt.subplots(figsize=(20, 8))
        colors = ["#e74c3c", "#3498db", "#2ecc71"]

        for idx, param in enumerate(available_temps):
            mean_col = f"{param}_mean"
            min_col = f"{param}_min"
            max_col = f"{param}_max"

            ax.plot(
                self.meteo_data["date"],
                self.meteo_data[mean_col],
                linewidth=2.5,
                color=colors[idx],
                alpha=0.9,
                label=f"{self.param_names_en.get(param, param)} (Mean)",
                marker="o",
                markersize=4,
            )

            if min_col in self.meteo_data.columns and max_col in self.meteo_data.columns:
                ax.fill_between(
                    self.meteo_data["date"],
                    self.meteo_data[min_col],
                    self.meteo_data[max_col],
                    alpha=0.2,
                    color=colors[idx],
                )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="0°C")
        ax.set_xlabel("Time", fontsize=14, fontweight="bold")
        ax.set_ylabel("Temperature (°C)", fontsize=14, fontweight="bold")
        ax.set_title("Beijing Temperature Time Series (NC)", fontsize=18, fontweight="bold", pad=30)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=11, framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
        plt.xticks(rotation=45, ha="right")

        if "t2m_mean" in self.meteo_data.columns:
            stats_text = (
                "2m Temperature Statistics:\n"
                f"Sample: {len(self.meteo_data)} months\n"
                f"Mean: {self.meteo_data['t2m_mean'].mean():.2f} °C\n"
                f"Median: {self.meteo_data['t2m_mean'].median():.2f} °C\n"
                f"Max: {self.meteo_data['t2m_mean'].max():.2f} °C\n"
                f"Min: {self.meteo_data['t2m_mean'].min():.2f} °C"
            )
            ax.text(
                0.02,
                0.02,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"温度时间序列图已保存至: {save_path}")
        plt.show()

    def plot_wind_speed_analysis(self, save_path: str = "wind_speed_analysis_nc.png") -> None:
        if self.meteo_data.empty:
            print("错误: 无数据可用于风速分析")
            return

        print("\n开始绘制风速分析图...")

        if "wind_speed_10m" not in self.meteo_data.columns and {
            "u10_mean",
            "v10_mean",
        }.issubset(self.meteo_data.columns):
            self.meteo_data["wind_speed_10m"] = np.sqrt(
                self.meteo_data["u10_mean"] ** 2 + self.meteo_data["v10_mean"] ** 2
            )
        if "wind_speed_100m" not in self.meteo_data.columns and {
            "u100_mean",
            "v100_mean",
        }.issubset(self.meteo_data.columns):
            self.meteo_data["wind_speed_100m"] = np.sqrt(
                self.meteo_data["u100_mean"] ** 2 + self.meteo_data["v100_mean"] ** 2
            )

        available = [
            col for col in ["wind_speed_10m", "wind_speed_100m"] if col in self.meteo_data.columns
        ]
        if not available:
            print("警告: 无可用风速变量")
            return

        fig, axes = plt.subplots(2, 1, figsize=(20, 12))

        if "wind_speed_10m" in available:
            axes[0].plot(
                self.meteo_data["date"],
                self.meteo_data["wind_speed_10m"],
                linewidth=2.5,
                color="#3498db",
                alpha=0.9,
                label="10m Wind Speed",
                marker="o",
                markersize=4,
            )
        if "wind_speed_100m" in available:
            axes[0].plot(
                self.meteo_data["date"],
                self.meteo_data["wind_speed_100m"],
                linewidth=2.5,
                color="#e74c3c",
                alpha=0.9,
                label="100m Wind Speed",
                marker="s",
                markersize=4,
            )

        axes[0].set_xlabel("Time", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Wind Speed (m/s)", fontsize=14, fontweight="bold")
        axes[0].set_title("Beijing Wind Speed Time Series (NC)", fontsize=16, fontweight="bold", pad=15)
        axes[0].grid(True, alpha=0.3, linestyle="--")
        axes[0].legend(loc="upper right", fontsize=11)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

        wind_data = []
        labels = []
        if "wind_speed_10m" in available:
            wind_data.append(self.meteo_data["wind_speed_10m"].dropna())
            labels.append("10m Wind Speed")
        if "wind_speed_100m" in available:
            wind_data.append(self.meteo_data["wind_speed_100m"].dropna())
            labels.append("100m Wind Speed")

        axes[1].hist(
            wind_data,
            bins=30,
            label=labels,
            color=["#3498db", "#e74c3c"][: len(wind_data)],
            alpha=0.7,
            edgecolor="white",
            linewidth=1,
        )
        axes[1].set_xlabel("Wind Speed (m/s)", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Frequency", fontsize=14, fontweight="bold")
        axes[1].set_title("Wind Speed Distribution (NC)", fontsize=16, fontweight="bold", pad=15)
        axes[1].grid(True, alpha=0.3, linestyle="--")
        axes[1].legend(loc="upper right", fontsize=11)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"风速分析图已保存至: {save_path}")
        plt.show()

    def plot_precipitation_analysis(self, save_path: str = "precipitation_analysis_nc.png") -> None:
        if self.meteo_data.empty:
            print("错误: 无数据可用于降水分析")
            return

        print("\n开始绘制降水分析图...")

        if "tp_mean" not in self.meteo_data.columns and "avg_tprate_mean" not in self.meteo_data.columns:
            print("警告: 无可用降水变量")
            return

        fig, axes = plt.subplots(2, 1, figsize=(20, 12))

        if "tp_mean" in self.meteo_data.columns:
            axes[0].bar(
                self.meteo_data["date"],
                self.meteo_data["tp_mean"],
                width=20,
                color="#3498db",
                alpha=0.7,
                label="Total Precipitation",
            )
            axes[0].set_xlabel("Time", fontsize=14, fontweight="bold")
            axes[0].set_ylabel("Precipitation (mm)", fontsize=14, fontweight="bold")
            axes[0].set_title("Beijing Precipitation Time Series (NC)", fontsize=16, fontweight="bold", pad=15)
            axes[0].grid(True, alpha=0.3, linestyle="--", axis="y")
            axes[0].legend(loc="upper right", fontsize=11)
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

        if "avg_tprate_mean" in self.meteo_data.columns:
            axes[1].plot(
                self.meteo_data["date"],
                self.meteo_data["avg_tprate_mean"],
                linewidth=2.5,
                color="#2ecc71",
                alpha=0.9,
                label="Mean Precipitation Rate",
                marker="o",
                markersize=4,
            )
            axes[1].set_xlabel("Time", fontsize=14, fontweight="bold")
            axes[1].set_ylabel("Precipitation Rate (mm/h)", fontsize=14, fontweight="bold")
            axes[1].set_title(
                "Beijing Mean Precipitation Rate Time Series (NC)", fontsize=16, fontweight="bold", pad=15
            )
            axes[1].grid(True, alpha=0.3, linestyle="--")
            axes[1].legend(loc="upper right", fontsize=11)
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
        else:
            axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"降水分析图已保存至: {save_path}")
        plt.show()

    def plot_seasonal_analysis(self, save_path: str = "seasonal_analysis_nc.png") -> None:
        if self.meteo_data.empty:
            print("错误: 无数据可用于季节分析")
            return

        print("\n开始绘制季节分析图...")

        def get_season(month: int) -> str:
            if month in (12, 1, 2):
                return "Winter"
            if month in (3, 4, 5):
                return "Spring"
            if month in (6, 7, 8):
                return "Summer"
            return "Autumn"

        self.meteo_data["season"] = self.meteo_data["month"].astype(int).apply(get_season)
        season_order = ["Spring", "Summer", "Autumn", "Winter"]

        fig, axes = plt.subplots(2, 2, figsize=(20, 18))

        if "t2m_mean" in self.meteo_data.columns:
            try:
                plot_data = self.meteo_data[["season", "t2m_mean"]].dropna()
                if not plot_data.empty:
                    sns.boxplot(
                        data=plot_data,
                        x="season",
                        y="t2m_mean",
                        order=season_order,
                        palette="Set2",
                        ax=axes[0, 0],
                    )
                    axes[0, 0].set_xlabel("Season", fontsize=12, fontweight="bold")
                    axes[0, 0].set_ylabel("Temperature (°C)", fontsize=12, fontweight="bold")
                    axes[0, 0].set_title(
                        "2m Temperature Seasonal Distribution (NC)",
                        fontsize=14,
                        fontweight="bold",
                        pad=20,
                    )
                    axes[0, 0].grid(True, alpha=0.3, linestyle="--", axis="y")
            except Exception as err:
                print(f"温度季节分布绘图失败: {err}")
                axes[0, 0].text(0.5, 0.5, "Insufficient data", transform=axes[0, 0].transAxes)

        target_for_second = (
            "wind_speed_10m"
            if "wind_speed_10m" in self.meteo_data.columns
            else "sp_mean"
            if "sp_mean" in self.meteo_data.columns
            else None
        )
        if target_for_second:
            try:
                plot_data = self.meteo_data[["season", target_for_second]].dropna()
                if not plot_data.empty:
                    palette = "Set3" if target_for_second == "wind_speed_10m" else "coolwarm"
                    ylabel = "Wind Speed (m/s)" if target_for_second == "wind_speed_10m" else "Surface Pressure (Pa)"
                    title = (
                        "10m Wind Speed Seasonal Distribution (NC)"
                        if target_for_second == "wind_speed_10m"
                        else "Surface Pressure Seasonal Distribution (NC)"
                    )
                    sns.boxplot(
                        data=plot_data,
                        x="season",
                        y=target_for_second,
                        order=season_order,
                        palette=palette,
                        ax=axes[0, 1],
                    )
                    axes[0, 1].set_xlabel("Season", fontsize=12, fontweight="bold")
                    axes[0, 1].set_ylabel(ylabel, fontsize=12, fontweight="bold")
                    axes[0, 1].set_title(title, fontsize=14, fontweight="bold", pad=20)
                    axes[0, 1].grid(True, alpha=0.3, linestyle="--", axis="y")
            except Exception as err:
                print(f"第二子图绘制失败: {err}")
                axes[0, 1].text(0.5, 0.5, "Insufficient data", transform=axes[0, 1].transAxes)

        if "tp_mean" in self.meteo_data.columns:
            try:
                plot_data = self.meteo_data[["season", "tp_mean"]].dropna()
                if not plot_data.empty:
                    sns.boxplot(
                        data=plot_data,
                        x="season",
                        y="tp_mean",
                        order=season_order,
                        palette="Blues",
                        ax=axes[1, 0],
                    )
                    axes[1, 0].set_xlabel("Season", fontsize=12, fontweight="bold")
                    axes[1, 0].set_ylabel("Precipitation (mm)", fontsize=12, fontweight="bold")
                    axes[1, 0].set_title(
                        "Precipitation Seasonal Distribution (NC)",
                        fontsize=14,
                        fontweight="bold",
                        pad=20,
                    )
                    axes[1, 0].grid(True, alpha=0.3, linestyle="--", axis="y")
            except Exception as err:
                print(f"降水季节分布绘图失败: {err}")
                axes[1, 0].text(0.5, 0.5, "Insufficient data", transform=axes[1, 0].transAxes)

        if "blh_mean" in self.meteo_data.columns:
            try:
                plot_data = self.meteo_data[["season", "blh_mean"]].dropna()
                if not plot_data.empty:
                    sns.boxplot(
                        data=plot_data,
                        x="season",
                        y="blh_mean",
                        order=season_order,
                        palette="Greens",
                        ax=axes[1, 1],
                    )
                    axes[1, 1].set_xlabel("Season", fontsize=12, fontweight="bold")
                    axes[1, 1].set_ylabel("Boundary Layer Height (m)", fontsize=12, fontweight="bold")
                    axes[1, 1].set_title(
                        "Boundary Layer Height Seasonal Distribution (NC)",
                        fontsize=14,
                        fontweight="bold",
                        pad=20,
                    )
                    axes[1, 1].grid(True, alpha=0.3, linestyle="--", axis="y")
            except Exception as err:
                print(f"边界层高度季节分布绘图失败: {err}")
                axes[1, 1].text(0.5, 0.5, "Insufficient data", transform=axes[1, 1].transAxes)

        plt.tight_layout(h_pad=4.0, w_pad=2.0)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"季节分析图已保存至: {save_path}")
        plt.show()

    def save_data_summary(self, save_path: str = "meteorological_summary_nc.csv") -> None:
        if self.meteo_data.empty:
            print("错误: 无数据可导出")
            return

        print("\n开始保存气象统计汇总...")

        self.meteo_data.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"完整统计已保存至: {save_path}")

        numeric_cols = [
            col
            for col in self.meteo_data.select_dtypes(include=[np.number]).columns
            if col not in {"year", "month"}
        ]
        summary = self.meteo_data[numeric_cols].describe().T
        summary["parameter"] = summary.index
        summary = summary[
            ["parameter", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        ]
        summary_path = save_path.replace(".csv", "_statistics.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"统计摘要已保存至: {summary_path}")

    # -------------------------------------------------------------------------
    # 主流程
    # -------------------------------------------------------------------------
    def run_analysis(self) -> None:
        print("=" * 60)
        print("Beijing Meteorological Data Analysis (NC)")
        print("=" * 60)

        self.load_all_data()
        if self.meteo_data.empty:
            print("错误: 数据加载失败，请检查数据路径与文件格式")
            return

        self.plot_temperature_timeseries()
        self.plot_wind_speed_analysis()
        self.plot_precipitation_analysis()
        self.plot_seasonal_analysis()
        self.save_data_summary()

        print("\n分析完成!")


def main() -> None:
    data_dir = r"E:\DATA Science\ERA5-Beijing-NC"
    if not os.path.exists(data_dir):
        fallback_paths = [
            r"C:\Users\IU\Desktop\something\ERA5-Beijing-NC",
            r"C:\Users\IU\Desktop\ERA5-Data",
            r"C:\Users\IU\Desktop\something\Short-essay-Beijing\Statistics\Meteorological\NC",
        ]
        for path in fallback_paths:
            if os.path.exists(path):
                data_dir = path
                print(f"使用备选路径: {data_dir}")
                break
        else:
            print("错误: 未找到 ERA5 NetCDF 数据目录，请调整 main() 中的 data_dir")
            return

    print(f"数据目录: {data_dir}")
    print("如需更改数据路径，请修改 main() 函数中的 data_dir 变量\n")

    analyzer = MeteorologicalAnalyzerNC(data_dir=data_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()


