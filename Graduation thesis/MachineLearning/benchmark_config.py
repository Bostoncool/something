from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetTypeConfig:
    key: str
    priority: str = "aux"  # core / aux / unused
    paths: list[str] = field(default_factory=list)
    column_map: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ClusterInputConfig:
    key: str
    display_name: str
    daily_input: list[str] = field(default_factory=list)
    column_map: dict[str, str] = field(default_factory=dict)
    dataset_types: dict[str, DatasetTypeConfig] = field(default_factory=dict)


@dataclass
class ModelRunConfig:
    key: str
    script_name: str
    enabled: bool = True
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    clusters: list[ClusterInputConfig]
    models: list[ModelRunConfig]
    train_end_year: int = 2021
    valid_year: int = 2022
    test_year: int = 2023
    seed: int = 42
    include_era5_daily: bool = True
    device: str = ""


def _default_dataset_types_by_cluster() -> dict[str, dict[str, DatasetTypeConfig]]:
    # Priority table from user requirements.
    return {
        "BTH": {
            "pm25": DatasetTypeConfig(key="pm25", priority="core"),
            "era5": DatasetTypeConfig(key="era5", priority="core"),
            "industry_emission": DatasetTypeConfig(key="industry_emission", priority="aux"),
            "energy": DatasetTypeConfig(key="energy", priority="core"),
            "road_density": DatasetTypeConfig(key="road_density", priority="core"),
            "night_light": DatasetTypeConfig(key="night_light", priority="aux"),
            "map_vector": DatasetTypeConfig(key="map_vector", priority="aux"),
            "industrial_land": DatasetTypeConfig(key="industrial_land", priority="aux"),
            "land_use": DatasetTypeConfig(key="land_use", priority="unused", enabled=False),
            "gdp_population": DatasetTypeConfig(key="gdp_population", priority="aux"),
            "nev_stock": DatasetTypeConfig(key="nev_stock", priority="unused", enabled=False),
            "vegetation": DatasetTypeConfig(key="vegetation", priority="unused", enabled=False),
        },
        "YRD": {
            "pm25": DatasetTypeConfig(key="pm25", priority="core"),
            "era5": DatasetTypeConfig(key="era5", priority="core"),
            "industry_emission": DatasetTypeConfig(key="industry_emission", priority="core"),
            "energy": DatasetTypeConfig(key="energy", priority="aux"),
            "road_density": DatasetTypeConfig(key="road_density", priority="aux"),
            "night_light": DatasetTypeConfig(key="night_light", priority="aux"),
            "map_vector": DatasetTypeConfig(key="map_vector", priority="aux"),
            "industrial_land": DatasetTypeConfig(key="industrial_land", priority="aux"),
            "land_use": DatasetTypeConfig(key="land_use", priority="core"),
            "gdp_population": DatasetTypeConfig(key="gdp_population", priority="unused", enabled=False),
            "nev_stock": DatasetTypeConfig(key="nev_stock", priority="core"),
            "vegetation": DatasetTypeConfig(key="vegetation", priority="unused", enabled=False),
        },
        "PRD": {
            "pm25": DatasetTypeConfig(key="pm25", priority="core"),
            "era5": DatasetTypeConfig(key="era5", priority="core"),
            "industry_emission": DatasetTypeConfig(key="industry_emission", priority="core"),
            "energy": DatasetTypeConfig(key="energy", priority="aux"),
            "road_density": DatasetTypeConfig(key="road_density", priority="core"),
            "night_light": DatasetTypeConfig(key="night_light", priority="aux"),
            "map_vector": DatasetTypeConfig(key="map_vector", priority="aux"),
            "industrial_land": DatasetTypeConfig(key="industrial_land", priority="aux"),
            "land_use": DatasetTypeConfig(key="land_use", priority="aux"),
            "gdp_population": DatasetTypeConfig(key="gdp_population", priority="aux"),
            "nev_stock": DatasetTypeConfig(key="nev_stock", priority="unused", enabled=False),
            "vegetation": DatasetTypeConfig(key="vegetation", priority="unused", enabled=False),
        },
    }


def build_default_config() -> BenchmarkConfig:
    dataset_defaults = _default_dataset_types_by_cluster()
    return BenchmarkConfig(
        clusters=[
            ClusterInputConfig(key="YRD", display_name="长三角", dataset_types=dataset_defaults["YRD"]),
            ClusterInputConfig(key="PRD", display_name="珠三角", dataset_types=dataset_defaults["PRD"]),
            ClusterInputConfig(key="BTH", display_name="京津冀", dataset_types=dataset_defaults["BTH"]),
        ],
        models=[
            ModelRunConfig(key="xgboost", script_name="XGBOOST.py"),
            ModelRunConfig(key="lightgbm", script_name="LightGBM.py"),
            ModelRunConfig(
                key="cnn_lstm",
                script_name="CNN-LSTM.py",
                extra_args={"seq-len": 14, "epochs": 200, "batch-size": 512, "lr": 1e-3},
            ),
            ModelRunConfig(
                key="st_transformer",
                script_name="ST-Transformer.py",
                extra_args={
                    "seq-len": 14,
                    "epochs": 200,
                    "batch-size": 512,
                    "lr": 1e-3,
                    "d-model": 64,
                    "nhead": 4,
                },
            ),
        ],
    )


def _as_list_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _to_cluster_list(raw: Any, defaults: list[ClusterInputConfig]) -> list[ClusterInputConfig]:
    default_map = {item.key: item for item in defaults}
    if raw is None:
        return defaults

    output: list[ClusterInputConfig] = []
    for key, default_item in default_map.items():
        cluster_raw = raw.get(key, {}) if isinstance(raw, dict) else {}
        if not isinstance(cluster_raw, dict):
            cluster_raw = {}
        raw_column_map = cluster_raw.get("column_map", {})
        if not isinstance(raw_column_map, dict):
            raw_column_map = {}
        raw_dataset_types = cluster_raw.get("dataset_types", {})
        if not isinstance(raw_dataset_types, dict):
            raw_dataset_types = {}
        parsed_dataset_types: dict[str, DatasetTypeConfig] = {}
        for ds_key, ds_default in default_item.dataset_types.items():
            ds_raw = raw_dataset_types.get(ds_key, {})
            if not isinstance(ds_raw, dict):
                ds_raw = {}
            ds_column_map = ds_raw.get("column_map", {})
            if not isinstance(ds_column_map, dict):
                ds_column_map = {}
            priority = str(ds_raw.get("priority", ds_default.priority)).strip().lower() or ds_default.priority
            if priority not in {"core", "aux", "unused"}:
                priority = ds_default.priority
            parsed_dataset_types[ds_key] = DatasetTypeConfig(
                key=ds_key,
                priority=priority,
                paths=_as_list_str(ds_raw.get("paths", ds_default.paths)),
                column_map={str(k): str(v) for k, v in ds_column_map.items()},
                enabled=bool(ds_raw.get("enabled", ds_default.enabled)),
            )
        output.append(
            ClusterInputConfig(
                key=key,
                display_name=str(cluster_raw.get("display_name", default_item.display_name)),
                daily_input=_as_list_str(cluster_raw.get("daily_input", default_item.daily_input)),
                column_map={str(k): str(v) for k, v in raw_column_map.items()},
                dataset_types=parsed_dataset_types,
            )
        )
    return output


def _to_model_list(raw: Any, defaults: list[ModelRunConfig]) -> list[ModelRunConfig]:
    default_map = {item.key: item for item in defaults}
    if raw is None:
        return defaults

    output: list[ModelRunConfig] = []
    for key, default_item in default_map.items():
        model_raw = raw.get(key, {}) if isinstance(raw, dict) else {}
        if not isinstance(model_raw, dict):
            model_raw = {}
        raw_extra_args = model_raw.get("extra_args", default_item.extra_args)
        if not isinstance(raw_extra_args, dict):
            raw_extra_args = default_item.extra_args
        output.append(
            ModelRunConfig(
                key=key,
                script_name=str(model_raw.get("script_name", default_item.script_name)),
                enabled=bool(model_raw.get("enabled", default_item.enabled)),
                extra_args=dict(raw_extra_args),
            )
        )
    return output


def load_config_from_dict(raw: dict[str, Any]) -> BenchmarkConfig:
    defaults = build_default_config()
    clusters = _to_cluster_list(raw.get("clusters"), defaults.clusters)
    models = _to_model_list(raw.get("models"), defaults.models)
    return BenchmarkConfig(
        clusters=clusters,
        models=models,
        train_end_year=int(raw.get("train_end_year", defaults.train_end_year)),
        valid_year=int(raw.get("valid_year", defaults.valid_year)),
        test_year=int(raw.get("test_year", defaults.test_year)),
        seed=int(raw.get("seed", defaults.seed)),
        include_era5_daily=bool(raw.get("include_era5_daily", defaults.include_era5_daily)),
        device=str(raw.get("device", defaults.device)).strip(),
    )


def resolve_paths(path_list: list[str]) -> list[str]:
    return [str(Path(path).expanduser().resolve()) for path in path_list if str(path).strip()]
