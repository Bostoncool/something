from __future__ import annotations

from pathlib import Path
import json
import hashlib
from typing import Any, Callable

import pandas as pd


AUTO_RUN_DIR = Path(__file__).resolve().parent
CACHE_DIR = AUTO_RUN_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _hash_config(config: dict[str, Any] | None) -> str:
    """根据配置生成稳定的短哈希，用于区分不同版本缓存。"""
    if not config:
        return "default"
    dumped = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(dumped.encode("utf-8")).hexdigest()[:8]


def load_df_cached(
    cache_name: str,
    loader_func: Callable[..., pd.DataFrame],
    loader_kwargs: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    通用 DataFrame 缓存入口。

    - cache_name: 逻辑名称，如 'bth_panel'、'yrd_panel' 等。
    - loader_func: 实际读取数据的函数，返回 DataFrame。
    - loader_kwargs: 传入 loader_func 的关键字参数。
    - config: 影响数据内容的关键配置（区域、源文件路径、城市列表等），用于生成版本哈希。
    """
    loader_kwargs = loader_kwargs or {}
    version = _hash_config(config)
    cache_path = CACHE_DIR / f"{cache_name}__{version}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    df = loader_func(**loader_kwargs)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.to_parquet(cache_path, index=False)
    return df

