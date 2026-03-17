"""
特征重要性分析：读取各模型/区域的 feature_importance.csv，汇总输出到四个 CSV 文件
- 全域融合模型
- BTH（京津冀）
- YRD（长三角）
- PRD（珠三角）
"""

import pandas as pd
from pathlib import Path

# 输出目录（与脚本同目录）
OUTPUT_DIR = Path(__file__).parent

# 数据路径配置
DATA_PATHS = {
    "全域融合模型": [
        r"H:\大论文Result\大论文图\机器学习结果\catboost_daily_pm25\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\xgboost_daily_pm25\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\lightgbm_daily_pm25\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\tabtransformer_daily_pm25\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\lightgbm_bth_seasonal_daily_pm25\feature_importance_winter.csv",
        r"H:\大论文Result\大论文图\机器学习结果\lightgbm_bth_seasonal_daily_pm25\feature_importance_non_winter.csv",
    ],
    "BTH": [
        r"H:\大论文Result\大论文图\机器学习结果\lightgbm_bth_daily_pm25\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\catboost_daily_pm25\bth\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\xgboost_daily_pm25\bth\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\tabtransformer_daily_pm25\bth\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\lightgbm_daily_pm25\bth\feature_importance.csv",
    ],
    "YRD": [
        r"H:\大论文Result\大论文图\机器学习结果\lightgbm_daily_pm25\yrd\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\xgboost_daily_pm25\yrd\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\catboost_daily_pm25\yrd\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\tabtransformer_daily_pm25\yrd\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\adaboost_daily_pm25\yrd\feature_importance.csv",
    ],
    "PRD": [
        r"H:\大论文Result\大论文图\机器学习结果\lightgbm_daily_pm25\prd\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\xgboost_daily_pm25\prd\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\catboost_daily_pm25\prd\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\rf_daily_pm25\prd\feature_importance.csv",
        r"H:\大论文Result\大论文图\机器学习结果\adaboost_daily_pm25\prd\feature_importance.csv",
    ],
}

# 模型名称映射（从路径提取简短名称）
MODEL_NAMES = {
    "全域融合模型": [
        "catboost",
        "xgboost",
        "lightgbm",
        "tabtransformer",
        "lightgbm_winter",
        "lightgbm_non_winter",
    ],
    "BTH": [
        "lightgbm_bth",
        "catboost",
        "xgboost",
        "tabtransformer",
        "lightgbm",
    ],
    "YRD": [
        "lightgbm",
        "xgboost_lightgbm",
        "catboost",
        "tabtransformer",
        "adaboost",
    ],
    "PRD": [
        "lightgbm",
        "xgboost",
        "catboost",
        "rf",
        "adaboost",
    ],
}


def load_feature_importance(path: str, model_name: str) -> pd.DataFrame:
    """读取单个 feature_importance.csv，返回 (feature, model_name, importance) 格式"""
    df = pd.read_csv(path, encoding="utf-8")
    # 兼容 feature/importance 或 feature/importance/cluster 等格式
    if "feature" not in df.columns or "importance" not in df.columns:
        raise ValueError(f"文件 {path} 缺少 feature 或 importance 列，实际列: {list(df.columns)}")
    out = df[["feature", "importance"]].copy()
    out["model"] = model_name
    return out


def merge_and_pivot(df_list: list[pd.DataFrame], model_names: list[str]) -> pd.DataFrame:
    """将多个模型的 feature importance 合并为宽表（feature 为行，各模型为列）"""
    # 合并所有数据
    combined = pd.concat(df_list, ignore_index=True)
    # 透视：feature 为索引，model 为列，importance 为值
    pivot = combined.pivot_table(
        index="feature", columns="model", values="importance", aggfunc="first"
    )
    # 确保列顺序与 model_names 一致
    pivot = pivot.reindex(columns=model_names)
    return pivot.reset_index()


def main():
    for region, paths in DATA_PATHS.items():
        model_names = MODEL_NAMES[region]
        dfs = []
        for path, name in zip(paths, model_names):
            try:
                df = load_feature_importance(path, name)
                dfs.append(df)
            except FileNotFoundError:
                print(f"[WARN] File not found, skip: {path}")
            except Exception as e:
                print(f"[错误] 读取失败 {path}: {e}")
                raise

        if not dfs:
            print(f"[WARN] {region} has no valid data, skip")
            continue

        merged = merge_and_pivot(dfs, model_names)
        out_path = OUTPUT_DIR / f"feature_importance_{region}.csv"
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Output: {out_path} ({len(merged)} features, {len(model_names)} models)")


if __name__ == "__main__":
    main()
