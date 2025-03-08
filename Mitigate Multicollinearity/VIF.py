#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
方差膨胀因子(VIF)计算与多重共线性缓解示例
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.set(font='SimHei')  # 设置seaborn默认字体

def generate_multicollinear_data(n_samples=1000, n_features=5, noise=0.1, random_state=42):
    """
    生成具有多重共线性的数据
    
    参数:
    n_samples: 样本数量
    n_features: 特征数量
    noise: 噪声水平
    random_state: 随机种子
    
    返回:
    X_df: 特征数据框
    y: 目标变量
    """
    # 生成基础数据
    X_base, y = make_regression(n_samples=n_samples, 
                               n_features=n_features-1,  # 减1是因为我们将添加一个共线性特征
                               n_informative=n_features-1, 
                               noise=noise,
                               random_state=random_state)
    
    # 将X转换为DataFrame以便于操作
    feature_names = [f'X{i+1}' for i in range(n_features-1)]
    X_df = pd.DataFrame(X_base, columns=feature_names)
    
    # 添加一个与X1高度相关的特征
    X_df['X5'] = X_df['X1'] * 0.8 + np.random.normal(0, 0.1, n_samples)
    
    return X_df, y

def calculate_vif(X):
    """
    计算方差膨胀因子
    
    参数:
    X: 特征数据框
    
    返回:
    vif_df: 包含VIF值的数据框
    """
    # 添加常数项
    X_with_const = add_constant(X)
    
    # 计算VIF
    vif_data = pd.DataFrame()
    vif_data["特征"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                       for i in range(X_with_const.shape[1])]
    
    # 删除常数项的VIF
    vif_data = vif_data.iloc[1:]
    
    return vif_data

def plot_correlation_matrix(X):
    """
    绘制相关性矩阵热图
    
    参数:
    X: 特征数据框
    """
    plt.figure(figsize=(10, 8))
    corr_matrix = X.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                mask=mask, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5)
    
    plt.title("特征相关性矩阵", fontsize=15)
    plt.tight_layout()
    plt.show()

def plot_vif(vif_df):
    """
    绘制VIF条形图
    
    参数:
    vif_df: 包含VIF值的数据框
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x="特征", y="VIF", data=vif_df)
    plt.axhline(y=5, color='r', linestyle='-', alpha=0.3, label="VIF=5 (警戒线)")
    plt.axhline(y=10, color='r', linestyle='--', alpha=0.7, label="VIF=10 (严重线)")
    
    plt.title("各特征的方差膨胀因子(VIF)", fontsize=15)
    plt.ylabel("VIF值")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def mitigate_multicollinearity(X, y, vif_threshold=10):
    """
    缓解多重共线性
    
    参数:
    X: 特征数据框
    y: 目标变量
    vif_threshold: VIF阈值，超过此值的特征将被移除
    
    返回:
    X_reduced: 移除高VIF特征后的数据框
    removed_features: 被移除的特征列表
    """
    X_working = X.copy()
    removed_features = []
    max_vif = float('inf')
    
    print("开始缓解多重共线性...")
    
    while max_vif > vif_threshold:
        vif_df = calculate_vif(X_working)
        max_vif = vif_df['VIF'].max()
        
        if max_vif > vif_threshold:
            max_vif_feature = vif_df.loc[vif_df['VIF'].idxmax(), '特征']
            print(f"移除特征 {max_vif_feature} (VIF = {max_vif:.2f})")
            
            removed_features.append(max_vif_feature)
            X_working = X_working.drop(max_vif_feature, axis=1)
    
    print(f"多重共线性缓解完成。保留了 {X_working.shape[1]} 个特征，移除了 {len(removed_features)} 个特征。")
    return X_working, removed_features

def compare_models(X_original, X_reduced, y):
    """
    比较原始模型和缓解多重共线性后的模型
    
    参数:
    X_original: 原始特征数据框
    X_reduced: 缓解多重共线性后的特征数据框
    y: 目标变量
    """
    # 原始模型
    model_original = LinearRegression()
    model_original.fit(X_original, y)
    y_pred_original = model_original.predict(X_original)
    r2_original = model_original.score(X_original, y)
    
    # 缓解多重共线性后的模型
    model_reduced = LinearRegression()
    model_reduced.fit(X_reduced, y)
    y_pred_reduced = model_reduced.predict(X_reduced)
    r2_reduced = model_reduced.score(X_reduced, y)
    
    print("\n模型比较:")
    print(f"原始模型 (特征数: {X_original.shape[1]}):")
    print(f"  R² = {r2_original:.4f}")
    print(f"  系数: {model_original.coef_}")
    
    print(f"\n缓解多重共线性后的模型 (特征数: {X_reduced.shape[1]}):")
    print(f"  R² = {r2_reduced:.4f}")
    print(f"  系数: {model_reduced.coef_}")
    
    # 绘制系数比较
    coef_original = pd.Series(model_original.coef_, index=X_original.columns)
    coef_reduced = pd.Series(model_reduced.coef_, index=X_reduced.columns)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    coef_original.plot(kind='bar')
    plt.title("原始模型系数")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    coef_reduced.plot(kind='bar')
    plt.title("缓解多重共线性后的模型系数")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    print("生成具有多重共线性的数据...")
    X, y = generate_multicollinear_data(n_samples=1000, n_features=5)
    
    print("\n数据概览:")
    print(X.head())
    
    print("\n计算特征之间的相关性...")
    plot_correlation_matrix(X)
    
    print("\n计算方差膨胀因子(VIF)...")
    vif_df = calculate_vif(X)
    print(vif_df)
    
    # 绘制VIF条形图
    plot_vif(vif_df)
    
    # 缓解多重共线性
    X_reduced, removed_features = mitigate_multicollinearity(X, y, vif_threshold=5)
    
    # 计算缓解后的VIF
    if X_reduced.shape[1] > 1:  # 确保至少有两个特征才能计算VIF
        print("\n缓解多重共线性后的VIF:")
        vif_reduced = calculate_vif(X_reduced)
        print(vif_reduced)
        plot_vif(vif_reduced)
    
    # 比较模型
    compare_models(X, X_reduced, y)
    
    print("\n多重共线性分析完成!")

if __name__ == "__main__":
    main()
