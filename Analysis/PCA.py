#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主成分分析(PCA)示例代码
包含数据生成、PCA实现、可视化和结果解释
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data():
    """
    加载示例数据集(鸢尾花数据集)
    """
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # 创建DataFrame以便于数据操作
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]
    
    print("数据集形状:", X.shape)
    print("特征名称:", feature_names)
    print("目标类别:", target_names)
    
    return df, X, y, feature_names, target_names

def explore_data(df):
    """
    数据探索
    """
    print("\n数据集前5行:")
    print(df.head())
    
    print("\n数据集统计描述:")
    print(df.describe())
    
    # 绘制特征之间的散点图矩阵
    plt.figure(figsize=(12, 10))
    sns.pairplot(df, hue='species', height=2.5)
    plt.tight_layout()
    plt.savefig('pairplot.png')
    
    # 绘制特征相关性热图
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.drop('species', axis=1).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('特征相关性热图')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    
    return correlation_matrix

def perform_pca(X, feature_names):
    """
    执行PCA分析
    """
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 创建PCA对象并拟合数据
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # 计算解释方差比例
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print("\nPCA解释方差比例:")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"主成分 {i+1}: {ratio:.4f} ({cumulative_variance_ratio[i]:.4f} 累计)")
    
    # 创建包含主成分的DataFrame
    pca_df = pd.DataFrame(
        data=X_pca,
        columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
    )
    
    # 获取主成分的特征向量(加载矩阵)
    loadings = pca.components_
    loadings_df = pd.DataFrame(
        loadings.T,
        columns=[f'PC{i+1}' for i in range(loadings.shape[0])],
        index=feature_names
    )
    
    print("\n主成分加载矩阵(特征向量):")
    print(loadings_df)
    
    return pca, X_pca, pca_df, explained_variance_ratio, cumulative_variance_ratio, loadings_df

def visualize_pca_results(X_pca, y, target_names, explained_variance_ratio, cumulative_variance_ratio, loadings_df):
    """
    可视化PCA结果
    """
    # 绘制解释方差比例
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, label='单个方差比例')
    plt.step(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, where='mid', label='累计方差比例')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% 方差阈值')
    plt.xlabel('主成分')
    plt.ylabel('解释方差比例')
    plt.title('PCA解释方差比例')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('explained_variance.png')
    
    # 绘制前两个主成分的散点图
    plt.figure(figsize=(12, 10))
    colors = ['navy', 'turquoise', 'darkorange']
    for i, target_name in enumerate(target_names):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                   color=colors[i], alpha=0.8, lw=2, label=target_name)
    plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2%})')
    plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.2%})')
    plt.title('PCA: 前两个主成分')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('pca_scatter.png')
    
    # 绘制加载图(特征对主成分的贡献)
    plt.figure(figsize=(10, 8))
    for i, feature in enumerate(loadings_df.index):
        plt.arrow(0, 0, loadings_df.iloc[i, 0]*5, loadings_df.iloc[i, 1]*5, 
                 head_width=0.1, head_length=0.1, fc='r', ec='r')
        plt.text(loadings_df.iloc[i, 0]*5.2, loadings_df.iloc[i, 1]*5.2, feature)
    
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2%})')
    plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.2%})')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.title('PCA加载图: 特征对主成分的贡献')
    plt.tight_layout()
    plt.savefig('pca_loadings.png')
    
    # 绘制3D散点图(前三个主成分)
    if X_pca.shape[1] >= 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, target_name in enumerate(target_names):
            ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], X_pca[y == i, 2],
                      color=colors[i], alpha=0.8, label=target_name)
        
        ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.2%})')
        ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.2%})')
        ax.set_zlabel(f'PC3 ({explained_variance_ratio[2]:.2%})')
        ax.set_title('PCA: 前三个主成分')
        plt.legend()
        plt.tight_layout()
        plt.savefig('pca_3d.png')

def interpret_results(loadings_df, explained_variance_ratio):
    """
    解释PCA结果
    """
    print("\nPCA结果解释:")
    
    # 对每个主成分，找出贡献最大的特征
    for i in range(loadings_df.shape[1]):
        pc = f'PC{i+1}'
        var_explained = explained_variance_ratio[i]
        
        # 按绝对值排序，找出最重要的特征
        sorted_loadings = loadings_df[pc].abs().sort_values(ascending=False)
        top_features = sorted_loadings.index[:2]  # 取前两个最重要的特征
        
        print(f"\n{pc} (解释方差: {var_explained:.2%}):")
        print(f"  主要特征: {', '.join(top_features)}")
        
        # 显示正负贡献
        pos_features = loadings_df[loadings_df[pc] > 0][pc].sort_values(ascending=False).index[:2]
        neg_features = loadings_df[loadings_df[pc] < 0][pc].sort_values().index[:2]
        
        if len(pos_features) > 0:
            print(f"  正向贡献特征: {', '.join(pos_features)}")
        if len(neg_features) > 0:
            print(f"  负向贡献特征: {', '.join(neg_features)}")

def main():
    """
    主函数
    """
    print("开始PCA分析...")
    
    # 加载数据
    df, X, y, feature_names, target_names = load_data()
    
    # 数据探索
    correlation_matrix = explore_data(df)
    
    # 执行PCA
    pca, X_pca, pca_df, explained_variance_ratio, cumulative_variance_ratio, loadings_df = perform_pca(X, feature_names)
    
    # 可视化结果
    visualize_pca_results(X_pca, y, target_names, explained_variance_ratio, cumulative_variance_ratio, loadings_df)
    
    # 解释结果
    interpret_results(loadings_df, explained_variance_ratio)
    
    print("\nPCA分析完成！所有图表已保存。")

if __name__ == "__main__":
    main()
