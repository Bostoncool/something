#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
聚类分析示例代码
包含K-means聚类和层次聚类两种方法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings('ignore')

# 设置随机种子，保证结果可重现
np.random.seed(42)

def generate_sample_data(n_samples=300):
    """
    生成示例数据
    
    参数:
    n_samples: 样本数量
    
    返回:
    X: 特征数据
    y: 真实的类别标签
    """
    # 生成有4个簇的数据
    X, y = make_blobs(n_samples=n_samples, centers=4, cluster_std=0.60, random_state=42)
    
    # 转换为DataFrame以便于处理
    df = pd.DataFrame(X, columns=['特征1', '特征2'])
    df['真实类别'] = y
    
    return df

def preprocess_data(df):
    """
    数据预处理
    
    参数:
    df: 输入的DataFrame
    
    返回:
    X_scaled: 标准化后的特征数据
    """
    # 提取特征
    X = df[['特征1', '特征2']].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

def plot_original_data(df):
    """
    绘制原始数据散点图
    
    参数:
    df: 包含特征和真实类别的DataFrame
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='特征1', y='特征2', hue='真实类别', 
                    palette='viridis', data=df, s=50)
    plt.title('原始数据分布')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend(title='真实类别')
    plt.show()

def find_optimal_k(X_scaled, max_k=10):
    """
    使用肘部法则和轮廓系数找到最优的K值
    
    参数:
    X_scaled: 标准化后的特征数据
    max_k: 最大的K值
    
    返回:
    optimal_k: 最优的K值
    """
    inertia = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # 绘制肘部法则图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, 'o-', markersize=8)
    plt.title('肘部法则')
    plt.xlabel('聚类数量 (k)')
    plt.ylabel('惯性 (Inertia)')
    plt.grid(True)
    
    # 绘制轮廓系数图
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'o-', markersize=8)
    plt.title('轮廓系数')
    plt.xlabel('聚类数量 (k)')
    plt.ylabel('轮廓系数')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 找到轮廓系数最大的K值
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"根据轮廓系数，最优的聚类数量为: {optimal_k}")
    
    return optimal_k

def perform_kmeans(X_scaled, n_clusters, df):
    """
    执行K-means聚类
    
    参数:
    X_scaled: 标准化后的特征数据
    n_clusters: 聚类数量
    df: 原始DataFrame
    
    返回:
    df_kmeans: 包含聚类结果的DataFrame
    """
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 将聚类结果添加到DataFrame
    df_kmeans = df.copy()
    df_kmeans['KMeans聚类'] = clusters
    
    # 计算轮廓系数
    silhouette_avg = silhouette_score(X_scaled, clusters)
    print(f"K-means聚类的轮廓系数: {silhouette_avg:.4f}")
    
    return df_kmeans, kmeans

def plot_kmeans_clusters(df_kmeans, kmeans, X_scaled):
    """
    绘制K-means聚类结果
    
    参数:
    df_kmeans: 包含聚类结果的DataFrame
    kmeans: 训练好的KMeans模型
    X_scaled: 标准化后的特征数据
    """
    # 绘制聚类结果
    plt.figure(figsize=(12, 5))
    
    # 绘制聚类结果散点图
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='特征1', y='特征2', hue='KMeans聚类', 
                    palette='viridis', data=df_kmeans, s=50)
    plt.title('K-means聚类结果')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend(title='聚类')
    
    # 绘制聚类中心和决策边界
    plt.subplot(1, 2, 2)
    
    # 创建网格
    h = 0.02
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 预测网格点的聚类
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.viridis)
    
    # 绘制数据点
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, 
                cmap=plt.cm.viridis, s=50, alpha=0.8)
    
    # 绘制聚类中心
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='X')
    
    plt.title('K-means聚类边界和中心')
    plt.xlabel('特征1 (标准化)')
    plt.ylabel('特征2 (标准化)')
    
    plt.tight_layout()
    plt.show()

def perform_hierarchical_clustering(X_scaled, n_clusters, df):
    """
    执行层次聚类
    
    参数:
    X_scaled: 标准化后的特征数据
    n_clusters: 聚类数量
    df: 原始DataFrame
    
    返回:
    df_hierarchical: 包含聚类结果的DataFrame
    """
    # 执行层次聚类
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    clusters = hierarchical.fit_predict(X_scaled)
    
    # 将聚类结果添加到DataFrame
    df_hierarchical = df.copy()
    df_hierarchical['层次聚类'] = clusters
    
    # 计算轮廓系数
    silhouette_avg = silhouette_score(X_scaled, clusters)
    print(f"层次聚类的轮廓系数: {silhouette_avg:.4f}")
    
    return df_hierarchical

def plot_dendrogram(X_scaled):
    """
    绘制层次聚类的树状图
    
    参数:
    X_scaled: 标准化后的特征数据
    """
    # 计算层次聚类的链接矩阵
    linked = linkage(X_scaled, 'ward')
    
    # 绘制树状图
    plt.figure(figsize=(12, 6))
    dendrogram(linked, orientation='top', distance_sort='descending', 
               show_leaf_counts=True)
    plt.title('层次聚类树状图')
    plt.xlabel('样本索引')
    plt.ylabel('距离')
    plt.axhline(y=6, c='k', linestyle='--', alpha=0.5)
    plt.text(X_scaled.shape[0]/2, 6.5, '建议的聚类数量截断线', ha='center')
    plt.show()

def plot_hierarchical_clusters(df_hierarchical):
    """
    绘制层次聚类结果
    
    参数:
    df_hierarchical: 包含层次聚类结果的DataFrame
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='特征1', y='特征2', hue='层次聚类', 
                    palette='viridis', data=df_hierarchical, s=50)
    plt.title('层次聚类结果')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend(title='聚类')
    plt.show()

def compare_clustering_methods(df_kmeans, df_hierarchical):
    """
    比较不同聚类方法的结果
    
    参数:
    df_kmeans: 包含K-means聚类结果的DataFrame
    df_hierarchical: 包含层次聚类结果的DataFrame
    """
    # 创建一个包含两种聚类结果的DataFrame
    df_compare = df_kmeans.copy()
    df_compare['层次聚类'] = df_hierarchical['层次聚类']
    
    # 绘制比较图
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='特征1', y='特征2', hue='KMeans聚类', 
                    palette='viridis', data=df_compare, s=50)
    plt.title('K-means聚类结果')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend(title='聚类')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='特征1', y='特征2', hue='层次聚类', 
                    palette='viridis', data=df_compare, s=50)
    plt.title('层次聚类结果')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend(title='聚类')
    
    plt.tight_layout()
    plt.show()
    
    # 创建一个交叉表来比较两种聚类方法的一致性
    cross_tab = pd.crosstab(df_compare['KMeans聚类'], df_compare['层次聚类'], 
                           rownames=['K-means'], colnames=['层次聚类'])
    
    print("K-means聚类与层次聚类的交叉表:")
    print(cross_tab)
    
    # 计算两种聚类方法的一致性
    total = cross_tab.sum().sum()
    agreement = sum(cross_tab.values.max(axis=1))
    agreement_rate = agreement / total
    
    print(f"两种聚类方法的一致性: {agreement_rate:.4f}")

def main():
    """
    主函数，执行聚类分析的完整流程
    """
    print("="*50)
    print("聚类分析示例")
    print("="*50)
    
    # 1. 生成示例数据
    print("\n1. 生成示例数据")
    df = generate_sample_data(n_samples=300)
    print(f"数据形状: {df.shape}")
    print(df.head())
    
    # 2. 绘制原始数据
    print("\n2. 绘制原始数据")
    plot_original_data(df)
    
    # 3. 数据预处理
    print("\n3. 数据预处理")
    X_scaled = preprocess_data(df)
    print("数据标准化完成")
    
    # 4. 寻找最优的K值
    print("\n4. 寻找最优的K值")
    optimal_k = find_optimal_k(X_scaled)
    
    # 5. 执行K-means聚类
    print("\n5. 执行K-means聚类")
    df_kmeans, kmeans = perform_kmeans(X_scaled, optimal_k, df)
    
    # 6. 绘制K-means聚类结果
    print("\n6. 绘制K-means聚类结果")
    plot_kmeans_clusters(df_kmeans, kmeans, X_scaled)
    
    # 7. 绘制层次聚类的树状图
    print("\n7. 绘制层次聚类的树状图")
    plot_dendrogram(X_scaled)
    
    # 8. 执行层次聚类
    print("\n8. 执行层次聚类")
    df_hierarchical = perform_hierarchical_clustering(X_scaled, optimal_k, df)
    
    # 9. 绘制层次聚类结果
    print("\n9. 绘制层次聚类结果")
    plot_hierarchical_clusters(df_hierarchical)
    
    # 10. 比较不同聚类方法
    print("\n10. 比较不同聚类方法")
    compare_clustering_methods(df_kmeans, df_hierarchical)
    
    print("\n聚类分析完成!")

if __name__ == "__main__":
    main()
