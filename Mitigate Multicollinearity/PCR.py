#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主成分回归(Principal Component Regression, PCR)示例
PCR是一种结合了主成分分析(PCA)和线性回归的技术，用于处理多重共线性问题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子，确保结果可重现
np.random.seed(42)

# 1. 生成具有多重共线性的数据
def generate_multicollinear_data(n_samples=200, n_features=10, noise=0.5, random_state=42):
    """生成具有多重共线性的数据"""
    # 生成基础数据
    X, y = make_regression(n_samples=n_samples, 
                          n_features=n_features, 
                          n_informative=5,  # 只有5个特征是真正有信息量的
                          noise=noise, 
                          random_state=random_state)
    
    # 引入多重共线性：将一些特征设为其他特征的线性组合
    X[:, 5] = 0.7 * X[:, 0] + 0.3 * X[:, 1]
    X[:, 6] = 0.5 * X[:, 1] + 0.5 * X[:, 2]
    X[:, 7] = 0.8 * X[:, 0] + 0.2 * X[:, 3]
    X[:, 8] = 0.6 * X[:, 2] + 0.4 * X[:, 4]
    X[:, 9] = 0.9 * X[:, 0] + 0.1 * X[:, 4]
    
    # 转换为DataFrame以便更好地展示
    feature_names = [f'特征{i+1}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='目标变量')
    
    return X_df, y_series

# 2. 实现PCR类
class PrincipalComponentRegression:
    """主成分回归实现类"""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.regressor = LinearRegression()
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('pca', self.pca),
            ('regressor', self.regressor)
        ])
        
    def fit(self, X, y):
        """训练PCR模型"""
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        """使用PCR模型进行预测"""
        return self.pipeline.predict(X)
    
    def score(self, X, y):
        """评估PCR模型"""
        return self.pipeline.score(X, y)
    
    def get_pca_explained_variance(self):
        """获取PCA解释的方差比例"""
        return self.pca.explained_variance_ratio_

# 3. 主函数
def main():
    # 生成数据
    print("正在生成具有多重共线性的数据...")
    X, y = generate_multicollinear_data(n_samples=200, n_features=10)
    
    # 查看数据的相关性矩阵，检查多重共线性
    print("\n计算特征间的相关性矩阵...")
    correlation_matrix = X.corr()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    
    # 可视化相关性矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('特征相关性矩阵')
    plt.tight_layout()
    plt.show()
    
    # 使用不同数量的主成分训练PCR模型
    max_components = min(X.shape[0], X.shape[1])
    n_components_range = range(1, max_components + 1)
    
    # 存储不同主成分数量的模型性能
    train_scores = []
    test_scores = []
    explained_variances = []
    
    for n_components in n_components_range:
        pcr = PrincipalComponentRegression(n_components=n_components)
        pcr.fit(X_train, y_train)
        
        # 计算训练集和测试集的R²分数
        train_score = pcr.score(X_train, y_train)
        test_score = pcr.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        
        # 获取累积解释方差
        explained_variance = np.sum(pcr.get_pca_explained_variance())
        explained_variances.append(explained_variance)
        
        print(f"主成分数量: {n_components}, 训练集R²: {train_score:.4f}, 测试集R²: {test_score:.4f}, "
              f"累积解释方差: {explained_variance:.4f}")
    
    # 可视化不同主成分数量的模型性能
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(n_components_range, train_scores, 'o-', label='训练集')
    plt.plot(n_components_range, test_scores, 's-', label='测试集')
    plt.xlabel('主成分数量')
    plt.ylabel('R²分数')
    plt.title('不同主成分数量的模型性能')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(n_components_range, explained_variances, 'o-')
    plt.xlabel('主成分数量')
    plt.ylabel('累积解释方差比例')
    plt.title('主成分累积解释方差')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 选择最佳主成分数量
    best_n_components = np.argmax(test_scores) + 1
    print(f"\n最佳主成分数量: {best_n_components}")
    
    # 使用最佳主成分数量训练最终模型
    final_pcr = PrincipalComponentRegression(n_components=best_n_components)
    final_pcr.fit(X_train, y_train)
    
    # 在测试集上评估最终模型
    y_pred = final_pcr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n最终模型评估:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 可视化预测结果
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('PCR模型预测结果')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 比较PCR与普通线性回归
    print("\n比较PCR与普通线性回归:")
    
    # 普通线性回归
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)
    
    print(f"普通线性回归 - MSE: {lr_mse:.4f}, R²: {lr_r2:.4f}")
    print(f"PCR (n_components={best_n_components}) - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # 交叉验证比较
    print("\n使用5折交叉验证比较模型:")
    
    # PCR交叉验证
    pcr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=best_n_components)),
        ('regressor', LinearRegression())
    ])
    pcr_cv_scores = cross_val_score(pcr_pipeline, X, y, cv=5, scoring='r2')
    
    # 普通线性回归交叉验证
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    lr_cv_scores = cross_val_score(lr_pipeline, X, y, cv=5, scoring='r2')
    
    print(f"PCR交叉验证R²: {pcr_cv_scores.mean():.4f} ± {pcr_cv_scores.std():.4f}")
    print(f"线性回归交叉验证R²: {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")

if __name__ == "__main__":
    main()
