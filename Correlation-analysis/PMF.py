#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PMF (Positive Matrix Factorization) 分析实现
这是一个用于环境数据源解析的受体模型
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class PMFAnalysis:
    """
    PMF分析类，用于执行正矩阵因子分解分析
    """
    
    def __init__(self, data_file=None, data=None, uncertainty=None):
        """
        初始化PMF分析对象
        
        参数:
            data_file: 数据文件路径，CSV格式
            data: 直接提供的数据矩阵，pandas DataFrame格式
            uncertainty: 不确定性矩阵，与data同样大小的DataFrame
        """
        if data_file is not None:
            self.data = pd.read_csv(data_file, index_col=0)
        elif data is not None:
            self.data = data
        else:
            raise ValueError("必须提供数据文件路径或数据矩阵")
            
        # 如果没有提供不确定性矩阵，则使用数据的10%作为不确定性估计
        if uncertainty is None:
            self.uncertainty = self.data * 0.1
        else:
            self.uncertainty = uncertainty
            
        self.n_components = None
        self.model = None
        self.W = None  # 源谱图矩阵
        self.H = None  # 源贡献矩阵
        self.explained_variance_ = None
        
    def preprocess_data(self):
        """
        数据预处理：处理缺失值，标准化等
        """
        # 填充缺失值
        self.data = self.data.fillna(self.data.mean())
        
        # 确保所有值为非负
        if (self.data < 0).any().any():
            print("警告: 数据中存在负值，将被替换为0")
            self.data[self.data < 0] = 0
            
        # 标准化数据
        self.scaler = StandardScaler()
        self.scaled_data = pd.DataFrame(
            self.scaler.fit_transform(self.data),
            index=self.data.index,
            columns=self.data.columns
        )
        
        return self.scaled_data
    
    def run_pmf(self, n_components, **kwargs):
        """
        运行PMF模型
        
        参数:
            n_components: 要提取的因子数量
            **kwargs: 传递给NMF的其他参数
        """
        self.n_components = n_components
        
        # 使用sklearn的NMF作为PMF的实现
        self.model = NMF(
            n_components=n_components,
            init='random',
            solver='cd',
            max_iter=1000,
            random_state=42,
            **kwargs
        )
        
        # 拟合模型
        self.W = self.model.fit_transform(self.data)
        self.H = self.model.components_
        
        # 计算重构误差
        self.reconstruction_err_ = self.model.reconstruction_err_
        
        # 计算解释方差
        reconstructed = np.dot(self.W, self.H)
        residuals = self.data.values - reconstructed
        ss_total = np.sum((self.data.values - np.mean(self.data.values))**2)
        ss_residual = np.sum(residuals**2)
        self.explained_variance_ = 1 - (ss_residual / ss_total)
        
        print(f"重构误差: {self.reconstruction_err_:.4f}")
        print(f"解释方差: {self.explained_variance_:.4f}")
        
        return self.W, self.H
    
    def get_factor_profiles(self):
        """
        获取因子谱图（源谱图）
        """
        if self.H is None:
            raise ValueError("请先运行PMF模型")
            
        factor_profiles = pd.DataFrame(
            self.H,
            columns=self.data.columns,
            index=[f"因子{i+1}" for i in range(self.n_components)]
        )
        
        return factor_profiles
    
    def get_factor_contributions(self):
        """
        获取因子贡献（源贡献）
        """
        if self.W is None:
            raise ValueError("请先运行PMF模型")
            
        factor_contributions = pd.DataFrame(
            self.W,
            index=self.data.index,
            columns=[f"因子{i+1}" for i in range(self.n_components)]
        )
        
        return factor_contributions
    
    def plot_factor_profiles(self, normalize=True, figsize=(12, 8)):
        """
        绘制因子谱图
        
        参数:
            normalize: 是否归一化因子谱图
            figsize: 图形大小
        """
        if self.H is None:
            raise ValueError("请先运行PMF模型")
            
        factor_profiles = self.get_factor_profiles()
        
        if normalize:
            # 归一化每个因子谱图
            factor_profiles = factor_profiles.div(factor_profiles.sum(axis=1), axis=0)
        
        plt.figure(figsize=figsize)
        
        for i in range(self.n_components):
            plt.subplot(self.n_components, 1, i+1)
            factor_profiles.iloc[i].plot(kind='bar')
            plt.title(f"因子 {i+1} 谱图")
            plt.ylabel("贡献")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
        
        plt.tight_layout()
        plt.savefig('factor_profiles.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_factor_contributions(self, figsize=(12, 8)):
        """
        绘制因子贡献时间序列
        
        参数:
            figsize: 图形大小
        """
        if self.W is None:
            raise ValueError("请先运行PMF模型")
            
        factor_contributions = self.get_factor_contributions()
        
        plt.figure(figsize=figsize)
        
        for i in range(self.n_components):
            plt.subplot(self.n_components, 1, i+1)
            factor_contributions[f"因子{i+1}"].plot()
            plt.title(f"因子 {i+1} 贡献")
            plt.ylabel("贡献")
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('factor_contributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_explained_variance(self, max_components=10, figsize=(10, 6)):
        """
        绘制不同因子数量的解释方差
        
        参数:
            max_components: 最大测试的因子数量
            figsize: 图形大小
        """
        explained_var = []
        reconstruction_err = []
        
        for n in range(1, max_components + 1):
            model = NMF(
                n_components=n,
                init='random',
                solver='cd',
                max_iter=1000,
                random_state=42
            )
            
            W = model.fit_transform(self.data)
            H = model.components_
            
            # 计算重构误差
            reconstruction_err.append(model.reconstruction_err_)
            
            # 计算解释方差
            reconstructed = np.dot(W, H)
            residuals = self.data.values - reconstructed
            ss_total = np.sum((self.data.values - np.mean(self.data.values))**2)
            ss_residual = np.sum(residuals**2)
            explained_var.append(1 - (ss_residual / ss_total))
        
        plt.figure(figsize=figsize)
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, max_components + 1), explained_var, 'o-', linewidth=2)
        plt.xlabel('因子数量')
        plt.ylabel('解释方差')
        plt.title('不同因子数量的解释方差')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, max_components + 1), reconstruction_err, 'o-', linewidth=2)
        plt.xlabel('因子数量')
        plt.ylabel('重构误差')
        plt.title('不同因子数量的重构误差')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('explained_variance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return explained_var, reconstruction_err
    
    def plot_factor_correlation(self, figsize=(10, 8)):
        """
        绘制因子之间的相关性热图
        
        参数:
            figsize: 图形大小
        """
        if self.W is None:
            raise ValueError("请先运行PMF模型")
            
        factor_contributions = self.get_factor_contributions()
        corr = factor_contributions.corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('因子贡献相关性')
        plt.tight_layout()
        plt.savefig('factor_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_results(self, output_dir='.'):
        """
        保存分析结果
        
        参数:
            output_dir: 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if self.W is not None and self.H is not None:
            # 保存因子谱图
            factor_profiles = self.get_factor_profiles()
            factor_profiles.to_csv(os.path.join(output_dir, 'factor_profiles.csv'))
            
            # 保存因子贡献
            factor_contributions = self.get_factor_contributions()
            factor_contributions.to_csv(os.path.join(output_dir, 'factor_contributions.csv'))
            
            # 保存重构数据
            reconstructed = np.dot(self.W, self.H)
            reconstructed_df = pd.DataFrame(
                reconstructed,
                index=self.data.index,
                columns=self.data.columns
            )
            reconstructed_df.to_csv(os.path.join(output_dir, 'reconstructed_data.csv'))
            
            # 保存模型参数
            with open(os.path.join(output_dir, 'model_info.txt'), 'w') as f:
                f.write(f"因子数量: {self.n_components}\n")
                f.write(f"重构误差: {self.reconstruction_err_:.4f}\n")
                f.write(f"解释方差: {self.explained_variance_:.4f}\n")
                
            print(f"结果已保存到 {output_dir} 目录")
        else:
            print("没有可保存的结果，请先运行PMF模型")


# 示例用法
if __name__ == "__main__":
    # 创建一个示例数据集
    # 在实际应用中，您应该加载自己的数据
    np.random.seed(42)
    
    # 创建3个源谱图
    n_species = 10
    n_samples = 100
    
    # 创建源谱图矩阵 (3 x n_species)
    true_profiles = np.zeros((3, n_species))
    true_profiles[0, :3] = np.random.rand(3)  # 第一个源主要影响前3个物种
    true_profiles[1, 3:7] = np.random.rand(4)  # 第二个源主要影响中间4个物种
    true_profiles[2, 7:] = np.random.rand(3)  # 第三个源主要影响最后3个物种
    
    # 归一化源谱图
    true_profiles = true_profiles / true_profiles.sum(axis=1)[:, np.newaxis]
    
    # 创建时间变化的源贡献
    t = np.linspace(0, 4*np.pi, n_samples)
    true_contributions = np.zeros((n_samples, 3))
    true_contributions[:, 0] = 5 + 3 * np.sin(t)  # 第一个源的贡献
    true_contributions[:, 1] = 3 + 2 * np.cos(t/2)  # 第二个源的贡献
    true_contributions[:, 2] = 4 + np.sin(t/3) + np.cos(t/4)  # 第三个源的贡献
    
    # 确保所有贡献都是正的
    true_contributions = np.abs(true_contributions)
    
    # 生成观测数据矩阵
    X = np.dot(true_contributions, true_profiles)
    
    # 添加一些噪声
    noise = np.random.normal(0, 0.1, X.shape)
    X_noisy = X + noise
    X_noisy[X_noisy < 0] = 0  # 确保所有值为非负
    
    # 创建DataFrame
    species = [f"物种{i+1}" for i in range(n_species)]
    samples = [f"样本{i+1}" for i in range(n_samples)]
    
    data_df = pd.DataFrame(X_noisy, index=samples, columns=species)
    
    # 创建PMF分析对象
    pmf = PMFAnalysis(data=data_df)
    
    # 数据预处理
    pmf.preprocess_data()
    
    # 绘制不同因子数量的解释方差
    explained_var, reconstruction_err = pmf.plot_explained_variance(max_components=6)
    
    # 运行PMF模型，提取3个因子
    pmf.run_pmf(n_components=3)
    
    # 绘制因子谱图
    pmf.plot_factor_profiles()
    
    # 绘制因子贡献
    pmf.plot_factor_contributions()
    
    # 绘制因子相关性
    pmf.plot_factor_correlation()
    
    # 保存结果
    pmf.save_results()
    
    print("PMF分析完成！")
