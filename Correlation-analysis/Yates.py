#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Yates分析模块 - 用于农业和生物学实验的方差分析
主要用于分析完全随机区组设计(RCBD)实验数据
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
sns.set(font='SimHei')  # 设置seaborn默认字体

class YatesAnalysis:
    """
    Yates分析类，用于执行完全随机区组设计(RCBD)的方差分析
    """
    
    def __init__(self, data=None, treatments=None, blocks=None, response=None):
        """
        初始化Yates分析对象
        
        参数:
            data (DataFrame): 包含实验数据的DataFrame
            treatments (str): 处理因素的列名
            blocks (str): 区组因素的列名
            response (str): 响应变量的列名
        """
        self.data = data
        self.treatments = treatments
        self.blocks = blocks
        self.response = response
        self.model = None
        self.anova_results = None
    
    def generate_sample_data(self, n_treatments=4, n_blocks=3, seed=42):
        """
        生成样本数据用于演示
        
        参数:
            n_treatments (int): 处理水平数量
            n_blocks (int): 区组数量
            seed (int): 随机种子
        
        返回:
            DataFrame: 生成的样本数据
        """
        np.random.seed(seed)
        
        # 创建处理效应
        treatment_effects = np.random.normal(0, 2, n_treatments)
        
        # 创建区组效应
        block_effects = np.random.normal(0, 1, n_blocks)
        
        # 创建数据框
        treatments = []
        blocks = []
        responses = []
        
        # 基础响应值
        base_response = 10
        
        # 生成数据
        for i in range(n_treatments):
            for j in range(n_blocks):
                treatments.append(f"处理{i+1}")
                blocks.append(f"区组{j+1}")
                
                # 响应值 = 基础值 + 处理效应 + 区组效应 + 随机误差
                response = base_response + treatment_effects[i] + block_effects[j] + np.random.normal(0, 1)
                responses.append(response)
        
        # 创建DataFrame
        self.data = pd.DataFrame({
            '处理': treatments,
            '区组': blocks,
            '产量': responses
        })
        
        self.treatments = '处理'
        self.blocks = '区组'
        self.response = '产量'
        
        return self.data
    
    def fit(self):
        """
        使用statsmodels执行方差分析
        
        返回:
            DataFrame: ANOVA表结果
        """
        if self.data is None:
            raise ValueError("请先提供数据或使用generate_sample_data生成样本数据")
        
        # 创建模型公式
        formula = f"{self.response} ~ C({self.treatments}) + C({self.blocks})"
        
        # 拟合模型
        self.model = ols(formula, data=self.data).fit()
        
        # 执行方差分析
        self.anova_results = anova_lm(self.model, typ=2)
        
        return self.anova_results
    
    def summary(self):
        """
        打印分析结果摘要
        """
        if self.model is None:
            self.fit()
        
        print("Yates分析结果摘要:")
        print("=" * 50)
        print("\n模型摘要:")
        print(self.model.summary())
        
        print("\n方差分析表:")
        print(self.anova_results)
        
        # 计算处理均值
        treatment_means = self.data.groupby(self.treatments)[self.response].mean()
        print("\n各处理均值:")
        print(treatment_means)
        
        # 计算区组均值
        block_means = self.data.groupby(self.blocks)[self.response].mean()
        print("\n各区组均值:")
        print(block_means)
        
        # 执行多重比较（如果处理因素显著）
        p_value = self.anova_results.loc[f"C({self.treatments})", "PR(>F)"]
        if p_value < 0.05:
            print("\n处理因素显著 (p < 0.05)，执行多重比较:")
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            tukey = pairwise_tukeyhsd(endog=self.data[self.response],
                                     groups=self.data[self.treatments],
                                     alpha=0.05)
            print(tukey)
        else:
            print(f"\n处理因素不显著 (p = {p_value:.4f})，不执行多重比较")
    
    def plot_boxplot(self):
        """
        绘制处理水平的箱线图
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.treatments, y=self.response, data=self.data)
        plt.title("各处理水平的响应变量箱线图")
        plt.xlabel("处理")
        plt.ylabel(self.response)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def plot_interaction(self):
        """
        绘制处理和区组的交互图
        """
        plt.figure(figsize=(12, 7))
        
        # 计算每个处理-区组组合的平均值
        interaction_data = self.data.groupby([self.treatments, self.blocks])[self.response].mean().reset_index()
        
        # 绘制交互图
        sns.lineplot(x=self.treatments, y=self.response, hue=self.blocks, 
                    data=interaction_data, marker='o', markersize=10)
        
        plt.title("处理和区组的交互效应图")
        plt.xlabel("处理")
        plt.ylabel(self.response)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title="区组")
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self):
        """
        绘制残差分析图
        """
        if self.model is None:
            self.fit()
        
        # 获取残差
        residuals = self.model.resid
        fitted = self.model.fittedvalues
        
        # 创建残差图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 残差 vs 拟合值
        sns.scatterplot(x=fitted, y=residuals, ax=axes[0, 0])
        axes[0, 0].axhline(y=0, color='r', linestyle='-')
        axes[0, 0].set_title("残差 vs 拟合值")
        axes[0, 0].set_xlabel("拟合值")
        axes[0, 0].set_ylabel("残差")
        axes[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 残差QQ图
        sm.qqplot(residuals, line='45', ax=axes[0, 1])
        axes[0, 1].set_title("残差QQ图")
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # 残差直方图
        sns.histplot(residuals, kde=True, ax=axes[1, 0])
        axes[1, 0].set_title("残差直方图")
        axes[1, 0].set_xlabel("残差")
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 残差 vs 处理
        treatment_order = self.data[self.treatments].unique()
        residual_by_treatment = pd.DataFrame({
            self.treatments: self.data[self.treatments],
            'residuals': residuals
        })
        
        sns.boxplot(x=self.treatments, y='residuals', data=residual_by_treatment, ax=axes[1, 1])
        axes[1, 1].set_title(f"残差 vs {self.treatments}")
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()


# 示例用法
if __name__ == "__main__":
    # 创建Yates分析对象
    yates = YatesAnalysis()
    
    # 生成样本数据
    data = yates.generate_sample_data(n_treatments=4, n_blocks=3)
    print("生成的样本数据:")
    print(data.head())
    
    # 执行方差分析
    anova_results = yates.fit()
    print("\n方差分析结果:")
    print(anova_results)
    
    # 打印摘要
    yates.summary()
    
    # 绘制图形
    yates.plot_boxplot()
    yates.plot_interaction()
    yates.plot_residuals()
    
    print("\n分析完成!")
