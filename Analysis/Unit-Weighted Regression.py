#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
加权求和模型(Weighted Sum Model, WSM)实现
这是一种多准则决策方法，用于评估多个备选方案
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class WeightedSumModel:
    """
    加权求和模型(WSM)的实现
    
    参数:
    - criteria_weights: 各评价准则的权重
    - is_benefit: 布尔值列表，表示各准则是否为效益型(True)或成本型(False)
    """
    
    def __init__(self, criteria_weights, is_benefit):
        # 确保权重和为1
        self.weights = np.array(criteria_weights) / np.sum(criteria_weights)
        self.is_benefit = np.array(is_benefit, dtype=bool)
        
        if len(self.weights) != len(self.is_benefit):
            raise ValueError("权重和准则类型的数量必须相同")
    
    def normalize_data(self, data):
        """
        对数据进行归一化处理
        
        参数:
        - data: 原始数据矩阵，行为备选方案，列为评价准则
        
        返回:
        - 归一化后的数据矩阵
        """
        data = np.array(data, dtype=float)
        normalized = np.zeros_like(data)
        
        for j in range(data.shape[1]):
            if self.is_benefit[j]:
                # 效益型指标: 值越大越好，使用最大最小值归一化
                min_val = np.min(data[:, j])
                max_val = np.max(data[:, j])
                if max_val > min_val:
                    normalized[:, j] = (data[:, j] - min_val) / (max_val - min_val)
                else:
                    normalized[:, j] = 1.0  # 如果所有值相同
            else:
                # 成本型指标: 值越小越好，使用最大最小值归一化的倒数
                min_val = np.min(data[:, j])
                max_val = np.max(data[:, j])
                if max_val > min_val:
                    normalized[:, j] = (max_val - data[:, j]) / (max_val - min_val)
                else:
                    normalized[:, j] = 1.0  # 如果所有值相同
                    
        return normalized
    
    def evaluate(self, data):
        """
        评估各备选方案
        
        参数:
        - data: 原始数据矩阵，行为备选方案，列为评价准则
        
        返回:
        - 各备选方案的得分和排名
        """
        # 归一化数据
        normalized_data = self.normalize_data(data)
        
        # 计算加权得分
        scores = np.dot(normalized_data, self.weights)
        
        # 计算排名（得分越高，排名越靠前）
        ranks = np.argsort(-scores) + 1
        
        return scores, ranks

def visualize_results(alternatives, scores, ranks, criteria, weights):
    """
    可视化WSM的结果
    
    参数:
    - alternatives: 备选方案名称列表
    - scores: 各备选方案的得分
    - ranks: 各备选方案的排名
    - criteria: 评价准则名称列表
    - weights: 各评价准则的权重
    """
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        '备选方案': alternatives,
        '得分': scores,
        '排名': ranks
    }).sort_values('排名')
    
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制得分条形图
    sns.barplot(x='得分', y='备选方案', data=results_df, ax=ax1, palette='viridis')
    ax1.set_title('各备选方案得分')
    ax1.set_xlabel('得分')
    ax1.set_ylabel('备选方案')
    
    # 在条形图上添加得分和排名标签
    for i, (score, rank) in enumerate(zip(results_df['得分'], results_df['排名'])):
        ax1.text(score + 0.01, i, f'得分: {score:.4f} (排名: {rank})', va='center')
    
    # 绘制权重饼图
    ax2.pie(weights, labels=criteria, autopct='%1.1f%%', startangle=90, shadow=True)
    ax2.set_title('评价准则权重分布')
    ax2.axis('equal')  # 确保饼图是圆的
    
    plt.tight_layout()
    plt.show()

def main():
    """
    WSM示例：选择最佳手机方案
    """
    # 定义评价准则
    criteria = ['价格', '电池寿命', '相机质量', '处理器性能', '屏幕质量']
    
    # 定义准则权重
    weights = [0.25, 0.20, 0.20, 0.15, 0.20]
    
    # 定义准则类型（True为效益型，False为成本型）
    is_benefit = [False, True, True, True, True]  # 价格是成本型（越低越好），其他是效益型（越高越好）
    
    # 定义备选方案
    alternatives = ['手机A', '手机B', '手机C', '手机D', '手机E']
    
    # 各备选方案在各准则上的原始评分数据
    # 行：备选方案，列：评价准则
    data = np.array([
        [3500, 8, 7, 8, 9],    # 手机A
        [2800, 6, 9, 7, 8],    # 手机B
        [4200, 9, 6, 9, 7],    # 手机C
        [3000, 7, 8, 6, 9],    # 手机D
        [3800, 8, 9, 8, 6]     # 手机E
    ])
    
    # 创建WSM模型
    wsm = WeightedSumModel(weights, is_benefit)
    
    # 评估备选方案
    scores, ranks = wsm.evaluate(data)
    
    # 打印结果
    print("加权求和模型(WSM)结果:")
    print("-" * 50)
    print(f"{'备选方案':<10} {'得分':<10} {'排名':<10}")
    print("-" * 50)
    for i, alternative in enumerate(alternatives):
        print(f"{alternative:<10} {scores[i]:<10.4f} {ranks[i]:<10}")
    
    # 可视化结果
    visualize_results(alternatives, scores, ranks, criteria, weights)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        '备选方案': alternatives,
        '得分': scores,
        '排名': ranks
    }).sort_values('排名')
    
    print("\n排序结果:")
    print(results_df)
    
    # 创建原始数据DataFrame
    data_df = pd.DataFrame(data, index=alternatives, columns=criteria)
    print("\n原始数据:")
    print(data_df)

if __name__ == "__main__":
    main()
