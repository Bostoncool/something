import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class WeightedSumModel:
    """
    加权总和模型(Weighted Sum Model, WSM)实现
    
    WSM是一种多准则决策方法，用于评估多个备选方案。
    每个备选方案在多个准则上有不同的表现，每个准则有不同的权重。
    """
    
    def __init__(self, weights=None):
        """
        初始化WSM模型
        
        参数:
            weights: 各准则的权重向量，如果为None则假设等权重
        """
        self.weights = weights
        self.normalized_data = None
        self.scores = None
        self.rankings = None
    
    def normalize_data(self, data, benefit_criteria):
        """
        对数据进行归一化处理
        
        参数:
            data: 原始数据矩阵，行为备选方案，列为准则
            benefit_criteria: 布尔列表，表示每个准则是否为效益型(越大越好)
                             True表示效益型，False表示成本型(越小越好)
        
        返回:
            归一化后的数据矩阵
        """
        normalized = np.zeros_like(data, dtype=float)
        
        for j in range(data.shape[1]):
            if benefit_criteria[j]:  # 效益型指标
                # 最大值归一化
                normalized[:, j] = data[:, j] / np.max(data[:, j]) if np.max(data[:, j]) != 0 else 0
            else:  # 成本型指标
                # 最小值归一化（倒数变换）
                min_val = np.min(data[:, j])
                if min_val != 0:
                    normalized[:, j] = min_val / data[:, j]
                else:
                    # 避免除零错误
                    normalized[:, j] = 1 / (data[:, j] + 1e-10)
        
        self.normalized_data = normalized
        return normalized
    
    def calculate_scores(self, data, benefit_criteria):
        """
        计算每个备选方案的加权总分
        
        参数:
            data: 原始数据矩阵
            benefit_criteria: 布尔列表，表示每个准则是否为效益型
        
        返回:
            每个备选方案的加权总分
        """
        # 归一化数据
        normalized_data = self.normalize_data(data, benefit_criteria)
        
        # 如果未指定权重，则假设等权重
        if self.weights is None:
            self.weights = np.ones(data.shape[1]) / data.shape[1]
        
        # 计算加权总分
        self.scores = np.dot(normalized_data, self.weights)
        
        # 计算排名（从高到低）
        self.rankings = np.argsort(-self.scores)
        
        return self.scores
    
    def get_rankings(self):
        """获取备选方案的排名（从最优到最差）"""
        if self.rankings is None:
            raise ValueError("请先调用calculate_scores方法计算分数")
        return self.rankings
    
    def visualize_results(self, alternative_names=None, criteria_names=None):
        """
        可视化WSM结果
        
        参数:
            alternative_names: 备选方案的名称列表
            criteria_names: 准则的名称列表
        """
        if self.scores is None or self.normalized_data is None:
            raise ValueError("请先调用calculate_scores方法计算分数")
        
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 创建一个图形对象
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 准备数据
        n_alternatives = len(self.scores)
        if alternative_names is None:
            alternative_names = [f"方案{i+1}" for i in range(n_alternatives)]
        
        if criteria_names is None:
            criteria_names = [f"准则{i+1}" for i in range(self.normalized_data.shape[1])]
        
        # 绘制总分条形图
        sorted_indices = self.rankings
        sorted_scores = self.scores[sorted_indices]
        sorted_names = [alternative_names[i] for i in sorted_indices]
        
        bars = ax1.barh(sorted_names, sorted_scores, color='skyblue')
        ax1.set_title('备选方案总分排名')
        ax1.set_xlabel('加权总分')
        ax1.set_ylabel('备选方案')
        
        # 在条形上添加分数标签
        for i, bar in enumerate(bars):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{sorted_scores[i]:.4f}', va='center')
        
        # 创建热图显示归一化后的数据
        sorted_normalized_data = self.normalized_data[sorted_indices]
        
        sns.heatmap(sorted_normalized_data, annot=True, cmap='YlGnBu', 
                   xticklabels=criteria_names, yticklabels=sorted_names, ax=ax2)
        ax2.set_title('归一化后的准则得分')
        
        plt.tight_layout()
        plt.show()


# 示例使用
if __name__ == "__main__":
    # 示例数据：5个备选方案，4个评价准则
    # 行：备选方案，列：评价准则
    data = np.array([
        [8, 7, 2, 1],  # 方案1
        [5, 3, 7, 5],  # 方案2
        [7, 5, 6, 4],  # 方案3
        [9, 9, 7, 3],  # 方案4
        [6, 5, 5, 8],  # 方案5
    ])
    
    # 准则权重
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    
    # 准则类型（True表示效益型，False表示成本型）
    benefit_criteria = [True, True, True, False]
    
    # 备选方案和准则的名称
    alternative_names = ["方案A", "方案B", "方案C", "方案D", "方案E"]
    criteria_names = ["性能", "质量", "服务", "成本"]
    
    # 创建WSM模型
    wsm = WeightedSumModel(weights)
    
    # 计算得分
    scores = wsm.calculate_scores(data, benefit_criteria)
    
    # 获取排名
    rankings = wsm.get_rankings()
    
    # 打印结果
    print("加权总分:")
    for i, score in enumerate(scores):
        print(f"{alternative_names[i]}: {score:.4f}")
    
    print("\n排名（从最优到最差）:")
    for rank, idx in enumerate(rankings):
        print(f"第{rank+1}名: {alternative_names[idx]} (得分: {scores[idx]:.4f})")
    
    # 可视化结果
    wsm.visualize_results(alternative_names, criteria_names)
    
    # 创建一个DataFrame来展示详细结果
    results_df = pd.DataFrame({
        '备选方案': alternative_names,
        '加权总分': scores,
        '排名': [list(rankings).index(i) + 1 for i in range(len(scores))]
    })
    
    print("\n详细结果:")
    print(results_df.sort_values('排名'))
