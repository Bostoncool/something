#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ANOVA-同步成分分析(ASCA)实现
ASCA结合了方差分析(ANOVA)和主成分分析(PCA)的优点，
适用于多因素实验设计中的多变量数据分析。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import itertools

# 设置中文字体
try:
    # 尝试设置中文字体，如果失败则使用默认字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")

class ASCA:
    """ANOVA-同步成分分析(ASCA)类"""
    
    def __init__(self):
        """初始化ASCA对象"""
        self.effect_matrices = {}
        self.pca_models = {}
        self.variance_explained = {}
        self.scores = {}
        self.loadings = {}
        self.anova_results = None
        self.design_info = None
        
    def fit(self, X, design, formula=None):
        """
        拟合ASCA模型
        
        参数:
        X : DataFrame, 响应变量数据矩阵 (样本 × 变量)
        design : DataFrame, 实验设计矩阵 (样本 × 因素)
        formula : str, 可选，ANOVA模型公式
        
        返回:
        self : 返回ASCA对象
        """
        # 保存原始数据
        self.X = X
        self.design = design
        self.design_info = {
            'factors': list(design.columns),
            'factor_levels': {col: design[col].unique() for col in design.columns}
        }
        
        # 数据中心化
        self.X_centered = X - X.mean()
        
        # 如果没有提供公式，则自动构建完整的ANOVA模型公式
        if formula is None:
            factors = list(design.columns)
            main_effects = factors.copy()
            
            # 添加所有可能的交互效应
            interactions = []
            for i in range(2, len(factors) + 1):
                for combo in itertools.combinations(factors, i):
                    interactions.append(':'.join(combo))
            
            if interactions:
                formula = 'X ~ ' + ' + '.join(main_effects) + ' + ' + ' + '.join(interactions)
            else:
                formula = 'X ~ ' + ' + '.join(main_effects)
        
        self.formula = formula
        
        # 对每个响应变量执行ANOVA分解
        effect_matrices = {}
        anova_results = {}
        
        # 合并设计矩阵和响应变量
        for col in X.columns:
            data = design.copy()
            data[col] = X[col]
            
            # 使用statsmodels执行ANOVA
            model = ols(formula.replace('X', col), data=data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            anova_results[col] = anova_table
            
            # 提取每个效应的贡献
            for effect in anova_table.index:
                if effect != 'Residual':
                    if effect not in effect_matrices:
                        effect_matrices[effect] = pd.DataFrame(0, index=X.index, columns=X.columns)
                    
                    # 计算效应矩阵
                    effect_matrix = self._calculate_effect_matrix(model, effect, col)
                    effect_matrices[effect][col] = effect_matrix
        
        # 添加残差矩阵
        residual_matrix = self.X_centered.copy()
        for effect, matrix in effect_matrices.items():
            residual_matrix -= matrix
        
        effect_matrices['Residual'] = residual_matrix
        self.effect_matrices = effect_matrices
        self.anova_results = anova_results
        
        # 对每个效应矩阵执行PCA
        for effect, matrix in effect_matrices.items():
            pca = PCA()
            scores = pca.fit_transform(matrix)
            
            self.pca_models[effect] = pca
            self.scores[effect] = scores
            self.loadings[effect] = pca.components_
            self.variance_explained[effect] = pca.explained_variance_ratio_
        
        return self
    
    def _calculate_effect_matrix(self, model, effect, response_var):
        """计算特定效应的效应矩阵"""
        # 获取模型参数
        params = model.params
        
        # 提取与当前效应相关的参数
        effect_params = [p for p in params.index if p.startswith(effect)]
        
        # 计算效应矩阵
        effect_matrix = np.zeros(len(model.model.exog))
        
        for param in effect_params:
            param_idx = list(params.index).index(param)
            effect_matrix += params[param] * model.model.exog[:, param_idx]
        
        return effect_matrix
    
    def plot_scores(self, effect, pc1=0, pc2=1, color_by=None, ax=None):
        """
        绘制指定效应的得分散点图
        
        参数:
        effect : str, 要绘制的效应名称
        pc1 : int, 第一主成分索引
        pc2 : int, 第二主成分索引
        color_by : str, 可选，用于着色的设计因素
        ax : matplotlib.axes, 可选，用于绘图的轴对象
        
        返回:
        ax : matplotlib.axes, 绘图轴对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        scores = self.scores[effect]
        var_explained1 = self.variance_explained[effect][pc1] * 100
        var_explained2 = self.variance_explained[effect][pc2] * 100
        
        if color_by is not None and color_by in self.design.columns:
            groups = self.design[color_by].unique()
            for group in groups:
                mask = self.design[color_by] == group
                ax.scatter(scores[mask, pc1], scores[mask, pc2], label=f'{color_by}={group}')
            ax.legend()
        else:
            ax.scatter(scores[:, pc1], scores[:, pc2])
        
        ax.set_xlabel(f'PC{pc1+1} ({var_explained1:.1f}%)')
        ax.set_ylabel(f'PC{pc2+1} ({var_explained2:.1f}%)')
        ax.set_title(f'{effect} 效应的PCA得分图')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        return ax
    
    def plot_loadings(self, effect, pc=0, ax=None):
        """
        绘制指定效应的载荷条形图
        
        参数:
        effect : str, 要绘制的效应名称
        pc : int, 主成分索引
        ax : matplotlib.axes, 可选，用于绘图的轴对象
        
        返回:
        ax : matplotlib.axes, 绘图轴对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        loadings = self.loadings[effect][pc]
        var_explained = self.variance_explained[effect][pc] * 100
        
        # 按绝对值大小排序
        sorted_idx = np.argsort(np.abs(loadings))[::-1]
        sorted_loadings = loadings[sorted_idx]
        sorted_labels = self.X.columns[sorted_idx]
        
        # 只显示前15个变量
        n_display = min(15, len(sorted_labels))
        
        colors = ['b' if x < 0 else 'r' for x in sorted_loadings[:n_display]]
        ax.bar(range(n_display), sorted_loadings[:n_display], color=colors)
        ax.set_xticks(range(n_display))
        ax.set_xticklabels(sorted_labels[:n_display], rotation=45, ha='right')
        ax.set_xlabel('变量')
        ax.set_ylabel('载荷系数')
        ax.set_title(f'{effect} 效应的PC{pc+1} ({var_explained:.1f}%)载荷图')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        return ax
    
    def plot_variance_explained(self, ax=None):
        """
        绘制各效应解释的方差比例
        
        参数:
        ax : matplotlib.axes, 可选，用于绘图的轴对象
        
        返回:
        ax : matplotlib.axes, 绘图轴对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # 计算每个效应矩阵的总方差
        effect_var = {}
        total_var = np.sum(np.var(self.X_centered, axis=0))
        
        for effect, matrix in self.effect_matrices.items():
            effect_var[effect] = np.sum(np.var(matrix, axis=0)) / total_var * 100
        
        # 绘制条形图
        effects = list(effect_var.keys())
        variances = list(effect_var.values())
        
        # 按方差大小排序
        sorted_idx = np.argsort(variances)[::-1]
        sorted_effects = [effects[i] for i in sorted_idx]
        sorted_variances = [variances[i] for i in sorted_idx]
        
        ax.bar(sorted_effects, sorted_variances)
        ax.set_xlabel('效应')
        ax.set_ylabel('解释的方差比例 (%)')
        ax.set_title('各效应解释的方差比例')
        ax.set_ylim(0, 100)
        
        # 旋转x轴标签以防止重叠
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        return ax
    
    def summary(self):
        """打印ASCA模型摘要"""
        print("ANOVA-同步成分分析(ASCA)摘要")
        print("=" * 50)
        print(f"数据维度: {self.X.shape[0]} 样本 × {self.X.shape[1]} 变量")
        print(f"实验设计: {len(self.design_info['factors'])} 因素")
        
        for factor, levels in self.design_info['factor_levels'].items():
            print(f"  - {factor}: {len(levels)} 水平 ({', '.join(map(str, levels))})")
        
        print("\nANOVA模型公式:")
        print(f"  {self.formula}")
        
        print("\n效应方差分解:")
        total_var = np.sum(np.var(self.X_centered, axis=0))
        
        for effect, matrix in self.effect_matrices.items():
            effect_var = np.sum(np.var(matrix, axis=0))
            var_ratio = effect_var / total_var * 100
            print(f"  - {effect}: {var_ratio:.2f}% 的总方差")
        
        print("\nPCA模型:")
        for effect, pca in self.pca_models.items():
            var_exp = self.variance_explained[effect]
            print(f"  - {effect}:")
            print(f"    * PC1: {var_exp[0]*100:.2f}%")
            print(f"    * PC2: {var_exp[1]*100:.2f}%")
            print(f"    * 累计(PC1+PC2): {(var_exp[0]+var_exp[1])*100:.2f}%")


# 示例: 生成模拟数据并应用ASCA
if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 模拟实验设计: 2个因素 (处理和时间)
    # 处理: 对照组和处理组
    # 时间: 3个时间点
    n_replicates = 5  # 每组重复次数
    n_variables = 20  # 变量数量
    
    # 创建实验设计矩阵
    treatments = np.repeat(['对照组', '处理组'], n_replicates * 3)
    time_points = np.tile(np.repeat(['t1', 't2', 't3'], n_replicates), 2)
    
    design = pd.DataFrame({
        '处理': treatments,
        '时间': time_points
    })
    
    # 生成响应变量数据
    # 基础噪声
    X = np.random.normal(0, 1, (len(design), n_variables))
    
    # 添加处理效应 (影响前10个变量)
    treatment_effect = np.zeros((len(design), n_variables))
    treatment_effect[design['处理'] == '处理组', :10] = 2
    
    # 添加时间效应 (影响后10个变量)
    time_effect = np.zeros((len(design), n_variables))
    for i, t in enumerate(['t1', 't2', 't3']):
        time_effect[design['时间'] == t, 10:] = i
    
    # 添加交互效应 (影响中间10个变量)
    interaction_effect = np.zeros((len(design), n_variables))
    interaction_effect[(design['处理'] == '处理组') & (design['时间'] == 't3'), 5:15] = 3
    
    # 合并所有效应
    X = X + treatment_effect + time_effect + interaction_effect
    
    # 转换为DataFrame
    X = pd.DataFrame(X, columns=[f'变量{i+1}' for i in range(n_variables)])
    
    # 应用ASCA
    asca = ASCA()
    asca.fit(X, design)
    
    # 打印摘要
    asca.summary()
    
    # 创建可视化
    plt.figure(figsize=(15, 10))
    
    # 1. 方差分解图
    plt.subplot(2, 2, 1)
    asca.plot_variance_explained()
    
    # 2. 处理效应的得分图
    plt.subplot(2, 2, 2)
    asca.plot_scores('处理', color_by='处理')
    
    # 3. 时间效应的得分图
    plt.subplot(2, 2, 3)
    asca.plot_scores('时间', color_by='时间')
    
    # 4. 交互效应的得分图
    plt.subplot(2, 2, 4)
    asca.plot_scores('处理:时间')
    
    plt.tight_layout()
    plt.savefig('ASCA_scores.png', dpi=300)
    
    # 绘制载荷图
    plt.figure(figsize=(15, 10))
    
    # 1. 处理效应的载荷图
    plt.subplot(2, 2, 1)
    asca.plot_loadings('处理')
    
    # 2. 时间效应的载荷图
    plt.subplot(2, 2, 2)
    asca.plot_loadings('时间')
    
    # 3. 交互效应的载荷图
    plt.subplot(2, 2, 3)
    asca.plot_loadings('处理:时间')
    
    # 4. 残差的载荷图
    plt.subplot(2, 2, 4)
    asca.plot_loadings('Residual')
    
    plt.tight_layout()
    plt.savefig('ASCA_loadings.png', dpi=300)
    
    print("\n分析完成! 图表已保存为 'ASCA_scores.png' 和 'ASCA_loadings.png'")
