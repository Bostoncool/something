#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多因素分析(Multiple Factor Analysis, MFA)示例代码
MFA是一种处理多组变量的多变量分析方法，可以看作是对多组变量同时进行的主成分分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.font_manager as fm

# 设置中文字体显示
try:
    # 尝试设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")

# 创建示例数据
# 假设我们有三组变量：感官评价、理化指标和消费者偏好
np.random.seed(42)  # 设置随机种子以确保结果可重复

# 样本数量
n_samples = 30

# 第一组变量：感官评价 (4个变量)
sensory_data = pd.DataFrame({
    '口感': np.random.normal(5, 1.5, n_samples),
    '香气': np.random.normal(6, 1.2, n_samples),
    '外观': np.random.normal(4, 1.8, n_samples),
    '风味': np.random.normal(5.5, 1.3, n_samples)
})

# 第二组变量：理化指标 (3个变量)
physical_data = pd.DataFrame({
    'pH值': np.random.normal(6.5, 0.5, n_samples),
    '酸度': np.random.normal(3.2, 0.8, n_samples),
    '糖度': np.random.normal(12, 2.5, n_samples)
})

# 第三组变量：消费者偏好 (3个变量)
consumer_data = pd.DataFrame({
    '购买意愿': np.random.normal(7, 1.5, n_samples),
    '推荐意愿': np.random.normal(6.5, 1.8, n_samples),
    '价格接受度': np.random.normal(5.5, 2.0, n_samples)
})

# 为数据集添加一些相关性
# 假设口感好的样品通常pH值较低，购买意愿较高
physical_data['pH值'] = physical_data['pH值'] - 0.3 * sensory_data['口感'] + np.random.normal(0, 0.3, n_samples)
consumer_data['购买意愿'] = consumer_data['购买意愿'] + 0.4 * sensory_data['口感'] + np.random.normal(0, 0.5, n_samples)

# 合并所有数据
all_data = pd.concat([sensory_data, physical_data, consumer_data], axis=1)

# 添加样品标识
all_data['样品'] = [f'样品{i+1}' for i in range(n_samples)]
all_data.set_index('样品', inplace=True)

print("数据集概览:")
print(all_data.head())
print("\n数据集描述性统计:")
print(all_data.describe())

# 数据标准化
scaler = StandardScaler()
scaled_data = pd.DataFrame(
    scaler.fit_transform(all_data),
    index=all_data.index,
    columns=all_data.columns
)

# 定义变量组
groups = {
    '感官评价': sensory_data.columns,
    '理化指标': physical_data.columns,
    '消费者偏好': consumer_data.columns
}

# 实现MFA的主要步骤
def perform_mfa(data, groups):
    """
    执行多因素分析(MFA)
    
    参数:
    data: 包含所有变量的DataFrame
    groups: 字典，键为组名，值为该组包含的变量列表
    
    返回:
    global_pca: 全局PCA结果
    partial_factor_scores: 各组的部分因子得分
    group_weights: 各组的权重
    """
    # 第1步：对每组变量进行单独的PCA
    group_pcas = {}
    group_weights = {}
    
    for group_name, variables in groups.items():
        group_data = data[variables]
        pca = PCA()
        pca.fit(group_data)
        group_pcas[group_name] = pca
        # 使用第一个特征值的倒数作为权重
        group_weights[group_name] = 1 / pca.explained_variance_[0]
    
    # 第2步：构建加权数据矩阵
    weighted_data = pd.DataFrame()
    for group_name, variables in groups.items():
        weight = group_weights[group_name]
        for var in variables:
            weighted_data[f"{var}"] = data[var] * np.sqrt(weight)
    
    # 第3步：对加权数据进行全局PCA
    global_pca = PCA()
    global_pca_result = global_pca.fit_transform(weighted_data)
    global_pca_df = pd.DataFrame(
        global_pca_result,
        index=data.index,
        columns=[f"F{i+1}" for i in range(global_pca_result.shape[1])]
    )
    
    # 第4步：计算每组的部分因子得分
    partial_factor_scores = {}
    for group_name, variables in groups.items():
        group_data = data[variables]
        # 将原始数据投影到全局PCA空间
        partial_scores = group_data.values @ global_pca.components_.T * np.sqrt(group_weights[group_name])
        partial_factor_scores[group_name] = pd.DataFrame(
            partial_scores,
            index=data.index,
            columns=[f"F{i+1}" for i in range(partial_scores.shape[1])]
        )
    
    return global_pca, global_pca_df, partial_factor_scores, group_weights

# 执行MFA
global_pca, global_scores, partial_scores, group_weights = perform_mfa(scaled_data, groups)

# 输出MFA结果
print("\nMFA分析结果:")
print(f"各组权重: {group_weights}")
print("\n主成分解释方差比例:")
print(global_pca.explained_variance_ratio_[:5])  # 显示前5个主成分
print("\n累积解释方差比例:")
print(np.cumsum(global_pca.explained_variance_ratio_)[:5])  # 显示前5个主成分的累积方差

# 可视化结果
# 1. 碎石图
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(global_pca.explained_variance_ratio_) + 1), 
        global_pca.explained_variance_ratio_, alpha=0.7, color='skyblue')
plt.plot(range(1, len(global_pca.explained_variance_ratio_) + 1), 
         np.cumsum(global_pca.explained_variance_ratio_), 'ro-')
plt.xlabel('主成分')
plt.ylabel('解释方差比例')
plt.title('MFA碎石图')
plt.xticks(range(1, len(global_pca.explained_variance_ratio_) + 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('MFA_scree_plot.png', dpi=300)

# 2. 变量因子载荷图
plt.figure(figsize=(12, 10))
loadings = global_pca.components_.T * np.sqrt(global_pca.explained_variance_)

# 为不同组的变量使用不同颜色
colors = {'感官评价': 'red', '理化指标': 'blue', '消费者偏好': 'green'}
markers = {'感官评价': 'o', '理化指标': 's', '消费者偏好': '^'}

# 绘制变量载荷
for group_name, variables in groups.items():
    for i, var in enumerate(variables):
        var_idx = list(all_data.columns).index(var)
        plt.arrow(0, 0, loadings[var_idx, 0], loadings[var_idx, 1], 
                  head_width=0.05, head_length=0.05, fc=colors[group_name], ec=colors[group_name])
        plt.text(loadings[var_idx, 0]*1.15, loadings[var_idx, 1]*1.15, 
                 var, color=colors[group_name], ha='center', va='center')

# 绘制单位圆
circle = plt.Circle((0,0), 1, fill=False, color='gray', linestyle='--')
plt.gca().add_patch(circle)

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel(f'F1 ({global_pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'F2 ({global_pca.explained_variance_ratio_[1]:.2%})')
plt.title('MFA变量因子载荷图')
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
for group_name, color in colors.items():
    plt.scatter([], [], color=color, marker=markers[group_name], label=group_name)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('MFA_variable_loadings.png', dpi=300)

# 3. 样本因子得分图
plt.figure(figsize=(12, 10))

# 绘制全局得分
plt.scatter(global_scores['F1'], global_scores['F2'], 
            color='black', s=100, alpha=0.7, label='全局得分')

# 为每个样本添加标签
for i, sample in enumerate(global_scores.index):
    plt.text(global_scores['F1'].iloc[i], global_scores['F2'].iloc[i], 
             sample, fontsize=9)

# 绘制部分因子得分
for group_name, scores in partial_scores.items():
    plt.scatter(scores['F1'], scores['F2'], 
                color=colors[group_name], marker=markers[group_name], 
                s=50, alpha=0.5, label=f'{group_name}部分得分')
    
    # 连接全局得分和部分得分
    for i in range(len(global_scores)):
        plt.plot([global_scores['F1'].iloc[i], scores['F1'].iloc[i]], 
                 [global_scores['F2'].iloc[i], scores['F2'].iloc[i]], 
                 color=colors[group_name], alpha=0.3, linestyle='-')

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel(f'F1 ({global_pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'F2 ({global_pca.explained_variance_ratio_[1]:.2%})')
plt.title('MFA样本因子得分图')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('MFA_sample_scores.png', dpi=300)

# 4. 组的贡献图
plt.figure(figsize=(10, 6))
group_contrib = {}

# 计算每组对主成分的贡献
for group_name, variables in groups.items():
    var_indices = [list(all_data.columns).index(var) for var in variables]
    group_loadings = loadings[var_indices, :]
    # 计算每组对每个主成分的贡献
    group_contrib[group_name] = np.sum(group_loadings**2, axis=0) / np.sum(loadings**2, axis=0)

# 转换为DataFrame便于绘图
group_contrib_df = pd.DataFrame(group_contrib, index=[f'F{i+1}' for i in range(loadings.shape[1])])

# 绘制堆叠条形图
group_contrib_df.iloc[:5].T.plot(kind='bar', stacked=False, figsize=(10, 6), 
                               colormap='viridis')
plt.xlabel('变量组')
plt.ylabel('贡献比例')
plt.title('各组对主成分的贡献')
plt.legend(title='主成分')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('MFA_group_contributions.png', dpi=300)

print("\nMFA分析完成，图表已保存。")

# 如果需要，可以保存结果到CSV文件
global_scores.to_csv('MFA_global_scores.csv')
for group_name, scores in partial_scores.items():
    scores.to_csv(f'MFA_{group_name}_partial_scores.csv')

if __name__ == "__main__":
    print("多因素分析(MFA)示例运行完成")
