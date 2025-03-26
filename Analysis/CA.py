#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
相关性分析(Correlation Analysis)示例
包含多种相关系数计算及可视化:
- Pearson、Spearman和Kendall相关系数
- Point-Biserial相关系数
- Phi系数
- Cramér's V系数
- Tetrachoric相关系数
- Polychoric相关系数
- 典型相关分析(CCA)
- 偏相关系数
- 距离相关系数
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from dcor import distance_correlation
from statsmodels.stats.contingency_tables import Table
import pingouin as pg
import statsmodels.api as sm

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.set(font='SimHei')  # 设置seaborn默认字体

# 设置随机种子以确保结果可重现
np.random.seed(42)
random.seed(42)

def generate_sample_data(n_samples=100):
    """
    生成示例数据集，包含不同类型的相关关系
    
    参数:
        n_samples: 样本数量
        
    返回:
        pandas DataFrame 包含多个变量
    """
    # 线性相关变量
    x1 = np.random.normal(0, 1, n_samples)
    y1 = 2 * x1 + np.random.normal(0, 0.5, n_samples)  # 强正相关
    
    # 非线性相关变量
    x2 = np.random.uniform(-3, 3, n_samples)
    y2 = x2**2 + np.random.normal(0, 1, n_samples)  # 非线性相关
    
    # 负相关变量
    x3 = np.random.normal(0, 1, n_samples)
    y3 = -1.5 * x3 + np.random.normal(0, 0.7, n_samples)  # 负相关
    
    # 无相关变量
    x4 = np.random.normal(0, 1, n_samples)
    y4 = np.random.normal(0, 1, n_samples)  # 无相关
    
    # 分类变量
    categories = ['A', 'B', 'C', 'D']
    cat_var = [random.choice(categories) for _ in range(n_samples)]
    
    # 二分类变量 (用于Point-Biserial和Phi系数)
    binary_var1 = np.random.binomial(1, 0.5, n_samples)
    binary_var2 = np.random.binomial(1, 0.5, n_samples)
    
    # 多分类变量 (用于Cramér's V)
    multi_cat1 = np.random.choice(['类别1', '类别2', '类别3'], n_samples)
    multi_cat2 = np.random.choice(['组A', '组B', '组C', '组D'], n_samples)
    
    # 潜在连续变量的二分类化 (用于Tetrachoric相关)
    latent1 = np.random.normal(0, 1, n_samples)
    latent2 = 0.7 * latent1 + np.random.normal(0, np.sqrt(1 - 0.7**2), n_samples)
    binary_latent1 = (latent1 > 0).astype(int)
    binary_latent2 = (latent2 > 0).astype(int)
    
    # 潜在连续变量的多分类化 (用于Polychoric相关)
    poly_latent1 = np.random.normal(0, 1, n_samples)
    poly_latent2 = 0.6 * poly_latent1 + np.random.normal(0, np.sqrt(1 - 0.6**2), n_samples)
    
    # 将连续变量转换为有序分类变量
    def to_ordinal(x, n_categories=4):
        quantiles = np.linspace(0, 1, n_categories+1)[1:-1]
        thresholds = np.quantile(x, quantiles)
        categories = np.zeros(len(x), dtype=int)
        for i, threshold in enumerate(thresholds):
            categories[x > threshold] = i + 1
        return categories
    
    ordinal1 = to_ordinal(poly_latent1)
    ordinal2 = to_ordinal(poly_latent2)
    
    # 多变量集合 (用于典型相关分析)
    X_set = np.random.normal(0, 1, (n_samples, 3))
    Y_set = 0.5 * X_set + np.random.normal(0, 0.8, (n_samples, 3))
    
    # 创建DataFrame
    df = pd.DataFrame({
        '变量X1': x1,
        '变量Y1': y1,
        '变量X2': x2,
        '变量Y2': y2,
        '变量X3': x3,
        '变量Y3': y3,
        '变量X4': x4,
        '变量Y4': y4,
        '分类变量': cat_var,
        '二分类变量1': binary_var1,
        '二分类变量2': binary_var2,
        '多分类变量1': multi_cat1,
        '多分类变量2': multi_cat2,
        '二分潜变量1': binary_latent1,
        '二分潜变量2': binary_latent2,
        '有序变量1': ordinal1,
        '有序变量2': ordinal2
    })
    
    # 添加多变量集合
    for i in range(3):
        df[f'X集合{i+1}'] = X_set[:, i]
        df[f'Y集合{i+1}'] = Y_set[:, i]
    
    return df

def calculate_correlations(df, numeric_columns=None):
    """
    计算不同类型的相关系数
    
    参数:
        df: pandas DataFrame
        numeric_columns: 要计算相关性的数值列列表
        
    返回:
        pearson_corr, spearman_corr, kendall_corr: 三种相关系数矩阵
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # 计算Pearson相关系数（线性相关）
    pearson_corr = df[numeric_columns].corr(method='pearson')
    
    # 计算Spearman相关系数（秩相关，对非线性单调关系敏感）
    spearman_corr = df[numeric_columns].corr(method='spearman')
    
    # 计算Kendall相关系数（秩相关，对异常值更稳健）
    kendall_corr = df[numeric_columns].corr(method='kendall')
    
    return pearson_corr, spearman_corr, kendall_corr

def visualize_correlations(pearson_corr, spearman_corr, kendall_corr):
    """
    可视化不同的相关系数矩阵
    
    参数:
        pearson_corr, spearman_corr, kendall_corr: 三种相关系数矩阵
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 设置热图参数
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Pearson相关系数热图
    sns.heatmap(pearson_corr, annot=True, cmap=cmap, vmin=-1, vmax=1, 
                square=True, linewidths=.5, ax=axes[0], fmt=".2f")
    axes[0].set_title('Pearson相关系数\n(线性相关)')
    
    # Spearman相关系数热图
    sns.heatmap(spearman_corr, annot=True, cmap=cmap, vmin=-1, vmax=1, 
                square=True, linewidths=.5, ax=axes[1], fmt=".2f")
    axes[1].set_title('Spearman相关系数\n(秩相关，适用于非线性单调关系)')
    
    # Kendall相关系数热图
    sns.heatmap(kendall_corr, annot=True, cmap=cmap, vmin=-1, vmax=1, 
                square=True, linewidths=.5, ax=axes[2], fmt=".2f")
    axes[2].set_title('Kendall相关系数\n(秩相关，对异常值更稳健)')
    
    plt.tight_layout()
    plt.savefig('correlation_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_scatter_relationships(df):
    """
    绘制散点图矩阵以可视化变量间关系
    
    参数:
        df: pandas DataFrame
    """
    # 选择要绘制的变量对
    pairs = [
        ('变量X1', '变量Y1'),  # 线性正相关
        ('变量X2', '变量Y2'),  # 非线性相关
        ('变量X3', '变量Y3'),  # 线性负相关
        ('变量X4', '变量Y4')   # 无相关
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (x, y) in enumerate(pairs):
        # 计算相关系数
        pearson_r, pearson_p = stats.pearsonr(df[x], df[y])
        spearman_r, spearman_p = stats.spearmanr(df[x], df[y])
        
        # 绘制散点图和回归线
        sns.regplot(x=x, y=y, data=df, ax=axes[i], scatter_kws={'alpha':0.6})
        
        # 添加相关系数信息
        axes[i].set_title(f'{x} vs {y}')
        axes[i].annotate(f'Pearson r: {pearson_r:.2f} (p={pearson_p:.3f})\n'
                         f'Spearman ρ: {spearman_r:.2f} (p={spearman_p:.3f})',
                         xy=(0.05, 0.95), xycoords='axes fraction',
                         ha='left', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('correlation_scatterplots.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_categorical_relationships(df, cat_col='分类变量'):
    """
    分析分类变量与数值变量之间的关系
    
    参数:
        df: pandas DataFrame
        cat_col: 分类变量的列名
    """
    # 选择数值变量
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 使用箱线图可视化分类变量与数值变量的关系
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols[:4]):  # 只取前4个数值变量
        sns.boxplot(x=cat_col, y=col, data=df, ax=axes[i])
        axes[i].set_title(f'{cat_col} vs {col}')
        
        # 进行ANOVA检验
        categories = df[cat_col].unique()
        samples = [df[df[cat_col] == cat][col].values for cat in categories]
        f_stat, p_val = stats.f_oneway(*samples)
        
        axes[i].annotate(f'ANOVA: F={f_stat:.2f}, p={p_val:.3f}',
                         xy=(0.05, 0.95), xycoords='axes fraction',
                         ha='left', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('categorical_relationships.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_point_biserial(df, continuous_col, binary_col):
    """
    计算点二列相关系数 (Point-Biserial Correlation)
    用于一个连续变量和一个二分类变量之间的相关性分析
    
    参数:
        df: pandas DataFrame
        continuous_col: 连续变量的列名
        binary_col: 二分类变量的列名
        
    返回:
        correlation: 相关系数
        p_value: p值
    """
    # 确保二分类变量是0和1
    if not set(df[binary_col].unique()).issubset({0, 1}):
        raise ValueError("二分类变量必须只包含0和1")
    
    # 使用scipy.stats中的pointbiserialr函数
    correlation, p_value = stats.pointbiserialr(df[continuous_col], df[binary_col])
    
    return correlation, p_value

def calculate_phi_coefficient(df, binary_col1, binary_col2):
    """
    计算Phi系数 (Phi Coefficient)
    用于两个二分类变量之间的相关性分析
    
    参数:
        df: pandas DataFrame
        binary_col1: 第一个二分类变量的列名
        binary_col2: 第二个二分类变量的列名
        
    返回:
        phi: Phi系数
        p_value: p值
    """
    # 创建列联表
    contingency_table = pd.crosstab(df[binary_col1], df[binary_col2])
    
    # 计算卡方值和p值
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # 计算Phi系数
    n = contingency_table.sum().sum()
    phi = np.sqrt(chi2 / n)
    
    # 如果两个变量负相关，则Phi系数为负
    if contingency_table.iloc[0, 0] * contingency_table.iloc[1, 1] < contingency_table.iloc[0, 1] * contingency_table.iloc[1, 0]:
        phi = -phi
    
    return phi, p_value

def calculate_cramers_v(df, cat_col1, cat_col2):
    """
    计算Cramér's V系数
    用于两个分类变量之间的相关性分析
    
    参数:
        df: pandas DataFrame
        cat_col1: 第一个分类变量的列名
        cat_col2: 第二个分类变量的列名
        
    返回:
        v: Cramér's V系数
        p_value: p值
    """
    # 创建列联表
    contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
    
    # 计算卡方值和p值
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # 计算Cramér's V系数
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape
    v = np.sqrt(chi2 / (n * min(r-1, k-1)))
    
    return v, p_value

def calculate_tetrachoric(df, binary_col1, binary_col2):
    """
    计算四分相关系数 (Tetrachoric Correlation)
    用于两个二分类变量之间的相关性分析，假设它们是由潜在连续变量二分化而来
    
    参数:
        df: pandas DataFrame
        binary_col1: 第一个二分类变量的列名
        binary_col2: 第二个二分类变量的列名
        
    返回:
        rho: 四分相关系数
    """
    # 创建2x2列联表
    contingency_table = pd.crosstab(df[binary_col1], df[binary_col2])
    
    # 使用statsmodels计算四分相关系数
    table = Table(contingency_table.values)
    rho = table.tetrachoric()[0]
    
    return rho

def calculate_polychoric(df, ordinal_col1, ordinal_col2):
    """
    计算多分相关系数 (Polychoric Correlation)
    用于两个有序分类变量之间的相关性分析，假设它们是由潜在连续变量分类化而来
    
    参数:
        df: pandas DataFrame
        ordinal_col1: 第一个有序分类变量的列名
        ordinal_col2: 第二个有序分类变量的列名
        
    返回:
        rho: 多分相关系数
    """
    # 使用pingouin库计算多分相关系数
    result = pg.polychoric_corr(df[ordinal_col1], df[ordinal_col2])
    rho = result[0]  # 相关系数
    
    return rho

def calculate_canonical_correlation(df, X_cols, Y_cols):
    """
    计算典型相关系数 (Canonical Correlation)
    用于两组变量之间的相关性分析
    
    参数:
        df: pandas DataFrame
        X_cols: 第一组变量的列名列表
        Y_cols: 第二组变量的列名列表
        
    返回:
        correlations: 典型相关系数列表
        X_loadings: X变量的载荷矩阵
        Y_loadings: Y变量的载荷矩阵
    """
    # 提取数据
    X = df[X_cols].values
    Y = df[Y_cols].values
    
    # 标准化数据
    X_std = StandardScaler().fit_transform(X)
    Y_std = StandardScaler().fit_transform(Y)
    
    # 计算典型相关系数
    n_components = min(len(X_cols), len(Y_cols))
    cca = CCA(n_components=n_components)
    cca.fit(X_std, Y_std)
    
    # 计算相关系数
    X_c, Y_c = cca.transform(X_std, Y_std)
    correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
    
    # 计算载荷
    X_loadings = cca.x_loadings_
    Y_loadings = cca.y_loadings_
    
    return correlations, X_loadings, Y_loadings

def calculate_partial_correlation(df, var1, var2, control_vars):
    """
    计算偏相关系数 (Partial Correlation)
    用于测量两个变量之间的相关性，同时控制其他变量的影响
    
    参数:
        df: pandas DataFrame
        var1: 第一个变量的列名
        var2: 第二个变量的列名
        control_vars: 控制变量的列名列表
        
    返回:
        r: 偏相关系数
        p_value: p值
    """
    # 使用pingouin库计算偏相关系数
    result = pg.partial_corr(data=df, x=var1, y=var2, covar=control_vars)
    r = result['r'].values[0]
    p_value = result['p-val'].values[0]
    
    return r, p_value

def calculate_distance_correlation(df, var1, var2):
    """
    计算距离相关系数 (Distance Correlation)
    用于检测非线性关系，不依赖于线性假设
    
    参数:
        df: pandas DataFrame
        var1: 第一个变量的列名
        var2: 第二个变量的列名
        
    返回:
        dcor: 距离相关系数
    """
    # 计算距离相关系数
    dcor = distance_correlation(df[var1].values, df[var2].values)
    
    return dcor

def visualize_additional_correlations(df):
    """
    可视化额外的相关性分析方法
    
    参数:
        df: pandas DataFrame
    """
    # 创建一个包含所有相关系数的字典
    results = {}
    
    # 1. Point-Biserial相关
    pb_corr, pb_p = calculate_point_biserial(df, '变量X1', '二分类变量1')
    results['Point-Biserial (X1 vs 二分类1)'] = pb_corr
    
    # 2. Phi系数
    phi, phi_p = calculate_phi_coefficient(df, '二分类变量1', '二分类变量2')
    results['Phi系数 (二分类1 vs 二分类2)'] = phi
    
    # 3. Cramér's V系数
    cv, cv_p = calculate_cramers_v(df, '多分类变量1', '多分类变量2')
    results['Cramér\'s V (多分类1 vs 多分类2)'] = cv
    
    # 4. 四分相关系数
    tetra = calculate_tetrachoric(df, '二分潜变量1', '二分潜变量2')
    results['四分相关 (二分潜变量1 vs 二分潜变量2)'] = tetra
    
    # 5. 多分相关系数
    poly = calculate_polychoric(df, '有序变量1', '有序变量2')
    results['多分相关 (有序变量1 vs 有序变量2)'] = poly
    
    # 6. 典型相关系数
    X_cols = ['X集合1', 'X集合2', 'X集合3']
    Y_cols = ['Y集合1', 'Y集合2', 'Y集合3']
    cca_corrs, _, _ = calculate_canonical_correlation(df, X_cols, Y_cols)
    for i, corr in enumerate(cca_corrs):
        results[f'典型相关 {i+1}'] = corr
    
    # 7. 偏相关系数
    partial_corr, partial_p = calculate_partial_correlation(df, '变量X1', '变量Y1', ['变量X3'])
    results['偏相关 (X1 vs Y1 | X3)'] = partial_corr
    
    # 8. 距离相关系数
    dcor = calculate_distance_correlation(df, '变量X2', '变量Y2')
    results['距离相关 (X2 vs Y2)'] = dcor
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(results)), list(results.values()), color='skyblue')
    plt.xticks(range(len(results)), list(results.keys()), rotation=45, ha='right')
    plt.ylabel('相关系数值')
    plt.title('不同相关性分析方法的结果比较')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('additional_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def main():
    """主函数"""
    print("开始相关性分析示例...")
    
    # 生成示例数据
    df = generate_sample_data(n_samples=150)
    print("\n数据集前5行:")
    print(df.head())
    
    # 数据描述性统计
    print("\n数据描述性统计:")
    print(df.describe())
    
    # 计算基本相关系数
    numeric_cols = ['变量X1', '变量Y1', '变量X2', '变量Y2', '变量X3', '变量Y3', '变量X4', '变量Y4']
    pearson_corr, spearman_corr, kendall_corr = calculate_correlations(df, numeric_cols)
    
    print("\nPearson相关系数矩阵:")
    print(pearson_corr)
    
    print("\nSpearman相关系数矩阵:")
    print(spearman_corr)
    
    print("Kendall相关系数矩阵:")
    print(kendall_corr)
    
    # 可视化相关系数矩阵
    print("\n生成相关系数热图...")
    visualize_correlations(pearson_corr, spearman_corr, kendall_corr)
    
    # 绘制散点图矩阵
    print("\n生成散点图矩阵...")
    plot_scatter_relationships(df)
    
    # 分析分类变量与数值变量的关系
    print("\n分析分类变量与数值变量的关系...")
    analyze_categorical_relationships(df)
    
    # 计算和可视化额外的相关性分析方法
    print("\n计算额外的相关性分析方法...")
    additional_results = visualize_additional_correlations(df)
    
    # 打印额外的相关性分析结果
    print("\n额外的相关性分析结果:")
    for name, value in additional_results.items():
        print(f"{name}: {value:.4f}")
    
    # Point-Biserial相关示例
    pb_corr, pb_p = calculate_point_biserial(df, '变量X1', '二分类变量1')
    print(f"\nPoint-Biserial相关 (变量X1 vs 二分类变量1): r = {pb_corr:.4f}, p = {pb_p:.4f}")
    
    # Phi系数示例
    phi, phi_p = calculate_phi_coefficient(df, '二分类变量1', '二分类变量2')
    print(f"Phi系数 (二分类变量1 vs 二分类变量2): φ = {phi:.4f}, p = {phi_p:.4f}")
    
    # Cramér's V系数示例
    cv, cv_p = calculate_cramers_v(df, '多分类变量1', '多分类变量2')
    print(f"Cramér's V系数 (多分类变量1 vs 多分类变量2): V = {cv:.4f}, p = {cv_p:.4f}")
    
    # 四分相关系数示例
    tetra = calculate_tetrachoric(df, '二分潜变量1', '二分潜变量2')
    print(f"四分相关系数 (二分潜变量1 vs 二分潜变量2): ρ = {tetra:.4f}")
    
    # 多分相关系数示例
    poly = calculate_polychoric(df, '有序变量1', '有序变量2')
    print(f"多分相关系数 (有序变量1 vs 有序变量2): ρ = {poly:.4f}")
    
    # 典型相关系数示例
    X_cols = ['X集合1', 'X集合2', 'X集合3']
    Y_cols = ['Y集合1', 'Y集合2', 'Y集合3']
    cca_corrs, X_loadings, Y_loadings = calculate_canonical_correlation(df, X_cols, Y_cols)
    print("\n典型相关系数:")
    for i, corr in enumerate(cca_corrs):
        print(f"典型相关 {i+1}: {corr:.4f}")
    
    # 偏相关系数示例
    partial_corr, partial_p = calculate_partial_correlation(df, '变量X1', '变量Y1', ['变量X3'])
    print(f"\n偏相关系数 (变量X1 vs 变量Y1 | 变量X3): r = {partial_corr:.4f}, p = {partial_p:.4f}")
    
    # 距离相关系数示例
    dcor = calculate_distance_correlation(df, '变量X2', '变量Y2')
    print(f"距离相关系数 (变量X2 vs 变量Y2): dCor = {dcor:.4f}")
    
    print("\n相关性分析完成!")

if __name__ == "__main__":
    main()
