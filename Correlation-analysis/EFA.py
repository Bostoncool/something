#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
探索性因子分析(EFA)示例代码
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

# 设置中文字体显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")

def generate_sample_data(n_samples=500):
    """
    生成示例数据用于因子分析
    
    参数:
    n_samples: 样本数量
    
    返回:
    pandas DataFrame 包含模拟的问卷数据
    """
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 假设我们有三个潜在因子
    latent_factors = np.random.normal(0, 1, size=(n_samples, 3))
    
    # 创建10个观测变量，它们由潜在因子线性组合而成
    # 变量1-4主要受第一个因子影响
    # 变量5-7主要受第二个因子影响
    # 变量8-10主要受第三个因子影响
    observed_vars = np.zeros((n_samples, 10))
    
    # 第一个因子影响的变量
    observed_vars[:, 0] = 0.8 * latent_factors[:, 0] + 0.2 * latent_factors[:, 1] + np.random.normal(0, 0.3, n_samples)
    observed_vars[:, 1] = 0.7 * latent_factors[:, 0] + 0.1 * latent_factors[:, 2] + np.random.normal(0, 0.4, n_samples)
    observed_vars[:, 2] = 0.9 * latent_factors[:, 0] + 0.1 * latent_factors[:, 1] + np.random.normal(0, 0.2, n_samples)
    observed_vars[:, 3] = 0.6 * latent_factors[:, 0] + 0.3 * latent_factors[:, 2] + np.random.normal(0, 0.3, n_samples)
    
    # 第二个因子影响的变量
    observed_vars[:, 4] = 0.2 * latent_factors[:, 0] + 0.8 * latent_factors[:, 1] + np.random.normal(0, 0.3, n_samples)
    observed_vars[:, 5] = 0.1 * latent_factors[:, 0] + 0.9 * latent_factors[:, 1] + np.random.normal(0, 0.2, n_samples)
    observed_vars[:, 6] = 0.3 * latent_factors[:, 2] + 0.7 * latent_factors[:, 1] + np.random.normal(0, 0.4, n_samples)
    
    # 第三个因子影响的变量
    observed_vars[:, 7] = 0.2 * latent_factors[:, 1] + 0.8 * latent_factors[:, 2] + np.random.normal(0, 0.3, n_samples)
    observed_vars[:, 8] = 0.1 * latent_factors[:, 0] + 0.9 * latent_factors[:, 2] + np.random.normal(0, 0.2, n_samples)
    observed_vars[:, 9] = 0.3 * latent_factors[:, 1] + 0.7 * latent_factors[:, 2] + np.random.normal(0, 0.4, n_samples)
    
    # 将数据转换为DataFrame
    column_names = [f'问题{i+1}' for i in range(10)]
    df = pd.DataFrame(observed_vars, columns=column_names)
    
    # 将数据缩放到1-5的范围，模拟李克特量表
    df = 1 + 4 * (df - df.min()) / (df.max() - df.min())
    df = df.round()  # 四舍五入到整数
    
    return df

def check_efa_assumptions(df):
    """
    检查数据是否适合进行因子分析
    
    参数:
    df: 包含观测变量的DataFrame
    
    返回:
    适合性评估结果
    """
    # 计算Bartlett球形度检验
    chi_square_value, p_value = calculate_bartlett_sphericity(df)
    print(f"Bartlett球形度检验:")
    print(f"卡方值: {chi_square_value:.4f}")
    print(f"p值: {p_value:.8f}")
    print(f"结论: {'数据适合因子分析' if p_value < 0.05 else '数据不适合因子分析'}")
    print()
    
    # 计算KMO值
    kmo_all, kmo_model = calculate_kmo(df)
    print(f"KMO采样充分性检验:")
    print(f"总体KMO: {kmo_model:.4f}")
    print(f"各变量KMO: \n{pd.Series(kmo_all, index=df.columns).round(4)}")
    
    kmo_criteria = ""
    if kmo_model >= 0.9:
        kmo_criteria = "极佳"
    elif kmo_model >= 0.8:
        kmo_criteria = "优秀"
    elif kmo_model >= 0.7:
        kmo_criteria = "中等"
    elif kmo_model >= 0.6:
        kmo_criteria = "一般"
    elif kmo_model >= 0.5:
        kmo_criteria = "勉强接受"
    else:
        kmo_criteria = "不适合"
    
    print(f"KMO值评价: {kmo_criteria}")
    print()
    
    # 计算相关矩阵
    corr_matrix = df.corr()
    
    # 绘制相关矩阵热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('变量相关矩阵热图')
    plt.tight_layout()
    plt.show()
    
    return kmo_model, p_value

def determine_factors_number(df):
    """
    确定最佳因子数量
    
    参数:
    df: 包含观测变量的DataFrame
    
    返回:
    建议的因子数量
    """
    # 标准化数据
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    # 计算特征值以确定因子数量
    fa = FactorAnalyzer()
    fa.fit(df_scaled)
    ev, v = fa.get_eigenvalues()
    
    # 绘制碎石图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ev) + 1), ev, 'o-', linewidth=2, color='blue')
    plt.axhline(y=1, color='red', linestyle='--')
    plt.title('碎石图')
    plt.xlabel('因子数量')
    plt.ylabel('特征值')
    plt.grid(True)
    plt.show()
    
    # 计算解释的方差比例
    explained_variance = ev / sum(ev) * 100
    cumulative_variance = np.cumsum(explained_variance)
    
    # 绘制解释方差比例图
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='skyblue')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', color='red', marker='o')
    plt.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='70% 方差')
    plt.title('解释方差比例')
    plt.xlabel('因子数量')
    plt.ylabel('解释方差百分比')
    plt.legend(['累积方差', '各因子方差', '70% 方差阈值'])
    plt.grid(True)
    plt.show()
    
    # 输出特征值和解释方差
    eigenvalue_df = pd.DataFrame({
        '特征值': ev,
        '解释方差比例(%)': explained_variance,
        '累积解释方差(%)': cumulative_variance
    })
    eigenvalue_df.index = [f'因子{i+1}' for i in range(len(ev))]
    print("特征值和解释方差:")
    print(eigenvalue_df.round(4))
    print()
    
    # 根据Kaiser准则(特征值>1)确定因子数量
    n_factors_kaiser = sum(ev > 1)
    print(f"根据Kaiser准则(特征值>1)建议的因子数量: {n_factors_kaiser}")
    
    # 根据解释方差比例(>70%)确定因子数量
    n_factors_variance = np.argmax(cumulative_variance >= 70) + 1
    print(f"根据累积解释方差(>70%)建议的因子数量: {n_factors_variance}")
    
    # 返回建议的因子数量(取两者的较大值)
    return max(n_factors_kaiser, n_factors_variance)

def perform_factor_analysis(df, n_factors, rotation='varimax'):
    """
    执行因子分析
    
    参数:
    df: 包含观测变量的DataFrame
    n_factors: 因子数量
    rotation: 旋转方法
    
    返回:
    因子分析结果
    """
    # 标准化数据
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    # 执行因子分析
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method='principal')
    fa.fit(df_scaled)
    
    # 获取因子载荷
    loadings = fa.loadings_
    loadings_df = pd.DataFrame(loadings, index=df.columns, 
                              columns=[f'因子{i+1}' for i in range(n_factors)])
    
    # 获取共同度
    communalities = fa.get_communalities()
    communalities_df = pd.DataFrame({'共同度': communalities}, index=df.columns)
    
    # 获取特殊方差
    specific_variance = fa.get_uniquenesses()
    specific_variance_df = pd.DataFrame({'特殊方差': specific_variance}, index=df.columns)
    
    # 合并结果
    result_df = pd.concat([loadings_df, communalities_df, specific_variance_df], axis=1)
    
    # 打印因子载荷矩阵
    print(f"因子载荷矩阵 (旋转方法: {rotation}):")
    print(result_df.round(4))
    print()
    
    # 计算因子得分
    factor_scores = fa.transform(df_scaled)
    factor_scores_df = pd.DataFrame(factor_scores, 
                                   columns=[f'因子{i+1}得分' for i in range(n_factors)])
    
    # 绘制因子载荷热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings_df, annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title(f'因子载荷矩阵热图 (旋转方法: {rotation})')
    plt.tight_layout()
    plt.show()
    
    # 如果因子数量为2或3，绘制因子载荷散点图
    if n_factors >= 2:
        plt.figure(figsize=(10, 8))
        for i, var in enumerate(df.columns):
            plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.02, head_length=0.03, 
                     fc='blue', ec='blue', alpha=0.7)
            plt.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, var, fontsize=12)
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.grid(alpha=0.3)
        plt.title('因子载荷散点图 (因子1 vs 因子2)')
        plt.xlabel('因子1')
        plt.ylabel('因子2')
        plt.tight_layout()
        plt.show()
        
        if n_factors >= 3:
            # 绘制3D因子载荷图
            try:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                for i, var in enumerate(df.columns):
                    ax.text(loadings[i, 0], loadings[i, 1], loadings[i, 2], var, fontsize=10)
                    ax.plot([0, loadings[i, 0]], [0, loadings[i, 1]], [0, loadings[i, 2]], 'b-', alpha=0.6)
                
                ax.scatter(loadings[:, 0], loadings[:, 1], loadings[:, 2], color='red', s=50, alpha=0.7)
                
                # 绘制坐标轴
                ax.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.2)
                ax.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.2)
                ax.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.2)
                
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                ax.set_xlabel('因子1')
                ax.set_ylabel('因子2')
                ax.set_zlabel('因子3')
                ax.set_title('3D因子载荷图')
                plt.tight_layout()
                plt.show()
            except:
                print("无法绘制3D图形，请确保已安装mpl_toolkits")
    
    return loadings_df, factor_scores_df

def interpret_factors(loadings_df, threshold=0.5):
    """
    解释因子结构
    
    参数:
    loadings_df: 因子载荷DataFrame
    threshold: 载荷阈值，高于此值的变量被认为与因子相关
    
    返回:
    因子解释结果
    """
    n_factors = loadings_df.shape[1]
    
    print(f"因子结构解释 (载荷阈值: {threshold}):")
    for i in range(n_factors):
        factor_name = f"因子{i+1}"
        # 找出载荷绝对值大于阈值的变量
        high_loading_vars = loadings_df.index[abs(loadings_df[factor_name]) >= threshold].tolist()
        
        if high_loading_vars:
            print(f"\n{factor_name} 主要由以下变量构成:")
            for var in high_loading_vars:
                loading = loadings_df.loc[var, factor_name]
                print(f"  - {var}: {loading:.4f}")
        else:
            print(f"\n{factor_name} 没有变量的载荷绝对值超过阈值 {threshold}")
    
    # 检查是否有变量没有高载荷
    all_high_loading_vars = set()
    for i in range(n_factors):
        factor_name = f"因子{i+1}"
        high_loading_vars = loadings_df.index[abs(loadings_df[factor_name]) >= threshold].tolist()
        all_high_loading_vars.update(high_loading_vars)
    
    low_loading_vars = set(loadings_df.index) - all_high_loading_vars
    if low_loading_vars:
        print("\n以下变量在所有因子上的载荷都低于阈值:")
        for var in low_loading_vars:
            print(f"  - {var}")
    
    # 检查交叉载荷(一个变量在多个因子上有高载荷)
    cross_loading_vars = []
    for var in loadings_df.index:
        high_loading_factors = []
        for i in range(n_factors):
            factor_name = f"因子{i+1}"
            if abs(loadings_df.loc[var, factor_name]) >= threshold:
                high_loading_factors.append(factor_name)
        
        if len(high_loading_factors) > 1:
            cross_loading_vars.append((var, high_loading_factors))
    
    if cross_loading_vars:
        print("\n以下变量在多个因子上有高载荷(交叉载荷):")
        for var, factors in cross_loading_vars:
            print(f"  - {var}: {', '.join(factors)}")

def main():
    """
    主函数，执行完整的探索性因子分析流程
    """
    print("=" * 80)
    print("探索性因子分析(EFA)示例")
    print("=" * 80)
    
    # 生成示例数据
    print("生成示例数据...")
    df = generate_sample_data(n_samples=500)
    
    # 显示数据基本信息
    print("\n数据概览:")
    print(f"样本数量: {df.shape[0]}")
    print(f"变量数量: {df.shape[1]}")
    print("\n数据前5行:")
    print(df.head())
    
    # 数据描述性统计
    print("\n描述性统计:")
    print(df.describe().round(2))
    
    # 检查数据是否适合进行因子分析
    print("\n检查数据是否适合进行因子分析...")
    kmo, p_value = check_efa_assumptions(df)
    
    if kmo < 0.5 or p_value >= 0.05:
        print("警告: 数据可能不适合进行因子分析!")
        print("KMO值应大于0.5，Bartlett球形度检验的p值应小于0.05")
        print("尽管如此，我们仍将继续分析以作示例")
    
    # 确定最佳因子数量
    print("\n确定最佳因子数量...")
    n_factors = determine_factors_number(df)
    print(f"\n建议的因子数量: {n_factors}")
    
    # 执行因子分析
    print("\n执行因子分析...")
    loadings_df, factor_scores_df = perform_factor_analysis(df, n_factors, rotation='varimax')
    
    # 解释因子结构
    print("\n解释因子结构...")
    interpret_factors(loadings_df, threshold=0.5)
    
    print("\n探索性因子分析完成!")

if __name__ == "__main__":
    main()
