import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib as mpl

# 设置中文字体显示
try:
    # 尝试设置中文字体
    font = FontProperties(fname=r'C:\Windows\Fonts\SimHei.ttf')
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("无法设置中文字体，图表中文可能显示为方块")

# 设置更好看的图表风格
sns.set_style('whitegrid')
plt.style.use('seaborn-v0_8-pastel')

# ================ 1. Brinson 模型绩效归因 ================

def brinson_attribution(portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns, sectors):
    """
    使用Brinson模型进行绩效归因分析
    
    参数:
    portfolio_weights: 投资组合权重 DataFrame，索引为股票代码，列为日期
    benchmark_weights: 基准指数权重 DataFrame，索引为股票代码，列为日期
    portfolio_returns: 投资组合收益率 DataFrame，索引为股票代码，列为日期
    benchmark_returns: 基准指数收益率 DataFrame，索引为股票代码，列为日期
    sectors: Series，索引为股票代码，值为行业分类
    
    返回:
    归因结果 DataFrame
    """
    # 确保所有输入数据具有相同的股票和日期
    common_stocks = portfolio_weights.index.intersection(benchmark_weights.index)
    common_stocks = common_stocks.intersection(portfolio_returns.index)
    common_stocks = common_stocks.intersection(benchmark_returns.index)
    common_stocks = common_stocks.intersection(sectors.index)
    
    dates = portfolio_weights.columns.intersection(benchmark_weights.columns)
    dates = dates.intersection(portfolio_returns.columns)
    dates = dates.intersection(benchmark_returns.columns)
    
    # 筛选共同的股票和日期
    portfolio_weights = portfolio_weights.loc[common_stocks, dates]
    benchmark_weights = benchmark_weights.loc[common_stocks, dates]
    portfolio_returns = portfolio_returns.loc[common_stocks, dates]
    benchmark_returns = benchmark_returns.loc[common_stocks, dates]
    sectors = sectors.loc[common_stocks]
    
    # 初始化结果DataFrame
    results = pd.DataFrame(index=dates, columns=['总超额收益', '资产配置效应', '个股选择效应', '交互效应'])
    
    for date in dates:
        # 获取当前日期的数据
        pw = portfolio_weights[date]
        bw = benchmark_weights[date]
        pr = portfolio_returns[date]
        br = benchmark_returns[date]
        
        # 计算行业级别的权重和收益率
        sector_portfolio_weights = pw.groupby(sectors).sum()
        sector_benchmark_weights = bw.groupby(sectors).sum()
        
        # 计算行业级别的收益率（加权平均）
        sector_portfolio_returns = pd.Series(index=sector_portfolio_weights.index, dtype=float)
        sector_benchmark_returns = pd.Series(index=sector_benchmark_weights.index, dtype=float)
        
        for sector in sector_portfolio_weights.index:
            sector_stocks = sectors[sectors == sector].index
            if len(sector_stocks) > 0:
                # 行业内的投资组合权重和基准权重
                sector_pw = pw.loc[sector_stocks]
                sector_bw = bw.loc[sector_stocks]
                
                # 归一化行业内权重
                if sector_pw.sum() > 0:
                    sector_pw = sector_pw / sector_pw.sum() * sector_portfolio_weights[sector]
                if sector_bw.sum() > 0:
                    sector_bw = sector_bw / sector_bw.sum() * sector_benchmark_weights[sector]
                
                # 计算行业收益率
                sector_pr = pr.loc[sector_stocks]
                sector_br = br.loc[sector_stocks]
                
                sector_portfolio_returns[sector] = (sector_pw * sector_pr).sum() / sector_pw.sum() if sector_pw.sum() > 0 else 0
                sector_benchmark_returns[sector] = (sector_bw * sector_br).sum() / sector_bw.sum() if sector_bw.sum() > 0 else 0
        
        # Brinson模型计算
        # 1. 资产配置效应：基准收益率下，投资组合权重与基准权重的差异带来的影响
        allocation_effect = ((sector_portfolio_weights - sector_benchmark_weights) * sector_benchmark_returns).sum()
        
        # 2. 个股选择效应：投资组合权重下，投资组合收益率与基准收益率的差异带来的影响
        selection_effect = (sector_benchmark_weights * (sector_portfolio_returns - sector_benchmark_returns)).sum()
        
        # 3. 交互效应：权重差异和收益率差异的交互作用
        interaction_effect = ((sector_portfolio_weights - sector_benchmark_weights) * 
                             (sector_portfolio_returns - sector_benchmark_returns)).sum()
        
        # 4. 总超额收益
        total_excess_return = (pw * pr).sum() - (bw * br).sum()
        
        # 存储结果
        results.loc[date, '总超额收益'] = total_excess_return
        results.loc[date, '资产配置效应'] = allocation_effect
        results.loc[date, '个股选择效应'] = selection_effect
        results.loc[date, '交互效应'] = interaction_effect
    
    return results

# ================ 2. 因子归因分析 ================

def factor_attribution(portfolio_returns, factor_returns, portfolio_exposures):
    """
    使用因子模型进行绩效归因分析
    
    参数:
    portfolio_returns: Series，投资组合的总收益率
    factor_returns: DataFrame，因子收益率，列为因子名称
    portfolio_exposures: DataFrame，投资组合对各因子的暴露度，列为因子名称
    
    返回:
    归因结果 DataFrame
    """
    # 初始化结果DataFrame
    results = pd.DataFrame(index=portfolio_returns.index, 
                          columns=list(factor_returns.columns) + ['特质收益'])
    
    # 计算每个因子的贡献
    for factor in factor_returns.columns:
        results[factor] = portfolio_exposures[factor] * factor_returns[factor]
    
    # 计算特质收益（总收益减去所有因子贡献）
    factor_contribution = results.drop(columns=['特质收益']).sum(axis=1)
    results['特质收益'] = portfolio_returns - factor_contribution
    
    # 计算累计贡献
    cumulative_results = results.cumsum()
    
    return results, cumulative_results

# ================ 示例数据生成 ================

def generate_sample_data(n_stocks=50, n_days=252, n_sectors=10, seed=42):
    """生成示例数据用于绩效归因分析"""
    np.random.seed(seed)
    
    # 生成日期索引
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='B')
    
    # 生成股票代码
    stocks = [f'股票{i+1:03d}' for i in range(n_stocks)]
    
    # 生成行业分类
    sectors = pd.Series(np.random.choice([f'行业{i+1}' for i in range(n_sectors)], size=n_stocks), index=stocks)
    
    # 生成投资组合权重（随时间变化）
    portfolio_weights = pd.DataFrame(np.random.dirichlet(np.ones(n_stocks), size=n_days).T,
                                    index=stocks, columns=dates)
    
    # 生成基准指数权重（随时间变化）
    benchmark_weights = pd.DataFrame(np.random.dirichlet(np.ones(n_stocks), size=n_days).T,
                                    index=stocks, columns=dates)
    
    # 生成股票收益率
    # 假设市场因子、规模因子和价值因子
    market_factor = np.random.normal(0.0005, 0.01, n_days)  # 市场因子收益率
    size_factor = np.random.normal(0.0002, 0.005, n_days)   # 规模因子收益率
    value_factor = np.random.normal(0.0001, 0.004, n_days)  # 价值因子收益率
    
    # 股票对因子的暴露度
    market_exposure = np.random.normal(1, 0.2, n_stocks)
    size_exposure = np.random.normal(0, 1, n_stocks)
    value_exposure = np.random.normal(0, 1, n_stocks)
    
    # 生成股票收益率
    stock_returns = np.zeros((n_stocks, n_days))
    for i in range(n_stocks):
        for j in range(n_days):
            # 系统性收益
            systematic_return = (market_exposure[i] * market_factor[j] + 
                                size_exposure[i] * size_factor[j] + 
                                value_exposure[i] * value_factor[j])
            # 特质收益
            idiosyncratic_return = np.random.normal(0, 0.015)
            stock_returns[i, j] = systematic_return + idiosyncratic_return
    
    portfolio_returns = pd.DataFrame(stock_returns, index=stocks, columns=dates)
    
    # 基准收益率与投资组合收益率相似，但有一些差异
    benchmark_returns = portfolio_returns + np.random.normal(0, 0.002, (n_stocks, n_days))
    
    # 生成因子收益率数据
    factor_returns = pd.DataFrame({
        '市场因子': market_factor,
        '规模因子': size_factor,
        '价值因子': value_factor
    }, index=dates)
    
    # 生成投资组合对因子的暴露度
    portfolio_exposures = pd.DataFrame({
        '市场因子': np.random.normal(1.05, 0.1, n_days),  # 略微高于市场
        '规模因子': np.random.normal(0.2, 0.3, n_days),   # 偏向小盘股
        '价值因子': np.random.normal(0.1, 0.3, n_days)    # 略微偏向价值股
    }, index=dates)
    
    # 计算投资组合总收益率
    portfolio_total_returns = pd.Series(index=dates)
    for date in dates:
        portfolio_total_returns[date] = (portfolio_weights[date] * portfolio_returns[date]).sum()
    
    return {
        'portfolio_weights': portfolio_weights,
        'benchmark_weights': benchmark_weights,
        'portfolio_returns': portfolio_returns,
        'benchmark_returns': benchmark_returns,
        'sectors': sectors,
        'factor_returns': factor_returns,
        'portfolio_exposures': portfolio_exposures,
        'portfolio_total_returns': portfolio_total_returns,
        'dates': dates
    }

# ================ 可视化函数 ================

def plot_brinson_attribution(attribution_results):
    """绘制Brinson模型归因结果"""
    # 计算累计效应
    cumulative_results = attribution_results.cumsum()
    
    plt.figure(figsize=(12, 8))
    
    # 绘制累计效应
    cumulative_results.plot(ax=plt.gca())
    
    plt.title('Brinson模型绩效归因分析 - 累计效应', fontproperties=font, fontsize=15)
    plt.xlabel('日期', fontproperties=font)
    plt.ylabel('累计贡献', fontproperties=font)
    plt.legend(prop=font)
    plt.grid(True, alpha=0.3)
    
    # 绘制饼图显示各效应的贡献占比
    plt.figure(figsize=(10, 8))
    
    # 使用最终累计值
    final_values = cumulative_results.iloc[-1]
    abs_values = final_values.abs()
    contribution_pct = abs_values / abs_values.sum() * 100
    
    # 设置颜色
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    # 绘制饼图
    plt.pie(contribution_pct, labels=contribution_pct.index, autopct='%1.1f%%', 
            startangle=90, colors=colors, textprops={'fontproperties': font})
    plt.axis('equal')
    plt.title('Brinson模型各效应贡献占比', fontproperties=font, fontsize=15)
    
    return plt

def plot_factor_attribution(factor_results, cumulative_factor_results):
    """绘制因子归因结果"""
    plt.figure(figsize=(12, 8))
    
    # 绘制累计因子贡献
    cumulative_factor_results.plot(ax=plt.gca())
    
    plt.title('因子归因分析 - 累计贡献', fontproperties=font, fontsize=15)
    plt.xlabel('日期', fontproperties=font)
    plt.ylabel('累计贡献', fontproperties=font)
    plt.legend(prop=font)
    plt.grid(True, alpha=0.3)
    
    # 绘制堆叠图显示每日因子贡献
    plt.figure(figsize=(14, 8))
    
    factor_results.plot(kind='bar', stacked=True, ax=plt.gca(), alpha=0.7)
    
    plt.title('因子归因分析 - 每日贡献', fontproperties=font, fontsize=15)
    plt.xlabel('日期', fontproperties=font)
    plt.ylabel('收益贡献', fontproperties=font)
    plt.legend(prop=font)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 只显示部分日期标签，避免拥挤
    ax = plt.gca()
    for i, label in enumerate(ax.xaxis.get_ticklabels()):
        if i % 20 != 0:
            label.set_visible(False)
    
    return plt

# ================ 主函数 ================

def main():
    print("生成示例数据...")
    data = generate_sample_data(n_stocks=50, n_days=252)
    
    print("执行Brinson模型绩效归因分析...")
    brinson_results = brinson_attribution(
        data['portfolio_weights'], 
        data['benchmark_weights'], 
        data['portfolio_returns'], 
        data['benchmark_returns'], 
        data['sectors']
    )
    
    print("执行因子归因分析...")
    factor_results, cumulative_factor_results = factor_attribution(
        data['portfolio_total_returns'],
        data['factor_returns'],
        data['portfolio_exposures']
    )
    
    print("绘制Brinson模型归因结果...")
    plot_brinson_attribution(brinson_results)
    
    print("绘制因子归因结果...")
    plot_factor_attribution(factor_results, cumulative_factor_results)
    
    # 显示结果摘要
    print("\n===== Brinson模型归因结果摘要 =====")
    print(brinson_results.describe())
    
    print("\n===== 因子归因结果摘要 =====")
    print(factor_results.describe())
    
    # 保存图表
    plt.savefig('brinson_attribution.png')
    plt.savefig('factor_attribution.png')
    
    print("分析完成！图表已保存。")
    plt.show()

if __name__ == "__main__":
    main()
