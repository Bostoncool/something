"""
北京PM2.5浓度预测 - ARIMA模型
使用ARIMA/SARIMA时间序列模型进行单变量预测

特点:
- 纯时间序列方法（仅使用PM2.5历史值）
- SARIMA支持季节性分析
- auto_arima自动参数优化
- 完整的残差诊断
- ACF/PACF分析
- ADF平稳性检验

数据来源:
- 污染数据: Benchmark数据集 (PM2.5浓度)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import pickle
from pathlib import Path
import multiprocessing

warnings.filterwarnings('ignore')

# 获取CPU核心数
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT - 1)

# 尝试导入tqdm进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("提示: tqdm未安装，进度显示将使用简化版本。")

# 时间序列库
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from scipy import stats

# auto_arima
try:
    import pmdarima as pm
    AUTO_ARIMA_AVAILABLE = True
except ImportError:
    print("警告: pmdarima未安装，将跳过auto_arima优化。")
    print("      可使用 'pip install pmdarima' 安装。")
    AUTO_ARIMA_AVAILABLE = False

# 评估指标
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 设置随机种子
np.random.seed(42)

print("=" * 80)
print("北京PM2.5浓度预测 - ARIMA模型")
print("=" * 80)

# ============================== 第1部分: 配置和路径设置 ==============================
print("\n配置参数...")

# 数据路径
pollution_all_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)'
pollution_extra_path = r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)'

# 输出路径
output_dir = Path('./output')
output_dir.mkdir(exist_ok=True)

# 模型保存路径
model_dir = Path('./models')
model_dir.mkdir(exist_ok=True)

# 日期范围
start_date = datetime(2015, 1, 1)
end_date = datetime(2024, 12, 31)

print(f"数据时间范围: {start_date.date()} 至 {end_date.date()}")
print(f"目标变量: PM2.5浓度（时间序列）")
print(f"输出目录: {output_dir}")
print(f"模型保存目录: {model_dir}")
print(f"CPU核心数: {CPU_COUNT}, 并行工作线程: {MAX_WORKERS}")

# ============================== 第2部分: 数据加载函数 ==============================
def daterange(start, end):
    """生成日期序列"""
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(n)

def find_file(base_path, date_str, prefix):
    """查找指定日期的文件"""
    filename = f"{prefix}_{date_str}.csv"
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def read_pollution_day(date):
    """读取单日污染数据"""
    date_str = date.strftime('%Y%m%d')
    all_file = find_file(pollution_all_path, date_str, 'beijing_all')
    
    if not all_file:
        return None
    
    try:
        df_all = pd.read_csv(all_file, encoding='utf-8', on_bad_lines='skip')
        
        # 过滤掉24小时平均和AQI
        df_all = df_all[~df_all['type'].str.contains('_24h|AQI', na=False)]
        
        # 只保留PM2.5
        df_pm25 = df_all[df_all['type'] == 'PM2.5']
        
        if df_pm25.empty:
            return None
        
        # 转换为长格式
        df_pm25 = df_pm25.melt(id_vars=['date', 'hour', 'type'], 
                                var_name='station', value_name='value')
        df_pm25['value'] = pd.to_numeric(df_pm25['value'], errors='coerce')
        
        # 删除负值
        df_pm25 = df_pm25[df_pm25['value'] >= 0]
        
        # 按日期聚合（所有站点平均）
        daily_mean = df_pm25.groupby('date')['value'].mean()
        
        if len(daily_mean) == 0:
            return None
        
        # 将索引转换为datetime格式
        daily_mean.index = pd.to_datetime(daily_mean.index, format='%Y%m%d', errors='coerce')
        
        return daily_mean
    except Exception as e:
        return None

def read_all_pollution():
    """并行读取所有污染数据"""
    print("\n正在加载污染数据...")
    print(f"使用 {MAX_WORKERS} 个并行工作线程")
    dates = list(daterange(start_date, end_date))
    pollution_series = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(read_pollution_day, date): date for date in dates}
        
        if TQDM_AVAILABLE:
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="加载污染数据", unit="天"):
                result = future.result()
                if result is not None:
                    pollution_series.append(result)
        else:
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    pollution_series.append(result)
                if i % 500 == 0 or i == len(futures):
                    print(f"  已处理 {i}/{len(futures)} 天 ({i/len(futures)*100:.1f}%)")
    
    if pollution_series:
        print(f"  成功读取 {len(pollution_series)}/{len(dates)} 天的数据")
        print("  正在合并数据...")
        pm25_series = pd.concat(pollution_series)
        pm25_series = pm25_series.sort_index()
        
        # 去重（保留第一个）
        pm25_series = pm25_series[~pm25_series.index.duplicated(keep='first')]
        
        # 填充缺失日期
        full_index = pd.date_range(start=pm25_series.index.min(), 
                                    end=pm25_series.index.max(), 
                                    freq='D')
        pm25_series = pm25_series.reindex(full_index)
        
        # 前向填充然后均值填充
        pm25_series = pm25_series.ffill().bfill()
        pm25_series = pm25_series.fillna(pm25_series.mean())
        
        print(f"污染数据加载完成，形状: {pm25_series.shape}")
        return pm25_series
    return pd.Series()

# ============================== 第3部分: 数据加载和预处理 ==============================
print("\n" + "=" * 80)
print("第1步: 数据加载和预处理")
print("=" * 80)

pm25_series = read_all_pollution()

# 检查数据加载情况
print("\n数据加载检查:")
print(f"  PM2.5序列形状: {pm25_series.shape}")

if pm25_series.empty:
    print("\n⚠️ 警告: 污染数据为空！请检查数据路径和文件。")
    import sys
    sys.exit(1)

print(f"  时间范围: {pm25_series.index.min()} 至 {pm25_series.index.max()}")
print(f"  数据点数: {len(pm25_series)}")

# 基本统计信息
print(f"\nPM2.5统计信息:")
print(f"  均值: {pm25_series.mean():.2f} μg/m³")
print(f"  标准差: {pm25_series.std():.2f} μg/m³")
print(f"  最小值: {pm25_series.min():.2f} μg/m³")
print(f"  最大值: {pm25_series.max():.2f} μg/m³")
print(f"  中位数: {pm25_series.median():.2f} μg/m³")
print(f"  缺失值: {pm25_series.isna().sum()}")

# ============================== 第4部分: 时间序列分析 ==============================
print("\n" + "=" * 80)
print("第2步: 时间序列平稳性分析")
print("=" * 80)

# ADF检验（增强迪基-福勒检验）
print("\n进行ADF平稳性检验...")
adf_result = adfuller(pm25_series.dropna())
print(f"\nADF检验结果:")
print(f"  ADF统计量: {adf_result[0]:.4f}")
print(f"  p值: {adf_result[1]:.4f}")
print(f"  临界值:")
for key, value in adf_result[4].items():
    print(f"    {key}: {value:.4f}")

if adf_result[1] < 0.05:
    print(f"\n✓ 序列是平稳的 (p={adf_result[1]:.4f} < 0.05)")
else:
    print(f"\n⚠ 序列非平稳 (p={adf_result[1]:.4f} >= 0.05)，ARIMA会自动进行差分")

# ============================== 第5部分: 数据集划分 ==============================
print("\n" + "=" * 80)
print("第3步: 数据集划分")
print("=" * 80)

# 按时间顺序划分：训练集70%，验证集15%，测试集15%
n_samples = len(pm25_series)
train_size = int(n_samples * 0.70)
val_size = int(n_samples * 0.15)

y_train = pm25_series.iloc[:train_size]
y_val = pm25_series.iloc[train_size:train_size + val_size]
y_test = pm25_series.iloc[train_size + val_size:]

print(f"\n训练集: {len(y_train)} 样本 ({len(y_train)/n_samples*100:.1f}%)")
print(f"  时间范围: {y_train.index.min().date()} 至 {y_train.index.max().date()}")
print(f"  PM2.5: {y_train.mean():.2f} ± {y_train.std():.2f} μg/m³")

print(f"\n验证集: {len(y_val)} 样本 ({len(y_val)/n_samples*100:.1f}%)")
print(f"  时间范围: {y_val.index.min().date()} 至 {y_val.index.max().date()}")
print(f"  PM2.5: {y_val.mean():.2f} ± {y_val.std():.2f} μg/m³")

print(f"\n测试集: {len(y_test)} 样本 ({len(y_test)/n_samples*100:.1f}%)")
print(f"  时间范围: {y_test.index.min().date()} 至 {y_test.index.max().date()}")
print(f"  PM2.5: {y_test.mean():.2f} ± {y_test.std():.2f} μg/m³")

# ============================== 第6部分: 基础ARIMA模型 ==============================
print("\n" + "=" * 80)
print("第4步: 基础SARIMA模型训练")
print("=" * 80)

# 基础模型参数
order_basic = (5, 1, 5)  # (p, d, q)
seasonal_order_basic = (1, 1, 1, 12)  # (P, D, Q, s)

print(f"\n基础模型参数:")
print(f"  ARIMA阶数 (p,d,q): {order_basic}")
print(f"  季节阶数 (P,D,Q,s): {seasonal_order_basic}")

print("\n开始训练基础模型...")
model_basic = SARIMAX(y_train, 
                      order=order_basic,
                      seasonal_order=seasonal_order_basic,
                      enforce_stationarity=False,
                      enforce_invertibility=False)

fit_basic = model_basic.fit(disp=False, maxiter=200)

print(f"\n✓ 基础模型训练完成")
print(f"  AIC: {fit_basic.aic:.2f}")
print(f"  BIC: {fit_basic.bic:.2f}")
print(f"  对数似然: {fit_basic.llf:.2f}")

# 预测
print("\n进行预测...")
y_train_pred_basic = fit_basic.fittedvalues
y_val_pred_basic = fit_basic.forecast(steps=len(y_val))
y_test_pred_basic = fit_basic.forecast(steps=len(y_val) + len(y_test)).iloc[len(y_val):]

# 评估
def evaluate_model(y_true, y_pred, dataset_name):
    """评估模型性能"""
    # 确保索引对齐
    y_true_aligned = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred_aligned = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    
    r2 = r2_score(y_true_aligned, y_pred_aligned)
    rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
    mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
    mape = np.mean(np.abs((y_true_aligned - y_pred_aligned) / y_true_aligned)) * 100
    return {
        'Dataset': dataset_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

results_basic = []
results_basic.append(evaluate_model(y_train, y_train_pred_basic, 'Train'))
results_basic.append(evaluate_model(y_val, y_val_pred_basic, 'Validation'))
results_basic.append(evaluate_model(y_test, y_test_pred_basic, 'Test'))

results_basic_df = pd.DataFrame(results_basic)
print("\n基础模型性能:")
print(results_basic_df.to_string(index=False))

# ============================== 第7部分: 超参数优化 ==============================
print("\n" + "=" * 80)
print("第5步: 超参数优化（auto_arima）")
print("=" * 80)

if AUTO_ARIMA_AVAILABLE:
    print("\n使用auto_arima自动搜索最佳参数...")
    print("这可能需要几分钟时间...")
    
    auto_model = pm.auto_arima(
        y_train,
        start_p=1, start_q=1,
        max_p=7, max_q=7,
        d=None,  # 自动确定差分阶数
        seasonal=True, m=12,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        D=None,  # 自动确定季节差分阶数
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        n_jobs=-1
    )
    
    print(f"\n✓ auto_arima搜索完成")
    print(f"\n最佳模型: {auto_model.order} x {auto_model.seasonal_order}")
    print(f"  AIC: {auto_model.aic():.2f}")
    print(f"  BIC: {auto_model.bic():.2f}")
    
    best_order = auto_model.order
    best_seasonal_order = auto_model.seasonal_order
    
else:
    print("\n⚠ auto_arima不可用，使用手动优化参数")
    best_order = (3, 1, 3)
    best_seasonal_order = (1, 1, 1, 12)
    print(f"  ARIMA阶数 (p,d,q): {best_order}")
    print(f"  季节阶数 (P,D,Q,s): {best_seasonal_order}")

# ============================== 第8部分: 训练优化模型 ==============================
print("\n" + "=" * 80)
print("第6步: 使用最佳参数训练优化模型")
print("=" * 80)

print(f"\n优化模型参数:")
print(f"  ARIMA阶数 (p,d,q): {best_order}")
print(f"  季节阶数 (P,D,Q,s): {best_seasonal_order}")

print("\n开始训练优化模型...")
model_optimized = SARIMAX(y_train,
                          order=best_order,
                          seasonal_order=best_seasonal_order,
                          enforce_stationarity=False,
                          enforce_invertibility=False)

fit_optimized = model_optimized.fit(disp=False, maxiter=200)

print(f"\n✓ 优化模型训练完成")
print(f"  AIC: {fit_optimized.aic:.2f}")
print(f"  BIC: {fit_optimized.bic:.2f}")
print(f"  对数似然: {fit_optimized.llf:.2f}")

# 预测
print("\n进行预测...")
y_train_pred_opt = fit_optimized.fittedvalues
y_val_pred_opt = fit_optimized.forecast(steps=len(y_val))
y_test_pred_opt = fit_optimized.forecast(steps=len(y_val) + len(y_test)).iloc[len(y_val):]

# 评估
results_opt = []
results_opt.append(evaluate_model(y_train, y_train_pred_opt, 'Train'))
results_opt.append(evaluate_model(y_val, y_val_pred_opt, 'Validation'))
results_opt.append(evaluate_model(y_test, y_test_pred_opt, 'Test'))

results_opt_df = pd.DataFrame(results_opt)
print("\n优化模型性能:")
print(results_opt_df.to_string(index=False))

# ============================== 第9部分: 模型比较 ==============================
print("\n" + "=" * 80)
print("第7步: 模型性能比较")
print("=" * 80)

# 合并结果
results_basic_df['Model'] = 'ARIMA_Basic'
results_opt_df['Model'] = 'ARIMA_Optimized'
all_results = pd.concat([results_basic_df, results_opt_df])

# 重新排列列顺序
all_results = all_results[['Model', 'Dataset', 'R²', 'RMSE', 'MAE', 'MAPE']]

print("\n所有模型性能对比:")
print(all_results.to_string(index=False))

# 测试集性能对比
test_results = all_results[all_results['Dataset'] == 'Test'].sort_values('R²', ascending=False)
print("\n测试集性能排名:")
print(test_results.to_string(index=False))

# 性能提升
basic_test_r2 = results_basic_df[results_basic_df['Dataset'] == 'Test']['R²'].values[0]
opt_test_r2 = results_opt_df[results_opt_df['Dataset'] == 'Test']['R²'].values[0]
basic_test_rmse = results_basic_df[results_basic_df['Dataset'] == 'Test']['RMSE'].values[0]
opt_test_rmse = results_opt_df[results_opt_df['Dataset'] == 'Test']['RMSE'].values[0]

if basic_test_r2 != 0:
    r2_improvement = (opt_test_r2 - basic_test_r2) / abs(basic_test_r2) * 100
else:
    r2_improvement = 0

if basic_test_rmse != 0:
    rmse_improvement = (basic_test_rmse - opt_test_rmse) / basic_test_rmse * 100
else:
    rmse_improvement = 0

print(f"\n优化效果:")
print(f"  R²变化: {r2_improvement:+.2f}%")
print(f"  RMSE变化: {rmse_improvement:+.2f}%")

# ============================== 第10部分: 残差诊断 ==============================
print("\n" + "=" * 80)
print("第8步: 残差诊断")
print("=" * 80)

# 基础模型残差
residuals_basic = fit_basic.resid
print("\n基础模型残差统计:")
print(f"  均值: {residuals_basic.mean():.4f}")
print(f"  标准差: {residuals_basic.std():.4f}")

# Ljung-Box检验（检验残差自相关）
lb_test_basic = acorr_ljungbox(residuals_basic, lags=10, return_df=True)
print(f"\nLjung-Box检验 (基础模型):")
print(f"  p值 (lag 10): {lb_test_basic['lb_pvalue'].iloc[-1]:.4f}")
if lb_test_basic['lb_pvalue'].iloc[-1] > 0.05:
    print("  ✓ 残差无显著自相关（白噪声）")
else:
    print("  ⚠ 残差存在自相关")

# Jarque-Bera正态性检验
jb_test_basic = stats.jarque_bera(residuals_basic.dropna())
print(f"\nJarque-Bera正态性检验 (基础模型):")
print(f"  统计量: {jb_test_basic[0]:.4f}")
print(f"  p值: {jb_test_basic[1]:.4f}")
if jb_test_basic[1] > 0.05:
    print("  ✓ 残差服从正态分布")
else:
    print("  ⚠ 残差不服从正态分布")

# 优化模型残差
residuals_opt = fit_optimized.resid
print("\n优化模型残差统计:")
print(f"  均值: {residuals_opt.mean():.4f}")
print(f"  标准差: {residuals_opt.std():.4f}")

lb_test_opt = acorr_ljungbox(residuals_opt, lags=10, return_df=True)
print(f"\nLjung-Box检验 (优化模型):")
print(f"  p值 (lag 10): {lb_test_opt['lb_pvalue'].iloc[-1]:.4f}")
if lb_test_opt['lb_pvalue'].iloc[-1] > 0.05:
    print("  ✓ 残差无显著自相关（白噪声）")
else:
    print("  ⚠ 残差存在自相关")

jb_test_opt = stats.jarque_bera(residuals_opt.dropna())
print(f"\nJarque-Bera正态性检验 (优化模型):")
print(f"  统计量: {jb_test_opt[0]:.4f}")
print(f"  p值: {jb_test_opt[1]:.4f}")
if jb_test_opt[1] > 0.05:
    print("  ✓ 残差服从正态分布")
else:
    print("  ⚠ 残差不服从正态分布")

# ============================== 第11部分: 可视化 ==============================
print("\n" + "=" * 80)
print("第9步: 生成可视化图表")
print("=" * 80)

# 11.1 ACF和PACF图（原始序列）
print("生成ACF/PACF图...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 原始序列ACF
plot_acf(pm25_series.dropna(), lags=40, ax=axes[0, 0])
axes[0, 0].set_title('原始序列 - 自相关函数(ACF)', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('滞后期', fontsize=11)
axes[0, 0].set_ylabel('自相关系数', fontsize=11)

# 原始序列PACF
plot_pacf(pm25_series.dropna(), lags=40, ax=axes[0, 1])
axes[0, 1].set_title('原始序列 - 偏自相关函数(PACF)', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('滞后期', fontsize=11)
axes[0, 1].set_ylabel('偏自相关系数', fontsize=11)

# 差分序列ACF
pm25_diff = pm25_series.diff().dropna()
plot_acf(pm25_diff, lags=40, ax=axes[1, 0])
axes[1, 0].set_title('一阶差分序列 - 自相关函数(ACF)', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('滞后期', fontsize=11)
axes[1, 0].set_ylabel('自相关系数', fontsize=11)

# 差分序列PACF
plot_pacf(pm25_diff, lags=40, ax=axes[1, 1])
axes[1, 1].set_title('一阶差分序列 - 偏自相关函数(PACF)', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('滞后期', fontsize=11)
axes[1, 1].set_ylabel('偏自相关系数', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'acf_pacf_plots.png', dpi=300, bbox_inches='tight')
print("保存: acf_pacf_plots.png")
plt.close()

# 11.2 残差诊断图
print("生成残差诊断图...")
fig = plt.figure(figsize=(18, 12))

# 基础模型残差诊断
# QQ图
ax1 = plt.subplot(3, 2, 1)
stats.probplot(residuals_basic.dropna(), dist="norm", plot=plt)
ax1.set_title('基础模型 - QQ图', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 残差时间序列
ax2 = plt.subplot(3, 2, 2)
residuals_basic.plot(ax=ax2, color='blue', alpha=0.7)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_title('基础模型 - 残差时间序列', fontsize=12, fontweight='bold')
ax2.set_xlabel('日期', fontsize=11)
ax2.set_ylabel('残差', fontsize=11)
ax2.grid(True, alpha=0.3)

# 残差ACF
ax3 = plt.subplot(3, 2, 3)
plot_acf(residuals_basic.dropna(), lags=30, ax=ax3)
ax3.set_title('基础模型 - 残差ACF', fontsize=12, fontweight='bold')

# 优化模型残差诊断
# QQ图
ax4 = plt.subplot(3, 2, 4)
stats.probplot(residuals_opt.dropna(), dist="norm", plot=plt)
ax4.set_title('优化模型 - QQ图', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 残差时间序列
ax5 = plt.subplot(3, 2, 5)
residuals_opt.plot(ax=ax5, color='green', alpha=0.7)
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_title('优化模型 - 残差时间序列', fontsize=12, fontweight='bold')
ax5.set_xlabel('日期', fontsize=11)
ax5.set_ylabel('残差', fontsize=11)
ax5.grid(True, alpha=0.3)

# 残差ACF
ax6 = plt.subplot(3, 2, 6)
plot_acf(residuals_opt.dropna(), lags=30, ax=ax6)
ax6.set_title('优化模型 - 残差ACF', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'residual_diagnostics.png', dpi=300, bbox_inches='tight')
print("保存: residual_diagnostics.png")
plt.close()

# 11.3 预测vs实际值散点图
print("生成预测散点图...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

models_data = [
    ('Basic', y_train_pred_basic, y_train, 'Train'),
    ('Basic', y_val_pred_basic, y_val, 'Val'),
    ('Basic', y_test_pred_basic, y_test, 'Test'),
    ('Optimized', y_train_pred_opt, y_train, 'Train'),
    ('Optimized', y_val_pred_opt, y_val, 'Val'),
    ('Optimized', y_test_pred_opt, y_test, 'Test')
]

for idx, (model_name, y_pred, y_true, dataset) in enumerate(models_data):
    row = idx // 3
    col = idx % 3
    
    ax = axes[row, col]
    
    y_true_vals = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred_vals = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    
    # 散点图
    ax.scatter(y_true_vals, y_pred_vals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    
    # 理想预测线
    min_val = min(y_true_vals.min(), y_pred_vals.min())
    max_val = max(y_true_vals.max(), y_pred_vals.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测线')
    
    # 计算指标
    r2 = r2_score(y_true_vals, y_pred_vals)
    rmse = np.sqrt(mean_squared_error(y_true_vals, y_pred_vals))
    
    ax.set_xlabel('实际PM2.5浓度 (μg/m³)', fontsize=11)
    ax.set_ylabel('预测PM2.5浓度 (μg/m³)', fontsize=11)
    ax.set_title(f'ARIMA_{model_name} - {dataset}\nR²={r2:.4f}, RMSE={rmse:.2f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
print("保存: prediction_scatter.png")
plt.close()

# 11.4 时间序列预测对比
print("生成时间序列对比图...")
fig, axes = plt.subplots(2, 1, figsize=(18, 10))

# 测试集 - 基础模型
plot_range = min(300, len(y_test))
plot_idx = range(len(y_test) - plot_range, len(y_test))
time_idx = y_test.index[plot_idx]

axes[0].plot(time_idx, y_test.iloc[plot_idx], 'k-', label='实际值', 
             linewidth=2, alpha=0.8)
axes[0].plot(time_idx, y_test_pred_basic.iloc[plot_idx], 'b--', label='基础模型预测', 
             linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('日期', fontsize=12)
axes[0].set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
axes[0].set_title('ARIMA基础模型 - 时间序列预测对比（测试集最后300天）', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

# 测试集 - 优化模型
axes[1].plot(time_idx, y_test.iloc[plot_idx], 'k-', label='实际值', 
             linewidth=2, alpha=0.8)
axes[1].plot(time_idx, y_test_pred_opt.iloc[plot_idx], 'g--', label='优化模型预测', 
             linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('日期', fontsize=12)
axes[1].set_ylabel('PM2.5浓度 (μg/m³)', fontsize=12)
axes[1].set_title('ARIMA优化模型 - 时间序列预测对比（测试集最后300天）', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(output_dir / 'timeseries_comparison.png', dpi=300, bbox_inches='tight')
print("保存: timeseries_comparison.png")
plt.close()

# 11.5 残差分析散点图
print("生成残差分析图...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (model_name, y_pred, y_true, dataset) in enumerate(models_data):
    row = idx // 3
    col = idx % 3
    
    ax = axes[row, col]
    
    y_true_vals = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred_vals = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    
    residuals = y_true_vals - y_pred_vals
    
    ax.scatter(y_pred_vals, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('预测值 (μg/m³)', fontsize=11)
    ax.set_ylabel('残差 (μg/m³)', fontsize=11)
    ax.set_title(f'ARIMA_{model_name} - {dataset}\n残差均值={residuals.mean():.2f}, 标准差={residuals.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'residuals_analysis.png', dpi=300, bbox_inches='tight')
print("保存: residuals_analysis.png")
plt.close()

# 11.6 模型性能对比柱状图
print("生成模型对比图...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

test_results_plot = all_results[all_results['Dataset'] == 'Test']
models = test_results_plot['Model'].tolist()
x_pos = np.arange(len(models))
colors = ['blue', 'green']

metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
for i, metric in enumerate(metrics):
    axes[i].bar(x_pos, test_results_plot[metric], color=colors, alpha=0.7, 
                edgecolor='black', linewidth=1.5)
    axes[i].set_xticks(x_pos)
    axes[i].set_xticklabels(['Basic', 'Optimized'], fontsize=11)
    axes[i].set_ylabel(metric, fontsize=12)
    
    if metric == 'R²':
        axes[i].set_title(f'{metric} 对比\n(越大越好)', fontsize=12, fontweight='bold')
    else:
        axes[i].set_title(f'{metric} 对比\n(越小越好)', fontsize=12, fontweight='bold')
    
    axes[i].grid(True, alpha=0.3, axis='y')
    
    # 显示数值
    for j, v in enumerate(test_results_plot[metric]):
        if metric == 'MAPE':
            axes[i].text(j, v, f'{v:.1f}%', ha='center', va='bottom', 
                         fontsize=10, fontweight='bold')
        else:
            axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom', 
                         fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
print("保存: model_comparison.png")
plt.close()

# 11.7 误差分布直方图
print("生成误差分布图...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

errors_basic = y_test.values - y_test_pred_basic.values
errors_opt = y_test.values - y_test_pred_opt.values

axes[0].hist(errors_basic, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='零误差')
axes[0].set_xlabel('预测误差 (μg/m³)', fontsize=12)
axes[0].set_ylabel('频数', fontsize=12)
axes[0].set_title(f'基础模型 - 预测误差分布\n均值={errors_basic.mean():.2f}, 标准差={errors_basic.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].hist(errors_opt, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2.5, label='零误差')
axes[1].set_xlabel('预测误差 (μg/m³)', fontsize=12)
axes[1].set_ylabel('频数', fontsize=12)
axes[1].set_title(f'优化模型 - 预测误差分布\n均值={errors_opt.mean():.2f}, 标准差={errors_opt.std():.2f}', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
print("保存: error_distribution.png")
plt.close()

# 11.8 AIC/BIC对比图
print("生成AIC/BIC对比图...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

models_aic_bic = ['Basic', 'Optimized']
aic_values = [fit_basic.aic, fit_optimized.aic]
bic_values = [fit_basic.bic, fit_optimized.bic]

x = np.arange(len(models_aic_bic))
width = 0.35

bars1 = ax.bar(x - width/2, aic_values, width, label='AIC', color='steelblue', 
               edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, bic_values, width, label='BIC', color='coral', 
               edgecolor='black', linewidth=1.5)

ax.set_ylabel('信息准则值', fontsize=12)
ax.set_title('模型信息准则对比 (AIC/BIC)\n值越小越好', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_aic_bic, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# 显示数值
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'aic_bic_comparison.png', dpi=300, bbox_inches='tight')
print("保存: aic_bic_comparison.png")
plt.close()

# ============================== 第12部分: 保存结果 ==============================
print("\n" + "=" * 80)
print("第10步: 保存结果")
print("=" * 80)

# 保存模型性能
all_results.to_csv(output_dir / 'model_performance.csv', index=False, encoding='utf-8-sig')
print("保存: model_performance.csv")

# 保存最佳参数
best_params_df = pd.DataFrame([{
    'ARIMA_order_p': best_order[0],
    'ARIMA_order_d': best_order[1],
    'ARIMA_order_q': best_order[2],
    'Seasonal_order_P': best_seasonal_order[0],
    'Seasonal_order_D': best_seasonal_order[1],
    'Seasonal_order_Q': best_seasonal_order[2],
    'Seasonal_order_s': best_seasonal_order[3],
    'AIC': fit_optimized.aic,
    'BIC': fit_optimized.bic
}])
best_params_df.to_csv(output_dir / 'best_parameters.csv', index=False, encoding='utf-8-sig')
print("保存: best_parameters.csv")

# 保存预测结果
predictions_df = pd.DataFrame({
    'Date': y_test.index,
    'Actual': y_test.values,
    'Prediction_Basic': y_test_pred_basic.values,
    'Prediction_Optimized': y_test_pred_opt.values,
    'Error_Basic': y_test.values - y_test_pred_basic.values,
    'Error_Optimized': y_test.values - y_test_pred_opt.values
})
predictions_df.to_csv(output_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
print("保存: predictions.csv")

# 保存残差诊断结果
diagnostics_df = pd.DataFrame({
    'Model': ['Basic', 'Optimized'],
    'Residual_Mean': [residuals_basic.mean(), residuals_opt.mean()],
    'Residual_Std': [residuals_basic.std(), residuals_opt.std()],
    'Ljungbox_pvalue': [lb_test_basic['lb_pvalue'].iloc[-1], lb_test_opt['lb_pvalue'].iloc[-1]],
    'JarqueBera_statistic': [jb_test_basic[0], jb_test_opt[0]],
    'JarqueBera_pvalue': [jb_test_basic[1], jb_test_opt[1]]
})
diagnostics_df.to_csv(output_dir / 'diagnostics.csv', index=False, encoding='utf-8-sig')
print("保存: diagnostics.csv")

# 保存模型
with open(model_dir / 'arima_basic.pkl', 'wb') as f:
    pickle.dump(fit_basic, f)
print("保存: arima_basic.pkl")

with open(model_dir / 'arima_optimized.pkl', 'wb') as f:
    pickle.dump(fit_optimized, f)
print("保存: arima_optimized.pkl")

# ============================== 第13部分: 总结报告 ==============================
print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)

print("\n生成的文件:")
print("\nCSV文件:")
print("  - model_performance.csv       模型性能对比")
print("  - best_parameters.csv         最佳ARIMA参数")
print("  - predictions.csv             预测结果")
print("  - diagnostics.csv             残差诊断结果")

print("\n图表文件:")
print("  - acf_pacf_plots.png          ACF/PACF分析图")
print("  - residual_diagnostics.png    残差诊断图")
print("  - prediction_scatter.png      预测vs实际散点图")
print("  - timeseries_comparison.png   时间序列对比")
print("  - residuals_analysis.png      残差分析")
print("  - model_comparison.png        模型性能对比")
print("  - error_distribution.png      误差分布")
print("  - aic_bic_comparison.png      AIC/BIC对比")

print("\n模型文件:")
print("  - arima_basic.pkl             基础ARIMA模型")
print("  - arima_optimized.pkl         优化ARIMA模型")

# 最佳模型信息
best_model = test_results.iloc[0]
print(f"\n最佳模型: {best_model['Model']}")
print(f"  R² Score: {best_model['R²']:.4f}")
print(f"  RMSE: {best_model['RMSE']:.2f} μg/m³")
print(f"  MAE: {best_model['MAE']:.2f} μg/m³")
print(f"  MAPE: {best_model['MAPE']:.2f}%")

print(f"\n最佳ARIMA参数:")
print(f"  ARIMA阶数 (p,d,q): {best_order}")
print(f"  季节阶数 (P,D,Q,s): {best_seasonal_order}")
print(f"  AIC: {fit_optimized.aic:.2f}")
print(f"  BIC: {fit_optimized.bic:.2f}")

print("\n" + "=" * 80)
print("ARIMA PM2.5浓度预测完成！")
print("=" * 80)

