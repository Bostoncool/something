import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 生成具有多重共线性的合成数据
def generate_multicollinear_data(n_samples=200, n_features=10, noise=0.5, random_state=42):
    """
    生成具有多重共线性的合成数据
    
    参数:
    n_samples: 样本数量
    n_features: 特征数量
    noise: 噪声水平
    random_state: 随机种子
    
    返回:
    X: 特征矩阵
    y: 目标变量
    """
    X, y = make_regression(n_samples=n_samples, 
                          n_features=n_features, 
                          n_informative=5,  # 只有5个特征是真正有信息量的
                          noise=noise, 
                          random_state=random_state)
    
    # 添加多重共线性 - 使某些特征成为其他特征的线性组合
    X[:, 5] = X[:, 0] * 0.8 + X[:, 1] * 0.2 + np.random.normal(0, 0.1, n_samples)
    X[:, 6] = X[:, 1] * 0.7 + X[:, 2] * 0.3 + np.random.normal(0, 0.1, n_samples)
    X[:, 7] = X[:, 0] * 0.4 + X[:, 3] * 0.6 + np.random.normal(0, 0.1, n_samples)
    
    return X, y

# 生成数据
X, y = generate_multicollinear_data()

# 将数据转换为DataFrame以便于分析
feature_names = [f'特征_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['目标'] = y

# 查看数据的基本统计信息
print("数据基本统计信息:")
print(df.describe().round(2))

# 计算特征之间的相关性矩阵
correlation_matrix = df.corr().round(2)
print("\n特征相关性矩阵:")
print(correlation_matrix)

# 可视化相关性矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('特征相关性热力图')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建标准化器
scaler = StandardScaler()

# 训练普通线性回归模型
lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# 训练Ridge回归模型
# 使用网格搜索找到最佳的alpha值
param_grid = {'ridge__alpha': np.logspace(-3, 3, 7)}
ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

grid_search = GridSearchCV(ridge_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# 获取最佳模型
best_ridge_model = grid_search.best_estimator_
best_alpha = grid_search.best_params_['ridge__alpha']
print(f"\n最佳alpha值: {best_alpha}")

# 使用最佳模型进行预测
ridge_pred = best_ridge_model.predict(X_test)

# 评估模型性能
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

print("\n模型性能比较:")
print(f"线性回归 - MSE: {lr_mse:.4f}, R²: {lr_r2:.4f}")
print(f"Ridge回归 - MSE: {ridge_mse:.4f}, R²: {ridge_r2:.4f}")

# 提取系数
lr_coef = lr_model.named_steps['linear'].coef_
ridge_coef = best_ridge_model.named_steps['ridge'].coef_

# 创建系数比较DataFrame
coef_df = pd.DataFrame({
    '特征': feature_names,
    '线性回归系数': lr_coef,
    'Ridge回归系数': ridge_coef
})

print("\n模型系数比较:")
print(coef_df)

# 可视化系数比较
plt.figure(figsize=(12, 8))
bar_width = 0.35
index = np.arange(len(feature_names))

plt.bar(index, lr_coef, bar_width, label='线性回归系数', color='blue', alpha=0.7)
plt.bar(index + bar_width, ridge_coef, bar_width, label='Ridge回归系数', color='red', alpha=0.7)

plt.xlabel('特征')
plt.ylabel('系数值')
plt.title('线性回归与Ridge回归系数比较')
plt.xticks(index + bar_width / 2, feature_names, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('coefficient_comparison.png')
plt.close()

# 可视化不同alpha值对Ridge回归系数的影响
alphas = np.logspace(-3, 3, 7)
coefs = []

for alpha in alphas:
    ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])
    ridge.fit(X_train, y_train)
    coefs.append(ridge.named_steps['ridge'].coef_)

# 绘制系数随alpha变化的曲线
plt.figure(figsize=(12, 8))
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha值 (对数刻度)')
plt.ylabel('系数值')
plt.title('Ridge回归系数随正则化强度(alpha)的变化')
plt.axis('tight')
plt.legend([f'特征_{i}' for i in range(X.shape[1])])
plt.tight_layout()
plt.savefig('ridge_path.png')
plt.close()

# 学习曲线 - 比较不同训练集大小下的模型性能
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, title, ylim=None, cv=5, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("训练样本数")
    plt.ylabel("得分")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="训练集得分")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="验证集得分")
    plt.legend(loc="best")
    return plt

# 绘制学习曲线
plot_learning_curve(
    Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=best_alpha))]),
    X, y, "Ridge回归学习曲线 (alpha={})".format(best_alpha))
plt.savefig('learning_curve.png')
plt.close()

print("\n分析总结:")
print("1. Ridge回归通过引入L2正则化项，有效缓解了多重共线性问题")
print("2. 与普通线性回归相比，Ridge回归的系数更加稳定，模型泛化能力更强")
print("3. 最佳alpha值为{}，在此值下模型取得了最佳性能".format(best_alpha))
print("4. Ridge回归特别适合处理特征之间存在高度相关性的数据集")
print("5. 所有结果已保存为图表，可以直观查看模型性能和系数变化")
