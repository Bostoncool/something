# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 数据加载和预处理函数
def load_and_preprocess_data(file_path):
    """加载并预处理PM2.5数据"""
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 将日期字符串转换为datetime对象
    df['date'] = pd.to_datetime(df['date'])
    
    # 提取时间特征
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    
    return df

# 特征工程函数
def feature_engineering(df):
    """创建特征"""
    # 计算滚动平均
    df['pm25_rolling_mean_7d'] = df['pm25'].rolling(window=7).mean()
    df['pm25_rolling_std_7d'] = df['pm25'].rolling(window=7).std()
    
    # 处理缺失值
    df = df.fillna(method='ffill')
    
    return df

# 模型训练函数
def train_xgboost_model(X_train, y_train, X_test, y_test):
    """训练XGBoost模型"""
    # 设置模型参数
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # 创建和训练模型
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    return model

# 模型评估函数
def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f'均方根误差 (RMSE): {rmse:.2f}')
    print(f'R² 分数: {r2:.2f}')
    
    return predictions

# 可视化函数
def plot_results(y_test, predictions):
    """绘制预测结果"""
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('PM2.5 预测值 vs 实际值')
    plt.tight_layout()
    plt.show()

def main():
    # 加载数据
    file_path = 'CHAP_PM2.5_D1K_20230101_V1.csv'  # 替换为实际文件路径
    df = load_and_preprocess_data(file_path)
    
    # 特征工程
    df = feature_engineering(df)
    
    # 准备特征和目标变量
    feature_columns = ['year', 'month', 'day', 'dayofweek', 
                      'pm25_rolling_mean_7d', 'pm25_rolling_std_7d']
    X = df[feature_columns]
    y = df['pm25']
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = train_xgboost_model(X_train, y_train, X_test, y_test)
    
    # 评估模型
    predictions = evaluate_model(model, X_test, y_test)
    
    # 可视化结果
    plot_results(y_test, predictions)
    
    # 特征重要性可视化
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model)
    plt.title('特征重要性')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
