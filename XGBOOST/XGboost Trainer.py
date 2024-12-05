import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    """加载和预处理数据"""
    # 读取数据
    print("正在加载数据...")
    data = pd.read_csv(file_path)
    
    # 显示基本信息
    print("\n数据基本信息：")
    print(data.info())
    
    # 检查缺失值
    print("\n缺失值统计：")
    print(data.isnull().sum())
    
    # 填充缺失值
    data.fillna(data.mean(), inplace=True)
    
    # 对分类变量进行编码
    data = pd.get_dummies(data)
    
    return data

def train_xgboost_model(X_train, X_test, y_train, y_test, is_classification=False):
    """训练XGBoost模型"""
    # 创建DMatrix数据结构
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 设置参数
    params = {
        'max_depth': 3,
        'eta': 0.1,
        'objective': 'binary:logistic' if is_classification else 'reg:squarederror',
        'eval_metric': 'logloss' if is_classification else 'rmse',
        'silent': 1
    }
    
    # 设置评估列表
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    
    # 训练模型
    print("\n开始训练模型...")
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=100,
        evals=evallist,
        early_stopping_rounds=10,
        verbose_eval=True
    )
    
    return model

def evaluate_model(model, X_test, y_test, is_classification=False):
    """评估模型性能"""
    dtest = xgb.DMatrix(X_test)
    predictions = model.predict(dtest)
    
    if is_classification:
        predictions = [1 if p > 0.5 else 0 for p in predictions]
        accuracy = accuracy_score(y_test, predictions)
        print(f"\n模型准确率: {accuracy:.4f}")
        return accuracy
    else:
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"\n均方根误差 (RMSE): {rmse:.4f}")
        return rmse

def plot_feature_importance(model, X):
    """绘制特征重要性图"""
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10)
    plt.title('特征重要性排序')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    # 设置文件路径
    file_path = 'your_data.csv'  # 请替换为您的CSV文件路径
    target_column = 'target'      # 请替换为您的目标变量列名
    is_classification = True      # 根据您的问题类型设置（分类或回归）
    
    try:
        # 加载和预处理数据，在此改为自己数据储存的路径
        data = load_and_preprocess_data(file_path)
        
        # 分离特征和目标变量，在此改为自己的目标变量
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        model = train_xgboost_model(X_train, X_test, y_train, y_test, is_classification)
        
        # 评估模型
        evaluate_model(model, X_test, y_test, is_classification)
        
        # 绘制特征重要性
        plot_feature_importance(model, X)
        
        # 保存模型
        model.save_model('xgboost_model.json')
        print("\n模型已保存为 'xgboost_model.json'")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()
