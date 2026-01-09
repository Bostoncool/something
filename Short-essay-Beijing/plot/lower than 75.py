import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# 文件路径列表，包含9个模型的数据文件路径
file_paths = [
    "H:/DATA Science/小论文Result/Fine_model/-BPNN/CSV2/bpnn_predictions__bpnn_optimized.csv",
    "H:/DATA Science/小论文Result/Fine_model/-CNN- LSTM-Transformer/CSV/output/predictions__transformer__test.csv",
    "H:/DATA Science/小论文Result/Fine_model/-CNN-GridSearch/CSV2/cnn_predictions__cnn_optimized__test.csv",
    "H:/DATA Science/小论文Result/Fine_model/-LightGBM/CSV2/plot_ts_lastyear_sampled__lightgbm_optimized.csv",
    "H:/DATA Science/小论文Result/Fine_model/-MLR_GAM/CSV/output/predictions__mlr.csv",
    "H:/DATA Science/小论文Result/Fine_model/-RF/CSV/output/rf_predictions__rf_optimized.csv",
    "H:/DATA Science/小论文Result/Fine_model/-SVR/CSV2/plot_ts_lastyear_sampled__svr_linear.csv",
    "H:/DATA Science/小论文Result/Fine_model/-Transformer/CSV/output/predictions__transformer__test.csv",
    "H:/DATA Science/小论文Result/Fine_model/-XGBOOST/CSV/output/xgboost_predictions__xgboost_optimized__test.csv"
        # 在此添加其他6个模型的文件路径
]

# 创建一个空的DataFrame来存储所有模型的评估结果
results = []

# 遍历每个文件路径
for file_path in file_paths:
    try:
        # 读取数据
        data = pd.read_csv(file_path)
        
        # 获取列名
        date_col = data.columns[0]
        actual_col = data.columns[1]
        pred_col = data.columns[2]
        
        # 确保第一列（索引0）是日期格式 - 使用列名避免FutureWarning
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        
        # 确保第二列和第三列是数值类型
        data[actual_col] = pd.to_numeric(data[actual_col], errors='coerce')
        data[pred_col] = pd.to_numeric(data[pred_col], errors='coerce')
        
        # 删除包含NaN的行
        data = data.dropna(subset=[date_col, actual_col, pred_col])

        # 筛选符合条件的数据：2024年01月01日至2024年12月31日，PM2.5真实值在0到75之间
        # 第一列（索引0）：日期，第二列（索引1）：实际值，第三列（索引2）：预测值
        start_date = pd.to_datetime('2024-01-01')
        end_date = pd.to_datetime('2024-12-31')
        filtered_data = data[(data[date_col] >= start_date) & (data[date_col] <= end_date) & 
                             (data[actual_col] >= 0) & (data[actual_col] <= 75) &
                             (data[actual_col].notna()) & (data[pred_col].notna())]

        # 检查过滤后的数据是否为空
        if len(filtered_data) == 0:
            print(f"警告: {file_path} 过滤后没有数据，跳过此文件")
            continue

        # 提取真实值和预测值
        y_actual = filtered_data[actual_col].values  # 第二列：实际值
        y_pred = filtered_data[pred_col].values      # 第三列：预测值

        # 计算评估指标
        r2 = r2_score(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

        # 获取模型名称（从文件名中提取）
        model_name = file_path.split('/')[4]

        # 将结果添加到列表中
        results.append({
            'Model': model_name,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        continue

# 将所有结果存储到一个DataFrame中
results_df = pd.DataFrame(results)

# 保存结果为CSV文件
results_df.to_csv('lower than 75.csv', index=False)

print("模型评估结果已保存为 'lower than 75.csv'")
