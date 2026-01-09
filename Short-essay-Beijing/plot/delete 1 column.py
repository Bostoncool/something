import pandas as pd

# 加载CSV文件
file_path = "H:/DATA Science/小论文Result/Fine_model/-LightGBM/CSV2/plot_ts_lastyear_sampled__lightgbm_optimized.csv"
data = pd.read_csv(file_path)

# 删除第一列
data = data.drop(data.columns[0], axis=1)

# 将修改后的数据保存到原文件路径
data.to_csv(file_path, index=False)
