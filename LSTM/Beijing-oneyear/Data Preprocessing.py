from pandas import read_csv
import pandas as pd
import numpy as np

# 首先读取数据，不进行日期解析
dataset = read_csv(r'C:\Users\IU\Desktop\something\Long-short term memory\PRSA_Data\PRSA_data_2010.1.1-2014.12.31.csv')

# 读取数据后立即打印列名
print("数据集的列名:", dataset.columns.tolist())

# 合并年月日时为日期时间列
dataset['date'] = pd.to_datetime(dataset[['year', 'month', 'day', 'hour']].astype(str).agg('-'.join, axis=1), 
                                format='%Y-%m-%d-%H')

# 设置日期列为索引
dataset.set_index('date', inplace=True)

# 删除不需要的列
dataset.drop(['No', 'year', 'month', 'day', 'hour'], axis=1, inplace=True)

# 设置列名
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']

# 检查缺失值的情况
print("缺失值统计：")
print(dataset.isna().sum())

# 检查是否存在字符串形式的'NA'
print("\n'NA'字符串的数量：")
print((dataset == 'NA').sum())

# 更全面的缺失值处理
dataset['pollution'] = dataset['pollution'].replace('NA', np.nan)  # 将字符串'NA'转换为np.nan
dataset['pollution'] = pd.to_numeric(dataset['pollution'], errors='coerce')  # 确保数据类型为数值型
dataset['pollution'].fillna(0, inplace=True)  # 填充缺失值

# 再次检查缺失值情况
print("\n处理后的缺失值统计：")
print(dataset.isna().sum())

# 删除前24小时的数据
dataset = dataset[24:]

# 查看前5行数据
print(dataset.head(5))

# 保存到文件
dataset.to_csv('pollution.csv')