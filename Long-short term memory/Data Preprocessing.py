from pandas import read_csv
import pandas as pd

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

# 填充缺失值
dataset['pollution'].fillna(0, inplace=True)

# 删除前24小时的数据
dataset = dataset[24:]

# 查看前5行数据
print(dataset.head(5))

# 保存到文件
dataset.to_csv('pollution.csv')