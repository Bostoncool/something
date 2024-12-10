from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import numpy as np

# 添加LSTM模型类定义
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
# 加载数据集
dataset = read_csv('pollution.csv', header=0, index_col=0)  # 读取CSV文件,第一行为列名,第一列为索引
values = dataset.values  # 将DataFrame转换为numpy数组

# 首先确保所有字符串数据被清理
for i in range(values.shape[1]):
    if isinstance(values[0,i], str):
        values[:,i] = [x.strip() if isinstance(x, str) else x for x in values[:,i]]

# 然后对风向列进行编码
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4].astype(str))

# 最后转换为float32
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())
# split into train and test sets
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# 将数据转换为PyTorch张量
train_X = torch.FloatTensor(train_X)
train_y = torch.FloatTensor(train_y)
test_X = torch.FloatTensor(test_X)
test_y = torch.FloatTensor(test_y)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义模型参数
input_dim = train_X.shape[2]  # 特征数量
hidden_dim = 50  # LSTM隐藏层维度
model = LSTMModel(input_dim, hidden_dim)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.L1Loss()  # MAE损失
optimizer = torch.optim.Adam(model.parameters())

# 训练模型
train_losses = []  # 训练集损失
val_losses = []  # 测试集损失
epochs = 100   # 训练次数
batch_size = 72  # 批次大小

for epoch in range(epochs):
    model.train()
    # 批次训练
    for i in range(0, len(train_X), batch_size):
        batch_X = train_X[i:i+batch_size].to(device)
        batch_y = train_y[i:i+batch_size].to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
    
    # 评估
    model.eval()
    with torch.no_grad():
        # 训练集损失
        train_outputs = model(train_X.to(device))
        train_loss = criterion(train_outputs.squeeze(), train_y.to(device))
        train_losses.append(train_loss.item())
        
        # 测试集损失
        test_outputs = model(test_X.to(device))
        val_loss = criterion(test_outputs.squeeze(), test_y.to(device))
        val_losses.append(val_loss.item())
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: train_loss={train_loss.item():.4f}, val_loss={val_loss.item():.4f}')

# 绘制损失曲线
pyplot.plot(train_losses, label='train')
pyplot.plot(val_losses, label='test')
pyplot.legend()
pyplot.show()

# 预测
model.eval()
with torch.no_grad():
    yhat = model(test_X.to(device)).cpu().numpy()

# 重塑数据
test_X = test_X.reshape(test_X.shape[0], test_X.shape[2]).numpy()

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.numpy().reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)