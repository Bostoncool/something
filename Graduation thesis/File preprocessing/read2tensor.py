import xarray as xr
import torch

# 1) 读文件
ds = xr.open_dataset('file.nc')         # 不读进内存
var = ds['your_var']                    # 取出变量，仍是延迟加载
values = var.values                     # 真正读进内存，得到 numpy.ndarray

# 2) 整理维度顺序
# 神经网络一般把“样本维度”放最前，假设原数据是 (lat, lon, time)
# 想按时间切片训练 → (time, lat, lon)
values = values.transpose(2, 0, 1)

# 3) 转张量
tensor = torch.from_numpy(values).float()   # float32
# 如需归一化/去缺失值
tensor = tensor.where(~torch.isnan(tensor), torch.tensor(0.0))  # 例：NaN→0

# 4) 构造 DataLoader（示例）
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(tensor)             # 可额外加入标签
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

"""
文件太大放不进内存：用 xarray.open_mfdataset('*.nc', chunks={'time': 100}) 
做分块延迟读取，再 torch.as_tensor(chunk.values) 逐块训练。

多变量输入：把多个变量 concat 成通道维，如 (batch, channel, lat, lon)。

自动填充值：NetCDF 的 _FillValue / missing_value 属性可通过 var.encoding['_FillValue'] 读出，
再用 tensor.masked_fill 统一处理。

如果是文件夹，用glob递归读取所有符合.nc格式的文件，再拼接到一起。
"""