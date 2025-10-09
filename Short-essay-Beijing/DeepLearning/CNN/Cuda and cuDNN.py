import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA is available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")

# 三种Python的版本，需要不同的方式进行调度，好在是完成了。
# 最开始的时候是需要手动添加path，然后进行安装的。
