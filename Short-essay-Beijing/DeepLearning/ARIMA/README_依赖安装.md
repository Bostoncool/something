# ARIMA时序分析 - 依赖包安装说明

## 📦 所需依赖包

本ARIMA时序分析脚本需要以下Python包：

### 核心必需包
- **statsmodels** (>=0.12.0) - 时序分析核心库，提供SARIMAX模型
- **scipy** (>=1.7.0) - 科学计算库，用于统计检验
- **pandas** (>=1.3.0) - 数据处理
- **numpy** (>=1.20.0) - 数值计算
- **matplotlib** (>=3.3.0) - 数据可视化
- **scikit-learn** (>=0.24.0) - 评估指标

### 可选但推荐包
- **pmdarima** (>=1.8.0) - ARIMA自动参数优化
- **tqdm** (>=4.60.0) - 进度条显示

## 🚀 安装方法

### 方法1: 使用requirements.txt (推荐)

```bash
cd ARIMA
pip install -r requirements.txt
```

### 方法2: 使用安装脚本

```bash
cd ARIMA
chmod +x install_dependencies.sh
./install_dependencies.sh
```

### 方法3: 手动安装

```bash
# 安装核心包
pip install statsmodels>=0.12.0 scipy>=1.7.0 pandas>=1.3.0 numpy>=1.20.0 matplotlib>=3.3.0 scikit-learn>=0.24.0

# 安装可选包
pip install pmdarima>=1.8.0 tqdm>=4.60.0
```

## ✅ 验证安装

运行以下命令验证包是否正确安装：

```python
python3 -c "
import statsmodels
import scipy
import pandas
import numpy
print('✅ 所有核心包已正确安装')
print(f'statsmodels: {statsmodels.__version__}')
print(f'scipy: {scipy.__version__}')
"
```

## 🔧 常见问题解决

### 问题1: ImportError: No module named 'statsmodels'

**解决方案:**
```bash
pip install statsmodels
```

### 问题2: statsmodels版本过旧，缺少某些功能

**解决方案:**
```bash
pip install --upgrade statsmodels
```

### 问题3: scipy导入错误

**解决方案:**
```bash
pip install --upgrade scipy
```

### 问题4: pmdarima安装失败

**解决方案:**
```bash
# 先安装编译依赖
pip install numpy scipy scikit-learn statsmodels

# 再安装pmdarima
pip install pmdarima
```

### 问题5: 在conda环境中安装

```bash
conda install -c conda-forge statsmodels scipy pandas numpy matplotlib scikit-learn
pip install pmdarima tqdm
```

## 📝 代码改进说明

代码已添加以下改进：

1. **错误处理**: 所有时序包导入都添加了try-except错误处理
2. **友好提示**: 当包缺失时，会显示清晰的错误信息和安装指导
3. **依赖检查**: 在程序启动时检查关键包是否可用

## 🎯 使用说明

安装完所有依赖后，直接运行：

```bash
python ARIMA-CSV.py
```

如果缺少任何包，程序会显示详细的错误信息和安装指导。

