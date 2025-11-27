#!/bin/bash
# ARIMA时序分析依赖包安装脚本

echo "=========================================="
echo "ARIMA时序分析 - 依赖包安装脚本"
echo "=========================================="
echo ""

# 检查pip是否可用
if ! command -v pip &> /dev/null; then
    echo "❌ 错误: pip 未找到，请先安装 Python 和 pip"
    exit 1
fi

echo "正在安装核心依赖包..."
echo ""

# 升级pip
echo "1. 升级 pip..."
pip install --upgrade pip -q

# 安装核心包
echo "2. 安装 pandas, numpy..."
pip install pandas>=1.3.0 numpy>=1.20.0 -q

echo "3. 安装 matplotlib..."
pip install matplotlib>=3.3.0 -q

echo "4. 安装 statsmodels (时序分析核心库)..."
pip install statsmodels>=0.12.0 -q

echo "5. 安装 scipy..."
pip install scipy>=1.7.0 -q

echo "6. 安装 scikit-learn..."
pip install scikit-learn>=0.24.0 -q

echo "7. 安装 pmdarima (ARIMA自动优化)..."
pip install pmdarima>=1.8.0 -q

echo "8. 安装 tqdm (进度条)..."
pip install tqdm>=4.60.0 -q

echo ""
echo "=========================================="
echo "✅ 所有依赖包安装完成!"
echo "=========================================="
echo ""
echo "验证安装..."
python3 -c "
try:
    import statsmodels
    import scipy
    import pandas
    import numpy
    import matplotlib
    import sklearn
    print('✅ 所有核心包已正确安装')
    print(f'   statsmodels: {statsmodels.__version__}')
    print(f'   scipy: {scipy.__version__}')
    print(f'   pandas: {pandas.__version__}')
    print(f'   numpy: {numpy.__version__}')
except ImportError as e:
    print(f'❌ 导入错误: {e}')
    exit(1)
"

echo ""
echo "现在可以运行 ARIMA-CSV.py 了!"

