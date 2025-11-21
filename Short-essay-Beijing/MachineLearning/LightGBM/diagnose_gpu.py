#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU配置诊断脚本
用于逐一排查LightGBM GPU支持问题
"""

import sys
import subprocess
import os

print("=" * 80)
print("LightGBM GPU配置诊断工具")
print("=" * 80)
print()

# ========== 步骤1: 检查LightGBM版本和GPU支持 ==========
print("【步骤1】检查LightGBM版本和GPU支持...")
print("-" * 80)

try:
    import lightgbm as lgb
    lgb_version = lgb.__version__
    print(f"✓ LightGBM版本: {lgb_version}")
    
    # 检查库路径
    try:
        import lightgbm.libpath as libpath_module
        lib_path = libpath_module.find_lib_path()[0]
        print(f"✓ LightGBM库路径: {lib_path}")
        
        # 检查是否有GPU相关的库文件
        lib_dir = os.path.dirname(lib_path)
        if os.path.exists(lib_dir):
            lib_files = os.listdir(lib_dir)
            gpu_lib_files = [f for f in lib_files if 'gpu' in f.lower() or 'cuda' in f.lower()]
            if gpu_lib_files:
                print(f"✓ 发现GPU相关库文件: {', '.join(gpu_lib_files[:5])}")
            else:
                print("⚠️  未发现明显的GPU相关库文件")
                print("   （这可能正常，取决于LightGBM的构建方式）")
    except Exception as e:
        print(f"⚠️  无法检查库文件: {e}")
        
except ImportError:
    print("❌ LightGBM未安装")
    print("\n解决方案: pip install lightgbm")
    sys.exit(1)
except Exception as e:
    print(f"❌ 检查LightGBM时出错: {e}")
    sys.exit(1)

# ========== 步骤2: 检查CUDA驱动 ==========
print("\n【步骤2】检查CUDA驱动（nvidia-smi）...")
print("-" * 80)

try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("✓ nvidia-smi命令执行成功")
        lines = result.stdout.split('\n')
        
        # 提取GPU信息
        gpu_count = 0
        cuda_version = None
        for i, line in enumerate(lines):
            if 'NVIDIA-SMI' in line:
                print(f"  {line}")
            elif 'Driver Version' in line:
                print(f"  {line}")
            elif 'CUDA Version' in line:
                cuda_version = line.strip()
                print(f"  {cuda_version}")
            elif '|' in line and ('GPU' in line or 'Name' in line):
                if gpu_count == 0:
                    print(f"  {line}")
                gpu_count += 1
            elif gpu_count > 0 and '|' in line and not line.strip().startswith('|'):
                break
        
        if gpu_count == 0:
            print("⚠️  未找到GPU设备")
    else:
        print(f"❌ nvidia-smi执行失败 (返回码: {result.returncode})")
        print(f"错误输出: {result.stderr}")
        print("\n解决方案: 安装NVIDIA驱动")
        print("  访问 https://www.nvidia.com/Download/index.aspx 下载并安装驱动")
        sys.exit(1)
except FileNotFoundError:
    print("❌ nvidia-smi命令未找到")
    print("\n解决方案: 安装NVIDIA驱动")
    print("  访问 https://www.nvidia.com/Download/index.aspx 下载并安装驱动")
    sys.exit(1)
except subprocess.TimeoutExpired:
    print("❌ nvidia-smi命令超时")
    print("\n解决方案: 检查GPU驱动是否正常")
    sys.exit(1)
except Exception as e:
    print(f"❌ 检查CUDA驱动时出错: {e}")
    sys.exit(1)

# ========== 步骤3: 检查CUDA工具包 ==========
print("\n【步骤3】检查CUDA工具包（nvcc）...")
print("-" * 80)

try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("✓ nvcc命令执行成功")
        lines = result.stdout.split('\n')
        cuda_version_line = [line for line in lines if 'release' in line.lower()]
        if cuda_version_line:
            print(f"  {cuda_version_line[0]}")
        else:
            print("  CUDA工具包已安装，但无法确定版本")
    else:
        print(f"⚠️  nvcc执行失败 (返回码: {result.returncode})")
        print("  注意: nvcc不是必需的，LightGBM GPU支持主要依赖CUDA运行时库")
except FileNotFoundError:
    print("⚠️  nvcc命令未找到")
    print("  注意: nvcc不是必需的，LightGBM GPU支持主要依赖CUDA运行时库")
    print("  如果GPU训练失败，可以尝试安装CUDA工具包:")
    print("    conda install -c nvidia cuda-toolkit")
except Exception as e:
    print(f"⚠️  检查CUDA工具包时出错: {e}")
    print("  这通常不影响LightGBM GPU支持")

# ========== 步骤4: 检查Python环境中的CUDA支持 ==========
print("\n【步骤4】检查Python环境中的CUDA支持...")
print("-" * 80)

# 检查PyTorch
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ PyTorch检测到CUDA支持")
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️  PyTorch未检测到CUDA支持")
except ImportError:
    print("⚠️  PyTorch未安装（不影响LightGBM GPU支持）")
except Exception as e:
    print(f"⚠️  检查PyTorch CUDA支持时出错: {e}")

# 检查TensorFlow
try:
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'):
        print(f"✓ TensorFlow检测到GPU设备")
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            print(f"  {gpu}")
    else:
        print("⚠️  TensorFlow未检测到GPU设备")
except ImportError:
    print("⚠️  TensorFlow未安装（不影响LightGBM GPU支持）")
except Exception as e:
    print(f"⚠️  检查TensorFlow GPU支持时出错: {e}")

# ========== 步骤5: 测试LightGBM GPU功能 ==========
print("\n【步骤5】测试LightGBM GPU功能...")
print("-" * 80)

try:
    import numpy as np
    
    print("  创建测试数据集...")
    test_X = np.random.rand(100, 10).astype(np.float32)
    test_y = np.random.rand(100).astype(np.float32)
    
    print("  创建LightGBM Dataset（GPU模式）...")
    gpu_params = {
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'max_bin': 63
    }
    test_data = lgb.Dataset(test_X, label=test_y, params=gpu_params)
    print("  ✓ Dataset创建成功")
    
    test_params = {
        'objective': 'regression',
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'num_gpu': 1,
        'max_bin': 63,
        'verbose': -1
    }
    
    print("  尝试使用GPU训练测试模型...")
    model = lgb.train(
        test_params, 
        test_data, 
        num_boost_round=1, 
        callbacks=[lgb.log_evaluation(period=0)]
    )
    print("  ✓ GPU训练测试成功！")
    
    print("\n" + "=" * 80)
    print("✓ 所有检查通过！GPU加速可用")
    print("=" * 80)
    print("  LightGBM可以正常使用GPU进行训练")
    print("=" * 80)
    
except Exception as e:
    error_msg = str(e).lower()
    print(f"  ❌ GPU训练测试失败")
    print(f"  错误类型: {type(e).__name__}")
    print(f"  错误详情: {e}")
    
    print("\n" + "=" * 80)
    print("❌ GPU功能测试失败")
    print("=" * 80)
    
    # 根据错误信息提供针对性建议
    if 'gpu' in error_msg or 'cuda' in error_msg or 'device' in error_msg:
        print("\n诊断结果: LightGBM无法使用GPU")
        print("\n可能的原因:")
        print("1. LightGBM安装的是CPU版本，不支持GPU")
        print("2. CUDA运行时库版本不匹配")
        print("3. GPU设备无法访问")
        
        print("\n解决方案:")
        print("1. 重新安装支持GPU的LightGBM:")
        print("   方法1（推荐，使用conda）:")
        print("     conda install -c conda-forge lightgbm")
        print("   方法2（使用pip + 环境变量）:")
        print("     pip uninstall lightgbm")
        print("     LIGHTGBM_GPU=1 pip install lightgbm")
        print("   方法3（从源码编译）:")
        print("     git clone --recursive https://github.com/microsoft/LightGBM")
        print("     cd LightGBM && mkdir build && cd build")
        print("     cmake -DUSE_GPU=1 .. && make -j4")
        print("     cd ../python-package && python setup.py install")
        
        print("\n2. 检查CUDA运行时库:")
        print("   - 确保CUDA运行时库版本与LightGBM兼容")
        print("   - 可以尝试: conda install -c nvidia cuda-runtime")
        
    elif 'platform' in error_msg or 'device_id' in error_msg:
        print("\n诊断结果: GPU设备ID或平台配置问题")
        print("\n解决方案:")
        print("1. 检查GPU是否被其他进程占用: nvidia-smi")
        print("2. 尝试修改gpu_platform_id和gpu_device_id参数")
        
    else:
        print("\n诊断结果: 未知错误")
        print("\n建议:")
        print("1. 查看完整错误信息以获取更多线索")
        print("2. 检查LightGBM版本: pip show lightgbm")
        print("3. 尝试重新安装支持GPU的LightGBM版本")
    
    print("\n" + "=" * 80)
    sys.exit(1)

print("\n诊断完成！")

