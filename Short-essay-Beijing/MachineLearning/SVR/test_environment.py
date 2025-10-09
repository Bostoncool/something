"""
环境测试脚本
============
运行此脚本以验证所有依赖包是否正确安装
"""

import sys

def test_imports():
    """测试所有必要的包是否可以导入"""
    
    print("=" * 80)
    print("SVR PM2.5预测项目 - 环境测试")
    print("=" * 80)
    print(f"\nPython版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print("\n" + "-" * 80)
    print("检查依赖包...")
    print("-" * 80)
    
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'joblib': 'joblib'
    }
    
    results = {}
    
    for import_name, package_name in packages.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', '未知版本')
            results[package_name] = ('✓', version)
            print(f"✓ {package_name:20s} {version}")
        except ImportError as e:
            results[package_name] = ('✗', str(e))
            print(f"✗ {package_name:20s} 未安装")
    
    print("\n" + "-" * 80)
    print("测试结果汇总")
    print("-" * 80)
    
    success_count = sum(1 for status, _ in results.values() if status == '✓')
    total_count = len(results)
    
    print(f"\n成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 恭喜！所有依赖包已正确安装，环境配置完成！")
        print("\n您现在可以运行以下脚本:")
        print("  1. python SVR_PM25_Prediction.py  # 训练模型")
        print("  2. python predict_PM25.py         # 使用模型预测")
        return True
    else:
        print("\n⚠️ 警告：部分依赖包未安装")
        print("\n请运行以下命令安装缺失的包:")
        print("  pip install -r requirements.txt")
        
        missing_packages = [name for name, (status, _) in results.items() if status == '✗']
        print(f"\n缺失的包: {', '.join(missing_packages)}")
        return False

def test_data_paths():
    """测试数据路径是否存在"""
    import os
    
    print("\n" + "-" * 80)
    print("检查数据路径...")
    print("-" * 80)
    
    paths = {
        '污染物数据(all)': r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\all(AQI+PM2.5+PM10)',
        '污染物数据(extra)': r'C:\Users\IU\Desktop\Datebase Origin\Benchmark\extra(SO2+NO2+CO+O3)',
        'ERA5气象数据': r'C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-CSV'
    }
    
    all_exist = True
    
    for name, path in paths.items():
        exists = os.path.exists(path)
        status = '✓' if exists else '✗'
        print(f"{status} {name:20s}: {path}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✓ 所有数据路径都存在")
    else:
        print("\n⚠️ 部分数据路径不存在，请检查路径是否正确")
        print("   如果路径不同，请修改SVR_PM25_Prediction.py中的路径设置")
    
    return all_exist

def test_simple_prediction():
    """测试简单的预测功能"""
    
    print("\n" + "-" * 80)
    print("测试基本功能...")
    print("-" * 80)
    
    try:
        import numpy as np
        import pandas as pd
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        
        # 创建简单的测试数据
        print("\n创建测试数据...")
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        # 训练简单的SVR模型
        print("训练简单的SVR模型...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_scaled, y)
        
        # 预测
        print("进行预测...")
        y_pred = model.predict(X_scaled[:10])
        
        print(f"✓ 预测成功！预测了{len(y_pred)}个样本")
        print(f"  预测值示例: {y_pred[:3]}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def main():
    """主函数"""
    
    # 测试导入
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n" + "=" * 80)
        print("请先安装缺失的依赖包，然后重新运行此测试脚本")
        print("=" * 80)
        return
    
    # 测试数据路径
    paths_ok = test_data_paths()
    
    # 测试基本功能
    function_ok = test_simple_prediction()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"依赖包安装: {'✓ 通过' if imports_ok else '✗ 失败'}")
    print(f"数据路径检查: {'✓ 通过' if paths_ok else '⚠️ 警告'}")
    print(f"基本功能测试: {'✓ 通过' if function_ok else '✗ 失败'}")
    
    if imports_ok and function_ok:
        print("\n🎉 环境配置完成！您可以开始使用SVR预测模型了。")
        
        if not paths_ok:
            print("\n⚠️ 注意: 数据路径需要调整")
            print("   请在SVR_PM25_Prediction.py中修改数据路径设置")
    else:
        print("\n❌ 环境配置存在问题，请解决上述问题后重试")
    
    print("=" * 80)

if __name__ == '__main__':
    main()

