import netCDF4 as nc
 
def inspect_nc_file(nc_file_path):
    """
    详细查看NC文件结构
    """
    try:
        with nc.Dataset(nc_file_path, 'r') as nc_file:
            print("=" * 50)
            print("NC文件基本信息:")
            print("=" * 50)
            
            # 文件格式和维度
            print(f"文件格式: {nc_file.data_model}")
            print(f"文件维度: {nc_file.dimensions}")
            print(f"文件变量: {list(nc_file.variables.keys())}")
            
            print("\n" + "=" * 50)
            print("维度详细信息:")
            print("=" * 50)
            for dim_name, dim in nc_file.dimensions.items():
                print(f"{dim_name}: {len(dim)} 个元素")
            
            print("\n" + "=" * 50)
            print("变量详细信息:")
            print("=" * 50)
            for var_name, var in nc_file.variables.items():
                print(f"\n变量名: {var_name}")
                print(f"  维度: {var.dimensions}")
                print(f"  形状: {var.shape}")
                print(f"  数据类型: {var.dtype}")
                print(f"  属性: {dict(var.__dict__)}")
                
                # 显示部分数据样本
                if len(var.shape) <= 2:  # 只显示低维数据的样本
                    sample_data = var[:]
                    if hasattr(sample_data, 'flatten'):
                        sample_data = sample_data.flatten()
                    print(f"  数据样本: {sample_data[:5]}...")  # 显示前5个值
            
            print("\n" + "=" * 50)
            print("全局属性:")
            print("=" * 50)
            for attr_name in nc_file.ncattrs():
                print(f"{attr_name}: {getattr(nc_file, attr_name)}")
                
    except Exception as e:
        print(f"错误: {e}")
 
# 使用示例
inspect_nc_file(r"C:\Users\IU\Desktop\Datebase Origin\ERA5-Beijing-NC\2m_dewpoint_temperature\201501.nc")