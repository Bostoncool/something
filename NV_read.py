import netCDF4 as nc
import pandas as pd
import numpy as np
from pathlib import Path
import logging

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def read_nc_file(file_path):
    """
    读取NC文件并返回数据集
    
    Parameters:
        file_path (str): NC文件的路径
    
    Returns:
        netCDF4.Dataset: NC数据集对象
    """
    try:
        return nc.Dataset(file_path, 'r')
    except Exception as e:
        logging.error(f"读取NC文件时出错: {str(e)}")
        raise

def nc_to_dataframe(nc_dataset):
    """
    将NC数据集转换为pandas DataFrame，只提取PM2.5数据
    """
    try:
        # 首先打印数据集信息
        logging.info("NC文件变量信息:")
        for var_name, var in nc_dataset.variables.items():
            logging.info(f"变量名: {var_name}")
            logging.info(f"维度: {var.dimensions}")
            logging.info(f"形状: {var.shape}")
            logging.info("------------------------")
        
        # 检查是否存在PM2.5变量
        pm25_names = ['PM25', 'pm25', 'PM2.5', 'pm2.5', 'PM2_5', 'pm2_5', 
                     'PM25_concentration', 'PM25_mass', 'PM25_surface']
        pm25_var = None
        
        # 查找PM2.5变量
        for name in pm25_names:
            if name in nc_dataset.variables:
                pm25_var = nc_dataset.variables[name]
                pm25_name = name
                break
        
        if pm25_var is None:
            raise ValueError("未找到PM2.5变量，请确认文件中包含PM2.5相关变量")
        
        # 提取PM2.5数据
        pm25_data = pm25_var[:]
        
        # 如果数据是多维数组，需要进行展平处理
        if pm25_data.ndim > 1:
            logging.info(f"PM2.5数据维度为{pm25_data.ndim}维，进行展平处理")
            pm25_data = pm25_data.flatten()
        
        # 创建DataFrame
        df = pd.DataFrame({pm25_name: pm25_data})
        
        # 添加基本的数据描述
        logging.info("\nPM2.5数据信息:")
        logging.info(f"数据范围: {pm25_data.min():.2f} 到 {pm25_data.max():.2f}")
        logging.info(f"数据点数量: {len(pm25_data)}")
        logging.info(f"平均值: {pm25_data.mean():.2f}")
        logging.info(f"标准差: {pm25_data.std():.2f}")
        
        # 检查是否有单位信息
        if hasattr(pm25_var, 'units'):
            logging.info(f"数据单位: {pm25_var.units}")
        
        # 检查异常值
        logging.info("\n数据质量检查:")
        logging.info(f"缺失值数量: {np.isnan(pm25_data).sum()}")
        logging.info(f"负值数量: {(pm25_data < 0).sum() if isinstance(pm25_data, np.ndarray) else 0}")
        
        return df
    
    except Exception as e:
        logging.error(f"转换数据时出错: {str(e)}")
        raise

def save_to_csv(df, output_path):
    """
    将DataFrame保存为CSV文件
    
    Parameters:
        df (pd.DataFrame): 要保存的数据框
        output_path (str): 输出CSV文件的路径
    """
    try:
        df.to_csv(output_path, index=False, encoding='utf-8')
        logging.info(f"数据已成功保存到: {output_path}")
    except Exception as e:
        logging.error(f"保存CSV文件时出错: {str(e)}")
        raise

def main():
    """主函数"""
    setup_logging()
    
    try:
        # 获取用户输入的文件路径并处理路径格式
        input_path = input("请输入NC文件的路径: ").strip().replace('"', '').replace("'", '')
        
        # 转换为Path对象并解析绝对路径
        file_path = Path(input_path).resolve()
        
        # 检查文件是否存在
        if not file_path.exists():
            logging.error(f"找不到文件: {file_path}")
            logging.info("请确保：")
            logging.info("1. 文件路径中的斜杠方向是否正确（建议使用/或\\\\）")
            logging.info("2. 文件扩展名是否为.nc")
            logging.info("3. 文件确实存在于指定位置")
            return
        
        # 生成输出文件路径
        output_path = str(file_path.with_suffix('.csv'))
        
        # 读取NC文件
        logging.info(f"正在读取NC文件: {file_path}")
        nc_dataset = read_nc_file(str(file_path))
        
        # 转换为DataFrame
        logging.info("正在转换数据...")
        df = nc_to_dataframe(nc_dataset)
        
        # 显示数据基本信息
        logging.info("\n数据预览:")
        print(df.head())
        print("\n数据信息:")
        print(df.info())
        
        # 保存为CSV
        logging.info("正在保存为CSV文件...")
        save_to_csv(df, output_path)
        
        # 关闭NC文件
        nc_dataset.close()
        
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        return
    
if __name__ == "__main__":
    main()
