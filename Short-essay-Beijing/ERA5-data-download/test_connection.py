import cdsapi
import time
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_cds_connection():
    """
    测试CDS API连接
    """
    try:
        logging.info("正在测试CDS API连接...")
        
        # 创建客户端
        c = cdsapi.Client()
        
        # 构建一个最小的测试请求
        test_request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': 'surface_pressure',
            'year': '2024',
            'month': '01',
            'day': ['01', '02'],  # 只测试2天
            'time': ['00:00', '12:00'],  # 只测试2个时间点
            'area': [41, 115, 39, 117]  # 北京小区域
        }
        
        logging.info("提交测试请求...")
        
        # 提交请求
        r = c.retrieve('reanalysis-era5-single-levels', test_request)
        
        logging.info("请求提交成功！")
        logging.info(f"下载URL: {r.location}")
        
        # 尝试下载一个小文件
        logging.info("开始下载测试文件...")
        r.download('test_connection.nc')
        
        logging.info("✅ 连接测试成功！文件已下载为 test_connection.nc")
        return True
        
    except Exception as e:
        logging.error(f"❌ 连接测试失败: {e}")
        return False

def check_cds_config():
    """
    检查CDS配置文件
    """
    import os
    
    # 检查配置文件路径
    config_paths = [
        os.path.expanduser('~/.cdsapirc'),
        os.path.expanduser('~/cdsapirc'),
        '.cdsapirc'
    ]
    
    logging.info("检查CDS配置文件...")
    
    for path in config_paths:
        if os.path.exists(path):
            logging.info(f"✅ 找到配置文件: {path}")
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    if 'url' in content and 'key' in content:
                        logging.info("✅ 配置文件格式正确")
                        return True
                    else:
                        logging.warning("⚠️ 配置文件可能不完整")
            except Exception as e:
                logging.error(f"❌ 读取配置文件失败: {e}")
        else:
            logging.info(f"❌ 未找到配置文件: {path}")
    
    logging.error("❌ 未找到有效的CDS配置文件")
    logging.info("请按照以下步骤配置CDS API:")
    logging.info("1. 访问 https://cds.climate.copernicus.eu/")
    logging.info("2. 注册账户并登录")
    logging.info("3. 在个人资料页面获取API密钥")
    logging.info("4. 创建 ~/.cdsapirc 文件并添加配置")
    return False

if __name__ == '__main__':
    logging.info("=== CDS API 连接测试 ===")
    
    # 检查配置
    if check_cds_config():
        # 测试连接
        if test_cds_connection():
            logging.info("🎉 所有测试通过！可以开始下载数据")
        else:
            logging.error("💥 连接测试失败，请检查网络和配置")
    else:
        logging.error("💥 配置检查失败，请先配置CDS API")