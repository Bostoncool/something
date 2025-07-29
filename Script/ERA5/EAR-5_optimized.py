import cdsapi
import time
from datetime import datetime

def create_client():
    """创建CDS客户端"""
    return cdsapi.Client()

def download_small_dataset():
    """方案1：下载小数据集（推荐用于测试）"""
    print("=== 方案1：下载小数据集 ===")
    
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "2m_temperature",
            "total_precipitation"
        ],
        "year": ["2024"],
        "month": ["01", "02"],  # 只下载1-2月
        "day": ["01", "15"],    # 只下载每月1号和15号
        "time": ["00:00", "12:00"],  # 每天2次
        "data_format": "netcdf",
        "area": [41, 115, 39, 117]
    }
    
    target = 'download_small_test.nc'
    client = create_client()
    
    try:
        print("开始下载小数据集...")
        client.retrieve("reanalysis-era5-single-levels", request, target)
        print(f"下载完成：{target}")
        return True
    except Exception as e:
        print(f"下载失败：{e}")
        return False

def download_by_year():
    """方案2：按年份分批下载"""
    print("=== 方案2：按年份分批下载 ===")
    
    years = ["2020", "2021", "2022", "2023", "2024"]
    base_request = {
        "product_type": ["reanalysis"],
        "variable": [
            "2m_temperature",
            "mean_sea_level_pressure",
            "total_precipitation"
        ],
        "month": [
            "01", "02", "03", "04", "05", "06",
            "07", "08", "09", "10", "11", "12"
        ],
        "day": [
            "01", "02", "03", "04", "05", "06",
            "07", "08", "09", "10", "11", "12",
            "13", "14", "15", "16", "17", "18",
            "19", "20", "21", "22", "23", "24",
            "25", "26", "27", "28", "29", "30",
            "31"
        ],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "area": [41, 115, 39, 117]
    }
    
    client = create_client()
    
    for year in years:
        request = base_request.copy()
        request["year"] = [year]
        target = f'download_{year}.nc'
        
        try:
            print(f"正在下载 {year} 年数据...")
            client.retrieve("reanalysis-era5-single-levels", request, target)
            print(f"下载完成：{target}")
            time.sleep(5)  # 避免请求过于频繁
        except Exception as e:
            print(f"下载 {year} 年数据失败：{e}")
            continue

def download_by_month():
    """方案3：按月下载（更高时间分辨率）"""
    print("=== 方案3：按月下载 ===")
    
    months = ["01", "02", "03", "04", "05", "06", 
              "07", "08", "09", "10", "11", "12"]
    
    base_request = {
        "product_type": ["reanalysis"],
        "variable": [
            "2m_temperature",
            "mean_sea_level_pressure",
            "total_precipitation"
        ],
        "year": ["2024"],
        "day": [
            "01", "02", "03", "04", "05", "06",
            "07", "08", "09", "10", "11", "12",
            "13", "14", "15", "16", "17", "18",
            "19", "20", "21", "22", "23", "24",
            "25", "26", "27", "28", "29", "30",
            "31"
        ],
        "time": [
            "00:00", "01:00", "02:00", "03:00",
            "04:00", "05:00", "06:00", "07:00",
            "08:00", "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00", "15:00",
            "16:00", "17:00", "18:00", "19:00",
            "20:00", "21:00", "22:00", "23:00"
        ],
        "data_format": "netcdf",
        "area": [41, 115, 39, 117]
    }
    
    client = create_client()
    
    for month in months:
        request = base_request.copy()
        request["month"] = [month]
        target = f'download_2024_{month}.nc'
        
        try:
            print(f"正在下载 2024年{month}月 数据...")
            client.retrieve("reanalysis-era5-single-levels", request, target)
            print(f"下载完成：{target}")
            time.sleep(3)  # 避免请求过于频繁
        except Exception as e:
            print(f"下载 2024年{month}月 数据失败：{e}")
            continue

def download_custom():
    """方案4：自定义下载参数"""
    print("=== 方案4：自定义下载参数 ===")
    
    # 用户可以根据需要修改这些参数
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "2m_temperature",
            "mean_sea_level_pressure",
            "total_precipitation"
        ],
        "year": ["2024"],
        "month": ["01", "02", "03"],  # 可以修改月份
        "day": ["01", "15"],          # 可以修改日期
        "time": ["00:00", "12:00"],   # 可以修改时间
        "data_format": "netcdf",
        "area": [41, 115, 39, 117]
    }
    
    target = 'download_custom.nc'
    client = create_client()
    
    try:
        print("开始下载自定义数据集...")
        client.retrieve("reanalysis-era5-single-levels", request, target)
        print(f"下载完成：{target}")
        return True
    except Exception as e:
        print(f"下载失败：{e}")
        return False

def main():
    """主函数：选择下载方案"""
    print("ERA5数据下载工具")
    print("=" * 50)
    print("请选择下载方案：")
    print("1. 小数据集（用于测试）")
    print("2. 按年份分批下载")
    print("3. 按月下载（高时间分辨率）")
    print("4. 自定义参数下载")
    
    choice = input("请输入选择（1-4）：").strip()
    
    if choice == "1":
        download_small_dataset()
    elif choice == "2":
        download_by_year()
    elif choice == "3":
        download_by_month()
    elif choice == "4":
        download_custom()
    else:
        print("无效选择，默认使用方案1")
        download_small_dataset()

if __name__ == "__main__":
    main()