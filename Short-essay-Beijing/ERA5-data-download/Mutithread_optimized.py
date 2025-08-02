import cdsapi
import calendar
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# ---------- 1. 配置日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('era5_download.log'),
        logging.StreamHandler()
    ]
)

# ---------- 2. 通用配置 ----------
MAX_WORKERS = 1        # 降低并发数，避免429错误
RETRY_ATTEMPTS = 3     # 重试次数
RETRY_DELAY = 300      # 重试间隔（秒）
REQUEST_DELAY = 60     # 请求间隔（秒）
SAVE_DIR = r"F:\surface_pressure"

# 确保保存目录存在
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 3. 优化的下载函数 ----------
def task_worker(year, month, attempt=1):
    """
    优化的单线程任务：添加重试机制和错误处理
    """
    c = cdsapi.Client()
    
    # 构建请求参数
    req = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': 'surface_pressure',
        'year': str(year),
        'month': str(month).zfill(2),
        'day': [str(d).zfill(2) for d in range(1, calendar.monthrange(year, month)[1] + 1)],
        'time': [f"{h:02d}:00" for h in range(0, 24, 6)],  # 减少到每6小时一次
        'area': [41, 115, 39, 117]          # 北京小区域
    }

    try:
        logging.info(f"开始下载 {year}-{month:02d} (尝试 {attempt}/{RETRY_ATTEMPTS})")
        
        # 添加请求延迟
        if attempt > 1:
            time.sleep(RETRY_DELAY)
        
        r = c.retrieve('reanalysis-era5-single-levels', req)
        url = r.location
        fname = f"{year}{month:02d}.nc"
        filepath = os.path.join(SAVE_DIR, fname)
        
        # 直接下载文件，不使用IDM
        logging.info(f"下载文件: {fname}")
        r.download(filepath)
        
        logging.info(f"[成功] {fname} 下载完成")
        return True
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"[错误] {year}-{month:02d}: {error_msg}")
        
        # 处理特定错误类型
        if "429" in error_msg or "Too Many Requests" in error_msg:
            if attempt < RETRY_ATTEMPTS:
                logging.warning(f"遇到限流错误，等待 {RETRY_DELAY} 秒后重试...")
                time.sleep(RETRY_DELAY)
                return task_worker(year, month, attempt + 1)
            else:
                logging.error(f"达到最大重试次数，跳过 {year}-{month:02d}")
                return False
                
        elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            if attempt < RETRY_ATTEMPTS:
                logging.warning(f"连接超时，等待 {RETRY_DELAY} 秒后重试...")
                time.sleep(RETRY_DELAY)
                return task_worker(year, month, attempt + 1)
            else:
                logging.error(f"连接问题，跳过 {year}-{month:02d}")
                return False
        else:
            logging.error(f"未知错误，跳过 {year}-{month:02d}")
            return False

# ---------- 4. 主程序 ----------
if __name__ == '__main__':
    # 测试用的较小数据集
    years = range(2024, 2025)  # 只下载2024年
    months = range(1, 3)       # 只下载1-2月
    
    # 生成任务列表
    tasks = [(y, m) for y in years for m in months]
    
    logging.info(f"开始下载任务，共 {len(tasks)} 个文件")
    logging.info(f"保存目录: {SAVE_DIR}")
    
    success_count = 0
    total_count = len(tasks)
    
    # 使用单线程避免并发问题
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(task_worker, y, m) for y, m in tasks]
        
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                if result:
                    success_count += 1
                
                # 添加请求间隔
                if i < total_count:
                    logging.info(f"等待 {REQUEST_DELAY} 秒后继续下一个请求...")
                    time.sleep(REQUEST_DELAY)
                    
            except Exception as e:
                logging.error(f"任务执行异常: {e}")
    
    logging.info(f"下载完成！成功: {success_count}/{total_count}")