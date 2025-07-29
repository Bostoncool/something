import cdsapi
import calendar
from subprocess import call
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- 1. 通用配置 ----------
MAX_WORKERS = 2        # 并行线程数（≤4 比较稳）
IDM_PATH = r"C:\Program Files (x86)\Internet Download Manager\IDMan.exe"
SAVE_DIR = r"D:\2m_dewpoint_temperature"

# ---------- 2. 打包 + 下载 ----------
def task_worker(year, month):
    """
    单线程任务：提交打包 → 拿到 url → 推给 IDM
    """
    c = cdsapi.Client()
    req = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': '2m_temperature',
        'year': str(year),
        'month': str(month).zfill(2),
        'day': [str(d).zfill(2) for d in range(1, calendar.monthrange(year, month)[1] + 1)],
        'time': [f"{h:02d}:00" for h in range(24)],
        'area': [41, 115, 39, 117]          # 北京小区域，可改
    }

    try:
        r = c.retrieve('reanalysis-era5-single-levels', req)
        url = r.location
        fname = f"{year}{month:02d}.nc"
        # 推给 IDM
        call([IDM_PATH, '/d', url, '/p', SAVE_DIR, '/f', fname, '/a'])
        call([IDM_PATH, '/s'])
        print(f"[OK] {fname} 已加入 IDM 队列")
        return True
    except Exception as e:
        print(f"[ERROR] {year}-{month:02d}: {e}")
        return False

# ---------- 3. 主程序 ----------
if __name__ == '__main__':
    years  = range(2015, 2025)
    months = range(1, 13)

    # 生成任务列表
    tasks = [(y, m) for y in years for m in months]

    with ThreadPoolExecutor(max_workers = MAX_WORKERS) as pool:
        futures = [pool.submit(task_worker, y, m) for y, m in tasks]
        for f in as_completed(futures):
            f.result()          # 这里只是为了把异常抛出来