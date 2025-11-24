import os
import sys
import cdsapi
import calendar
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from subprocess import call

# --------------------  1. Global Configuration  --------------------
MAX_WORKERS   = 3               # Number of parallel threads, CDS official recommendation ≤6
IDM_PATH      = r"C:\Program Files (x86)\Internet Download Manager\IDMan.exe"
ROOT_DIR      = Path(r"F:\ERA5_PM25_PM10")     # Root directory
ERA5_DATASET  = "reanalysis-era5-single-levels"
ERA5PL_DATASET= "reanalysis-era5-pressure-levels"
LAND_DATASET  = "reanalysis-era5-land"

# ERA5 single-level variable list
SINGLE_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "mean_sea_level_pressure",
    "total_precipitation",
    "boundary_layer_height",
    "total_cloud_cover",
    "surface_solar_radiation_downwards",
    "surface_net_solar_radiation",
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
    "evaporation",
    "convective_available_potential_energy",
    "convective_inhibition",
    "instantaneous_10m_wind_gust",
    "forecast_albedo",
    "soil_temperature_level_1",
    "volumetric_soil_water_layer_1",
    "snow_depth",
    "skin_temperature",
    "total_column_water_vapour"
]

# ERA5 pressure-level variable list (pressure level list)
PRESSURE_LEVELS = ["1000", "850", "700", "500"]
PRESS_VARS = ["u_component_of_wind",
              "v_component_of_wind",
              "temperature",
              "relative_humidity",
              "geopotential",
              "specific_humidity",
              "vertical_velocity"]

# ERA5-Land specific variables (higher resolution 0.1°)
LAND_VARS = [
    "leaf_area_index_high_vegetation",
    "potential_evaporation",
    "runoff",
    "snow_albedo",
    "evaporation_from_vegetation_transpiration"
]

# Download time range
YEARS  = list(range(2000, 2025))     # 2000-2024
AREA   = [54, 73, 4, 135]          # Beijing region example, can be changed to China region

# --------------------  2. Utility Functions  --------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def build_request(dataset, variable, year, month, level=None):
    """Build cdsapi request dictionary based on dataset type"""
    base = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "year": str(year),
        "month": str(month).zfill(2),
        "day": [str(d).zfill(2) for d in range(1, calendar.monthrange(year, month)[1] + 1)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA
    }
    if dataset == ERA5_DATASET:
        base["variable"] = variable
    elif dataset == ERA5PL_DATASET:
        base["variable"] = variable
        base["pressure_level"] = level
    elif dataset == LAND_DATASET:
        base["variable"] = variable
    return base

def download_one(dataset, variable, year, month, level=None):
    """Single-threaded task: submit cdsapi → get url → push to IDM"""
    c = cdsapi.Client()
    req_dict = build_request(dataset, variable, year, month, level)

    # Determine folder path
    if dataset == ERA5_DATASET:
        folder = ensure_dir(ROOT_DIR / "ERA5_single" / variable)
    elif dataset == ERA5PL_DATASET:
        folder = ensure_dir(ROOT_DIR / "ERA5_pressure" / f"{variable}_{level}hPa")
    elif dataset == LAND_DATASET:
        folder = ensure_dir(ROOT_DIR / "ERA5_Land" / variable)

    fname = f"{variable}_{year}{month:02d}.nc"
    local_file = folder / fname

    # Skip if already exists
    if local_file.exists():
        print(f"[SKIP] {local_file} already exists")
        return True

    try:
        r = c.retrieve(dataset, req_dict)
        url = r.location
        call([IDM_PATH, "/d", url, "/p", str(folder), "/f", fname, "/a"])
        call([IDM_PATH, "/s"])
        print(f"[OK] {fname} added to IDM queue")
        return True
    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
        return False

# --------------------  3. Main Program: Generate Task Pool  --------------------
def generate_tasks():
    tasks = []
    for year in YEARS:
        for month in range(1, 13):
            # 1) ERA5 single-level
            for var in SINGLE_VARS:
                tasks.append((ERA5_DATASET, var, year, month, None))
            # 2) ERA5 pressure-level
            for var in PRESS_VARS:
                for lev in PRESSURE_LEVELS:
                    tasks.append((ERA5PL_DATASET, var, year, month, lev))
            # 3) ERA5-Land
            for var in LAND_VARS:
                tasks.append((LAND_DATASET, var, year, month, None))
    return tasks

# --------------------  4. Start Concurrent Downloads  --------------------
if __name__ == "__main__":
    if not ROOT_DIR.exists():
        ROOT_DIR.mkdir(parents=True)

    all_tasks = generate_tasks()
    print(f"Total {len(all_tasks)} tasks, ready to submit...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(download_one, *t) for t in all_tasks]
        for f in as_completed(futures):
            f.result()          # Catch exceptions