import os
import glob
import multiprocessing as mp
import pandas as pd
import netCDF4 as nc
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class NetCDFToCSVConverter:
    def __init__(self, input_folder, output_folder, num_processes=None):
        """
        Initialize NetCDF to CSV converter
        
        Args:
            input_folder: Input folder path
            output_folder: Output folder path
            num_processes: Number of processes, defaults to CPU core count
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.num_processes = num_processes or mp.cpu_count()
        
        # Create output folder
        os.makedirs(self.output_folder, exist_ok=True)
        
    def process_single_file(self, nc_file_path):
        """
        Process a single NetCDF file
        
        Args:
            nc_file_path: NetCDF file path
            
        Returns:
            tuple: (input file path, output file path, status, error message)
        """
        try:
            # Get filename without extension
            file_name = os.path.splitext(os.path.basename(nc_file_path))[0]
            output_file_path = os.path.join(self.output_folder, f"{file_name}.csv")
            
            # Read NetCDF file
            with nc.Dataset(nc_file_path, 'r') as dataset:
                # Check if required variables exist
                if 'PM2.5' not in dataset.variables:
                    return nc_file_path, output_file_path, "failed", "Missing PM2.5 variable"
                
                # Get dimension information
                lat_dim = dataset.dimensions['lat']
                lon_dim = dataset.dimensions['lon']
                
                # Get variable data
                lat_data = dataset.variables['lat'][:]
                lon_data = dataset.variables['lon'][:]
                pm25_data = dataset.variables['PM2.5'][:]
                
                # Get PM2.5 attributes
                pm25_attrs = dataset.variables['PM2.5'].__dict__
                fill_value = pm25_attrs.get('_FillValue', 65535)
                scale_factor = pm25_attrs.get('scale_factor', 0.1)
                add_offset = pm25_attrs.get('add_offset', 0.0)
                units = pm25_attrs.get('units', 'µg/m3')
            
            # Handle missing values and data scaling
            pm25_data = pm25_data.astype(np.float32)
            pm25_data[pm25_data == fill_value] = np.nan
            pm25_data = pm25_data * scale_factor + add_offset
            
            # Create grid
            lon_grid, lat_grid = np.meshgrid(lon_data, lat_data)
            
            # Flatten data and create DataFrame
            df_data = {
                'latitude': lat_grid.flatten(),
                'longitude': lon_grid.flatten(),
                'PM25': pm25_data.flatten()
            }
            
            df = pd.DataFrame(df_data)
            
            # Remove missing values to reduce file size
            df = df.dropna(subset=['PM25'])
            
            # Add unit information as comment
            metadata_comment = f"# Units: PM25 = {units}\n"
            metadata_comment += f"# Data scaled with: value = raw * {scale_factor} + {add_offset}\n"
            metadata_comment += f"# Original dimensions: lat={len(lat_data)}, lon={len(lon_data)}\n"
            metadata_comment += f"# Processed data points: {len(df)}\n"
            
            # Write CSV file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(metadata_comment)
                df.to_csv(f, index=False)
            
            return nc_file_path, output_file_path, "success", None
            
        except Exception as e:
            return nc_file_path, "", "failed", str(e)
    
    def batch_convert(self, file_pattern="*.nc"):
        """
        Batch convert NetCDF files to CSV format (recursively search all subfolders)
        
        Args:
            file_pattern: File matching pattern
        """
        # Check if input folder exists
        if not os.path.exists(self.input_folder):
            print(f"Error: Input folder does not exist: {self.input_folder}")
            print("Please check if the path is correct, or if the USB drive is properly connected")
            return
        
        print(f"Searching folder: {self.input_folder}")
        
        # Recursively get all NetCDF files (including subfolders)
        # Use os.walk to ensure recursive search
        nc_files = []
        for root, dirs, files in os.walk(self.input_folder):
            for file in files:
                if file.endswith('.nc'):
                    nc_files.append(os.path.join(root, file))
        
        # If no files found, try listing directory contents for debugging
        if not nc_files:
            print(f"No {file_pattern} files found in folder {self.input_folder}")
            print("\nDebug info:")
            print(f"Folder exists: {os.path.exists(self.input_folder)}")
            print(f"Is directory: {os.path.isdir(self.input_folder)}")
            
            # Try listing directory contents
            try:
                dir_contents = os.listdir(self.input_folder)
                print(f"Directory contents: {dir_contents[:10]}...")  # Only show first 10
                
                # Try different search patterns
                print("\nTrying different search patterns:")
                patterns_to_try = ["*.nc", "**/*.nc", "**/*.NC", "*/*.nc"]
                for pattern in patterns_to_try:
                    test_files = glob.glob(os.path.join(self.input_folder, pattern), recursive=True)
                    print(f"  Pattern '{pattern}': found {len(test_files)} files")
                    if test_files:
                        print(f"    Example file: {test_files[0]}")
                
                # Check subfolder contents
                print("\nChecking subfolder contents:")
                for subfolder in dir_contents[:3]:  # Only check first 3 subfolders
                    subfolder_path = os.path.join(self.input_folder, subfolder)
                    if os.path.isdir(subfolder_path):
                        try:
                            sub_contents = os.listdir(subfolder_path)
                            nc_files_in_sub = [f for f in sub_contents if f.endswith('.nc')]
                            print(f"  {subfolder}: {len(nc_files_in_sub)} .nc files")
                            if nc_files_in_sub:
                                print(f"    Example: {nc_files_in_sub[0]}")
                        except Exception as e:
                            print(f"  {subfolder}: Error - {e}")
            except Exception as e:
                print(f"Cannot access directory: {e}")
            return
        
        print(f"Found {len(nc_files)} NetCDF files")
        print(f"Using {self.num_processes} processes for conversion")
        print(f"Output folder: {self.output_folder}")
        print("-" * 50)
        
        # Use multiprocessing for processing
        with mp.Pool(processes=self.num_processes) as pool:
            results = list(tqdm(
                pool.imap(self.process_single_file, nc_files),
                total=len(nc_files),
                desc="Conversion progress"
            ))
        
        # Count results
        successful = 0
        failed = 0
        
        print("\nConversion results:")
        print("-" * 50)
        for input_file, output_file, status, error_msg in results:
            if status == "success":
                print(f"[OK] {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
                successful += 1
            else:
                print(f"[FAIL] {os.path.basename(input_file)} -> failed: {error_msg}")
                failed += 1
        
        print("-" * 50)
        print(f"Success: {successful}, Failed: {failed}")

def test_path_access():
    """
    Test path access functionality
    """
    possible_paths = [
        r"G:\2000-2023[PM2.5(NC)]",
        r"G:\Bodhi_Tree\2000-2023[PM2.5(NC)]",
        r"G:\\",
        r"G:\Bodhi_Tree"
    ]
    
    print("Testing possible paths:")
    for path in possible_paths:
        exists = os.path.exists(path)
        is_dir = os.path.isdir(path) if exists else False
        print(f"  {path}: exists={exists}, is_dir={is_dir}")
        
        if exists and is_dir:
            try:
                contents = os.listdir(path)
                print(f"    contents: {contents[:5]}...")
            except Exception as e:
                print(f"    cannot list contents: {e}")

def main():
    """
    Main function - Set input and output paths here
    """
    # ================================
    # Set your folder paths here
    # ================================
    
    # First test path access
    test_path_access()
    print("\n" + "="*50 + "\n")
    
    # Input folder path (folder containing .nc files)
    INPUT_FOLDER = r"G:\2000-2023[PM2.5(NC)]"  # Modify to your input folder path
    
    # Output folder path (folder where .csv files will be saved)
    OUTPUT_FOLDER = r"G:\2000-2023[PM2.5(CSV)]"  # Modify to your output folder path
    
    # Number of processes (defaults to CPU core count, can be adjusted as needed)
    NUM_PROCESSES = mp.cpu_count()  # Can be set to a specific number, e.g., 4
    
    # ================================
    # Execute conversion
    # ================================
    
    converter = NetCDFToCSVConverter(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        num_processes=NUM_PROCESSES
    )
    
    # Start batch conversion
    converter.batch_convert()

if __name__ == "__main__":
    # This protection is needed when using multiprocessing on Windows
    mp.freeze_support()
    main()