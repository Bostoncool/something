import os
import shutil
import re
from pathlib import Path


def extract_year_from_filename(filename):
    # Remove file extension
    name_without_ext = filename.replace('.csv', '')
    
    # Match different filename patterns
    patterns = [
        r'CHAP_PM2\.5_D1K_(\d{4})\d{4}_V4',  # Daily data: YYYYMMDD
        r'CHAP_PM2\.5_M1K_(\d{4})\d{2}_V4',  # Monthly data: YYYYMM
        r'CHAP_PM2\.5_Y1K_(\d{4})_V4'        # Yearly data: YYYY
    ]
    
    for pattern in patterns:
        match = re.match(pattern, name_without_ext)
        if match:
            return match.group(1)
    
    return None


def organize_files_by_year(source_dir='.', dry_run=False):
    """
    Organize CSV files by year into corresponding folders
    
    Parameters:
        source_dir: Source directory path, defaults to current directory
        dry_run: If True, only print operations without actually moving files
    """
    source_path = Path(source_dir)
    
    # Get all CSV files matching the pattern
    csv_files = [f for f in source_path.glob('CHAP_PM2.5_*.csv')]
    
    if not csv_files:
        print("No CSV files matching naming rules found")
        return
    
    print(f"Found {len(csv_files)} files\n")
    
    # Statistics
    moved_count = 0
    failed_count = 0
    year_folders = {}
    
    for csv_file in csv_files:
        filename = csv_file.name
        year = extract_year_from_filename(filename)
        
        if year is None:
            print(f"⚠ Unable to extract year from filename: {filename}")
            failed_count += 1
            continue
        
        # Create year folder path
        year_folder = source_path / year
        
        # Record file count for each year
        if year not in year_folders:
            year_folders[year] = []
        year_folders[year].append(filename)
        
        # Target file path
        target_file = year_folder / filename
        
        if dry_run:
            print(f"[Simulation] {filename} -> {year}/{filename}")
        else:
            # Create year folder (if it doesn't exist)
            if not year_folder.exists():
                year_folder.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created folder: {year}/")
            
            # Move file
            try:
                shutil.move(str(csv_file), str(target_file))
                print(f"✓ Moved: {filename} -> {year}/")
                moved_count += 1
            except Exception as e:
                print(f"✗ Move failed: {filename} - {str(e)}")
                failed_count += 1
    
    # Output statistics
    print("\n" + "="*60)
    print("Processing Summary:")
    print("="*60)
    
    if not dry_run:
        print(f"Successfully moved: {moved_count} files")
        print(f"Failed: {failed_count} files")
    else:
        print(f"Will move: {len(csv_files) - failed_count} files")
        print(f"Unrecognized: {failed_count} files")
    
    print(f"\nDistribution by year:")
    for year in sorted(year_folders.keys()):
        print(f"  {year}: {len(year_folders[year])} files")


def main():
    """Main function"""
    import sys
    
    print("="*60)
    print("CSV File Year Organization Tool")
    print("="*60)
    print()
    
    # Get target directory
    if len(sys.argv) > 1:
        # If path parameter provided via command line
        target_dir = Path(sys.argv[1])
    else:
        # Interactive path input
        print("Please enter the folder path to organize:")
        print("(Press Enter to use the script's current directory)")
        user_input = input("Path: ").strip()
        
        if user_input:
            target_dir = Path(user_input)
        else:
            target_dir = Path(__file__).parent
    
    # Verify directory exists
    if not target_dir.exists():
        print(f"\n✗ Error: Directory does not exist: {target_dir}")
        return
    
    if not target_dir.is_dir():
        print(f"\n✗ Error: Path is not a directory: {target_dir}")
        return
    
    print(f"\nTarget directory: {target_dir.absolute()}\n")
    
    # First preview (simulation run)
    print("[Preview Mode] View operations to be performed...\n")
    organize_files_by_year(target_dir, dry_run=True)
    
    print("\n" + "="*60)
    response = input("\nConfirm to execute the above operations? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\nStarting to move files...\n")
        organize_files_by_year(target_dir, dry_run=False)
        print("\n✓ All operations completed!")
    else:
        print("\nOperation cancelled")


if __name__ == "__main__":
    main()

