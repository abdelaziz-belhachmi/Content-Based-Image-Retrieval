#!/usr/bin/env python3
"""
3D Pottery Benchmark Dataset Downloader

Downloads the 3D Pottery Benchmark Dataset from IPET/Athena Research Centre
and organizes it for use with the 3D CBIR system.

Dataset: http://www.ipet.gr/~akoutsou/benchmark/
Reference: A. Koutsoudis et al., "3D Pottery Content Based Retrieval based on 
           Pose Normalisation and Segmentation", Journal of Cultural Heritage, 
           Vol. 11 (2010), pp. 329-338.

Author: CBIR System
Date: January 2026
"""

import os
import sys
import zipfile
import urllib.request
import urllib.error
import shutil
from pathlib import Path
import time

# Configuration
BENCHMARK_URL = "http://www.ipet.gr/~akoutsou/benchmark/dataset/3DPotteryDataset_v_1.zip"
THUMBNAILS_URL = "http://www.ipet.gr/~akoutsou/benchmark/dataset/thumbnails_v_1.zip"
GROUNDTRUTH_URL = "http://www.ipet.gr/~akoutsou/benchmark/dataset/3D%20Pottery_Groundtruth_and_Metadata.xls"

# Get the script directory as base path
BASE_DIR = Path(__file__).parent.absolute()
DATA_3D_DIR = BASE_DIR / "Data_3D"
MODELS_DIR = DATA_3D_DIR / "models"
THUMBNAILS_DIR = DATA_3D_DIR / "thumbnails"
TEMP_DIR = DATA_3D_DIR / "temp"


def print_banner():
    """Print script banner."""
    print("=" * 70)
    print("   3D Pottery Benchmark Dataset Downloader")
    print("   Source: IPET/Athena Research Centre")
    print("   http://www.ipet.gr/~akoutsou/benchmark/")
    print("=" * 70)
    print()


def create_directories():
    """Create necessary directories."""
    print("[1/5] Creating directories...")
    
    directories = [DATA_3D_DIR, MODELS_DIR, THUMBNAILS_DIR, TEMP_DIR]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print()


def download_file(url: str, destination: Path, description: str) -> bool:
    """
    Download a file with progress indicator.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        description: Description for progress display
        
    Returns:
        True if successful, False otherwise
    """
    print(f"  Downloading: {description}")
    print(f"  URL: {url}")
    
    try:
        # Create a request with a user agent
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) CBIR-System/1.0'}
        )
        
        with urllib.request.urlopen(request, timeout=300) as response:
            total_size = response.headers.get('Content-Length')
            
            if total_size:
                total_size = int(total_size)
                print(f"  Size: {total_size / (1024*1024):.1f} MB")
            
            # Download with progress
            downloaded = 0
            block_size = 8192
            
            with open(destination, 'wb') as out_file:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    if total_size:
                        progress = (downloaded / total_size) * 100
                        print(f"\r  Progress: {progress:.1f}% ({downloaded / (1024*1024):.1f} MB)", end='')
            
            print(f"\n  ✓ Downloaded to: {destination}")
            return True
            
    except urllib.error.HTTPError as e:
        print(f"\n  ✗ HTTP Error {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"\n  ✗ URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"\n  ✗ Error: {str(e)}")
        return False


def extract_zip(zip_path: Path, extract_to: Path, description: str) -> bool:
    """
    Extract a ZIP file.
    
    Args:
        zip_path: Path to the ZIP file
        extract_to: Directory to extract to
        description: Description for display
        
    Returns:
        True if successful, False otherwise
    """
    print(f"  Extracting: {description}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            print(f"  Files in archive: {total_files}")
            
            # Extract all
            for i, file in enumerate(file_list):
                zip_ref.extract(file, extract_to)
                if (i + 1) % 100 == 0:
                    print(f"\r  Extracted: {i + 1}/{total_files}", end='')
            
            print(f"\n  ✓ Extracted to: {extract_to}")
            return True
            
    except zipfile.BadZipFile:
        print(f"  ✗ Invalid ZIP file: {zip_path}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False


def organize_models():
    """Organize extracted models into category folders."""
    print("[4/5] Organizing models by category...")
    
    # Find all OBJ files in the extracted directory
    obj_files = list(TEMP_DIR.rglob("*.obj"))
    
    if not obj_files:
        print("  ⚠ No OBJ files found in extracted content")
        # Try to find them in a subdirectory
        obj_files = list(DATA_3D_DIR.rglob("*.obj"))
    
    print(f"  Found {len(obj_files)} OBJ files")
    
    # Move OBJ files to models directory
    moved_count = 0
    for obj_file in obj_files:
        dest = MODELS_DIR / obj_file.name
        if not dest.exists():
            shutil.copy2(obj_file, dest)
            moved_count += 1
    
    print(f"  ✓ Copied {moved_count} models to {MODELS_DIR}")
    
    # Copy groundtruth/metadata files to Data_3D root
    groundtruth_files = list(TEMP_DIR.rglob("*.xls")) + list(TEMP_DIR.rglob("*.xlsx"))
    for gt_file in groundtruth_files:
        dest = DATA_3D_DIR / gt_file.name
        if not dest.exists():
            shutil.copy2(gt_file, dest)
            print(f"  ✓ Copied groundtruth file: {gt_file.name}")
    
    # Also copy any txt files (may contain class info)
    txt_files = list(TEMP_DIR.rglob("*.txt"))
    for txt_file in txt_files:
        dest = DATA_3D_DIR / txt_file.name
        if not dest.exists():
            shutil.copy2(txt_file, dest)
    
    # Move thumbnail files
    jpg_files = list(TEMP_DIR.rglob("*.jpg")) + list(TEMP_DIR.rglob("*.jpeg"))
    for jpg_file in jpg_files:
        dest = THUMBNAILS_DIR / jpg_file.name
        if not dest.exists():
            shutil.copy2(jpg_file, dest)
    
    print(f"  ✓ Copied {len(jpg_files)} thumbnails to {THUMBNAILS_DIR}")
    print()


def cleanup():
    """Clean up temporary files."""
    print("[5/5] Cleaning up temporary files...")
    
    try:
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            print(f"  ✓ Removed temporary directory")
    except Exception as e:
        print(f"  ⚠ Could not remove temp directory: {e}")
    
    print()


def print_summary():
    """Print download summary."""
    print("=" * 70)
    print("   DOWNLOAD COMPLETE")
    print("=" * 70)
    print()
    
    # Count files
    obj_count = len(list(MODELS_DIR.glob("*.obj"))) if MODELS_DIR.exists() else 0
    jpg_count = len(list(THUMBNAILS_DIR.glob("*.jpg"))) if THUMBNAILS_DIR.exists() else 0
    
    print(f"  Location: {DATA_3D_DIR}")
    print(f"  3D Models (OBJ): {obj_count}")
    print(f"  Thumbnails (JPG): {jpg_count}")
    print()
    print("  Directory Structure:")
    print(f"    {DATA_3D_DIR}/")
    print(f"    ├── models/        # OBJ files for indexing")
    print(f"    └── thumbnails/    # Preview images")
    print()
    print("  Next Steps:")
    print("    1. Start the Flask API: cd flask_api && python app.py")
    print("    2. Start Django: cd django_app && python manage.py runserver")
    print("    3. Navigate to 'Recherche 3D' > 'Construire Index'")
    print(f"    4. Enter path: {MODELS_DIR}")
    print("    5. Click 'Lancer l'Indexation'")
    print()
    print("  Reference (please cite):")
    print("    A. Koutsoudis et al., '3D Pottery Content Based Retrieval")
    print("    based on Pose Normalisation and Segmentation',")
    print("    Journal of Cultural Heritage, Vol. 11 (2010), pp. 329-338.")
    print()


def download_benchmark():
    """Main download function."""
    print_banner()
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Download main dataset
    print("[2/5] Downloading 3D Pottery Dataset...")
    zip_path = TEMP_DIR / "3DPotteryDataset_v_1.zip"
    
    success = download_file(
        BENCHMARK_URL,
        zip_path,
        "3D Pottery Dataset (main archive)"
    )
    
    if not success:
        print("\n  Trying alternative download method...")
        # Alternative: try with HTTPS
        alt_url = BENCHMARK_URL.replace("http://", "https://")
        success = download_file(alt_url, zip_path, "3D Pottery Dataset (HTTPS)")
    
    if not success:
        print("\n  ⚠ Could not download automatically.")
        print(f"  Please download manually from: {BENCHMARK_URL}")
        print(f"  And extract to: {DATA_3D_DIR}")
        return False
    
    print()
    
    # Step 3: Extract
    print("[3/5] Extracting archive...")
    if zip_path.exists():
        extract_zip(zip_path, TEMP_DIR, "3D Pottery Dataset")
    print()
    
    # Step 4: Organize
    organize_models()
    
    # Step 5: Cleanup
    cleanup()
    
    # Summary
    print_summary()
    
    return True


def main():
    """Main entry point."""
    try:
        success = download_benchmark()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
