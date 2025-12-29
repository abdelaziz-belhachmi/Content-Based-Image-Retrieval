"""
Script to find and fix corrupt JPEG images in the dataset.
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil


def check_and_fix_image(image_path, backup=True):
    """
    Check if an image is valid and try to fix it if corrupt.
    
    Args:
        image_path: Path to the image file
        backup: Whether to backup the original file
    
    Returns:
        tuple: (is_fixed, error_message)
    """
    try:
        # Try to open and verify the image
        with Image.open(image_path) as img:
            img.verify()
        
        # Try to load the image data
        with Image.open(image_path) as img:
            img.load()
        
        return True, None
    
    except Exception as e:
        # Image is corrupt, try to fix it
        error_msg = str(e)
        
        try:
            # Backup original if requested
            if backup:
                backup_path = image_path.parent / f"{image_path.stem}_backup{image_path.suffix}"
                if not backup_path.exists():
                    shutil.copy2(image_path, backup_path)
            
            # Try to open and resave the image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Save with high quality
                img.save(image_path, 'JPEG', quality=95, optimize=True)
            
            # Verify the fixed image
            with Image.open(image_path) as img:
                img.verify()
            
            return True, f"Fixed: {error_msg}"
        
        except Exception as fix_error:
            return False, f"Cannot fix: {fix_error}"


def scan_and_fix_dataset(dataset_dir, remove_unfixable=False):
    """
    Scan the entire dataset and fix corrupt images.
    
    Args:
        dataset_dir: Path to dataset directory
        remove_unfixable: Whether to remove images that cannot be fixed
    """
    dataset_dir = Path(dataset_dir)
    
    # Find all image directories
    image_dirs = []
    for split in ['train', 'val']:
        img_dir = dataset_dir / split / 'images'
        if img_dir.exists():
            image_dirs.append(img_dir)
    
    if not image_dirs:
        print(f"No image directories found in {dataset_dir}")
        return
    
    # Statistics
    stats = {
        'total': 0,
        'valid': 0,
        'fixed': 0,
        'corrupt': 0,
        'removed': 0
    }
    
    corrupt_files = []
    
    print("=" * 60)
    print("Scanning and fixing images...")
    print("=" * 60)
    
    # Process each directory
    for img_dir in image_dirs:
        print(f"\nProcessing: {img_dir}")
        
        # Find all images
        image_files = []
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
            image_files.extend(img_dir.glob(f'*{ext}'))
        
        # Process images
        for img_path in tqdm(image_files, desc=f"  {img_dir.name}"):
            stats['total'] += 1
            
            is_ok, message = check_and_fix_image(img_path, backup=True)
            
            if is_ok:
                if message:
                    # Was fixed
                    stats['fixed'] += 1
                    print(f"\n  ✓ Fixed: {img_path.name}")
                else:
                    # Was already valid
                    stats['valid'] += 1
            else:
                # Cannot be fixed
                stats['corrupt'] += 1
                corrupt_files.append((img_path, message))
                print(f"\n  ✗ Corrupt: {img_path.name} - {message}")
                
                if remove_unfixable:
                    try:
                        # Remove image
                        img_path.unlink()
                        
                        # Remove corresponding label
                        label_path = img_path.parent.parent / 'labels' / f"{img_path.stem}.txt"
                        if label_path.exists():
                            label_path.unlink()
                        
                        stats['removed'] += 1
                        print(f"    Removed: {img_path.name}")
                    except Exception as e:
                        print(f"    Failed to remove: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images: {stats['total']}")
    print(f"  Valid: {stats['valid']}")
    print(f"  Fixed: {stats['fixed']}")
    print(f"  Corrupt (unfixable): {stats['corrupt']}")
    if remove_unfixable:
        print(f"  Removed: {stats['removed']}")
    
    if corrupt_files and not remove_unfixable:
        print("\n" + "=" * 60)
        print("CORRUPT FILES (cannot be fixed)")
        print("=" * 60)
        for file_path, error in corrupt_files:
            print(f"  {file_path}")
            print(f"    Error: {error}")
        
        print(f"\nTo remove these files, run with --remove flag")
    
    print("\n" + "=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Find and fix corrupt images in dataset')
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='Path to dataset directory (default: dataset)')
    parser.add_argument('--remove', action='store_true',
                        help='Remove images that cannot be fixed')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create backup of original files')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    dataset_dir = script_dir / args.dataset
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return 1
    
    print(f"Dataset directory: {dataset_dir.absolute()}")
    
    if args.remove:
        confirm = input("\n⚠ WARNING: This will remove unfixable images. Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Cancelled.")
            return 0
    
    scan_and_fix_dataset(dataset_dir, remove_unfixable=args.remove)
    
    return 0


if __name__ == "__main__":
    exit(main())