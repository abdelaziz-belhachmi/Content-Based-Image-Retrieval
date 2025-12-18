"""
Script to filter images and annotations, keeping only matched pairs.
- Removes images without corresponding annotations
- Removes annotations without corresponding images
- Preserves the original folder structure
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict


def get_base_name(filename):
    """Get filename without extension."""
    return Path(filename).stem


def find_synset_folder(category_path):
    """Find the synset folder (image folder) in a category directory."""
    for item in os.listdir(category_path):
        item_path = os.path.join(category_path, item)
        if os.path.isdir(item_path) and item != "Annotation":
            return item, item_path
    return None, None


def get_annotation_synset_folder(annotation_path):
    """Find the synset folder inside the Annotation directory."""
    if not os.path.exists(annotation_path):
        return None, None
    for item in os.listdir(annotation_path):
        item_path = os.path.join(annotation_path, item)
        if os.path.isdir(item_path):
            return item, item_path
    return None, None


def filter_category(category_path, dry_run=True):
    """
    Filter a single category folder, keeping only matched image-annotation pairs.
    
    Args:
        category_path: Path to the category folder (e.g., Data/Ananas)
        dry_run: If True, only print what would be deleted without actually deleting
    
    Returns:
        dict with statistics about the filtering
    """
    stats = {
        'images_with_annotations': 0,
        'images_without_annotations': 0,
        'annotations_without_images': 0,
        'deleted_images': [],
        'deleted_annotations': []
    }
    
    category_name = os.path.basename(category_path)
    
    # Find image folder (synset folder)
    synset_name, image_folder = find_synset_folder(category_path)
    if not image_folder:
        print(f"  [WARNING] No image folder found in {category_name}")
        return stats
    
    # Find annotation folder
    annotation_base = os.path.join(category_path, "Annotation")
    annot_synset, annotation_folder = get_annotation_synset_folder(annotation_base)
    if not annotation_folder:
        print(f"  [WARNING] No annotation folder found in {category_name}")
        return stats
    
    # Get all images and their base names
    image_files = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    for f in os.listdir(image_folder):
        file_path = os.path.join(image_folder, f)
        if os.path.isfile(file_path):
            ext = Path(f).suffix.lower()
            if ext in image_extensions:
                base_name = get_base_name(f)
                image_files[base_name] = f
    
    # Get all annotations and their base names
    annotation_files = {}
    annotation_extensions = {'.xml'}
    
    for f in os.listdir(annotation_folder):
        file_path = os.path.join(annotation_folder, f)
        if os.path.isfile(file_path):
            ext = Path(f).suffix.lower()
            if ext in annotation_extensions:
                base_name = get_base_name(f)
                annotation_files[base_name] = f
    
    # Find matched pairs
    image_base_names = set(image_files.keys())
    annotation_base_names = set(annotation_files.keys())
    
    matched = image_base_names & annotation_base_names
    images_only = image_base_names - annotation_base_names
    annotations_only = annotation_base_names - image_base_names
    
    stats['images_with_annotations'] = len(matched)
    stats['images_without_annotations'] = len(images_only)
    stats['annotations_without_images'] = len(annotations_only)
    
    # Delete images without annotations
    for base_name in images_only:
        filename = image_files[base_name]
        file_path = os.path.join(image_folder, filename)
        stats['deleted_images'].append(filename)
        if not dry_run:
            os.remove(file_path)
    
    # Delete annotations without images
    for base_name in annotations_only:
        filename = annotation_files[base_name]
        file_path = os.path.join(annotation_folder, filename)
        stats['deleted_annotations'].append(filename)
        if not dry_run:
            os.remove(file_path)
    
    return stats


def filter_data_folder(data_path, dry_run=True):
    """
    Filter all categories in the Data folder.
    
    Args:
        data_path: Path to the Data folder
        dry_run: If True, only print what would be deleted without actually deleting
    """
    if not os.path.exists(data_path):
        print(f"Error: Data folder not found at {data_path}")
        return
    
    print("=" * 70)
    print(f"{'DRY RUN - ' if dry_run else ''}Filtering Data Folder: {data_path}")
    print("=" * 70)
    
    total_stats = {
        'total_matched': 0,
        'total_images_deleted': 0,
        'total_annotations_deleted': 0,
        'categories_processed': 0
    }
    
    # Get all category folders
    categories = []
    for item in sorted(os.listdir(data_path)):
        item_path = os.path.join(data_path, item)
        if os.path.isdir(item_path):
            categories.append((item, item_path))
    
    for category_name, category_path in categories:
        print(f"\n📁 Processing: {category_name}")
        print("-" * 50)
        
        stats = filter_category(category_path, dry_run)
        
        print(f"  ✅ Matched pairs (kept): {stats['images_with_annotations']}")
        print(f"  🗑️  Images without annotations: {stats['images_without_annotations']}")
        print(f"  🗑️  Annotations without images: {stats['annotations_without_images']}")
        
        if stats['deleted_images']:
            print(f"\n  Images to {'delete' if dry_run else 'deleted'}:")
            for img in stats['deleted_images'][:5]:  # Show first 5
                print(f"    - {img}")
            if len(stats['deleted_images']) > 5:
                print(f"    ... and {len(stats['deleted_images']) - 5} more")
        
        if stats['deleted_annotations']:
            print(f"\n  Annotations to {'delete' if dry_run else 'deleted'}:")
            for ann in stats['deleted_annotations'][:5]:  # Show first 5
                print(f"    - {ann}")
            if len(stats['deleted_annotations']) > 5:
                print(f"    ... and {len(stats['deleted_annotations']) - 5} more")
        
        total_stats['total_matched'] += stats['images_with_annotations']
        total_stats['total_images_deleted'] += stats['images_without_annotations']
        total_stats['total_annotations_deleted'] += stats['annotations_without_images']
        total_stats['categories_processed'] += 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Categories processed: {total_stats['categories_processed']}")
    print(f"Total matched pairs (kept): {total_stats['total_matched']}")
    print(f"Total images {'to delete' if dry_run else 'deleted'}: {total_stats['total_images_deleted']}")
    print(f"Total annotations {'to delete' if dry_run else 'deleted'}: {total_stats['total_annotations_deleted']}")
    
    if dry_run:
        print("\n⚠️  This was a DRY RUN. No files were actually deleted.")
        print("    To actually delete files, run with dry_run=False")
    else:
        print("\n✅ Filtering complete! Only matched pairs remain.")
    
    return total_stats


def main():
    # Set the path to your Data folder
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
    
    # First, do a dry run to see what would be deleted
    print("\n" + "🔍 PHASE 1: DRY RUN (Preview)" + "\n")
    filter_data_folder(data_path, dry_run=True)
    
    # Ask for confirmation before actually deleting
    print("\n" + "-" * 70)
    response = input("\n⚠️  Do you want to proceed with actual deletion? (yes/no): ").strip().lower()
    
    if response == 'yes':
        print("\n" + "🗑️ PHASE 2: ACTUAL DELETION" + "\n")
        filter_data_folder(data_path, dry_run=False)
    else:
        print("\n❌ Deletion cancelled. No files were modified.")


if __name__ == "__main__":
    main()
