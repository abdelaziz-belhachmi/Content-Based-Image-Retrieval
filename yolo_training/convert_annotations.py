"""
Script to convert Pascal VOC annotations to YOLO format.
Also splits the dataset into train and validation sets.
"""

import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path
from tqdm import tqdm


# Category mapping: folder_name -> (class_id, class_name, synset_id)
CATEGORIES = {
    'Ananas': (0, 'pineapple', 'n07753275'),
    'Apple': (1, 'apple', 'n07742313'),
    'BellPepper': (2, 'bell_pepper', 'n07720875'),
    'Bicycle': (3, 'bicycle', 'n02834778'),
    'Broccoli': (4, 'broccoli', 'n07714990'),
    'Bus': (5, 'bus', 'n02924116'),
    'Car': (6, 'car', 'n02958343'),
    'Cat': (7, 'cat', 'n02121620'),
    'Dog': (8, 'dog', 'n02084071'),
    'Elephant': (9, 'elephant', 'n02504013'),
    'Horse': (10, 'horse', 'n02374451'),
    'Lemon': (11, 'lemon', 'n07749582'),
    'Motorcycle': (12, 'motorcycle', 'n03790512'),
    'Strawberry': (13, 'strawberry', 'n07745940'),
    'Tomato': (14, 'tomato', 'n07734017'),
}


def parse_voc_annotation(xml_path):
    """
    Parse Pascal VOC annotation file and extract bounding box information.
    
    Returns:
        dict with 'width', 'height', and 'objects' list
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    objects = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        objects.append({
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })
    
    return {
        'width': width,
        'height': height,
        'objects': objects
    }


def convert_to_yolo(annotation, class_id):
    """
    Convert VOC annotation to YOLO format.
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values are normalized [0, 1]
    
    Returns:
        List of YOLO format strings
    """
    width = annotation['width']
    height = annotation['height']
    
    yolo_lines = []
    for obj in annotation['objects']:
        # Calculate center and dimensions
        x_center = ((obj['xmin'] + obj['xmax']) / 2) / width
        y_center = ((obj['ymin'] + obj['ymax']) / 2) / height
        box_width = (obj['xmax'] - obj['xmin']) / width
        box_height = (obj['ymax'] - obj['ymin']) / height
        
        # Clamp values to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        box_width = max(0, min(1, box_width))
        box_height = max(0, min(1, box_height))
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
    
    return yolo_lines


def convert_dataset(data_dir, output_dir, train_ratio=0.8):
    """
    Convert entire dataset from Pascal VOC to YOLO format.
    
    Args:
        data_dir: Path to the Data folder
        output_dir: Path to output YOLO dataset
        train_ratio: Ratio of training data (default 0.8)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    train_images = output_dir / 'train' / 'images'
    train_labels = output_dir / 'train' / 'labels'
    val_images = output_dir / 'val' / 'images'
    val_labels = output_dir / 'val' / 'labels'
    
    for dir_path in [train_images, train_labels, val_images, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        'total': 0,
        'train': 0,
        'val': 0,
        'errors': 0,
        'per_class': {cat: 0 for cat in CATEGORIES}
    }
    
    # Process each category
    for category_name, (class_id, class_label, synset_id) in CATEGORIES.items():
        category_path = data_dir / category_name
        if not category_path.exists():
            print(f"Warning: Category folder not found: {category_name}")
            continue
        
        # Find annotation and image folders
        annotation_folder = category_path / 'Annotation' / synset_id
        image_folder = category_path / synset_id
        
        if not annotation_folder.exists():
            # Try to find any synset folder in Annotation
            annot_base = category_path / 'Annotation'
            if annot_base.exists():
                subdirs = [d for d in annot_base.iterdir() if d.is_dir()]
                if subdirs:
                    annotation_folder = subdirs[0]
        
        if not image_folder.exists():
            # Try to find any synset folder (not Annotation)
            subdirs = [d for d in category_path.iterdir() if d.is_dir() and d.name != 'Annotation']
            if subdirs:
                image_folder = subdirs[0]
        
        if not annotation_folder.exists() or not image_folder.exists():
            print(f"Warning: Could not find annotation or image folder for {category_name}")
            continue
        
        print(f"\nProcessing {category_name} (class {class_id}: {class_label})...")
        
        # Get all annotation files
        xml_files = list(annotation_folder.glob('*.xml'))
        
        # Shuffle for random split
        random.shuffle(xml_files)
        
        # Split into train and val
        split_idx = int(len(xml_files) * train_ratio)
        train_files = xml_files[:split_idx]
        val_files = xml_files[split_idx:]
        
        # Process files
        for xml_file, is_train in tqdm(
            [(f, True) for f in train_files] + [(f, False) for f in val_files],
            desc=f"  {category_name}"
        ):
            try:
                # Parse annotation
                annotation = parse_voc_annotation(xml_file)
                
                if not annotation['objects']:
                    continue
                
                # Find corresponding image
                base_name = xml_file.stem
                image_path = None
                for ext in ['.JPEG', '.jpeg', '.jpg', '.JPG', '.png', '.PNG']:
                    candidate = image_folder / f"{base_name}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break
                
                if image_path is None:
                    stats['errors'] += 1
                    continue
                
                # Convert to YOLO format
                yolo_lines = convert_to_yolo(annotation, class_id)
                
                if not yolo_lines:
                    continue
                
                # Determine output directories
                if is_train:
                    out_img_dir = train_images
                    out_lbl_dir = train_labels
                    stats['train'] += 1
                else:
                    out_img_dir = val_images
                    out_lbl_dir = val_labels
                    stats['val'] += 1
                
                # Create unique filename with category prefix
                new_filename = f"{category_name}_{base_name}"
                
                # Copy image
                shutil.copy2(image_path, out_img_dir / f"{new_filename}{image_path.suffix}")
                
                # Write YOLO label
                with open(out_lbl_dir / f"{new_filename}.txt", 'w') as f:
                    f.write('\n'.join(yolo_lines))
                
                stats['total'] += 1
                stats['per_class'][category_name] += 1
                
            except Exception as e:
                stats['errors'] += 1
                print(f"    Error processing {xml_file}: {e}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Total images processed: {stats['total']}")
    print(f"  Training set: {stats['train']}")
    print(f"  Validation set: {stats['val']}")
    print(f"  Errors: {stats['errors']}")
    print("\nPer-class distribution:")
    for cat, count in stats['per_class'].items():
        print(f"  {cat}: {count}")
    
    return stats


def main():
    # Get script directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    # Input and output paths
    data_dir = project_dir / 'Data'
    output_dir = script_dir / 'dataset'
    
    print("=" * 60)
    print("Pascal VOC to YOLO Format Converter")
    print("=" * 60)
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Convert dataset
    convert_dataset(data_dir, output_dir, train_ratio=0.8)
    
    print(f"\nDataset ready at: {output_dir}")
    print("You can now run training with: python train.py")


if __name__ == "__main__":
    main()
