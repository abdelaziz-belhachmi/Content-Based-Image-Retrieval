"""
YOLOv8n Training Script
Fine-tunes YOLOv8n on the custom 15-class dataset.
"""

import os
from pathlib import Path
from ultralytics import YOLO


def train_model(
    data_yaml='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    device='auto',
    project='runs',
    name='yolov8n_custom'
):
    """
    Train YOLOv8n model on custom dataset.
    
    Args:
        data_yaml: Path to data.yaml configuration
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        patience: Early stopping patience
        device: 'cuda', 'cpu', or 'auto'
        project: Project name for saving results
        name: Experiment name
    """
    # Load pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    print("=" * 60)
    print("YOLOv8n Training Configuration")
    print("=" * 60)
    print(f"Data config: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Early stopping patience: {patience}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        save=True,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        deterministic=True,
        plots=True,
        # Data augmentation
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=0.0,  # Rotation (+/- deg)
        translate=0.1,  # Translation (+/- fraction)
        scale=0.5,    # Scale (+/- gain)
        shear=0.0,    # Shear (+/- deg)
        flipud=0.0,   # Flip up-down (probability)
        fliplr=0.5,   # Flip left-right (probability)
        mosaic=1.0,   # Mosaic augmentation (probability)
        mixup=0.0,    # Mixup augmentation (probability)
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Save best model to models directory
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    best_model_path = Path(project) / name / 'weights' / 'best.pt'
    if best_model_path.exists():
        import shutil
        dest_path = models_dir / 'yolov8n_custom_best.pt'
        shutil.copy2(best_model_path, dest_path)
        print(f"Best model saved to: {dest_path}")
    
    return results


def validate_model(model_path, data_yaml='data.yaml'):
    """
    Validate trained model on validation set.
    """
    model = YOLO(model_path)
    
    results = model.val(
        data=data_yaml,
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.6,
        device='auto'
    )
    
    return results


def test_inference(model_path, image_path):
    """
    Test inference on a single image.
    """
    model = YOLO(model_path)
    
    results = model(image_path)
    
    for r in results:
        print(f"Detected {len(r.boxes)} objects")
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f"  Class: {r.names[cls]}, Confidence: {conf:.2f}, Box: {xyxy}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8n Training Script')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validate', 'test'],
                        help='Mode: train, validate, or test')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--model', type=str, default='models/yolov8n_custom_best.pt',
                        help='Model path for validation/testing')
    parser.add_argument('--image', type=str, help='Image path for testing')
    
    args = parser.parse_args()
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    if args.mode == 'train':
        train_model(
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device
        )
    elif args.mode == 'validate':
        validate_model(args.model)
    elif args.mode == 'test':
        if not args.image:
            print("Error: --image required for test mode")
            return
        test_inference(args.model, args.image)


if __name__ == "__main__":
    main()
