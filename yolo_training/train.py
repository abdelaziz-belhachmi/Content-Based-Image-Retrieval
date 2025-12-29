"""
YOLOv8n Training Script
Fine-tunes YOLOv8n on the custom 15-class dataset.
"""

import os
import sys
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt


def check_cuda_available():
    """
    Check if CUDA is available and return device information.
    
    Returns:
        tuple: (is_cuda_available, device_info_string)
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            info = f"Found {device_count} GPU(s): {', '.join(device_names)}"
            return True, info
        else:
            return False, "No CUDA devices found"
    except Exception as e:
        return False, f"Error checking CUDA: {e}"


def prompt_device_selection():
    """
    Prompt user to select CPU or GPU for training.
    
    Returns:
        str: Device string ('cpu', '0', '1', etc.)
    """
    cuda_available, info = check_cuda_available()
    
    print("\n" + "=" * 60)
    print("DEVICE SELECTION")
    print("=" * 60)
    print(info)
    print()
    
    if cuda_available:
        print("Available options:")
        print("  [1] GPU (CUDA) - Recommended for faster training")
        print("  [2] CPU - Slower but works on any system")
        print()
        
        while True:
            choice = input("Select device (1 for GPU, 2 for CPU): ").strip()
            
            if choice == '1':
                import torch
                device_count = torch.cuda.device_count()
                
                if device_count > 1:
                    print(f"\nMultiple GPUs detected ({device_count} GPUs)")
                    for i in range(device_count):
                        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
                    print(f"  [a] All GPUs (0,1,2,...)")
                    
                    while True:
                        gpu_choice = input(f"\nSelect GPU (0-{device_count-1} or 'a' for all): ").strip().lower()
                        
                        if gpu_choice == 'a':
                            device = ','.join(str(i) for i in range(device_count))
                            print(f"\n✓ Selected: All GPUs ({device})")
                            return device
                        elif gpu_choice.isdigit() and 0 <= int(gpu_choice) < device_count:
                            print(f"\n✓ Selected: GPU {gpu_choice} ({torch.cuda.get_device_name(int(gpu_choice))})")
                            return gpu_choice
                        else:
                            print(f"Invalid choice. Please enter 0-{device_count-1} or 'a'")
                else:
                    print(f"\n✓ Selected: GPU (CUDA)")
                    return '0'
            
            elif choice == '2':
                print("\n✓ Selected: CPU")
                print("⚠ Warning: CPU training will be significantly slower than GPU")
                return 'cpu'
            
            else:
                print("Invalid choice. Please enter 1 or 2")
    
    else:
        print("⚠ CUDA not available. Using CPU.")
        print()
        
        # Check if torch is CPU-only
        try:
            import torch
            if not torch.cuda.is_available():
                print("📋 To enable GPU training, reinstall PyTorch with CUDA:")
                print("   1. Check your CUDA version: nvidia-smi")
                print("   2. Uninstall current PyTorch: pip uninstall torch torchvision torchaudio")
                print("   3. Install CUDA version:")
                print("      - CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                print("      - CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                print("      - CUDA 12.4: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        except:
            pass
        
        print()
        input("Press Enter to continue with CPU training...")
        return 'cpu'


def save_training_results(results, output_dir, model_name='yolov8n_custom'):
    """
    Save detailed training results to JSON and generate visualizations.
    
    Args:
        results: Training results object
        output_dir: Directory to save results
        model_name: Name of the model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
    try:
        metrics = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'final_metrics': {
                'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
            },
            'training_params': {
                'epochs_completed': len(results.results_dict.get('train/box_loss', [])) if hasattr(results, 'results_dict') else 0,
            }
        }
        
        # Save to JSON
        with open(output_dir / 'training_results.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Training results saved to: {output_dir / 'training_results.json'}")
        
        return metrics
    except Exception as e:
        print(f"Warning: Could not save detailed results: {e}")
        return None


def print_training_summary(results, training_time=None):
    """
    Print a comprehensive training summary.
    
    Args:
        results: Training results object
        training_time: Training duration in seconds
    """
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    try:
        # Extract final metrics
        metrics_dict = results.results_dict if hasattr(results, 'results_dict') else {}
        
        print(f"\n📊 Final Metrics:")
        print(f"  mAP@50:     {metrics_dict.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@50-95:  {metrics_dict.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision:  {metrics_dict.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall:     {metrics_dict.get('metrics/recall(B)', 0):.4f}")
        
        if training_time:
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            seconds = int(training_time % 60)
            print(f"\n⏱️  Training Time: {hours}h {minutes}m {seconds}s")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"Note: Could not extract all metrics: {e}")


def train_model(
    data_yaml='data.yaml',
    epochs=5,
    imgsz=640,
    batch=8,
    patience=20,
    device='0',
    project='runs',
    name='yolov8n_finetuned',
    freeze_layers=10,
    learning_rate=0.001
):
    """
    Fine-tune YOLOv8n model on custom dataset.
    
    Fine-tuning freezes the early backbone layers and only trains the
    later layers and detection head on your custom dataset.
    
    Args:
        data_yaml: Path to data.yaml configuration
        epochs: Number of training epochs (default: 50 for fine-tuning)
        imgsz: Image size for training
        batch: Batch size
        patience: Early stopping patience
        device: 'cuda', 'cpu', or GPU index
        project: Project name for saving results
        name: Experiment name
        freeze_layers: Number of layers to freeze (default: 10 for fine-tuning)
        learning_rate: Initial learning rate (default: 0.001 for fine-tuning)
    
    Returns:
        Training results object
    """
    # Verify data.yaml exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Data configuration file not found: {data_yaml}")
    
    # Load pretrained YOLOv8n model
    print("\nLoading YOLOv8n pretrained model...")
    try:
        model_path = Path('yolov8n.pt')
        if not model_path.exists():
            print("Downloading YOLOv8n weights...")
        model = YOLO('yolov8n.pt')
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("YOLOv8n FINE-TUNING Configuration")
    print("=" * 60)
    print(f"Mode:               FINE-TUNING (not training from scratch)")
    print(f"Data config:        {data_path.absolute()}")
    print(f"Epochs:             {epochs}")
    print(f"Image size:         {imgsz}")
    print(f"Batch size:         {batch}")
    print(f"Patience:           {patience}")
    print(f"Device:             {device}")
    print(f"Learning rate:      {learning_rate} (lower for fine-tuning)")
    print(f"Frozen layers:      {freeze_layers} (backbone frozen, head trainable)")
    print("=" * 60)
    print()
    print("ℹ️  Fine-tuning Strategy:")
    print(f"   - Freezing first {freeze_layers} layers (backbone)")
    print(f"   - Training detection head + last layers on your data")
    print(f"   - Using lower learning rate to preserve pretrained features")
    print(f"   - Faster training, requires less data")
    print("=" * 60)
    print()
    
    # Record start time
    import time
    start_time = time.time()
    
    # Train the model
    print("🚀 Starting fine-tuning...\n")
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        save=True,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,  # Keep pretrained weights
        optimizer='AdamW',  # Better for fine-tuning
        verbose=False,
        seed=42,
        deterministic=False,
        plots=True,
        workers=2,
        lr0=learning_rate,  # Lower LR for fine-tuning
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Lighter augmentation for fine-tuning
        hsv_h=0.01,  # Reduced
        hsv_s=0.5,   # Reduced
        hsv_v=0.3,   # Reduced
        degrees=0.0,
        translate=0.05,  # Reduced
        scale=0.3,       # Reduced
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,      # Reduced
        mixup=0.0,
        copy_paste=0.0,
        # Fine-tuning: freeze backbone layers
        freeze=freeze_layers,
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE")
    print("=" * 60)
    
    # Print training summary
    print_training_summary(results, training_time)
    
    # Save best model to models directory
    script_dir = Path(__file__).parent
    models_dir = script_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Construct the path to the best model
    best_model_path = Path(project) / name / 'weights' / 'best.pt'
    last_model_path = Path(project) / name / 'weights' / 'last.pt'
    
    # If project path is relative, make it absolute from script dir
    if not best_model_path.is_absolute():
        best_model_path = script_dir / best_model_path
    if not last_model_path.is_absolute():
        last_model_path = script_dir / last_model_path
    
    print(f"\n📁 Saving models to: {models_dir.absolute()}")
    
    # Copy best model
    if best_model_path.exists():
        dest_path = models_dir / 'yolov8n_finetuned_best.pt'
        try:
            shutil.copy2(best_model_path, dest_path)
            file_size = dest_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ Best fine-tuned model: {dest_path.name} ({file_size:.2f} MB)")
        except Exception as e:
            print(f"✗ Error copying best model: {e}")
    else:
        print(f"✗ Warning: Best model not found at {best_model_path}")
    
    # Copy last model
    if last_model_path.exists():
        dest_path = models_dir / 'yolov8n_finetuned_last.pt'
        try:
            shutil.copy2(last_model_path, dest_path)
            file_size = dest_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ Last fine-tuned model: {dest_path.name} ({file_size:.2f} MB)")
        except Exception as e:
            print(f"✗ Error copying last model: {e}")
    
    # Save detailed results
    results_dir = script_dir / 'results' / name
    save_training_results(results, results_dir, model_name=name)
    
    # Print final paths
    print(f"\n📊 Training artifacts:")
    print(f"  Weights:     {best_model_path.parent}")
    print(f"  Results:     {Path(project) / name}")
    print(f"  Plots:       {Path(project) / name}")
    
    return results


def validate_model(model_path, data_yaml='data.yaml', save_json=True):
    """
    Validate trained model on validation set.
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to data.yaml configuration
        save_json: Whether to save results to JSON
    
    Returns:
        Validation results object
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Data configuration file not found: {data_yaml}")
    
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    
    print("Running validation...")
    results = model.val(
        data=str(data_path),
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.6,
        device='auto',
        save_json=save_json,
        save_hybrid=False,
        plots=True
    )
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    try:
        metrics = results.results_dict
        print(f"\n📊 Metrics:")
        print(f"  mAP@50:     {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@50-95:  {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision:  {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall:     {metrics.get('metrics/recall(B)', 0):.4f}")
    except Exception as e:
        print(f"Note: Could not extract metrics: {e}")
    
    print("=" * 60)
    
    return results


def test_inference(model_path, image_path, conf_threshold=0.25, save_output=True):
    """
    Test inference on a single image.
    
    Args:
        model_path: Path to trained model weights
        image_path: Path to test image
        conf_threshold: Confidence threshold for detections
        save_output: Whether to save annotated image
    
    Returns:
        Inference results object
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"Running inference on: {image_path}")
    results = model(str(image_path), conf=conf_threshold, save=save_output)
    
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    
    for r in results:
        print(f"\n📸 Image: {image_path.name}")
        print(f"Detected {len(r.boxes)} objects:\n")
        
        if len(r.boxes) == 0:
            print("  No objects detected")
        else:
            for idx, box in enumerate(r.boxes, 1):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                print(f"  [{idx}] Class: {r.names[cls]}")
                print(f"      Confidence: {conf:.3f}")
                print(f"      BBox: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
        
        if save_output:
            output_dir = Path('runs/detect')
            print(f"\n💾 Annotated image saved to: {output_dir}")
    
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8n Training Script with Fine-Tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard fine-tuning (recommended)
  python train.py --epochs 50 --freeze 10 --lr 0.001
  
  # Quick fine-tuning test
  python train.py --epochs 10 --batch 8 --freeze 10
  
  # Deeper fine-tuning (train more layers)
  python train.py --epochs 50 --freeze 5 --lr 0.001
  
  # Light fine-tuning (freeze more, only train head)
  python train.py --epochs 30 --freeze 15 --lr 0.0005
  
  # Fine-tuning on CPU (slower)
  python train.py --device cpu --epochs 20 --batch 4
  
  # Validate fine-tuned model
  python train.py --mode validate --model models/yolov8n_finetuned_best.pt
  
  # Test inference
  python train.py --mode test --model models/yolov8n_finetuned_best.pt --image test.jpg
        """
    )
    
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'validate', 'test'],
                        help='Mode: train, validate, or test')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of fine-tuning epochs (default: 50)')
    parser.add_argument('--batch', type=int, default=8, 
                        help='Batch size (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640, 
                        help='Image size (default: 640)')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device: 0, cpu, or 0,1,2,3 for multi-GPU')
    parser.add_argument('--freeze', type=int, default=10,
                        help='Number of backbone layers to freeze (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001 for fine-tuning)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--model', type=str, 
                        help='Model path for validation/testing')
    parser.add_argument('--image', type=str, 
                        help='Image path for testing')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for inference (default: 0.25)')
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='Path to data.yaml (default: data.yaml)')
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Working directory: {script_dir.absolute()}")
    
    try:
        if args.mode == 'train':
            # Prompt for device selection if not specified
            if args.device is None:
                device = prompt_device_selection()
            else:
                device = args.device
                print(f"\nUsing specified device: {device}")
            
            # Determine training mode
            mode_desc = f"Fine-tuning with {args.freeze} frozen layers" if args.freeze > 0 else "Full training (no frozen layers)"
            print(f"\n🔧 Mode: {mode_desc}")
            
            if args.freeze == 0:
                print("⚠️  Warning: No layers frozen. This is full training, not fine-tuning!")
                print("   For fine-tuning, use --freeze 10 (or higher)")
            
            train_model(
                data_yaml=args.data,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=device,
                patience=args.patience,
                freeze_layers=args.freeze,
                learning_rate=args.lr
            )
        
        elif args.mode == 'validate':
            if not args.model:
                # Try to find the fine-tuned model
                default_model = script_dir / 'models' / 'yolov8n_finetuned_best.pt'
                if default_model.exists():
                    args.model = str(default_model)
                else:
                    parser.error("--model required for validate mode")
            validate_model(args.model, args.data)
        
        elif args.mode == 'test':
            if not args.model:
                # Try to find the fine-tuned model
                default_model = script_dir / 'models' / 'yolov8n_finetuned_best.pt'
                if default_model.exists():
                    args.model = str(default_model)
                else:
                    parser.error("--model required for test mode")
            if not args.image:
                parser.error("--image required for test mode")
            test_inference(args.model, args.image, conf_threshold=args.conf)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        return 1
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())