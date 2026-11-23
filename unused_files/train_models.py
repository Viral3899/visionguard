#!/usr/bin/env python3
"""
VisionGuard AI - Model Training Script
Trains custom YOLO models on downloaded Hugging Face datasets.
Combines: FalsePositive_Fix (audit_dataset, strict validation) + RTSP_Surveillance (HF datasets)
"""
import os
import sys
import shutil
import json
import math
import hashlib
from collections import Counter
from pathlib import Path
from typing import Optional, Dict, List

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / 'models'
DATASETS_DIR = ROOT / 'datasets'

# Training configurations for each model
TRAIN_CONFIGS = {
    'fire_smoke': {
        'dataset': 'fire_smoke',
        'base_model': 'yolov8n.pt',
        'output': 'yolov26-fire-smoke.pt',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'classes': ['fire', 'smoke'],
    },
    'violence': {
        'dataset': 'violence',
        'base_model': 'yolov8n.pt',
        'output': 'fight-yolov8-nano.pt',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'classes': ['fight', 'violence'],
    },
    'threat': {
        'dataset': 'threat',
        'base_model': 'yolov8n.pt',
        'output': 'threat-detection-yolov8n.pt',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'classes': ['gun', 'knife', 'weapon', 'threat'],
    },
}

def audit_dataset(path):
    """Refuse missing labels/splits and leaking identical train/val images.
    
    From FalsePositive_Fix: strict dataset validation.
    """
    path = Path(path)
    import yaml
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    names = config.get('names')
    if not isinstance(names, (dict, list)) or not names:
        raise ValueError('data.yaml requires labeled class names')
    total_labels = Counter()
    hashes = {}
    for split in ('train', 'val'):
        base = Path(config['path'])
        images = sorted((base / split / 'images').glob('*'))
        if not images:
            raise ValueError(f'{split} has no labeled images')
        for img in images:
            lbl = base / split / 'labels' / (img.stem + '.txt')
            if not lbl.is_file():
                raise ValueError(f'Image lacks corresponding label file: {img}')
            digest = hashlib.sha256(img.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError('Train/validation contain exactly duplicate images')
            hashes[digest] = split
            for line in lbl.read_text(encoding='utf-8').splitlines():
                values = line.split()
                if len(values) != 5:
                    raise ValueError(f'Malformed YOLO annotation: {lbl}')
                cls = int(values[0])
                coords = [float(v) for v in values[1:]]
                if not 0 <= cls < len(names) or not all(math.isfinite(x) for x in coords):
                    raise ValueError(f'Invalid label index / nonfinite coordinate: {lbl}')
                total_labels[(split, cls)] += 1
    for cls in range(len(names)):
        if total_labels[('train',cls)] == 0:
            raise ValueError(f'Class {names[cls]} has no TRAIN bounding boxes')
        if total_labels[('val',cls)] == 0:
            raise ValueError(f'Class {names[cls]} has no VAL bounding boxes; add real labeled examples')
    return {'classes': list(names.values()) if isinstance(names,dict) else names,
            'train_boxes': {str(k):v for (split,k),v in total_labels.items() if split=='train'},
            'val_boxes': {str(k):v for (split,k),v in total_labels.items() if split=='val'}}

# Training configurations for each model
TRAIN_CONFIGS = {
    'fire_smoke': {
        'dataset': 'fire_smoke',
        'base_model': 'yolov8n.pt',
        'output': 'yolov26-fire-smoke.pt',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'classes': ['fire', 'smoke'],
    },
    'violence': {
        'dataset': 'violence',
        'base_model': 'yolov8n.pt',
        'output': 'fight-yolov8-nano.pt',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'classes': ['fight', 'violence'],
    },
    'threat': {
        'dataset': 'threat',
        'base_model': 'yolov8n.pt',
        'output': 'threat-detection-yolov8n.pt',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'classes': ['gun', 'knife', 'weapon', 'threat'],
    },
}

def prepare_yolo_dataset(dataset_name: str, classes: List[str]) -> Optional[Path]:
    """Convert HF dataset to YOLO format with data.yaml."""
    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return None
    
    # Look for existing data.yaml
    yaml_path = dataset_path / 'data.yaml'
    if yaml_path.exists():
        return yaml_path
    
    # Find images/labels directories
    images_dir = None
    labels_dir = None
    
    # Check common structures
    for split in ['train', 'val', 'test', '']:
        base = dataset_path / split if split else dataset_path
        img_dir = base / 'images'
        lbl_dir = base / 'labels'
        if img_dir.exists() and lbl_dir.exists():
            images_dir = img_dir
            labels_dir = lbl_dir
            break
        # Also check flat structure
        if (base / 'images').exists() or list(base.glob('*.jpg')) or list(base.glob('*.png')):
            images_dir = base
            labels_dir = base
            break
    
    if not images_dir:
        print(f"No images directory found in {dataset_path}")
        return None
    
    # Create data.yaml
    yaml_content = f"""path: {dataset_path}
train: {images_dir}
val: {images_dir}
names:
"""
    for i, cls in enumerate(classes):
        yaml_content += f"  {i}: {cls}\n"
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created data.yaml: {yaml_path}")
    return yaml_path

def train_model(config_key: str, device: str = 'auto') -> bool:
    """Train a single model."""
    if config_key not in TRAIN_CONFIGS:
        print(f"Unknown config: {config_key}")
        return False
    
    config = TRAIN_CONFIGS[config_key]
    
    print(f"\n{'='*60}")
    print(f"Training: {config_key}")
    print(f"Output: {config['output']}")
    print(f"Epochs: {config['epochs']}")
    print(f"{'='*60}\n")
    
    # Prepare dataset
    yaml_path = prepare_yolo_dataset(config['dataset'], config['classes'])
    if not yaml_path:
        print(f"Failed to prepare dataset for {config_key}")
        return False
    
    # Check if base model exists
    base_model = config['base_model']
    if not Path(base_model).exists() and not (MODELS_DIR / base_model).exists():
        print(f"Downloading base model: {base_model}")
        # Will be auto-downloaded by ultralytics
    
    try:
        from ultralytics import YOLO
        
        print(f"Loading base model: {base_model}")
        model = YOLO(base_model)
        
        # Determine device
        if device == 'auto':
            import torch
            device = 0 if torch.cuda.is_available() else 'cpu'
        
        print(f"Training on device: {device}")
        
        # Train
        results = model.train(
            data=str(yaml_path),
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            device=device,
            project=str(MODELS_DIR / 'runs'),
            name=f'train_{config_key}',
            exist_ok=True,
            patience=20,
            save_period=10,
            verbose=True,
            plots=True,
        )
        
        # Copy best model to models dir
        best_path = Path(results.save_dir) / 'weights' / 'best.pt'
        last_path = Path(results.save_dir) / 'weights' / 'last.pt'
        
        output_path = MODELS_DIR / config['output']
        
        if best_path.exists():
            shutil.copy2(best_path, output_path)
            print(f"\n✓ Best model saved to: {output_path}")
        elif last_path.exists():
            shutil.copy2(last_path, output_path)
            print(f"\n✓ Last model saved to: {output_path}")
        else:
            print(f"\n✗ No model weights found in {results.save_dir}/weights/")
            return False
        
        # Also copy to root for easy access
        shutil.copy2(output_path, ROOT / config['output'])
        print(f"✓ Model also copied to project root")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_model(model_path: Path, dataset_yaml: Path) -> bool:
    """Validate trained model on dataset."""
    try:
        from ultralytics import YOLO
        
        print(f"Validating model: {model_path}")
        model = YOLO(str(model_path))
        
        results = model.val(
            data=str(dataset_yaml),
            split='val',
            verbose=True,
        )
        
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        
        return True
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train VisionGuard AI Models')
    parser.add_argument('models', nargs='*', help='Models to train (default: all)')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--device', default='auto', help='Device: auto, cpu, 0, 1, etc.')
    parser.add_argument('--epochs', type=int, help='Override epochs for all models')
    parser.add_argument('--validate', action='store_true', help='Validate after training')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available models:")
        for key, config in TRAIN_CONFIGS.items():
            print(f"  {key}: {config['output']} (dataset: {config['dataset']}, epochs: {config['epochs']})")
        return
    
    models_to_train = args.models if args.models else list(TRAIN_CONFIGS.keys())
    
    # Override epochs if specified
    if args.epochs:
        for key in models_to_train:
            if key in TRAIN_CONFIGS:
                TRAIN_CONFIGS[key]['epochs'] = args.epochs
    
    MODELS_DIR.mkdir(exist_ok=True)
    
    print(f"Training models: {', '.join(models_to_train)}")
    print(f"Device: {args.device}")
    
    results = {}
    for model_key in models_to_train:
        if model_key in TRAIN_CONFIGS:
            success = train_model(model_key, device=args.device)
            results[model_key] = success
        else:
            print(f"Unknown model: {model_key}")
            results[model_key] = False
    
    # Validate if requested
    if args.validate:
        print("\nValidating trained models...")
        for model_key, success in results.items():
            if success and model_key in TRAIN_CONFIGS:
                config = TRAIN_CONFIGS[model_key]
                model_path = MODELS_DIR / config['output']
                yaml_path = prepare_yolo_dataset(config['dataset'], config['classes'])
                if yaml_path:
                    validate_model(model_path, yaml_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("Training Summary:")
    print(f"{'='*60}")
    for model_key, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {model_key}: {status}")

if __name__ == '__main__':
    main()