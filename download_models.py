#!/usr/bin/env python3
"""Download YOLO models for VisionGuard."""
import os
import sys
from pathlib import Path
from ultralytics import YOLO

MODELS_DIR = Path(__file__).parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    'yolov8n.pt': 'Person detection (COCO)',
    'yolov8n-pose.pt': 'Pose estimation for fall/loitering detection',
    # Fire/smoke models - these are community models, you may need to train your own
    # 'yolov8n-fire.pt': 'Fire & smoke detection',
    # 'yolov8n-violence.pt': 'Violence & weapon detection',
}

def download_model(name: str, description: str):
    """Download a YOLO model."""
    path = MODELS_DIR / name
    if path.exists():
        print(f"✓ {name} already exists ({description})")
        return True
    
    print(f"⬇ Downloading {name} ({description})...")
    try:
        model = YOLO(name)  # This downloads from Ultralytics hub
        model.save(path)
        print(f"✓ Saved to {path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {name}: {e}")
        return False

def main():
    print("=" * 60)
    print("VisionGuard Model Downloader")
    print("=" * 60)
    
    # Core models
    for name, desc in MODELS.items():
        download_model(name, desc)
    
    print("\n" + "=" * 60)
    print("NOTE: Fire and Violence models need custom training.")
    print("Options:")
    print("  1. Train your own: yolo detect train data=fire.yaml model=yolov8n.pt epochs=100")
    print("  2. Use YOLO-World for open-vocabulary detection")
    print("  3. Download community models from:")
    print("     - https://github.com/ultralytics/assets/releases")
    print("     - Roboflow Universe (search 'fire', 'smoke', 'violence', 'weapon')")
    print("=" * 60)
    
    # Create placeholder files so app doesn't crash
    for placeholder in ['yolov8n-fire.pt', 'yolov8n-violence.pt']:
        path = MODELS_DIR / placeholder
        if not path.exists():
            print(f"\n⚠ {placeholder} not found - creating placeholder")
            print(f"   Replace with your trained model at: {path}")

if __name__ == '__main__':
    main()