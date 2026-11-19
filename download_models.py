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
        if not download_model(name, desc):
            raise SystemExit(1)
    
    print("\n" + "=" * 60)
    print("Fire weights are included in the release. Run python check_fire.py to verify your configuration.")
    print("Violence detection needs a separately qualified model.")

if __name__ == '__main__':
    main()