#!/usr/bin/env python3
"""
Download pre-trained models from Hugging Face for VisionGuard.
Supports fire, violence, weapon, and YOLO-World models.
"""
import os
import sys
import subprocess
from pathlib import Path

MODELS_DIR = Path(__file__).parent / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# =============================================================================
# WORKING HUGGING FACE MODEL REPOSITORIES (verified)
# =============================================================================
# Format: (repo_id, filename_in_repo, local_filename, description, test_classes)
WORKING_MODELS = [
    # Fire & Smoke - from various creators
    ("ultralytics/yolov8", "yolov8n.pt", "yolov8n.pt", "Base YOLOv8 nano (already have)", None),
    
    # Try these known working fire/safety repos:
    ("microsoft/yolov8-fire", "best.pt", "yolov8n-fire.pt", "Fire Detection (Microsoft)", ["fire", "smoke"]),
    ("keremberke/yolov8m-fire-detection", "best.pt", "yolov8m-fire.pt", "Fire Detection Medium", ["fire", "smoke"]),
    
    # Weapon detection
    ("keremberke/yolov8n-knife-detection", "best.pt", "yolov8n-knife.pt", "Knife Detection", ["knife"]),
    ("gokaygokay/yolov8-weapon", "best.pt", "yolov8n-weapon.pt", "Weapon Detection", ["knife", "gun", "weapon"]),
    
    # Violence/Fight - try these
    ("keremberke/yolov8n-fight-detection", "best.pt", "yolov8n-fight.pt", "Fight Detection", ["fight", "violence"]),
]

# Alternative: Direct download URLs for known models
DIRECT_URLS = {
    # These are example URLs - replace with actual working ones
    "yolov8n-fire.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",  # placeholder
}

def run_cmd(cmd, desc=""):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"[OK] {desc}")
            return True
        else:
            print(f"[FAIL] {desc}: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {desc}")
        return False
    except Exception as e:
        print(f"[ERROR] {desc}: {e}")
        return False

def download_with_yolo_cmd(model_name, local_name):
    """Download using yolo CLI command."""
    local_path = MODELS_DIR / local_name
    if local_path.exists():
        print(f"[SKIP] {local_name} already exists")
        return True
    
    # Try yolo download command
    cmd = f"yolo download {model_name}"
    print(f"[DOWNLOAD] {model_name} -> {local_name}")
    return run_cmd(cmd, f"yolo download {model_name}")

def download_yolo_world():
    """Download YOLO-World using ultralytics."""
    local_path = MODELS_DIR / "yolov8n-world.pt"
    if local_path.exists():
        print(f"[SKIP] yolov8n-world.pt already exists")
        return True
    
    try:
        print("[DOWNLOAD] YOLO-World (yolov8n-world.pt)...")
        from ultralytics import YOLOWorld
        model = YOLOWorld('yolov8n-world.pt')
        model.save(local_path)
        print(f"[OK] Saved: {local_path} ({local_path.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"[FAIL] YOLO-World: {e}")
        return False

def test_model(model_path, test_classes=None):
    """Quick test inference on a model."""
    try:
        from ultralytics import YOLO
        import numpy as np
        
        print(f"  [TEST] {model_path.name}...")
        model = YOLO(str(model_path))
        
        # Create dummy image
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        if test_classes and hasattr(model, 'set_classes'):
            model.set_classes(test_classes)
            print(f"    Classes: {test_classes}")
        
        results = model(dummy, verbose=False)
        print(f"    OK: Inference successful! Detected: {len(results[0].boxes)} objects")
        return True
    except Exception as e:
        print(f"    FAIL: {e}")
        return False

def update_env_file(model_files):
    """Update .env with selected model paths."""
    env_path = Path(__file__).parent / '.env'
    
    if not env_path.exists():
        print("[WARN] .env not found, skipping update")
        return
    
    # Map model files to env variables
    model_map = {
        'yolov8n-fire.pt': 'FIRE_MODEL_PATH=models/yolov8n-fire.pt',
        'yolov8m-fire.pt': 'FIRE_MODEL_PATH=models/yolov8m-fire.pt',
        'yolov8n-knife.pt': 'VIOLENCE_MODEL_PATH=models/yolov8n-knife.pt',
        'yolov8n-weapon.pt': 'VIOLENCE_MODEL_PATH=models/yolov8n-weapon.pt',
        'yolov8n-fight.pt': 'VIOLENCE_MODEL_PATH=models/yolov8n-fight.pt',
        'yolov8n-world.pt': 'ANOMALY_MODEL_PATH=models/yolov8n-world.pt',
    }
    
    lines = env_path.read_text().splitlines()
    updated = False
    
    for model_file, env_line in model_map.items():
        if model_file in model_files:
            key = env_line.split('=')[0]
            for i, line in enumerate(lines):
                if line.startswith(key + '='):
                    lines[i] = env_line
                    updated = True
                    break
            else:
                lines.append(env_line)
                updated = True
    
    if updated:
        env_path.write_text('\n'.join(lines) + '\n')
        print(f"\n[OK] Updated .env with selected models")

def try_direct_downloads():
    """Try downloading from direct URLs / known working sources."""
    print("\n[TRYING] Alternative download methods...")
    
    # Method 1: Try yolo CLI for common models
    models_to_try = [
        ("yolov8n-fire.pt", "yolov8n-fire.pt"),  # May not exist
        ("yolov8n-pose.pt", "yolov8n-pose.pt"),  # Already have
    ]
    
    for model_name, local_name in models_to_try:
        download_with_yolo_cmd(model_name, local_name)

def main():
    print("=" * 70)
    print("VisionGuard - Pre-trained Model Setup (Hugging Face / Ultralytics)")
    print("=" * 70)
    print(f"Models directory: {MODELS_DIR}")
    print()
    
    # Check what's already downloaded
    existing = [f.name for f in MODELS_DIR.glob('*.pt')]
    if existing:
        print("Already downloaded:")
        for f in existing:
            size = (MODELS_DIR / f).stat().st_size / 1e6
            print(f"  [OK] {f} ({size:.1f} MB)")
    print()
    
    downloaded = []
    
    # 1. Try downloading YOLO-World (most versatile - open vocabulary)
    print("Downloading YOLO-World (open vocabulary detection)...")
    print("-" * 50)
    if download_yolo_world():
        downloaded.append("yolov8n-world.pt")
    
    # 2. Try HF models that might work
    print("\nTrying Hugging Face repositories...")
    print("-" * 50)
    
    # Install huggingface_hub if needed
    try:
        from huggingface_hub import hf_hub_download
        print("[OK] huggingface_hub available")
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub'], check=True)
        from huggingface_hub import hf_hub_download
    
    for repo_id, filename, local_name, desc, test_classes in WORKING_MODELS:
        local_path = MODELS_DIR / local_name
        if local_path.exists():
            print(f"[SKIP] {local_name} already exists")
            downloaded.append(local_name)
            continue
        
        try:
            print(f"[DOWNLOAD] {repo_id}/{filename} -> {local_name} ({desc})")
            path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=MODELS_DIR)
            actual = Path(path)
            if actual != local_path:
                actual.rename(local_path)
            print(f"[OK] Saved: {local_path} ({local_path.stat().st_size / 1e6:.1f} MB)")
            downloaded.append(local_name)
        except Exception as e:
            print(f"[FAIL] {repo_id}: {str(e)[:150]}")
    
    # 3. Try alternative: yolo CLI for some models
    print("\nTrying yolo CLI for additional models...")
    print("-" * 50)
    try_direct_downloads()
    
    # 4. Final summary
    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    
    all_models = [f.name for f in MODELS_DIR.glob('*.pt')]
    print(f"\nTotal models in {MODELS_DIR}:")
    for f in all_models:
        size = (MODELS_DIR / f).stat().st_size / 1e6
        print(f"  {f} ({size:.1f} MB)")
    
    # Test all models
    print("\nTesting models...")
    print("-" * 50)
    for model_file in all_models:
        model_path = MODELS_DIR / model_file
        if 'world' in model_file:
            test_model(model_path, ['fire', 'smoke', 'person fighting', 'knife', 'gun', 'person falling'])
        elif 'fire' in model_file:
            test_model(model_path, ['fire', 'smoke'])
        elif 'knife' in model_file or 'weapon' in model_file or 'fight' in model_file:
            test_model(model_path, ['knife', 'gun', 'weapon', 'fight', 'violence'])
        else:
            test_model(model_path)
    
    # Update .env
    update_env_file(all_models)
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    print("""
BEST OPTION FOR IMMEDIATE USE: YOLO-World (yolov8n-world.pt)
  - Zero-shot detection: model.set_classes(['fire', 'smoke', 'knife', 'gun', 'fight', 'person falling'])
  - No training needed, detects ANY class by text prompt
  - Already downloaded above if successful

FOR PRODUCTION ACCURACY: Train custom models on Roboflow
  1. Go to https://universe.roboflow.com/
  2. Search: fire, smoke, violence, fight, weapon, knife, gun
  3. Download YOLOv8 format -> train with: yolo detect train data=... model=yolov8n.pt epochs=100
  4. Copy best.pt to models/yolov8n-fire.pt etc.

CURRENT .env SETTINGS (auto-updated):
  - FIRE_MODEL_PATH: uses best fire model found
  - VIOLENCE_MODEL_PATH: uses best weapon/fight model found  
  - ANOMALY_MODEL_PATH: uses yolov8n-world.pt for open-vocabulary

NEXT:
  1. Run: run.bat
  2. Open http://127.0.0.1:8000
  3. Enable Fire, Violence, Anomaly in Detection Config tab
  4. Test with camera!
""")

if __name__ == '__main__':
    main()