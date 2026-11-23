#!/usr/bin/env python3
"""
VisionGuard AI - Complete Setup Script
Installs GPU-enabled PyTorch, downloads all models from Hugging Face,
prepares datasets, and sets up the complete surveillance system.
"""
import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
import json

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / 'models'
DATA_DIR = ROOT / 'data'
DATASETS_DIR = ROOT / 'datasets'

# Hugging Face model repositories
HF_MODELS = {
    # Fire & Smoke Detection
    'fire_smoke': {
        'repo': 'SalahALHaismawi/yolov26-fire-detection',
        'filename': 'best.pt',  # Will try common names
        'local': 'yolov26-fire-smoke.pt',
        'classes': ['fire', 'smoke'],
    },
    
    # Violence/Fight Detection
    'violence': {
        'repo': 'Musawer14/fight_detection_yolov8',
        'filename': 'Yolo_nano_weights.pt',
        'local': 'fight-yolov8-nano.pt',
        'classes': ['fight', 'violence'],
    },
    
    # 3D CNN Violence Detection (Keras .h5 format, not PyTorch)
    # 'violence_3dcnn': {
    #     'repo': 'Musawer14/fight_detection_3DCNN',
    #     'filename': 'fight_detection_model.h5',
    #     'local': 'fight-3dcnn.h5',
    #     'classes': ['fight', 'violence'],
    # },
    
    # Weapon Detection - Threat Detection YOLOv8n
    'weapon': {
        'repo': 'Subh775/Threat-Detection-YOLOv8n',
        'filename': 'weights/best.pt',  # Actual path in repo
        'local': 'threat-detection-yolov8n.pt',
        'classes': ['gun', 'knife', 'weapon', 'threat'],
    },
    
    # Knife Detection - not available as public model yet
    # 'knife': {
    #     'repo': 'syzenzz/cctv-knife-detection-dataset',
    #     'filename': 'best.pt',
    #     'local': 'cctv-knife-detection.pt',
    #     'classes': ['knife'],
    # },
}

# Dataset repositories for training
HF_DATASETS = {
    'fire_smoke': 'SalahALHaismawi/yolov26-fire-detection',
    'violence': 'Musawer14/fight_detection_yolov8',
    'threat': 'Subh775/Threat-Detection-YOLOv8n',
}

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(msg: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{msg}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_step(msg: str):
    print(f"{Colors.BLUE}▶ {msg}{Colors.END}")

def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_error(msg: str):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def run_cmd(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run command and return result."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True)
    if check and result.returncode != 0:
        print_error(f"Command failed: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result

def check_cuda() -> Tuple[bool, str]:
    """Check if CUDA is available and get version.
    
    Uses multiple detection methods:
    1. nvidia-smi (most reliable)
    2. Windows WMI query
    3. Linux lspci
    4. Check if CUDA toolkit is installed
    """
    cuda_version = ""
    
    # Method 1: nvidia-smi (works on Windows and Linux)
    try:
        result = run_cmd(['nvidia-smi'], check=False)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    # Format: "CUDA Version: 12.4"
                    cuda_ver = line.split('CUDA Version:')[1].split()[0]
                    print_step(f"Detected CUDA via nvidia-smi: {cuda_ver}")
                    return True, cuda_ver
            # GPU found but couldn't parse version
            print_step("GPU detected via nvidia-smi (version unknown)")
            return True, "unknown"
    except FileNotFoundError:
        pass
    except Exception:
        pass
    
    # Method 2: Windows WMI query for GPU
    if platform.system() == 'Windows':
        try:
            result = run_cmd([
                'wmic', 'path', 'win32_VideoController', 'get', 'name'
            ], check=False)
            if result.returncode == 0 and 'NVIDIA' in result.stdout.upper():
                print_step("NVIDIA GPU detected via WMI")
                # Try to get CUDA version from registry or nvcuda.dll
                cuda_version = get_windows_cuda_version()
                return True, cuda_version
        except:
            pass
    
    # Method 3: Linux lspci
    if platform.system() == 'Linux':
        try:
            result = run_cmd(['lspci'], check=False)
            if result.returncode == 0 and 'NVIDIA' in result.stdout.upper():
                print_step("NVIDIA GPU detected via lspci")
                return True, "unknown"
        except:
            pass
    
    # Method 4: Check for CUDA toolkit installation
    cuda_version = get_cuda_toolkit_version()
    if cuda_version:
        print_step(f"CUDA toolkit found: {cuda_version}")
        return True, cuda_version
    
    return False, ""

def get_windows_cuda_version() -> str:
    """Try to get CUDA version on Windows from registry or DLL."""
    try:
        # Check nvcuda.dll version
        import ctypes
        nvcuda = ctypes.WinDLL('nvcuda')
        # If we can load it, CUDA is installed
        # Try to get version from registry
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA")
        i = 0
        while True:
            try:
                name, value, _ = winreg.EnumValue(key, i)
                if 'Version' in name:
                    return value
                i += 1
            except OSError:
                break
    except:
        pass
    return ""

def get_cuda_toolkit_version() -> str:
    """Get CUDA version from nvcc or CUDA installation."""
    try:
        result = run_cmd(['nvcc', '--version'], check=False)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    # Format: "Cuda compilation tools, release 12.4, V12.4.99"
                    parts = line.split('release')
                    if len(parts) > 1:
                        ver = parts[1].strip().split(',')[0].strip()
                        return ver
    except:
        pass
    return ""

def get_pytorch_index_url(cuda_available: bool, cuda_version: str) -> str:
    """Get the appropriate PyTorch index URL based on CUDA version."""
    if not cuda_available:
        return "https://download.pytorch.org/whl/cpu"
    
    # If version unknown, try latest stable CUDA 12.x
    if not cuda_version or cuda_version == "unknown":
        print_step("CUDA version unknown, trying CUDA 12.1 wheels (most compatible)")
        return "https://download.pytorch.org/whl/cu121"
    
    # Parse major.minor version
    try:
        major_minor = '.'.join(cuda_version.split('.')[:2])
        major = int(major_minor.split('.')[0])
        minor = int(major_minor.split('.')[1])
    except:
        print_warning(f"Could not parse CUDA version: {cuda_version}, defaulting to CUDA 12.1")
        return "https://download.pytorch.org/whl/cu121"
    
    # Map to PyTorch supported CUDA versions
    # PyTorch typically supports: cu118, cu121, cu124, cu126
    if major == 12:
        if minor >= 6:
            return "https://download.pytorch.org/whl/cu126"
        elif minor >= 4:
            return "https://download.pytorch.org/whl/cu124"
        elif minor >= 1:
            return "https://download.pytorch.org/whl/cu121"
        else:
            return "https://download.pytorch.org/whl/cu118"
    elif major == 11:
        if minor >= 8:
            return "https://download.pytorch.org/whl/cu118"
        else:
            return "https://download.pytorch.org/whl/cu118"
    else:
        print_warning(f"Unsupported CUDA major version: {major}, defaulting to CUDA 12.1")
        return "https://download.pytorch.org/whl/cu121"

def install_pytorch(cuda_available: bool, cuda_version: str) -> bool:
    """Install PyTorch with appropriate CUDA support.
    
    Tries GPU version first, falls back to CPU if it fails.
    """
    if not cuda_available:
        print_step("No CUDA GPU detected, installing CPU PyTorch")
        return install_pytorch_cpu()
    
    index_url = get_pytorch_index_url(cuda_available, cuda_version)
    cuda_label = f"CUDA {cuda_version}" if cuda_version and cuda_version != "unknown" else "CUDA"
    
    print_step(f"Installing PyTorch with {cuda_label} support from {index_url}")
    
    try:
        run_cmd([
            sys.executable, '-m', 'pip', 'install', 
            'torch', 'torchvision', 'torchaudio',
            '--index-url', index_url
        ])
        print_success(f"PyTorch with {cuda_label} support installed successfully")
        
        # Verify CUDA actually works
        if verify_cuda():
            print_success("CUDA verification passed!")
            return True
        else:
            print_warning("PyTorch installed but CUDA not working, falling back to CPU")
            return install_pytorch_cpu()
            
    except subprocess.CalledProcessError as e:
        print_warning(f"GPU PyTorch installation failed: {e}")
        print_step("Falling back to CPU PyTorch...")
        return install_pytorch_cpu()

def install_pytorch_cpu() -> bool:
    """Install CPU-only PyTorch."""
    try:
        run_cmd([
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cpu'
        ])
        print_success("CPU PyTorch installed successfully")
        return False
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install CPU PyTorch: {e}")
        return False

def verify_cuda() -> bool:
    """Verify CUDA is working with PyTorch."""
    try:
        result = run_cmd([
            sys.executable, '-c', 
            "import torch; "
            "cuda_ok = torch.cuda.is_available(); "
            "print(f'CUDA: {cuda_ok}'); "
            "if cuda_ok: print(f'GPU: {torch.cuda.get_device_name(0)}'); "
            "print(f'PyTorch: {torch.__version__}'); "
            "print(f'CUDA Version: {torch.version.cuda}')"
        ], check=False)
        
        if result.returncode == 0:
            print(result.stdout.strip())
            return 'CUDA: True' in result.stdout
        return False
    except Exception as e:
        print_warning(f"CUDA verification error: {e}")
        return False

def check_python_cuda_compatibility() -> Tuple[bool, str]:
    """Check if current Python version supports CUDA PyTorch wheels."""
    py_version = sys.version_info
    py_key = f"{py_version.major}.{py_version.minor}"
    
    # PyTorch CUDA wheels availability:
    # Python 3.8-3.12: Full CUDA support
    # Python 3.13+: Limited/No CUDA wheels on PyPI yet
    if py_version.major == 3 and py_version.minor <= 12:
        return True, py_key
    elif py_version.major == 3 and py_version.minor >= 13:
        return False, f"{py_key} (Python 3.13+ has limited CUDA wheel support on PyPI)"
    return True, py_key

def install_requirements() -> bool:
    """Install all requirements from requirements.txt."""
    print_step("Installing Python dependencies...")
    try:
        run_cmd([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print_success("All dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install requirements: {e}")
        return False

def download_hf_model(repo: str, filename: str, local_path: Path) -> bool:
    """Download model from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
        
        print_step(f"Downloading {filename} from {repo}...")
        weights = hf_hub_download(repo_id=repo, filename=filename)
        shutil.copy2(weights, local_path)
        print_success(f"Downloaded to {local_path}")
        return True
    except Exception as e:
        print_warning(f"Could not download {filename} from {repo}: {e}")
        # Try common alternative filenames
        for alt_name in ['best.pt', 'model.pt', 'weights.pt', 'yolov8n.pt', 'last.pt']:
            if alt_name == filename:
                continue
            try:
                print_step(f"Trying alternative: {alt_name}...")
                weights = hf_hub_download(repo_id=repo, filename=alt_name)
                shutil.copy2(weights, local_path)
                print_success(f"Downloaded {alt_name} to {local_path}")
                return True
            except:
                continue
        return False

def download_all_models() -> dict:
    """Download all models from Hugging Face."""
    print_header("Downloading Models from Hugging Face")
    
    MODELS_DIR.mkdir(exist_ok=True)
    
    results = {}
    for key, info in HF_MODELS.items():
        local_path = MODELS_DIR / info['local']
        if local_path.exists():
            print_success(f"Already exists: {info['local']}")
            results[key] = True
            continue
        
        success = download_hf_model(info['repo'], info['filename'], local_path)
        results[key] = success
        
        if not success:
            print_warning(f"Failed to download {key} - will use fallback")
    
    return results

def download_datasets() -> bool:
    """Download datasets for training if needed."""
    print_header("Checking/Downloading Training Datasets")
    
    DATASETS_DIR.mkdir(exist_ok=True)
    
    try:
        from huggingface_hub import snapshot_download
        
        for name, repo in HF_DATASETS.items():
            dataset_path = DATASETS_DIR / name
            if dataset_path.exists() and any(dataset_path.iterdir()):
                print_success(f"Dataset already exists: {name}")
                continue
            
            print_step(f"Downloading dataset: {repo}...")
            try:
                snapshot_download(
                    repo_id=repo,
                    repo_type='dataset',
                    local_dir=dataset_path,
                    local_dir_use_symlinks=False
                )
                print_success(f"Dataset downloaded: {name}")
            except Exception as e:
                print_warning(f"Could not download dataset {repo}: {e}")
        
        return True
    except ImportError:
        print_warning("huggingface_hub not available for dataset download")
        return False

def prepare_yolo_dataset(dataset_name: str) -> Optional[Path]:
    """Convert HF dataset to YOLO format."""
    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        return None
    
    # Look for data.yaml or create one
    yaml_path = dataset_path / 'data.yaml'
    if yaml_path.exists():
        return yaml_path
    
    # Try to find images and labels
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    
    if not images_dir.exists():
        # Try common structures
        for split in ['train', 'val', 'test']:
            img_dir = dataset_path / split / 'images'
            lbl_dir = dataset_path / split / 'labels'
            if img_dir.exists():
                images_dir = img_dir
                labels_dir = lbl_dir
                break
    
    if images_dir.exists():
        # Create data.yaml
        classes = HF_MODELS.get(dataset_name, {}).get('classes', ['object'])
        yaml_content = f"""path: {dataset_path}
train: {images_dir}
val: {images_dir}
names:
"""
        for i, cls in enumerate(classes):
            yaml_content += f"  {i}: {cls}\n"
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        return yaml_path
    
    return None

def train_model(model_key: str, epochs: int = 50, imgsz: int = 640, batch: int = 16) -> bool:
    """Train a YOLO model on downloaded dataset."""
    print_header(f"Training Model: {model_key}")
    
    info = HF_MODELS.get(model_key)
    if not info:
        print_error(f"Unknown model key: {model_key}")
        return False
    
    dataset_name = model_key.replace('_', '-').split('-')[0]  # rough mapping
    yaml_path = prepare_yolo_dataset(dataset_name)
    
    if not yaml_path:
        print_warning(f"No dataset found for {model_key}, skipping training")
        return False
    
    model_path = MODELS_DIR / info['local']
    base_model = 'yolov8n.pt'  # Start from pretrained
    
    try:
        from ultralytics import YOLO
        
        print_step(f"Loading base model: {base_model}")
        model = YOLO(base_model)
        
        print_step(f"Training for {epochs} epochs...")
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=0 if verify_cuda() else 'cpu',
            project=str(MODELS_DIR / 'runs'),
            name=f'train_{model_key}',
            exist_ok=True,
            patience=10,
            save_period=10,
        )
        
        # Copy best model to models dir
        best_path = Path(results.save_dir) / 'weights' / 'best.pt'
        if best_path.exists():
            shutil.copy2(best_path, model_path)
            print_success(f"Trained model saved to {model_path}")
            return True
        else:
            print_error("Training completed but best.pt not found")
            return False
            
    except Exception as e:
        print_error(f"Training failed: {e}")
        return False

def create_env_file() -> bool:
    """Create .env file with all model paths configured."""
    print_header("Creating Configuration (.env)")
    
    env_path = ROOT / '.env'
    if env_path.exists():
        print_success(".env already exists")
        return True
    
    env_content = """# =============================================================================
# VisionGuard AI Surveillance - Configuration
# Auto-generated by setup_all.py
# =============================================================================

# -----------------------------------------------------------------------------
# Server Settings
# -----------------------------------------------------------------------------
HOST=127.0.0.1
PORT=8000

# -----------------------------------------------------------------------------
# Core Person Detection (YOLOv8 + ByteTrack)
# -----------------------------------------------------------------------------
MODEL_PATH=yolov8n.pt
DEVICE=auto
CONFIDENCE=0.50
DETECT_CLASS_IDS=0
IMGSZ=416
MAX_FRAME_WIDTH=960
FRAME_SKIP=1
JPEG_QUALITY=72
SNAPSHOT_RETENTION_DAYS=30
EVENT_COOLDOWN_SECONDS=20
INCIDENT_MIN_CONFIDENCE=0.50
INCIDENT_CONFIRMATION_FRAMES=2

# -----------------------------------------------------------------------------
# Fire & Smoke Detection (SalahALHaismawi/yolov26-fire-detection)
# -----------------------------------------------------------------------------
FIRE_MODEL_PATH=models/yolov26-fire-smoke.pt
FIRE_CONFIDENCE=0.30
FIRE_IMGSZ=768
FIRE_CLASSES=0,1

# -----------------------------------------------------------------------------
# Violence / Fight Detection (Musawer14/fight_detection_yolov8)
# -----------------------------------------------------------------------------
VIOLENCE_MODEL_PATH=models/fight-yolov8-nano.pt
VIOLENCE_CONFIDENCE=0.25
VIOLENCE_IMGSZ=768
VIOLENCE_EVERY_N_FRAMES=2

# -----------------------------------------------------------------------------
# Weapon Detection (Subh775/Threat-Detection-YOLOv8n, shravya11/weapon-detection)
# -----------------------------------------------------------------------------
WEAPON_MODEL_PATH=models/threat-detection-yolov8n.pt
WEAPON_CONFIDENCE=0.25
WEAPON_IMGSZ=960
WEAPON_EVERY_N_FRAMES=2

# -----------------------------------------------------------------------------
# Knife Detection (syzenzz/cctv-knife-detection-dataset)
# -----------------------------------------------------------------------------
KNIFE_MODEL_PATH=models/cctv-knife-detection.pt
KNIFE_CONFIDENCE=0.30
KNIFE_IMGSZ=768
KNIFE_EVERY_N_FRAMES=2

# -----------------------------------------------------------------------------
# Anomaly Detection (Pose-based: Falls, Loitering)
# -----------------------------------------------------------------------------
ANOMALY_MODEL_PATH=yolov8n-pose.pt
ANOMALY_CONFIDENCE=0.30
ENABLE_FALL_DETECTION=true
ENABLE_LOITERING_DETECTION=false
LOITERING_TIME_SECONDS=30

# YOLO-World open vocabulary classes
YOLO_WORLD_CLASSES=fire,smoke,person fighting,knife,a person holding a knife,gun,a person holding a gun,weapon,pistol,rifle,blade,person falling

# -----------------------------------------------------------------------------
# General Object Detection (COCO classes)
# -----------------------------------------------------------------------------
OBJECT_MODEL_PATH=yolov8n.pt
OBJECT_CONFIDENCE=0.40
OBJECT_IMGSZ=960
OBJECT_EVERY_N_FRAMES=3
KNIFE_TILE_SCAN=false
OBJECT_CLASSES=43

# -----------------------------------------------------------------------------
# Telegram Alerts (optional)
# -----------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
TELEGRAM_ALERT_TYPES=fire_detected,smoke,violence,weapon,fall_detected,unusual_activity
TELEGRAM_COOLDOWN_SECONDS=60
TELEGRAM_MIN_CONFIDENCE=0.50

# -----------------------------------------------------------------------------
# TTS Voice Siren Alerts (pyttsx3 - offline Windows SAPI)
# -----------------------------------------------------------------------------
TTS_ENABLED=true
TTS_VOICE_RATE=180
TTS_VOICE_VOLUME=1.0
TTS_ALERT_TYPES=fire,smoke,violence,weapon,intrusion,fall_detected
SIREN_DURATION_SECONDS=10
SIREN_REPEAT_COUNT=3

# -----------------------------------------------------------------------------
# Dashboard Security
# -----------------------------------------------------------------------------
DASHBOARD_PASSWORD=
ADMIN_USERNAME=admin
APP_TIMEZONE=Asia/Kolkata
ADMIN_PASSWORD=
SESSION_SECRET=CHANGE_THIS_TO_A_LONG_RANDOM_SECRET
ALLOWED_ORIGINS=
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print_success(".env created with all model paths configured")
    return True

def create_run_script() -> bool:
    """Create cross-platform run script."""
    print_header("Creating Run Scripts")
    
    # Windows batch file
    bat_content = """@echo off
cd /d "%~dp0"
echo ============================================
echo VisionGuard AI Surveillance - Starting
echo ============================================
echo.

if not exist .venv\\Scripts\\python.exe (
  echo Virtual environment not found.
  echo Run python setup_all.py first.
  pause
  exit /b 1
)

echo Checking CUDA availability...
.venv\\Scripts\\python.exe -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.

echo Starting VisionGuard server...
echo Dashboard: http://127.0.0.1:8000
echo API Docs:  http://127.0.0.1:8000/docs
echo.
echo Press Ctrl+C to stop

.venv\\Scripts\\python.exe -m app.main
pause
"""
    
    with open(ROOT / 'run.bat', 'w') as f:
        f.write(bat_content)
    
    # Linux/Mac shell script
    sh_content = """#!/bin/bash
cd "$(dirname "$0")"
echo "============================================"
echo "VisionGuard AI Surveillance - Starting"
echo "============================================"
echo

if [ ! -f .venv/bin/python ]; then
  echo "Virtual environment not found."
  echo "Run python setup_all.py first."
  exit 1
fi

echo "Checking CUDA availability..."
.venv/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo

echo "Starting VisionGuard server..."
echo "Dashboard: http://127.0.0.1:8000"
echo "API Docs:  http://127.0.0.1:8000/docs"
echo
echo "Press Ctrl+C to stop"

.venv/bin/python -m app.main
"""
    
    with open(ROOT / 'run.sh', 'w') as f:
        f.write(sh_content)
    
    os.chmod(ROOT / 'run.sh', 0o755)
    
    print_success("Run scripts created (run.bat, run.sh)")
    return True

def verify_installation() -> bool:
    """Verify the installation works."""
    print_header("Verifying Installation")
    
    try:
        result = run_cmd([
            sys.executable, '-c', 
            "import torch; import cv2; import ultralytics; import fastapi; "
            "print(f'PyTorch: {torch.__version__}'); "
            "print(f'CUDA: {torch.cuda.is_available()}'); "
            "print(f'OpenCV: {cv2.__version__}'); "
            "print(f'Ultralytics: {ultralytics.__version__}'); "
            "print(f'FastAPI: {fastapi.__version__}')"
        ], check=False)
        
        if result.returncode == 0:
            print_success("All imports successful")
            print(result.stdout)
            return True
        else:
            print_error("Import verification failed")
            print(result.stderr)
            return False
    except Exception as e:
        print_error(f"Verification failed: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='VisionGuard AI Complete Setup')
    parser.add_argument('--cuda', action='store_true', help='Force CUDA installation')
    parser.add_argument('--cpu', action='store_true', help='Force CPU installation')
    parser.add_argument('--download-only', action='store_true', help='Only download models')
    parser.add_argument('--train', nargs='*', help='Train specific models (or all)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--skip-verify', action='store_true', help='Skip verification')
    parser.add_argument('--no-models', action='store_true', help='Skip model downloads')
    
    args = parser.parse_args()
    
    print_header("VisionGuard AI - Complete Setup")
    
    # Check Python/CUDA compatibility
    py_compat, py_info = check_python_cuda_compatibility()
    if not py_compat:
        print_warning(f"Python {py_info} - CUDA wheels may not be available on PyPI")
        print_step("Will attempt GPU install but may fall back to CPU")
    else:
        print_step(f"Python version: {py_info} (CUDA compatible)")
    
    # Check CUDA
    cuda_available, cuda_version = check_cuda()
    if args.cpu:
        cuda_available = False
        print_step("Forced CPU mode (--cpu flag)")
    elif args.cuda:
        cuda_available = True
        print_step("Forced CUDA mode (--cuda flag)")
    
    print_step(f"Platform: {platform.system()} {platform.release()}")
    print_step(f"Python: {sys.version.split()[0]}")
    print_step(f"CUDA Available: {cuda_available}" + (f" (v{cuda_version})" if cuda_version else ""))
    
    if cuda_available and not py_compat:
        print_warning("GPU detected but Python version may not have CUDA wheels")
        print_step("Attempting GPU install anyway...")
    
    # Create directories
    MODELS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    DATASETS_DIR.mkdir(exist_ok=True)
    
    # Install PyTorch
    if not args.download_only:
        gpu_success = install_pytorch(cuda_available, cuda_version)
        if gpu_success:
            verify_cuda()
        else:
            print_step("Running with CPU-only PyTorch")
            print_step("For GPU acceleration, ensure:")
            print_step("  1. NVIDIA GPU with drivers installed")
            print_step("  2. CUDA Toolkit 11.8+ or 12.1+ installed")
            print_step("  3. Python 3.8-3.12 (3.13+ has limited CUDA support)")
    
    # Install requirements
    if not args.download_only:
        install_requirements()
    
    # Download models
    if not args.no_models:
        download_all_models()
    
    # Download datasets
    if not args.download_only:
        download_datasets()
    
    # Train models if requested
    if args.train is not None:
        models_to_train = args.train if args.train else list(HF_MODELS.keys())
        for model_key in models_to_train:
            if model_key in HF_MODELS:
                train_model(model_key, epochs=args.epochs)
            else:
                print_warning(f"Unknown model: {model_key}")
    
    # Create config
    create_env_file()
    
    # Create run scripts
    create_run_script()
    
    # Verify
    if not args.skip_verify and not args.download_only:
        verify_installation()
    
    print_header("Setup Complete!")
    print(f"""
{Colors.GREEN}VisionGuard AI is ready!{Colors.END}

Next steps:
1. Run the server:
   {Colors.BOLD}Windows:{Colors.END}  run.bat
   {Colors.BOLD}Linux/Mac:{Colors.END} ./run.sh
   {Colors.BOLD}Direct:{Colors.END}  python -m app.main

2. Open dashboard: http://127.0.0.1:8000
3. API docs: http://127.0.0.1:8000/docs

Configure Telegram alerts in .env if needed:
  TELEGRAM_BOT_TOKEN=your_bot_token
  TELEGRAM_CHAT_ID=your_chat_id

Models directory: {MODELS_DIR}
Datasets directory: {DATASETS_DIR}
""")

if __name__ == '__main__':
    main()