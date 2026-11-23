#!/usr/bin/env python3
"""
VisionGuard AI - Quick Start Script
One-click setup and run for the complete surveillance system.
"""
import os
import sys
import subprocess
import platform
import ctypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(cmd, **kwargs):
    """Run command with proper error handling."""
    print(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    return subprocess.run(cmd, **kwargs)

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"✗ Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    
    # Warn about Python 3.13+ CUDA limitations
    if version.major == 3 and version.minor >= 13:
        print("⚠ Python 3.13+ has limited CUDA wheel support on PyPI")
        print("  GPU acceleration may not be available, will use CPU")
    return True

def check_cuda() -> tuple:
    """Check CUDA availability using multiple methods."""
    cuda_version = ""
    
    # Method 1: nvidia-smi (most reliable)
    try:
        result = run(['nvidia-smi'], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    ver = line.split('CUDA Version:')[1].split()[0]
                    print(f"✓ CUDA {ver} detected via nvidia-smi")
                    return True, ver
            print("✓ NVIDIA GPU detected via nvidia-smi (version unknown)")
            return True, "unknown"
    except FileNotFoundError:
        pass
    except Exception:
        pass
    
    # Method 2: Windows WMI
    if platform.system() == 'Windows':
        try:
            result = run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], capture_output=True, text=True, check=False)
            if result.returncode == 0 and 'NVIDIA' in result.stdout.upper():
                print("✓ NVIDIA GPU detected via WMI")
                # Try to get CUDA version from registry
                cuda_version = get_windows_cuda_version()
                return True, cuda_version
        except:
            pass
    
    # Method 3: Linux lspci
    if platform.system() == 'Linux':
        try:
            result = run(['lspci'], capture_output=True, text=True, check=False)
            if result.returncode == 0 and 'NVIDIA' in result.stdout.upper():
                print("✓ NVIDIA GPU detected via lspci")
                return True, "unknown"
        except:
            pass
    
    # Method 4: Check CUDA toolkit (nvcc)
    cuda_version = get_cuda_toolkit_version()
    if cuda_version:
        print(f"✓ CUDA toolkit found: {cuda_version}")
        return True, cuda_version
    
    print("⚠ No CUDA GPU detected, will use CPU")
    return False, ""

def get_windows_cuda_version() -> str:
    """Try to get CUDA version on Windows from registry."""
    try:
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
    """Get CUDA version from nvcc."""
    try:
        result = run(['nvcc', '--version'], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    parts = line.split('release')
                    if len(parts) > 1:
                        ver = parts[1].strip().split(',')[0].strip()
                        return ver
    except:
        pass
    return ""

def get_pytorch_index_url(cuda_available: bool, cuda_version: str) -> str:
    """Get the appropriate PyTorch index URL."""
    if not cuda_available:
        return "https://download.pytorch.org/whl/cpu"
    
    if not cuda_version or cuda_version == "unknown":
        return "https://download.pytorch.org/whl/cu121"
    
    try:
        major_minor = '.'.join(cuda_version.split('.')[:2])
        major = int(major_minor.split('.')[0])
        minor = int(major_minor.split('.')[1])
    except:
        return "https://download.pytorch.org/whl/cu121"
    
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
        return "https://download.pytorch.org/whl/cu121"

def create_venv():
    """Create virtual environment."""
    venv_path = ROOT / '.venv'
    if venv_path.exists():
        print("✓ Virtual environment exists")
        return True
    
    print("Creating virtual environment...")
    result = run([sys.executable, '-m', 'venv', '.venv'], check=False)
    if result.returncode == 0:
        print("✓ Virtual environment created")
        return True
    else:
        print("✗ Failed to create virtual environment")
        return False

def get_python():
    """Get venv python path."""
    if platform.system() == 'Windows':
        return ROOT / '.venv' / 'Scripts' / 'python.exe'
    return ROOT / '.venv' / 'bin' / 'python'

def install_pytorch(python, cuda_available, cuda_version):
    """Install PyTorch with CUDA support, with fallback to CPU."""
    index_url = get_pytorch_index_url(cuda_available, cuda_version)
    
    if cuda_available:
        label = f"CUDA {cuda_version}" if cuda_version and cuda_version != "unknown" else "CUDA"
        print(f"Installing PyTorch with {label} support...")
    else:
        print("Installing CPU PyTorch...")
    
    # Try GPU version first
    result = run([str(python), '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', index_url], check=False)
    
    if result.returncode == 0:
        # Verify CUDA works
        verify = run([str(python), '-c', "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"], capture_output=True, text=True, check=False)
        if verify.returncode == 0 and 'CUDA: True' in verify.stdout:
            print("✓ PyTorch with CUDA installed and verified!")
            print(verify.stdout.strip())
            return True
        else:
            print("⚠ PyTorch installed but CUDA not working, falling back to CPU...")
    
    # Fallback to CPU
    print("Installing CPU PyTorch...")
    result = run([str(python), '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu'], check=False)
    if result.returncode == 0:
        print("✓ CPU PyTorch installed")
        return False
    else:
        print("✗ PyTorch installation failed")
        return None

def install_requirements(python):
    """Install all requirements."""
    print("Installing requirements...")
    result = run([str(python), '-m', 'pip', 'install', '-r', 'requirements.txt'], check=False)
    return result.returncode == 0

def run_setup(python):
    """Run the main setup script."""
    print("Running complete setup (downloading models, creating config)...")
    result = run([str(python), 'setup_all.py', '--skip-verify'], check=False)
    return result.returncode == 0

def run_server(python):
    """Run the VisionGuard server."""
    print("Starting VisionGuard server...")
    print("Dashboard: http://127.0.0.1:8000")
    print("API Docs:  http://127.0.0.1:8000/docs")
    print("Press Ctrl+C to stop\n")
    run([str(python), '-m', 'app.main'])

def main():
    print("=" * 60)
    print("VisionGuard AI - Quick Start")
    print("=" * 60)
    
    # Check Python
    if not check_python_version():
        sys.exit(1)
    
    # Check CUDA
    cuda_available, cuda_version = check_cuda()
    
    # Create venv
    if not create_venv():
        sys.exit(1)
    
    python = get_python()
    if not python.exists():
        print(f"✗ Python not found in venv: {python}")
        sys.exit(1)
    
    # Upgrade pip
    run([str(python), '-m', 'pip', 'install', '--upgrade', 'pip'], check=False)
    
    # Install PyTorch
    gpu_result = install_pytorch(python, cuda_available, cuda_version)
    if gpu_result is None:
        print("✗ PyTorch installation failed")
        sys.exit(1)
    elif gpu_result:
        print("✓ GPU acceleration enabled!")
    else:
        print("⚠ Running in CPU mode (no GPU acceleration)")
        print("  For GPU support, install NVIDIA drivers + CUDA Toolkit 11.8+ or 12.1+")
        print("  And use Python 3.8-3.12")
    
    # Install requirements
    if not install_requirements(python):
        print("✗ Requirements installation failed")
        sys.exit(1)
    
    # Run setup
    if not run_setup(python):
        print("✗ Setup failed")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ Setup complete! Starting server...")
    print("=" * 60 + "\n")
    
    # Run server
    run_server(python)

if __name__ == '__main__':
    main()