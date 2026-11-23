"""Install VisionGuard with exactly 64-bit Python 3.12 and GPU-aware PyTorch.

Launch with `py -3.12 bootstrap_env.py` on Windows. Preexisting virtual environments
from other Python versions are backed up rather than silently reused/deleted.
CUDA builds are installed from the official PyTorch wheel indices and tested by
actually allocating and computing on the GPU.
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
VENV = ROOT / '.venv'
PYTHON = VENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
CUDA_WHEEL_INDEX = 'https://download.pytorch.org/whl/'


def require_python312(version=None, bits=None):
    version = sys.version_info if version is None else version
    bits = bits if bits is not None else (64 if sys.maxsize > 2**32 else 32)
    if tuple(version[:2]) != (3, 12) or bits != 64:
        raise RuntimeError(
            f'Python 3.12 (64-bit) required; detected {version[0]}.{version[1]} ({bits}-bit). '
            'On Windows, install Python 3.12 x64 from https://www.python.org/downloads/ '
            'then run ONE_CLICK_SETUP_AND_RUN.bat (or py -3.12 bootstrap_env.py).'
        )


def call(*args):
    command = [str(x) for x in args]
    print('>', ' '.join(command), flush=True)
    subprocess.run(command, check=True, cwd=str(ROOT))


def existing_venv_is_python312():
    if not PYTHON.is_file():
        return False
    try:
        result = subprocess.run(
            [str(PYTHON), '-c', 'import sys; print(sys.version_info[:2] == (3,12) and sys.maxsize > 2**32)'],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0 and result.stdout.strip() == 'True'
    except (OSError, subprocess.TimeoutExpired):
        return False


def ensure_venv():
    if VENV.exists() and not existing_venv_is_python312():
        suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = ROOT / f'.venv_backup_{suffix}'
        sequence = 1
        while backup.exists():
            backup = ROOT / f'.venv_backup_{suffix}_{sequence}'
            sequence += 1
        print(f'Existing virtual environment is not Python 3.12 x64; moving it to {backup.name}')
        try:
            VENV.rename(backup)
        except OSError as exc:
            raise RuntimeError(
                f'Could not back up old .venv ({exc}). Close terminals using that '
                'environment and rerun the installer. Your existing files were not deleted.'
            ) from exc
    if not PYTHON.is_file():
        print('Creating Python 3.12 virtual environment:', VENV)
        venv.create(VENV, with_pip=True)
    if not existing_venv_is_python312():
        raise RuntimeError('Virtual environment is not Python 3.12 x64; installation stopped.')


def find_nvidia_smi():
    exe = shutil.which('nvidia-smi')
    if exe:
        return exe
    if os.name == 'nt':
        for candidate in (
            Path(os.getenv('WINDIR', r'C:\Windows')) / 'System32' / 'nvidia-smi.exe',
            Path(os.getenv('ProgramFiles', r'C:\Program Files')) / 'NVIDIA Corporation' / 'NVSMI' / 'nvidia-smi.exe',
        ):
            if candidate.is_file():
                return str(candidate)
    return None


def get_nvidia_gpu():
    """Return (found, max CUDA runtime supported by driver), not toolkit version."""
    exe = find_nvidia_smi()
    if not exe:
        return False, None
    try:
        result = subprocess.run(
            [exe, '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return False, None
        print('NVIDIA GPU(s):', result.stdout.strip())
        capability = subprocess.run([exe], capture_output=True, text=True, timeout=15)
        match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', capability.stdout)
        driver_cuda = (int(match.group(1)), int(match.group(2))) if match else None
        print('Driver-reported maximum CUDA version:', driver_cuda or 'unknown')
        return True, driver_cuda
    except (OSError, subprocess.TimeoutExpired):
        return False, None


def cuda_indices(driver_cuda):
    """Official wheel indices; CUDA 12.8 is no longer built in PyTorch 2.12+."""
    if driver_cuda is not None and driver_cuda < (13, 0):
        return ('cu126',)  # older driver: do not waste time downloading CUDA 13
    return ('cu130', 'cu126')


GPU_CHECK = (
    'import torch, torchvision; '
    'assert torch.cuda.is_available(), "CUDA unavailable"; '
    'x=torch.ones(4, device="cuda"); '
    'assert x.sum().item()==4.0; '
    'print("Verified CUDA tensor:", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
)
CPU_CHECK = (
    'import torch, torchvision; '
    'x=torch.ones(4, device="cpu"); assert x.sum().item()==4.0; '
    'print("Verified PyTorch CPU:", torch.__version__)'
)


def torch_check(require_gpu):
    return subprocess.run(
        [str(PYTHON), '-c', GPU_CHECK if require_gpu else CPU_CHECK],
        capture_output=True, text=True,
    )


def install_torch(require_gpu, driver_cuda, reinstall=False):
    status = torch_check(require_gpu)
    print('Current PyTorch check:', status.stdout.strip() or status.stderr.strip()[-500:])
    if status.returncode == 0 and not reinstall:
        return
    if require_gpu:
        for cuda in cuda_indices(driver_cuda):
            print(f'Installing PyTorch GPU build: {cuda}')
            try:
                call(PYTHON, '-m', 'pip', 'install', '--upgrade', '--force-reinstall',
                     'torch', 'torchvision', '--index-url', CUDA_WHEEL_INDEX + cuda)
            except subprocess.CalledProcessError as exc:
                print(f'[Install failed: {cuda}]', exc)
                continue
            check = torch_check(require_gpu=True)
            if check.returncode == 0:
                print('[GPU confirmed]', check.stdout.strip())
                return
            print(f'[CUDA {cuda} runtime failed]', check.stderr.strip()[-900:])
        raise RuntimeError(
            'NVIDIA GPU found but no usable CUDA PyTorch build was verified. '
            'Update the NVIDIA driver and retry (or explicitly run py -3.12 bootstrap_env.py --cpu). '
            'PyTorch is not reported as GPU-ready unless a CUDA tensor test passes.'
        )
    call(PYTHON, '-m', 'pip', 'install', '--upgrade', '--force-reinstall' if reinstall else '--upgrade',
         'torch', 'torchvision', '--index-url', CUDA_WHEEL_INDEX + 'cpu')
    check = torch_check(require_gpu=False)
    if check.returncode != 0:
        raise RuntimeError('PyTorch CPU check failed: ' + check.stderr[-800:])
    print('[CPU confirmed]', check.stdout.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true', help='Explicitly use CPU-only PyTorch')
    parser.add_argument('--reinstall-torch', action='store_true')
    args = parser.parse_args(argv)
    try:
        require_python312()
        ensure_venv()
        call(PYTHON, '-m', 'pip', 'install', '--upgrade', 'pip')
        detected, driver_cuda = get_nvidia_gpu()
        require_gpu = detected and not args.cpu
        if args.cpu:
            print('CPU mode explicitly selected; NVIDIA GPU will not be used.')
        elif not detected:
            print('No NVIDIA GPU detected through nvidia-smi; using CPU PyTorch.')
        install_torch(require_gpu, driver_cuda, args.reinstall_torch or (args.cpu and detected))
        call(PYTHON, '-m', 'pip', 'install', '-r', ROOT / 'requirements.txt')
        # Requirements may resolve a different torch wheel: ALWAYS retest after pip.
        post = torch_check(require_gpu)
        if post.returncode != 0:
            raise RuntimeError('PyTorch check failed after requirements installation: ' + post.stderr[-1000:])
        print('[Final PyTorch check]', post.stdout.strip())
        call(PYTHON, ROOT / 'verify_runtime.py', *(['--require-gpu'] if require_gpu else []))
        print('VisionGuard Python 3.12 installation finished successfully.')
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'INSTALL ERROR: {exc}\n')


if __name__ == '__main__':
    main()
