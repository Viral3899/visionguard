"""Check 64-bit Python 3.12, PyTorch, torchvision, and real CUDA computation."""
import argparse
import sys


def verify(require_gpu=False):
    version = sys.version_info
    bits = 64 if sys.maxsize > 2**32 else 32
    print(f'Python: {version.major}.{version.minor}.{version.micro} ({bits}-bit)')
    if (version.major, version.minor, bits) != (3, 12, 64):
        print('ERROR: VisionGuard requires Python 3.12 x64. Re-run the setup installer.')
        return False
    try:
        import torch
        import torchvision
        print('PyTorch:', torch.__version__, 'TorchVision:', torchvision.__version__)
        print('PyTorch built for CUDA:', torch.version.cuda or 'none')
        available = torch.cuda.is_available()
        print('CUDA available:', available)
        if available:
            tensor = torch.ones(4, device='cuda')
            assert tensor.sum().item() == 4.0
            print('GPU:', torch.cuda.get_device_name(0))
            print('GPU computation: PASS')
        else:
            print('Execution mode: CPU')
        if require_gpu and not available:
            print('ERROR: GPU was required but CUDA is not available.')
            return False
        return True
    except Exception as exc:
        print('ERROR: PyTorch runtime check failed:', exc)
        return False


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--require-gpu', action='store_true')
    args = p.parse_args()
    raise SystemExit(0 if verify(args.require_gpu) else 1)
