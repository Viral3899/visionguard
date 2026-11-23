"""Regression coverage for Python 3.12 installer and modern CUDA wheel routing."""
from pathlib import Path
import pytest
from bootstrap_env import cuda_indices, require_python312, ROOT, GPU_CHECK


def test_python_312_x64_allowed():
    require_python312((3, 12, 11), 64)


@pytest.mark.parametrize('version,bits', [((3, 11, 9), 64), ((3, 13, 1), 64), ((3, 12, 2), 32)])
def test_wrong_python_version_or_architecture_rejected(version, bits):
    with pytest.raises(RuntimeError, match='Python 3.12'):
        require_python312(version, bits)


def test_cuda_wheels_match_driver_generation():
    assert cuda_indices((12, 8)) == ('cu126',)
    assert cuda_indices((13, 0)) == ('cu130', 'cu126')
    assert cuda_indices(None) == ('cu130', 'cu126')
    assert 'device="cuda"' in GPU_CHECK


def test_windows_install_launches_python312_and_runtime_check():
    installer = (ROOT / 'install_windows.bat').read_text()
    assert 'py -3.12' in installer
    assert 'py -3.11' not in installer
    assert 'verify_runtime.py' in installer
    trainer = (ROOT / 'TRAIN_AND_RUN.bat').read_text()
    assert 'verify_runtime.py --require-gpu' in trainer
