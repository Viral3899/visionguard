from pathlib import Path
from setup_models import MODEL_SPECS, check_model_file

def test_model_specs_present():
    assert 'yolov8n.pt' in MODEL_SPECS
    assert 'yolov8n-pose.pt' in MODEL_SPECS
    assert 'yolov26-fire-smoke.pt' in MODEL_SPECS
    assert 'threat-yolov8n.pt' in MODEL_SPECS
    assert 'fight-yolov8-nano.pt' in MODEL_SPECS

def test_check_model_file_on_existing():
    root = Path(__file__).resolve().parent.parent
    p = root / 'yolov8n.pt'
    if not p.is_file():
        p = root / 'models' / 'yolov8n.pt'
    if p.is_file():
        assert check_model_file(p) is True
