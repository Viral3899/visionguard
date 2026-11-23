"""Explicit, auditable Hugging Face model manifest for VisionGuard.

Third-party .pt files can contain executable pickle. Download only trusted repositories.
Only the known checkpoint filenames below are used, never '<weights>.pt'.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'

@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    filename: str
    output: str
    kind: str = 'yolo'

SPECS = (
    ModelSpec('fire', 'SalahALHaismawi/yolov26-fire-detection', 'best.pt', 'yolov26-fire-smoke.pt'),
    ModelSpec('fight', 'Musawer14/fight_detection_yolov8', 'Yolo_nano_weights.pt', 'fight-yolov8-nano.pt'),
    ModelSpec('threat', 'Subh775/Threat-Detection-YOLOv8n', 'weights/best.pt', 'threat-yolov8n.pt'),
    ModelSpec('3dcnn', 'Musawer14/fight_detection_3DCNN', 'fight_detection_model.h5', 'fight_detection_model.h5', 'keras'),
)

CORE_MODELS = ('yolov8n.pt', 'yolov8n-pose.pt')


def labels(model):
    raw = model.names
    return [str(v) for v in (raw.values() if isinstance(raw, dict) else raw)]
