"""Fire model installer: optional --local to install pre-downloaded trusted .pt file."""
import argparse
import shutil
from pathlib import Path
from model_registry import MODELS, SPECS, labels
from install_security_models import verify_labels, install_pt

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--local',type=Path)
    p.add_argument('--accept-checkpoint-risk',action='store_true')
    a=p.parse_args()
    if not a.accept_checkpoint_risk: p.error('Pass --accept-checkpoint-risk for trusted third-party checkpoints')
    if a.local:
        from ultralytics import YOLO
        verify_labels('fire',labels(YOLO(str(a.local))))
        MODELS.mkdir(exist_ok=True)
        shutil.copy2(a.local,MODELS/SPECS[0].output)
    else:install_pt(SPECS[0])
if __name__=='__main__':main()
