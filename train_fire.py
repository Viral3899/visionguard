"""Train a fire/smoke YOLO detector using audited D-Fire train/val/test splits.

The original YOLOv26 checkpoint is an independent pretrained inference option;
we do not assume its task/architecture matches an arbitrary YOLOv8 trainer.
"""
import argparse
import json
import math
from pathlib import Path
import shutil

from app.config import ROOT
from model_registry import MODELS, labels
from prepare_fire_dataset import OUTPUT, prepare
from train_models import audit_dataset


def train(epochs=40,imgsz=640,batch=-1, max_train_images=0,
          skip_download=False,allow_cpu=False,min_map50=.05,
          activate=False,source=None):
    if epochs < 1 or imgsz < 128: raise ValueError('Invalid epochs/imgsz')
    data=Path(source) if source else prepare(max_train_images, skip_download=skip_download)
    audit=audit_dataset(data)
    if audit['classes'] != ['smoke','fire']:
        raise ValueError(f'D-Fire must be smoke/fire, got {audit["classes"]}')
    import torch
    from ultralytics import YOLO
    device=0 if torch.cuda.is_available() else 'cpu'
    if device=='cpu' and not allow_cpu:
        raise RuntimeError('No CUDA GPU for fire training; use --allow-cpu explicitly')
    MODELS.mkdir(exist_ok=True)
    base=MODELS/'yolov8n.pt'
    model=YOLO(str(base) if base.is_file() else 'yolov8n.pt')
    model.train(data=str(data),epochs=epochs,imgsz=imgsz,batch=batch,workers=2,
                device=device,project=str(ROOT/'runs'/'train'),name='fire',exist_ok=True,
                patience=max(5,min(20,epochs//2)),plots=True,seed=42)
    best=Path(model.trainer.best)
    if not best.is_file():raise RuntimeError('No fire best checkpoint')
    trained=YOLO(str(best))
    if [x.lower() for x in labels(trained)]!=['smoke','fire']:
        raise RuntimeError('Trained fire classes mismatch D-Fire annotations')
    metrics=trained.val(data=str(data),split='val',device=device,imgsz=imgsz,
                        project=str(ROOT/'runs'/'val'),name='fire',exist_ok=True)
    map50=float(metrics.box.map50)
    if not math.isfinite(map50) or map50 < min_map50:
        raise RuntimeError(f'Fire model failed validation gate (mAP@50={map50}); not promoting')
    # Independent source test split is informative, NOT a guarantee of site performance.
    test_metrics=trained.val(data=str(data),split='test',device=device,imgsz=imgsz,
                             project=str(ROOT/'runs'/'test'),name='fire',exist_ok=True)
    report={'source':str(data),'device':str(device),'class_names':['smoke','fire'],
            'val_map50':map50,'test_map50':float(test_metrics.box.map50),
            'epochs':epochs,'production_accepted':False,
            'note':'Independent CCTV site evaluation is required; dataset test results alone do not establish deployability.'}
    target=MODELS/'visionguard-fire-trained.pt'
    staging=MODELS/'.fire-staging.pt'
    shutil.copy2(best,staging)
    staging.replace(target)
    report['trained_weights']=str(target)
    (MODELS/'fire_training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # No automatic deployment activation based ONLY on internal mAP.
    print(json.dumps(report,indent=2))
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--batch',type=int,default=-1)
    p.add_argument('--max-train-images',type=int,default=0)
    p.add_argument('--skip-download',action='store_true')
    p.add_argument('--allow-cpu',action='store_true')
    p.add_argument('--data',type=Path)
    a=p.parse_args()
    train(epochs=a.epochs,imgsz=a.imgsz,batch=a.batch,max_train_images=a.max_train_images,
          skip_download=a.skip_download,allow_cpu=a.allow_cpu,source=a.data)
