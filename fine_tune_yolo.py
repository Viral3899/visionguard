"""Optional YOLO fine-tuning of fire/fight on externally supplied, annotated YOLO datasets.

No fire/fight datasets were supplied in this request. Requires a real data.yaml.
Automatically downloads no unrelated or unlicensed data.
"""
import argparse
import json
import shutil
from pathlib import Path

from model_registry import MODELS, ROOT, labels
from install_security_models import verify_labels

WEIGHTS = {'fire': ('yolov26-fire-smoke.pt','yolov26-fire-custom.pt'),
           'fight':('fight-yolov8-nano.pt','fight-yolov8-custom.pt'),
           'weapon':('threat-yolov8n.pt','visionguard-weapon-trained.pt')}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=WEIGHTS,required=True)
    p.add_argument('--data',type=Path,required=True,help='Your annotated Ultralytics data.yaml')
    p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--activate',action='store_true',help='For fire/fight, update .env to use custom weights')
    args=p.parse_args()
    if args.epochs<1:p.error('--epochs must be positive')
    if not args.data.is_file():p.error('Missing data.yaml')
    import torch
    from ultralytics import YOLO
    base_name, output_name=WEIGHTS[args.task]
    base=MODELS/base_name
    if not base.is_file():p.error('Missing '+str(base)+'; install checkpoints first')
    device=0 if torch.cuda.is_available() else 'cpu'
    model=YOLO(str(base))
    if getattr(model, 'task', 'detect') != 'detect':
        p.error('Cannot fine-tune a classification model with YOLO object boxes')
    model.train(data=str(args.data.resolve()),device=device,epochs=args.epochs,imgsz=args.imgsz,
                batch=-1,project=str(ROOT/'runs'/'train'),name='custom_'+args.task,exist_ok=True)
    best=Path(model.trainer.best)
    verified=YOLO(str(best))
    if getattr(verified, 'task', 'detect') != 'detect':
        p.error('Training created an unsupported non-detection model')
    verify_labels(args.task if args.task!='weapon' else 'weapon',labels(verified))
    metrics=verified.val(data=str(args.data.resolve()),device=device,imgsz=args.imgsz,
                         project=str(ROOT/'runs'/'val'),name='custom_'+args.task,exist_ok=True)
    dest=MODELS/output_name
    stage=MODELS/'.custom-staging.pt'
    shutil.copy2(best,stage);stage.replace(dest)
    print('Saved',dest,'Validation mAP@50:',float(metrics.box.map50))
    if args.activate and args.task in ('fire','fight'):
        env=ROOT/'.env'
        key='FIRE_MODEL_PATH' if args.task=='fire' else 'VIOLENCE_MODEL_PATH'
        if not env.is_file():p.error('Missing .env: run init_config.py first')
        text=env.read_text(encoding='utf-8')
        import re
        text,n=re.subn(r'(?m)^'+key+r'=.*$',key+'=models/'+output_name,text,count=1)
        if n==0:text+='\n'+key+'=models/'+output_name+'\n'
        env.write_text(text,encoding='utf-8')
        print('Activated checkpoint in .env; restart server.')
    print('Run external CCTV holdout evaluation before using for safety alerts.')

if __name__=='__main__':main()
