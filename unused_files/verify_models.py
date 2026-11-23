"""Inspect actual installed checkpoint metadata. No fake 'ready' statuses."""
import argparse
from pathlib import Path
from model_registry import MODELS, SPECS, CORE_MODELS, labels
from install_security_models import verify_labels

def verify():
    from ultralytics import YOLO
    rows=[]
    for spec in SPECS:
        path=MODELS/spec.output
        if not path.is_file():
            rows.append((spec.key,'MISSING','',str(path))); continue
        if spec.kind != 'yolo':
            rows.append((spec.key,'DOWNLOADED (inference opt-in; requires TensorFlow)','',str(path))); continue
        try:
            names=labels(YOLO(str(path)))
            verify_labels('fire' if spec.key=='fire' else 'fight' if spec.key=='fight' else 'weapon', names)
            rows.append((spec.key,'READY',','.join(names),str(path)))
        except Exception as exc:
            rows.append((spec.key,'INVALID',str(exc),str(path)))
    for filename in ('visionguard-weapon-trained.pt','visionguard-fire-trained.pt'):
        path=MODELS/filename
        if path.is_file():
            try:
                model=YOLO(str(path))
                names=labels(model)
                verify_labels('fire' if 'fire' in filename else 'fight' if 'fight' in filename else 'weapon',names)
                rows.append((filename,'READY',','.join(names),str(path)))
            except Exception as exc:rows.append((filename,'INVALID',str(exc),str(path)))
    for filename in CORE_MODELS:
        path=MODELS/filename
        rows.append((filename,'PRESENT' if path.is_file() else 'MISSING','',str(path)))
    classifier = MODELS / 'visionguard-r3d18-fight.pt'
    if classifier.is_file():
        try:
            import torch
            checkpoint = torch.load(classifier, map_location='cpu', weights_only=True)
            if (checkpoint.get('format') != 'visionguard-r3d18-v1' or
                    checkpoint.get('classes') != ['normal','fight'] or
                    checkpoint.get('frames') != 16 or checkpoint.get('size') != 112):
                raise ValueError('Unexpected classifier label/preprocessing metadata')
            rows.append(('torch_fight_classifier','READY (opt-in)', 'normal/fight',str(classifier)))
        except Exception as exc:
            rows.append(('torch_fight_classifier','INVALID',str(exc),str(classifier)))
    else:
        rows.append(('torch_fight_classifier','NOT TRAINED','requires labeled video clips',str(classifier)))
    for name, status, detail, path in rows:
        print(f'{name:16s} {status:20s} {path} {detail[:150]}')
    try:
        import torch
        print('Torch:',torch.__version__,'CUDA available:',torch.cuda.is_available())
        if torch.cuda.is_available():print('GPU:',torch.cuda.get_device_name(0))
    except ImportError:print('PyTorch not installed')
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--require',choices=['fire','fight','threat'])
    args=parser.parse_args()
    rows=verify()
    if args.require and not any(n==args.require and s=='READY' for n,s,_,_ in rows):raise SystemExit(1)
