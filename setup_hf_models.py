"""Install locally supplied specialist model weights without downloading from HF."""
import argparse
import hashlib
import shutil
from pathlib import Path
from model_registry import MODELS, labels
from install_security_models import verify_labels

def install(src,output,kind):
    from ultralytics import YOLO
    path=Path(src).expanduser().resolve()
    if path.suffix.lower()!='.pt' or not path.is_file():raise ValueError('Expected existing .pt checkpoint')
    names=labels(YOLO(str(path)))
    verify_labels(kind,names)
    MODELS.mkdir(exist_ok=True)
    dest=MODELS/output
    if path!=dest.resolve():shutil.copy2(path,dest)
    print('Installed',dest,'classes:',names,'SHA256:',hashlib.sha256(dest.read_bytes()).hexdigest())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weapon');p.add_argument('--violence');p.add_argument('--fire')
    args=p.parse_args()
    if not any((args.weapon,args.violence,args.fire)):p.error('Specify --weapon, --violence or --fire')
    if args.weapon:install(args.weapon,'threat-yolov8n.pt','weapon')
    if args.violence:install(args.violence,'fight-yolov8-nano.pt','fight')
    if args.fire:install(args.fire,'yolov26-fire-smoke.pt','fire')
if __name__=='__main__':main()
