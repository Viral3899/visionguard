"""Download audited D-Fire images/YOLO boxes, create distinct train/val/test.

Source hanvithSai/gavin-dfire-raw, original D-Fire, CC BY 4.0.
Source class ids are 0=smoke, 1=fire (NEVER guess alternate mapping).
Training data are not shipped in the ZIP; download needs disk + bandwidth.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

from app.config import ROOT
from prepare_datasets import IMAGE_EXT, valid_yolo

RAW = ROOT / 'data' / 'datasets' / 'raw' / 'dfire'
OUTPUT = ROOT / 'data' / 'datasets' / 'visionguard_fire'
DATASET_ID = 'hanvithSai/gavin-dfire-raw'
NAMES = {0:'smoke', 1:'fire'}


def fetch_dataset(directory=RAW):
    from huggingface_hub import snapshot_download
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    return Path(snapshot_download(repo_id=DATASET_ID,repo_type='dataset',
                                  allow_patterns=['train/images/**','train/labels/**',
                                                  'test/images/**','test/labels/**','README.md'],
                                  local_dir=str(directory)))


def pairs(root, split):
    imgs=root/split/'images'
    lbls=root/split/'labels'
    if not imgs.is_dir() or not lbls.is_dir():
        raise FileNotFoundError(f'Missing D-Fire {split}/images and {split}/labels under {root}')
    rows=[]
    for image in sorted(imgs.rglob('*')):
        if image.suffix.lower() not in IMAGE_EXT:
            continue
        label=lbls/image.relative_to(imgs).with_suffix('.txt')
        if not label.is_file():
            # Never invent negative labels when annotations are missing.
            continue
        raw_lines=label.read_text(encoding='utf-8').splitlines()
        valid=valid_yolo('\n'.join(raw_lines), {0:0,1:1})
        # Don't silently throw invalid positive boxes away and call images negative.
        if raw_lines and len(valid) != len([line for line in raw_lines if line.strip()]):
            continue
        rows.append((image,valid))
    return rows


def hardlink_or_copy(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    try:
        os.link(source,dest)
    except OSError:
        shutil.copy2(source,dest)


def materialize(source, output=OUTPUT, max_train_images=0, val_fraction=.15, seed=42):
    if not 0.05 <= val_fraction <= .35:
        raise ValueError('Validation fraction must be 0.05 to 0.35')
    source,output=Path(source),Path(output)
    train=pairs(source,'train')
    test=pairs(source,'test')
    if not train or not test:
        raise ValueError('D-Fire train/test annotation pairs are missing')
    rng=random.Random(seed)
    rng.shuffle(train)
    if max_train_images:
        if max_train_images < 20: raise ValueError('max_train_images must be >=20 or zero (all)')
        train=train[:max_train_images]
    n_val=max(2,round(val_fraction*len(train)))
    split_rows={'train':train[n_val:],'val':train[:n_val],'test':test}
    # Validate labels and duplicates across splits, without silently mixing train+test.
    counts=Counter(); hashes={}
    for split,rows in split_rows.items():
        for image, lines in rows:
            digest=hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest]!=split:
                raise ValueError(f'D-Fire exact duplicate between {hashes[digest]} and {split}: {image}')
            hashes[digest]=split
            for line in lines: counts[(split,int(line.split()[0]))]+=1
    for cls in NAMES:
        if counts[('train',cls)]<2 or counts[('val',cls)]<1 or counts[('test',cls)]<1:
            raise ValueError(f'{NAMES[cls]} labels missing from train, val or test')
    output.mkdir(parents=True,exist_ok=True)
    for split in split_rows:
        shutil.rmtree(output/split,ignore_errors=True)
    for split,rows in split_rows.items():
        for image, lines in rows:
            # Prefix source relative names to avoid stem collisions.
            relative=image.relative_to(source/('test' if split=='test' else 'train')/'images')
            output_name=(hashlib.sha1(str(relative).encode()).hexdigest()[:12]+'_'+image.name)
            dest=output/split/'images'/output_name
            hardlink_or_copy(image,dest)
            lbl=output/split/'labels'/(Path(output_name).stem+'.txt')
            lbl.parent.mkdir(parents=True,exist_ok=True)
            lbl.write_text('\n'.join(lines)+ ('\n' if lines else ''),encoding='utf-8')
    import yaml
    config={'path':str(output.resolve()),'train':'train/images',
            'val':'val/images','test':'test/images','names':NAMES,'nc':len(NAMES)}
    (output/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    report={'source':DATASET_ID,'license':'CC BY 4.0 (original D-Fire)',
            'source_classes':NAMES,'train_images':len(split_rows['train']),
            'val_images':len(split_rows['val']),'test_images':len(split_rows['test']),
            'label_boxes':{f'{split}_{NAMES[cls]}':counts[(split,cls)] for split in split_rows for cls in NAMES},
            'warnings':['D-Fire is mostly ground-level and wildfire imagery. Independent indoor CCTV validation is mandatory.',
                        'Image-level split does not guarantee different cameras/scenes between splits.',
                        'Existing raw test split is not used for training or checkpoint selection.']}
    (output/'dataset_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return output/'data.yaml'


def prepare(max_train_images=0,skip_download=False, source=RAW, output=OUTPUT):
    source=Path(source)
    if not skip_download:
        source=fetch_dataset(source)
    return materialize(source,output,max_train_images=max_train_images)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--skip-download',action='store_true')
    ap.add_argument('--max-train-images',type=int,default=0,help='0 uses ALL available image pairs')
    args=ap.parse_args()
    print(prepare(args.max_train_images,args.skip_download))
