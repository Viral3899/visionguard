"""Download *labeled* fight/normal videos and preserve upstream train/val/test splits.

Source: OmarNabill/event-detection (Fighting, Other video clips).
Original clips have different upstream licenses. Explicitly review and accept
these terms BEFORE use. Do not use automatic pseudo-labels as ground truth.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil

from app.config import ROOT

DATASET_ID = 'OmarNabill/event-detection'
RAW = ROOT/'data'/'datasets'/'raw'/'fight_event'
OUTPUT = ROOT/'data'/'fight_clips'
CLASS_MAP = {'fighting':'fight','other':'normal'}


def snapshot(directory=RAW):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=DATASET_ID,repo_type='dataset',
                                 allow_patterns=['videos/fighting/**','videos/other/**','metadata.csv',
                                                 'README.md','running_license_review.csv'],
                                 local_dir=str(directory)))


def import_clips(root,output=OUTPUT,max_per_class=0):
    root,output=Path(root),Path(output)
    metadata=root/'metadata.csv'
    if not metadata.is_file():
        raise FileNotFoundError('metadata.csv is required to preserve real clip labels and splits')
    rows=[]
    counts={}
    with metadata.open('r',encoding='utf-8-sig',newline='') as fp:
        for record in csv.DictReader(fp):
            label=CLASS_MAP.get(str(record.get('label','')).lower().strip())
            relative=record.get('file_name','').replace('\\','/')
            split=record.get('split','').lower().strip()
            if not label or split not in ('train','val','test'):
                continue
            if Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise ValueError('Unsafe upstream metadata path')
            if not relative.startswith('videos/'+('fighting' if label=='fight' else 'other')+'/'):
                continue
            file=root/relative
            if not file.is_file():
                continue
            if max_per_class and counts.get(label,0)>=max_per_class:
                continue
            counts[label]=counts.get(label,0)+1
            rows.append((file,label,split,relative,record.get('group','')))
    split_counts={(label,split):sum(x[1]==label and x[2]==split for x in rows)
                  for label in CLASS_MAP.values() for split in ('train','val')}
    if any(n < 5 for n in split_counts.values()):
        raise ValueError('Fighting/Other dataset must provide at least 5 source-labeled videos per class and split. '
                         f'Found {split_counts}')
    output.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for file,label,split,rel,group in rows:
        # Stable dataset prefix avoids basename collisions and does not flatten source identity.
        dest=output/label/(file.parent.name+'_'+file.name)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.is_file():
            try:os.link(file,dest)
            except OSError:shutil.copy2(file,dest)
        manifest.append({'file':str(dest.relative_to(output)).replace('\\','/'),
                         'label':label,'split':split,'group':group,'origin':rel})
    (output/'split_manifest.json').write_text(json.dumps({'source':DATASET_ID,'rows':manifest,
        'license_warning':'Review original fight_surv and other source terms. This ZIP does not redistribute any clip.'},
        indent=2),encoding='utf-8')
    return {'clips':len(manifest),'splits':{f'{a}_{b}':split_counts[(a,b)] for a,b in split_counts}}


def prepare(accept_license_review=False,skip_download=False,max_per_class=0):
    if not accept_license_review:
        raise PermissionError('Review OmarNabill/event-detection upstream licenses and pass --accept-license-review')
    root=RAW if skip_download else snapshot()
    return import_clips(root,max_per_class=max_per_class)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--accept-license-review',action='store_true')
    ap.add_argument('--skip-download',action='store_true')
    ap.add_argument('--max-per-class',type=int,default=0,help='0 uses ALL available labeled clips')
    a=ap.parse_args()
    print(json.dumps(prepare(a.accept_license_review,a.skip_download,a.max_per_class),indent=2))
