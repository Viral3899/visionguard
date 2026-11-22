"""Train a REAL 3D CNN video classifier on user-labeled normal and fight clips.

Expected:
  data/fight_clips/normal/*.mp4
  data/fight_clips/fight/*.mp4

DO NOT train this classifier with knife bounding boxes or unlabeled files.
Original source recordings should not appear in both folders/splits. It does not
make fight *object boxes*, only an action label for a 16-frame video window.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import cv2

from app.torch_fight import FRAMES, SIZE, build_r3d, preprocess_frames
from model_registry import MODELS, ROOT

VALID = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v'}


def read_clip(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Unable to open video clip: {path}')
    frames = []
    try:
        while len(frames) < 2500:
            ok, frame = cap.read()
            if not ok:
                break
            # Resize BEFORE caching to avoid gigabytes of full-HD frames in RAM.
            frames.append(cv2.resize(frame, (SIZE, SIZE)))
    finally:
        cap.release()
    if len(frames) < FRAMES:
        raise ValueError(f'Clip requires at least {FRAMES} frames: {path}')
    # Uniformly sample across *the video* not duplicate the last still frame.
    indices = [round(i * (len(frames)-1) / (FRAMES-1)) for i in range(FRAMES)]
    return preprocess_frames([frames[i] for i in indices])


def catalog(folder, min_clips=5):
    folder = Path(folder)
    manifest = folder / 'split_manifest.json'
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        train, val = [], []
        from collections import defaultdict
        seen = set()
        counts = Counter()
        for entry in data['rows']:
            label = entry['label']
            split = entry['split']
            if label not in ('normal', 'fight') or split not in ('train', 'val', 'test'):
                raise ValueError('Invalid upstream split label')
            relative = Path(entry['file'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe clip manifest path')
            path = folder / relative
            if not path.is_file():
                raise FileNotFoundError(f'Missing metadata-labeled video {path}')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                raise ValueError('Duplicate fight video in the upstream manifest')
            seen.add(digest)
            target = train if split == 'train' else val if split == 'val' else None
            if target is not None:
                target.append((path, int(label == 'fight')))
                counts[(label,split)] += 1
        for label in ('normal','fight'):
            for split in ('train','val'):
                if counts[(label,split)] < min_clips:
                    raise ValueError(f'{label} {split} needs {min_clips} real labeled clips')
        return train, val
    random_seed = 42
    groups = {}
    all_digests = set()
    for index, name in enumerate(('normal','fight')):
        videos = sorted(p for p in (folder / name).rglob('*') if p.suffix.lower() in VALID) if (folder/name).is_dir() else []
        unique = []
        for path in videos:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in all_digests:
                raise ValueError('Duplicate video found across action classes or within dataset: '+str(path))
            all_digests.add(digest)
            unique.append((path, index))
        if len(unique) < min_clips:
            raise ValueError(f'{name}: need >= {min_clips} independently labeled video clips; found {len(unique)}. '
                             'This is not a dataset that can be auto-labeled safely.')
        rng = random.Random(random_seed + index)
        rng.shuffle(unique)
        groups[name] = unique
    train = []
    val = []
    for group in groups.values():
        n_val = max(1, round(len(group)*.2))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


def train(folder, epochs=10, batch_size=2, lr=3e-4, activate=False, allow_cpu=False, min_class_recall=.40):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch must be positive')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and not allow_cpu:
        raise RuntimeError('No CUDA GPU; R3D-18 is expensive on CPU. Use --allow-cpu to opt in.')
    training, validation = catalog(folder)

    class Clips(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, index):
            path, label = self.rows[index]
            return read_clip(path), torch.tensor(label, dtype=torch.long)

    trainloader = DataLoader(Clips(training), batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(Clips(validation), batch_size=batch_size, shuffle=False, num_workers=0)
    net = build_r3d(pretrained=True).to(device)
    # For a limited set of videos, train the action head, not an entire random network.
    for name, parameter in net.named_parameters():
        parameter.requires_grad = name.startswith('fc.')
    opt = torch.optim.Adam(net.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_class_recall = 0.0
    best_state = None
    history = []
    for epoch in range(epochs):
        # Keep frozen R3D backbone BatchNorm running statistics unchanged.
        net.eval()
        net.fc.train()
        seen = 0
        loss_sum = 0.0
        for clips, label in trainloader:
            clips, label = clips.to(device), label.to(device)
            opt.zero_grad()
            loss = criterion(net(clips), label)
            loss.backward()
            opt.step()
            seen += len(label)
            loss_sum += float(loss.item()) * len(label)
        net.eval()
        correct = Counter()
        totals = Counter()
        with torch.inference_mode():
            for clips, label in valloader:
                predictions = net(clips.to(device)).argmax(1).cpu().tolist()
                for true, pred in zip(label.tolist(), predictions):
                    totals[true] += 1
                    correct[true] += int(true == pred)
        acc = sum(correct.values()) / max(1,sum(totals.values()))
        report = {'epoch':epoch+1, 'train_loss':loss_sum/max(1,seen), 'val_accuracy':acc,
                  'normal_recall':correct[0]/max(1,totals[0]),
                  'fight_recall':correct[1]/max(1,totals[1])}
        history.append(report)
        print(report,flush=True)
        class_recall = min(report['normal_recall'], report['fight_recall'])
        # Select for balanced performance rather than predicting only 'normal'.
        if class_recall > best_class_recall or (class_recall == best_class_recall and acc > best_accuracy):
            best_class_recall = class_recall
            best_accuracy = acc
            best_state = {name: tensor.detach().cpu().clone() for name,tensor in net.state_dict().items()}
    if best_state is None or best_class_recall < min_class_recall:
        raise RuntimeError(f'Fight classifier recall for weakest class={best_class_recall:.3f} '
                           f'below gate {min_class_recall:.3f}; not promoting or activating')
    MODELS.mkdir(exist_ok=True)
    path = MODELS / 'visionguard-r3d18-fight.pt'
    stage = path.with_suffix('.staging')
    torch.save({'format':'visionguard-r3d18-v1','classes':['normal','fight'],
                'frames':FRAMES,'size':SIZE, 'state_dict':best_state}, stage)
    stage.replace(path)
    summary = {'trained_file':str(path), 'device':device, 'train_clips':len(training),
               'val_clips':len(validation), 'best_holdout_accuracy':best_accuracy, 'weakest_class_recall':best_class_recall, 
               'history':history, 'notes':
                 'Small random split by video file is not independent of the underlying camera/scene. '
                 'Do not claim production accuracy without a separate, externally labeled test set.'}
    (MODELS/'fight_classifier_report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if activate:
        env = ROOT / '.env'
        if not env.is_file():
            raise FileNotFoundError('Run setup before --activate; .env missing')
        import re
        content = env.read_text(encoding='utf-8')
        content, n = re.subn(r'(?m)^ENABLE_FIGHT_CLASSIFIER=.*$', 'ENABLE_FIGHT_CLASSIFIER=true', content)
        if not n: content += '\nENABLE_FIGHT_CLASSIFIER=true\n'
        env.write_text(content, encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data'/'fight_clips')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--activate',action='store_true')
    p.add_argument('--allow-cpu',action='store_true')
    a=p.parse_args()
    train(a.data,a.epochs,a.batch,activate=a.activate,allow_cpu=a.allow_cpu)
