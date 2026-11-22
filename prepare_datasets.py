"""Download annotated CCTV weapons datasets and prepare a clean YOLO DETECTION dataset.

Only boxes from YOLO and Pascal VOC annotations are used. Image classification
folders are explicitly excluded. Files are split at *image* level with SHA256
exact duplicate removal; footage from different frames/cameras must additionally
be separated at recording level in an external independent evaluation set.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'datasets' / 'raw'
PREPARED = ROOT / 'data' / 'datasets' / 'visionguard_weapon'
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
# Internal/source IDs: 0=person is reserved but MUST NOT be an output class.
# Person detection comes from the separate COCO model. Training the weapon
# checkpoint on 'person' would make weapon-free footage misleadingly labeled.
CLASS_NAMES = ['person [EXCLUDED]', 'knife', 'gun']
VOC_MAP = {'knife':1, 'blade':1, 'dagger':1, 'machete':1,
           'gun':2, 'pistol':2, 'handgun':2, 'firearm':2, 'rifle':2, 'revolver':2}


def fetch(repo, dest, allow_patterns):
    from huggingface_hub import snapshot_download
    print('Downloading annotated dataset:', repo, flush=True)
    return Path(snapshot_download(repo_id=repo, repo_type='dataset',
                                  allow_patterns=allow_patterns, local_dir=str(dest)))


def valid_yolo(text, mapping):
    result = []
    for line in text.splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        try:
            source_id = int(values[0])
            x, y, w, h = (float(x) for x in values[1:])
        except (ValueError, OverflowError):
            continue
        if source_id not in mapping:
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            continue
        if x-w/2 < -0.001 or x+w/2 > 1.001 or y-h/2 < -0.001 or y+h/2 > 1.001:
            continue
        result.append(f'{mapping[source_id]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}')
    return result


def parse_voc(xml_file):
    try:
        root = ET.parse(xml_file).getroot()
    except (ET.ParseError, OSError):
        return []
    size = root.find('size')
    if size is None:
        return []
    try:
        width = float(size.findtext('width'))
        height = float(size.findtext('height'))
        if width <= 0 or height <= 0:
            return []
    except (TypeError, ValueError):
        return []
    lines = []
    for obj in root.findall('object'):
        name = ' '.join((obj.findtext('name') or '').strip().lower().split())
        if name not in VOC_MAP:
            continue
        bbox = obj.find('bndbox')
        if bbox is None:
            continue
        try:
            x1, y1, x2, y2 = [float(bbox.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
        except (TypeError, ValueError):
            continue
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f'{VOC_MAP[name]} {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} '
                     f'{(x2-x1)/width:.6f} {(y2-y1)/height:.6f}')
    return lines


def collect_syzenzz(path):
    directory = path / 'Knife_Dataset'
    images, labels = directory / 'images', directory / 'labels'
    if not images.is_dir() or not labels.is_dir():
        raise RuntimeError('Missing Knife_Dataset/images or Knife_Dataset/labels in syzenzz download')
    for img in sorted(images.rglob('*')):
        if img.suffix.lower() not in IMAGE_EXT:
            continue
        relative = img.relative_to(images)
        label = labels / relative.with_suffix('.txt')
        if not label.is_file():
            continue
        # Source YAML: 0 person, 1 knife (NOT our threat pretrained model classes).
        lines = valid_yolo(label.read_text(encoding='utf-8'), {1:1})
        if lines:
            yield img, lines, 'syzenzz'


def collect_shravya(path):
    """Pair XML and image in matching parts, not by stem across whole repo.

    The repository has distinct part directories with potentially repeated stems.
    A global stem cache would silently attach a box to the WRONG image.
    """
    for category in ('Knife_detection', 'Pistol detection'):
        directory = path / category
        if not directory.is_dir():
            continue
        for xml in sorted(directory.rglob('*.xml')):
            lines = parse_voc(xml)
            if not lines:
                continue
            parent = xml.parent
            # Common HF layout: part_1/annotations/X.xml + part_1/images/X.jpg
            candidates = [parent, parent.parent / 'images', parent.parent / 'Images',
                          parent.parent / 'JPEGImages', directory / 'images']
            found = []
            for folder in candidates:
                for ext in IMAGE_EXT:
                    p = folder / f'{xml.stem}{ext}'
                    if p.is_file():
                        found.append(p)
            # Only pair unambiguous annotation-to-image paths.
            matches = sorted(set(found))
            if len(matches) == 1:
                yield matches[0], lines, 'shravya'


def materialize(records, root=PREPARED, seed=42, val_fraction=0.2,
                negatives_dir=None):
    """Make deterministic disjoint train/val sets; omit classes with no labels.

    Negative images need *verified* absence of target objects. We never assume
    arbitrary unlabeled classification examples are clean negative images.
    """
    if not 0 < val_fraction < 0.5:
        raise ValueError('Expected 0 < validation fraction < 0.5')
    unique = {}
    duplicate_count = 0
    for img, lines, origin in records:
        if not img.is_file() or not lines:
            continue
        digest = hashlib.sha256(img.read_bytes()).hexdigest()
        if digest in unique:
            duplicate_count += 1
            if unique[digest][1] != lines:
                raise ValueError(f'Conflicting annotations for duplicated image {img}')
            continue
        unique[digest] = (img, lines, origin)
    if negatives_dir is not None:
        for img in Path(negatives_dir).rglob('*'):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXT:
                continue
            digest = hashlib.sha256(img.read_bytes()).hexdigest()
            if digest in unique and unique[digest][1]:
                raise ValueError('A verified-negative image duplicates a positive bounding-box example: ' + str(img))
            if digest not in unique:
                unique[digest] = (img, [], 'verified_negative')
    if len(unique) < 5:
        raise RuntimeError(f'Only {len(unique)} images available; at least 5 required')
    active = sorted({int(line.split()[0]) for _, lines, _ in unique.values() for line in lines})
    if not active:
        raise RuntimeError('No object bounding boxes were found; cannot train a detector')
    if not set(active).issubset({1,2}):
        raise ValueError('Weapon-only training requires knife/gun boxes; do not train a person class')
    new_ids = {source: new for new, source in enumerate(active)}
    names = {new: CLASS_NAMES[source] for source, new in new_ids.items()}
    items = sorted(unique.items())
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = max(1, round(len(items) * val_fraction))
    # Preserve at least one labeled example of each adequately represented class
    # on both sides where feasible. No claim of independent real-camera coverage.
    candidates = items[:n_val]
    training = items[n_val:]
    for source in active:
        group = lambda seq: sum(any(int(line.split()[0]) == source for line in row[1][1]) for row in seq)
        if group(candidates) == 0 and group(training) >= 2:
            candidate = next((item for item in training if any(int(l.split()[0]) == source for l in item[1][1])), None)
            replacement = next((item for item in candidates if not any(int(l.split()[0]) == source for l in item[1][1])), None)
            if candidate is not None and replacement is not None:
                training.remove(candidate)
                candidates.remove(replacement)
                candidates.append(candidate)
                training.append(replacement)
        if group(training) == 0 and group(candidates) >= 2:
            candidate = next((item for item in candidates if any(int(l.split()[0]) == source for l in item[1][1])), None)
            replacement = next((item for item in training if not any(int(l.split()[0]) == source for l in item[1][1])), None)
            if candidate is not None and replacement is not None:
                candidates.remove(candidate)
                training.remove(replacement)
                training.append(candidate)
                candidates.append(replacement)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val'):
        shutil.rmtree(root / split, ignore_errors=True)
    counts = Counter()
    box_counts = {'train': Counter(), 'val': Counter()}
    for split, seq in [('train', training), ('val', candidates)]:
        for digest, (img, lines, origin) in seq:
            out_img = root / split / 'images' / (digest[:32] + img.suffix.lower())
            out_lbl = root / split / 'labels' / (digest[:32] + '.txt')
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out_img)
            converted = []
            for line in lines:
                fields = line.split()
                index = int(fields[0])
                converted.append(str(new_ids[index]) + ' ' + ' '.join(fields[1:]))
                box_counts[split][CLASS_NAMES[index]] += 1
            out_lbl.write_text('\n'.join(converted) + ('\n' if converted else ''), encoding='utf-8')
            counts[split] += 1
            counts[f'{origin}_{split}'] += 1
            if not lines:
                counts[f'{split}_negative_images'] += 1
    import yaml
    config = {'path': str(root.resolve()), 'train':'train/images', 'val':'val/images',
              'nc':len(names), 'names':names}
    yaml_path = root / 'data.yaml'
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    report = {'images':len(items), 'exact_duplicates_removed':duplicate_count,
              'counts':dict(counts), 'classes':list(names.values()),
              'boxes_by_class':{k:dict(v) for k,v in box_counts.items()},
              'seed':seed, 'val_fraction':val_fraction,
              'warnings': [
                  'Synthetic CCTV sample is only 114 pictures, insufficient alone for reliable deployment.',
                  'Train/val files are exactly disjoint, but perceptual near-duplicates and scene leakage are not excluded.',
                  'Independent test camera footage with verified negatives is required for false-positive evaluation.',
                  'No output classes without source bounding boxes are created.',
                  'The person class is excluded: core COCO model already detects people.']}
    (root / 'dataset_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return yaml_path


def prepare(source='both', root=PREPARED, dry_run=False, negatives_dir=None):
    records = []
    if source in ('both', 'syzenzz'):
        if dry_run:
            print('Would download syzenzz synthetic CCTV knife dataset (~175 MB)')
        else:
            path = fetch('syzenzz/cctv-knife-detection-dataset', RAW/'syzenzz', ['Knife_Dataset/**'])
            records.extend(collect_syzenzz(path))
    if source in ('both', 'shravya'):
        if dry_run:
            print('Would download shravya11 only detection annotation subsets (not classification images)')
        else:
            path = fetch('shravya11/weapon-detection', RAW/'shravya',
                         ['Knife_detection/**', 'Pistol detection/**', 'License.md'])
            records.extend(collect_shravya(path))
    if dry_run:
        return None
    if not records:
        raise RuntimeError('No valid annotated images found; check source directory layouts and download logs.')
    return materialize(records, root, negatives_dir=negatives_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=['syzenzz','shravya','both'], default='both')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--output', type=Path, default=PREPARED)
    p.add_argument('--verified-negatives', type=Path, help='Images confirmed to have no target objects')
    a = p.parse_args()
    prepare(a.source, a.output, a.dry_run, a.verified_negatives)
