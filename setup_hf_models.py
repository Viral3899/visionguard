#!/usr/bin/env python3
"""Safely install user-supplied, trusted custom weapon/violence YOLO weights.

The previous edition contained unverified Hugging Face repository names and
placeholder URLs. This command intentionally does not download or replace
bundled fire models from unverified sources.

Usage: python setup_hf_models.py --weapon C:\\Users\\me\\Downloads\\best.pt
       python setup_hf_models.py --violence C:\\Users\\me\\Downloads\\fight.pt
"""
import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from app.security_labels import weapon_class, is_fight_label


def install(source: str, target: Path, allowed: set[str]):
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file() or source_path.suffix.lower() != '.pt':
        raise ValueError(f'Cannot read a YOLO .pt checkpoint: {source_path}')
    # Torch .pt files may execute pickle code during loading. Only user-verified
    # weights from a trusted source should be passed to this operation.
    from ultralytics import YOLO
    model = YOLO(str(source_path))
    names = model.names
    labels = {str(v).strip().lower() for v in (names.values() if isinstance(names, dict) else names)}
    if not (any(weapon_class(x) for x in labels) if allowed == {'weapon'} else
            any(is_fight_label(x) for x in labels)) or len(labels) > 40:
        raise ValueError(f'The checkpoint does not look like a specialist model: {sorted(labels)}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_path != target.resolve():
        shutil.copyfile(source_path, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f'Installed {target.name} ({target.stat().st_size:,} bytes)')
    print(f'Model classes: {sorted(labels)}')
    print(f'SHA-256: {digest}')
    print('Validate performance on representative labelled videos before using for security alerts.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weapon', help='Path to a trusted specialist knife/weapon YOLO checkpoint')
    parser.add_argument('--violence', help='Path to a trusted trained violence/fight detector')
    args = parser.parse_args()
    if not (args.weapon or args.violence):
        parser.print_help()
        print('\nThere is no verified custom knife/fight checkpoint bundled in this project.')
        return
    if args.weapon:
        install(args.weapon, ROOT / 'models' / 'cs2-yolo12-weapon-detection.pt',
                {'weapon'})
    if args.violence:
        install(args.violence, ROOT / 'models' / 'fight-yolov8-nano.pt',
                {'violence'})


if __name__ == '__main__':
    main()
