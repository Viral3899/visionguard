"""Download verified-by-filename Hugging Face repos and validate specialist classes."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from model_registry import SPECS, MODELS, ROOT, CORE_MODELS, labels
from app.security_labels import weapon_class, is_fight_label


def verify_labels(kind, names):
    cleaned = [str(x).strip() for x in names]
    if kind == 'weapon':
        if not any(weapon_class(x) for x in cleaned):
            raise ValueError('Missing recognized knife/gun/weapon labels: ' + repr(cleaned))
    elif kind == 'fight':
        if not any(is_fight_label(x) for x in cleaned):
            raise ValueError('Missing a positive fight/violence class: ' + repr(cleaned))
    elif kind == 'fire':
        if not any(x.lower() in ('fire', 'smoke', 'flame', 'flames') for x in cleaned):
            raise ValueError('Missing fire/smoke labels: ' + repr(cleaned))
    else:
        raise ValueError('Unknown label class group ' + str(kind))
    return True


def install_pt(spec, *, force=False, dry_run=False):
    """Download first, validate labels, then atomically move checkpoint into production."""
    dest = MODELS / spec.output
    if dest.is_file() and dest.stat().st_size > 100_000 and not force:
        print(f'[cached] {dest}'); return dest
    print('[download]', spec.repo, spec.filename)
    if dry_run: return dest
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    src = Path(hf_hub_download(repo_id=spec.repo, filename=spec.filename))
    MODELS.mkdir(exist_ok=True)
    staging = MODELS / f'.{spec.key}-checking.pt'
    try:
        shutil.copyfile(src, staging)
        model = YOLO(str(staging))
        actual = labels(model)
        verify_labels('fire' if spec.key == 'fire' else 'fight' if spec.key == 'fight' else 'weapon', actual)
        print('[verified labels]', spec.key, actual)
        staging.replace(dest)
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    print('[installed]', dest, hashlib.sha256(dest.read_bytes()).hexdigest()[:16])
    return dest


def download_base(dry_run=False):
    from ultralytics import YOLO
    MODELS.mkdir(exist_ok=True)
    for filename in CORE_MODELS:
        path = MODELS / filename
        if path.is_file():
            print('[cached]', path); continue
        print('[download] Ultralytics', filename)
        if dry_run: continue
        model = YOLO(filename)
        from_path = Path(getattr(model, 'ckpt_path', filename))
        if not from_path.is_file():
            raise FileNotFoundError('Missing downloaded base checkpoint ' + str(from_path))
        shutil.copyfile(from_path, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-checkpoint-risk', action='store_true', help='I trust these third-party pickle checkpoints')
    parser.add_argument('--include-3dcnn', action='store_true', help='Also download optional Keras model (not enabled automatically)')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--only', choices=['fire','fight','threat','3dcnn'], help='Install only one checkpoint')
    parser.add_argument('--skip-base', action='store_true')
    args = parser.parse_args(argv)
    if not args.accept_checkpoint_risk and not args.dry_run:
        parser.error('You must explicitly acknowledge risk with --accept-checkpoint-risk. See README.')
    if not args.skip_base and not args.only and not args.dry_run:
        download_base()
    wanted = [s for s in SPECS if (s.key == args.only if args.only else s.key != '3dcnn' or args.include_3dcnn)]
    status = {}
    for spec in wanted:
        try:
            if spec.kind == 'yolo':
                path = install_pt(spec, force=args.force, dry_run=args.dry_run)
            else:
                from huggingface_hub import hf_hub_download
                dest = MODELS / spec.output
                if not dest.is_file() or args.force:
                    print('[download optional]', spec.repo, spec.filename)
                    if not args.dry_run:
                        src = hf_hub_download(repo_id=spec.repo, filename=spec.filename)
                        MODELS.mkdir(exist_ok=True)
                        shutil.copyfile(src, dest)
                path = dest
            status[spec.key] = {'status': ('ready' if spec.kind=='yolo' else 'downloaded_unverified_classes') if not args.dry_run else 'planned', 'path': str(path)}
        except Exception as exc:
            print(f'[ERROR] {spec.key}: {exc}')
            status[spec.key] = {'status':'error', 'detail':str(exc)}
    if not args.dry_run:
        MODELS.mkdir(exist_ok=True)
        (MODELS / 'install_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(json.dumps(status, indent=2))
    return 1 if any(x['status'] == 'error' for x in status.values()) else 0

if __name__ == '__main__':
    raise SystemExit(main())
