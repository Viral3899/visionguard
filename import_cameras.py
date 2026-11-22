"""Idempotent import of RTSP/HTTP/USB cameras into existing VisionGuard DB.

Copy cameras.example.json to cameras.json, substitute ${NAME} with .env values.
No camera credentials are printed. Never deletes existing cameras.
"""
import argparse
import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from app.config import ROOT, settings
from app.database import Database
from app.schemas import CameraCreate

ENV = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')


def resolve_source(value, env=None):
    env = os.environ if env is None else env
    if not isinstance(value, str):
        raise ValueError('camera source must be a string')
    def replace(match):
        key = match.group(1)
        if not env.get(key):
            raise ValueError(f'Camera source environment variable {key} is absent or empty')
        return env[key]
    return ENV.sub(replace, value)


def parse_config(path, env=None):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = data['cameras'] if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError('cameras.json must be a list or an object containing cameras: []')
    parsed, names, sources = [], set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Each camera entry must be an object')
        entry = {**row, 'source': resolve_source(row.get('source'), env)}
        enabled = CameraCreate.model_validate(entry)
        if enabled.name.casefold() in names or enabled.source in sources:
            raise ValueError('Duplicate camera name or stream source')
        names.add(enabled.name.casefold())
        sources.add(enabled.source)
        config = row.get('features') or {}
        if not isinstance(config, dict):
            raise ValueError('features must be an object')
        allowed = {'fire_enabled','violence_enabled','anomaly_enabled','object_enabled','telegram_enabled','tts_enabled','sensitivity'}
        if set(config) - allowed:
            raise ValueError(f'Unsupported per-camera feature fields: {set(config) - allowed}')
        parsed.append((enabled, config))
    return parsed


def import_cameras(path, db_path=None, update_existing=False, dry_run=False, env=None):
    load_dotenv(ROOT / '.env')
    parsed = parse_config(path, env)
    db = Database(db_path or settings.data_dir / 'visionguard.db')
    db.init()
    existing = db.list_cameras(safe=False)
    by_source = {c['source']: c for c in existing}
    by_name = {c['name'].casefold(): c for c in existing}
    stats = {'configured':len(parsed), 'added':0,'unchanged':0,'updated':0}
    for cam, feature in parsed:
        found = by_source.get(cam.source) or by_name.get(cam.name.casefold())
        if found:
            if update_existing:
                changes = cam.model_dump(exclude={'client_id'})
                if not dry_run:
                    db.update_camera(found['id'], changes)
                    if feature: db.update_camera_config(found['id'], feature)
                stats['updated'] += 1
            else:
                stats['unchanged'] += 1
        else:
            if not dry_run:
                created = db.create_camera(cam)
                if feature: db.update_camera_config(created['id'], feature)
            stats['added'] += 1
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT/'cameras.json')
    ap.add_argument('--update-existing', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args=ap.parse_args()
    if not args.config.is_file():
        print(f'No {args.config.name}; add cameras in dashboard or copy cameras.example.json')
    else:
        import_cameras(args.config, update_existing=args.update_existing, dry_run=args.dry_run)
