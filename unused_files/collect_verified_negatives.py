"""Extract *clean* CCTV frames from a segment you explicitly verified as weapon-free.

Never use already-annotated predictions/screenshots as clean negatives. A gun or
knife in an 'empty' negative image creates contradictory training supervision.

Example:
 .venv\\Scripts\\python.exe collect_verified_negatives.py --input "normal.mp4" \\
   --start 0 --end 30 --interval 2 --confirm-no-weapons
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import cv2


def collect(video, out, start, end, interval, *, confirmed=False, limit=200):
    if not confirmed:
        raise ValueError('Explicit --confirm-no-weapons required after reviewing the original CLEAN video')
    if not (0 <= start < end and interval > 0 and limit > 0):
        raise ValueError('Expected 0 <= start < end, interval > 0 and limit > 0')
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('Cannot decode input video')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Invalid FPS: unable to extract exact timestamps')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_id = hashlib.sha256(str(video).encode('utf-8')).hexdigest()[:12]
    folder = Path(out) / f'{video.stem[:45]}_{source_id}'
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    records = []
    try:
        t = float(start)
        while t < end and count < limit:
            target = int(round(t * fps))
            if total > 0 and target >= total:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            name = f'frame_{target:09d}.jpg'
            out_file = folder / name
            if not cv2.imwrite(str(out_file), frame):
                raise OSError(f'Cannot write image {out_file}')
            records.append({'image':name, 'video_second':round(target/fps, 3), 'frame':target})
            count += 1
            t += interval
    finally:
        cap.release()
    manifest = folder / '_review_manifest.json'
    manifest.write_text(json.dumps({'source':str(video),'confirmed_no_weapons':True,
                                    'assumption':'Operator reviewed this clean clip segment. '
                                                 'Re-review extracted images before training.',
                                    'items': records},indent=2),encoding='utf-8')
    print(f'Extracted {count} candidate negative images into {folder}. '
          'REVIEW them before using --verified-negatives.')
    return folder, count


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('data/verified_negatives'))
    p.add_argument('--start', type=float, required=True)
    p.add_argument('--end', type=float, required=True)
    p.add_argument('--interval', type=float, default=2)
    p.add_argument('--limit', type=int, default=200)
    p.add_argument('--confirm-no-weapons', action='store_true')
    a = p.parse_args()
    collect(a.input, a.output, a.start, a.end, a.interval,
            confirmed=a.confirm_no_weapons, limit=a.limit)
