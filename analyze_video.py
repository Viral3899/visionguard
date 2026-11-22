"""Offline full-video debugging with REAL model outputs, original frames and JSONL.

Usage:
  .venv\Scripts\python.exe analyze_video.py --input "C:\\clips\\test.mp4"
  .venv\Scripts\python.exe analyze_video.py --input test.mp4 --only fire,weapon

Every source frame is written to an MP4. Inference defaults to every frame. A
candidate is NOT a confirmed security incident. Lower --diagnostic-conf only
for debugging; incident evidence still respects the independent >=50% floor.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

import cv2


def analyze(path, output=None, only='fire,fight,weapon,anomaly,object',
            every=1, diagnostic_conf=None, limit=0, device='auto'):
    from app.config import settings
    from app.detectors import DetectionManager
    from app.incidents import IncidentTracker, eligible_event, draw_event_box
    if every < 1 or limit < 0:
        raise ValueError('--every must be >=1 and --limit must be >=0')
    if diagnostic_conf is not None and not (0.05 <= diagnostic_conf <= 1):
        raise ValueError('Diagnostic confidence must be between .05 and 1')
    input_path = Path(path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output is None:
        output = input_path.parent / (input_path.stem + '_visionguard.mp4')
    output = Path(output).resolve()
    if output == input_path:
        raise ValueError('Output may not overwrite input')
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f'OpenCV failed to decode: {input_path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 2 or height < 2:
        cap.release()
        raise RuntimeError('Invalid video dimensions')
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Failed to open MP4 writer (try another output folder)')
    classes = {s.strip().lower() for s in only.split(',') if s.strip()}
    allowed = {'fire','fight','weapon','anomaly','object'}
    if not classes or classes - allowed:
        raise ValueError('Only --only fire,fight,weapon,anomaly,object supported')
    manager = DetectionManager()
    # Unlike the live camera worker, offline analysis has no upstream person
    # tracker. Build person context so the strict weapon gate can operate.
    person_model = None
    person_context_error = None
    if 'weapon' in classes:
        try:
            from ultralytics import YOLO
            from app.config import local_model_path
            person_model = YOLO(local_model_path(settings.model_path))
        except Exception as exc:
            person_context_error = str(exc)
            print('WARNING: person context detector unavailable:', exc, flush=True)
    manager.fire_every_n_frames = 1
    manager.violence_every_n_frames = 1
    manager.weapon_every_n_frames = 1
    manager.anomaly_every_n_frames = 1
    manager.object_every_n_frames = 1
    loaded = {
        'fire': manager.fire_detector.status == 'ready',
        'fight': manager.violence_detector.model is not None or manager.temporal_fight_detector is not None
                 or manager.torch_fight_classifier is not None,
        'weapon': any(x is not None and x.model is not None for x in (
            manager.weapon_detector, manager.trained_weapon_detector)),
        'anomaly': getattr(manager.anomaly_detector, 'model', None) is not None,
        'object': manager.object_detector.model is not None,
    }
    if not any(loaded.get(key) for key in classes):
        writer.release(); cap.release()
        output.unlink(missing_ok=True)
        raise RuntimeError('None of the selected detector models loaded. Run verify_models.py; '
                           'do not interpret this as no threats in your video.')
    if diagnostic_conf is not None:
        for model in (manager.fire_detector, manager.violence_detector,
                      manager.weapon_detector, manager.trained_weapon_detector,
                      manager.object_detector):
            if model is not None:
                model.confidence = diagnostic_conf
    selected_config = {'fire_enabled': int('fire' in classes),
                       'violence_enabled': int(bool(classes & {'fight','weapon'})),
                       'anomaly_enabled': int('anomaly' in classes),
                       'object_enabled': int('object' in classes)}
    model_info = {}
    for key, model in [('fire',manager.fire_detector),('fight',manager.violence_detector),
                       ('weapon',manager.weapon_detector),('weapon_trained',manager.trained_weapon_detector),
                       ('object',manager.object_detector)]:
        if model is not None and model.model is not None:
            model_info[key] = {'file':model.model_path,
                               'labels':list(model.model.names.values()) if isinstance(model.model.names,dict)
                               else list(model.model.names)}
    incidents = IncidentTracker(settings.incident_min_confidence,
                                settings.incident_confirmation_frames,
                                max_missed_frames=max(2,every))
    evidence = output.with_suffix('.events.jsonl')
    summary_path = output.with_suffix('.summary.json')
    frame_number = 0
    sampled = 0
    raw_counts, accepted_counts, suppressed_counts, confirmed_counts = Counter(), Counter(), Counter(), Counter()
    start = time.monotonic()
    try:
        with evidence.open('w', encoding='utf-8') as log:
            while not limit or frame_number < limit:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_number += 1
                if frame_number % every == 0 or (frame_number == 1 and every > 1):
                    sampled += 1
                    # Obtain person context from an independent person checkpoint.
                    tracks = []
                    if person_model is not None:
                        try:
                            kw = {'classes':[0], 'conf':max(.40, settings.confidence),
                                  'imgsz':settings.imgsz, 'verbose':False}
                            if settings.device != 'auto': kw['device'] = settings.device
                            people_result = person_model.predict(frame, **kw)[0]
                            if people_result.boxes is not None:
                                for j, b in enumerate(people_result.boxes.xyxy.cpu().tolist()):
                                    tracks.append({'id': j, 'bbox': b})
                        except Exception as exc:
                            person_context_error = str(exc)
                            person_model = None
                    # Manager processes all selected models on clean imagery.
                    raw = manager.process_frame(frame, tracks, 'offline-video', selected_config,
                                                timestamp=(frame_number-1)/fps)
                    confirmed = incidents.update(raw, (frame_number-1)/fps)
                    for e in raw:
                        accepted_counts[e.get('type','unknown')] += 1
                    all_candidates = getattr(manager, 'last_raw_candidates', raw)
                    for e in all_candidates:
                        raw_counts[e.get('type','unknown')] += 1
                    for e in getattr(manager, 'last_suppressed', []):
                        suppressed_counts[e.get('suppressed_reason', 'unknown')] += 1
                    for e in confirmed:
                        confirmed_counts[e.get('type','unknown')] += 1
                    # Confirmed threats ONLY are displayed; no misleading raw boxes.
                    for e in incidents.active_events:
                        draw_event_box(frame, e)
                    suppressed = getattr(manager, 'last_suppressed', [])
                    log.write(json.dumps({'frame':frame_number,
                                          'seconds':round((frame_number-1)/fps,3),
                                          'accepted_candidates':raw,
                                          'raw_candidates':getattr(manager, 'last_raw_candidates', raw),
                                          'suppressed':suppressed,
                                          'confirmed':confirmed}, default=float) + '\n')
                else:
                    # Update even without inference, so stale events expire.
                    incidents.update([], (frame_number-1)/fps)
                status = f'FRAME {frame_number} | ANALYZED {sampled} | CONFIRMED {sum(confirmed_counts.values())}'
                cv2.rectangle(frame,(0,0),(min(width,700),36),(18,25,35),-1)
                cv2.putText(frame,status,(10,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                writer.write(frame)
                if frame_number % 100 == 0:
                    print(f'Frames {frame_number}  raw {dict(raw_counts)}  confirmed {dict(confirmed_counts)}',flush=True)
    finally:
        writer.release()
        cap.release()
    if frame_number == 0:
        raise RuntimeError('Video contains no decodable frames')
    summary = {'source':str(input_path), 'video':str(output), 'events_jsonl':str(evidence),
               'frames_read':frame_number, 'frames_analyzed':sampled,
               'fps':fps, 'video_seconds':round(frame_number/fps,2),
               'elapsed_seconds':round(time.monotonic()-start,2),
               'models_loaded':loaded, 'model_metadata':model_info,
               'person_context_available':person_context_error is None if 'weapon' in classes else None,
               'person_context_error':person_context_error,
               'raw_counts':dict(raw_counts), 'accepted_counts':dict(accepted_counts),
               'suppressed_by_reason':dict(suppressed_counts),
               'confirmed_counts':dict(confirmed_counts),
               'incident_minimum':settings.incident_min_confidence,
               'diagnostic_confidence':diagnostic_conf,
               'warning':'Predictions are model hypotheses, not ground truth. '
                         'A zero event count is not proof of safety; view raw per-frame candidates and model statuses.'}
    summary_path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return summary

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--only', default='fire,fight,weapon,anomaly,object')
    p.add_argument('--every', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--device',choices=['auto','cpu','0'], default='auto')
    p.add_argument('--diagnostic-conf', type=float,
                   help='Lower YOLO candidate threshold for debugging; does not alter incident floor')
    args = p.parse_args()
    if args.device != 'auto':
        os.environ['DEVICE'] = args.device
    analyze(args.input,args.output,args.only,args.every,args.diagnostic_conf,args.limit,args.device)
