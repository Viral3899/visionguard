"""Evaluate confirmed video-level events on USER-LABELED independent CCTV clips.

You supply ground truth: a JSON manifest with positive and negative videos.
Example:
{"cases": [
  {"video":"C:/clips/fight_01.mp4", "expected":["violence"]},
  {"video":"C:/clips/kitchen_normal.mp4", "expected":[]}
]}
Run: .venv\\Scripts\\python.exe evaluate_video_cases.py --manifest cases.json

This benchmark intentionally DOES NOT infer labels from file names, source model
confidence, or downloaded training images. Use clips from unseen cameras/scenes.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

ALLOWED = {'fire_detected','smoke','violence','weapon','fall_detected','unusual_activity'}


def score_cases(records):
    summary = {k: Counter() for k in ALLOWED}
    details = []
    for row in records:
        expected = set(row['expected'])
        predicted = set(row['predicted'])
        if expected-ALLOWED or predicted-ALLOWED:
            raise ValueError('Unknown event class in video truth/predictions')
        for event in ALLOWED:
            if event in expected and event in predicted:summary[event]['tp'] += 1
            elif event in predicted:summary[event]['fp'] += 1
            elif event in expected:summary[event]['fn'] += 1
            else:summary[event]['tn'] += 1
        details.append({'video':row['video'], 'expected':sorted(expected),'predicted':sorted(predicted),
                        'misses':sorted(expected-predicted),'false_alarms':sorted(predicted-expected)})
    metrics = {}
    for name, counts in sorted(summary.items()):
        if counts['tp']+counts['fp']+counts['fn']==0:
            continue
        precision = (counts['tp']/(counts['tp']+counts['fp'])
                     if counts['tp']+counts['fp'] else None)
        recall = (counts['tp']/(counts['tp']+counts['fn'])
                  if counts['tp']+counts['fn'] else None)
        metrics[name] = dict(counts)
        metrics[name].update({'video_level_precision':precision, 'video_level_recall':recall})
    return {'videos':len(details), 'metrics':metrics, 'details':details,
            'limitations':'Video-level labels cannot assess localization / per-frame precision. '
                          'Split entire source recordings/cameras across training and evaluation.'}


def evaluate(manifest, out_dir=None):
    from analyze_video import analyze
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text(encoding='utf-8'))
    cases = data.get('cases')
    if not isinstance(cases,list) or not cases:
        raise ValueError('Manifest must have a nonempty cases array')
    out_dir = Path(out_dir or manifest.parent/'visionguard_evaluation').resolve()
    out_dir.mkdir(parents=True,exist_ok=True)
    rows = []
    for index, case in enumerate(cases):
        video = Path(case['video'])
        if not video.is_absolute():video = manifest.parent/video
        expected = case['expected']
        if not isinstance(expected,list) or any(not isinstance(x,str) for x in expected):
            raise ValueError('expected must be a list of event type names')
        result = analyze(video, out_dir/f'case_{index+1:04}.mp4', only='fire,fight,weapon,anomaly,object')
        rows.append({'video':str(video),'expected':expected,
                     'predicted':[e for e,n in result['confirmed_counts'].items() if n>0]})
    report = score_cases(rows)
    (out_dir/'benchmark_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path)
    a = p.parse_args()
    evaluate(a.manifest,a.output_dir)
