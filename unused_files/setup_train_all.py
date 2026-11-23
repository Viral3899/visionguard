"""One command: verify GPU, fetch pretrained models and annotated data, train 3 tasks.

All actual downloads/training occur locally on the user's machine, not in ZIP.
Person detection and pretrained fight YOLO remain pretrained: no correct labeled
person/fight-box dataset is inferred from fire or violence classification clips.
The unknown-output Keras .h5 is not used as a surrogate for a labeled classifier.
"""
from datetime import datetime,timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from app.config import ROOT

REPORT = ROOT / 'data' / 'training_pipeline_status.json'


def plan(args):
    steps=[('runtime',['verify_runtime.py']+([] if args.allow_cpu else ['--require-gpu']))]
    if not args.offline:
        steps.append(('pretrained_models',['install_security_models.py','--accept-checkpoint-risk','--include-3dcnn']))
    if args.download_only:
        if not args.skip_weapon:
            steps.append(('weapon_dataset',['prepare_datasets.py']))
        if not args.skip_fire:
            steps.append(('fire_dataset',['prepare_fire_dataset.py','--max-train-images',str(args.fire_max_images)]))
        if not args.skip_fight:
            if not args.fight_license_reviewed and not args.offline:
                raise PermissionError('Fight dataset licenses must be reviewed before download.')
            if not args.offline:
                steps.append(('fight_labeled_videos',['download_fight_dataset.py','--accept-license-review',
                                                      '--max-per-class',str(args.fight_max_per_class)]))
    else:
        if not args.skip_weapon:
            steps.append(('weapon_finetune', ['train_models.py','--source','both','--epochs',str(args.weapon_epochs),
                         '--base','yolov8n']+(['--skip-download'] if args.offline else [])+
                         (['--allow-cpu'] if args.allow_cpu else [])))
        if not args.skip_fire:
            steps.append(('fire_finetune',['train_fire.py','--epochs',str(args.fire_epochs),
                         '--max-train-images',str(args.fire_max_images)]+
                         (['--skip-download'] if args.offline else [])+
                         (['--allow-cpu'] if args.allow_cpu else [])))
        if not args.skip_fight:
            if not args.fight_license_reviewed and not args.offline:
                raise PermissionError('Review licenses for OmarNabill/event-detection original video sources, '
                                      'then pass --fight-license-reviewed. No fight video download occurs otherwise.')
            if not args.offline:
                steps.append(('fight_labeled_videos',['download_fight_dataset.py',
                                                      '--accept-license-review','--max-per-class',str(args.fight_max_per_class)]))
            steps.append(('fight_video_classifier',['train_fight_classifier.py','--epochs',str(args.fight_epochs)] +
                         (['--allow-cpu'] if args.allow_cpu else [])))
    return steps


def run_steps(steps,execute=None, report_path=REPORT, continue_on_error=True):
    report_path=Path(report_path)
    execute=execute or (lambda args:subprocess.run(args,cwd=ROOT,check=False).returncode)
    status={'started_at':datetime.now(timezone.utc).isoformat(),
            'steps':{},'complete':False,
            'warnings':['Pretrained weights are not automatically retrained without task-compatible annotations.',
                        'Successful training/holdout scores are NOT proof of real-world CCTV accuracy.',
                        'License reviews and third-party serialized checkpoint trust are operator responsibilities.']}
    for name,cmd in steps:
        args=[sys.executable, str(ROOT/cmd[0]), *cmd[1:]]
        print(f'\n========== {name.upper()} ==========\n',flush=True)
        try:
            exit_code=execute(args)
            state='success' if exit_code==0 else 'failed'
            status['steps'][name]={'status':state,'exit_code':exit_code,'command':[Path(cmd[0]).name,*cmd[1:]]}
        except Exception as exc:
            status['steps'][name]={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
        report_path.parent.mkdir(parents=True,exist_ok=True)
        report_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
        if status['steps'][name]['status']=='failed' and (not continue_on_error or name=='runtime'):
            break
    status['complete']=all(value['status']=='success' for value in status['steps'].values())
    status['ended_at']=datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(f'\nPipeline summary: {report_path}\n',json.dumps({x:y['status'] for x,y in status['steps'].items()},indent=2),flush=True)
    return 0 if status['complete'] else 1


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--accept-checkpoint-risk',action='store_true')
    ap.add_argument('--accept-data-licenses',action='store_true',
                    help='Confirm D-Fire/weapon dataset licenses have been reviewed before downloads')
    ap.add_argument('--fight-license-reviewed',action='store_true',
                    help='Explicitly acknowledge original clip license review')
    ap.add_argument('--allow-cpu',action='store_true')
    ap.add_argument('--offline',action='store_true',help='Use predownloaded models and prepared datasets')
    ap.add_argument('--download-only',action='store_true')
    ap.add_argument('--skip-fire',action='store_true')
    ap.add_argument('--skip-weapon',action='store_true')
    ap.add_argument('--skip-fight',action='store_true')
    ap.add_argument('--fire-epochs',type=int,default=40)
    ap.add_argument('--weapon-epochs',type=int,default=40)
    ap.add_argument('--fight-epochs',type=int,default=10)
    ap.add_argument('--fire-max-images',type=int,default=0,
                    help='0 trains on ALL D-Fire annotated training images (~21k total)')
    ap.add_argument('--fight-max-per-class',type=int,default=0,
                    help='0 uses ALL available labeled clips')
    ap.add_argument('--stop-on-error',action='store_true')
    args=ap.parse_args(argv)
    if not args.offline and not (args.accept_checkpoint_risk and args.accept_data_licenses):
        ap.error('Online downloads require BOTH --accept-checkpoint-risk and --accept-data-licenses. '
                 'Review Hugging Face repo licenses and checkpoint serialization risks before starting.')
    if args.offline and not args.download_only:
        expected=[]
        if not args.skip_weapon and not (ROOT/'data'/'datasets'/'visionguard_weapon'/'data.yaml').is_file():
            expected.append('prepared weapon data.yaml')
        if not args.skip_fire and not (ROOT/'data'/'datasets'/'visionguard_fire'/'data.yaml').is_file():
            expected.append('prepared fire data.yaml')
        if not args.skip_fight and not (ROOT/'data'/'fight_clips'/'split_manifest.json').is_file():
            expected.append('prepared fight split_manifest.json')
        if expected:
            ap.error('Offline mode missing: '+', '.join(expected))
    try:steps=plan(args)
    except PermissionError as exc:ap.error(str(exc))
    print('Training tasks: '+', '.join(name for name,_ in steps),flush=True)
    return run_steps(steps,continue_on_error=not args.stop_on_error)


if __name__=='__main__':
    raise SystemExit(main())
