"""Explicit opt-in to trained weights AFTER independent site video evaluation.

Internal mAP/recall alone is not deployment acceptance. This tool checks
training provenance and minimum holdout gates, then updates .env atomically.
"""
import argparse
import json
from pathlib import Path
import os
import re
from app.config import ROOT
from model_registry import MODELS


def training_gate(kind):
    if kind=='fire':
        info=json.loads((MODELS/'fire_training_report.json').read_text(encoding='utf-8'))
        if float(info['val_map50']) < .30 or float(info['test_map50']) < .25:
            raise ValueError('Fire model minimum validation/test gate unmet')
        if not (MODELS/'visionguard-fire-trained.pt').is_file():
            raise FileNotFoundError('Missing trained fire weights')
        return {'FIRE_MODEL_PATH':'models/visionguard-fire-trained.pt'}
    if kind=='weapon':
        info=json.loads((MODELS/'training_report.json').read_text(encoding='utf-8'))
        if float(info['metrics']['holdout_map50']) < .30:
            raise ValueError('Weapon model minimum validation gate unmet')
        if not (MODELS/'visionguard-weapon-trained.pt').is_file():
            raise FileNotFoundError('Missing trained weapon weights')
        return {'ENABLE_TRAINED_WEAPON_MODEL':'true'}
    if kind=='fight':
        info=json.loads((MODELS/'fight_classifier_report.json').read_text(encoding='utf-8'))
        if float(info.get('weakest_class_recall',0)) < .60:
            raise ValueError('Fight classifier per-class recall gate unmet')
        if not (MODELS/'visionguard-r3d18-fight.pt').is_file():
            raise FileNotFoundError('Missing trained fight classifier weights')
        return {'ENABLE_FIGHT_CLASSIFIER':'true'}
    raise ValueError(kind)


def update_env(values,path=None):
    env=Path(path or ROOT/'.env')
    if not env.is_file(): raise FileNotFoundError('Initialize .env before model activation')
    text=env.read_text(encoding='utf-8')
    for key,value in values.items():
        pattern=rf'(?m)^{re.escape(key)}=.*$'
        if re.search(pattern,text):
            text=re.sub(pattern,lambda _:f'{key}={value}',text)
        else:text+=f'\n{key}={value}\n'
    temp=env.with_suffix('.env.tmp')
    temp.write_text(text,encoding='utf-8')
    os.replace(temp,env)


def activate(tasks,approved=False,path=None):
    if not approved:raise PermissionError('Independent labeled site-video validation approval is required')
    changes={}
    for task in tasks:
        changes.update(training_gate(task))
    update_env(changes,path)
    return changes


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',default='fire,weapon,fight')
    p.add_argument('--approve-site-validation',action='store_true',
                   help='I have reviewed ground-truth CCTV clips for both misses and false alerts')
    args=p.parse_args()
    print(activate([name.strip() for name in args.models.split(',')],args.approve_site_validation))
