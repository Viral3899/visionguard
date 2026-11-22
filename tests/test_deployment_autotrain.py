"""New deployment and dataset-integrity regression tests, without network access."""
import os
os.environ['ADMIN_PASSWORD'] = ''
os.environ['SESSION_SECRET'] = 'test-secret-for-testing-only'
os.environ['DASHBOARD_PASSWORD'] = ''

import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from import_cameras import parse_config, import_cameras
from app.database import Database
from prepare_fire_dataset import materialize as fire_materialize
from download_fight_dataset import import_clips
from train_fight_classifier import catalog
from setup_train_all import run_steps
from activate_models import training_gate, activate
from app.config import ROOT


def test_camera_import_idempotent_and_env_expansion(tmp_path):
    cfg=tmp_path/'cameras.json'
    cfg.write_text(json.dumps({'cameras':[
        {'name':'Gate One','source':'${CAM_ONE}', 'features':{'fire_enabled':True,'tts_enabled':False}},
        {'name':'Rear Hall','source':'0','enabled':False}
    ]}),encoding='utf-8')
    rows=parse_config(cfg,env={'CAM_ONE':'rtsp://user:pass@192.168.9.9/stream'})
    assert len(rows)==2
    db_path=tmp_path/'test.db'
    assert import_cameras(cfg,db_path,env={'CAM_ONE':'rtsp://user:pass@192.168.9.9/stream'})['added']==2
    assert import_cameras(cfg,db_path,env={'CAM_ONE':'rtsp://user:pass@192.168.9.9/stream'})['unchanged']==2
    db=Database(db_path)
    assert len(db.list_cameras())==2
    assert 'pass' not in json.dumps(db.list_cameras())
    assert db.get_camera_config(db.list_cameras()[0]['id'])['tts_enabled']==0


def test_camera_import_rejects_undefined_secrets(tmp_path):
    cfg=tmp_path/'cams.json'
    cfg.write_text(json.dumps([{'name':'Example','source':'${UNSET}'}]))
    with pytest.raises(ValueError,match='UNSET'):
        parse_config(cfg,env={})


def test_camera_import_rejects_duplicate_source(tmp_path):
    cfg=tmp_path/'cams.json'
    cfg.write_text(json.dumps([{'name':'One','source':'0'},{'name':'Two','source':'0'}]))
    with pytest.raises(ValueError,match='Duplicate'):
        parse_config(cfg)


def test_fire_dataset_correct_labels_and_independent_splits(tmp_path):
    root=tmp_path/'source'
    for split, n in [('train',20),('test',6)]:
        for i in range(n):
            image=root/split/'images'/f'{i}.png'
            image.parent.mkdir(parents=True,exist_ok=True)
            val=(i+1)*2 if split=='train' else (i+1)*2+125
            cv2.imwrite(str(image),np.full((12,12,3),val%256,dtype=np.uint8))
            label=root/split/'labels'/f'{i}.txt'
            label.parent.mkdir(parents=True,exist_ok=True)
            label.write_text(f'{i%2} .5 .5 .3 .3\n')
    dest=tmp_path/'converted'
    config=fire_materialize(root,dest,seed=5)
    import yaml
    data=yaml.safe_load(config.read_text())
    assert data['names']=={0:'smoke',1:'fire'}
    assert data['test']=='test/images'
    report=json.loads((dest/'dataset_report.json').read_text())
    assert report['train_images'] + report['val_images']==20
    assert report['test_images']==6


def test_fire_dataset_rejects_bad_annotations_without_pseudonegative(tmp_path):
    root=tmp_path/'src'
    for split in ('train','test'):
        imgs=root/split/'images'; labels=root/split/'labels'
        imgs.mkdir(parents=True);labels.mkdir(parents=True)
        cv2.imwrite(str(imgs/'a.png'),np.zeros((10,10,3),dtype=np.uint8))
        (labels/'a.txt').write_text('99 .5 .5 .3 .3')
    with pytest.raises(ValueError):fire_materialize(root,tmp_path/'out')


def test_manifest_preserves_labels_and_splits(tmp_path):
    root=tmp_path/'original';output=tmp_path/'clips'
    metadata=root/'metadata.csv'
    root.mkdir()
    with metadata.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['file_name','label','split','group'])
        writer.writeheader()
        for key,label in [('fighting','Fighting'),('other','Other')]:
            for split in ('train','val','test'):
                for i in range(5):
                    folder=root/'videos'/key
                    folder.mkdir(parents=True,exist_ok=True)
                    file=folder/f'{split}_{i}.mp4'
                    file.write_bytes(f'{label}-{split}-{i}'.encode())
                    writer.writerow({'file_name':file.relative_to(root).as_posix(),
                                     'label':label,'split':split,'group':split})
    info=import_clips(root,output)
    assert info['clips']==30
    train,val=catalog(output)
    assert len(train)==10 and len(val)==10
    assert set(x[1] for x in val)=={0,1}


def test_runner_fails_loudly_when_a_stage_fails(tmp_path):
    calls=[]
    def execute(cmd):
        calls.append(Path(cmd[1]).name)
        return 1 if 'weapon' in Path(cmd[1]).name else 0
    steps=[('weapon', ['weapon.py']),('fire',['fire.py'])]
    status=tmp_path/'status.json'
    assert run_steps(steps,execute=execute,report_path=status)==1
    log=json.loads(status.read_text())
    assert log['steps']['weapon']['status']=='failed'
    assert log['steps']['fire']['status']=='success'
    assert log['complete'] is False


def test_activation_refuses_without_real_site_review():
    with pytest.raises(PermissionError):activate(['fire'],approved=False)


def test_multiple_cameras_have_independent_worker_limit(tmp_path,monkeypatch):
    from app.engine import CameraManager
    from app.schemas import CameraCreate
    from app import engine
    db=Database(tmp_path/'cams.db');db.init()
    for i in range(3):db.create_camera(CameraCreate(name=f'Camera {i}',source=str(i)))
    starts=[]
    class FakeWorker:
        def __init__(self,camera,*args):self.camera=camera
        def start(self):starts.append(self.camera['id'])
        def stop(self):pass
        def telemetry(self):return {'status':'online'}
    monkeypatch.setattr(engine,'CameraWorker',FakeWorker)
    from dataclasses import replace
    monkeypatch.setattr(engine,'settings',replace(engine.settings,max_active_cameras=2))
    m=CameraManager(db,None,None)
    m.start()
    assert len(starts)==2
    assert sum(m.telemetry(c['id'])['status']=='capacity_limit' for c in db.list_cameras())==1
