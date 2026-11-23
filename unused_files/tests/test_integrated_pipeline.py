"""Regression tests for label-safe training, real clip processing and incident safety."""
import os
os.environ['ADMIN_PASSWORD'] = ''
os.environ['SESSION_SECRET'] = 'test-secret-for-testing-only'
os.environ['DASHBOARD_PASSWORD'] = ''

from pathlib import Path
from types import SimpleNamespace
import json
import sys

import cv2
import numpy as np
import pytest
import torch

from app.incidents import IncidentTracker, same_scene_box
from app.torch_fight import preprocess_frames, FRAMES, SIZE
from app.detectors import BaseDetector
from prepare_datasets import materialize
from train_models import audit_dataset
from train_fight_classifier import catalog


def image(path, gray):
    assert cv2.imwrite(str(path), np.full((16, 16, 3), gray, dtype=np.uint8))
    return path


def test_mislabeled_duplicated_image_is_rejected(tmp_path):
    pic = image(tmp_path/'one.png', 10)
    with pytest.raises(ValueError, match='Conflicting annotations'):
        materialize([(pic,['1 .5 .5 .1 .1'],'A'),
                     (pic,['2 .5 .5 .1 .1'],'B')], tmp_path/'prepared')


def test_no_untrained_class_is_created(tmp_path):
    records = [(image(tmp_path/f'{i}.png',i*20),['1 .5 .5 .2 .2'],'sample')
               for i in range(7)]
    yaml_file = materialize(records, tmp_path/'prepared')
    audit = audit_dataset(yaml_file)
    assert audit['classes'] == ['knife']
    assert all(x != 'gun' for x in audit['classes'])


def test_negative_images_are_explicit_and_never_promoted_as_a_class(tmp_path):
    negatives = tmp_path/'negatives'
    negatives.mkdir()
    for i in range(3): image(negatives/f'n{i}.png',201+i)
    records = [(image(tmp_path/f'{i}.png',i*20),['1 .5 .5 .2 .2'],'CCTV') for i in range(7)]
    yaml_file = materialize(records, tmp_path/'prepared', negatives_dir=negatives)
    report = json.loads((yaml_file.parent/'dataset_report.json').read_text())
    assert report['images'] == 10
    assert report['counts']['train_negative_images'] + report['counts']['val_negative_images'] == 3


def test_spatial_confirmation_does_not_merge_different_knives():
    tracker = IncidentTracker(minimum=.5, confirmations=2)
    def event(box):
        return {'type':'weapon','subtype':'knife','confidence':.88,'bbox':box}
    a=[.01,.01,.10,.10]
    b=[.75,.75,.85,.85]
    assert not same_scene_box(a,b)
    assert tracker.update([event(a)],0.) == []
    assert tracker.update([event(b)],.1) == []
    assert len(tracker.update([event(b)],.2)) == 1


def test_pretrained_classification_checkpoint_not_assumed_yolo_boxes(monkeypatch):
    class BadModel:
        task='classify'
        names={0:'normal',1:'fight'}
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=lambda path:BadModel()))
    assert BaseDetector('synthetic-cls.pt',.6).model is None


def test_gpu_fight_preprocessing_uses_16_rgb_frames():
    blue = np.zeros((48,64,3), dtype=np.uint8)
    blue[:,:,0] = 255
    tensor = preprocess_frames([blue] * FRAMES)
    assert tensor.shape == (3, FRAMES, SIZE, SIZE)
    assert isinstance(tensor, torch.Tensor)
    # BGR blue becomes RGB blue, so channel 2 must exceed channel 0 after norm.
    assert float(tensor[2].mean()) > float(tensor[0].mean())


def test_clip_catalog_requires_both_real_labeled_video_sets(tmp_path):
    (tmp_path/'normal').mkdir()
    (tmp_path/'fight').mkdir()
    for i in range(5):
        (tmp_path/'normal'/f'{i}.mp4').write_bytes(bytes([i+1]))
        (tmp_path/'fight'/f'{i}.mp4').write_bytes(bytes([i+10]))
    train, val = catalog(tmp_path)
    assert len(train) == 8 and len(val) == 2
    (tmp_path/'fight'/'4.mp4').write_bytes(b'\x01')
    with pytest.raises(ValueError,match='Duplicate video'):
        catalog(tmp_path)


def test_classifier_two_independent_windows_not_fabricated_score():
    tracker = IncidentTracker(confirmations=2)
    event = {'type':'violence', 'subtype':'temporal_fight_classifier',
             'source_model':'torch_r3d18_fight', 'confidence':.86,
             'bbox':[0.,0.,1.,1.], 'temporal_confirmed':True}
    assert len(tracker.update([event], 3.0)) == 1
    assert tracker.update([event], 3.1) == []


def test_video_diagnostic_writes_all_frames_and_real_json(monkeypatch, tmp_path):
    import app.detectors
    from analyze_video import analyze
    clip = tmp_path/'source.avi'
    writer = cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,64))
    assert writer.isOpened()
    for _ in range(5):writer.write(np.zeros((64,64,3),dtype=np.uint8))
    writer.release()
    missing = SimpleNamespace(model=None, model_path='missing.pt')
    class FakeManager:
        def __init__(self):
            self.fire_detector=SimpleNamespace(model=None,status='model_unavailable')
            self.violence_detector=missing
            self.weapon_detector=SimpleNamespace(model=SimpleNamespace(names={0:'knife'}),model_path='knife.pt')
            self.trained_weapon_detector=None
            self.temporal_fight_detector=None
            self.torch_fight_classifier=None
            self.anomaly_detector=missing
            self.object_detector=missing
        def process_frame(self, frame, tracks, camera_id, config, *, timestamp):
            return [{'type':'weapon','subtype':'knife','confidence':.79,
                     'bbox':[.1,.1,.3,.3], 'source_model':'knife.pt'}]
    monkeypatch.setattr(app.detectors,'DetectionManager',FakeManager)
    output=tmp_path/'result.mp4'
    report=analyze(clip, output=output, only='weapon')
    assert output.is_file() and output.stat().st_size > 0
    assert report['frames_read']==5 and report['frames_analyzed']==5
    assert report['confirmed_counts']=={'weapon':1}
    lines=output.with_suffix('.events.jsonl').read_text().splitlines()
    assert len(lines)==5 and json.loads(lines[0])['raw_candidates'][0]['confidence']==.79
    playback=cv2.VideoCapture(str(output))
    count=0
    while playback.read()[0]: count+=1
    playback.release()
    assert count==5


def test_shravya_parts_pair_annotations_only_with_same_part(tmp_path):
    from prepare_datasets import collect_shravya
    root = tmp_path/'shravya'
    for part, gray in [('part_1', 20), ('part_2', 210)]:
        files = root/'Knife_detection'/part
        (files/'images').mkdir(parents=True)
        (files/'annotations').mkdir()
        image(files/'images'/'same.png', gray)
        (files/'annotations'/'same.xml').write_text(
            '<annotation><size><width>16</width><height>16</height></size>'
            '<object><name>knife</name><bndbox><xmin>1</xmin><ymin>1</ymin>'
            '<xmax>8</xmax><ymax>8</ymax></bndbox></object></annotation>')
    pairs = list(collect_shravya(root))
    assert len(pairs) == 2
    assert {pair[0].parent.parent.name for pair in pairs} == {'part_1','part_2'}


def test_manual_video_benchmark_precision_and_false_alarms():
    from evaluate_video_cases import score_cases
    report = score_cases([
        {'video':'fight.mp4','expected':['violence'],'predicted':['violence']},
        {'video':'normal.mp4','expected':[],'predicted':['violence']},
        {'video':'gun.mp4','expected':['weapon'],'predicted':[]},
    ])
    assert report['videos'] == 3
    assert report['metrics']['violence']['tp'] == 1
    assert report['metrics']['violence']['fp'] == 1
    assert report['metrics']['violence']['video_level_precision'] == .5
    assert report['metrics']['weapon']['fn'] == 1
