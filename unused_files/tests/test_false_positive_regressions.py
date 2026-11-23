"""Tests generated from the user's *failure patterns*, not from ground-truth model inference.

The screenshots are evidence of false alarms, not annotated training images.
"""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.detectors import DetectionManager, DetectionResult, FireDetector, SpecialistWeaponDetector
from app.incidents import IncidentTracker
from app.threat_policy import ThreatPolicy
from collect_verified_negatives import collect

FRAME = np.zeros((480, 640, 3), np.uint8)
PERSON = [{'id': 1, 'bbox': [30, 0, 370, 470]}]


def threat(label, score, coords):
    return {'type': 'weapon', 'subtype': label, 'confidence': score,
            'bbox': [coords[0]/640, coords[1]/480, coords[2]/640, coords[3]/480]}


@pytest.mark.parametrize('label,score,coords,reason', [
    ('gun', .80, (0,31,171,350), 'below_class_threshold'),
    ('knife', .54, (109,175,172,351), 'below_class_threshold'),
    ('knife', .38, (438,48,633,416), 'below_class_threshold'),
    ('weapon', .79, (4,56,344,462), 'below_class_threshold'),
    ('gun', .99, (0,31,171,350), 'implausible_bbox_size'),
    ('knife', .95, (439,48,633,416), 'implausible_bbox_size'),
    ('weapon', .99, (4,56,344,462), 'implausible_bbox_size'),
])
def test_user_room_false_alarm_patterns_are_blocked(label, score, coords, reason):
    # Scores/boxes taken from the visible annotations approximately, NOT YOLO output.
    passes, why = ThreatPolicy().review(threat(label, score, coords), FRAME.shape, PERSON)
    assert not passes and why == reason


def test_plausible_weapon_near_person_survives_policy_with_true_score():
    policy = ThreatPolicy()
    gun = threat('gun', .91, (150, 220, 188, 260))
    assert policy.review(gun, FRAME.shape, PERSON) == (True, 'passed')
    assert gun['confidence'] == .91
    assert policy.review(gun, FRAME.shape, []) == (False, 'no_person_near_weapon')
    distant = threat('knife', .95, (555, 210, 575, 265))
    assert policy.review(distant, FRAME.shape, PERSON) == (False, 'no_person_near_weapon')
    assert ThreatPolicy(require_person=False).review(distant, FRAME.shape, []) == (True, 'passed')


def test_unknown_and_malformed_weapon_are_never_critical_alerts():
    policy = ThreatPolicy()
    assert policy.review({'type':'weapon','subtype':'alien','confidence':.99,
                          'bbox':[.1,.1,.2,.2]}, FRAME.shape, PERSON)[0] is False
    assert policy.review({'type':'weapon','subtype':'gun','confidence':float('nan'),
                          'bbox':[.1,.1,.2,.2]}, FRAME.shape, PERSON)[0] is False
    assert policy.review({'type':'weapon','subtype':'gun','confidence':.99,
                          'bbox':[.9,.5,.2,.7]}, FRAME.shape, PERSON)[0] is False


def test_detection_manager_does_not_draw_unverified_weapon_boxes(monkeypatch):
    # Integration: a real DetectionManager with fake model outputs. No weights needed.
    import app.detectors as dm
    monkeypatch.setattr(dm.BaseDetector, '_load_model', lambda self: None)
    monkeypatch.setattr(dm.AnomalyDetector, '_load_model', lambda self: None)
    manager = DetectionManager()
    manager.violence_detector.detect = lambda frame: None
    detections = DetectionResult([[0, 31, 171, 350], [150,220,188,260]],
                                 [0,0], [.8,.92], {0:'gun'})
    manager.weapon_detector = SimpleNamespace(
        model_path='fake-test-model.pt', detect=lambda frame:detections,
        analyze_weapons=SpecialistWeaponDetector.analyze_weapons)
    manager.trained_weapon_detector = None
    input_image = FRAME.copy()
    events = manager.process_frame(input_image, PERSON, 'camera',
                                   {'violence_enabled':1, 'fire_enabled':0,
                                    'anomaly_enabled':0, 'object_enabled':0})
    assert np.array_equal(input_image, FRAME), 'Raw YOLO boxes must not be displayed'
    assert len(events) == 1 and events[0]['confidence'] == .92
    assert len(manager.last_raw_candidates) == 2
    assert len(manager.last_suppressed) == 1
    assert manager.last_suppressed[0]['suppressed_reason'] == 'below_class_threshold'


def test_incident_tracker_only_displays_confirmed_not_single_frame():
    tracker = IncidentTracker(minimum=.5, confirmations=2)
    e = {'type':'weapon','subtype':'gun','confidence':.91,'bbox':[.1,.2,.2,.3]}
    assert tracker.update([e],0) == []
    assert tracker.active_events == []
    assert len(tracker.update([e],.1)) == 1
    assert tracker.active_events == [e]
    assert tracker.update([e],.2) == []
    assert tracker.active_events == [e]
    tracker.update([],.3)
    assert tracker.active_events == []


def test_fire_matchbox_single_73_percent_is_not_an_alert(monkeypatch):
    from app import detectors
    monkeypatch.setattr(detectors.BaseDetector, '_load_model',
                        lambda self: setattr(self, 'model', SimpleNamespace(names={0:'fire'})))
    detector = FireDetector()
    r = DetectionResult([[250,245,305,302]], [0], [.73], {0:'fire'})
    for t in (0, .1, .2):
        events = detector.check_fire_hazard(FRAME, r, t)
        assert events and events[0]['type'] == 'fire_watch'


def test_small_flame_cannot_alert_from_only_one_frame(monkeypatch):
    from app import detectors
    monkeypatch.setattr(detectors.BaseDetector, '_load_model',
                        lambda self: setattr(self, 'model', SimpleNamespace(names={0:'flame'})))
    detector = FireDetector()
    frame = np.zeros((100,100,3),dtype=np.uint8)
    r = DetectionResult([[10,10,12,12]], [0], [.79], {0:'flame'})
    assert detector.check_fire_hazard(frame,r,0)[0]['type'] == 'fire_watch'
    assert detector.check_fire_hazard(frame,r,.1)[0]['type'] == 'fire_watch'
    assert detector.check_fire_hazard(frame,r,.2)[0]['type'] == 'fire_detected'
    assert detector.check_fire_hazard(frame,r,.3)[0]['type'] == 'fire_detected'


def test_negative_still_extraction_requires_verified_clean_clip(tmp_path):
    video = tmp_path/'normal.avi'
    vw = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,64))
    assert vw.isOpened()
    for i in range(20):
        vw.write(np.full((64,64,3), 20+i, dtype=np.uint8))
    vw.release()
    with pytest.raises(ValueError,match='Explicit'):
        collect(video,tmp_path/'negatives',0,1,.5)
    folder, count = collect(video,tmp_path/'negatives',0,1,.5,confirmed=True)
    assert count == 2 and len(list(folder.glob('*.jpg'))) == 2
    assert (folder/'_review_manifest.json').is_file()


def test_weapon_finetuning_does_not_train_person_or_turn_people_into_negatives(tmp_path):
    """Person boxes and weapon-free PERSON images must not define a class."""
    from prepare_datasets import valid_yolo, materialize
    from train_models import audit_dataset
    source = '0 .5 .5 .4 .6\n1 .55 .53 .08 .12\n'
    assert valid_yolo(source, {1:1}) == ['1 0.550000 0.530000 0.080000 0.120000']
    photos = []
    for i in range(8):
        path = tmp_path/f'{i}.png'
        cv2.imwrite(str(path),np.full((16,16,3),i*20,np.uint8))
        photos.append((path, ['1 .5 .5 .2 .2'], 'knife'))
    yaml_file = materialize(photos, tmp_path/'prepared')
    assert audit_dataset(yaml_file)['classes'] == ['knife']
    with pytest.raises(ValueError,match='Weapon-only'):
        materialize([(item[0], ['0 .5 .5 .2 .2'], 'person') for item in photos], tmp_path/'bad')
