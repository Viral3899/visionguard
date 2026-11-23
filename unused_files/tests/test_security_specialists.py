import numpy as np
import pytest
from app.security_labels import CS2_FIREARMS, is_fight_label, weapon_class
from app.detectors import DetectionResult, DetectionManager, SpecialistWeaponDetector, ViolenceDetector
from app.incidents import IncidentTracker
from install_security_models import verify_labels


@pytest.mark.parametrize('name', ['Fight','FIGHT','violence', 'Violence/Fight','assault','Aggressive', 'Person Fighting'])
def test_positive_fight_labels(name):
    assert is_fight_label(name)


@pytest.mark.parametrize('name', ['NoViolence', 'No Fight', 'NON-VIOLENCE','nonfight','normal', 'person','knife','not fighting','0'])
def test_negative_or_unrelated_labels(name):
    assert not is_fight_label(name)


def test_game_gun_names_are_not_real_cctv_detections():
    assert len(CS2_FIREARMS) == 19
    assert all(weapon_class(x) is None for x in CS2_FIREARMS)
    assert 'knife' not in CS2_FIREARMS
    assert weapon_class('knife') == 'knife'  # stock COCO / different custom weights
    assert weapon_class('person') is None
    assert weapon_class('NoWeapon') is None


def test_reject_wrong_specialist_model():
    with pytest.raises(ValueError): verify_labels('weapon', ['AK47', 'M4A1-S'])
    assert verify_labels('fight', ['NoViolence','Violence'])
    with pytest.raises(ValueError): verify_labels('weapon', ['person','bus'])
    with pytest.raises(ValueError): verify_labels('fight', ['NoViolence','Normal'])


def test_weapon_event_true_raw_score_and_location():
    result = DetectionResult([[10,20,30,40], [0,0,50,50]], [0,1], [0.77,0.95],
                             {0:'gun', 1:'person'})
    events = SpecialistWeaponDetector.analyze_weapons(np.zeros((100,100,3),dtype=np.uint8), result)
    assert len(events) == 1
    assert events[0]['type'] == 'weapon'
    assert events[0]['subtype'] == 'gun'
    assert events[0]['model_label'] == 'gun'
    assert events[0]['confidence'] == 0.77
    assert events[0]['bbox'] == [.1,.2,.3,.4]


def test_fight_event_excludes_no_violence_and_confidence_not_inflated():
    v = ViolenceDetector.__new__(ViolenceDetector)
    result = DetectionResult([[0,0,10,10],[10,10,30,30]], [0,1], [.99,.54],
                             {0:'NoViolence',1:'Violence'})
    events = v.analyze_violence(np.zeros((40,40,3),dtype=np.uint8), result)
    assert len(events) == 1 and events[0]['confidence'] == .54
    assert events[0]['type'] == 'violence'


def test_two_frame_confirmation_one_picture_only():
    tracker = IncidentTracker(minimum=.5, confirmations=2)
    event = dict(type='weapon', subtype='gun', confidence=.62,
                 bbox=[.1,.1,.4,.4])
    assert tracker.update([event], 0.0) == []
    assert len(tracker.update([event], 0.1)) == 1
    assert tracker.update([event], .2) == []
    assert tracker.update([dict(event, confidence=.49)], .3) == []
