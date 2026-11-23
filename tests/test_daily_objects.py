import numpy as np
import pytest
from app.detectors import ObjectDetector, DetectionResult, format_daily_object_label


def test_format_daily_object_label_cf_score_and_aliases():
    assert format_daily_object_label('cell phone', 0.92) == 'MOBILE | CF: 92%'
    assert format_daily_object_label('mobile', 0.854) == 'MOBILE | CF: 85%'
    assert format_daily_object_label('chair', 0.88) == 'CHAIR | CF: 88%'
    assert format_daily_object_label('powerbank', 0.76) == 'POWERBANK | CF: 76%'
    assert format_daily_object_label('power bank', 0.80) == 'POWERBANK | CF: 80%'
    assert format_daily_object_label('battery', 0.70) == 'POWERBANK | CF: 70%'
    assert format_daily_object_label('laptop', 0.95) == 'LAPTOP | CF: 95%'


def test_draw_daily_objects_mobile_chair_powerbank():
    detector = ObjectDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_result = DetectionResult(
        boxes=[
            [20, 20, 100, 150],    # mobile / cell phone
            [120, 50, 260, 300],   # chair
            [300, 100, 420, 220]   # powerbank
        ],
        classes=[67, 56, 99],
        confidences=[0.91, 0.84, 0.79],
        class_names={67: 'cell phone', 56: 'chair', 99: 'powerbank'}
    )
    drawn = detector.draw_daily_objects(frame.copy(), mock_result)
    sub_mobile = drawn[20:150, 20:100]
    sub_chair = drawn[50:300, 120:260]
    sub_powerbank = drawn[100:220, 300:420]
    assert np.any(sub_mobile > 0), "Mobile bounding box was not drawn"
    assert np.any(sub_chair > 0), "Chair bounding box was not drawn"
    assert np.any(sub_powerbank > 0), "Powerbank bounding box was not drawn"


def test_draw_daily_objects():
    detector = ObjectDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_result = DetectionResult(
        boxes=[[50, 50, 150, 200], [200, 100, 350, 300]],
        classes=[67, 39],  # cell phone, bottle
        confidences=[0.85, 0.78],
        class_names={67: 'cell phone', 39: 'bottle'}
    )
    drawn = detector.draw_daily_objects(frame.copy(), mock_result)
    sub1 = drawn[50:200, 50:150]
    sub2 = drawn[100:300, 200:350]
    assert np.any(sub1 > 0), "Cell phone box was not drawn"
    assert np.any(sub2 > 0), "Bottle box was not drawn"


def test_daily_objects_filter_person_and_weapons():
    detector = ObjectDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_result = DetectionResult(
        boxes=[[10, 10, 100, 100], [150, 150, 250, 250]],
        classes=[0, 62],  # person (handled by tracker), tv (daily object)
        confidences=[0.90, 0.82],
        class_names={0: 'person', 62: 'tv'}
    )
    drawn = detector.draw_daily_objects(frame.copy(), mock_result)
    # Person box should be skipped by draw_daily_objects (person has ByteTrack and its own renderer)
    assert np.all(drawn[10:100, 10:100] == 0), "Person box should not be drawn by draw_daily_objects"
    # TV box should be drawn
    assert np.any(drawn[150:250, 150:250] > 0), "TV box should be drawn"
