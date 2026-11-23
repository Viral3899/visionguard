import numpy as np
import pytest
from app.detectors import ObjectDetector, DetectionResult


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
