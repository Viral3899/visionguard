import numpy as np
import pytest
from app.detectors import draw_security_bounding_boxes, ObjectDetector, DetectionResult


def test_alarming_activity_persists_when_daily_objects_disabled():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Alarming event (e.g. weapon / violence)
    alarming_events = [{
        'type': 'weapon',
        'subtype': 'gun',
        'confidence': 0.92,
        'bbox': [0.2, 0.2, 0.4, 0.5]
    }]

    # 2. Daily life object (e.g. laptop / backpack)
    daily_result = DetectionResult(
        boxes=[[400, 300, 550, 450]],
        classes=[63],
        confidences=[0.88],
        class_names={63: 'laptop'}
    )

    detector = ObjectDetector()

    # Mode A: show_objects = False -> only alarming event drawn
    show_objects = False
    frame_no_objects = frame.copy()
    if show_objects:
        frame_no_objects = detector.draw_daily_objects(frame_no_objects, daily_result)
    frame_no_objects = draw_security_bounding_boxes(frame_no_objects, alarming_events)

    # Weapon region must be drawn
    weapon_region = frame_no_objects[int(0.2*480):int(0.5*480), int(0.2*640):int(0.4*640)]
    assert np.any(weapon_region > 0), "Weapon box must be drawn even when show_objects is False"
    # Laptop region must NOT be drawn
    laptop_region = frame_no_objects[300:450, 400:550]
    assert np.all(laptop_region == 0), "Laptop box must not be drawn when show_objects is False"

    # Mode B: show_objects = True -> both drawn
    frame_with_objects = frame.copy()
    frame_with_objects = detector.draw_daily_objects(frame_with_objects, daily_result)
    frame_with_objects = draw_security_bounding_boxes(frame_with_objects, alarming_events)
    assert np.any(frame_with_objects[int(0.2*480):int(0.5*480), int(0.2*640):int(0.4*640)] > 0)
    assert np.any(frame_with_objects[300:450, 400:550] > 0)
