import time
import numpy as np
import pytest
from app.business_analytics import LossPreventionEngine, box_iou, box_contains_point


def test_box_iou_and_contains():
    b1 = [0, 0, 10, 10]
    b2 = [5, 5, 15, 15]
    iou = box_iou(b1, b2)
    assert 0.14 < iou < 0.15

    assert box_contains_point(b1, (5, 5))
    assert not box_contains_point(b1, (12, 12))


def test_loss_prevention_concealment_detection():
    engine = LossPreventionEngine(proximity_threshold=100.0, concealment_frames=2, cooldown_seconds=5.0)
    now = 1000.0

    # Frame 1: Person #101 standing near cell phone
    persons = [{'id': 101, 'bbox': [100, 100, 200, 300]}]
    detected_objects = [{'bbox': [150, 150, 180, 190], 'class_name': 'cell phone'}]
    events = engine.process((480, 640), detected_objects, persons, now=now)
    assert len(events) == 0
    assert len(engine.active_items) == 1

    # Frame 2: Cell phone disappears (concealed), person remains
    now += 0.1
    events = engine.process((480, 640), [], persons, now=now)
    assert len(events) == 0  # 1 missed frame, needs 2

    # Frame 3: Second missed frame -> should trigger inventory_shrinkage alert
    now += 0.1
    events = engine.process((480, 640), [], persons, now=now)
    assert len(events) == 1
    assert events[0]['type'] == 'inventory_shrinkage'
    assert events[0]['person_id'] == 101
    assert events[0]['item'] == 'cell phone'
    assert events[0]['severity'] == 'high'

    # Frame 4: Duplicate frame within cooldown should NOT produce duplicate event
    now += 0.1
    events = engine.process((480, 640), [], persons, now=now)
    assert len(events) == 0


def test_loss_prevention_ignores_non_theft_objects():
    engine = LossPreventionEngine()
    now = 1000.0
    persons = [{'id': 101, 'bbox': [100, 100, 200, 300]}]
    detected_objects = [{'bbox': [150, 150, 250, 250], 'class_name': 'car'}]
    events = engine.process((480, 640), detected_objects, persons, now=now)
    assert len(events) == 0
    assert len(engine.active_items) == 0
