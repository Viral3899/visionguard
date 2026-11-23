import time
import pytest
from app.business_analytics import WorkerSafetyEngine


def test_worker_slip_and_fall():
    engine = WorkerSafetyEngine(fall_aspect_ratio_min=1.15, fall_velocity_threshold=50.0, immobility_seconds=5.0)
    now = 1000.0

    # Person #42 starts standing: width=60, height=180 (AR = 0.33)
    p_standing = [{'id': 42, 'bbox': [100, 100, 160, 280]}]
    events1 = engine.process((480, 640), p_standing, now=now)
    assert len(events1) == 0

    # Person #42 falls suddenly: centroid drops rapidly and person becomes recumbent: width=190, height=70 (AR = 2.71)
    now += 0.2
    p_fallen = [{'id': 42, 'bbox': [100, 280, 290, 350]}]
    events2 = engine.process((480, 640), p_fallen, now=now)
    assert len(events2) == 1
    assert events2[0]['type'] == 'worker_slip_fall'
    assert events2[0]['person_id'] == 42
    assert events2[0]['severity'] == 'critical'


def test_worker_prolonged_immobility_unresponsive():
    engine = WorkerSafetyEngine(fall_aspect_ratio_min=1.15, fall_velocity_threshold=50.0, immobility_seconds=3.0)
    now = 1000.0

    # Initial fall
    p_fallen = [{'id': 7, 'bbox': [100, 280, 290, 350]}]
    events = engine.process((480, 640), p_fallen, now=now)
    assert any(e['type'] == 'worker_slip_fall' for e in events)

    # Worker stays down for 2.0s (< 3.0s threshold)
    now += 2.0
    events = engine.process((480, 640), p_fallen, now=now)
    assert not any(e['type'] == 'unresponsive_worker' for e in events)

    # Worker stays down past 3.0s (e.g. 3.5s total)
    now += 1.5
    events = engine.process((480, 640), p_fallen, now=now)
    unresp = [e for e in events if e['type'] == 'unresponsive_worker']
    assert len(unresp) == 1
    assert unresp[0]['person_id'] == 7
    assert unresp[0]['severity'] == 'critical'
    assert unresp[0]['duration_seconds'] >= 3.0
