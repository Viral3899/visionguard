import time
import pytest
from app.business_analytics import PerimeterTrespassEngine


def test_perimeter_directional_crossing():
    engine = PerimeterTrespassEngine(tailgating_gap_seconds=2.0, cooldown_seconds=5.0)
    tripwire = [(0.0, 0.5), (1.0, 0.5)]  # Horizontal tripwire across frame mid-height
    now = 1000.0

    # Person #1 moves from above the line (y=0.40) to below the line (y=0.60)
    p1_f1 = [{'id': 1, 'foot': (0.5, 0.40), 'bbox': [200, 100, 280, 192]}]
    events1 = engine.process((480, 640), p1_f1, tripwire=tripwire, now=now)
    assert len(events1) == 0

    now += 0.2
    p1_f2 = [{'id': 1, 'foot': (0.5, 0.60), 'bbox': [200, 200, 280, 288]}]
    events2 = engine.process((480, 640), p1_f2, tripwire=tripwire, now=now)
    assert len(events2) == 1
    assert events2[0]['type'] == 'perimeter_trespass'
    assert events2[0]['person_id'] == 1
    assert events2[0]['severity'] == 'critical'


def test_perimeter_tailgating_detection():
    engine = PerimeterTrespassEngine(tailgating_gap_seconds=2.0, cooldown_seconds=5.0)
    tripwire = [(0.0, 0.5), (1.0, 0.5)]
    now = 1000.0

    # Person #1 crosses line
    engine.process((480, 640), [{'id': 1, 'foot': (0.5, 0.40)}], tripwire=tripwire, now=now)
    now += 0.1
    events1 = engine.process((480, 640), [{'id': 1, 'foot': (0.5, 0.60)}], tripwire=tripwire, now=now)
    assert any(e['type'] == 'perimeter_trespass' for e in events1)

    # Person #2 follows Person #1 closely within 0.8s
    now += 0.4
    engine.process((480, 640), [{'id': 2, 'foot': (0.52, 0.40)}], tripwire=tripwire, now=now)
    now += 0.4  # delta t = 0.8s from Person 1 crossing
    events2 = engine.process((480, 640), [{'id': 2, 'foot': (0.52, 0.60)}], tripwire=tripwire, now=now)

    # Should detect tailgating
    tailgate_events = [e for e in events2 if e['type'] == 'tailgating_detected']
    assert len(tailgate_events) == 1
    assert tailgate_events[0]['severity'] == 'critical'
    assert 1 in tailgate_events[0]['person_ids']
    assert 2 in tailgate_events[0]['person_ids']
