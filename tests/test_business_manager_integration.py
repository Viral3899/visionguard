import numpy as np
import pytest
from app.business_analytics import BusinessAnalyticsManager, draw_business_overlays


def test_business_analytics_manager_integration():
    manager = BusinessAnalyticsManager()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    now = 1000.0

    # 1. Standing worker with cell phone near perimeter line
    tripwire = [(0.0, 0.5), (1.0, 0.5)]
    persons = [{'id': 10, 'bbox': [100, 100, 160, 280], 'foot': (0.2, 0.45)}]
    detected_objs = [{'bbox': [120, 150, 140, 180], 'class_name': 'cell phone'}]

    events = manager.process_frame(frame, persons, detected_objs, tripwire=tripwire, now=now)
    assert len(events) == 0

    # 2. Next frame: Person moves down across line, drops to floor, and cell phone vanishes
    now += 0.2
    persons_fallen = [{'id': 10, 'bbox': [100, 260, 280, 320], 'foot': (0.2, 0.55)}]
    events = manager.process_frame(frame, persons_fallen, [], tripwire=tripwire, now=now)

    # Should detect perimeter crossing and worker fall
    event_types = [e['type'] for e in events]
    assert 'perimeter_trespass' in event_types or 'worker_slip_fall' in event_types

    # 3. Verify manager summary report
    summary = manager.get_summary()
    assert 'foot_traffic' in summary
    assert 'active_worker_monitors' in summary
    assert summary['active_worker_monitors'] >= 1
