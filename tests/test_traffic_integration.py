import numpy as np
from app.traffic import TrafficEnforcementManager, draw_traffic_overlays

def test_traffic_manager_processes_and_renders():
    mgr = TrafficEnforcementManager()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # 1 motorcycle + 3 riders
    moto = {'id': 1, 'class_id': 3, 'bbox': [100, 100, 240, 250]}
    riders = [
        {'id': 10, 'bbox': [110, 70, 150, 170]},
        {'id': 11, 'bbox': [150, 70, 190, 170]},
        {'id': 12, 'bbox': [190, 70, 230, 170]},
    ]
    violations, plates = mgr.process_traffic(frame, [moto], riders, stop_line=None)
    assert len(violations) >= 1
    assert any(v['type'] == 'triple_riding' for v in violations)

    annotated = draw_traffic_overlays(frame, violations, plates)
    assert np.any(annotated > 0)

def test_draw_traffic_overlays_renders_red_light_and_plates():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    violations = [
        {
            'type': 'red_light_violation',
            'severity': 'critical',
            'vehicle_id': 5,
            'vehicle_bbox': [100, 120, 250, 280],
            'confidence': 0.95,
            'message': 'Red Light Jump'
        }
    ]
    plates = [
        {
            'bbox': [120, 240, 220, 275],
            'plate_text': 'DL01CA9999',
            'confidence': 0.92
        }
    ]
    annotated = draw_traffic_overlays(frame, violations, plates)
    assert annotated is not None
    assert np.any(annotated > 0)
