import numpy as np
from app.engine import draw_hud_overlay, draw_interactive_zones

def test_draw_hud_overlay():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = draw_hud_overlay(
        frame,
        counts={'people': 2, 'vehicles': 1, 'electronics': 1, 'threats': 0, 'daily': 3},
        fps=29.8,
        device_str='CUDA:0',
        active_threat=False
    )
    assert annotated is not None
    assert np.any(annotated > 0)

def test_draw_hud_overlay_with_active_threat():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = draw_hud_overlay(
        frame,
        counts={'people': 1, 'threats': 1},
        fps=30.0,
        device_str='CPU',
        active_threat=True
    )
    assert annotated is not None
    assert np.any(annotated > 0)

def test_draw_interactive_zones_alert_state():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    zone = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]
    annotated = draw_interactive_zones(frame, zone=zone, is_intruded=True)
    assert annotated is not None
    assert np.any(annotated > 0)
