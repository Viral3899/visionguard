import numpy as np
import pytest
from app.business_analytics import FootTrafficAnalytics, draw_business_overlays


def test_foot_traffic_grid_accumulation():
    analytics = FootTrafficAnalytics(grid_size=(8, 8))
    now = 1000.0

    # Person walking at center (x=0.5, y=0.5)
    persons = [{'id': 1, 'foot': (0.5, 0.5)}]
    grid = analytics.update((480, 640), persons, now=now)
    assert grid[4, 4] > 0.0

    # Add more time to center cell
    now += 0.5
    grid = analytics.update((480, 640), persons, now=now)
    assert grid[4, 4] >= 0.4


def test_blindspot_deadzone_detection():
    analytics = FootTrafficAnalytics(grid_size=(8, 8))
    now = 1000.0

    # Simulate heavy traffic in center cell (4, 4)
    for _ in range(10):
        analytics.update((480, 640), [{'id': 1, 'foot': (0.5, 0.5)}], now=now)
        now += 0.2

    summary = analytics.get_summary()
    assert summary['max_dwell_seconds'] > 0.0
    assert summary['blindspot_cells_count'] > 0
    # Corner (0, 0) should be classified as a blindspot/deadzone
    blindspot_coords = [b['grid_coord'] for b in summary['blindspots']]
    assert (0, 0) in blindspot_coords


def test_heatmap_overlay_rendering():
    analytics = FootTrafficAnalytics(grid_size=(8, 8))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Accumulate foot traffic
    analytics.update((480, 640), [{'id': 1, 'foot': (0.5, 0.5)}], now=100.0)

    # Render heatmap overlay
    heatmap_frame = analytics.render_heatmap(frame.copy(), alpha=0.4)
    assert heatmap_frame.shape == (480, 640, 3)
    # The center should have non-zero heatmap color
    center_patch = heatmap_frame[230:250, 310:330]
    assert np.any(center_patch > 0)

    # Test draw_business_overlays
    events = [
        {'type': 'inventory_shrinkage', 'details': 'Shoplifting test', 'bbox': [100, 100, 200, 200]},
        {'type': 'worker_slip_fall', 'details': 'Slip fall test', 'bbox': [300, 200, 450, 300]}
    ]
    overlay_frame = draw_business_overlays(frame.copy(), events, foot_traffic=analytics, show_heatmap=True)
    assert overlay_frame.shape == (480, 640, 3)
