import numpy as np
from app.trajectory import TrajectoryTracker

def test_trajectory_updates_and_prunes():
    tracker = TrajectoryTracker(max_history=5, max_idle_seconds=2.0)
    # Track 1 appears at time 10.0
    tracker.update([{'id': 1, 'bbox': [100, 100, 200, 200]}], now=10.0)
    assert tracker.get_dwell_time(1, now=10.0) == 0.0
    pts = tracker.get_trajectory(1)
    assert len(pts) == 1
    assert pts[0] == (150, 200)  # center foot

    # Track 1 moves at time 12.0
    tracker.update([{'id': 1, 'bbox': [110, 100, 210, 200]}], now=12.0)
    assert tracker.get_dwell_time(1, now=12.0) == 2.0
    assert len(tracker.get_trajectory(1)) == 2

    # Prune stale tracks after 15.0 (15 - 12 = 3s > 2s max_idle)
    tracker.update([], now=15.0)
    assert tracker.get_trajectory(1) == []

def test_draw_trajectories_on_frame():
    tracker = TrajectoryTracker(max_history=5)
    tracker.update([{'id': 1, 'bbox': [100, 100, 120, 120]}], now=1.0)
    tracker.update([{'id': 1, 'bbox': [110, 110, 130, 130]}], now=2.0)
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    annotated = tracker.draw(frame)
    assert np.any(annotated > 0)
