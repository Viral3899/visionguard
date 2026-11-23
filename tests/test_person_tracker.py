import time
import numpy as np
import pytest
from app.person_tracker import PersonDeduplicator, extract_person_signature, compare_signatures


def test_signature_extraction_and_comparison():
    # Create two distinctly colored person crops (e.g., Red vs Blue)
    img_red = np.zeros((100, 50, 3), dtype=np.uint8)
    img_red[:, :] = (0, 0, 255)  # BGR Red

    img_blue = np.zeros((100, 50, 3), dtype=np.uint8)
    img_blue[:, :] = (255, 0, 0)  # BGR Blue

    sig_red1 = extract_person_signature(img_red, [0, 0, 50, 100])
    sig_red2 = extract_person_signature(img_red, [0, 0, 50, 100])
    sig_blue = extract_person_signature(img_blue, [0, 0, 50, 100])

    assert sig_red1 is not None
    assert sig_blue is not None

    # Red compared to red should be near 1.0
    sim_same = compare_signatures(sig_red1, sig_red2)
    assert sim_same > 0.95

    # Red compared to blue should be low
    sim_diff = compare_signatures(sig_red1, sig_blue)
    assert sim_diff < 0.5


def test_person_deduplication_continuous_tracking():
    tracker = PersonDeduplicator(cooldown_seconds=300.0)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[20:120, 20:60] = (0, 255, 0)  # Green shirt person

    # Person 1 detected at t=0
    persons_t0 = [{'id': 1, 'bbox': [20, 20, 60, 120]}]
    in_frame, total_5m = tracker.update(frame, persons_t0, now=1000.0)
    assert in_frame == 1
    assert total_5m == 1

    # Person 1 still in frame at t=10s
    in_frame, total_5m = tracker.update(frame, persons_t0, now=1010.0)
    assert in_frame == 1
    assert total_5m == 1


def test_person_deduplication_reentry_within_5min():
    tracker = PersonDeduplicator(cooldown_seconds=300.0, sim_threshold=0.75)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[20:120, 20:60] = (0, 120, 255)  # Orange shirt person

    # Person 1 enters at t=0
    tracker.update(frame, [{'id': 1, 'bbox': [20, 20, 60, 120]}], now=1000.0)
    assert tracker.total_unique_passers == 1

    # Person 1 leaves frame at t=10s (empty frame)
    in_frame, total_5m = tracker.update(frame, [], now=1010.0)
    assert in_frame == 0
    assert total_5m == 1

    # Person 1 re-enters at t=120s (2 minutes later) with NEW track ID 2
    in_frame, total_5m = tracker.update(frame, [{'id': 2, 'bbox': [20, 20, 60, 120]}], now=1120.0)
    assert in_frame == 1
    # MUST NOT count the same person again within 5 minutes
    assert total_5m == 1


def test_person_deduplication_different_person_increments():
    tracker = PersonDeduplicator(cooldown_seconds=300.0, sim_threshold=0.75)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Person 1: Red
    frame[20:120, 20:60] = (0, 0, 255)
    tracker.update(frame, [{'id': 1, 'bbox': [20, 20, 60, 120]}], now=1000.0)
    assert tracker.total_unique_passers == 1

    # Person 2: Blue
    frame[20:120, 100:140] = (255, 0, 0)
    in_frame, total_5m = tracker.update(frame, [
        {'id': 1, 'bbox': [20, 20, 60, 120]},
        {'id': 2, 'bbox': [100, 20, 140, 120]}
    ], now=1005.0)

    assert in_frame == 2
    assert total_5m == 2


def test_person_deduplication_reentry_after_5min_recounts():
    tracker = PersonDeduplicator(cooldown_seconds=300.0, sim_threshold=0.75)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[20:120, 20:60] = (50, 150, 50)

    # Person 1 enters at t=1000.0
    tracker.update(frame, [{'id': 1, 'bbox': [20, 20, 60, 120]}], now=1000.0)
    assert tracker.total_unique_passers == 1

    # Person leaves frame
    tracker.update(frame, [], now=1010.0)

    # 305 seconds later (> 5 minutes cooldown), same person re-enters with ID 3
    in_frame, total_5m = tracker.update(frame, [{'id': 3, 'bbox': [20, 20, 60, 120]}], now=1315.0)
    assert in_frame == 1
    # After 5 minutes, counting again is expected
    assert total_5m == 2
