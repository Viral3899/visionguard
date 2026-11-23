import numpy as np
import cv2
from app.traffic import classify_traffic_light_color, SignalViolationDetector

def test_classify_traffic_light_red_and_green():
    # Synthetic red light (red patch on top third of 60x20 image)
    red_crop = np.zeros((60, 20, 3), dtype=np.uint8)
    red_crop[5:15, 5:15] = (0, 0, 255)  # BGR Red
    assert classify_traffic_light_color(red_crop) == 'red'

    # Synthetic green light (green patch on bottom third)
    green_crop = np.zeros((60, 20, 3), dtype=np.uint8)
    green_crop[45:55, 5:15] = (0, 255, 0)  # BGR Green
    assert classify_traffic_light_color(green_crop) == 'green'

def test_signal_violation_triggers_on_red_only():
    detector = SignalViolationDetector()
    stop_line = [[0.1, 0.5], [0.9, 0.5]]  # horizontal line at y=0.5
    vehicle = {'id': 1, 'bbox': [100, 240, 200, 300]}  # crossed y=250 (0.5 of 500)
    
    # When signal is RED -> violation
    violations_red = detector.check_violations(
        vehicles=[vehicle],
        signal_state='red',
        stop_line=stop_line,
        frame_shape=(500, 600)
    )
    assert len(violations_red) == 1
    assert violations_red[0]['type'] == 'red_light_violation'

    # When signal is GREEN -> no violation
    violations_green = detector.check_violations(
        vehicles=[vehicle],
        signal_state='green',
        stop_line=stop_line,
        frame_shape=(500, 600)
    )
    assert len(violations_green) == 0
