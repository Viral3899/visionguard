import numpy as np
import cv2
from app.traffic import SeatbeltDetector

def test_seatbelt_detected_with_diagonal_strap():
    detector = SeatbeltDetector()
    # Torso image with diagonal seatbelt line drawn
    crop = np.full((120, 100, 3), 180, dtype=np.uint8)
    cv2.line(crop, (15, 10), (85, 110), (30, 30, 30), 5)  # dark diagonal strap
    res = detector.check_seatbelt(crop)
    assert res['wearing_seatbelt'] is True

def test_seatbelt_violation_when_no_strap():
    detector = SeatbeltDetector()
    crop = np.full((120, 100, 3), 180, dtype=np.uint8)  # uniform plain shirt, no strap
    res = detector.check_seatbelt(crop)
    assert res['wearing_seatbelt'] is False
