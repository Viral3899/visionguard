import numpy as np
import cv2
from app.traffic import LicensePlateOCR, clean_plate_text

def test_clean_plate_text():
    assert clean_plate_text("MH-12-AB-1234!") == "MH12AB1234"
    assert clean_plate_text(" dl 01  ca 9999 ") == "DL01CA9999"

def test_license_plate_localization_and_reading():
    ocr = LicensePlateOCR()
    # Synthetic vehicle front bumper with high-contrast license plate
    frame = np.full((200, 300, 3), 60, dtype=np.uint8)
    plate_x1, plate_y1, plate_x2, plate_y2 = 80, 130, 220, 170
    cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (240, 240, 240), -1)
    cv2.putText(frame, "DL01CA99", (plate_x1 + 10, plate_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)
    
    candidates = ocr.extract_plates(frame)
    assert len(candidates) >= 1
    assert candidates[0]['plate_text'] != ""
