# Intelligent Traffic & Road Safety Enforcement Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an intelligent road safety and traffic violation enforcement suite for VisionGuard: Automatic Number Plate Recognition (ANPR / Plate OCR), Red-Light Signal Jumping detection, Motorcycle Triple-Riding violation detection, and Driver Seatbelt compliance monitoring.

**Architecture:** Create a modular traffic analytics engine in `app/traffic.py` containing `TripleRidingDetector`, `SignalViolationDetector`, `SeatbeltDetector`, `LicensePlateOCR`, and `TrafficEnforcementManager`. Integrate traffic violation detection and custom overlay rendering into `CameraWorker` in `app/engine.py`.

**Tech Stack:** Python 3.12, OpenCV, PyTorch, Ultralytics YOLOv8, NumPy, FastAPI.

**Spec:** [traffic_enforcement_design.md](file:///C:/Users/viral/.gemini/antigravity-cli/brain/637c22a2-440a-4c14-9270-598cc4e7fab4/traffic_enforcement_design.md)

## Global Constraints
- Must not break any of the existing 34 unit and integration tests.
- High performance: traffic analytics must execute within video frame processing budgets without dropping stream FPS.
- Robust character extraction: license plate OCR must have clean regex normalization and handle varying illumination/aspect ratios.
- Non-intrusive: violations emit structured alert events into `visionguard.db` and display prominent alert badges on frame.

## Review Focus
1. Rider count association must correctly differentiate 1 or 2 riders (legal) from 3+ riders (triple riding violation).
2. Traffic light color classification must handle diverse illumination and avoid false red signals during daylight/glare.
3. Stop-line breach must only trigger on red signal state, not on green or yellow.
4. License plate text parser must normalize spacing and strip special characters cleanly.
5. All traffic violation bounding boxes must remain clamped inside frame dimensions.

---

### Task 1: Motorcycle Triple-Riding Detector

**Files:**
- Create: `app/traffic.py`
- Create: `tests/test_traffic_triple_riding.py`

**Interfaces:**
- Input: `motorcycles: list[dict]`, `persons: list[dict]`, `frame_shape: tuple[int, int]`
- Output: `list[dict]` of triple-riding violation events.

- [ ] **Step 1: Write the failing test for `TripleRidingDetector`**

```python
# tests/test_traffic_triple_riding.py
from app.traffic import TripleRidingDetector

def test_triple_riding_single_and_double_rider_allowed():
    detector = TripleRidingDetector()
    moto = {'id': 1, 'bbox': [100, 100, 200, 250]}
    # 2 riders overlapping motorcycle
    persons = [
        {'id': 10, 'bbox': [110, 80, 150, 180]},
        {'id': 11, 'bbox': [150, 80, 190, 180]},
    ]
    violations = detector.detect_violations([moto], persons)
    assert len(violations) == 0

def test_triple_riding_three_riders_violation():
    detector = TripleRidingDetector()
    moto = {'id': 2, 'bbox': [100, 100, 250, 250]}
    # 3 riders overlapping motorcycle
    persons = [
        {'id': 20, 'bbox': [110, 70, 145, 170]},
        {'id': 21, 'bbox': [145, 70, 180, 170]},
        {'id': 22, 'bbox': [180, 70, 220, 170]},
    ]
    violations = detector.detect_violations([moto], persons)
    assert len(violations) == 1
    assert violations[0]['type'] == 'triple_riding'
    assert violations[0]['rider_count'] == 3
    assert violations[0]['motorcycle_id'] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_traffic_triple_riding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.traffic'`

- [ ] **Step 3: Implement `TripleRidingDetector` in `app/traffic.py`**
- Compute horizontal and vertical overlap between rider bboxes and motorcycle bounding envelopes.
- Count riders per motorcycle.
- If `rider_count >= 3`, emit violation dict with severity `'high'`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_traffic_triple_riding.py -v`
Expected: PASS

---

### Task 2: Traffic Light State & Red-Light Signal Violation Detector

**Files:**
- Modify: `app/traffic.py`
- Create: `tests/test_traffic_signal.py`

**Interfaces:**
- Input: `classify_traffic_light_color(crop: np.ndarray) -> str`
- Input: `SignalViolationDetector.check_violations(vehicles, traffic_lights, stop_line, frame)`
- Output: `list[dict]` of red-light jumping violations.

- [ ] **Step 1: Write the failing test for `SignalViolationDetector`**

```python
# tests/test_traffic_signal.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_traffic_signal.py -v`
Expected: FAIL with `cannot import name 'classify_traffic_light_color'`

- [ ] **Step 3: Implement `classify_traffic_light_color` and `SignalViolationDetector` in `app/traffic.py`**
- In `classify_traffic_light_color`:
  - Split crop into upper (red) and lower (green) halves/thirds.
  - Apply HSV masks for Red (`H: 0-10 & 160-180`) and Green (`H: 35-85`).
  - Return dominant active color.
- In `SignalViolationDetector`:
  - Check vehicle bounding box crossing past `stop_line`.
  - Trigger violation if `signal_state == 'red'`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_traffic_signal.py -v`
Expected: PASS

---

### Task 3: Driver Seatbelt Compliance Detector

**Files:**
- Modify: `app/traffic.py`
- Create: `tests/test_traffic_seatbelt.py`

**Interfaces:**
- Input: `SeatbeltDetector.check_seatbelt(driver_crop: np.ndarray) -> dict`
- Output: `{'wearing_seatbelt': bool, 'confidence': float}`

- [ ] **Step 1: Write the failing test for `SeatbeltDetector`**

```python
# tests/test_traffic_seatbelt.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_traffic_seatbelt.py -v`
Expected: FAIL with `cannot import name 'SeatbeltDetector'`

- [ ] **Step 3: Implement `SeatbeltDetector` in `app/traffic.py`**
- Uses oriented edge gradients and Hough line detection in the $30^\circ - 65^\circ$ diagonal range across the occupant torso.
- Returns compliance dictionary.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_traffic_seatbelt.py -v`
Expected: PASS

---

### Task 4: License Plate Localizer & ANPR OCR

**Files:**
- Modify: `app/traffic.py`
- Create: `tests/test_traffic_plate_ocr.py`

**Interfaces:**
- Input: `vehicle_crop: np.ndarray`
- Output: `list[dict]` containing detected plate bounding box, clean plate text, and confidence.

- [ ] **Step 1: Write the failing test for `LicensePlateOCR`**

```python
# tests/test_traffic_plate_ocr.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_traffic_plate_ocr.py -v`
Expected: FAIL with `cannot import name 'LicensePlateOCR'`

- [ ] **Step 3: Implement `LicensePlateOCR` and text cleaner in `app/traffic.py`**
- Candidate extraction via contour aspect ratios (2.0 to 4.5), area filtering, and thresholding.
- Character extraction pipeline with clean alphanumeric normalization.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_traffic_plate_ocr.py -v`
Expected: PASS

---

### Task 5: Traffic Enforcement Manager & Visualizer Integration

**Files:**
- Modify: `app/traffic.py`
- Modify: `app/engine.py`
- Create: `tests/test_traffic_integration.py`

**Interfaces:**
- `TrafficEnforcementManager.process_frame(frame, objects, persons, stop_line) -> tuple[list[dict], np.ndarray]`
- `draw_traffic_overlays(frame, violations, plates) -> np.ndarray`

- [ ] **Step 1: Write integration test**

```python
# tests/test_traffic_integration.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_traffic_integration.py -v`
Expected: FAIL with `cannot import name 'TrafficEnforcementManager'`

- [ ] **Step 3: Implement `TrafficEnforcementManager` and `draw_traffic_overlays` in `app/traffic.py` and hook into `app/engine.py`**
- Render distinct high-contrast badges for all traffic violations and plate OCR tags.
- Hook into `CameraWorker._run()` in `app/engine.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_traffic_integration.py -v`
Expected: PASS

---

### Task 6: Full Regression Testing & Verification

- [ ] **Step 1: Run complete test suite across all modules**
Run: `.venv\Scripts\python.exe -m pytest`
Expected: All tests pass (original 34 + new traffic tests).

- [ ] **Step 2: End-to-end pipeline verification**
Run synthetic pipeline check ensuring full compatibility.
