# Advanced Vision & Turnkey Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide full-spectrum multi-category bounding boxes across all 80 COCO classes, advanced vision analytics (motion trajectory trails, dwell-time counters, skeleton pose visualizer, interactive zone triggers, and live detection HUD), and a 1-click `setup.bat` installer that provisions `.venv`, installs GPU/CPU dependencies, and auto-downloads/verifies all required AI models.

**Architecture:** Extend the VisionGuard FastAPI + OpenCV + Ultralytics pipeline. Generalize object detection by removing knife-only filtering in configuration and adding category color mapping. Add a lightweight `TrajectoryTracker` in `app/trajectory.py` for motion trails and dwell times. Upgrade frame rendering in `app/engine.py` and `app/detectors.py` to draw category badges, motion tails, skeleton connectors, interactive zones, and a live HUD. Provide a robust `setup_models.py` and `setup.bat` for headless automated installation and verification.

**Tech Stack:** Python 3.12, PyTorch, TorchVision, Ultralytics YOLOv8, ByteTrack, OpenCV, FastAPI, Hugging Face Hub, Windows Batch (`.bat`).

**Spec:** [design_doc.md](file:///C:/Users/viral/.gemini/antigravity-cli/brain/637c22a2-440a-4c14-9270-598cc4e7fab4/design_doc.md)

## Global Constraints
- Target platform: Windows 10/11 64-bit, Python 3.12 (with support for Python 3.10-3.11).
- Non-breaking: all existing 23 test cases in `tests/` must remain passing.
- Fail-safe bounding boxes: bounding boxes must render even if ByteTrack has not yet assigned a track ID.
- Alarming events guarantee: security alerts (weapons, violence, fire, smoke, falls) must always be rendered with high-visibility alert boxes regardless of UI display toggles.

## Review Focus
1. Empty `OBJECT_CLASSES` in `.env` must be parsed as `None`/empty tuple (meaning all 80 COCO classes) without raising ValueError.
2. Trajectory history must be bounded (e.g. 25 points per track) and pruned when tracks expire to avoid unbounded memory growth.
3. Fallback when `result.boxes.id is None` in `engine.py` must render person boxes cleanly with track placeholder instead of dropping the box.
4. `setup_models.py` must handle network errors gracefully and verify model weights integrity.
5. In video frames with narrow dimensions (e.g. mobile streams), the HUD and badges must remain clamped within the frame boundaries.

---

### Task 1: Configuration & Multi-Category Color-Coded Bounding Boxes

**Files:**
- Modify: `app/config.py`
- Modify: `.env`
- Modify: `.env.example`
- Modify: `app/detectors.py`
- Create: `tests/test_multi_category_boxes.py`

**Interfaces:**
- Consumes: `settings.object_classes`, `DetectionResult`
- Produces: `get_category_style(class_name: str) -> dict`, `draw_daily_objects(...)` rendering all COCO categories with designated colors and badges.

- [ ] **Step 1: Write the failing test for multi-category styling and empty object_classes parsing**

```python
# tests/test_multi_category_boxes.py
import numpy as np
from app.detectors import ObjectDetector, DetectionResult, get_category_style, format_daily_object_label
from app.config import Settings

def test_empty_object_classes_allows_all():
    s = Settings(object_classes=())
    assert s.object_classes == ()

def test_category_styles():
    assert get_category_style('cell phone')['category'] == 'electronics'
    assert get_category_style('car')['category'] == 'vehicle'
    assert get_category_style('backpack')['category'] == 'daily'
    assert get_category_style('dog')['category'] == 'animal'
    assert get_category_style('chair')['category'] == 'furniture'

def test_draw_daily_objects_renders_multiple_categories():
    detector = ObjectDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = DetectionResult(
        boxes=[[10, 10, 80, 80], [100, 100, 200, 200]],
        classes=[67, 2],  # cell phone, car
        confidences=[0.92, 0.88],
        class_names={67: 'cell phone', 2: 'car'}
    )
    annotated = detector.draw_daily_objects(frame, result)
    assert annotated is not None
    assert annotated.shape == frame.shape
    assert np.any(annotated > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_multi_category_boxes.py -v`
Expected: FAIL with missing `get_category_style` or import error.

- [ ] **Step 3: Implement category styles and update `ObjectDetector.draw_daily_objects` in `app/detectors.py` and `app/config.py`**
- In `app/detectors.py`:
  - Implement `CATEGORY_MAP` and `CATEGORY_COLORS`:
    - `vehicle`: `(255, 165, 0)` (Azure / Cobalt Blue)
    - `electronics`: `(220, 50, 220)` (Neon Violet)
    - `daily`: `(0, 190, 255)` (Warm Amber / Gold)
    - `animal`: `(50, 220, 50)` (Lime Green)
    - `furniture`: `(180, 180, 120)` (Slate Gray)
  - Implement `get_category_style(class_name: str) -> dict`.
  - Update `draw_daily_objects` to dynamically choose color and tag based on class category, draw corner accents, and render dark translucent badge with white text.
- In `app/config.py`:
  - Ensure `OBJECT_CLASSES` parses empty string `""` as empty tuple `()` (signifying all classes enabled).
- In `.env` and `.env.example`:
  - Update `OBJECT_CLASSES=` to empty by default.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_multi_category_boxes.py -v`
Expected: PASS

---

### Task 2: Motion Trajectory Trails & Dwell Time Tracker

**Files:**
- Create: `app/trajectory.py`
- Modify: `app/engine.py`
- Create: `tests/test_trajectory.py`

**Interfaces:**
- Consumes: Person/vehicle track IDs, bounding boxes, timestamps
- Produces: `TrajectoryTracker`:
  - `update(tracks: list[dict], now: float)`
  - `get_trajectory(track_id: int) -> list[tuple[int, int]]`
  - `get_dwell_time(track_id: int, now: float) -> float`
  - `draw_trajectories(frame: np.ndarray, color_map: dict | None = None) -> np.ndarray`

- [ ] **Step 1: Write the failing test for `TrajectoryTracker`**

```python
# tests/test_trajectory.py
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

    # Prune stale tracks
    tracker.update([], now=15.0)
    assert tracker.get_trajectory(1) == []

def test_draw_trajectories_on_frame():
    tracker = TrajectoryTracker(max_history=5)
    tracker.update([{'id': 1, 'bbox': [100, 100, 120, 120]}], now=1.0)
    tracker.update([{'id': 1, 'bbox': [110, 110, 130, 130]}], now=2.0)
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    annotated = tracker.draw(frame)
    assert np.any(annotated > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_trajectory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.trajectory'`

- [ ] **Step 3: Implement `app/trajectory.py` and integrate into `app/engine.py`**
- Implement `TrajectoryTracker` in `app/trajectory.py`:
  - Maintains `_histories: dict[int, deque[tuple[int, int]]]`
  - Maintains `_first_seen: dict[int, float]`
  - Maintains `_last_seen: dict[int, float]`
  - Method `draw(frame, color_map=None)` draws antialiased fading lines with increasing thickness towards current position, and a glowing endpoint dot.
- Integrate into `CameraWorker` in `app/engine.py`:
  - Initialize `self.trajectory_tracker = TrajectoryTracker(max_history=25, max_idle_seconds=5.0)`
  - In `_run()`, call `self.trajectory_tracker.update(persons, now=time.monotonic())`
  - Draw trajectories on `frame`.
  - Format person box label to include dwell time: `f"PERSON #{track_id} ({dwell_str}) | {conf:.0%}"`. If dwell time > `settings.loitering_time_seconds`, color label warning amber.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_trajectory.py -v`
Expected: PASS

---

### Task 3: Live Detection Summary HUD, Interactive Zones & Skeleton Pose Visualizer

**Files:**
- Modify: `app/engine.py`
- Modify: `app/detectors.py`
- Create: `tests/test_hud_and_zones.py`

**Interfaces:**
- Consumes: Active detection counts, camera zones/tripwires, person keypoints, confirmed events
- Produces:
  - `_draw_hud(frame, counts, fps, device_name)` in `app/engine.py`
  - `_draw_interactive_geometry(frame, analytics, active_alerts)` in `app/engine.py`
  - Skeleton pose overlay connection lines in `app/detectors.py`

- [ ] **Step 1: Write the failing test for HUD and zone drawing**

```python
# tests/test_hud_and_zones.py
import numpy as np
from app.engine import draw_hud_overlay, draw_interactive_zones

def test_draw_hud_overlay():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = draw_hud_overlay(
        frame,
        counts={'people': 2, 'vehicles': 1, 'electronics': 1, 'threats': 0},
        fps=29.8,
        device_str='CUDA:0',
        active_threat=False
    )
    assert annotated is not None
    assert np.any(annotated > 0)

def test_draw_interactive_zones_alert_state():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    zone = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]
    annotated = draw_interactive_zones(frame, zone=zone, is_intruded=True)
    assert annotated is not None
    assert np.any(annotated > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_hud_and_zones.py -v`
Expected: FAIL with missing `draw_hud_overlay` / `draw_interactive_zones`

- [ ] **Step 3: Implement HUD overlay, interactive zones, and skeleton connectors**
- In `app/engine.py`:
  - Implement `draw_hud_overlay(frame, counts, fps, device_str, active_threat)`:
    - Renders sleek top banner bar with glassmorphism dark background `(14, 20, 30)` and thin cyan/accent border.
    - Displays active object counts: `👤 People: X | 🚗 Vehicles: Y | 📱 Devices: Z | 🎯 Total: N`
    - Displays engine metrics: `30.2 FPS | GPU` and security badge `● SECURE` (or pulsating `[!] THREAT ALERT` in red if active threat).
  - Implement `draw_interactive_zones(frame, zone, tripwire, is_intruded, tripwire_crossed)`:
    - Normal state: subtle cyan line / polygon with 15% opacity fill.
    - Alert state (intrusion/tripwire crossed): glowing neon red with `[!] ZONE INTRUSION` banner.
  - Fix person box fallback: if `result.boxes.id is None`, still extract bounding boxes and render them cleanly as `PERSON | {conf:.0%}` so people never vanish on tracking latency.
- In `app/detectors.py`:
  - Enhance `_draw_keypoints` to connect the 17 standard COCO keypoints (head/face, arms/shoulders, torso/hips, legs/knees/ankles) with antialiased colored bones.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_hud_and_zones.py -v`
Expected: PASS

---

### Task 4: Turnkey `setup_models.py` Automated Downloader & Verifier

**Files:**
- Create: `setup_models.py`
- Create: `tests/test_setup_models.py`

**Interfaces:**
- Consumes: Hugging Face repos and Ultralytics YOLO core weights
- Produces:
  - CLI script: `python setup_models.py [--force] [--check-only]`
  - Validates and downloads:
    - `yolov8n.pt`
    - `yolov8n-pose.pt`
    - `yolov26-fire-smoke.pt`
    - `threat-yolov8n.pt`
    - `fight-yolov8-nano.pt`
  - Function: `verify_and_download_all_models(models_dir: Path, force: bool = False) -> dict[str, bool]`

- [ ] **Step 1: Write the failing test for `setup_models.py`**

```python
# tests/test_setup_models.py
from pathlib import Path
from setup_models import MODEL_SPECS, check_model_file

def test_model_specs_present():
    assert 'yolov8n.pt' in MODEL_SPECS
    assert 'yolov8n-pose.pt' in MODEL_SPECS
    assert 'yolov26-fire-smoke.pt' in MODEL_SPECS
    assert 'threat-yolov8n.pt' in MODEL_SPECS
    assert 'fight-yolov8-nano.pt' in MODEL_SPECS

def test_check_model_file_on_existing():
    # yolov8n.pt exists in root or models
    root = Path(__file__).resolve().parent.parent
    p = root / 'yolov8n.pt'
    if not p.is_file():
        p = root / 'models' / 'yolov8n.pt'
    if p.is_file():
        assert check_model_file(p) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_setup_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'setup_models'`

- [ ] **Step 3: Implement `setup_models.py`**
- Implement `setup_models.py`:
  - Define `MODEL_SPECS` with URLs, repos, target paths, and expected minimum file size.
  - Core models (`yolov8n.pt`, `yolov8n-pose.pt`): downloaded via Ultralytics or direct official releases.
  - Community specialist models (`yolov26-fire-smoke.pt`, `threat-yolov8n.pt`, `fight-yolov8-nano.pt`): downloaded via `huggingface_hub.hf_hub_download` to staging file `.staging-<key>.pt`, verified using `YOLO(str(staging))`, and atomically moved to `models/<name>`.
  - CLI flags: `--check-only`, `--force`.
  - Prints clean tabular status report.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_setup_models.py -v`
Expected: PASS

---

### Task 5: Turnkey `setup.bat` 1-Click Windows Installer

**Files:**
- Create: `setup.bat`
- Create: `tests/test_setup_bat.py`

**Interfaces:**
- Consumes: System Python, NVIDIA GPU / CUDA drivers
- Produces: Automated Windows first-time installation and readiness:
  - Python check -> `.venv` creation -> PyTorch (CUDA / CPU) -> Requirements -> Config -> Model Download & Verification -> Launch prompt.

- [ ] **Step 1: Write test verifying `setup.bat` existence and key phases**

```python
# tests/test_setup_bat.py
from pathlib import Path

def test_setup_bat_structure():
    root = Path(__file__).resolve().parent.parent
    bat = root / 'setup.bat'
    assert bat.is_file(), "setup.bat must exist in project root"
    content = bat.read_text(encoding='utf-8', errors='ignore')
    assert "bootstrap_env" in content or "venv" in content
    assert "setup_models.py" in content
    assert "requirements.txt" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_setup_bat.py -v`
Expected: FAIL with `setup.bat must exist in project root`

- [ ] **Step 3: Implement `setup.bat`**
- Create `setup.bat` in the workspace root:
  - Section 1: Check Python (64-bit Python 3.12, 3.11, or 3.10 via `py -3.12` or `python`).
  - Section 2: Create virtual environment `.venv` if not present.
  - Section 3: Check NVIDIA GPU (`nvidia-smi`). If CUDA driver is available, install PyTorch with CUDA (`--index-url https://download.pytorch.org/whl/cu124`). Otherwise install PyTorch CPU.
  - Section 4: Install `requirements.txt`.
  - Section 5: Initialize `.env` from `.env.example` if `.env` does not exist.
  - Section 6: Run `.venv\Scripts\python.exe setup_models.py` to auto-download all required AI models into `models/`.
  - Section 7: Verification check and prompt: `Setup Complete! Press 1 to Start VisionGuard Live, or any other key to exit.`

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_setup_bat.py -v`
Expected: PASS

---

### Task 6: Full Regression Testing & Verification

**Files:**
- Run all tests: `tests/`
- Test sample video analysis: `test_video.mp4`
- Modify: `README.md`

- [ ] **Step 1: Run all unit and integration tests**

Run: `.venv\Scripts\python.exe -m pytest -v`
Expected: PASS (all existing 23 tests + all new tests pass)

- [ ] **Step 2: Test live frame annotation pipeline with test video**
- Run a short pipeline verification script on `test_video.mp4` to ensure:
  - Category bounding boxes are drawn with accents.
  - Trajectories are computed and drawn.
  - HUD overlay is visible.
  - No exceptions are raised.

- [ ] **Step 3: Update documentation in `README.md`**
- Document the new 1-click `setup.bat`.
- Document the multi-category bounding boxes, motion trails, skeleton pose overlay, and HUD.

---
