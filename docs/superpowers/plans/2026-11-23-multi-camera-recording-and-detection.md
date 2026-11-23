# Multi-Camera Independent Recording & Daily Life Object Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide separate per-camera video recording controls with on-video timer overlays and camera-specific folders, expand object detection to daily life objects, and add an overlay display toggle where daily objects can be shown/hidden while alarming activity alerts are always visible.

**Architecture:** 
1. `VideoRecorder` in `app/recorder.py` is upgraded to write video clips into per-camera subfolders (`data/recordings/<camera_id>/`) and burn an on-frame recording timestamp and running elapsed timer onto each frame.
2. `ObjectDetector` and `DetectionManager` in `app/detectors.py` are expanded to detect all 80 COCO daily-life objects (electronics, bags, bottles, furniture, vehicles, animals, etc.).
3. `CameraWorker` in `app/engine.py` supports a `show_objects` toggle per camera: when enabled, daily objects and people are drawn; when disabled, everyday objects are hidden, but `draw_security_bounding_boxes` ALWAYS runs so alarming security threats (weapons, violence, fire, smoke, falls, intrusions) remain visible at all times.
4. The dashboard UI (`index.html`, `app.js`, `style.css`) adds independent record buttons with live timers on every camera grid tile and sidebar item, adds a "Show Objects" toggle button on the focus view and grid, and supports concurrent multi-camera recording.

**Tech Stack:** Python 3.12, FastAPI, OpenCV, Ultralytics YOLOv8, SQLite, HTML5, Vanilla JavaScript.

**Spec:** User prompt:
> "in here recording button is seperate for each camera having seperate recording button shoud be there and recording saving with timer in it in seperte folder or seperate timeing also more object detection objects need to check also it need to detect daily life objec in the detection camera you need to show button where detection need to show but alarming activity also need to show always"

## Global Constraints

- Must run on Windows 10/11 with Python 3.12.
- Backward compatibility: existing recordings in `data/recordings/` must still be playable and deletable without 404s.
- All existing 16 unit and integration tests in `tests/` must continue to pass.
- Alarming activity (weapons, violence, fire, smoke, falls, intrusions) must NEVER be hidden by any display toggle.

## Review Focus

1. Concurrent recording: starting recording on Camera A must not affect or reset recording on Camera B.
2. File resolution: `/api/recordings/{filename}` must resolve files located in camera subfolders without failing regex or path checks.
3. Timer rendering: on-frame recording timer must handle variable FPS and resolutions without crashing or overflowing frame boundaries.
4. Detection toggle persistence: toggling object visibility off hides everyday items (cups, chairs, backpacks) but leaves high-visibility alert boxes for weapons, fire, and violence untouched.
5. Performance: running daily-life object detection must maintain real-time frame rates and not block frame streaming.

---

### Task 1: Camera Subfolder Storage & Burned-In Recording Timer

**Files:**
- Modify: `app/recorder.py:22-195`
- Modify: `app/main.py:732-752`
- Test: `tests/test_recorder.py`
- Test: `tests/test_api_recording.py`

**Interfaces:**
- `VideoRecorder.start_recording(camera_id, camera_name, width, height, fps)` -> creates subfolder `output_dir / safe_cam_id`, instantiates `_RecordingSession` with destination `output_dir / safe_cam_id / filename`.
- `_RecordingSession._write_loop()` -> burns on-frame recording timer banner `● REC HH:MM:SS | YYYY-MM-DD HH:MM:SS | Camera Name` onto recorded frames before writing to `VideoWriter`.
- `GET /api/recordings/{filename}` -> finds file in subfolders or fallback to root recordings folder.

- [ ] **Step 1: Write failing test for camera subfolder and timer overlay in recorder**

In `tests/test_recorder.py`, add `test_recording_saved_in_camera_subfolder_with_timer`:
```python
def test_recording_saved_in_camera_subfolder_with_timer(temp_dir, temp_db):
    recorder = VideoRecorder(output_dir=temp_dir, db=temp_db)
    cam_id = 'cam_hallway'
    cam_name = 'Hallway East'
    rec_info = recorder.start_recording(cam_id, cam_name, width=320, height=240, fps=10.0)
    
    # Verify file path is in camera subfolder
    file_path = Path(rec_info['file_path'])
    assert file_path.parent.name == cam_id
    assert file_path.parent == temp_dir / cam_id

    # Write frames
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(12):
        recorder.write_frame(cam_id, frame)
        time.sleep(0.01)

    completed = recorder.stop_recording(cam_id)
    assert Path(completed['file_path']).is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_recorder.py::test_recording_saved_in_camera_subfolder_with_timer -v`
Expected: FAIL (assertion `file_path.parent.name == cam_id` fails because currently saved directly in root `temp_dir`).

- [ ] **Step 3: Implement camera subfolder and burned-in timer in `app/recorder.py`**

In `app/recorder.py`:
1. In `VideoRecorder.start_recording`:
   ```python
   cam_folder = self.output_dir / safe_cam_id
   cam_folder.mkdir(parents=True, exist_ok=True)
   file_path = cam_folder / filename
   ```
2. In `_RecordingSession._write_loop`:
   Before `self.writer.write(frame)`, burn the timer overlay:
   ```python
   elapsed_sec = max(0.0, time.monotonic() - self.start_monotonic)
   hrs = int(elapsed_sec // 3600)
   mins = int((elapsed_sec % 3600) // 60)
   secs = int(elapsed_sec % 60)
   now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
   timer_text = f"REC {hrs:02d}:{mins:02d}:{secs:02d} | {now_str} | {self.camera_name}"
   
   # Draw red circle indicator + text banner on top
   overlay_frame = frame.copy()
   cv2.circle(overlay_frame, (18, 20), 6, (0, 0, 255), -1)
   cv2.putText(overlay_frame, timer_text, (32, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
   self.writer.write(overlay_frame)
   ```
3. In `app/main.py`:
   Update `get_recording_video` and `delete_recording` to locate the file by querying `db.get_recording(filename)` first, and if found using its `file_path`. If not in DB, search `recordings_dir.glob(f"*/{filename}")` and `recordings_dir / filename`.

- [ ] **Step 4: Run recorder tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_recorder.py tests/test_api_recording.py -v`
Expected: PASS (all tests pass).

- [ ] **Step 5: Commit changes**

```bash
git add app/recorder.py app/main.py tests/test_recorder.py
git commit -m "feat: store camera recordings in dedicated subfolders with burned-in timer overlay"
```

---

### Task 2: Expand Object Detection to Daily Life Objects

**Files:**
- Modify: `app/config.py:124-131`
- Modify: `app/detectors.py:1061-1116, 1297-1307`
- Test: `tests/test_daily_objects.py` (new)

**Interfaces:**
- `settings.object_classes`: tuple of class IDs for daily life objects (or empty/all COCO classes).
- `ObjectDetector.detect(frame)`: detects daily life objects (bottles, phones, backpacks, chairs, vehicles, pets, etc.) and returns `DetectionResult`.
- `ObjectDetector.draw_daily_objects(frame, result)`: draws bounding boxes and clean labels in distinct colors (e.g. cyan/lime/light blue) for everyday items.

- [ ] **Step 1: Write test for daily life object detection and drawing**

Create `tests/test_daily_objects.py`:
```python
import numpy as np
import pytest
from app.detectors import ObjectDetector, DetectionResult

def test_draw_daily_objects():
    detector = ObjectDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_result = DetectionResult(
        boxes=[[50, 50, 150, 200], [200, 100, 350, 300]],
        classes=[67, 39],  # cell phone, bottle
        confidences=[0.85, 0.78],
        class_names={67: 'cell phone', 39: 'bottle'}
    )
    drawn = detector.draw_daily_objects(frame.copy(), mock_result)
    sub1 = drawn[50:200, 50:150]
    sub2 = drawn[100:300, 200:350]
    assert np.any(sub1 > 0), "Cell phone box was not drawn"
    assert np.any(sub2 > 0), "Bottle box was not drawn"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_daily_objects.py -v`
Expected: FAIL (`AttributeError: 'ObjectDetector' object has no attribute 'draw_daily_objects'`).

- [ ] **Step 3: Implement daily life object detection in `app/config.py` and `app/detectors.py`**

1. In `app/config.py`:
   Update default `OBJECT_CLASSES` in `settings`:
   Change default from `'43'` (knife only) to all standard daily-life COCO classes or `None`/empty tuple meaning detect all COCO classes:
   ```python
   # Daily life objects from COCO: vehicles, animals, accessories, tableware, food, furniture, electronics, etc.
   object_classes: tuple[int, ...] = tuple(
       int(n.strip()) for n in os.getenv('OBJECT_CLASSES', '').split(',') if n.strip()
   )
   ```
   When `object_classes` is empty, Ultralytics YOLO runs on all classes.
2. In `app/detectors.py`:
   In `ObjectDetector`:
   Add `draw_daily_objects(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray`:
   Draws rounded/compact bounding boxes with color `(235, 180, 30)` (cyan/teal/amber) and label `f"{name} {conf:.0%}"`, filtering out person (which has dedicated ByteTrack tracker) and weapons/fire (which have specialized detectors).
3. In `DetectionManager.process_frame`:
   Store `self.last_object_result` so `CameraWorker` can optionally draw it based on the camera's display toggle.

- [ ] **Step 4: Run daily objects test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_daily_objects.py -v`
Expected: PASS.

- [ ] **Step 5: Commit changes**

```bash
git add app/config.py app/detectors.py tests/test_daily_objects.py
git commit -m "feat: enable daily-life object detection for COCO categories with dedicated renderer"
```

---

### Task 3: Detection Display Toggle with Persistent Alarming Activity

**Files:**
- Modify: `app/database.py:72-82, 114-121`
- Modify: `app/schemas.py` & `app/main.py:75-84`
- Modify: `app/engine.py:200-254`
- Test: `tests/test_alarming_vs_daily_objects.py` (new)

**Interfaces:**
- `camera_config.show_objects`: integer flag (1 = show daily objects & person boxes, 0 = hide daily objects).
- `CameraConfigUpdate`: add `show_objects: bool | None = None`.
- `CameraWorker._run()`:
  - If `show_objects` is True: render person tracking boxes and daily life objects (`draw_daily_objects`).
  - If `show_objects` is False: do NOT draw person or daily object boxes.
  - **ALWAYS**: `draw_security_bounding_boxes(frame, all_events, persons=persons)` is executed regardless of `show_objects`! Weapons, violence, fire, smoke, intrusion, and falls are NEVER hidden.

- [ ] **Step 1: Write test verifying alarming activity is drawn even when `show_objects=False`**

Create `tests/test_alarming_vs_daily_objects.py`:
```python
import numpy as np
import pytest
from app.detectors import draw_security_bounding_boxes, ObjectDetector, DetectionResult

def test_alarming_activity_persists_when_daily_objects_disabled():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 1. Alarming event (e.g. weapon / violence)
    alarming_events = [{
        'type': 'weapon',
        'subtype': 'gun',
        'confidence': 0.92,
        'bbox': [0.2, 0.2, 0.4, 0.5]
    }]
    
    # 2. Daily life object (e.g. laptop / backpack)
    daily_result = DetectionResult(
        boxes=[[400, 300, 550, 450]],
        classes=[63],
        confidences=[0.88],
        class_names={63: 'laptop'}
    )
    
    detector = ObjectDetector()
    
    # Mode A: show_objects = False -> only alarming event drawn
    show_objects = False
    frame_no_objects = frame.copy()
    if show_objects:
        frame_no_objects = detector.draw_daily_objects(frame_no_objects, daily_result)
    frame_no_objects = draw_security_bounding_boxes(frame_no_objects, alarming_events)
    
    # Weapon region must be drawn
    weapon_region = frame_no_objects[int(0.2*480):int(0.5*480), int(0.2*640):int(0.4*640)]
    assert np.any(weapon_region > 0), "Weapon box must be drawn even when show_objects is False"
    # Laptop region must NOT be drawn
    laptop_region = frame_no_objects[300:450, 400:550]
    assert np.all(laptop_region == 0), "Laptop box must not be drawn when show_objects is False"

    # Mode B: show_objects = True -> both drawn
    frame_with_objects = frame.copy()
    frame_with_objects = detector.draw_daily_objects(frame_with_objects, daily_result)
    frame_with_objects = draw_security_bounding_boxes(frame_with_objects, alarming_events)
    assert np.any(frame_with_objects[int(0.2*480):int(0.5*480), int(0.2*640):int(0.4*640)] > 0)
    assert np.any(frame_with_objects[300:450, 400:550] > 0)
```

- [ ] **Step 2: Run test to verify logic**

Run: `.venv\Scripts\python.exe -m pytest tests/test_alarming_vs_daily_objects.py -v`
Expected: PASS.

- [ ] **Step 3: Add `show_objects` to database, API, and `CameraWorker`**

1. In `app/database.py`:
   Add `show_objects INTEGER NOT NULL DEFAULT 1` to `camera_config` table creation and migration.
2. In `app/main.py`:
   Add `show_objects: bool | None = None` to `CameraConfigUpdate`.
   Handle updating `show_objects` in `PATCH /api/cameras/{camera_id}/config`.
3. In `app/engine.py`:
   Check `show_objects = bool(self.camera_config.get('show_objects', 1))` in `CameraWorker._run()`.
   If `show_objects`:
   - Draw person tracking boxes.
   - Draw daily life objects from `self.detection_manager.last_object_result`.
   Always draw `draw_security_bounding_boxes(frame, all_events, persons=persons)` and HUD alert banner.

- [ ] **Step 4: Run full test suite to verify no regressions**

Run: `.venv\Scripts\python.exe -m pytest tests -v`
Expected: PASS.

- [ ] **Step 5: Commit changes**

```bash
git add app/database.py app/main.py app/engine.py tests/test_alarming_vs_daily_objects.py
git commit -m "feat: add show_objects toggle ensuring alarming activity is always displayed"
```

---

### Task 4: UI Controls — Per-Camera Recording Buttons, Timers & Detection Toggle

**Files:**
- Modify: `app/static/index.html`
- Modify: `app/static/style.css`
- Modify: `app/static/app.js`
- Test: `tests/test_ui_api_integration.py`

**Interfaces:**
- Every camera tile in the Multi-Camera Grid has:
  - An independent `⏺ Rec` / `⏹ Stop (00:15)` button.
  - Clicking this button calls `toggleRecording(cam.id)` without triggering grid focus view.
  - A running timer displayed on the active recording tile.
- Every camera has a `Show Objects` button in focus view and on each tile / toolbar:
  - Toggles `show_objects` setting on/off via `PATCH /api/cameras/{camera_id}/config`.
- Recordings table lists camera name, subfolder path/filename, duration, and download link.

- [ ] **Step 1: Write integration test for UI elements and API endpoint updates**

In `tests/test_ui_api_integration.py`, add checks for:
- `show_objects` in camera config endpoint.
- HTML elements for detection toggle button (`toggle-objects-btn`).
- JavaScript function `toggleCameraRecording` and `toggleObjectsDisplay`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ui_api_integration.py -v`
Expected: FAIL.

- [ ] **Step 3: Update `index.html`, `style.css`, and `app.js`**

1. In `app/static/index.html`:
   - Add `<button class="outline-btn small-btn" id="toggle-objects-btn" title="Toggle Daily Life Object Bounding Boxes (O)">👁 Objects: ON</button>` in the single camera focus action bar and grid toolbar.
   - Add status pill or badge for recording and object detection.
2. In `app/static/style.css`:
   - Add styles for `.tile-rec-btn` (red dot, pulsating recording indicator, timer font, hover effects).
   - Add styles for `.tile-objects-btn` and active detection indicator.
3. In `app/static/app.js`:
   - In `renderCameraGrid()`:
     Render the per-camera record button directly on each camera tile header:
     ```html
     <button class="tile-rec-btn ${cam.is_recording ? 'active-rec' : ''}" data-cam-id="${esc(cam.id)}">
       ${cam.is_recording ? `● REC <span class="tile-rec-timer" data-start="${cam.recording_start || ''}"></span>` : '⏺ Rec'}
     </button>
     ```
     Add click listener with `e.stopPropagation()` calling `toggleRecording(cam.id)`.
   - Implement `toggleObjectsDisplay(cameraId)`: calls `PATCH /api/cameras/{id}/config` with `{ show_objects: !current }`.
   - Update timers every second for all cameras currently recording.

- [ ] **Step 4: Run integration tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ui_api_integration.py -v`
Expected: PASS.

- [ ] **Step 5: Commit changes**

```bash
git add app/static/index.html app/static/style.css app/static/app.js tests/test_ui_api_integration.py
git commit -m "feat: add per-camera recording buttons with live timers and daily object toggle to UI"
```

---

### Task 5: End-to-End Verification & Documentation

**Files:**
- Create / Update: `pytest.ini` (to point `testpaths = tests`)
- Update: `README.md`
- Test: Full test suite

- [ ] **Step 1: Create `pytest.ini` to pin `testpaths = tests`**

Create `pytest.ini` with:
```ini
[pytest]
testpaths = tests
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
```

- [ ] **Step 2: Run full test suite across all modules**

Run: `.venv\Scripts\python.exe -m pytest -v`
Expected: All tests pass with 0 errors.

- [ ] **Step 3: Update documentation in `README.md`**

Document the new per-camera recording buttons, timer burned-in video output, camera subfolder organization, daily life object detection, and the detection toggle button with guaranteed alarming activity visibility.

- [ ] **Step 4: Commit all changes**

```bash
git add pytest.ini README.md
git commit -m "docs: document per-camera recording, timer overlay, daily objects, and alarm guarantee"
```
