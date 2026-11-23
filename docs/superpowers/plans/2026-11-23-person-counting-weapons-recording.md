# Person Counting (5-Min Dedup), Weapon & Irregular Activity Bounding Boxes, and Video Recording Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 5-minute person deduplication counting, real-time bounding boxes for weapons and irregular activities on video feeds, manual start/stop video recording with storage and playback, and clean git commits.

**Architecture:** 
1. `app/person_tracker.py`: Appearance-based (HSV histogram) and Track-ID temporal deduplication window (300 seconds / 5 minutes) to count unique persons passing through without recounting the same person within 5 minutes.
2. `app/engine.py` & `app/detectors.py`: Render high-visibility bounding boxes for weapons (guns, knives) and irregular activity (violence, falls, loitering, unusual activity, restricted intrusions) directly onto live stream frames and recorded video.
3. `app/recorder.py`: Thread-safe `cv2.VideoWriter` recording engine for starting and stopping video captures per camera, saving to `data/recordings/` with complete metadata stored in SQLite.
4. `app/main.py` & Web UI (`index.html`, `app.js`, `style.css`): API endpoints and dashboard UI for recording start/stop, recording indicators, recordings list/playback, and real-time 5-minute passer counter stats.

**Tech Stack:** Python 3.12, OpenCV (`cv2.VideoWriter`, color histograms), Ultralytics YOLOv8, ByteTrack, FastAPI, SQLite, Vanilla JS/CSS.

**Spec:** User request: count persons / passers with 5-minute deduplication, render bounding boxes for weapons and irregular activity, start/stop video recording, and commit changes.

## Global Constraints
- Target platform: Windows 11 with Python 3.12, OpenCV 4.14, PyTorch with CUDA.
- Storage directory for recordings: `data/recordings/`.
- Person deduplication timeout: 300.0 seconds (5 minutes).
- Bounding box rendering must happen directly on the frame in `CameraWorker._run` before MJPEG encoding and before video recording writer.
- All code must run without external cloud dependencies; local processing only.
- Frequent, clean git commits.

## Review Focus
1. Rapid reappearance of the same person with a new ByteTrack ID within 5 minutes must match appearance histogram and NOT increment unique passer count.
2. Person leaving the frame and staying away for > 5 minutes (300 seconds) MUST increment unique passer count upon return.
3. Weapon and irregular activity bounding boxes must display proper colors, labels, and confidence percentages without flickering or obscuring essential views.
4. Video recording start/stop must be thread-safe, handle unexpected camera disconnection gracefully, and cleanly finalize the MP4 file header without corruption.
5. Deleting or playing recordings in UI must handle missing or corrupted files safely without 500 crashes.

---

### Task 1: Person Counting with 5-Minute Appearance & Track Deduplication

**Files:**
- Create: `app/person_tracker.py`
- Test: `tests/test_person_tracker.py`
- Modify: `app/analytics.py`, `app/engine.py`

**Interfaces:**
- Produces: `PersonDeduplicator` class in `app/person_tracker.py`:
  - `update(frame: np.ndarray, persons: list[dict], now: float = None) -> tuple[int, int]` (returns `(current_in_frame, unique_passed_5m)`)
  - `get_counts() -> dict[str, int]` (`{'in_frame': X, 'unique_5m': Y}`)
  - `reset()`

- [ ] **Step 1: Write the failing unit tests for PersonDeduplicator**
Create `tests/test_person_tracker.py` testing:
- Single person stays in frame for 10 frames -> count is 1.
- Person leaves and re-enters 2 minutes later (new track ID, same appearance) -> count remains 1.
- Second distinct person enters -> count becomes 2.
- Person re-enters after 301 seconds (> 5 minutes) -> count becomes 3.

- [ ] **Step 2: Run test to verify it fails**
Run: `.venv\Scripts\python -m pytest tests/test_person_tracker.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `PersonDeduplicator` in `app/person_tracker.py`**
Implement:
- Color histogram extraction: HSV 2-part upper/lower torso histogram from person bbox.
- Track ID cache + appearance comparison with correlation threshold (> 0.82).
- 300.0s expiration window for deduplication registry.
- Tracking of `unique_5m` total and `current_in_frame`.

- [ ] **Step 4: Integrate PersonDeduplicator into `app/engine.py`**
- Initialize `self.person_deduplicator = PersonDeduplicator(cooldown_seconds=300.0)` in `CameraWorker`.
- Call `in_frame, unique_5m = self.person_deduplicator.update(frame, persons, now=time.monotonic())`.
- Update telemetry to include `unique_people_5m`.
- Render HUD badge on frame: `PEOPLE: {in_frame} | 5M UNIQUE: {unique_5m}`.

- [ ] **Step 5: Run tests to verify they pass**
Run: `.venv\Scripts\python -m pytest tests/test_person_tracker.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**
```bash
git add app/person_tracker.py app/engine.py tests/test_person_tracker.py
git commit -m "feat(analytics): add person counter with 5-minute appearance deduplication"
```

---

### Task 2: Real-time Bounding Box Rendering for Weapons & Irregular Activities

**Files:**
- Modify: `app/detectors.py`, `app/engine.py`, `app/incidents.py`
- Test: `tests/test_bounding_boxes.py`

**Interfaces:**
- Produces: `draw_security_overlays(frame: np.ndarray, events: list[dict], conf_floor: float = 0.35) -> np.ndarray`
  - High-visibility Neon Red / Magenta bounding box for weapons (`gun`, `knife`, `blade`, `pistol`, `rifle`).
  - High-visibility Orange / Yellow / Red bounding box for irregular activities (`violence`, `fight`, `fall_detected`, `loitering`, `unusual_activity`, `intrusion`).
  - Clear label tag containing event name and confidence percentage.
  - Collision-avoidant text badge with background fill.

- [ ] **Step 1: Write unit tests for security overlay rendering**
Create `tests/test_bounding_boxes.py`:
- Test weapon event draws rectangle with red/magenta color and label `WEAPON: KNIFE 85%`.
- Test irregular activity event (violence, fall, loitering) draws bounding box with expected label and color.
- Test handles unnormalized and normalized bboxes safely.

- [ ] **Step 2: Run test to verify it fails**
Run: `.venv\Scripts\python -m pytest tests/test_bounding_boxes.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `draw_security_overlays` in `app/detectors.py` and call in `app/engine.py`**
- In `app/detectors.py` or `app/incidents.py`, add `draw_security_overlays(frame, events)`.
- In `CameraWorker._run` in `app/engine.py`:
  - Collect weapon detections from `raw_detections` and `all_events`.
  - Collect irregular activity detections (violence, falls, loitering, unusual activity, intrusions).
  - Call `draw_security_overlays(frame, active_threat_events)` directly on `frame` so it appears on live MJPEG stream and in recordings.

- [ ] **Step 4: Run tests to verify they pass**
Run: `.venv\Scripts\python -m pytest tests/test_bounding_boxes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add app/detectors.py app/engine.py app/incidents.py tests/test_bounding_boxes.py
git commit -m "feat(vision): render bounding boxes for weapons and irregular activities on live stream"
```

---

### Task 3: Video Recording Backend Engine & Database Persistence

**Files:**
- Create: `app/recorder.py`
- Modify: `app/database.py`, `app/engine.py`, `app/main.py`
- Test: `tests/test_recorder.py`

**Interfaces:**
- Produces:
  - `VideoRecorder`:
    - `start_recording(camera_id: str, camera_name: str, width: int, height: int, fps: float) -> str` (returns `recording_id`)
    - `write_frame(camera_id: str, frame: np.ndarray)`
    - `stop_recording(camera_id: str) -> dict | None`
    - `get_status(camera_id: str) -> dict`
    - `is_recording(camera_id: str) -> bool`
  - Database schema: `recordings` table in `app/database.py`:
    - `(id TEXT PRIMARY KEY, camera_id TEXT, camera_name TEXT, filename TEXT, file_path TEXT, start_time TEXT, end_time TEXT, duration_seconds REAL, file_size_bytes INTEGER, created_at TEXT)`
  - API endpoints:
    - `POST /api/cameras/{cam_id}/recording/start`
    - `POST /api/cameras/{cam_id}/recording/stop`
    - `GET /api/cameras/{cam_id}/recording/status`
    - `GET /api/recordings`
    - `GET /api/recordings/{filename}` (stream/download)
    - `DELETE /api/recordings/{filename}`

- [ ] **Step 1: Write unit tests for VideoRecorder and Database recording records**
Create `tests/test_recorder.py`:
- Test starting a recording creates video file in `data/recordings/`.
- Test writing 10 frames and stopping completes an openable MP4 video with valid duration and size.
- Test recording status reporting (`is_recording`, `duration_seconds`).
- Test database insertion and retrieval of recording record.

- [ ] **Step 2: Run test to verify it fails**
Run: `.venv\Scripts\python -m pytest tests/test_recorder.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `app/recorder.py` and update `app/database.py`**
- Implement queue-based asynchronous `VideoRecorder` using OpenCV `cv2.VideoWriter(..., cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))`.
- Add `recordings` table and methods (`store_recording`, `list_recordings`, `get_recording`, `delete_recording`) in `app/database.py`.
- Hook `recorder.write_frame(self.camera['id'], frame)` into `CameraWorker._run` right after overlays are drawn.

- [ ] **Step 4: Implement API endpoints in `app/main.py`**
- Add start, stop, status endpoints for camera recording.
- Add list, download, and delete endpoints for stored recordings.

- [ ] **Step 5: Run tests to verify they pass**
Run: `.venv\Scripts\python -m pytest tests/test_recorder.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**
```bash
git add app/recorder.py app/database.py app/engine.py app/main.py tests/test_recorder.py
git commit -m "feat(recorder): add video recording engine with start/stop APIs and database persistence"
```

---

### Task 4: Frontend UI for Start/Stop Recording and 5-Minute Person Counter

**Files:**
- Modify: `app/static/index.html`, `app/static/app.js`, `app/static/style.css`

**Interfaces:**
- UI components:
  - Header actions & toolbar: **"⏺ Start Recording"** / **"⏹ Stop Recording"** button.
  - Recording HUD badge on video: `● REC 00:14`.
  - Stats card: `PEOPLE IN VIEW: X` and `5-MIN UNIQUE PASSERS: Y`.
  - Grid View tiles: Recording indicator badge on recording camera tiles.
  - Video Recordings management section / modal to list, watch, download, and delete recordings.

- [ ] **Step 1: Add HTML elements in `index.html`**
- Add Record button to the video player header and tool actions.
- Add REC indicator badge to `video-viewport`.
- Add stat card or counter for 5-minute unique passers.
- Add "📹 Recordings" tab or section to the dashboard with recordings table and modal video player.

- [ ] **Step 2: Add styles in `style.css`**
- Add styles for active recording button (glowing red pulse).
- Add styles for `.rec-badge` with pulsing red dot.
- Add styles for recordings list table and preview modal.

- [ ] **Step 3: Implement client logic in `app.js`**
- Implement `toggleRecording(camId)` calling start/stop endpoints.
- Poll recording status and update recording timer (`00:00`).
- Update stats and camera telemetry display to show `5m unique passers`.
- Implement `loadRecordings()`, `deleteRecording()`, and `playRecordingModal()`.

- [ ] **Step 4: Verify UI in browser / automated test**
Create `tests/test_ui_api_integration.py` to verify all new endpoints return valid responses.

- [ ] **Step 5: Commit**
```bash
git add app/static/index.html app/static/app.js app/static/style.css tests/test_ui_api_integration.py
git commit -m "feat(ui): add video recording controls, recording indicator, and 5-minute unique passer stats"
```

---

### Task 5: End-to-End Verification & Final Git Commit

**Files:**
- Verify all modified files: `app/person_tracker.py`, `app/detectors.py`, `app/engine.py`, `app/recorder.py`, `app/database.py`, `app/main.py`, `app/static/index.html`, `app/static/app.js`, `app/static/style.css`.

- [ ] **Step 1: Run complete test suite**
Run: `.venv\Scripts\python -m pytest tests/ -v`
Expected: ALL PASS.

- [ ] **Step 2: Start the application in a test run and verify healthy startup**
Check `/api/cameras`, `/api/stats`, `/api/recordings` all respond with HTTP 200.

- [ ] **Step 3: Final git commit and status check**
Review `git status` and `git log` to ensure all changes are committed cleanly.
