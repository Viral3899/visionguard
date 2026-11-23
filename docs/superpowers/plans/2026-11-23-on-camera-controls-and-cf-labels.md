# On-Camera View Controls, CF Score Labels & Daily Classes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add on-video floating camera view buttons for recording start/stop and object detection toggle, format bounding box labels with object name and explicit "CF: XX%" (confidence score), support mobile, powerbank, chair and daily life classes, and provide automated test coverage.

**Architecture:**
1. In `app/detectors.py`: Update `draw_daily_objects()` to map class names (e.g. `cell phone` -> `MOBILE`, `chair` -> `CHAIR`, `powerbank` -> `POWERBANK`, etc.) and format the label as `f"{name} | CF: {conf:.0%}"` (or `CF: {conf:.2f}`) with bounding box.
2. In `app/static/index.html` & `app/static/style.css`: Add floating `.cam-view-overlay-controls` directly on `#video-viewport` containing `#cam-view-rec-btn` and `#cam-view-obj-btn` with sleek cyber/surveillance styling.
3. In `app/static/app.js`: Wire up `#cam-view-rec-btn` to `toggleRecording()` and `#cam-view-obj-btn` to `toggleObjectsDisplay()`, keeping them synced with active camera state and live timer.
4. In `tests/`: Add automated tests in `test_daily_objects.py` and `test_ui_api_integration.py` for on-camera view buttons, CF score format, and object classes (mobile, chair, powerbank).

## Tasks

### Task 1: Label Formatting with Object Name, CF Score, and Class Aliases
- In `app/detectors.py`, update `draw_daily_objects` to use `CLASS_DISPLAY_NAMES` mapping (e.g. `cell phone` -> `MOBILE`, `chair` -> `CHAIR`, `powerbank` -> `POWERBANK`) and render label as `f"{display_name} | CF: {conf:.0%}"` with high-contrast bounding boxes.
- Test in `tests/test_daily_objects.py`.

### Task 2: On-Camera View Floating Controls in UI
- In `app/static/index.html`: Add `#cam-view-controls` overlay inside `#video-viewport` containing `#cam-view-rec-btn` and `#cam-view-obj-btn`.
- In `app/static/style.css`: Style `.cam-view-controls` with semi-transparent backdrop, responsive positioning, glowing recording pulse, and cyan object toggle.
- In `app/static/app.js`: Synchronize on-camera view buttons in `renderSelected()`, tick live timers, and handle click events.
- Test in `tests/test_ui_api_integration.py`.

### Task 3: Comprehensive Verification
- Run full pytest test suite (all tests passing).
- Document new on-camera view controls and CF score display.
