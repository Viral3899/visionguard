# Connected Device & Phone Link Camera Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide seamless auto-detection and registration for all connected camera devices (including Microsoft Phone Link "Connected Camera", USB webcams, virtual cameras, and mobile Wi-Fi streams) into VisionGuard.

**Architecture:** 
- A multi-backend hardware probe in `camera_utils.py` combining Windows WMI PnP device discovery with OpenCV DirectShow (`CAP_DSHOW`) and Media Foundation (`CAP_MSMF`) capture verification.
- Enriched camera discovery API (`/api/cameras/discover`) returning friendly names, device classifications (Built-in, Phone Link / Mobile, USB / External, Virtual), and resolutions.
- Dedicated CLI and batch utility (`phone_camera_setup.py` + `CONNECT_PHONE_CAMERA_WINDOWS.bat`) for 1-click detection, preview verification, and automatic database registration.
- Complete documentation in `PHONE_CAMERA_GUIDE.md` and `README.md`.

**Tech Stack:** Python 3.12, OpenCV (DSHOW + MSMF), Windows WMI / PnP Entity enumeration, FastAPI, pytest.

---

### Task 1: Enhanced Camera & Connected Device Enumeration (`camera_utils.py`)

**Files:**
- Modify: `camera_utils.py`
- Test: `tests/test_camera_utils.py`

**Interfaces:**
- Consumes: `cv2.VideoCapture`, `win32com.client` (or fallback WMI query)
- Produces: `get_connected_camera_devices() -> list[dict]`, `detect_cameras(max_index=10) -> list[dict]`

- [ ] **Step 1: Write test for camera detection and device enrichment in `tests/test_camera_utils.py`**
- [ ] **Step 2: Run test to verify it fails before implementation**
- [ ] **Step 3: Implement WMI friendly-name lookup and dual-backend capture probing in `camera_utils.py`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit changes**

---

### Task 2: Discovery API Updates (`app/main.py`)

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `camera_utils.detect_cameras()`
- Produces: `/api/cameras/discover` endpoint with enhanced metadata (`is_phone_candidate`, `friendly_name`, `resolution`)

- [ ] **Step 1: Write test for enhanced `/api/cameras/discover` response in `tests/test_api.py`**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Update `/api/cameras/discover` in `app/main.py` to use enriched detection**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit changes**

---

### Task 3: Interactive Phone Camera Setup CLI & Batch Launcher

**Files:**
- Create: `phone_camera_setup.py`
- Create: `CONNECT_PHONE_CAMERA_WINDOWS.bat`

**Interfaces:**
- Consumes: `camera_utils.detect_cameras()`, `Database.create_camera()`
- Produces: Command-line wizard to auto-detect Phone Link cameras, test frames, and register into database

- [ ] **Step 1: Implement `phone_camera_setup.py` with automatic device detection and database registration**
- [ ] **Step 2: Create `CONNECT_PHONE_CAMERA_WINDOWS.bat` launcher**
- [ ] **Step 3: Test `phone_camera_setup.py --list` and verify execution**
- [ ] **Step 4: Commit changes**

---

### Task 4: Documentation & Guide

**Files:**
- Create: `PHONE_CAMERA_GUIDE.md`
- Modify: `README.md`

- [ ] **Step 1: Create `PHONE_CAMERA_GUIDE.md` with Windows 11 Phone Link steps and Wi-Fi RTSP fallback**
- [ ] **Step 2: Update `README.md` with phone camera quick-start instructions**
- [ ] **Step 3: Commit changes**

---

### Task 5: Full Regression Testing & Git Push

- [ ] **Step 1: Run full pytest suite across all tests**
- [ ] **Step 2: Verify zero test failures**
- [ ] **Step 3: Push changes to `https://github.com/Viral3899/visionguard.git` on `main` branch**
