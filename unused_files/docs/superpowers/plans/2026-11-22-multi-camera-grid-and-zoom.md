# Multi-Camera Grid View and Digital PTZ Zoom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a comprehensive multi-camera matrix grid view with an instant Home button, double-click camera focus, and advanced digital PTZ zoom/pan controls for live video surveillance.

**Architecture:** Extend VisionGuard's web dashboard with an adaptive multi-camera grid layout that feeds simultaneous live MJPEG streams, an event-driven view mode switcher (`grid` vs `focus`), and a high-performance Digital PTZ controller that handles smooth scroll-wheel zoom, drag-to-pan, on-screen zoom HUD, and a real-time minimap crop indicator.

**Tech Stack:** HTML5, CSS Grid / Flexbox / Transforms, Vanilla JavaScript (ES6+), FastAPI backend.

**Spec:** [docs/superpowers/specs/2026-11-22-multi-camera-grid-and-zoom-design.md](file:///E:/Visionmaster/docs/superpowers/specs/2026-11-22-multi-camera-grid-and-zoom-design.md)

## Global Constraints
- Must not break existing single camera preview, detection rule drawing (zone/line), or alert mechanisms.
- Must support arbitrary number of cameras (0, 1, 2, 4, 9, 16) with responsive grid presets (Auto, 1x1, 2x2, 3x3, 4x4).
- Digital zoom must scale between 1.0x (100%) and 5.0x (500%) with boundary clamping on pan translation to prevent zooming into empty space.
- All actions must have keyboard shortcuts (`H`/`Esc` for Home/Grid, `+`/`-` for Zoom, `0` for Zoom Reset, `F` for Fullscreen).

## Review Focus
1. Switching between grid and focus mode without memory leaks or stale MJPEG stream connections.
2. Digital PTZ zoom centering at the exact mouse cursor coordinates during mouse-wheel interaction.
3. Clamping pan boundaries so video never pans out of view at any zoom factor.
4. Preserving drawing tools when in focus mode at 1.0x zoom and disabling pan during active zone/tripwire drawing mode.
5. Graceful fallback for cameras with errors, offline state, or zero configured cameras.

---

### Task 1: DOM Structure for Grid View, Home Navigation, and PTZ HUD

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes: Existing `#overview` section in `index.html`.
- Produces: 
  - `#home-grid-btn` and `#focus-back-btn` for Home navigation.
  - `#multi-camera-grid-section` containing grid controls bar (`#grid-layout-select`, `#grid-refresh-btn`) and `#camera-matrix-grid`.
  - `#ptz-controls-hud` containing `#ptz-zoom-in`, `#ptz-zoom-out`, `#ptz-zoom-reset`, `#ptz-zoom-level`, `#ptz-fullscreen`, and `#ptz-minimap`.

- [ ] **Step 1: Add Home button to topbar and sidebar navigation**
Add `⌂ Home (Grid)` button in the topbar next to the breadcrumbs and in the sidebar navigation.

- [ ] **Step 2: Add Multi-Camera Grid container section**
Add `#multi-camera-grid-section` directly above `#workspace-grid` with a dedicated header toolbar containing layout presets (Auto, 1×1, 2×2, 3×3, 4×4), active camera badge, and `#camera-matrix-grid`.

- [ ] **Step 3: Add Digital PTZ Floating HUD in player panel**
Add `#ptz-controls-hud` inside `#video-display` containing zoom buttons (`+`, `−`, `100%`), zoom level indicator, fullscreen toggle, and minimap viewport indicator `#ptz-minimap`.

- [ ] **Step 4: Verify HTML syntax and DOM structure**
Run validation command to ensure no unclosed tags or syntax errors in `index.html`.

- [ ] **Step 5: Commit**
```bash
git add app/static/index.html
git commit -m "feat(ui): add grid view, home button, and PTZ HUD to index.html"
```

---

### Task 2: CSS Styles for Grid Matrix, Camera Tiles, and PTZ Controls

**Files:**
- Modify: `app/static/style.css`

**Interfaces:**
- Consumes: DOM elements from Task 1.
- Produces: Clean, modern styles for `.camera-matrix-grid`, `.camera-tile`, `.tile-overlay`, `.ptz-controls-hud`, and `.ptz-minimap`.

- [ ] **Step 1: Add Grid Layout & Tile styles**
Add responsive CSS grid rules for `#camera-matrix-grid` with `.layout-auto`, `.layout-1`, `.layout-2`, `.layout-3`, `.layout-4`. Add styles for `.camera-tile` (16:9 aspect ratio, dark frame, hover glow, double-click badge).

- [ ] **Step 2: Add Tile Overlays and Indicators**
Add `.tile-top-bar` (camera name, status dot, people count) and `.tile-bottom-bar` (AI badges, double-click prompt).

- [ ] **Step 3: Add Digital PTZ HUD and Minimap styles**
Add `.ptz-controls-hud` floating bar at top-left/top-right of `#video-display`, `.ptz-btn`, `.ptz-zoom-badge`, and `#ptz-minimap` widget with dynamic red crop rectangle.

- [ ] **Step 4: Add Panning & Cursor styles**
Add `.ptz-draggable` cursor grab / grabbing states and transform transitions.

- [ ] **Step 5: Verify CSS styles**
Check CSS rules for conflicts or rendering bugs.

- [ ] **Step 6: Commit**
```bash
git add app/static/style.css
git commit -m "feat(ui): add grid matrix and PTZ HUD styling in style.css"
```

---

### Task 3: Multi-Camera Grid Engine, Double-Click Focus, and Home Navigation

**Files:**
- Modify: `app/static/app.js`

**Interfaces:**
- Consumes: `state.cameras`, `/stream/{camera_id}`.
- Produces:
  - `state.viewMode = 'grid' | 'focus'`
  - `state.gridLayout = 'auto' | 1 | 2 | 3 | 4`
  - `switchToGridView()`
  - `switchToFocusView(cameraId)`
  - `renderCameraGrid()`

- [ ] **Step 1: Add View Mode & Grid Layout state variables**
Define `state.viewMode = 'grid'`, `state.gridLayout = 'auto'`, and event listeners for Home buttons (`#home-grid-btn`, `#focus-back-btn`).

- [ ] **Step 2: Implement `renderCameraGrid()`**
Build dynamic grid generation that instantiates `<div class="camera-tile">` for each configured camera, loading `/stream/{cam.id}`, status light, metrics, and empty fallback state.

- [ ] **Step 3: Implement Double-Click on Camera Tile**
Attach `ondblclick` event to each `.camera-tile` that calls `switchToFocusView(cam.id)`. Also attach single-click to select camera without switching view.

- [ ] **Step 4: Implement Home Navigation and Breadcrumb updates**
Implement `switchToGridView()` which hides `#workspace-grid`, shows `#multi-camera-grid-section`, updates breadcrumbs to `OPERATIONS / MULTI-CAMERA GRID`, and updates sidebar active links.

- [ ] **Step 5: Verify Grid and Switch logic**
Verify switching from Grid to Focus on double-click and Focus back to Grid on Home button.

- [ ] **Step 6: Commit**
```bash
git add app/static/app.js
git commit -m "feat(ui): implement multi-camera grid rendering and navigation in app.js"
```

---

### Task 4: Advanced Digital PTZ Controller (Zoom In, Zoom Out, Pan, Minimap, Shortcuts)

**Files:**
- Modify: `app/static/app.js`

**Interfaces:**
- Consumes: `#video-feed`, `#video-display`, `#draw-overlay`.
- Produces:
  - `ptzController`: `{ zoomIn(), zoomOut(), reset(), setZoom(scale, centerX, centerY), pan(dx, dy) }`
  - Minimap viewport updater.
  - Keyboard shortcuts (`H`, `Esc`, `+`, `-`, `0`, `F`).

- [ ] **Step 1: Implement Digital PTZ State and Math Engine**
Create `ptzController` managing `scale` (1.0x to 5.0x), `translateX`, `translateY`, and clamping to ensure the video frame always fills the viewport without displaying blank margins.

- [ ] **Step 2: Implement Mouse Wheel Zoom at Cursor Position**
Attach `wheel` listener on `#video-display` (when not in drawing mode) computing zoom focal point at mouse (x, y) coordinates.

- [ ] **Step 3: Implement Click-and-Drag Panning**
Attach `mousedown`, `mousemove`, `mouseup`, `mouseleave` listeners to allow fluid drag-to-pan when `scale > 1.0`.

- [ ] **Step 4: Wire HUD buttons and Minimap Viewport**
Connect `#ptz-zoom-in`, `#ptz-zoom-out`, `#ptz-zoom-reset`, `#ptz-fullscreen` to `ptzController`. Update `#ptz-minimap` crop box on every transform change.

- [ ] **Step 5: Add Keyboard Shortcuts and Double-Click Zoom Reset**
Add `keydown` listeners: `H`/`Escape` -> Home/Grid view; `+`/`=` -> Zoom in; `-` -> Zoom out; `0` -> Reset zoom; `F` -> Fullscreen. Double-clicking on the zoomed video resets zoom to 100%.

- [ ] **Step 6: Commit**
```bash
git add app/static/app.js
git commit -m "feat(ui): add digital PTZ zoom, pan, and minimap controller in app.js"
```

---

### Task 5: End-to-End Verification and Integration Testing

**Files:**
- Create: `tests/test_grid_and_ptz.py`

**Interfaces:**
- Consumes: FastAPI test client, endpoints `/`, `/api/cameras`, `/stream/{camera_id}`, `/static/index.html`, `/static/app.js`, `/static/style.css`.
- Produces: Test verification results.

- [ ] **Step 1: Write integration tests**
Create `tests/test_grid_and_ptz.py` testing static asset delivery, presence of grid elements, PTZ HUD, camera endpoints, and JavaScript functions.

- [ ] **Step 2: Run pytest suite**
Execute `pytest tests/test_grid_and_ptz.py -v`.

- [ ] **Step 3: Verify live dashboard launch**
Verify that the FastAPI server loads without error and all endpoints respond with HTTP 200.

- [ ] **Step 4: Commit**
```bash
git add tests/test_grid_and_ptz.py
git commit -m "test: add verification tests for multi-camera grid and PTZ features"
```
