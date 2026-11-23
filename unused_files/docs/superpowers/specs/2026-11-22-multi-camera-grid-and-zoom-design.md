# Multi-Camera Grid View and Digital Zoom/Pan Design

**Date:** 2026-11-22  
**Topic:** Multi-camera surveillance matrix grid view, home navigation, double-click single-camera focus, and digital PTZ zoom/pan controls.

## 1. Intent & Why
In video surveillance and security operations (e.g. NVR systems, control rooms), operators need to simultaneously monitor all camera feeds in a unified grid wall. When an event or object of interest occurs on a particular camera, the operator needs to double-click to immediately open/focus on that camera, digitally zoom in and pan around to inspect fine details (e.g. faces, license plates, objects), and click a single "Home" button to return to the global grid overview.

## 2. Goals, Non-Goals, and Anti-Goals
- **Goals:**
  - Responsive multi-camera grid layout (Auto, 1x1, 2x2, 3x3, 4x4) showing live streams from all active cameras simultaneously.
  - Per-camera tile cards with live camera name, status indicator (online, reconnecting, offline), live detected people count, and AI detection badges.
  - Double-click on any camera card in the grid to immediately open it in Single-Camera Focused View.
  - Single-click or keyboard Home navigation (`⌂ Home (Grid)` button, breadcrumb, or `H`/`Esc` key) to return to the grid view.
  - Advanced Digital Zoom & Pan (Digital PTZ) for camera inspection:
    - Mouse scroll wheel zoom centered at cursor position.
    - Drag-to-pan when zoomed in (smooth grab hand cursor).
    - On-screen floating PTZ HUD: Zoom In (`+`), Zoom Out (`−`), Reset (`100%`), and Zoom level indicator (e.g., `100%` – `500%`).
    - Minimap / viewport indicator showing current zoom crop within the full frame.
    - Fullscreen toggle (`⛶`).
  - Seamless compatibility with existing zone drawing, tripwire tools, detection configuration, and alert popups.
- **Non-Goals:**
  - Hardware motorized mechanical PTZ protocols (Pelco-D, ONVIF PTZ motor commands) — this design provides high-precision digital software PTZ on the video feed.
- **Anti-Goals:**
  - Breaking existing zone/line editing capabilities or backend API endpoints.

## 3. Approaches Considered
- **Approach 1 (Selected): Integrated Multi-Mode Dashboard with Digital PTZ**
  - Seamless state switching between `grid` mode and `focus` mode inside the main dashboard.
  - Native CSS grid with dynamic column presets and auto-fit.
  - Dedicated Digital PTZ controller module with transform matrix (`scale`, `translate`), mouse wheel listener, pan gesture handling, and minimap HUD.
  - Direct Home button and double-click triggers.
- **Approach 2: Separate Video Wall Page/Tab**
  - Rejected because switching between different tabs creates UI friction and breaks context during incident investigation.
- **Approach 3: Modal Lightbox**
  - Rejected because lightboxes obscure surrounding telemetry and don't provide sufficient canvas for rule editing or side-by-side alerting.

## 4. Architecture & Interface Design
- **DOM / HTML Changes (`index.html`)**:
  - Add `home-grid-btn` in topbar and navigation.
  - Add `#multi-camera-grid-view` container with layout toolbar (presets `Auto`, `1×1`, `2×2`, `3×3`, `4×4`, snapshot all, refresh).
  - Add Digital PTZ HUD overlay on `#video-display` with zoom buttons (`+`, `−`, `100%`), zoom badge, minimap viewport, and fullscreen button.
- **Styling Changes (`style.css`)**:
  - Grid matrix styling with aspect ratio preservation (16:9), hover highlight, double-click cue, active status badge, and camera metrics overlay.
  - Digital PTZ container styles with `overflow: hidden`, cursor styling (`grab`, `grabbing`), and floating HUD controls.
  - Minimap HUD widget styles.
- **JavaScript Engine Changes (`app.js`)**:
  - Camera View Mode state: `state.viewMode = 'grid' | 'focus'`, `state.gridLayout = 'auto' | 1 | 2 | 3 | 4`.
  - Grid renderer: `renderCameraGrid()` creates/updates tiles for all cameras with live `/stream/<id>` feeds, status lights, and double-click handler.
  - Digital PTZ Controller:
    - State: `zoomLevel` (1.0 to 5.0), `panX`, `panY`, `isPanning`, `startX`, `startY`.
    - Handles `wheel`, `mousedown`, `mousemove`, `mouseup`, `mouseleave`, touch events.
    - Updates CSS transform `translate(${panX}px, ${panY}px) scale(${zoomLevel})` on `#video-feed` and `#draw-overlay`.
    - Updates minimap red viewport box according to current pan and zoom.
  - Double-click handler:
    - In Grid View: Double-clicking any camera tile calls `focusCamera(camId)`.
    - In Focus View: Double-clicking the video resets zoom to 100% (or toggles 2x zoom).
  - Home handler:
    - Clicking `⌂ Grid View` button, breadcrumb, or pressing `Esc`/`H` switches back to Grid View and resets PTZ zoom.
