import os
os.environ['HOST'] = '127.0.0.1'
os.environ['ADMIN_PASSWORD'] = 'admin123'
os.environ['SESSION_SECRET'] = 'test-secret-for-testing-only'
os.environ['DASHBOARD_PASSWORD'] = ''

from fastapi.testclient import TestClient
from app.main import create_app


def test_grid_and_ptz_html_elements(tmp_path):
    """Verify that index.html contains all multi-camera grid and PTZ elements."""
    app = create_app(tmp_path / 'test.db', start_cameras=False, auto_camera=False)
    with TestClient(app) as client:
        resp = client.get('/')
        assert resp.status_code == 200
        html = resp.text

        # Topbar & Home Button
        assert 'home-grid-btn' in html
        assert 'Home (Grid View)' in html
        assert 'LIVE CAMERA MATRIX' in html

        # Multi-camera Grid section
        assert 'multi-camera-grid-section' in html
        assert 'camera-matrix-grid' in html
        assert 'grid-layout-selector' in html
        assert 'grid-refresh-btn' in html

        # Focus View & PTZ HUD elements
        assert 'focus-back-btn' in html
        assert 'video-viewport' in html
        assert 'ptz-controls-hud' in html
        assert 'ptz-zoom-in' in html
        assert 'ptz-zoom-out' in html
        assert 'ptz-zoom-reset' in html
        assert 'ptz-zoom-badge' in html
        assert 'ptz-fullscreen' in html
        assert 'ptz-minimap-container' in html
        assert 'ptz-minimap-crop' in html


def test_grid_and_ptz_css_classes(tmp_path):
    """Verify that style.css contains styling for multi-camera grid and digital PTZ."""
    app = create_app(tmp_path / 'test.db', start_cameras=False, auto_camera=False)
    with TestClient(app) as client:
        resp = client.get('/static/style.css')
        assert resp.status_code == 200
        css = resp.text

        # Grid classes
        assert '.grid-panel' in css
        assert '#camera-matrix-grid' in css
        assert '.camera-tile' in css
        assert '.tile-top-bar' in css
        assert '.tile-bottom-bar' in css
        assert '.layout-auto' in css
        assert '.layout-2' in css

        # PTZ HUD classes
        assert '.video-viewport' in css
        assert '.ptz-controls-hud' in css
        assert '.ptz-btn' in css
        assert '.ptz-zoom-badge' in css
        assert '.ptz-minimap-container' in css
        assert '.ptz-minimap-crop' in css
        assert '.ptz-draggable' in css


def test_grid_and_ptz_js_logic(tmp_path):
    """Verify that app.js contains functions for grid view and PTZ controller."""
    app = create_app(tmp_path / 'test.db', start_cameras=False, auto_camera=False)
    with TestClient(app) as client:
        resp = client.get('/static/app.js')
        assert resp.status_code == 200
        js = resp.text

        # State & view mode
        assert "viewMode: 'grid'" in js
        assert "gridLayout: 'auto'" in js

        # Navigation & Grid functions
        assert 'function setViewMode(' in js
        assert 'function switchToGridView(' in js
        assert 'function switchToFocusView(' in js
        assert 'function renderCameraGrid(' in js

        # PTZ Controller
        assert 'const ptzController' in js
        assert 'zoomAt(' in js
        assert 'zoomIn()' in js
        assert 'zoomOut()' in js
        assert 'reset()' in js
        assert 'toggleFullscreen()' in js

        # Double-click triggers & home button
        assert 'ondblclick' in js
        assert 'home-grid-btn' in js


def test_multi_camera_creation_and_streaming_endpoints(tmp_path):
    """Verify multiple cameras can be created and queried for the grid view."""
    app = create_app(tmp_path / 'test.db', start_cameras=False, auto_camera=False)
    with TestClient(app) as client:
        # Authenticate first
        login_resp = client.post('/api/login', json={'username': 'admin', 'password': 'admin123'})
        assert login_resp.status_code == 200

        # Create 3 cameras
        cams = []
        for i in range(1, 4):
            r = client.post('/api/cameras', json={
                'name': f'Camera {i}',
                'source': str(i),
                'enabled': False
            })
            assert r.status_code == 201
            cams.append(r.json())

        # Check list returns all 3 cameras
        list_resp = client.get('/api/cameras')
        assert list_resp.status_code == 200
        cameras = list_resp.json()
        assert len(cameras) == 3

        # Stream endpoint exists for each camera
        for cam in cameras:
            cid = cam['id']
            # When disabled, stream returns 404
            stream_resp = client.get(f'/stream/{cid}')
            assert stream_resp.status_code == 404
