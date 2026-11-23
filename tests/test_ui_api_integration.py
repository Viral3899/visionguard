from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import create_app


@pytest.fixture
def test_client(tmp_path):
    db_file = tmp_path / 'test_ui.db'
    recordings_dir = tmp_path / 'recordings'
    recordings_dir.mkdir(parents=True, exist_ok=True)

    orig_data_dir = settings.data_dir
    object.__setattr__(settings, 'data_dir', tmp_path)

    app = create_app(db_path=db_file, start_cameras=False, auto_camera=False)
    db = app.state.db
    db.init()

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with db.connection() as cx:
        cx.execute(
            'INSERT INTO cameras (id, name, source, zone, tripwire, enabled, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
            ('cam_ui_1', 'Main Entrance', '0', '[]', '[]', 1, now)
        )
        cx.execute('INSERT INTO totals (camera_id) VALUES (?)', ('cam_ui_1',))

    client = TestClient(app)
    client.post('/api/login', json={'username': 'admin', 'password': settings.admin_password or 'admin123'})

    try:
        yield client, app
    finally:
        object.__setattr__(settings, 'data_dir', orig_data_dir)


def test_ui_html_has_required_elements():
    html_path = Path('app/static/index.html')
    assert html_path.is_file()
    content = html_path.read_text(encoding='utf-8')

    assert 'id="record-camera-btn"' in content
    assert 'id="rec-badge"' in content
    assert 'id="stat-passers"' in content
    assert 'id="recordings-panel"' in content
    assert 'id="recordings-body"' in content
    assert 'id="recording-video-player"' in content
    assert 'id="toggle-objects-btn"' in content


def test_ui_js_has_recording_and_dedup_functions():
    js_path = Path('app/static/app.js')
    assert js_path.is_file()
    content = js_path.read_text(encoding='utf-8')

    assert 'toggleRecording' in content
    assert 'refreshRecordings' in content
    assert 'startRecordingTimer' in content
    assert 'stopRecordingTimer' in content
    assert 'unique_people_5m' in content
    assert 'toggleObjectsDisplay' in content


def test_camera_config_api_show_objects(test_client):
    client, app = test_client
    res = client.get('/api/cameras')
    cams = res.json()
    cam_id = cams[0]['id']

    # Patch show_objects to False
    res = client.patch(f'/api/cameras/{cam_id}/config', json={'show_objects': False})
    assert res.status_code == 200
    assert res.json()['show_objects'] == 0

    # Patch show_objects back to True
    res = client.patch(f'/api/cameras/{cam_id}/config', json={'show_objects': True})
    assert res.status_code == 200
    assert res.json()['show_objects'] == 1


def test_camera_telemetry_includes_recording_and_dedup(test_client):
    client, app = test_client

    # Get cameras
    res = client.get('/api/cameras')
    assert res.status_code == 200
    cams = res.json()
    assert len(cams) >= 1
    cam = cams[0]
    assert 'unique_people_5m' in cam
    assert 'is_recording' in cam
    assert cam['is_recording'] is False

    # Start recording
    res = client.post(f'/api/cameras/{cam["id"]}/recording/start')
    assert res.status_code == 200
    assert res.json()['is_recording'] is True

    # Camera telemetry now reports is_recording: True
    res = client.get('/api/cameras')
    assert res.status_code == 200
    cam_updated = [c for c in res.json() if c['id'] == cam['id']][0]
    assert cam_updated['is_recording'] is True

    # Stop recording
    res = client.post(f'/api/cameras/{cam["id"]}/recording/stop')
    assert res.status_code == 200

    # Camera telemetry now reports is_recording: False
    res = client.get('/api/cameras')
    assert res.status_code == 200
    cam_stopped = [c for c in res.json() if c['id'] == cam['id']][0]
    assert cam_stopped['is_recording'] is False
