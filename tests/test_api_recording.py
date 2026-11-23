import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from app.database import Database
from app.main import create_app
from app.config import settings


@pytest.fixture
def test_app(tmp_path):
    db_file = tmp_path / 'test_app.db'
    recordings_dir = tmp_path / 'recordings'
    recordings_dir.mkdir(parents=True, exist_ok=True)
    
    orig_data_dir = settings.data_dir
    object.__setattr__(settings, 'data_dir', tmp_path)

    app = create_app(db_path=db_file, start_cameras=False, auto_camera=False)
    db = app.state.db
    db.init()

    # Insert a test camera
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with db.connection() as cx:
        cx.execute(
            'INSERT INTO cameras (id, name, source, zone, tripwire, enabled, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
            ('cam_rec_test', 'Test Camera 1', '0', '[]', '[]', 1, now)
        )
        cx.execute('INSERT INTO totals (camera_id) VALUES (?)', ('cam_rec_test',))

    client = TestClient(app)
    # Log in as admin
    client.post('/api/login', json={'username': 'admin', 'password': settings.admin_password or 'admin123'})

    try:
        yield client, app, tmp_path
    finally:
        object.__setattr__(settings, 'data_dir', orig_data_dir)


def test_api_stats_includes_5m_passers(test_app):
    client, app, _ = test_app
    res = client.get('/api/stats')
    assert res.status_code == 200
    data = res.json()
    assert 'unique_people_5m' in data
    assert 'current_people' in data
    assert 'active_recordings' in data


def test_api_recording_lifecycle(test_app):
    client, app, tmp_path = test_app
    cam_id = 'cam_rec_test'

    # Check initial status
    res = client.get(f'/api/cameras/{cam_id}/recording/status')
    assert res.status_code == 200
    assert res.json()['is_recording'] is False

    # Start recording
    res = client.post(f'/api/cameras/{cam_id}/recording/start')
    assert res.status_code == 200
    start_data = res.json()
    assert start_data['is_recording'] is True
    filename = start_data['filename']

    # Check status during recording
    res = client.get(f'/api/cameras/{cam_id}/recording/status')
    assert res.status_code == 200
    assert res.json()['is_recording'] is True

    # Write a frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    app.state.recorder.write_frame(cam_id, frame)

    # Stop recording
    res = client.post(f'/api/cameras/{cam_id}/recording/stop')
    assert res.status_code == 200
    stop_data = res.json()
    assert stop_data['filename'] == filename

    # List recordings
    res = client.get('/api/recordings')
    assert res.status_code == 200
    recs = res.json()
    assert len(recs) >= 1
    assert any(r['filename'] == filename for r in recs)

    # Download recording video
    res = client.get(f'/api/recordings/{filename}')
    assert res.status_code == 200
    assert res.headers['content-type'] == 'video/mp4'

    # Delete recording
    res = client.delete(f'/api/recordings/{filename}')
    assert res.status_code == 200
    assert res.json()['ok'] is True

    # Verify deleted
    res = client.get('/api/recordings')
    assert not any(r['filename'] == filename for r in res.json())
