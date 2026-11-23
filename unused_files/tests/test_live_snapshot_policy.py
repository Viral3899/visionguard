from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from app.database import Database
from app.engine import CameraWorker, EventBus
from app.schemas import CameraCreate
import app.engine as engine


def test_live_worker_only_saves_scored_incident_photos(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'settings', replace(engine.settings, data_dir=tmp_path))
    monkeypatch.setattr(engine, 'DetectionManager', lambda: SimpleNamespace())
    db = Database(tmp_path / 'events.db')
    db.init()
    new_camera = db.create_camera(CameraCreate(name='Snapshot policy', source='0', enabled=False))
    cam = db.get_camera(new_camera['id'], safe=False)
    received = []
    alerts = SimpleNamespace(process_event=lambda event, config: received.append(event))
    worker = CameraWorker(cam, db, EventBus(), alerts)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    worker._handle_event(frame, {'type': 'entry', 'track_id': 5, 'message': 'Normal entry'})
    worker._handle_event(frame, {'type': 'loitering', 'confidence': .91, 'message': 'Stationary person'})
    worker._handle_event(frame, {'type': 'weapon', 'subtype': 'knife', 'confidence': .49})
    assert not list((tmp_path / 'snapshots').glob('*.jpg'))
    worker._handle_event(frame, {
        'type': 'weapon', 'subtype': 'knife', 'confidence': .75,
        'bbox': [.15, .15, .65, .65], 'message': 'Knife model detection'
    })
    snapshots = list((tmp_path / 'snapshots').glob('*.jpg'))
    assert len(snapshots) == 1
    assert len(received) == 1
    assert received[0].metadata['confidence'] == .75
    assert {e['kind'] for e in db.events(limit=10)} == {'entry', 'weapon'}
    # Still the same incident identity: no additional snapshots.
    worker._handle_event(frame, {'type': 'weapon', 'subtype': 'knife', 'confidence': .93})
    assert len(list((tmp_path / 'snapshots').glob('*.jpg'))) == 1
