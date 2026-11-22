import os
os.environ['ADMIN_PASSWORD'] = ''
os.environ['SESSION_SECRET'] = 'test-secret-for-testing-only'
os.environ['DASHBOARD_PASSWORD'] = ''

import sys
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

import app.main as main
from app import detectors
from app.incidents import IncidentTracker, eligible_event, draw_event_box


def threat(kind, confidence, **extras):
    return {'type': kind, 'confidence': confidence, 'bbox': [.2, .2, .7, .7], **extras}


def test_no_normal_frames_no_loitering_no_confidence_boost():
    gate = IncidentTracker()
    assert not eligible_event(threat('loitering', .99))
    assert not eligible_event(threat('proximity_alert', .99))
    assert not eligible_event(threat('fire_watch', .99))
    assert not eligible_event(threat('weapon', .499))
    assert gate.update([threat('weapon', .499)], 0) == []
    assert gate.update([threat('weapon', .499)], .1) == []
    assert gate.update([threat('weapon', .50)], .2) == []
    assert len(gate.update([threat('weapon', .50)], .3)) == 1
    assert gate.update([threat('weapon', .99)], .4) == []
    assert gate.update([], 3) == []
    assert gate.update([], 3.01) == []
    assert gate.update([], 3.02) == []
    assert gate.update([threat('weapon', .75)], 3.1) == []
    assert len(gate.update([threat('weapon', .75)], 3.2)) == 1


def test_fire_escalation_keeps_photo_count_small():
    gate = IncidentTracker()
    assert gate.update([threat('smoke', .6)], 0) == []
    assert [e['type'] for e in gate.update([threat('smoke', .6)], .1)] == ['smoke']
    assert gate.update([threat('smoke', .6)], .2) == []
    assert [e['type'] for e in gate.update([threat('fire_detected', .8)], .3)] == ['fire_detected']
    assert gate.update([threat('fire_detected', .9)], .4) == []


def test_draw_bbox_creates_annotated_snapshot():
    frame = np.zeros((100, 100, 3), np.uint8)
    marked = draw_event_box(frame.copy(), threat('weapon', .76))
    assert np.any(marked[20:25, 20:70] != 0)
    assert not np.any(frame)


def test_upload_keeps_one_true_event_snapshot_and_secure_link(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'ultralytics', None)
    monkeypatch.setattr(main, 'settings', replace(main.settings, data_dir=tmp_path,
                                                 frame_skip=1, incident_confirmation_frames=2))

    class FakeDetectorManager:
        def __init__(self):
            self.fire_detector = SimpleNamespace(status='ready', model=object(),
                                                  diagnostics=lambda: {'status': 'ready'})
            self.violence_detector = SimpleNamespace(model=None)
            self.object_detector = SimpleNamespace(model=None)
            self.anomaly_detector = SimpleNamespace(model=None)
            self.object_every_n_frames = 1
            self.violence_every_n_frames = 1
            self.anomaly_every_n_frames = 1
            self.frame_number = 0
        def process_frame(self, frame, tracks, camera_id, config, timestamp=None):
            self.frame_number += 1
            # Normal activity first, then a real-model-like score for several
            # successive frames. No installation of YOLO is needed for this test.
            if self.frame_number <= 2:
                return [{'type': 'loitering', 'confidence': .99}]
            if self.frame_number <= 6:
                return [threat('fire_detected', .81, message='Test flame')]
            return []

    monkeypatch.setattr(detectors, 'DetectionManager', FakeDetectorManager)
    clip = tmp_path / 'sample.avi'
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'MJPG'), 10, (96, 96))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(np.zeros((96, 96, 3), np.uint8))
    writer.release()

    app = main.create_app(tmp_path / 'events.db', start_cameras=False, auto_camera=False)
    with TestClient(app) as client:
        with clip.open('rb') as stream:
            resp = client.post('/api/test-videos', files={'file': ('sample.avi', stream)})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data['frames_read'] == 8
        assert data['minimum_confidence'] >= .5
        assert data['confirmed_alert_types'] == ['fire_detected']
        assert len(data['confirmed_events']) == 1
        photo = data['confirmed_events'][0]['snapshot_url']
        assert photo and client.get(photo).status_code == 200
        assert client.get(photo).headers['content-type'] == 'image/jpeg'
        assert client.get(data['annotated_video_url']).status_code == 200
        assert client.get(photo.replace('jpg', 'png')).status_code == 404
        # No routine/loitering snapshots were ever written.
        assert len(list((tmp_path / 'test_videos').glob('*.jpg'))) == 1


def test_smoke_recurrence_cannot_inflate_raw_model_confidence(monkeypatch):
    class Model:
        names = {0: 'smoke'}
    monkeypatch.setattr(detectors.BaseDetector, '_load_model',
                        lambda self: setattr(self, 'model', Model()))
    fire = detectors.FireDetector()
    frame = np.zeros((100, 100, 3), np.uint8)
    detection = detectors.DetectionResult([[10, 10, 80, 80]], [0], [.47], {0: 'smoke'})
    assert fire.check_fire_hazard(frame, detection, 0)[0]['confidence'] == .47
    later = fire.check_fire_hazard(frame, detection, .1)[0]
    assert later['confidence'] == .47
    assert not eligible_event(later)
