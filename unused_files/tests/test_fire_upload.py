import os
os.environ['ADMIN_PASSWORD'] = ''
os.environ['SESSION_SECRET'] = 'test-secret-for-testing-only'
os.environ['DASHBOARD_PASSWORD'] = ''

import sys

import cv2
import numpy as np
from dataclasses import replace
from fastapi.testclient import TestClient

import app.main as main
from app import detectors


def test_missing_fire_model_on_real_upload(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'ultralytics', None)
    monkeypatch.setattr(main, 'settings', replace(main.settings, data_dir=tmp_path))
    monkeypatch.setattr(detectors.BaseDetector, '_load_model', lambda self: None)
    monkeypatch.setattr(detectors.AnomalyDetector, '__init__', lambda self: None)
    clip = tmp_path / 'sample.avi'
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 64))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    app = main.create_app(tmp_path / 'test.db', start_cameras=False, auto_camera=False)
    with TestClient(app) as client:
        with clip.open('rb') as stream:
            response = client.post('/api/test-videos', files={'file': ('sample.avi', stream)})
        assert response.status_code == 422, response.text
        assert 'No detection models are ready' in response.text
        assert not list((tmp_path / 'test_videos').glob('*annotated.mp4'))


def test_flame_label_and_frame_confirmation(monkeypatch):
    class Model:
        names = {0: 'Flame'}
    monkeypatch.setattr(detectors.BaseDetector, '_load_model', lambda self: setattr(self, 'model', Model()))
    detector = detectors.FireDetector()
    frame = np.zeros((64, 64, 3), np.uint8)
    result = detectors.DetectionResult([[1, 1, 50, 50]], [0], [.9], Model.names)
    assert detector.check_fire_hazard(frame, result)[0]['type'] == 'fire_watch'
    assert detector.check_fire_hazard(frame, result)[0]['type'] == 'fire_detected'
    Model.names = {0: 'person'}
    assert detectors.FireDetector().diagnostics()['status'] == 'unsupported_labels'


def test_fire_confirmation_uses_recorded_timestamps(monkeypatch):
    class Model:
        names = {0: 'fire'}
    monkeypatch.setattr(detectors.BaseDetector, '_load_model', lambda self: setattr(self, 'model', Model()))
    detector = detectors.FireDetector()
    frame = np.zeros((64, 64, 3), np.uint8)
    result = detectors.DetectionResult([[1, 1, 50, 50]], [0], [.9], Model.names)
    kinds = [detector.check_fire_hazard(frame, result, t)[0]['type'] for t in (0., 20., 21.)]
    assert kinds == ['fire_watch', 'fire_watch', 'fire_detected']
