import time
from types import SimpleNamespace

import cv2
import numpy as np

from app.alerts import AlertManager
from app.database import Database
from app.engine import CameraWorker, EventBus


class FakeTensor:
    def __init__(self, values): self.values = values
    def int(self): return self
    def cpu(self): return self
    def tolist(self): return self.values


class FakeModel:
    """Only a unit-test substitute for YOLO: no fake model in the application."""
    def __init__(self): self.call_count = 0
    def track(self, frame, **kwargs):
        assert kwargs['tracker']=='bytetrack.yaml'
        assert kwargs['persist'] is True
        self.call_count += 1
        h,w=frame.shape[:2]
        foot_y=.35 if self.call_count <= 3 else .65
        boxes=SimpleNamespace(id=FakeTensor([44]),
                              xyxy=FakeTensor([[w*.4,h*.12,w*.6,h*foot_y]]), cls=FakeTensor([0]))
        return [SimpleNamespace(boxes=boxes)]


def test_end_to_end_capture_analytics_and_snapshot(tmp_path, monkeypatch):
    file=tmp_path/'source.avi'
    writer=cv2.VideoWriter(str(file),cv2.VideoWriter_fourcc(*'MJPG'),15,(320,240))
    assert writer.isOpened()
    # Write more frames (200 frames = ~13 seconds at 15 fps) to avoid video end during test
    for _ in range(200): writer.write(np.zeros((240,320,3),np.uint8))
    writer.release()
    db=Database(tmp_path/'events.db'); db.init()
    from app.schemas import CameraCreate
    camera=db.create_camera(CameraCreate(name='Synthetic test',source=str(file)))
    full_camera=db.get_camera(camera['id'],safe=False)
    full_camera['zone']=[[.2,.2],[.8,.2],[.8,.9],[.2,.9]]
    full_camera['tripwire']=[[.2,.5],[.8,.5]]
    monkeypatch.setattr(CameraWorker,'_new_model',lambda self:FakeModel())
    # Disable anomaly detector for this test (uses YOLO-World which needs CLIP model)
    monkeypatch.setattr('app.engine.CameraWorker._init_anomaly_detector', lambda self: None)
    # Disable DetectionManager entirely for this test
    class MockDetectionManager:
        def process_frame(self, frame, person_tracks, camera_id, camera_config):
            return []
    monkeypatch.setattr('app.engine.DetectionManager', lambda: MockDetectionManager())
    from app.config import settings
    # Snapshot path is configured under the project data directory.
    alert_manager = AlertManager(db)
    worker=CameraWorker(full_camera,db,EventBus(), alert_manager); worker.start()
    try:
        until=time.monotonic()+10
        while time.monotonic()<until:
            types={e['kind'] for e in db.events(limit=20)}
            if 'intrusion' in types and 'entry' in types and worker.latest(): break
            time.sleep(.05)
        assert {'intrusion','entry'} <= {e['kind'] for e in db.events(limit=20)}
        assert worker.latest()[:3] == b'\xff\xd8\xff'
    finally:
        worker.stop()
    assert worker.status=='stopped'


def test_confidence_floor_discards_low_score_model_events():
    from app.engine import event_meets_confidence_floor
    assert event_meets_confidence_floor({'kind': 'fire_detected', 'confidence': 0.50})
    assert not event_meets_confidence_floor({'kind': 'fire_detected', 'confidence': 0.49})
    assert not event_meets_confidence_floor({'kind': 'fight', 'confidence': 'not-a-number'})
    assert event_meets_confidence_floor({'kind': 'intrusion'})  # analytics event has no model score
