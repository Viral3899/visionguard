"""Camera capture workers, per-camera YOLOv8+ByteTrack, MJPEG, alert bus."""
import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .alerts import AlertEvent, AlertManager
from .analytics import Analytics
from .config import settings
from .database import Database
from .detectors import DetectionManager

def event_meets_confidence_floor(event: dict, threshold: float = 0.30) -> bool:
    """Keep analytics events without model confidence; reject model scores below threshold.
    
    Lowered from 0.50 to 0.30 to allow detection of small flames (matchstick) and knives.
    """
    confidence = event.get('confidence')
    if confidence is None:
        return True
    try:
        return float(confidence) >= threshold
    except (TypeError, ValueError):
        return False


LOGGER = logging.getLogger(__name__)
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')


class EventBus:
    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()

    def publish(self, payload: dict):
        if self.loop and self.loop.is_running():
            try:
                self.loop.call_soon_threadsafe(self._broadcast, payload)
            except RuntimeError:
                pass

    def _broadcast(self, payload):
        for q in tuple(self.clients):
            if q.full():
                try: q.get_nowait()
                except asyncio.QueueEmpty: pass
            q.put_nowait(payload)


class CameraWorker:
    def __init__(self, camera: dict, db: Database, bus: EventBus, alert_manager: AlertManager):
        self.camera = camera
        self.db = db
        self.bus = bus
        self.alert_manager = alert_manager
        self.stop_signal = threading.Event()
        self.thread: threading.Thread | None = None
        self.jpeg: bytes | None = None
        self.jpeg_lock = threading.Lock()
        self.status = 'starting'
        self.error = ''
        self.current_people = 0
        self.fps = 0.0
        self.last_frame = 0.0
        self.camera_config = db.get_camera_config(camera['id']) or {}
        self._last_frame_error = ''
        self.detection_manager = DetectionManager()
        self._init_anomaly_detector()
    
    def _init_anomaly_detector(self):
        """Initialize anomaly detector (can be mocked in tests)."""
        # Anomaly detector is created in DetectionManager
        pass

    def start(self):
        self.thread = threading.Thread(target=self._run, name=f'camera-{self.camera["id"]}', daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_signal.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)

    def telemetry(self):
        return {'status': self.status, 'error': self.error, 'has_frame': self.latest() is not None,
                'current_people': self.current_people, 'fps': round(self.fps, 1),
                'last_frame_age_seconds': round(time.monotonic() - self.last_frame, 1) if self.last_frame else None}

    def latest(self):
        with self.jpeg_lock:
            return self.jpeg

    @staticmethod
    def _capture(src):
        """Open camera sources with Windows-friendly webcam backend and RTSP timeouts."""
        value = int(src) if src.isdecimal() else src
        if isinstance(value, str) and value.startswith(('rtsp://', 'rtsps://')):
            try:
                return cv2.VideoCapture(value, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000,
                ])
            except (TypeError, cv2.error):
                return cv2.VideoCapture(value, cv2.CAP_FFMPEG)
        if isinstance(value, int) and hasattr(cv2, 'CAP_DSHOW'):
            cap = cv2.VideoCapture(value, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(value)

    def _new_model(self):
        from ultralytics import YOLO
        return YOLO(settings.model_path)

    def _run(self):
        src = self.camera['source']
        while not self.stop_signal.is_set():
            cap = None
            try:
                with self.jpeg_lock: self.jpeg = None
                self.status = 'connecting'; self.error = ''
                cap = self._capture(src)
                if not cap.isOpened():
                    raise RuntimeError('Unable to open camera stream. Check URL, network, codec or webcam permissions.')
                self.status = 'loading model'
                model = self._new_model()
                analytics = Analytics(self.camera['zone'], self.camera['tripwire'], settings.event_cooldown_seconds)
                counter = 0
                last = time.monotonic()
                self.status = 'online'
                while not self.stop_signal.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError('Stream read failed or video ended; reconnecting')
                    counter += 1
                    if counter % settings.frame_skip:
                        continue
                    h, w = frame.shape[:2]
                    if w > settings.max_frame_width:
                        scale = settings.max_frame_width / w
                        frame = cv2.resize(frame, (settings.max_frame_width, round(h * scale)), interpolation=cv2.INTER_AREA)
                    h, w = frame.shape[:2]
                    
                    # Run person detection + tracking
                    # Auto-detect device: use CUDA if available, otherwise CPU
                    import torch
                    device = settings.device
                    if device == 'auto':
                        device = '0' if torch.cuda.is_available() else 'cpu'
                    
                    # CUDA automatically uses FP16 when available, no need for half/quantize parameter
                    # (half parameter is deprecated in newer Ultralytics versions)
                    
                    args: dict[str, Any] = {'persist': True, 'tracker': 'bytetrack.yaml',
                                            'classes': list(settings.detect_class_ids), 'conf': max(0.40, settings.confidence),
                                            'imgsz': settings.imgsz, 'verbose': False}
                    if device != 'auto':
                        args['device'] = device
                    result = model.track(frame, **args)[0]
                    persons = []
                    person_boxes = []
                    if result.boxes is not None and result.boxes.id is not None:
                        ids = result.boxes.id.int().cpu().tolist()
                        boxes = result.boxes.xyxy.cpu().tolist()
                        classes = result.boxes.cls.int().cpu().tolist()
                        # Create a detection result for person tracking to use improved draw
                        from .detectors import DetectionResult
                        person_result = DetectionResult(
                            boxes=boxes,
                            classes=classes,
                            confidences=result.boxes.conf.cpu().tolist() if result.boxes.conf is not None else [1.0]*len(boxes),
                            class_names=result.names if hasattr(result, 'names') else {0: 'person'}
                        )
                        # Draw person tracking with improved visuals
                        for track_id, box, class_id in zip(ids, boxes, classes):
                            x1, y1, x2, y2 = box
                            if class_id == 0:
                                persons.append({'id': track_id, 'foot': ((x1+x2)/(2*w), y2/h),
                                                'bbox': [float(x1), float(y1), float(x2), float(y2)]})
                                person_boxes.append([x1/w, y1/h, x2/w, y2/h])
                        # Use the detector's draw method for better visuals
                        color = (20, 220, 167)  # teal for persons
                        # Draw only person class (class_id == 0)
                        person_boxes_filtered = [b for b, c in zip(boxes, classes) if c == 0]
                        person_classes_filtered = [c for c in classes if c == 0]
                        person_conf_filtered = [conf for conf, c in zip(person_result.confidences, classes) if c == 0]
                        if person_boxes_filtered:
                            person_result_draw = DetectionResult(
                                boxes=person_boxes_filtered,
                                classes=person_classes_filtered,
                                confidences=person_conf_filtered,
                                class_names={0: 'person'}
                            )
                            # Use the detection manager's object detector to draw (or any detector)
                            frame = self.detection_manager.object_detector.draw(frame, person_result_draw, color)
                    
                    # Run analytics (intrusion, tripwire)
                    events = analytics.process(persons)
                    self.current_people = analytics.current_people
                    
                    # Run advanced detections
                    detection_events = self.detection_manager.process_frame(
                        frame, persons, self.camera['id'], self.camera_config)
                    
                    # Combine all events
                    all_events = events + (detection_events or [])
                    # A detector must return event dictionaries. Ignore malformed plugin/model
                    # outputs instead of killing the camera worker for every subsequent frame.
                    valid_events = []
                    for item in all_events:
                        if isinstance(item, dict):
                            # Ignore model-generated detections below the global 50% floor.
                            # Analytics events (zone/line crossings) have no confidence and remain valid.
                            if not event_meets_confidence_floor(item, 0.50):
                                continue
                            valid_events.append(item)
                            self._handle_event(frame, item)
                        else:
                            LOGGER.warning('Ignoring malformed detection event from %s: %s',
                                           self.camera['id'], type(item).__name__)
                    all_events = valid_events
                    
                    self._draw_geometry(frame, w, h)
                    cv2.rectangle(frame, (12,12), (204,48), (10,20,34), -1)
                    cv2.putText(frame, f'PEOPLE: {self.current_people}', (24,38),
                                cv2.FONT_HERSHEY_SIMPLEX, .67, (255,255,255), 2)
                    
                    # Show active detections on frame
                    self._draw_detection_status(frame, all_events)
                    
                    success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality])
                    if success:
                        with self.jpeg_lock: self.jpeg = encoded.tobytes()
                    now = time.monotonic()
                    self.fps = .8 * self.fps + .2 / max(.001, now - last)
                    last = now
                    self.last_frame = now
                    self.status = 'online'
            except Exception as exc:
                with self.jpeg_lock: self.jpeg = None
                self.status = 'reconnecting'
                self.error = str(exc).replace(src, '[stream]')[:180]
                LOGGER.warning('Camera %s: %s', self.camera['id'], self.error)
            finally:
                if cap is not None: cap.release()
            self.current_people = 0
            if self.stop_signal.wait(3): break
        self.status = 'stopped'

    def _handle_event(self, frame: np.ndarray, event: dict):
        """Process event: save to DB, send alerts, broadcast."""
        event_kind = event.get('type', event.get('kind', 'unknown'))
        track_id = event.get('track_id')
        message = event.get('message', f'{event_kind} detected')
        severity = event.get('severity', 'medium')
        camera_id = event.get('camera_id', self.camera['id'])
        
        # Save snapshot
        snapshot_filename = None
        dest = settings.data_dir / 'snapshots'
        dest.mkdir(parents=True, exist_ok=True)
        from uuid import uuid4
        filename = uuid4().hex + '.jpg'
        path = dest / filename
        if cv2.imwrite(str(path), frame):
            snapshot_filename = filename
        
        # Store in database
        metadata = {k: v for k, v in event.items() if k not in ('type', 'kind', 'track_id', 'message', 'severity', 'camera_id', 'bbox', 'confidence')}
        if 'bbox' in event:
            metadata['bbox'] = event['bbox']
        if 'confidence' in event:
            metadata['confidence'] = event['confidence']
        if 'subtype' in event:
            metadata['subtype'] = event['subtype']
        
        db_event = self.db.store_event(
            camera_id, self.camera['name'], event_kind, track_id, message,
            snapshot_filename, metadata
        )
        
        # Broadcast to WebSocket clients
        self.bus.publish({'type': 'event', 'event': db_event})
        
        # Trigger alerts
        alert_event = AlertEvent(
            event_id=db_event['id'],
            camera_id=camera_id,
            camera_name=self.camera['name'],
            kind=event_kind,
            severity=severity,
            message=message,
            snapshot=frame.copy(),
            metadata=metadata,
            timestamp=db_event['created_at']
        )
        self.alert_manager.process_event(alert_event, self.camera_config)

    def _draw_geometry(self, frame, w, h):
        zone = self.camera['zone']
        if len(zone) >= 3:
            import numpy as np
            poly = np.array([(round(x*w), round(y*h)) for x,y in zone], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], (48,60,200))
            cv2.addWeighted(overlay, .18, frame, .82, 0, frame)
            cv2.polylines(frame, [poly], True, (75,94,255), 2)
            cv2.putText(frame, 'RESTRICTED', tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, .5, (75,94,255), 2)
        line = self.camera['tripwire']
        if len(line) == 2:
            (x1,y1),(x2,y2) = line
            cv2.line(frame, (round(x1*w),round(y1*h)), (round(x2*w),round(y2*h)), (40,198,245), 3)
            cv2.putText(frame, 'COUNT LINE', (round(x1*w),max(18,round(y1*h)-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (40,198,245), 2)

    def _draw_detection_status(self, frame, events):
        """Draw active detection badges on frame."""
        if not events:
            return
        h, w = frame.shape[:2]
        y_offset = 60
        for event in events[:5]:  # Show max 5
            event_type = event.get('type', event.get('kind', 'unknown'))
            severity = event.get('severity', 'medium')
            colors = {'critical': (0, 0, 255), 'high': (0, 100, 255), 'medium': (0, 200, 255), 'low': (0, 255, 100)}
            color = colors.get(severity, (255, 255, 255))
            label = f"{event_type.upper()} [{severity.upper()}]"
            cv2.rectangle(frame, (12, y_offset), (12 + len(label)*12, y_offset + 22), (0, 0, 0), -1)
            cv2.putText(frame, label, (16, y_offset + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 24


class CameraManager:
    def __init__(self, db, bus, alert_manager):
        self.db = db
        self.bus = bus
        self.alert_manager = alert_manager
        self.lock = threading.RLock()
        self.workers: dict[str, CameraWorker] = {}

    def start(self):
        for camera in self.db.list_cameras(safe=False):
            if camera['enabled']:
                self.reload(camera['id'])

    def reload(self, camera_id):
        with self.lock:
            previous = self.workers.pop(camera_id, None)
            if previous: previous.stop()
            camera = self.db.get_camera(camera_id, safe=False)
            if camera and camera['enabled']:
                worker = CameraWorker(camera, self.db, self.bus, self.alert_manager)
                self.workers[camera_id] = worker
                worker.start()

    def stop(self):
        with self.lock:
            for worker in list(self.workers.values()): worker.stop()
            self.workers.clear()

    def telemetry(self, cam_id):
        with self.lock:
            worker = self.workers.get(cam_id)
        return worker.telemetry() if worker else {'status':'disabled', 'error':'', 'current_people':0, 'fps':0, 'last_frame_age_seconds':None}

    def frame(self, cam_id):
        with self.lock:
            worker = self.workers.get(cam_id)
        return worker.latest() if worker else None
    
    def get_camera_config(self, cam_id):
        with self.lock:
            worker = self.workers.get(cam_id)
        if worker:
            return worker.camera_config
        return self.db.get_camera_config(cam_id)
    
    def update_camera_config(self, cam_id, config):
        with self.lock:
            worker = self.workers.get(cam_id)
        current = self.db.get_camera_config(cam_id) or {}
        merged = {**current, **config}
        saved = self.db.update_camera_config(cam_id, merged)
        if worker:
            worker.camera_config = saved or merged
        return saved