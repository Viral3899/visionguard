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
from .config import local_model_path, settings
from .database import Database
from .detectors import DetectionManager
from .incidents import IncidentTracker, eligible_event, draw_event_box

def event_meets_confidence_floor(event: dict, threshold: float = 0.50) -> bool:
    """Keep analytics events without model confidence; reject model scores below threshold.
    
    The global policy rejects model events below 50%; model-specific thresholds cannot bypass it.
    """
    confidence = event.get('confidence')
    if confidence is None:
        return True
    try:
        return float(confidence) >= threshold
    except (TypeError, ValueError):
        return False


LOGGER = logging.getLogger(__name__)
# Still images are reserved for meaningful security detections, never routine
# presence/entry/exit/loitering frames.
SNAPSHOT_EVENT_KINDS = frozenset({
    'fire_detected', 'smoke', 'violence', 'weapon', 'fall_detected', 'unusual_activity'
})
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
        self._last_saved_event: dict[str, float] = {}
        self.detection_manager = DetectionManager()
        self.incidents = IncidentTracker(settings.incident_min_confidence,
                                         settings.incident_confirmation_frames,
                                         max_missed_frames=max(2, settings.object_every_n_frames))
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
        if isinstance(value, int) and os.name == 'nt' and hasattr(cv2, 'CAP_DSHOW'):
            cap = cv2.VideoCapture(value, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(value)

    def _new_model(self):
        from ultralytics import YOLO
        return YOLO(local_model_path(settings.model_path))

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
                # Keep only the newest frame so a slow inference loop does not build latency.
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
                    args: dict[str, Any] = {'persist': True, 'tracker': 'bytetrack.yaml',
                                            'classes': list(settings.detect_class_ids), 'conf': max(0.40, settings.confidence),
                                            'imgsz': settings.imgsz, 'verbose': False}
                    if settings.device != 'auto':
                        args['device'] = settings.device
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
                            confidences=(result.boxes.conf.cpu().tolist() if getattr(result.boxes, 'conf', None) is not None else [1.0]*len(boxes)),
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
                        if person_boxes_filtered and hasattr(self.detection_manager, 'object_detector'):
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
                    
                    # Analytics still update counters, but never generate routine
                    # snapshots or Telegram. Security snapshots require repeated,
                    # real >=50% model evidence and one still per active incident.
                    for item in events:
                        if isinstance(item, dict):
                            self._handle_event(frame, item)
                    raw_detections = [item for item in (detection_events or []) if isinstance(item, dict)]
                    confirmed = self.incidents.update(raw_detections, time.monotonic())
                    for item in confirmed:
                        self._handle_event(frame, item)
                    all_events = events + [item for item in raw_detections
                                           if eligible_event(item, settings.incident_min_confidence)]
                    
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
                import traceback
                LOGGER.warning('Camera %s: %s', self.camera['id'], self.error)
                LOGGER.debug('Camera %s traceback: %s', self.camera['id'], traceback.format_exc())
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

        # Avoid both DB noise and a false low-score event consuming the
        # cooldown for a subsequent true >=50% detection.
        if event_kind in {'loitering', 'proximity_alert', 'heat_source', 'fire_watch'}:
            return
        if event_kind in SNAPSHOT_EVENT_KINDS and not eligible_event(event, settings.incident_min_confidence):
            return
        
        # Suppress repeated per-frame detections. Save and alert once per event
        # identity during the configured cooldown, not once for every matching frame.
        identity = f"{event_kind}:{track_id if track_id is not None else event.get('subtype', 'global')}"
        now = time.monotonic()
        cooldown = max(1, int(settings.event_cooldown_seconds))
        previous = self._last_saved_event.get(identity, float('-inf'))
        if now - previous < cooldown:
            return
        self._last_saved_event[identity] = now

        # Save snapshot
        snapshot_filename = None
        confirmed_threat = event_kind in SNAPSHOT_EVENT_KINDS and eligible_event(event, settings.incident_min_confidence)
        if confirmed_threat:
            dest = settings.data_dir / 'snapshots'
            dest.mkdir(parents=True, exist_ok=True)
            from uuid import uuid4
            filename = uuid4().hex + '.jpg'
            path = dest / filename
            snapshot_frame = draw_event_box(frame.copy(), event)
            if cv2.imwrite(str(path), snapshot_frame):
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
        if confirmed_threat:
            self.alert_manager.process_event(alert_event, self.camera_config)

    def _draw_geometry(self, frame, w, h):
        zone = self.camera.get('zone', [])
        # Validate zone: must be list of [x, y] pairs with at least 3 points
        if isinstance(zone, list) and len(zone) >= 3:
            try:
                # Filter out invalid points (not 2-element sequences)
                valid_zone = [p for p in zone if isinstance(p, (list, tuple)) and len(p) == 2]
                if len(valid_zone) >= 3:
                    import numpy as np
                    poly = np.array([(round(x*w), round(y*h)) for x, y in valid_zone], dtype=np.int32)
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [poly], (48,60,200))
                    cv2.addWeighted(overlay, .18, frame, .82, 0, frame)
                    cv2.polylines(frame, [poly], True, (75,94,255), 2)
                    cv2.putText(frame, 'RESTRICTED', tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, .5, (75,94,255), 2)
            except (ValueError, TypeError, IndexError) as e:
                LOGGER.debug('Invalid zone data for camera %s: %s', self.camera['id'], e)
        
        line = self.camera.get('tripwire', [])
        # Validate tripwire: must be exactly 2 points, each with [x, y]
        if isinstance(line, list) and len(line) == 2:
            try:
                p1, p2 = line[0], line[1]
                if (isinstance(p1, (list, tuple)) and len(p1) == 2 and
                    isinstance(p2, (list, tuple)) and len(p2) == 2):
                    x1, y1 = p1
                    x2, y2 = p2
                    cv2.line(frame, (round(x1*w),round(y1*h)), (round(x2*w),round(y2*h)), (40,198,245), 3)
                    cv2.putText(frame, 'COUNT LINE', (round(x1*w),max(18,round(y1*h)-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (40,198,245), 2)
            except (ValueError, TypeError, IndexError) as e:
                LOGGER.debug('Invalid tripwire data for camera %s: %s', self.camera['id'], e)

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