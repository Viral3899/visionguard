"""FastAPI entry point. Localhost by default; opt-in password gate for LAN."""
import asyncio
import csv
import base64
import hashlib
import json
import secrets
import time
import hmac
import io
import logging
import re
from anyio import to_thread
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import local_model_path, settings
from .database import Database
from .engine import CameraManager, CameraWorker, EventBus
from .schemas import CameraCreate, CameraUpdate
from .alerts import AlertManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
LOGGER = logging.getLogger('visionguard')


def _ensure_default_camera(db):
    """Create a default webcam camera if none exists."""
    try:
        import cv2
        cameras = db.list_cameras()
        if cameras:
            return  # Already have cameras
        
        # Try to find a working camera
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    LOGGER.info(f'Auto-creating default camera: index {i} ({w}x{h})')
                    from uuid import uuid4
                    from datetime import datetime, timezone
                    import json
                    cam_id = uuid4().hex[:12]
                    now = datetime.now(timezone.utc).isoformat()
                    with db.connection() as cx:
                        cx.execute('INSERT INTO cameras (id,name,source,zone,tripwire,enabled,created_at,client_id) VALUES (?,?,?,?,?,?,?,NULL)', (
                            cam_id, f'Webcam {i}', str(i), json.dumps([]),
                            json.dumps([]), 1, now))
                        cx.execute('INSERT INTO totals(camera_id) VALUES (?)', (cam_id,))
                        cx.execute('''
                            INSERT INTO camera_config (camera_id, fire_enabled, violence_enabled, anomaly_enabled, object_enabled, telegram_enabled, tts_enabled, sensitivity)
                            VALUES (?, 1, 1, 1, 1, 1, 1, 1.0)
                        ''', (cam_id,))
                    return
    except Exception as e:
        LOGGER.debug(f'Auto camera setup skipped: {e}')


class Credentials(BaseModel):
    username: str = "admin"
    password: str


class CameraConfigUpdate(BaseModel):
    fire_enabled: bool | None = None
    violence_enabled: bool | None = None
    anomaly_enabled: bool | None = None
    object_enabled: bool | None = None
    telegram_enabled: bool | None = None
    tts_enabled: bool | None = None
    sensitivity: float | None = None


class TestAlertsRequest(BaseModel):
    camera_id: str | None = None


class CameraSourceTest(BaseModel):
    source: str = Field(min_length=1, max_length=2048)


def create_app(db_path: Path | None = None, start_cameras: bool = True, auto_camera: bool = True):
    db = Database(db_path or (settings.data_dir / 'visionguard.db'))
    bus = EventBus()
    alert_manager = AlertManager(db)
    manager = CameraManager(db, bus, alert_manager)
    
    secret = settings.session_secret.encode('utf-8')
    admin_password = settings.admin_password or settings.dashboard_password

    if admin_password.startswith('CHANGE_THIS'):
        raise RuntimeError('Replace the example administrator password before starting.')
    if settings.host not in {'localhost', '127.0.0.1', '::1'} and not admin_password:
        raise RuntimeError('Set ADMIN_PASSWORD before binding to a network interface.')
    if not secret or secret.startswith((b'CHANGE_THIS', b'change-this-visionguard-secret')):
        secret = secrets.token_bytes(32)

    def make_session(user_id, role, client_id=None):
        payload = json.dumps({'sub': user_id, 'role': role, 'client_id': client_id, 'exp': int(time.time()) + 86400}, separators=(',', ':')).encode()
        body = base64.urlsafe_b64encode(payload).decode().rstrip('=')
        sig = hmac.new(secret, body.encode(), 'sha256').hexdigest()
        return body + '.' + sig

    def session_user(cookie):
        if not cookie or '.' not in cookie: return None
        body, sig = cookie.rsplit('.', 1)
        expected = hmac.new(secret, body.encode(), 'sha256').hexdigest()
        if not hmac.compare_digest(sig, expected): return None
        try:
            payload = json.loads(base64.urlsafe_b64decode(body + '=' * (-len(body) % 4)))
            if payload.get('exp', 0) < time.time(): return None
            return payload
        except Exception: return None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.init()
        # Run optional webcam discovery only after schema initialization.
        if auto_camera:
            _ensure_default_camera(db)
        bus.loop = asyncio.get_running_loop()
        if start_cameras: manager.start()
        maintenance_task = asyncio.create_task(_maintenance(db))
        yield
        maintenance_task.cancel()
        try: await maintenance_task
        except asyncio.CancelledError: pass
        manager.stop()
        bus.loop = None

    app = FastAPI(title='VisionGuard Surveillance Analytics', version='2.1.0', lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.db, app.state.manager, app.state.bus, app.state.alert_manager = db, manager, bus, alert_manager
    app.mount('/static', StaticFiles(directory=settings.static_dir), name='static')

    def get_user(request):
        return session_user(request.cookies.get('vg_session'))

    def require_user(request):
        user = get_user(request)
        if (admin_password or db.list_clients()) and not user:
            raise HTTPException(401, 'Please sign in')
        if user and user.get('role')=='client':
            client=db.get_client(user.get('client_id'))
            if not client or not client.get('active'): raise HTTPException(401, 'Client account is disabled')
        return user or {'role': 'admin', 'sub': 'local-admin', 'client_id': None}

    def require_admin(request):
        user = require_user(request)
        if user['role'] != 'admin': raise HTTPException(403, 'Administrator access required')
        return user

    @app.middleware('http')
    async def authorization(request: Request, call_next):
        path = request.url.path
        if path.startswith('/api/') and path not in ('/api/session', '/api/login'):
            user = session_user(request.cookies.get('vg_session'))
            if (admin_password or db.list_clients()) and not user:
                return JSONResponse({'detail': 'Authentication required'}, status_code=401)
        return await call_next(request)

    @app.get('/')
    async def home():
        return FileResponse(settings.static_dir / 'index.html')

    @app.get('/health')
    async def health():
        cameras = db.list_cameras()
        online = sum(manager.telemetry(c['id']).get('status') == 'online' for c in cameras)
        return {'status': 'ok', 'cameras': len(cameras), 'online_cameras': online}

    @app.get('/api/session')
    async def session(request: Request):
        user = get_user(request)
        if user and user.get('role')=='client':
            client=db.get_client(user.get('client_id'))
            if not client or not client.get('active'): user=None
        required = bool(admin_password or db.list_clients())
        result={'password_required': required, 'authenticated': bool(user) or not required}
        if user: result.update(role=user.get('role'), username=user.get('sub'))
        elif not required: result.update(role='admin', username='local-admin')
        return result

    @app.post('/api/login')
    async def login(credentials: Credentials, response: Response, request: Request):
        role = None; client_id = None
        if credentials.username == settings.admin_username and admin_password and hmac.compare_digest(credentials.password.encode(), admin_password.encode()):
            role = 'admin'
        else:
            client = db.get_client_login(credentials.username)
            if client:
                salt, expected_hash = client['password_hash'].split('$',1)
                actual_hash = hashlib.pbkdf2_hmac('sha256', credentials.password.encode(), bytes.fromhex(salt), 180000).hex()
                valid = hmac.compare_digest(actual_hash, expected_hash)
                if valid: role, client_id = 'client', client['id']
        if not role: raise HTTPException(status_code=401, detail='Invalid username or password')
        token = make_session(credentials.username, role, client_id)
        response.set_cookie('vg_session', token, httponly=True, samesite='lax',
                            secure=request.url.scheme == 'https', max_age=86400, path='/')
        return {'ok': True, 'role': role}

    @app.post('/api/logout')
    async def logout(response: Response):
        response.delete_cookie('vg_session', path='/')
        return {'ok': True}

    @app.get('/api/clients')
    async def list_clients(request: Request):
        require_admin(request)
        return db.list_clients()

    @app.post('/api/clients', status_code=201)
    async def create_client(payload: dict, request: Request):
        require_admin(request)
        if not admin_password: raise HTTPException(503, 'Set ADMIN_PASSWORD in .env before enabling client accounts')
        name, username, password = (str(payload.get(k, '')).strip() for k in ('name','username','password'))
        if len(name) < 2 or len(username) < 3 or len(password) < 8:
            raise HTTPException(422, 'Client name, username (3+ chars), and password (8+ chars) are required')
        salt=secrets.token_bytes(16)
        password_hash=salt.hex()+'$'+hashlib.pbkdf2_hmac('sha256',password.encode(),salt,180000).hex()
        try: return db.create_client(name, username, password_hash)
        except Exception: raise HTTPException(409, 'Username already exists')

    @app.patch('/api/clients/{client_id}')
    async def set_client(client_id: str, payload: dict, request: Request):
        require_admin(request)
        if not db.set_client_active(client_id, bool(payload.get('active', True))): raise HTTPException(404, 'Client not found')
        return {'ok': True}

    @app.get('/api/cameras')
    async def list_cameras(request: Request):
        user = require_user(request)
        cameras = db.list_cameras()
        if user['role'] == 'client': cameras = [c for c in cameras if c.get('client_id') == user.get('client_id')]
        for cam in cameras:
            cam.update(manager.telemetry(cam['id']))
            cam['config'] = manager.get_camera_config(cam['id'])
        return cameras

    @app.get('/api/cameras/discover')
    async def discover_cameras(request: Request):
        require_admin(request)
        def scan():
            import camera_utils
            return camera_utils.detect_cameras(max_index=8)
        return await asyncio.to_thread(scan)

    @app.post('/api/cameras/test-source')
    async def test_camera_source(payload: CameraSourceTest, request: Request):
        require_admin(request)
        source = payload.source.strip()
        # Reuse the same supported source rules as camera creation.
        try:
            CameraCreate(name='Test Camera', source=source)
        except Exception as exc:
            raise HTTPException(422, str(exc))
        def probe():
            import cv2
            cap = CameraWorker._capture(source)
            try:
                if not cap.isOpened():
                    return {'ok': False, 'message': 'Source could not be opened. Check camera permissions, RTSP URL, network and stream credentials.'}
                ok, frame = cap.read()
                if not ok or frame is None:
                    return {'ok': False, 'message': 'Source opened but no video frame arrived. Check stream path, codec and camera availability.'}
                h, w = frame.shape[:2]
                return {'ok': True, 'message': f'Camera source verified: {w} × {h}', 'width': w, 'height': h}
            finally:
                cap.release()
        return await asyncio.to_thread(probe)

    @app.post('/api/cameras', status_code=201)
    async def add_camera(camera: CameraCreate, request: Request):
        user = require_user(request)
        if user['role'] == 'client': raise HTTPException(403, 'Only administrators can add cameras')
        if camera.client_id and not db.get_client(camera.client_id): raise HTTPException(422, 'Selected client does not exist')
        if any(c.get('source') == camera.source for c in db.list_cameras(safe=False)):
            raise HTTPException(409, 'This camera source is already configured. Select the existing camera or use a different source.')
        created = db.create_camera(camera)
        if created['enabled'] and start_cameras:
            try: manager.reload(created['id'])
            except Exception: LOGGER.exception('Camera worker start failed for %s', created['id'])
        return created

    @app.get('/api/cameras/{camera_id}/source')
    async def get_camera_source(camera_id: str, request: Request):
        require_admin(request)
        camera=db.get_camera(camera_id, safe=False)
        if not camera: raise HTTPException(404, 'Camera not found')
        return {'source': camera['source']}

    @app.patch('/api/cameras/{camera_id}')
    async def update_camera(camera_id: str, camera: CameraUpdate, request: Request):
        user = require_user(request)
        existing = db.get_camera(camera_id)
        if not existing: raise HTTPException(404, 'Camera not found')
        if user['role'] == 'client' and existing.get('client_id') != user.get('client_id'): raise HTTPException(404, 'Camera not found')
        changes=camera.model_dump(exclude_unset=True)
        if user['role'] == 'client' and {'source','client_id'} & set(changes): raise HTTPException(403, 'Only administrators can change camera sources or assignments')
        if changes.get('client_id') and not db.get_client(changes['client_id']): raise HTTPException(422, 'Selected client does not exist')
        updated = db.update_camera(camera_id, camera.model_dump(exclude_unset=True))
        if updated is None:
            raise HTTPException(404, 'Camera not found')
        if start_cameras: manager.reload(camera_id)
        return updated

    @app.get('/api/cameras/{camera_id}/config')
    async def get_camera_config(camera_id: str, request: Request):
        user = require_user(request); cam = db.get_camera(camera_id)
        if not cam or (user['role']=='client' and cam.get('client_id') != user.get('client_id')): raise HTTPException(404, 'Camera not found')
        config = manager.get_camera_config(camera_id)
        if config is None:
            raise HTTPException(404, 'Camera not found')
        return config

    @app.patch('/api/cameras/{camera_id}/config')
    async def update_camera_config(camera_id: str, config: CameraConfigUpdate, request: Request):
        user = require_user(request); cam = db.get_camera(camera_id)
        if not cam or (user['role']=='client' and cam.get('client_id') != user.get('client_id')): raise HTTPException(404, 'Camera not found')
        updated = manager.update_camera_config(camera_id, config.model_dump(exclude_unset=True))
        if updated is None:
            raise HTTPException(404, 'Camera not found')
        return updated

    @app.delete('/api/cameras/{camera_id}')
    async def delete_camera(camera_id: str, request: Request):
        user = require_user(request); existing = db.get_camera(camera_id)
        if not existing or user['role'] != 'admin': raise HTTPException(404, 'Camera not found')
        db.update_camera(camera_id, {'enabled': False})
        manager.reload(camera_id)
        db.delete_camera(camera_id)
        return {'ok': True}

    @app.post('/api/admin/clear-all-data')
    async def clear_all_data(request: Request):
        """Clear all events, alert logs, snapshots, and test videos for a fresh start. Admin only."""
        user = require_admin(request)
        try:
            db.clear_all_data()
            # Reload all cameras to reset their state
            for cam in db.list_cameras(safe=False):
                if cam['enabled']:
                    manager.reload(cam['id'])
            return {'ok': True, 'message': 'All events, alert logs, snapshots, and test videos cleared. Cameras reloaded.'}
        except Exception as e:
            raise HTTPException(500, f'Failed to clear data: {e}')

    @app.post('/api/cameras/{camera_id}/test')
    async def test_camera(camera_id: str, request: Request):
        user = require_user(request); camera = db.get_camera(camera_id)
        if not camera or (user['role']=='client' and camera.get('client_id') != user.get('client_id')): raise HTTPException(404, 'Camera not found')
        if start_cameras: manager.reload(camera_id)
        return {'camera_id': camera_id, 'source_configured': bool(camera.get('source')), 'telemetry': manager.telemetry(camera_id),
                'stream_url': f'/stream/{camera_id}', 'message': 'Connection attempt restarted; check status and live preview.'}

    upload_slot = asyncio.Lock()

    def video_output(directory, token, user):
        owner = str(user.get('client_id') or 'admin').encode()
        prefix = hashlib.sha256(owner).hexdigest()
        return directory / f'{prefix}_{token}_annotated.mp4'

    @app.post('/api/test-videos')
    async def test_uploaded_video(request: Request, file: UploadFile = File(...)):
        """Run an uploaded local test clip through the same on-device detectors.

        This is a diagnostic workflow: it returns an annotated MP4 and a summary.
        Telegram is sent only for confirmed unusual events scoring >=50%,
        and only when Telegram is configured. No loitering or normal-frame stills.
        """
        user = require_user(request)
        if upload_slot.locked():
            raise HTTPException(429, 'Another clip is being analysed. Retry after it finishes.')
        async with upload_slot:
            return await process_upload(user, file)

    async def process_upload(user, file):
        allowed = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in allowed:
            raise HTTPException(415, 'Upload an MP4, AVI, MOV, MKV, WEBM or M4V video.')
        test_dir = settings.data_dir / 'test_videos'
        test_dir.mkdir(parents=True, exist_ok=True)
        token = secrets.token_hex(12)
        input_path = test_dir / f'{token}{suffix}'
        output_path = video_output(test_dir, token, user)
        max_bytes = 250 * 1024 * 1024
        size = 0
        try:
            with input_path.open('wb') as target:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(413, 'Video is larger than the 250 MB testing limit.')
                    target.write(chunk)
        except Exception:
            input_path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

        def analyze():
            import cv2
            from collections import defaultdict
            from .detectors import DetectionManager
            from .alerts import AlertEvent
            from .incidents import IncidentTracker, eligible_event, draw_event_box
            
            # === VIDEO PROPERTIES LOGGING ===
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise RuntimeError('Uploaded file could not be decoded by OpenCV. Try MP4 (H.264) format.')
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            fps = float(fps) if 0 < fps < float('inf') else 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            LOGGER.info(f"=== VIDEO UPLOAD ANALYSIS STARTED ===")
            LOGGER.info(f"File: {file.filename} ({size} bytes)")
            LOGGER.info(f"Video Properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration, Codec: {fourcc_str}")
            LOGGER.info(f"Frame skip setting: {settings.frame_skip} (process every {settings.frame_skip} frame(s))")
            LOGGER.info("Upload incident score floor: %.0f%%; 2+ analyzed frames", settings.incident_min_confidence * 100)
            LOGGER.info("Upload alert channels: TTS configured=%s, Telegram configured=%s", settings.tts_enabled, bool(settings.telegram_bot_token and settings.telegram_chat_id))
            
            if width <= 0 or height <= 0:
                cap.release(); raise RuntimeError('Video has invalid dimensions.')
            
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            if not writer.isOpened():
                cap.release(); raise RuntimeError('Could not create annotated MP4 output.')

            try:
                detectors = DetectionManager()
            except Exception:
                cap.release()
                writer.release()
                raise
            # One missing specialist must not stop the other detectors. However,
            # never return "analysis complete: zero detections" when all models
            # are missing -- this is a setup error, not a negative test result.
            if not any((detectors.fire_detector.status == 'ready',
                        detectors.violence_detector.model is not None,
                        getattr(getattr(detectors, 'weapon_detector', None), 'model', None) is not None,
                        getattr(detectors.anomaly_detector, 'model', None) is not None,
                        detectors.object_detector.model is not None)):
                cap.release()
                writer.release()
                output_path.unlink(missing_ok=True)
                raise RuntimeError('No detection models are ready. Install fire/smoke and/or security specialist weights and run verify_models.py.')
            # Video testing emphasizes recall. Real-time cameras use configurable
            # sampling rates to balance accuracy against capture latency.
            detectors.object_every_n_frames = 1
            detectors.violence_every_n_frames = 1
            detectors.anomaly_every_n_frames = 1
            if getattr(detectors, 'weapon_detector', None) is not None:
                detectors.weapon_every_n_frames = 1
            person_model = None
            try:
                from ultralytics import YOLO
                person_model = YOLO(local_model_path(settings.model_path))
                LOGGER.info(f"Person tracking model loaded: {settings.model_path}")
            except Exception as exc:
                LOGGER.warning('Person tracker unavailable for uploaded video: %s', exc)

            # Suppress local sirens during uploads; Telegram is still eligible
            # for confirmed unusual activity when explicitly configured.
            test_config = {'fire_enabled': 1, 'violence_enabled': 1, 'anomaly_enabled': 1,
                           'object_enabled': 1, 'tts_enabled': 0, 'telegram_enabled': 1}
            incidents = IncidentTracker(settings.incident_min_confidence,
                                        settings.incident_confirmation_frames,
                                        max_missed_frames=1)
            emitted = set()
            confirmed_records = []
            event_counts = defaultdict(int)
            max_frames = 12000
            frame_index = 0
            processed = 0
            fps_device = 'cpu'
            use_half = False
            try:
                import torch
                if settings.device == 'auto' and torch.cuda.is_available(): 
                    fps_device = 'cuda'
                    use_half = True
                elif settings.device not in ('auto', 'cpu'): 
                    fps_device = settings.device
                    use_half = True
            except Exception:
                pass
            
            # Frame analysis logging
            frame_log_interval = max(1, int(fps * 5))  # Log every 5 seconds of video
            LOGGER.info(f"Starting frame analysis (logging every {frame_log_interval} frames)...")
            
            try:
                while frame_index < max_frames:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        LOGGER.info(f"End of video reached at frame {frame_index}")
                        break
                    frame_index += 1
                    
                    # Log video progress periodically
                    if frame_index % frame_log_interval == 0:
                        progress_pct = (frame_index / max(total_frames, 1)) * 100
                        LOGGER.info(f"Progress: frame {frame_index}/{total_frames} ({progress_pct:.1f}%), analyzed: {processed}, events so far: {dict(event_counts)}")
                    
                    tracks = []
                    if person_model is not None:
                        try:
                            track_args = {'persist': True, 'tracker': 'bytetrack.yaml',
                                'classes': [0], 'conf': max(.40, settings.confidence), 'imgsz': settings.imgsz,
                                'device': fps_device, 'verbose': False}
                            # CUDA automatically uses FP16 when available
                            result = person_model.track(frame, **track_args)[0]
                            if result.boxes is not None and result.boxes.id is not None:
                                ids = result.boxes.id.int().cpu().tolist()
                                boxes = result.boxes.xyxy.cpu().tolist()
                                for tid, box in zip(ids, boxes):
                                    tracks.append({'id': tid, 'bbox': [float(v) for v in box],
                                                   'foot': ((box[0]+box[2])/2/frame.shape[1], box[3]/frame.shape[0])})
                        except Exception as exc:
                            LOGGER.debug('Test video person tracking frame failed: %s', exc)
                    
                    frame_events = detectors.process_frame(frame, tracks, 'uploaded-video-test', test_config, timestamp=(frame_index - 1) / fps) or []
                    
                    # Only real model detections at/above the policy threshold
                    # are eligible. No loitering, proximity, regular frames or
                    # inflated smoke confidence can reach the alert channels.
                    frame_events = [e for e in frame_events
                                    if eligible_event(e, settings.incident_min_confidence)]
                    for event in frame_events:
                        event_counts[event['type']] += 1
                        draw_event_box(frame, event)
                    for event in incidents.update(frame_events, (frame_index - 1) / fps):
                        kind = event['type']
                        confidence = float(event['confidence'])
                        emitted.add(kind)
                        from uuid import uuid4
                        snapshot_name = uuid4().hex + '.jpg'
                        # Private per-user test snapshots with a dedicated URL.
                        prefix = video_output(test_dir, token, user).stem.removesuffix('_annotated')
                        snapshot_path = test_dir / f'{prefix}_{snapshot_name}'
                        snapshot_frame = draw_event_box(frame.copy(), event)
                        if not cv2.imwrite(str(snapshot_path), snapshot_frame):
                            snapshot_name = None
                        snapshot_url = (f'/api/test-videos/{token}/snapshots/{snapshot_name}'
                                        if snapshot_name else None)
                        confirmed_records.append({
                            'kind': kind, 'confidence': confidence,
                            'message': event.get('message', f'{kind} detected in uploaded video'),
                            'frame': frame_index,
                            'time_seconds': round((frame_index - 1) / fps, 2),
                            'bbox': event.get('bbox'),
                            'snapshot': snapshot_name,
                            'snapshot_url': snapshot_url
                        })
                        alert = AlertEvent(event_id=token + uuid4().hex, camera_id='uploaded-video-test',
                            camera_name='Uploaded video test', kind=kind,
                            severity=event.get('severity', 'high'),
                            message=event.get('message', f'{kind} detected in uploaded video'),
                            snapshot=snapshot_frame,
                            metadata={'confidence': confidence},
                            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'))
                        alert_manager.process_event(alert, test_config)
                        LOGGER.warning('CONFIRMED INCIDENT: %s (%.0f%%) on frame %d', kind, 100 * confidence, frame_index)
                    writer.write(frame)
                    processed += 1
            finally:
                cap.release(); writer.release()
            
            LOGGER.info(f"=== VIDEO ANALYSIS COMPLETE ===")
            LOGGER.info(f"Total frames read: {frame_index}, Frames analyzed: {processed}")
            LOGGER.info(f"Event counts: {dict(event_counts)}")
            LOGGER.info(f"Confirmed alerts (2+ detections): {sorted(emitted)}")
            LOGGER.info(f"Early warning note: Smoke/heat indicators are treated as early-warning clues. This model pipeline does not predict ignition with certainty.")
            
            if frame_index == 0:
                raise RuntimeError('The uploaded file contains no readable video frames.')
            return {'ok': True, 'filename': file.filename, 'size_bytes': size,
                    'frames_read': frame_index, 'frames_analyzed': processed,
                    'duration_seconds': round(frame_index / fps, 2),
                    'minimum_confidence': settings.incident_min_confidence,
                    'fire_detector': detectors.fire_detector.diagnostics(),
                    'models_loaded': {
                        'person': person_model is not None,
                        'fire': detectors.fire_detector.model is not None,
                        'violence': detectors.violence_detector.model is not None,
                        'weapon_specialist': getattr(getattr(detectors, 'weapon_detector', None), 'model', None) is not None,
                        'pose': getattr(detectors.anomaly_detector, 'model', None) is not None,
                        'object': detectors.object_detector.model is not None,
                    },
                    'warnings': (["Fire/smoke model unavailable or unsupported; those detections could not be verified"]
                                 if detectors.fire_detector.status != 'ready' else []) +
                                (["Dedicated violence model is missing; fights not reliably classified"]
                                 if detectors.violence_detector.model is None else []) +
                                (["CS2 firearm specialist is missing; stock COCO may miss guns and small knives"]
                                 if getattr(getattr(detectors, 'weapon_detector', None), 'model', None) is None else []) +
                                (["The CS2 firearm model is trained on game weapons, not real-life knives or CCTV footage"]
                                 if getattr(getattr(detectors, 'weapon_detector', None), 'model', None) is not None else []),
                    'model_labels': {
                        'violence': (detectors.violence_detector.model.names
                                     if detectors.violence_detector.model is not None else {}),
                        'weapon': (detectors.weapon_detector.model.names
                                   if getattr(detectors, 'weapon_detector', None) is not None
                                   and detectors.weapon_detector.model is not None else {}),
                    },
                    'model_sampling': {'fire_every_n_frames': 1, 'violence_every_n_frames': 1,
                                       'weapon_every_n_frames': 1, 'object_every_n_frames': 1},
                    'frame_limit_reached': frame_index == max_frames,
                    'events': dict(event_counts),
                    'confirmed_alert_types': sorted(emitted),
                    'confirmed_events': confirmed_records,
                    'early_warning_note': 'Smoke/heat indicators are treated as early-warning clues. This model pipeline does not predict ignition with certainty.',
                    'annotated_video_url': f'/api/test-videos/{token}/annotated'}

        try:
            result = await to_thread.run_sync(analyze)
            return result
        except HTTPException:
            raise
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            raise HTTPException(422, f'Video analysis failed: {str(exc)[:240]}')
        finally:
            input_path.unlink(missing_ok=True)

    @app.get('/api/test-videos/{token}/annotated')
    async def get_annotated_test_video(token: str, request: Request):
        user = require_user(request)
        if not re.fullmatch(r'[0-9a-f]{24}', token):
            raise HTTPException(404, 'Video not found')
        directory = settings.data_dir / 'test_videos'
        path = video_output(directory, token, user)
        if user['role'] == 'admin' and not path.exists():
            path = directory / f'{token}_annotated.mp4'  # Legacy unscoped output
        if not path.is_file():
            raise HTTPException(404, 'Annotated video not found')
        return FileResponse(path, media_type='video/mp4', filename='visionguard-test-annotated.mp4')

    @app.get('/api/test-videos/{token}/snapshots/{filename}')
    async def get_test_snapshot(token: str, filename: str, request: Request):
        """User-scoped confirmed incident photos from uploaded video tests."""
        user = require_user(request)
        if not re.fullmatch(r'[0-9a-f]{24}', token) or not re.fullmatch(r'[0-9a-f]{32}\.jpg', filename):
            raise HTTPException(404, 'Photo not found')
        directory = settings.data_dir / 'test_videos'
        prefix = video_output(directory, token, user).stem.removesuffix('_annotated')
        path = directory / f'{prefix}_{filename}'
        if not path.is_file():
            raise HTTPException(404, 'Photo not found')
        return FileResponse(path, media_type='image/jpeg', filename='incident.jpg')

    @app.get('/api/stats')
    async def stats(request: Request):
        user = require_user(request)
        cams = db.list_cameras()
        if user['role']=='client': cams=[c for c in cams if c.get('client_id')==user.get('client_id')]
        out = db.stats(client_id=user.get('client_id'))
        out.update(cameras=len(cams), online=sum(manager.telemetry(c['id'])['status']=='online' for c in cams),
                   current_people=sum(manager.telemetry(c['id'])['current_people'] for c in cams))
        return out

    @app.get('/api/events')
    async def events(request: Request, camera_id: str | None = None, kind: str | None = None, limit: int = 80):
        valid_kinds = ('intrusion', 'entry', 'exit', 'fire_detected', 'smoke', 'heat_source',
                       'violence', 'weapon', 'fight', 'assault', 'proximity_alert',
                       'fall_detected', 'loitering', 'unusual_activity', 'test')
        if kind not in (None, *valid_kinds):
            raise HTTPException(422, 'Unknown event type')
        if not 1 <= limit <= 500:
            raise HTTPException(422, 'Limit must be 1..500')
        user=require_user(request)
        records = db.events(camera_id=camera_id, kind=kind, limit=limit, client_id=user.get('client_id'))
        return records

    @app.get('/api/events.csv')
    async def export_csv(request: Request):
        user = require_user(request)
        records = db.events(limit=5000, client_id=user.get('client_id'))
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=['id', 'created_at', 'camera_id','camera_name','kind','track_id','message','snapshot','metadata'])
        writer.writeheader(); writer.writerows(records)
        return Response(buffer.getvalue(), media_type='text/csv',
                        headers={'Content-Disposition': 'attachment; filename="visionguard-events.csv"'})

    @app.get('/api/alert-log')
    async def alert_log(request: Request, event_id: str | None = None, limit: int = 100):
        user = require_user(request)
        if not 1 <= limit <= 500:
            raise HTTPException(422, 'Limit must be 1..500')
        return db.alert_log(event_id=event_id, limit=limit, client_id=user.get('client_id'))

    @app.get('/api/snapshots/{filename}')
    async def snapshot(filename: str, request: Request):
        user=require_user(request)
        if not re.fullmatch(r'[0-9a-f]{32}\.jpg', filename): raise HTTPException(404)
        path = settings.data_dir / 'snapshots' / filename
        if not path.is_file(): raise HTTPException(404)
        if user['role'] == 'client' and not db.client_owns_snapshot(user['client_id'], filename):
            raise HTTPException(404)
        return FileResponse(path, media_type='image/jpeg')

    @app.get('/stream/{camera_id}')
    async def stream(camera_id: str, request: Request):
        user = require_user(request)
        camera = db.get_camera(camera_id)
        if camera and user['role']=='client' and camera.get('client_id') != user.get('client_id'): raise HTTPException(404, 'Camera not found')
        if camera is None or not camera['enabled']:
            raise HTTPException(404, 'Camera not available')
        def frames():
            last_frame = None
            while True:
                image = manager.frame(camera_id)
                if image is None and last_frame is not None:
                    last_frame = None
                if image is not None and image is not last_frame:
                    last_frame = image
                    yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(image)).encode() + b'\r\n\r\n' + image + b'\r\n'
                elif image is None and last_frame is None:
                    # Keep MJPEG clients alive and show the real camera status while
                    # the source/model is starting or reconnecting.
                    import cv2, numpy as np
                    status = manager.telemetry(camera_id)
                    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(canvas, 'VISIONGUARD', (28, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (220,230,240), 2)
                    cv2.putText(canvas, str(status.get('status', 'connecting')).upper(), (28, 160), cv2.FONT_HERSHEY_SIMPLEX, .9, (80,210,190), 2)
                    detail = (status.get('error') or 'Waiting for the first camera frame')[:65]
                    cv2.putText(canvas, detail, (28, 215), cv2.FONT_HERSHEY_SIMPLEX, .48, (170,185,205), 1)
                    ok, encoded = cv2.imencode('.jpg', canvas)
                    if ok:
                        placeholder = encoded.tobytes()
                        yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(placeholder)).encode() + b'\r\n\r\n' + placeholder + b'\r\n'
                time_sleep = .12
                import time
                time.sleep(time_sleep)
        return StreamingResponse(frames(), media_type='multipart/x-mixed-replace; boundary=frame',
                                 headers={'Cache-Control':'no-cache, no-store', 'X-Content-Type-Options':'nosniff'})

    @app.websocket('/ws/alerts')
    async def alerts(websocket: WebSocket):
        user=session_user(websocket.cookies.get('vg_session'))
        if (admin_password or db.list_clients()) and not user:
            await websocket.close(code=1008); return
        user=user or {'role':'admin','client_id':None}
        if user.get('role')=='client':
            client=db.get_client(user.get('client_id'))
            if not client or not client.get('active'):
                await websocket.close(code=1008); return
        origin = websocket.headers.get('origin')
        if origin:
            o = urlsplit(origin)
            host = websocket.headers.get('host','')
            if origin not in settings.allowed_origins and o.netloc != host:
                await websocket.close(code=1008); return
        await websocket.accept()
        q = asyncio.Queue(maxsize=100)
        bus.clients.add(q)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=20)
                    if user['role']=='client':
                        camera_id = event.get('camera_id') or (event.get('event') or {}).get('camera_id')
                        cam = db.get_camera(camera_id) if camera_id else None
                        if not cam or cam.get('client_id') != user.get('client_id'): continue
                    await websocket.send_json(event)
                except asyncio.TimeoutError:
                    await websocket.send_json({'type':'heartbeat'})
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
        finally:
            bus.clients.discard(q)

    @app.post('/api/alerts/test')
    async def test_alerts(request: TestAlertsRequest):
        """Test all alert channels."""
        results = alert_manager.test_alerts()
        return {'results': results}

    @app.get('/api/detection-types')
    async def detection_types():
        """Get all supported detection event types."""
        return {
            'core': ['intrusion', 'entry', 'exit'],
            'fire': ['fire_detected', 'smoke'],
            'violence': ['violence', 'weapon', 'unusual_activity'],
            'anomaly': ['fall_detected', 'unusual_activity'],
            'severity_levels': ['critical', 'high', 'medium', 'low']
        }

    @app.get('/api/system/info')
    async def system_info():
        """Get system information."""
        try:
            import torch
            cuda_available=torch.cuda.is_available(); cuda_devices=torch.cuda.device_count() if cuda_available else 0
        except Exception:
            cuda_available=False; cuda_devices=0
        return {
            'version': '2.1.0',
            'cuda_available': cuda_available,
            'cuda_devices': cuda_devices,
            'models': {
                'person': settings.model_path,
                'fire': settings.fire_model_path,
                'violence': settings.violence_model_path,
                'anomaly': settings.anomaly_model_path,
                'object': settings.object_model_path
            },
            'alerts': {'tts_enabled': settings.tts_enabled}
        }

    @app.get('/api/v1/models')
    async def list_models():
        """OpenAI-compatible models list endpoint."""
        try:
            import torch
            cuda_available=torch.cuda.is_available(); cuda_devices=torch.cuda.device_count() if cuda_available else 0
        except Exception:
            cuda_available=False; cuda_devices=0
        return {
            'object': 'list',
            'data': [
                {'id': 'visionguard-yolov8n', 'object': 'model', 'owned_by': 'visionguard', 'permission': []},
                {'id': 'visionguard-fire', 'object': 'model', 'owned_by': 'visionguard', 'permission': []},
                {'id': 'visionguard-violence', 'object': 'model', 'owned_by': 'visionguard', 'permission': []},
                {'id': 'visionguard-anomaly', 'object': 'model', 'owned_by': 'visionguard', 'permission': []},
            ]
        }

    return app


async def _maintenance(db):
    while True:
        try:
            await asyncio.to_thread(db.purge_old)
        except Exception:
            LOGGER.exception('Retention maintenance failed')
        await asyncio.sleep(86400)


app = create_app()


def run():
    import uvicorn
    uvicorn.run('app.main:app', host=settings.host, port=settings.port, reload=False)


if __name__ == '__main__':
    run()