"""On-demand camera video recording engine with asynchronous writer queues."""
from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from .config import settings
from .database import Database

LOGGER = logging.getLogger(__name__)


class _RecordingSession:
    """Manages the background VideoWriter worker for one camera."""

    def __init__(
        self,
        recording_id: str,
        camera_id: str,
        camera_name: str,
        file_path: Path,
        width: int,
        height: int,
        fps: float,
    ):
        self.recording_id = recording_id
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.file_path = file_path
        self.filename = file_path.name
        self.width = width
        self.height = height
        self.fps = max(1.0, float(fps))
        self.start_monotonic = time.monotonic()
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.frames_written = 0

        self.queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=120)
        self.stop_event = threading.Event()

        # Initialize VideoWriter (mp4v codec for cross-platform MP4 support)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(self.file_path), fourcc, self.fps, (self.width, self.height))
        if not self.writer.isOpened():
            # Fallback to XVID if mp4v fails
            fallback_fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(str(self.file_path), fallback_fourcc, self.fps, (self.width, self.height))

        self.worker_thread = threading.Thread(
            target=self._write_loop,
            name=f'recorder-{self.camera_id}',
            daemon=True,
        )
        self.worker_thread.start()

    def _write_loop(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is None:
                break

            try:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                self.writer.write(frame)
                self.frames_written += 1
            except Exception as exc:
                LOGGER.error("Error writing frame to recording %s: %s", self.filename, exc)
            finally:
                self.queue.task_done()

        try:
            self.writer.release()
        except Exception:
            pass

    def enqueue_frame(self, frame: np.ndarray):
        if self.stop_event.is_set():
            return
        try:
            # If queue is full, drop older frames to maintain real-time performance
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    pass
            self.queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except (queue.Full, Exception):
            pass

        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)

        duration = max(0.1, time.monotonic() - self.start_monotonic)
        end_time = datetime.now(timezone.utc).isoformat()
        file_size = self.file_path.stat().st_size if self.file_path.is_file() else 0

        return {
            'id': self.recording_id,
            'camera_id': self.camera_id,
            'camera_name': self.camera_name,
            'filename': self.filename,
            'file_path': str(self.file_path),
            'start_time': self.start_time,
            'end_time': end_time,
            'duration_seconds': round(duration, 1),
            'file_size_bytes': file_size,
            'frames_written': self.frames_written,
            'created_at': self.start_time,
        }


class VideoRecorder:
    """Manages active camera recording sessions and persists metadata to SQLite."""

    def __init__(self, output_dir: Path | None = None, db: Database | None = None):
        self.output_dir = Path(output_dir or (settings.data_dir / 'recordings')).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db = db
        self.lock = threading.RLock()
        self.sessions: dict[str, _RecordingSession] = {}

    def start_recording(
        self,
        camera_id: str,
        camera_name: str,
        width: int = 640,
        height: int = 480,
        fps: float = 20.0,
    ) -> dict[str, Any]:
        with self.lock:
            existing = self.sessions.get(camera_id)
            if existing is not None:
                return {
                    'id': existing.recording_id,
                    'camera_id': existing.camera_id,
                    'camera_name': existing.camera_name,
                    'filename': existing.filename,
                    'is_recording': True,
                    'duration_seconds': round(time.monotonic() - existing.start_monotonic, 1),
                }

            rec_id = uuid4().hex
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_cam_id = "".join(c for c in camera_id if c.isalnum() or c in ('-', '_'))
            filename = f"rec_{safe_cam_id}_{timestamp_str}_{rec_id[:6]}.mp4"
            file_path = self.output_dir / filename

            session = _RecordingSession(
                recording_id=rec_id,
                camera_id=camera_id,
                camera_name=camera_name,
                file_path=file_path,
                width=width,
                height=height,
                fps=fps,
            )
            self.sessions[camera_id] = session

            LOGGER.info("Started video recording for camera %s -> %s", camera_id, filename)
            return {
                'id': rec_id,
                'camera_id': camera_id,
                'camera_name': camera_name,
                'filename': filename,
                'file_path': str(file_path),
                'is_recording': True,
                'start_time': session.start_time,
                'duration_seconds': 0.0,
            }

    def write_frame(self, camera_id: str, frame: np.ndarray):
        session = self.sessions.get(camera_id)
        if session is not None and frame is not None:
            session.enqueue_frame(frame)

    def stop_recording(self, camera_id: str) -> dict[str, Any] | None:
        with self.lock:
            session = self.sessions.pop(camera_id, None)

        if session is None:
            return None

        record_data = session.stop()
        LOGGER.info(
            "Stopped video recording for camera %s: %s (%s frames, %ss, %d bytes)",
            camera_id,
            record_data['filename'],
            record_data['frames_written'],
            record_data['duration_seconds'],
            record_data['file_size_bytes'],
        )

        if self.db is not None:
            try:
                saved = self.db.store_recording(record_data)
                return saved
            except Exception as exc:
                LOGGER.error("Failed to store recording metadata: %s", exc)

        return record_data

    def is_recording(self, camera_id: str) -> bool:
        return camera_id in self.sessions

    def get_status(self, camera_id: str) -> dict[str, Any]:
        session = self.sessions.get(camera_id)
        if session is None:
            return {'is_recording': False, 'camera_id': camera_id}
        duration = round(time.monotonic() - session.start_monotonic, 1)
        return {
            'is_recording': True,
            'camera_id': camera_id,
            'recording_id': session.recording_id,
            'filename': session.filename,
            'duration_seconds': duration,
            'frames_written': session.frames_written,
        }

    def stop_all(self):
        with self.lock:
            cam_ids = list(self.sessions.keys())
        for cid in cam_ids:
            self.stop_recording(cid)
