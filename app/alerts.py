"""Alert services: Telegram notifications and TTS voice sirens."""
import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import requests

from .config import settings
from .database import Database

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent


def get_urgent_speech_text(kind: str, camera_name: str = "", message: str = "") -> str:
    """Generate repeated urgency speech text matching user specification."""
    kind_lower = str(kind).strip().lower()
    cam_str = f"at {camera_name}" if camera_name else ""

    if 'fire' in kind_lower:
        return f"Fire! Fire! Fire! Fire detected {cam_str}."
    elif 'smoke' in kind_lower:
        return f"Fire! Smoke! Fire! Smoke detected {cam_str}."
    elif any(k in kind_lower for k in ('violence', 'fight', 'assault')):
        return f"Violence! Violence! Violence! Violence detected {cam_str}."
    elif any(k in kind_lower for k in ('weapon', 'knife', 'gun')):
        return f"Weapon! Weapon! Weapon! Weapon detected {cam_str}."
    elif 'intrusion' in kind_lower:
        return f"Intrusion! Intrusion! Intrusion! Zone intrusion {cam_str}."
    elif 'fall' in kind_lower:
        return f"Emergency! Fall! Fall! Fall detected {cam_str}."
    elif 'heat' in kind_lower:
        return f"Warning! Heat source! Fire hazard {cam_str}."
    else:
        word = kind_lower.replace('_', ' ').title()
        return f"Alert! Alert! Alert! {word} {cam_str}."


class SirenPlayer:
    """Plays the physical buzzer MP3 sound for 10 seconds on Windows using MCI."""

    def __init__(self, audio_path: Path | str | None = None):
        self.audio_path = Path(audio_path or (ROOT / 'connieconnie2-buzzer-227217.mp3')).resolve()
        self._lock = threading.Lock()
        self._current_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._current_alias: str | None = None

    def play_buzzer(self, duration_seconds: float = 10.0):
        """Play the buzzer sound in a background thread for duration_seconds."""
        if not self.audio_path.is_file():
            LOGGER.warning(f"Buzzer sound file not found at {self.audio_path}")
            return
        with self._lock:
            # Signal any existing playback to stop
            self._stop_event.set()
            if self._current_thread and self._current_thread.is_alive():
                self._current_thread.join(timeout=0.5)
            self._stop_event.clear()
            self._current_thread = threading.Thread(
                target=self._play_worker,
                args=(duration_seconds,),
                daemon=True,
                name="siren-buzzer-thread"
            )
            self._current_thread.start()

    def _play_worker(self, duration_seconds: float):
        import ctypes
        alias = f"buzzer_{uuid4().hex[:8]}"
        self._current_alias = alias
        mci = getattr(getattr(ctypes, 'windll', None), 'winmm', None)
        if not mci:
            LOGGER.warning("winmm.dll not available for MCI playback")
            return
        send_str = mci.mciSendStringW
        try:
            send_str(f'close {alias}', None, 0, 0)
            open_cmd = f'open "{self.audio_path}" type mpegvideo alias {alias}'
            err = send_str(open_cmd, None, 0, 0)
            if err != 0:
                LOGGER.warning(f"MCI open failed with code {err}")
                return
            send_str(f'play {alias} repeat', None, 0, 0)
            LOGGER.info(f"Buzzer siren started for {duration_seconds}s")
            start = time.monotonic()
            while time.monotonic() - start < duration_seconds and not self._stop_event.is_set():
                time.sleep(0.1)
        except Exception as e:
            LOGGER.error(f"Error playing buzzer siren: {e}")
        finally:
            try:
                send_str(f'stop {alias}', None, 0, 0)
                send_str(f'close {alias}', None, 0, 0)
                LOGGER.info("Buzzer siren stopped and closed cleanly")
            except Exception:
                pass
            if self._current_alias == alias:
                self._current_alias = None

    def stop(self):
        self._stop_event.set()


SUSPICIOUS_TELEGRAM_KINDS = {
    'fire_detected', 'smoke', 'violence', 'weapon', 'fall_detected', 'unusual_activity',
    'fight', 'knife', 'gun', 'assault',  # supported aliases from specialist models
}


def telegram_event_eligible(kind: str, confidence: Any, minimum: float = 0.50) -> bool:
    """Notify for allow-listed security events at or above the configured confidence floor."""
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return False
    normalized = str(kind).strip().lower()
    return normalized in SUSPICIOUS_TELEGRAM_KINDS and score >= minimum


@dataclass
class AlertEvent:
    event_id: str
    camera_id: str
    camera_name: str
    kind: str
    severity: str
    message: str
    snapshot: np.ndarray | None = None
    metadata: dict | None = None
    timestamp: str = ""


class TelegramAlerter:
    """Send alert notifications to Telegram."""
    
    def __init__(self, db: Database):
        self.db = db
        self.bot_token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self.enabled = bool(self.bot_token and self.chat_id)
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'VisionGuard/1.0'})
    
    def send_alert(self, event: AlertEvent) -> bool:
        """Send alert to Telegram with optional image."""
        if not self.enabled:
            return False
        
        if str(event.kind).strip().lower() not in settings.telegram_alert_types:
            return False
        
        # Format message
        severity_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
        emoji = severity_emoji.get(event.severity, '⚠️')
        
        # Event storage is UTC; render alert timestamps in the configured local timezone.
        display_timestamp = event.timestamp
        try:
            from datetime import datetime, timezone
            from zoneinfo import ZoneInfo
            parsed = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            display_timestamp = parsed.astimezone(ZoneInfo(settings.app_timezone)).strftime('%Y-%m-%d %H:%M:%S %Z')
        except Exception:
            pass
        caption = (
            f"{emoji} <b>{event.kind.upper()}</b> - {event.severity.upper()}\n"
            f"📷 Camera: {event.camera_name}\n"
            f"📍 {event.message}\n"
            f"🎯 Confidence: {float((event.metadata or {}).get('confidence', 0)):.0%}\n"
            f"⏰ {display_timestamp}"
        )
        
        success = False
        try:
            if event.snapshot is not None:
                # Encode image
                _, img_encoded = cv2.imencode('.jpg', event.snapshot, [cv2.IMWRITE_JPEG_QUALITY, 80])
                files = {'photo': ('alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
                data = {'chat_id': self.chat_id, 'caption': caption, 'parse_mode': 'HTML'}
                resp = self._session.post(
                    f'https://api.telegram.org/bot{self.bot_token}/sendPhoto',
                    data=data, files=files, timeout=10
                )
            else:
                data = {'chat_id': self.chat_id, 'text': caption, 'parse_mode': 'HTML'}
                resp = self._session.post(
                    f'https://api.telegram.org/bot{self.bot_token}/sendMessage',
                    data=data, timeout=10
                )
            
            if resp.status_code == 200:
                success = True
                LOGGER.info(f"Telegram alert sent for {event.kind}")
            else:
                LOGGER.error(f"Telegram API error: {resp.status_code} - {resp.text}")
        except Exception as e:
            LOGGER.error(f"Failed to send Telegram alert: {e}")
        
        # Log alert attempt
        self.db.log_alert(event.event_id, 'telegram', 'sent' if success else 'failed',
                         f"Telegram {'sent' if success else 'failed'}: {event.kind}")
        return success
    
    def send_test_message(self) -> bool:
        """Send a test message to verify configuration."""
        if not self.enabled:
            return False
        try:
            data = {'chat_id': self.chat_id, 'text': '✅ VisionGuard Telegram alerts configured successfully!', 'parse_mode': 'HTML'}
            resp = self._session.post(
                f'https://api.telegram.org/bot{self.bot_token}/sendMessage',
                data=data, timeout=10
            )
            return resp.status_code == 200
        except Exception as e:
            LOGGER.error(f"Telegram test failed: {e}")
            return False


class TTSAlerter:
    """Text-to-speech voice alerts using pyttsx3 (offline Windows SAPI)."""
    
    def __init__(self):
        self.enabled = settings.tts_enabled
        self.engine = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._queue: list[tuple[str, int]] = []  # (message, repeat_count)
        self._init_engine()
    
    def _init_engine(self):
        """Initialize pyttsx3 engine."""
        if not self.enabled:
            return
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', settings.tts_voice_rate)
            self.engine.setProperty('volume', settings.tts_voice_volume)
            
            # Try to set a good voice
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower() or 'hazel' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
            
            LOGGER.info("TTS engine initialized successfully")
        except Exception as e:
            LOGGER.error(f"Failed to initialize TTS engine: {e}")
            self.enabled = False
    
    def speak(self, message: str, repeat: int = 1):
        """Queue a message to be spoken."""
        if not self.enabled or not self.engine:
            return
        with self._lock:
            self._queue.append((message, repeat))
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
    
    def _run(self):
        """Process speech queue."""
        import pyttsx3
        while not self._stop_event.is_set():
            with self._lock:
                if not self._queue:
                    break
                message, repeat = self._queue.pop(0)
            
            for _ in range(repeat):
                if self._stop_event.is_set():
                    break
                try:
                    self.engine.say(message)
                    self.engine.runAndWait()
                    time.sleep(0.5)  # Brief pause between repeats
                except Exception as e:
                    LOGGER.error(f"TTS error: {e}")
                    break
    
    def trigger_alert(self, event: AlertEvent):
        """Trigger voice alert for an event."""
        if not self.enabled or event.kind not in settings.tts_alert_types:
            return
        
        message = get_urgent_speech_text(event.kind, event.camera_name, event.message)
        self.speak(message, max(1, settings.siren_repeat_count))
    
    def stop(self):
        """Stop current speech."""
        self._stop_event.set()
        if self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass
    
    def test_speak(self):
        """Test TTS with a sample urgent alert message."""
        if self.enabled:
            self.speak("Fire! Fire! Fire! Violence! Violence! Weapon! Weapon! VisionGuard voice alert system is ready.", 1)


class AlertManager:
    """Coordinates all alert channels."""
    
    def __init__(self, db: Database):
        self.db = db
        self.telegram = TelegramAlerter(db)
        self.tts = TTSAlerter()
        self.siren = SirenPlayer()
        self._last_alert_time: dict[str, float] = {}
    
    def process_event(self, event: AlertEvent, camera_config: dict):
        """Process event through all enabled alert channels."""
        event_id = event.event_id
        
        # Telegram is only for configured confirmed security event classes, per-camera opt-in,
        # and a score >= 50%. Analytics events without a model
        # confidence score are deliberately excluded from Telegram notifications.
        confidence = None
        if isinstance(event.metadata, dict):
            confidence = event.metadata.get('confidence')
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        kind = str(event.kind).strip().lower()
        telegram_enabled_for_camera = bool(camera_config.get('telegram_enabled', 0))
        if (self.telegram and self.telegram.enabled and telegram_enabled_for_camera
                and kind in settings.telegram_alert_types
                and telegram_event_eligible(kind, confidence, settings.telegram_min_confidence)):
            if self._check_cooldown(f'telegram:{event.camera_id}:{kind}', settings.telegram_cooldown_seconds):
                self.telegram.send_alert(event)

        # Siren buzzer sound (10s) and local TTS urgent announcement
        siren_eligible_kinds = {
            'fire', 'fire_detected', 'smoke', 'heat_source',
            'violence', 'weapon', 'fight', 'knife', 'gun', 'assault',
            'intrusion', 'fall_detected', 'unusual_activity'
        }
        is_siren = (kind in siren_eligible_kinds or
                    any(k in kind for k in ('fire', 'smoke', 'viol', 'weap', 'fight', 'knife', 'gun', 'intr')))

        if camera_config.get('tts_enabled', 1):
            if is_siren and self._check_cooldown(f'siren:{event.camera_id}:{kind}', 10):
                self.siren.play_buzzer(10.0)

            if self.tts.enabled and self._check_cooldown(f'tts:{event.camera_id}:{kind}', 10):
                self.tts.trigger_alert(event)
                self.db.log_alert(event_id, 'tts', 'triggered', f'TTS alert: {event.kind}')
    
    def _check_cooldown(self, kind: str, cooldown: int) -> bool:
        """Check if enough time has passed since last alert of this kind."""
        now = time.monotonic()
        last = self._last_alert_time.get(kind, float('-inf'))
        if now - last >= cooldown:
            self._last_alert_time[kind] = now
            return True
        return False
    
    def test_alerts(self) -> dict[str, bool]:
        """Test all alert channels."""
        results = {}
        # Trigger 10-second buzzer sound
        self.siren.play_buzzer(10.0)
        results['buzzer'] = True
        
        if self.tts.enabled:
            self.tts.test_speak()
            results['tts'] = True
        else:
            results['tts'] = False
        results['telegram'] = bool(self.telegram and self.telegram.send_test_message())
        return results