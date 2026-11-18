"""Environment configuration without heavyweight optional imports."""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')


def get_int(name: str, default: int, minimum: int = 0) -> int:
    val = int(os.getenv(name, str(default)))
    if val < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return val


def get_float(name: str, default: float, minimum: float = 0.0) -> float:
    val = float(os.getenv(name, str(default)))
    if val < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return val


@dataclass(frozen=True)
class Settings:
    # Server
    host: str = os.getenv('HOST', '127.0.0.1')
    port: int = get_int('PORT', 8000, 1)
    
    # Core YOLO (person detection + tracking)
    model_path: str = os.getenv('MODEL_PATH', 'yolov8n.pt')
    # auto | cpu | 0 (first CUDA GPU). Defaults to auto for GPU acceleration.
    device: str = os.getenv('DEVICE', 'auto')
    confidence: float = max(0.40, get_float('CONFIDENCE', 0.40, 0.0))
    detect_class_ids: tuple[int, ...] = tuple(int(n.strip()) for n in os.getenv('DETECT_CLASS_IDS', '0').split(',') if n.strip())
    imgsz: int = get_int('IMGSZ', 480, 128)  # Reduced from 640 for faster inference
    max_frame_width: int = get_int('MAX_FRAME_WIDTH', 640, 320)  # Reduced from 960
    frame_skip: int = get_int('FRAME_SKIP', 1, 1)  # Process every frame (was 2)
    jpeg_quality: int = get_int('JPEG_QUALITY', 70, 30)
    snapshot_retention_days: int = get_int('SNAPSHOT_RETENTION_DAYS', 30, 1)
    event_cooldown_seconds: int = get_int('EVENT_COOLDOWN_SECONDS', 20, 0)
    
    # Fire Detection - optimized for small flames (matchstick, candle)
    fire_model_path: str = os.getenv('FIRE_MODEL_PATH', 'models/yolov8n-fire.pt')
    fire_confidence: float = max(0.30, get_float('FIRE_CONFIDENCE', 0.30, 0.0))  # Lowered from 0.50
    fire_classes: tuple[int, ...] = tuple(int(n.strip()) for n in os.getenv('FIRE_CLASSES', '0,1').split(',') if n.strip())
    
    # Violence/Fight Detection
    violence_model_path: str = os.getenv('VIOLENCE_MODEL_PATH', 'models/yolov8n-violence.pt')
    violence_confidence: float = max(0.35, get_float('VIOLENCE_CONFIDENCE', 0.35, 0.0))  # Lowered from 0.50
    
    # Anomaly/Unusual Activity Detection - uses YOLO-World for open-vocabulary (knife, gun, etc.)
    anomaly_model_path: str = os.getenv('ANOMALY_MODEL_PATH', 'models/yolov8s-world.pt')  # Changed from pose to YOLO-World
    anomaly_confidence: float = max(0.30, get_float('ANOMALY_CONFIDENCE', 0.30, 0.0))  # Lowered from 0.50
    enable_fall_detection: bool = os.getenv('ENABLE_FALL_DETECTION', 'true').lower() == 'true'
    enable_loitering_detection: bool = os.getenv('ENABLE_LOITERING_DETECTION', 'true').lower() == 'true'
    loitering_time_seconds: int = get_int('LOITERING_TIME_SECONDS', 30, 5)
    # YOLO-World open vocabulary classes - includes weapons, fire, violence
    yolo_world_classes: tuple[str, ...] = tuple(x.strip() for x in os.getenv('YOLO_WORLD_CLASSES', 'fire,smoke,person fighting,knife,gun,weapon,pistol,rifle,blade,person falling').split(',') if x.strip())
    
    # General Object Detection (COCO classes)
    object_model_path: str = os.getenv('OBJECT_MODEL_PATH', 'yolov8n.pt')
    object_confidence: float = max(0.40, get_float('OBJECT_CONFIDENCE', 0.40, 0.0))
    object_classes: tuple[int, ...] = tuple(int(n.strip()) for n in os.getenv('OBJECT_CLASSES', '').split(',') if n.strip())
    
    # Telegram Alerts
    telegram_bot_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '').strip()
    telegram_alert_types: tuple[str, ...] = tuple(x.strip().lower() for x in os.getenv('TELEGRAM_ALERT_TYPES', 'intrusion,fire,fire_detected,smoke,violence,weapon,knife,gun,fight,assault,fall_detected,loitering,proximity_alert,heat_source').split(',') if x.strip())
    telegram_min_confidence: float = max(0.75, min(1.0, get_float('TELEGRAM_MIN_CONFIDENCE', 0.75, 0.0)))
    telegram_cooldown_seconds: int = get_int('TELEGRAM_COOLDOWN_SECONDS', 60, 0)
    
    # TTS Siren/Voice Alerts
    tts_enabled: bool = os.getenv('TTS_ENABLED', 'true').lower() == 'true'
    tts_voice_rate: int = get_int('TTS_VOICE_RATE', 180, 50)
    tts_voice_volume: float = get_float('TTS_VOICE_VOLUME', 1.0, 0.0)
    tts_alert_types: tuple[str, ...] = tuple(x.strip() for x in os.getenv('TTS_ALERT_TYPES', 'fire,smoke,heat_source,violence,weapon,intrusion,fall_detected').split(',') if x.strip())
    siren_duration_seconds: int = get_int('SIREN_DURATION_SECONDS', 10, 1)
    siren_repeat_count: int = get_int('SIREN_REPEAT_COUNT', 3, 1)
    
    # Dashboard
    dashboard_password: str = os.getenv('DASHBOARD_PASSWORD', '').strip()
    admin_username: str = os.getenv('ADMIN_USERNAME', 'admin').strip()
    admin_password: str = os.getenv('ADMIN_PASSWORD', '').strip()
    session_secret: str = os.getenv('SESSION_SECRET', 'change-this-visionguard-secret').strip()
    app_timezone: str = os.getenv('APP_TIMEZONE', 'Asia/Kolkata').strip()
    allowed_origins: tuple[str, ...] = tuple(x.strip() for x in os.getenv('ALLOWED_ORIGINS', '').split(',') if x.strip())
    
    # Paths
    data_dir: Path = ROOT / 'data'
    static_dir: Path = ROOT / 'app' / 'static'
    models_dir: Path = ROOT / 'models'


settings = Settings()