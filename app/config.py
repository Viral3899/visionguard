"""Unified Environment Configuration - combines FalsePositive_Fix strictness + RTSP_Surveillance features."""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')


def local_model_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    for candidate in (ROOT / path, ROOT / 'models' / path):
        if candidate.is_file():
            return str(candidate)
    return value


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
    # Limit concurrently loaded camera workers to avoid silent GPU OOM.
    max_active_cameras: int = get_int('MAX_ACTIVE_CAMERAS', 4, 1)
    inference_parallelism: int = get_int('INFERENCE_PARALLELISM', 1, 1)
    share_specialist_models: bool = os.getenv('SHARE_SPECIALIST_MODELS', 'true').lower() == 'true'

    # Core YOLO (person detection + tracking)
    model_path: str = os.getenv('MODEL_PATH', 'yolov8n.pt')
    # auto | cpu | 0 (first CUDA GPU). Defaults to auto for GPU acceleration.
    device: str = os.getenv('DEVICE', 'auto')
    confidence: float = max(0.40, get_float('CONFIDENCE', 0.40, 0.0))
    detect_class_ids: tuple[int, ...] = tuple(int(n.strip()) for n in os.getenv('DETECT_CLASS_IDS', '0').split(',') if n.strip())
    imgsz: int = get_int('IMGSZ', 416, 128)  # Reduced from 640 for faster inference
    max_frame_width: int = get_int('MAX_FRAME_WIDTH', 960, 320)  # Reduced from 960
    frame_skip: int = get_int('FRAME_SKIP', 1, 1)  # Process every frame (was 2)
    jpeg_quality: int = get_int('JPEG_QUALITY', 70, 30)
    snapshot_retention_days: int = get_int('SNAPSHOT_RETENTION_DAYS', 30, 1)
    event_cooldown_seconds: int = get_int('EVENT_COOLDOWN_SECONDS', 20, 0)
    # Only verified threat incidents produce stills and Telegram alerts.
    incident_min_confidence: float = max(0.50, get_float('INCIDENT_MIN_CONFIDENCE', 0.50))
    incident_confirmation_frames: int = get_int('INCIDENT_CONFIRMATION_FRAMES', 2, 1)

    # Fire Detection - optimized for small flames (matchstick, candle)
    fire_model_path: str = os.getenv('FIRE_MODEL_PATH', 'models/yolov26-fire-smoke.pt')
    fire_confidence: float = max(0.30, get_float('FIRE_CONFIDENCE', 0.30, 0.0))  # Lowered from 0.50
    fire_imgsz: int = get_int('FIRE_IMGSZ', 768, 128)
    fire_alert_confidence: float = get_float('FIRE_ALERT_MIN_SCORE', 0.80, 0.0)
    small_flame_alert_confidence: float = get_float('SMALL_FLAME_ALERT_MIN_SCORE', 0.75, 0.0)
    fire_classes: tuple[int, ...] = tuple(int(n.strip()) for n in os.getenv('FIRE_CLASSES', '0,1').split(',') if n.strip())

    # Violence/Fight Detection
    violence_model_path: str = os.getenv('VIOLENCE_MODEL_PATH', 'models/fight-yolov8-nano.pt')
    violence_confidence: float = max(0.20, get_float('VIOLENCE_CONFIDENCE', 0.30, 0.0))
    violence_imgsz: int = get_int('VIOLENCE_IMGSZ', 768, 128)
    violence_every_n_frames: int = get_int('VIOLENCE_EVERY_N_FRAMES', 1, 1)

    # Weapon/Threat Detection (Subh775/Threat-Detection-YOLOv8n)
    weapon_model_path: str = os.getenv('WEAPON_MODEL_PATH', 'models/threat-detection-yolov8n.pt')
    weapon_confidence: float = max(0.20, get_float('WEAPON_CONFIDENCE', 0.30, 0.0))
    weapon_imgsz: int = get_int('WEAPON_IMGSZ', 960, 128)
    weapon_every_n_frames: int = get_int('WEAPON_EVERY_N_FRAMES', 1, 1)

    # Optional trained weapon model (NOT included in the stock YOLO person model).
    trained_weapon_model_path: str = os.getenv('TRAINED_WEAPON_MODEL_PATH', 'models/visionguard-weapon-trained.pt')
    enable_trained_weapon: bool = os.getenv('ENABLE_TRAINED_WEAPON_MODEL', 'false').lower() == 'true'

    # Knife Detection (specialized model from syzenzz/cctv-knife-detection-dataset)
    knife_model_path: str = os.getenv('KNIFE_MODEL_PATH', 'models/cctv-knife-detection.pt')
    knife_confidence: float = max(0.25, get_float('KNIFE_CONFIDENCE', 0.30, 0.0))
    knife_imgsz: int = get_int('KNIFE_IMGSZ', 768, 128)
    knife_every_n_frames: int = get_int('KNIFE_EVERY_N_FRAMES', 1, 1)

    # Optional R3D-18 fight classifier (TorchVision, video clip classification)
    enable_fight_classifier: bool = os.getenv('ENABLE_FIGHT_CLASSIFIER', 'false').strip().lower() == 'true'
    fight_classifier_model_path: str = os.getenv('FIGHT_CLASSIFIER_MODEL_PATH', 'models/visionguard-r3d18-fight.pt')
    fight_classifier_confidence: float = get_float('FIGHT_CLASSIFIER_CONFIDENCE', 0.85, 0.0)
    fight_classifier_interval: int = get_int('FIGHT_CLASSIFIER_INTERVAL', 8, 1)

    # Optional legacy 3DCNN (TensorFlow/Keras) - opt-in only
    enable_3dcnn: bool = os.getenv('ENABLE_3DCNN', 'false').strip().lower() == 'true'
    cnn3d_model_path: str = os.getenv('CNN3D_MODEL_PATH', 'models/fight_detection_model.h5')
    cnn3d_positive_class: str = os.getenv('CNN3D_POSITIVE_CLASS', '').strip()
    cnn3d_confidence: float = get_float('CNN3D_CONFIDENCE', 0.75, 0.0)
    cnn3d_every_n_frames: int = get_int('CNN3D_EVERY_N_FRAMES', 8, 1)

    # Precision-first weapon policy (separate from permissive YOLO candidate thresholds).
    strict_weapon_policy: bool = os.getenv('STRICT_WEAPON_POLICY', 'true').lower() == 'true'
    weapon_require_person: bool = os.getenv('WEAPON_REQUIRE_PERSON', 'true').lower() == 'true'
    weapon_person_margin: float = get_float('WEAPON_PERSON_MARGIN', 0.08, 0.0)
    weapon_gun_min_score: float = get_float('WEAPON_GUN_MIN_SCORE', 0.85, 0.0)
    weapon_knife_min_score: float = get_float('WEAPON_KNIFE_MIN_SCORE', 0.75, 0.0)
    weapon_other_min_score: float = get_float('WEAPON_OTHER_MIN_SCORE', 0.90, 0.0)
    weapon_gun_max_area: float = get_float('WEAPON_GUN_MAX_AREA', 0.15, 0.0)
    weapon_knife_max_area: float = get_float('WEAPON_KNIFE_MAX_AREA', 0.10, 0.0)
    weapon_other_max_area: float = get_float('WEAPON_OTHER_MAX_AREA', 0.12, 0.0)

    # Anomaly/Unusual Activity Detection - uses YOLOv8-Pose or YOLO-World for open-vocabulary
    anomaly_model_path: str = os.getenv('ANOMALY_MODEL_PATH', 'yolov8n-pose.pt')
    anomaly_confidence: float = max(0.30, get_float('ANOMALY_CONFIDENCE', 0.30, 0.0))  # Lowered from 0.50
    enable_fall_detection: bool = os.getenv('ENABLE_FALL_DETECTION', 'true').lower() == 'true'
    enable_loitering_detection: bool = os.getenv('ENABLE_LOITERING_DETECTION', 'false').lower() == 'true'
    loitering_time_seconds: int = get_int('LOITERING_TIME_SECONDS', 30, 5)
    # YOLO-World open vocabulary classes - includes weapons, fire, violence, falls
    yolo_world_classes: tuple[str, ...] = tuple(x.strip() for x in os.getenv('YOLO_WORLD_CLASSES', 'fire,smoke,person fighting,knife,a person holding a knife,gun,a person holding a gun,weapon,pistol,rifle,blade,person falling').split(',') if x.strip())

    # General Object Detection (COCO classes - all daily life objects by default)
    object_model_path: str = os.getenv('OBJECT_MODEL_PATH', 'yolov8n.pt')
    object_confidence: float = max(0.40, get_float('OBJECT_CONFIDENCE', 0.40, 0.0))
    object_imgsz: int = get_int('OBJECT_IMGSZ', 640, 128)
    object_every_n_frames: int = get_int('OBJECT_EVERY_N_FRAMES', 1, 1)
    knife_tile_scan: bool = os.getenv('KNIFE_TILE_SCAN', 'false').lower() == 'true'
    object_classes: tuple[int, ...] = tuple(int(n.strip()) for n in os.getenv('OBJECT_CLASSES', '').split(',') if n.strip())

    # Telegram Alerts
    telegram_bot_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '').strip()
    telegram_alert_types: tuple[str, ...] = tuple(x.strip().lower() for x in os.getenv('TELEGRAM_ALERT_TYPES', 'fire_detected,smoke,violence,weapon,fall_detected,unusual_activity').split(',') if x.strip())
    telegram_min_confidence: float = max(0.50, get_float('TELEGRAM_MIN_CONFIDENCE', 0.50))
    telegram_cooldown_seconds: int = get_int('TELEGRAM_COOLDOWN_SECONDS', 60, 0)

    # TTS Siren/Voice Alerts (Offline Windows TTS)
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