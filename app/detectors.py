"""Unified Multi-model Detection: Fire, Violence, Weapon, Knife, Anomaly (Pose/Fall/Loitering), Objects.
Combines: FalsePositive_Fix (strict policy, multi-frame confirmation, ThreatPolicy, R3D-18) + 
RTSP_Surveillance (YOLO-World, KnifeDetector, Telegram/TTS alerts)"""
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .config import local_model_path, settings
from .security_labels import is_fight_label, weapon_class
from .threat_policy import ThreatPolicy

LOGGER = logging.getLogger(__name__)


CLASS_DISPLAY_NAMES = {
    'cell phone': 'MOBILE',
    'cellphone': 'MOBILE',
    'mobile phone': 'MOBILE',
    'phone': 'MOBILE',
    'mobile': 'MOBILE',
    'chair': 'CHAIR',
    'powerbank': 'POWERBANK',
    'power bank': 'POWERBANK',
    'battery': 'POWERBANK',
    'laptop': 'LAPTOP',
    'bottle': 'BOTTLE',
    'backpack': 'BACKPACK',
    'handbag': 'BAG',
    'suitcase': 'SUITCASE',
    'book': 'BOOK',
    'tv': 'TV / MONITOR',
    'tvmonitor': 'TV / MONITOR',
    'monitor': 'MONITOR',
    'mouse': 'MOUSE',
    'keyboard': 'KEYBOARD',
    'remote': 'REMOTE / GADGET',
    'cup': 'CUP',
    'fork': 'FORK',
    'spoon': 'SPOON',
    'bowl': 'BOWL',
    'clock': 'CLOCK',
    'vase': 'VASE',
    'scissors': 'SCISSORS',
    'dining table': 'TABLE',
    'couch': 'SOFA / COUCH',
    'bed': 'BED',
    'microwave': 'MICROWAVE',
    'oven': 'OVEN',
    'toaster': 'TOASTER',
    'sink': 'SINK',
    'refrigerator': 'FRIDGE',
}


def format_daily_object_label(class_name: str, conf: float) -> str:
    """Format object name and explicit CF score (e.g. 'MOBILE | CF: 92%')."""
    cleaned = str(class_name).strip().lower()
    display_name = CLASS_DISPLAY_NAMES.get(cleaned, cleaned.upper())
    return f"{display_name} | CF: {float(conf):.0%}"


def draw_security_bounding_boxes(
    frame: np.ndarray,
    events: list[dict],
    persons: list[dict] | None = None
) -> np.ndarray:
    """Render high-visibility bounding boxes and labels for weapons and irregular activities.

    Highlights weapons (guns, knives) in neon magenta/red, and irregular activities
    (violence, fights, falls, loitering, unusual activity, intrusions) in orange/amber/yellow.
    """
    import math
    if frame is None or not events:
        return frame
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return frame

    person_by_id = {}
    if persons:
        for p in persons:
            if isinstance(p, dict) and 'id' in p and 'bbox' in p:
                person_by_id[int(p['id'])] = p['bbox']

    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get('type', event.get('kind', ''))).lower()
        subtype = str(event.get('subtype', '')).lower()
        confidence = event.get('confidence')

        is_weapon = (
            event_type == 'weapon' or
            subtype in ('knife', 'blade', 'gun', 'pistol', 'rifle', 'weapon')
        )
        is_violence = event_type in ('violence', 'fight', 'assault') or 'fight' in subtype or 'violence' in subtype
        is_fall = event_type in ('fall_detected', 'fall') or 'fall' in subtype
        is_loitering = event_type == 'loitering' or 'loitering' in subtype
        is_unusual = event_type in ('unusual_activity', 'abnormal_activity') or 'unusual' in subtype
        is_intrusion = event_type in ('intrusion', 'restricted') or 'intrusion' in subtype
        is_fire = event_type in ('fire_detected', 'fire') or subtype in ('fire', 'flame')
        is_smoke = event_type in ('smoke',)

        if not (is_weapon or is_violence or is_fall or is_loitering or is_unusual or is_intrusion or is_fire or is_smoke):
            continue

        bbox = event.get('bbox')
        if not bbox and event.get('track_id') is not None:
            track_id = int(event['track_id'])
            bbox = person_by_id.get(track_id)

        if not bbox or len(bbox) != 4:
            continue

        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if max(x1, y1, x2, y2) <= 1.05 and (x2 <= 1.0 or y2 <= 1.0):
                px1 = int(round(x1 * w))
                py1 = int(round(y1 * h))
                px2 = int(round(x2 * w))
                py2 = int(round(y2 * h))
            else:
                px1 = int(round(x1))
                py1 = int(round(y1))
                px2 = int(round(x2))
                py2 = int(round(y2))

            px1, px2 = sorted((max(0, min(w - 1, px1)), max(0, min(w - 1, px2))))
            py1, py2 = sorted((max(0, min(h - 1, py1)), max(0, min(h - 1, py2))))
            if px2 <= px1 or py2 <= py1:
                continue

            conf_str = f" {float(confidence):.0%}" if confidence is not None and math.isfinite(float(confidence)) else ""

            if is_weapon:
                color = (255, 0, 255) if 'knife' in subtype or 'blade' in subtype else (0, 0, 255)
                sub_label = subtype if subtype else 'weapon'
                label = f"[!] WEAPON: {sub_label.upper()}{conf_str}"
            elif is_violence:
                color = (0, 69, 255)
                sub_label = subtype if subtype else 'violence'
                label = f"[!] VIOLENCE: {sub_label.upper()}{conf_str}"
            elif is_fall:
                color = (0, 215, 255)
                label = f"[!] IRREGULAR: FALL DETECTED{conf_str}"
            elif is_loitering:
                color = (211, 0, 148)
                label = f"[!] IRREGULAR: LOITERING{conf_str}"
            elif is_unusual:
                color = (60, 20, 220)
                label = f"[!] IRREGULAR: UNUSUAL ACTIVITY{conf_str}"
            elif is_intrusion:
                color = (30, 30, 220)
                label = f"[!] RESTRICTED ZONE INTRUSION"
            elif is_fire:
                color = (0, 0, 255)
                label = f"[!] FIRE DETECTED{conf_str}"
            elif is_smoke:
                color = (0, 140, 255)
                label = f"[!] SMOKE WARNING{conf_str}"
            else:
                color = (0, 200, 255)
                label = f"[!] ALERT: {event_type.upper()}{conf_str}"

            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 3, cv2.LINE_AA)

            line_len = max(8, min(24, (px2 - px1) // 4, (py2 - py1) // 4))
            cv2.line(frame, (px1, py1), (px1 + line_len, py1), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (px1, py1), (px1, py1 + line_len), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (px2, py1), (px2 - line_len, py1), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (px2, py1), (px2, py1 + line_len), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (px1, py2), (px1 + line_len, py2), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (px1, py2), (px1, py2 - line_len), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (px2, py2), (px2 - line_len, py2), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (px2, py2), (px2, py2 - line_len), (255, 255, 255), 2, cv2.LINE_AA)

            font_scale = 0.50
            font_thick = 1
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            pad = 4
            tag_y1 = max(0, py1 - th - 2 * pad)
            tag_y2 = tag_y1 + th + 2 * pad
            tag_x1 = px1
            tag_x2 = min(w - 1, px1 + tw + 2 * pad)

            cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), (10, 16, 26), -1)
            cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), color, 1)
            cv2.putText(
                frame, label, (tag_x1 + pad, tag_y2 - pad - 1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA
            )
        except Exception as exc:
            LOGGER.debug("Error drawing security box: %s", exc)

    return frame

# Shared specialist *weights* across cameras, but preserve each camera's own
# per-frame temporal state and keep YOLO calls serialized through a model lock.
# The person tracker is NOT pooled: persist=True needs per-camera track state.
_MODEL_POOL: dict[str, tuple[object, threading.RLock]] = {}
_MODEL_POOL_MUTEX = threading.RLock()


def merge_fallback_names(actual: dict, fallback: dict) -> dict:
    """Use labels embedded in model weights; fallback only for missing/numeric labels.

    Never overwrite real labels with guessed class-id mappings (e.g. class 0 !=
    automatically fight or fire). This prevents person detections becoming threats.
    """
    merged = dict(actual or {})
    for class_id, fallback_name in fallback.items():
        current = merged.get(class_id)
        if current is None or str(current).strip().isdigit():
            merged[class_id] = fallback_name
    return merged


@dataclass
class DetectionResult:
    boxes: list[list[float]]  # xyxy in input-frame pixels (not normalized)
    classes: list[int]
    confidences: list[float]
    class_names: dict[int, str]
    keypoints: list[list[float]] | None = None  # For pose detection: [person][keypoint][x,y]
    keypoint_confs: list[list[float]] | None = None  # For pose detection: [person][keypoint] confidence


class BaseDetector:
    """Base class for YOLO-based detectors with model pooling for GPU memory efficiency."""

    def __init__(self, model_path: str, confidence: float, classes: tuple[int, ...] | None = None):
        self.model_path = local_model_path(model_path)
        self.confidence = confidence
        self.classes = classes
        self.model = None
        self._lock = threading.RLock()
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            if not Path(self.model_path).is_file() and Path(self.model_path).parent != Path("."):
                raise FileNotFoundError(self.model_path)
            # Lock the pool during creation so a dozen cameras cannot each
            # allocate a copy of a 100 MB specialist model on the same GPU.
            with _MODEL_POOL_MUTEX:
                cache = _MODEL_POOL.get(str(self.model_path)) if settings.share_specialist_models else None
                if cache is not None:
                    self.model, self._lock = cache
                else:
                    candidate = YOLO(self.model_path)
                    if str(getattr(candidate, 'task', 'detect')) != 'detect':
                        raise ValueError(f'Expected YOLO bounding-box detector; got task={candidate.task!r}. '
                                         'Classification checkpoints are not object detectors.')
                    self.model = candidate
                    if settings.share_specialist_models:
                        _MODEL_POOL[str(self.model_path)] = (candidate, self._lock)
            LOGGER.info("Loaded object detector %s with classes %s", self.model_path, self.model.names)
        except Exception as e:
            LOGGER.error(f"Failed to load model {self.model_path}: {e}")
            self.model = None

    def detect(self, frame: np.ndarray, imgsz: int = 640, device: str = 'auto') -> DetectionResult | None:
        if self.model is None:
            return None
        with self._lock:
            args = {'conf': self.confidence, 'imgsz': imgsz, 'verbose': False}
            if device != 'auto':
                args['device'] = device
            if self.classes is not None:
                args['classes'] = list(self.classes)
            try:
                results = self.model(frame, **args)
                return self._parse_results(results[0])
            except Exception as e:
                LOGGER.error(f"Detection failed: {e}")
                return None

    def _parse_results(self, result) -> DetectionResult:
        boxes = []
        classes = []
        confidences = []
        keypoints = None
        keypoint_confs = None

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            classes = result.boxes.cls.int().cpu().numpy().tolist()
            confidences = result.boxes.conf.cpu().numpy().tolist()

        class_names = result.names if hasattr(result, 'names') else {}

        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy().tolist() if result.keypoints.xy is not None else None
            if result.keypoints.conf is not None:
                keypoint_confs = result.keypoints.conf.cpu().numpy().tolist()

        return DetectionResult(
            boxes=boxes,
            classes=classes,
            confidences=confidences,
            class_names=class_names,
            keypoints=keypoints,
            keypoint_confs=keypoint_confs
        )

    def draw(self, frame: np.ndarray, result: DetectionResult, color: tuple = (0, 255, 0),
             thickness: int = 2, font_scale: float = 0.48) -> np.ndarray:
        """Draw compact boxes with collision-aware labels kept inside the frame."""
        h, w = frame.shape[:2]
        occupied: list[tuple[int, int, int, int]] = []
        for box, cls, conf in zip(result.boxes, result.classes, result.confidences):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, thickness), cv2.LINE_AA)
            label = f"{result.class_names.get(cls, str(cls))} {conf:.0%}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            pad = 3
            # Prefer a label just inside the top edge of its own box.
            lx = max(0, min(w - tw - 2 * pad, x1))
            candidates = [y1, y1 + th + 2 * pad, max(0, y1 - th - 2 * pad)]
            ly = candidates[0]
            for candidate in candidates:
                rect = (lx, candidate, lx + tw + 2 * pad, candidate + th + 2 * pad)
                if not any(not (rect[2] < a or rect[0] > c or rect[3] < b or rect[1] > d)
                           for a, b, c, d in occupied):
                    ly = candidate
                    break
            ly = max(0, min(h - th - 2 * pad, ly))
            rect = (lx, ly, min(w - 1, lx + tw + 2 * pad), min(h - 1, ly + th + 2 * pad))
            overlay = frame.copy()
            cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
            cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
            cv2.putText(frame, label, (lx + pad, ly + th + pad - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            occupied.append(rect)
        return frame


class FireDetector(BaseDetector):
    """Fire and smoke detection using specialized YOLO model.

    Logic: Detect smoke FIRST, then confirm fire before alerting.
    - Smoke alone = warning (potential fire)
    - Fire confirmed = critical alert
    - Both smoke + fire = critical alert with high confidence
    - Small flames (matchstick, candle, lighter) = early warning with lower thresholds
    """

    def __init__(self):
        super().__init__(
            model_path=settings.fire_model_path,
            confidence=settings.fire_confidence,
            # Do not filter by guessed class IDs: fire model class order varies by dataset.
            classes=None
        )
        self.status = 'model_unavailable'
        self.inferences = 0
        self.failures = 0
        self.labels = []
        if self.model is not None:
            names = self.model.names
            items = names.items() if isinstance(names, dict) else enumerate(names)
            self.labels = [str(name).strip().lower() for index, name in items
                           if not self.classes or int(index) in self.classes]
            self.status = 'ready' if any(x in {'fire','flame','flames','smoke'} for x in self.labels) else 'unsupported_labels'
            if self.status != 'ready':
                LOGGER.error("Fire model has no supported labels among the enabled classes")
        # Track smoke detections for fire confirmation
        self.smoke_history: list[dict] = []
        self.max_smoke_age = 8.0  # seconds
        self.fire_history: list[tuple[float, list[float]]] = []
        self.fire_confirm_frames = 2
        self.fire_min_confidence = settings.fire_alert_confidence
        self.small_flame_min_confidence = settings.small_flame_alert_confidence
        self.small_flame_confirm_frames = 3

    def detect(self, frame: np.ndarray) -> DetectionResult | None:
        if self.status != 'ready':
            return None
        # Higher inference resolution preserves small flame/smoke regions.
        result = super().detect(frame, settings.fire_imgsz, settings.device)
        if result is None:
            self.failures += 1
        else:
            self.inferences += 1
        return result

    def diagnostics(self):
        status = self.status
        if status == 'ready':
            if self.failures:
                status = 'partial_failure' if self.inferences else 'inference_failed'
            elif not self.inferences:
                status = 'not_sampled'
        return {'status': status, 'enabled_labels': self.labels,
                'successful_frames': self.inferences, 'failed_frames': self.failures,
                'confidence_threshold': self.confidence}

    def check_fire_hazard(self, frame: np.ndarray, result: DetectionResult, now: float | None = None) -> list[dict]:
        """Check for fire hazards with smoke-first confirmation logic.

        Optimized for small flames (matchstick, candle, lighter) with lower thresholds.
        """
        hazards = []
        h, w = frame.shape[:2]
        now = time.monotonic() if now is None else now

        # Clean old smoke history
        self.smoke_history = [s for s in self.smoke_history if now - s['time'] < self.max_smoke_age]

        has_smoke = False
        has_fire = False
        smoke_conf = 0.0
        fire_conf = 0.0
        smoke_bbox = None
        fire_bbox = None
        fire_boxes = []  # Track all fire boxes for size analysis

        for box, cls, conf in zip(result.boxes, result.classes, result.confidences):
            class_name = result.class_names.get(cls, str(cls)).strip().lower()
            x1, y1, x2, y2 = box

            if class_name in {'smoke', 'smokes'}:
                has_smoke = True
                smoke_conf = max(smoke_conf, conf)
                smoke_bbox = [x1/w, y1/h, x2/w, y2/h]
                # Track smoke for fire confirmation
                self.smoke_history.append({
                    'time': now,
                    'confidence': conf,
                    'bbox': [x1/w, y1/h, x2/w, y2/h]
                })

            elif class_name in {'fire', 'flame', 'flames'}:
                has_fire = True
                fire_conf = max(fire_conf, conf)
                fire_bbox = [x1/w, y1/h, x2/w, y2/h]
                # Track fire box size for small flame detection
                box_area = ((x2 - x1) * (y2 - y1)) / (w * h)
                fire_boxes.append({'conf': conf, 'area': box_area, 'bbox': [x1/w, y1/h, x2/w, y2/h]})

        # Check for small flames (matchstick, candle, lighter) - very small box area
        small_flame_detected = False
        small_flame_conf = 0.0
        small_flame_bbox = None
        for fb in fire_boxes:
            # Small flame: area < 0.001 (0.1% of frame) - typical for matchstick at distance
            if fb['area'] < 0.001 and fb['conf'] >= self.small_flame_min_confidence:
                small_flame_detected = True
                small_flame_conf = max(small_flame_conf, fb['conf'])
                small_flame_bbox = fb['bbox']

        # Smoke-first, multi-frame confirmation: one red/orange-looking frame must
        # never create a critical fire alarm. Smoke is an early warning; fire is
        # critical only with smoke corroboration or repeated high-confidence frames.
        # Count distinct nearby fire detections, not unrelated bright objects in
        # different corners of the scene. Wall-clock/video timestamps expire history.
        from .incidents import same_scene_box
        self.fire_history = [(t, b) for t, b in self.fire_history if 0 <= now-t <= 4.0]
        if has_fire and fire_bbox:
            if self.fire_history and not same_scene_box(self.fire_history[-1][1], fire_bbox):
                self.fire_history = []
            # This method is invoked once per processed source frame.
            self.fire_history.append((now, fire_bbox))
        smoke_recent = any(same_scene_box(s['bbox'], fire_bbox) for s in self.smoke_history) if fire_bbox else False
        fire_persistent = len(self.fire_history) >= self.fire_confirm_frames

        # Standard fire confirmation (larger flames)
        fire_confirmed = has_fire and fire_conf >= self.fire_min_confidence and (smoke_recent or fire_persistent)

        # Small flame detection (matchstick, candle) - lower threshold, no smoke required
        small_flame_confirmed = (small_flame_detected and small_flame_conf >= self.small_flame_min_confidence
                                 and len(self.fire_history) >= self.small_flame_confirm_frames)

        if fire_confirmed:
            message = (f'FIRE CONFIRMED with smoke ({fire_conf:.0%} fire, {smoke_conf:.0%} smoke)'
                       if smoke_recent else f'Persistent fire confirmed across {len(self.fire_history)} frames ({fire_conf:.0%})')
            hazards.append({
                'type': 'fire_detected', 'severity': 'critical', 'class': 'fire',
                'confidence': fire_conf, 'bbox': fire_bbox or [0, 0, 0, 0],
                'message': message, 'smoke_present': smoke_recent,
                'smoke_confidence': smoke_conf if has_smoke else 0,
                'confirmation': 'smoke_corroborated' if smoke_recent else 'multi_frame'
            })
        elif small_flame_confirmed:
            # Small flame detected (matchstick, candle, lighter) - early warning
            hazards.append({
                'type': 'fire_detected', 'severity': 'high', 'class': 'small_flame',
                'confidence': small_flame_conf, 'bbox': small_flame_bbox or [0, 0, 0, 0],
                'message': f'Persistent small flame candidate ({small_flame_conf:.0%}) - review manually',
                'smoke_present': smoke_recent,
                'smoke_confidence': smoke_conf if has_smoke else 0,
                'confirmation': 'small_flame_multi_frame',
                'flame_size': 'small'
            })
        elif has_fire:
            hazards.append({
                'type': 'fire_watch', 'severity': 'medium', 'class': 'fire_candidate',
                'confidence': fire_conf, 'bbox': fire_bbox or [0, 0, 0, 0],
                'message': f'Possible fire signature ({fire_conf:.0%}); checking for smoke and persistence',
                'confirmation': 'pending'
            })

        elif has_smoke and not has_fire:
            # Smoke detected but no fire yet - WARNING level
            # Check if persistent smoke (multiple detections)
            recent_smoke = [s for s in self.smoke_history if now - s['time'] < 3.0]
            is_persistent = len(recent_smoke) >= 2

            severity = 'high' if is_persistent else 'medium'
            # Repetition is useful for confirmation, but must NEVER inflate the
            # model confidence over the 50% notification threshold.
            confidence = smoke_conf

            hazards.append({
                'type': 'smoke',
                'severity': severity,
                'class': 'smoke',
                'confidence': confidence,
                'bbox': smoke_bbox or [0, 0, 0, 0],
                'message': f'Smoke detected ({smoke_conf:.0%}) - {"persistent, potential fire risk" if is_persistent else "monitoring for fire confirmation"}',
                'persistent': is_persistent,
                'smoke_count': len(recent_smoke)
            })

        return hazards


class ViolenceDetector(BaseDetector):
    """Violence/Fight/Weapon detection for home security."""

    def __init__(self, model_path: str | None = None):
        self.specialist = model_path is not None
        super().__init__(
            model_path=model_path or settings.violence_model_path,
            confidence=settings.violence_confidence
        )
        # Never guess violence class IDs: class 0 is often 'person'.
        self.violence_classes = {}
        if self.model is not None and not any(is_fight_label(n) for n in
                  (self.model.names.values() if isinstance(self.model.names, dict) else self.model.names)):
            LOGGER.error('Fight detector %s has no positive fight/violence label. Disabling it.', self.model_path)
            self.model = None
        # Track weapon detections for home security
        self.weapon_history: dict[str, list[dict]] = {}  # track_id -> weapon detections
        # Lower threshold for knife specifically (knives are harder to detect)
        self.knife_min_confidence = 0.25
        self.weapon_min_confidence = 0.30

    def detect(self, frame: np.ndarray) -> DetectionResult | None:
        result = super().detect(frame, settings.violence_imgsz, settings.device)
        if result:
            result.class_names = merge_fallback_names(result.class_names, self.violence_classes)
        return result

    def analyze_violence(self, frame: np.ndarray, result: DetectionResult,
                         person_tracks: list[dict] | None = None) -> list[dict]:
        """Analyze violence indicators with home-security focus.

        Optimized for knife detection with lower confidence thresholds.
        """
        events = []
        h, w = frame.shape[:2]

        # Map person tracks by track_id for weapon-person association
        person_by_id = {t['id']: t for t in person_tracks} if person_tracks else {}

        for box, cls, conf in zip(result.boxes, result.classes, result.confidences):
            class_name = result.class_names.get(cls, str(cls)).strip().lower()
            x1, y1, x2, y2 = box

            if class_name in ('unusual_activity', 'abnormal_activity', 'suspicious_activity'):
                events.append({
                    'type': 'unusual_activity', 'severity': 'high', 'subtype': class_name,
                    'confidence': conf, 'bbox': [x1/w, y1/h, x2/w, y2/h],
                    'message': f'{class_name.replace("_", " ").capitalize()} detected with {conf:.0%} confidence',
                })
            elif is_fight_label(class_name):
                events.append({
                    'type': 'violence',
                    'severity': 'critical',
                    'subtype': class_name,
                    'confidence': conf,
                    'bbox': [x1/w, y1/h, x2/w, y2/h],
                    'message': f'{class_name.capitalize()} detected with {conf:.0%} confidence',
                    'source_model': Path(getattr(self, 'model_path', 'fight-yolo')).name
                })
            elif class_name in ('weapon', 'knife', 'gun', 'pistol', 'rifle', 'blade'):
                # Find closest person to associate weapon
                closest_person = None
                min_dist = float('inf')
                weapon_center = ((x1 + x2) / 2, (y1 + y2) / 2)

                for pid, person in person_by_id.items():
                    px1, py1, px2, py2 = person.get('bbox', [0, 0, 0, 0])
                    if px2 > px1 and py2 > py1:
                        person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
                        dist = np.hypot(weapon_center[0] - person_center[0], weapon_center[1] - person_center[1])
                        if dist < min_dist:
                            min_dist = dist
                            closest_person = pid

                # Track weapon for unusual activity detection
                weapon_key = f"{class_name}_{closest_person or 'unknown'}"
                now = time.monotonic()
                if weapon_key not in self.weapon_history:
                    self.weapon_history[weapon_key] = []
                self.weapon_history[weapon_key].append({'time': now, 'confidence': conf})

                # Clean old history (keep last 10 seconds)
                self.weapon_history[weapon_key] = [
                    w for w in self.weapon_history[weapon_key] if now - w['time'] < 10.0
                ]

                # Determine severity based on weapon type and persistence
                is_persistent = len(self.weapon_history[weapon_key]) >= 2  # Reduced from 3

                # Use lower thresholds for knives (harder to detect)
                min_conf = self.knife_min_confidence if class_name in ('knife', 'blade') else self.weapon_min_confidence

                if conf < min_conf:
                    # Below threshold, but track for persistence
                    if is_persistent and conf >= min_conf * 0.7:
                        # Persistent low-confidence detections can still trigger
                        pass
                    else:
                        continue  # Skip this detection

                severity = 'critical'
                if class_name in ('gun', 'pistol', 'rifle'):
                    severity = 'critical'
                elif class_name in ('knife', 'blade'):
                    severity = 'critical' if is_persistent else 'high'

                # Home security: unusual activity with knife
                activity_msg = f'{class_name.capitalize()} detected'
                if closest_person:
                    activity_msg += f' near person #{closest_person}'
                if is_persistent:
                    activity_msg += ' - PERSISTENT THREAT'

                events.append({
                    'type': 'weapon',
                    'severity': severity,
                    'subtype': class_name,
                    'confidence': conf,
                    'bbox': [x1/w, y1/h, x2/w, y2/h],
                    'message': f'{activity_msg} with {conf:.0%} confidence',
                    'source_model': Path(getattr(self, 'model_path', 'fight-yolo')).name,
                    'associated_person': closest_person,
                    'persistent': is_persistent,
                    'home_security_alert': True
                })

        # Being close to another person is NORMAL. Do not manufacture a 60%
        # fight-confidence score or create snapshots/alerts from proximity.
        return events


class SpecialistWeaponDetector(BaseDetector):
    """Weapon specialist, including CS2 firearm labels without pretending they are knives.

    Supports an alternative, user-supplied real knife checkpoint whose class names
    are knife/blade/gun/etc. No guesswork for class IDs or negative/unknown labels.
    """

    def __init__(self, model_path: str | None = None):
        super().__init__(model_path or settings.weapon_model_path, settings.weapon_confidence)
        if self.model is not None and not any(weapon_class(n) for n in
                  (self.model.names.values() if isinstance(self.model.names, dict) else self.model.names)):
            LOGGER.error('Specialist %s has no supported real-world weapon labels. Disabling it.', self.model_path)
            self.model = None

    def detect(self, frame: np.ndarray) -> DetectionResult | None:
        return super().detect(frame, settings.weapon_imgsz, settings.device)

    @staticmethod
    def analyze_weapons(frame: np.ndarray, result: DetectionResult) -> list[dict]:
        events = []
        h, w = frame.shape[:2]
        for box, cls, conf in zip(result.boxes, result.classes, result.confidences):
            raw_label = str(result.class_names.get(cls, str(cls))).strip()
            subtype = weapon_class(raw_label)
            if not subtype:
                continue
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue
            events.append({
                'type': 'weapon', 'severity': 'critical', 'subtype': subtype,
                'model_label': raw_label, 'confidence': float(conf),
                'bbox': [max(0., x1/w), max(0., y1/h), min(1., x2/w), min(1., y2/h)],
                'message': f'Weapon model detected {raw_label} ({conf:.0%})',
                'source_model': 'specialist_weapon',
            })
        return events


class KnifeDetector(BaseDetector):
    """Specialized knife detection using trained model (e.g., from syzenzz/cctv-knife-detection-dataset).

    This detector uses a model specifically trained for knife detection in CCTV footage,
    providing higher accuracy for small knives compared to general COCO detection.
    """

    def __init__(self):
        super().__init__(
            model_path=settings.knife_model_path,
            confidence=settings.knife_confidence
        )
        self.knife_min_confidence = 0.25

    def detect(self, frame: np.ndarray) -> DetectionResult | None:
        return super().detect(frame, settings.knife_imgsz, settings.device)

    def analyze_knives(self, frame: np.ndarray, result: DetectionResult) -> list[dict]:
        """Analyze knife detections with home-security focus."""
        events = []
        h, w = frame.shape[:2]

        for box, cls, conf in zip(result.boxes, result.classes, result.confidences):
            class_name = result.class_names.get(cls, str(cls)).strip().lower()
            x1, y1, x2, y2 = box

            if class_name in ('knife', 'blade', 'weapon'):
                if conf < self.knife_min_confidence:
                    continue

                events.append({
                    'type': 'weapon',
                    'severity': 'critical',
                    'subtype': class_name,
                    'confidence': conf,
                    'bbox': [x1/w, y1/h, x2/w, y2/h],
                    'message': f'Knife/Blade detected with {conf:.0%} confidence',
                    'home_security_alert': True,
                    'source_model': 'specialist_knife',
                })

        return events


class AnomalyDetector:
    """Anomaly detection: Fall, Loitering, Unusual poses using YOLOv8-Pose.
    Also supports YOLO-World for open-vocabulary detection (fire, violence, weapons, etc.)."""

    def __init__(self):
        self.model = None
        self._lock = threading.RLock()
        self._load_model()
        self.track_history: dict[int, list[tuple[float, float]]] = {}  # track_id -> positions
        self.fall_history: dict[int, int] = {}  # track_id -> frames on ground

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(local_model_path(settings.anomaly_model_path))
            LOGGER.info(f"Loaded pose model: {settings.anomaly_model_path}")
        except Exception as e:
            LOGGER.error(f"Failed to load pose model: {e}")
            self.model = None

    def detect(self, frame: np.ndarray, person_tracks: list[dict]) -> DetectionResult | None:
        """Detect poses for tracked persons (or open-vocabulary with YOLO-World)."""
        if self.model is None:
            return None

        # Auto-detect device: use CUDA if available, otherwise CPU
        import torch
        device = settings.device
        if device == 'auto':
            device = '0' if torch.cuda.is_available() else 'cpu'

        # Check if this is a YOLO-World model (has set_classes method)
        is_world_model = hasattr(self.model, 'set_classes')

        with self._lock:
            try:
                if is_world_model:
                    # YOLO-World: use open-vocabulary detection
                    self.model.set_classes(list(settings.yolo_world_classes))
                    results = self.model.predict(
                        frame, conf=settings.anomaly_confidence,
                        imgsz=settings.imgsz, device=device, verbose=False
                    )
                    return self._parse_world_results(results[0], person_tracks)
                else:
                    # Standard YOLO-Pose: track persons only
                    results = self.model.track(
                        frame, persist=True, tracker='bytetrack.yaml',
                        classes=[0], conf=settings.anomaly_confidence,
                        imgsz=settings.imgsz, device=device, verbose=False
                    )
                    return self._parse_pose_results(results[0], person_tracks)
            except Exception as e:
                LOGGER.error(f"Anomaly detection failed: {e}")
                return None

    def _parse_pose_results(self, result, person_tracks: list[dict]) -> DetectionResult:
        boxes = []
        classes = []
        confidences = []
        keypoints = []
        keypoint_confs = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            classes = result.boxes.cls.int().cpu().numpy().tolist()
            confidences = result.boxes.conf.cpu().numpy().tolist()

        if hasattr(result, 'keypoints') and result.keypoints is not None and result.keypoints.xy is not None:
            keypoints = result.keypoints.xy.cpu().numpy().tolist()
            # Also get keypoint confidences if available
            if result.keypoints.conf is not None:
                keypoint_confs = result.keypoints.conf.cpu().numpy().tolist()

        return DetectionResult(
            boxes=boxes,
            classes=classes,
            confidences=confidences,
            class_names={0: 'person'},
            keypoints=keypoints,
            keypoint_confs=keypoint_confs
        )

    def _parse_world_results(self, result, person_tracks: list[dict]) -> DetectionResult:
        """Parse YOLO-World open-vocabulary detection results."""
        boxes = []
        classes = []
        confidences = []
        class_names = {}

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            classes = result.boxes.cls.int().cpu().numpy().tolist()
            confidences = result.boxes.conf.cpu().numpy().tolist()
            # YOLO-World uses the class names we set
            class_names = {i: name for i, name in enumerate(settings.yolo_world_classes)}

        return DetectionResult(
            boxes=boxes,
            classes=classes,
            confidences=confidences,
            class_names=class_names,
            keypoints=None
        )

    def analyze_anomalies(self, frame: np.ndarray, pose_result: DetectionResult,
                          person_tracks: list[dict], now: float) -> list[dict]:
        """Analyze pose for falls, loitering, unusual behavior.
        Also handles YOLO-World open-vocabulary detections (fire, violence, weapons, etc.)."""
        events = []
        h, w = frame.shape[:2]

        # Handle YOLO-World open-vocabulary detections (no keypoints, but class names)
        if not pose_result.keypoints and pose_result.class_names:
            # This is a YOLO-World result - check detected classes directly
            for i, (box, cls, conf) in enumerate(zip(pose_result.boxes, pose_result.classes, pose_result.confidences)):
                class_name = pose_result.class_names.get(cls, str(cls)).strip().lower()
                x1, y1, x2, y2 = box

                # Map YOLO-World classes to event types
                if class_name in ('fire', 'flame'):
                    events.append({
                        'type': 'fire_detected',
                        'severity': 'critical',
                        'track_id': None,
                        'confidence': conf,
                        'bbox': [x1/w, y1/h, x2/w, y2/h],
                        'message': f'Fire detected with {conf:.0%} confidence'
                    })
                elif class_name in ('smoke',):
                    events.append({
                        'type': 'smoke',
                        'severity': 'high',
                        'track_id': None,
                        'confidence': conf,
                        'bbox': [x1/w, y1/h, x2/w, y2/h],
                        'message': f'Smoke detected with {conf:.0%} confidence'
                    })
                elif class_name in ('person fighting', 'fighting', 'fight', 'violence'):
                    events.append({
                        'type': 'violence',
                        'severity': 'critical',
                        'track_id': None,
                        'confidence': conf,
                        'bbox': [x1/w, y1/h, x2/w, y2/h],
                        'message': f'Violence/fight detected with {conf:.0%} confidence'
                    })
                elif class_name in ('knife', 'a person holding a knife', 'gun', 'a person holding a gun', 'weapon', 'pistol', 'rifle', 'blade'):
                    events.append({
                        'type': 'weapon',
                        'severity': 'critical',
                        'track_id': None,
                        'confidence': conf,
                        'bbox': [x1/w, y1/h, x2/w, y2/h],
                        'message': f'{class_name.capitalize()} detected with {conf:.0%} confidence',
                        'source_model': 'yolo_world'
                    })
                elif class_name in ('person falling', 'falling person', 'fall'):
                    events.append({
                        'type': 'fall_detected',
                        'severity': 'high',
                        'track_id': None,
                        'confidence': conf,
                        'bbox': [x1/w, y1/h, x2/w, y2/h],
                        'message': f'Person falling detected with {conf:.0%} confidence'
                    })
            # For world model, still do loitering detection via person tracks
            # (fall detection via pose is skipped since no keypoints)

        # Standard pose-based analysis (fall detection, loitering)
        if not pose_result.keypoints:
            # Still do loitering detection using person tracks
            if settings.enable_loitering_detection:
                for track in person_tracks:
                    track_id = track['id']
                    x1, y1, x2, y2 = track.get('bbox', [0,0,0,0])
                    if x2 > x1 and y2 > y1:  # valid bbox
                        center_x = (x1 + x2) / (2 * w)
                        center_y = (y1 + y2) / (2 * h)

                        if track_id not in self.track_history:
                            self.track_history[track_id] = []

                        self.track_history[track_id].append((center_x, center_y, now))
                        cutoff = now - settings.loitering_time_seconds
                        self.track_history[track_id] = [
                            (x, y, t) for x, y, t in self.track_history[track_id] if t > cutoff
                        ]

                        if len(self.track_history[track_id]) > 10:
                            positions = [(x, y) for x, y, t in self.track_history[track_id]]
                            x_coords, y_coords = zip(*positions)
                            x_range = max(x_coords) - min(x_coords)
                            y_range = max(y_coords) - min(y_coords)

                            if x_range < 0.1 and y_range < 0.1:
                                events.append({
                                    'type': 'loitering',
                                    'severity': 'medium',
                                    'track_id': track_id,
                                    'confidence': 0.7,
                                    'bbox': [x1/w, y1/h, x2/w, y2/h],
                                    'message': f'Person #{track_id} loitering detected (> {settings.loitering_time_seconds}s)'
                                })
            return events

        # COCO keypoint indices: 0=nose, 5=left_shoulder, 6=right_shoulder,
        # 11=left_hip, 12=right_hip, 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle

        # Get keypoint confidences for this frame
        kpt_confs = pose_result.keypoint_confs if pose_result.keypoint_confs else []

        for i, (kpts, box) in enumerate(zip(pose_result.keypoints, pose_result.boxes)):
            if i >= len(person_tracks):
                continue
            track = person_tracks[i]
            track_id = track['id']
            x1, y1, x2, y2 = box

            # Get keypoint confidences for this person
            person_kpt_confs = kpt_confs[i] if i < len(kpt_confs) else []

            # Fall detection: only reasonably horizontal poses; do not flag
            # someone sitting, kneeling or bending down as a fall.
            if settings.enable_fall_detection and len(kpts) >= 17:
                # Helper to get y coordinate with confidence check
                def get_kpt_y(idx):
                    if idx < len(kpts) and idx < len(person_kpt_confs):
                        return kpts[idx][1] if person_kpt_confs[idx] > 0.3 else None
                    return None

                # Helper to compute nanmean safely (returns None if all values are None)
                def safe_nanmean(values):
                    valid = [v for v in values if v is not None]
                    return np.nanmean(valid) if valid else None

                nose_y = get_kpt_y(0)
                l_shoulder_y = get_kpt_y(5)
                r_shoulder_y = get_kpt_y(6)
                l_hip_y = get_kpt_y(11)
                r_hip_y = get_kpt_y(12)
                l_ankle_y = get_kpt_y(15)
                r_ankle_y = get_kpt_y(16)

                # Average y positions (skip if both sides are None)
                shoulder_y = safe_nanmean([l_shoulder_y, r_shoulder_y])
                hip_y = safe_nanmean([l_hip_y, r_hip_y])
                ankle_y = safe_nanmean([l_ankle_y, r_ankle_y])

                def get_kpt_x(idx):
                    if idx < len(kpts) and idx < len(person_kpt_confs):
                        return kpts[idx][0] if person_kpt_confs[idx] > 0.3 else None
                    return None
                shoulder_x = safe_nanmean([get_kpt_x(5), get_kpt_x(6)])
                hip_x = safe_nanmean([get_kpt_x(11), get_kpt_x(12)])

                if shoulder_y is not None and hip_y is not None and ankle_y is not None:
                    # Fallen: shoulders and hips are at similar height to ankles (horizontal body)
                    vertical_span = ankle_y - shoulder_y
                    hip_shoulder_diff = abs(hip_y - shoulder_y)

                    horizontal = (shoulder_x is not None and hip_x is not None
                                  and abs(hip_x - shoulder_x) > 1.25 * hip_shoulder_diff)
                    if (horizontal and vertical_span < h * 0.3
                            and hip_shoulder_diff < h * 0.15
                            and (x2-x1) > (y2-y1) * 1.05):
                        self.fall_history[track_id] = self.fall_history.get(track_id, 0) + 1
                        if self.fall_history[track_id] >= 3:  # 3 consecutive frames
                            events.append({
                                'type': 'fall_detected',
                                'severity': 'high',
                                'track_id': track_id,
                                # Pose checkpoint provides confidence for a PERSON,
                                # not for a FALL action. Never portray the person
                                # score as a 50%+ verified fall classifier score.
                                'confidence': None,
                                'person_confidence': pose_result.confidences[i],
                                'provisional_pose_heuristic': True,
                                'bbox': [x1/w, y1/h, x2/w, y2/h],
                                'message': f'Person #{track_id} fall detected'
                            })
                    else:
                        self.fall_history[track_id] = 0

            # Loitering detection: track position history
            if settings.enable_loitering_detection:
                center_x = (x1 + x2) / (2 * w)
                center_y = (y1 + y2) / (2 * h)

                if track_id not in self.track_history:
                    self.track_history[track_id] = []

                self.track_history[track_id].append((center_x, center_y, now))

                # Keep only recent history
                cutoff = now - settings.loitering_time_seconds
                self.track_history[track_id] = [
                    (x, y, t) for x, y, t in self.track_history[track_id] if t > cutoff
                ]

                # Check if person stayed in small area for long time
                if len(self.track_history[track_id]) > 10:
                    positions = [(x, y) for x, y, t in self.track_history[track_id]]
                    x_coords, y_coords = zip(*positions)
                    x_range = max(x_coords) - min(x_coords)
                    y_range = max(y_coords) - min(y_coords)

                    if x_range < 0.1 and y_range < 0.1:  # Very small movement area
                        events.append({
                            'type': 'loitering',
                            'severity': 'medium',
                            'track_id': track_id,
                            'confidence': 0.7,
                            'bbox': [x1/w, y1/h, x2/w, y2/h],
                            'message': f'Person #{track_id} loitering detected (> {settings.loitering_time_seconds}s)'
                        })

        # Cleanup old tracks
        current_ids = {t['id'] for t in person_tracks}
        for tid in list(self.track_history.keys()):
            if tid not in current_ids:
                del self.track_history[tid]
        for tid in list(self.fall_history.keys()):
            if tid not in current_ids:
                del self.fall_history[tid]

        return events

    def _draw_keypoints(self, frame: np.ndarray, keypoints: list, boxes: list, keypoint_confs: list | None = None) -> np.ndarray:
        """Draw pose keypoints and skeleton on frame."""
        # COCO skeleton connections
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
            (5, 11), (6, 12), (11, 12),  # torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # legs
        ]
        for i, (kpts, box) in enumerate(zip(keypoints, boxes)):
            if kpts is None:
                continue
            person_confs = keypoint_confs[i] if keypoint_confs and i < len(keypoint_confs) else None
            for j, (x, y) in enumerate(kpts):
                if person_confs and person_confs[j] < 0.3:
                    continue
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
            for (a, b) in skeleton:
                if a < len(kpts) and b < len(kpts):
                    conf_a = person_confs[a] if person_confs else 1.0
                    conf_b = person_confs[b] if person_confs else 1.0
                    if conf_a > 0.3 and conf_b > 0.3:
                        cv2.line(frame, (int(kpts[a][0]), int(kpts[a][1])),
                                 (int(kpts[b][0]), int(kpts[b][1])), (0, 255, 255), 2)
        return frame


class ObjectDetector(BaseDetector):
    """General object detection for all COCO classes with optional knife tile scan."""

    def __init__(self):
        super().__init__(
            model_path=settings.object_model_path,
            confidence=settings.object_confidence,
            classes=settings.object_classes if settings.object_classes else None
        )

    def detect(self, frame: np.ndarray) -> DetectionResult | None:
        full = super().detect(frame, settings.object_imgsz, settings.device)
        if not settings.knife_tile_scan or full is None or self.model is None:
            return full
        # Optional high-recall scan: crop overlapping quadrants so a tiny knife
        # occupies more input pixels. Costs ~4 additional model passes.
        h, w = frame.shape[:2]
        if h < 240 or w < 240:
            return full
        cut_x, cut_y = w // 2, h // 2
        overlap_x, overlap_y = w // 10, h // 10
        crops = [
            (0, 0, min(w, cut_x + overlap_x), min(h, cut_y + overlap_y)),
            (max(0, cut_x - overlap_x), 0, w, min(h, cut_y + overlap_y)),
            (0, max(0, cut_y - overlap_y), min(w, cut_x + overlap_x), h),
            (max(0, cut_x - overlap_x), max(0, cut_y - overlap_y), w, h),
        ]
        all_boxes, all_classes, all_confs = list(full.boxes), list(full.classes), list(full.confidences)
        for left, top, right, bottom in crops:
            cropped = super().detect(frame[top:bottom, left:right], settings.object_imgsz, settings.device)
            if cropped is None:
                continue
            for box, cls, conf in zip(cropped.boxes, cropped.classes, cropped.confidences):
                all_boxes.append([box[0] + left, box[1] + top, box[2] + left, box[3] + top])
                all_classes.append(cls)
                all_confs.append(conf)
        # Remove duplicated boxes from overlapping crops (class-aware NMS).
        keep = []
        for i in sorted(range(len(all_boxes)), key=lambda k: all_confs[k], reverse=True):
            a = all_boxes[i]
            area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
            duplicate = False
            for j in keep:
                if all_classes[j] != all_classes[i]:
                    continue
                b = all_boxes[j]
                intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
                if intersection / max(1, area_a + area_b - intersection) >= 0.45:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(i)
        return DetectionResult([all_boxes[i] for i in keep], [all_classes[i] for i in keep],
                               [all_confs[i] for i in keep], full.class_names)

    def draw_daily_objects(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        color: tuple[int, int, int] = (235, 180, 30),
        thickness: int = 2,
        font_scale: float = 0.44
    ) -> np.ndarray:
        """Render high-clarity bounding boxes and labels for daily life objects.

        Excludes person (class 0, managed by person tracker) and weapon/fire classes
        which are managed by specialist detectors.
        """
        if frame is None or result is None or not result.boxes:
            return frame
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return frame

        excluded_names = {'person', 'fire', 'smoke', 'knife', 'gun', 'weapon', 'pistol', 'rifle', 'blade'}

        for box, cls, conf in zip(result.boxes, result.classes, result.confidences):
            class_name = str(result.class_names.get(cls, cls)).strip().lower()
            if cls == 0 or class_name in excluded_names:
                continue

            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            # Draw sleek bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, thickness), cv2.LINE_AA)

            # Draw corner accents
            clen = max(6, min(16, (x2 - x1) // 5, (y2 - y1) // 5))
            cv2.line(frame, (x1, y1), (x1 + clen, y1), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (x1, y1), (x1, y1 + clen), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2 - clen, y1), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2, y1 + clen), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1 + clen, y2), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1, y2 - clen), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2 - clen, y2), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2, y2 - clen), (255, 255, 255), 1, cv2.LINE_AA)

            # Compact label badge with object name and explicit CF score
            label = format_daily_object_label(class_name, conf)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            pad = 3
            tag_y1 = max(0, y1 - th - 2 * pad)
            tag_y2 = tag_y1 + th + 2 * pad
            tag_x1 = x1
            tag_x2 = min(w - 1, x1 + tw + 2 * pad)

            cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), (14, 20, 30), -1)
            cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), color, 1)
            cv2.putText(
                frame,
                label,
                (tag_x1 + pad, tag_y2 - pad - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame


class DetectionManager:
    """Manages all detectors and coordinates detection pipeline."""

    def __init__(self):
        self.fire_detector = FireDetector()
        self.violence_detector = ViolenceDetector()
        # Only construct the specialist if the checkpoint exists; otherwise report
        # unavailable rather than pretending the standard person model sees guns.
        weapon_path = local_model_path(settings.weapon_model_path)
        self.weapon_detector = (SpecialistWeaponDetector()
                                if Path(weapon_path).is_file() else None)
        trained_path = local_model_path(settings.trained_weapon_model_path)
        self.trained_weapon_detector = (
            SpecialistWeaponDetector(trained_path)
            if settings.enable_trained_weapon and Path(trained_path).is_file()
            and Path(trained_path).resolve() != Path(weapon_path).resolve()
            else None)
        # Knife detector (specialized model from syzenzz/cctv-knife-detection-dataset)
        knife_path = local_model_path(settings.knife_model_path)
        self.knife_detector = (KnifeDetector()
                               if Path(knife_path).is_file() else None)
        # Optional TorchVision R3D-18 fight classifier (video clip classification)
        self.torch_fight_classifier = None
        if settings.enable_fight_classifier:
            try:
                from .torch_fight import FightClipClassifier
                p = local_model_path(settings.fight_classifier_model_path)
                self.torch_fight_classifier = FightClipClassifier(
                    p, settings.fight_classifier_confidence,
                    settings.fight_classifier_interval, settings.device)
            except Exception as exc:
                LOGGER.warning('Optional R3D-18 classifier not loaded: %s', exc)
        # Optional legacy 3DCNN (TensorFlow/Keras) - opt-in only
        self.temporal_fight_detector = None
        if settings.enable_3dcnn:
            try:
                from .temporal_fight import TemporalFightDetector
                p = local_model_path(settings.cnn3d_model_path)
                self.temporal_fight_detector = TemporalFightDetector(
                    p, int(settings.cnn3d_positive_class), settings.cnn3d_confidence,
                    settings.cnn3d_every_n_frames)
            except Exception as exc:
                LOGGER.warning('3DCNN disabled: %s', exc)
        self.anomaly_detector = AnomalyDetector()
        self.object_detector = ObjectDetector()
        self.threat_policy = ThreatPolicy(
            enabled=settings.strict_weapon_policy,
            require_person=settings.weapon_require_person,
            person_margin=settings.weapon_person_margin,
            thresholds={'gun': settings.weapon_gun_min_score,
                        'knife': settings.weapon_knife_min_score,
                        'weapon': settings.weapon_other_min_score},
            max_areas={'gun': settings.weapon_gun_max_area,
                       'knife': settings.weapon_knife_max_area,
                       'weapon': settings.weapon_other_max_area},
        )
        self.last_suppressed: list[dict] = []
        self.last_raw_candidates: list[dict] = []
        self.last_object_result: DetectionResult | None = None
        self._last_telegram_alert: dict[str, float] = {}
        self._last_tts_alert: dict[str, float] = {}
        # Frame counters for running detectors at different frequencies
        self._frame_counter = 0
        # Run heavy detectors every N frames for performance
        self.fire_every_n_frames = 1    # Fire/smoke is safety-critical: sample every processed frame
        self.violence_every_n_frames = settings.violence_every_n_frames
        self.weapon_every_n_frames = settings.weapon_every_n_frames
        self.knife_every_n_frames = settings.knife_every_n_frames
        self.anomaly_every_n_frames = 5   # Run pose/open-vocabulary less often
        self.object_every_n_frames = settings.object_every_n_frames

    @staticmethod
    def _filter_result(result: DetectionResult, allowed: set[str]) -> DetectionResult:
        keep = [i for i, cls in enumerate(result.classes)
                if str(result.class_names.get(cls, str(cls))).strip().lower() in allowed]
        return DetectionResult(
            boxes=[result.boxes[i] for i in keep],
            classes=[result.classes[i] for i in keep],
            confidences=[result.confidences[i] for i in keep],
            class_names=result.class_names,
        )

    def process_frame(self, frame: np.ndarray, person_tracks: list[dict],
                      camera_id: str, camera_config: dict, *, timestamp: float | None = None) -> list[dict]:
        """Run all enabled detectors and return events.

        Optimized: Runs detectors at different frequencies for better FPS.
        """
        all_events = []
        self.last_suppressed = []
        self.last_raw_candidates = []
        # Never feed annotation overlays from earlier detectors back into models.
        inference_frame = frame.copy()
        now = time.monotonic() if timestamp is None else timestamp
        self._frame_counter += 1

        # Fire detection - run every frame (critical)
        if camera_config.get('fire_enabled', 1) and self._frame_counter % self.fire_every_n_frames == 0:
            fire_result = self.fire_detector.detect(inference_frame)
            if fire_result:
                hazards = self.fire_detector.check_fire_hazard(frame, fire_result, now)
                for hazard in hazards:
                    hazard['camera_id'] = camera_id
                    all_events.append(hazard)
                # Do not draw UNCONFIRMED fire hypotheses as red alert boxes.
                # The incident tracker draws evidence-backed confirmed events.

        # Violence detection - run every frame (critical)
        if camera_config.get('violence_enabled', 1) and self._frame_counter % self.violence_every_n_frames == 0:
            violence_result = self.violence_detector.detect(inference_frame)
            if violence_result:
                # Get person boxes for proximity analysis
                person_boxes = person_tracks if person_tracks else None
                violence_events = self.violence_detector.analyze_violence(
                    frame, violence_result, person_boxes)
                for event in violence_events:
                    event['camera_id'] = camera_id
                    all_events.append(event)

        # Named specialist models: generic pretrained threat + optional CCTV-fine-tuned weights.
        # Each model's own labels are used; class numbers are NEVER guessed.
        if camera_config.get('violence_enabled', 1) and self._frame_counter % self.weapon_every_n_frames == 0:
            for specialist in (self.weapon_detector, self.trained_weapon_detector):
                if specialist is None:
                    continue
                specialist_result = specialist.detect(inference_frame)
                if specialist_result is None:
                    continue
                for event in specialist.analyze_weapons(frame, specialist_result):
                    event['camera_id'] = camera_id
                    event['source_model'] = Path(specialist.model_path).name
                    all_events.append(event)

        # Knife detection (specialized model from syzenzz/cctv-knife-detection-dataset)
        if (self.knife_detector is not None and camera_config.get('violence_enabled', 1)
                and self._frame_counter % self.knife_every_n_frames == 0):
            knife_result = self.knife_detector.detect(inference_frame)
            if knife_result is not None:
                for event in self.knife_detector.analyze_knives(frame, knife_result):
                    event['camera_id'] = camera_id
                    all_events.append(event)

        # TorchVision R3D-18 fight classifier (video clip classification)
        if (self.torch_fight_classifier is not None
                and camera_config.get('violence_enabled', 1)):
            fight_event = self.torch_fight_classifier.process_frame(frame, now)
            if fight_event:
                fight_event['camera_id'] = camera_id
                all_events.append(fight_event)

        # Legacy 3DCNN (TensorFlow) - opt-in only
        if (self.temporal_fight_detector is not None
                and camera_config.get('violence_enabled', 1)):
            fight_event = self.temporal_fight_detector.process_frame(frame, now)
            if fight_event:
                fight_event['camera_id'] = camera_id
                all_events.append(fight_event)

        # Anomaly detection (falls, loitering, YOLO-World open-vocabulary) - run every N frames
        if (camera_config.get('anomaly_enabled', 1)
                and getattr(self.anomaly_detector, 'model', None) is not None
                and self._frame_counter % self.anomaly_every_n_frames == 0):
            pose_result = self.anomaly_detector.detect(inference_frame, person_tracks)
            if pose_result:
                anomaly_events = self.anomaly_detector.analyze_anomalies(
                    frame, pose_result, person_tracks, now)
                for event in anomaly_events:
                    event['camera_id'] = camera_id
                    all_events.append(event)
                # YOLO-World produces detections but no pose keypoints. In that
                # mode draw its true threat boxes, not just unlabelled video.
                if not pose_result.keypoints and anomaly_events:
                    from .incidents import draw_event_box, eligible_event
                    for event in anomaly_events:
                        if eligible_event(event, settings.incident_min_confidence):
                            draw_event_box(frame, event)
                # Draw pose keypoints
                if pose_result.keypoints:
                    frame = self.anomaly_detector._draw_keypoints(frame, pose_result.keypoints, pose_result.boxes, pose_result.keypoint_confs)

        # Object detection - run every N frames (least critical)
        if camera_config.get('object_enabled', 1) and self._frame_counter % self.object_every_n_frames == 0:
            object_result = self.object_detector.detect(inference_frame)
            if object_result:
                self.last_object_result = object_result
                # Filter for knife class (COCO class 43) if knife_tile_scan enabled
                if settings.knife_tile_scan:
                    knife_result = self._filter_result(object_result, {'knife'})
                    if knife_result.boxes:
                        self.object_detector.draw(frame, knife_result, (255, 165, 0))  # Orange for knife

        # Apply threat policy to weapon events (suppress false positives)
        filtered_events = []
        frame_h, frame_w = frame.shape[:2]
        for event in all_events:
            if event['type'] == 'weapon':
                # Convert to threat policy format
                candidate = {
                    'type': event.get('subtype', 'weapon'),
                    'confidence': event['confidence'],
                    'bbox': event['bbox'],
                    'model_label': event.get('model_label', ''),
                }
                decision = self.threat_policy.decide(candidate, person_tracks, (frame_h, frame_w))
                if decision.allow:
                    event['threat_policy'] = 'allowed'
                    event['policy_reason'] = decision.reason
                    filtered_events.append(event)
                else:
                    event['threat_policy'] = 'suppressed'
                    event['policy_reason'] = decision.reason
                    event['suppressed_reason'] = decision.reason  # For backward compatibility with tests
                    self.last_suppressed.append(event)
            else:
                filtered_events.append(event)

        self.last_raw_candidates = all_events.copy()
        return filtered_events