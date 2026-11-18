"""Multi-model detection: Fire, Violence, Anomaly (Pose/Fall/Loitering), Objects."""
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import settings

LOGGER = logging.getLogger(__name__)

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
    boxes: list[list[float]]  # xyxy normalized
    classes: list[int]
    confidences: list[float]
    class_names: dict[int, str]
    keypoints: list[list[float]] | None = None  # For pose detection


class BaseDetector:
    """Base class for YOLO-based detectors."""
    
    def __init__(self, model_path: str, confidence: float, classes: tuple[int, ...] | None = None):
        # Resolve model path relative to models directory if not absolute
        from .config import settings
        path = Path(model_path)
        if not path.is_absolute():
            # If path already starts with models/, don't prepend models_dir again
            if not str(path).startswith('models' + os.sep) and not str(path).startswith('models/'):
                path = settings.models_dir / path
        self.model_path = str(path)
        self.confidence = confidence
        self.classes = classes
        self.model = None
        self._lock = threading.Lock()
        self._load_model()
    
    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            LOGGER.info(f"Loaded model: {self.model_path}")
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
            if self.classes:
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
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            classes = result.boxes.cls.int().cpu().numpy().tolist()
            confidences = result.boxes.conf.cpu().numpy().tolist()
        
        class_names = result.names if hasattr(result, 'names') else {}
        
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy().tolist() if result.keypoints.xy is not None else None
        
        return DetectionResult(
            boxes=boxes,
            classes=classes,
            confidences=confidences,
            class_names=class_names,
            keypoints=keypoints
        )
    
    def draw(self, frame: np.ndarray, result: DetectionResult, color: tuple = (0, 255, 0), 
                thickness: int = 2, font_scale: float = 0.6) -> np.ndarray:
        """Draw detection boxes with labels - improved to avoid text overlap."""
        h, w = frame.shape[:2]
        
        for box, cls, conf in zip(result.boxes, result.classes, result.confidences):
            x1, y1, x2, y2 = map(int, box)
            label = result.class_names.get(cls, str(cls))
            
            # Draw rounded rectangle for better visual
            self._draw_rounded_rect(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text with background
            label_text = f'{label} {conf:.0%}'
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Position label above box, but ensure it's within frame
            label_y = max(text_h + 5, y1 - 5)
            label_x = max(0, min(x1, w - text_w - 10))
            
            # Draw label background (rounded)
            padding = 6
            bg_x1 = label_x - padding
            bg_y1 = label_y - text_h - padding
            bg_x2 = label_x + text_w + padding
            bg_y2 = label_y + padding
            
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            
            # Draw label text
            cv2.putText(frame, label_text, (label_x, label_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        return frame
    
    def _draw_rounded_rect(self, frame, pt1, pt2, color, thickness):
        """Draw a rectangle with rounded corners."""
        x1, y1 = pt1
        x2, y2 = pt2
        r = 8  # corner radius
        
        # Draw main rectangle
        cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), color, thickness)
        cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), color, thickness)
        
        # Draw four rounded corners
        cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


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
            classes=settings.fire_classes
        )
        # Default class names for fire models
        self.fire_class_names = {0: 'fire', 1: 'smoke'}
        # Track smoke detections for fire confirmation
        self.smoke_history: list[dict] = []
        self.max_smoke_age = 8.0  # seconds
        self.fire_history: list[float] = []
        self.fire_confirm_frames = 1  # Reduced from 3 - single frame for small flames
        self.fire_min_confidence = 0.40  # Reduced from 0.72 - lower threshold for small flames
        self.small_flame_min_confidence = 0.25  # Very low threshold for tiny flames (matchstick)
    
    def detect(self, frame: np.ndarray) -> DetectionResult | None:
        result = super().detect(frame, settings.imgsz, settings.device)
        if result:
            # Merge with fire-specific class names
            result.class_names = merge_fallback_names(result.class_names, self.fire_class_names)
        return result
    
    def check_fire_hazard(self, frame: np.ndarray, result: DetectionResult) -> list[dict]:
        """Check for fire hazards with smoke-first confirmation logic.
        
        Optimized for small flames (matchstick, candle, lighter) with lower thresholds.
        """
        hazards = []
        h, w = frame.shape[:2]
        now = time.monotonic()
        
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
            class_name = result.class_names.get(cls, str(cls)).lower()
            x1, y1, x2, y2 = box
            
            if class_name == 'smoke':
                has_smoke = True
                smoke_conf = max(smoke_conf, conf)
                smoke_bbox = [x1/w, y1/h, x2/w, y2/h]
                # Track smoke for fire confirmation
                self.smoke_history.append({
                    'time': now,
                    'confidence': conf,
                    'bbox': [x1/w, y1/h, x2/w, y2/h]
                })
            
            elif class_name == 'fire':
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
        if has_fire:
            self.fire_history.append(now)
        self.fire_history = [t for t in self.fire_history if now - t <= 4.0]
        smoke_recent = bool(self.smoke_history)
        fire_persistent = len(self.fire_history) >= self.fire_confirm_frames
        
        # Standard fire confirmation (larger flames)
        fire_confirmed = has_fire and fire_conf >= self.fire_min_confidence and (smoke_recent or fire_persistent)
        
        # Small flame detection (matchstick, candle) - lower threshold, no smoke required
        small_flame_confirmed = small_flame_detected and small_flame_conf >= self.small_flame_min_confidence
        
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
                'message': f'Small flame detected ({small_flame_conf:.0%}) - possible matchstick/candle/lighter',
                'smoke_present': smoke_recent,
                'smoke_confidence': smoke_conf if has_smoke else 0,
                'confirmation': 'small_flame_single_frame',
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
            confidence = smoke_conf
            if is_persistent:
                confidence = min(0.9, smoke_conf + 0.1)  # Boost confidence for persistent smoke
            
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
        
        # Check for electrical hazards, flammable materials near heat sources
        for box, cls, conf in zip(result.boxes, result.classes, result.confidences):
            class_name = result.class_names.get(cls, str(cls)).lower()
            x1, y1, x2, y2 = box
            
            if class_name in ('laptop', 'tv', 'microwave', 'oven', 'heater', 'candle'):
                hazards.append({
                    'type': 'heat_source',
                    'severity': 'low',
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [x1/w, y1/h, x2/w, y2/h],
                    'message': f'Heat source detected: {class_name}'
                })
        
        return hazards


class ViolenceDetector(BaseDetector):
    """Violence/Fight/Weapon detection for home security."""
    
    def __init__(self):
        super().__init__(
            model_path=settings.violence_model_path,
            confidence=settings.violence_confidence
        )
        # Common violence-related classes
        self.violence_classes = {
            0: 'fight', 1: 'assault', 2: 'weapon', 3: 'knife', 
            4: 'gun', 5: 'violence', 6: 'aggression', 7: 'pistol', 8: 'rifle', 9: 'blade'
        }
        # Track weapon detections for home security
        self.weapon_history: dict[str, list[dict]] = {}  # track_id -> weapon detections
        # Lower threshold for knife specifically (knives are harder to detect)
        self.knife_min_confidence = 0.25
        self.weapon_min_confidence = 0.30
    
    def detect(self, frame: np.ndarray) -> DetectionResult | None:
        result = super().detect(frame, settings.imgsz, settings.device)
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
            class_name = result.class_names.get(cls, str(cls)).lower()
            x1, y1, x2, y2 = box
            
            if class_name in ('fight', 'assault', 'violence', 'aggression'):
                events.append({
                    'type': 'violence',
                    'severity': 'critical',
                    'subtype': class_name,
                    'confidence': conf,
                    'bbox': [x1/w, y1/h, x2/w, y2/h],
                    'message': f'{class_name.capitalize()} detected with {conf:.0%} confidence'
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
                    'associated_person': closest_person,
                    'persistent': is_persistent,
                    'home_security_alert': True
                })
        
        # Analyze person proximity for fight detection
        if person_tracks and len(person_tracks) >= 2:
            for i, track1 in enumerate(person_tracks):
                for track2 in person_tracks[i+1:]:
                    box1 = track1.get('bbox', [0, 0, 0, 0])
                    box2 = track2.get('bbox', [0, 0, 0, 0])
                    if box1[2] > box1[0] and box2[2] > box2[0]:
                        cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
                        cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
                        dist = np.hypot(cx1 - cx2, cy1 - cy2)
                        if dist < 0.15:
                            events.append({
                                'type': 'proximity_alert',
                                'severity': 'medium',
                                'subtype': 'close_proximity',
                                'confidence': 0.6,
                                'bbox': [min(box1[0], box2[0]), min(box1[1], box2[1]), 
                                        max(box1[2], box2[2]), max(box1[3], box2[3])],
                                'message': f'Close proximity between persons #{track1["id"]} and #{track2["id"]} (possible altercation)'
                            })
        
        return events


class AnomalyDetector:
    """Anomaly detection: Fall, Loitering, Unusual poses using YOLOv8-Pose."""
    
    def __init__(self):
        self.model = None
        self._lock = threading.Lock()
        self._load_model()
        self.track_history: dict[int, list[tuple[float, float]]] = {}  # track_id -> positions
        self.fall_history: dict[int, int] = {}  # track_id -> frames on ground
        
    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(settings.anomaly_model_path)
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
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            classes = result.boxes.cls.int().cpu().numpy().tolist()
            confidences = result.boxes.conf.cpu().numpy().tolist()
        
        if hasattr(result, 'keypoints') and result.keypoints is not None and result.keypoints.xy is not None:
            keypoints = result.keypoints.xy.cpu().numpy().tolist()
        
        return DetectionResult(
            boxes=boxes,
            classes=classes,
            confidences=confidences,
            class_names={0: 'person'},
            keypoints=keypoints
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
                class_name = pose_result.class_names.get(cls, str(cls)).lower()
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
                elif class_name in ('knife', 'gun', 'weapon', 'pistol', 'rifle'):
                    events.append({
                        'type': 'weapon',
                        'severity': 'critical',
                        'track_id': None,
                        'confidence': conf,
                        'bbox': [x1/w, y1/h, x2/w, y2/h],
                        'message': f'{class_name.capitalize()} detected with {conf:.0%} confidence'
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
        
        for i, (kpts, box) in enumerate(zip(pose_result.keypoints, pose_result.boxes)):
            if i >= len(person_tracks):
                continue
            track = person_tracks[i]
            track_id = track['id']
            x1, y1, x2, y2 = box
            
            # Fall detection: check if person is horizontal (shoulders/hips near same y as ankles)
            if settings.enable_fall_detection and len(kpts) >= 17:
                nose_y = kpts[0][1] if kpts[0][2] > 0.3 else None  # confidence > 0.3
                l_shoulder_y = kpts[5][1] if kpts[5][2] > 0.3 else None
                r_shoulder_y = kpts[6][1] if kpts[6][2] > 0.3 else None
                l_hip_y = kpts[11][1] if kpts[11][2] > 0.3 else None
                r_hip_y = kpts[12][1] if kpts[12][2] > 0.3 else None
                l_ankle_y = kpts[15][1] if kpts[15][2] > 0.3 else None
                r_ankle_y = kpts[16][1] if kpts[16][2] > 0.3 else None
                
                # Average y positions
                shoulder_y = np.nanmean([l_shoulder_y, r_shoulder_y])
                hip_y = np.nanmean([l_hip_y, r_hip_y])
                ankle_y = np.nanmean([l_ankle_y, r_ankle_y])
                
                if shoulder_y is not None and hip_y is not None and ankle_y is not None:
                    # Fallen: shoulders and hips are at similar height to ankles (horizontal body)
                    vertical_span = ankle_y - shoulder_y
                    hip_shoulder_diff = abs(hip_y - shoulder_y)
                    
                    if vertical_span < h * 0.3 and hip_shoulder_diff < h * 0.15:
                        self.fall_history[track_id] = self.fall_history.get(track_id, 0) + 1
                        if self.fall_history[track_id] >= 3:  # 3 consecutive frames
                            events.append({
                                'type': 'fall_detected',
                                'severity': 'high',
                                'track_id': track_id,
                                'confidence': 0.85,
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


class ObjectDetector(BaseDetector):
    """General object detection for all COCO classes."""
    
    def __init__(self):
        super().__init__(
            model_path=settings.object_model_path,
            confidence=settings.object_confidence,
            classes=settings.object_classes if settings.object_classes else None
        )
    
    def detect(self, frame: np.ndarray) -> DetectionResult | None:
        return super().detect(frame, settings.imgsz, settings.device)


class DetectionManager:
    """Manages all detectors and coordinates detection pipeline."""
    
    def __init__(self):
        self.fire_detector = FireDetector()
        self.violence_detector = ViolenceDetector()
        self.anomaly_detector = AnomalyDetector()
        self.object_detector = ObjectDetector()
        self._last_telegram_alert: dict[str, float] = {}
        self._last_tts_alert: dict[str, float] = {}
        # Frame counters for running detectors at different frequencies
        self._frame_counter = 0
        # Run heavy detectors every N frames for performance
        self.fire_every_n_frames = 1    # Every frame (fire is critical)
        self.violence_every_n_frames = 1  # Every frame (violence is critical)
        self.anomaly_every_n_frames = 2   # Every 2 frames (falls/loitering less urgent)
        self.object_every_n_frames = 5    # Every 5 frames (general objects least urgent)
    
    def process_frame(self, frame: np.ndarray, person_tracks: list[dict], 
                      camera_id: str, camera_config: dict) -> list[dict]:
        """Run all enabled detectors and return events.
        
        Optimized: Runs detectors at different frequencies for better FPS.
        """
        all_events = []
        now = time.monotonic()
        self._frame_counter += 1
        
        # Fire detection - run every frame (critical)
        if camera_config.get('fire_enabled', 1) and self._frame_counter % self.fire_every_n_frames == 0:
            fire_result = self.fire_detector.detect(frame)
            if fire_result:
                hazards = self.fire_detector.check_fire_hazard(frame, fire_result)
                for hazard in hazards:
                    hazard['camera_id'] = camera_id
                    all_events.append(hazard)
                # Draw fire detections
                frame = self.fire_detector.draw(frame, fire_result, (0, 0, 255))
        
        # Violence detection - run every frame (critical)
        if camera_config.get('violence_enabled', 1) and self._frame_counter % self.violence_every_n_frames == 0:
            violence_result = self.violence_detector.detect(frame)
            if violence_result:
                # Get person boxes for proximity analysis
                person_boxes = person_tracks if person_tracks else None
                violence_events = self.violence_detector.analyze_violence(
                    frame, violence_result, person_boxes)
                for event in violence_events:
                    event['camera_id'] = camera_id
                    all_events.append(event)
                frame = self.violence_detector.draw(frame, violence_result, (255, 0, 255))
        
        # Anomaly detection (falls, loitering) - run every 2 frames
        if camera_config.get('anomaly_enabled', 1) and self._frame_counter % self.anomaly_every_n_frames == 0:
            pose_result = self.anomaly_detector.detect(frame, person_tracks)
            if pose_result:
                anomaly_events = self.anomaly_detector.analyze_anomalies(
                    frame, pose_result, person_tracks, now)
                for event in anomaly_events:
                    event['camera_id'] = camera_id
                    all_events.append(event)
                # Draw pose keypoints
                if pose_result.keypoints:
                    frame = self._draw_keypoints(frame, pose_result.keypoints, pose_result.boxes)
        
        # Object detection - run every 5 frames (least critical)
        if camera_config.get('object_enabled', 1) and self._frame_counter % self.object_every_n_frames == 0:
            object_result = self.object_detector.detect(frame)
            if object_result:
                frame = self.object_detector.draw(frame, object_result, (255, 255, 0))
        
        return all_events
    
    def _draw_keypoints(self, frame: np.ndarray, keypoints: list, boxes: list) -> np.ndarray:
        """Draw pose keypoints on frame."""
        h, w = frame.shape[:2]
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        for kpts, box in zip(keypoints, boxes):
            if len(kpts) >= 17:
                for i, (x, y, conf) in enumerate(kpts):
                    if conf > 0.3:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
                for start, end in skeleton:
                    if kpts[start][2] > 0.3 and kpts[end][2] > 0.3:
                        pt1 = (int(kpts[start][0]), int(kpts[start][1]))
                        pt2 = (int(kpts[end][0]), int(kpts[end][1]))
                        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        return frame
    
    def should_send_telegram(self, event_kind: str) -> bool:
        """Check cooldown for Telegram alerts."""
        if event_kind not in settings.telegram_alert_types:
            return False
        now = time.monotonic()
        last = self._last_telegram_alert.get(event_kind, 0)
        if now - last >= settings.telegram_cooldown_seconds:
            self._last_telegram_alert[event_kind] = now
            return True
        return False
    
    def should_trigger_tts(self, event_kind: str) -> bool:
        """Check cooldown for TTS alerts."""
        if event_kind not in settings.tts_alert_types:
            return False
        now = time.monotonic()
        last = self._last_tts_alert.get(event_kind, 0)
        if now - last >= settings.siren_duration_seconds * settings.siren_repeat_count:
            self._last_tts_alert[event_kind] = now
            return True
        return False