"""One evidence-backed snapshot/notification per unusual incident, not per frame.

All evidence confidences are raw model scores. We do not manufacture certainty
from motion, colors, time near another person, or repeated low-score detections.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

ALERTABLE_KINDS = frozenset({
    'fire_detected', 'smoke', 'violence', 'weapon', 'fall_detected',
    'unusual_activity',
})
_PRIORITY = {'fire_detected': 4, 'smoke': 2, 'violence': 3, 'weapon': 3,
             'fall_detected': 3, 'unusual_activity': 3}


def eligible_event(event: dict, minimum: float = 0.50) -> bool:
    if not isinstance(event, dict) or event.get('type', event.get('kind')) not in ALERTABLE_KINDS:
        return False
    try:
        confidence = float(event['confidence'])
        return math.isfinite(confidence) and minimum <= confidence <= 1.0
    except (KeyError, ValueError, TypeError):
        return False


def incident_key(event: dict) -> str:
    kind = event.get('type', event.get('kind'))
    # A fire often starts as smoke. Avoid storing two stills for that single scene,
    # but allow a new high-severity FIRE alert after an earlier smoke-only warning.
    if kind in ('fire_detected', 'smoke'):
        return 'fire_hazard'
    if kind == 'weapon':
        return 'weapon:' + str(event.get('subtype', 'weapon')).lower()
    if kind in ('fall_detected', 'unusual_activity') and event.get('track_id') is not None:
        return f'{kind}:{event["track_id"]}'
    return str(kind)


@dataclass
class _State:
    count: int
    last_seen: float
    emitted_priority: int = 0
    missed_frames: int = 0
    bbox: tuple | None = None


def same_scene_box(a, b):
    """Don't confirm separate objects at unrelated image locations as one threat.

    Allow overlapping boxes OR nearby centers (small object location jitter).
    Boxes are normalized; temporal full-frame classification naturally matches.
    """
    if a is None or b is None:
        return True  # legacy events without geometry should still be processed
    try:
        ax, ay, bx, by = [float(x) for x in a]
        cx, cy, dx, dy = [float(x) for x in b]
        if not all(math.isfinite(x) for x in (ax,ay,bx,by,cx,cy,dx,dy)):
            return False
        left, top = max(ax,cx), max(ay,cy)
        right, bottom = min(bx,dx), min(by,dy)
        intersection = max(0,right-left) * max(0,bottom-top)
        aarea = max(0,bx-ax) * max(0,by-ay)
        barea = max(0,dx-cx) * max(0,dy-cy)
        union = aarea+barea-intersection
        if union > 0 and intersection/union >= .08:
            return True
        acx, acy, bcx, bcy = (ax+bx)/2,(ay+by)/2,(cx+dx)/2,(cy+dy)/2
        # Scale tolerance to box size to admit a moving person while
        # refusing a knife on the opposite side of the frame.
        tolerance = max(.08, .6 * max(bx-ax, by-ay, dx-cx, dy-cy))
        return math.hypot(acx-bcx, acy-bcy) <= min(tolerance,.28)
    except (TypeError, ValueError):
        return False


class IncidentTracker:
    """Group recurring raw detections; emit once per active scene and on escalation.

    update() must be called for each *analyzed* frame, including empty detections.
    Uses timestamps from the video for offline tests, wall-clock for live streams.
    """
    def __init__(self, minimum: float = 0.50, confirmations: int = 2,
                 max_gap_seconds: float = 10.0, max_missed_frames: int = 2):
        self.minimum = minimum
        self.confirmations = max(1, confirmations)
        self.max_gap_seconds = max_gap_seconds
        self.max_missed_frames = max(0, max_missed_frames)
        self._states: dict[str, _State] = {}
        # Current *confirmed* events for video display; notifications emit only once.
        self.active_events: list[dict] = []

    def update(self, events: list[dict], now: float) -> list[dict]:
        # One best-supported event per threat type per analyzed frame.
        candidates: dict[str, dict] = {}
        for event in events:
            if not eligible_event(event, self.minimum):
                continue
            key = incident_key(event)
            prior = candidates.get(key)
            score = (_PRIORITY.get(event['type'], 1), float(event['confidence']))
            if prior is None or score > (_PRIORITY.get(prior['type'], 1), float(prior['confidence'])):
                candidates[key] = event

        for key, state in list(self._states.items()):
            if key not in candidates:
                state.missed_frames += 1
            if (state.missed_frames > self.max_missed_frames
                    or now - state.last_seen > self.max_gap_seconds):
                del self._states[key]

        confirmed = []
        self.active_events = []
        for key, event in candidates.items():
            state = self._states.get(key)
            if (state is None or now - state.last_seen > self.max_gap_seconds or
                    not same_scene_box(state.bbox, event.get('bbox'))):
                state = _State(0, now)
                self._states[key] = state
            # The optional 3D classifier has already required two different
            # 16-frame windows before producing this evidence. Do not require
            # it to emit on adjacent frames (it samples every N frames).
            if (event.get('source_model') == 'torch_r3d18_fight'
                    and event.get('subtype') == 'temporal_fight_classifier'
                    and event.get('temporal_confirmed') is True):
                state.count = max(state.count, self.confirmations)
            else:
                state.count += 1
            state.last_seen = now
            if isinstance(event.get('bbox'), (tuple, list)):
                state.bbox = tuple(event['bbox'])
            state.missed_frames = 0
            level = _PRIORITY.get(event['type'], 1)
            if state.count >= self.confirmations:
                self.active_events.append(event)
            if state.count >= self.confirmations and level > state.emitted_priority:
                state.emitted_priority = level
                confirmed.append(event)
        return confirmed


def draw_event_box(frame: np.ndarray, event: dict) -> np.ndarray:
    """Annotate the actual incident box; avoid writing generic normal frames."""
    bbox = event.get('bbox')
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return frame
    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return frame
        x1, x2 = sorted((max(0, min(w-1, int(x1*w))), max(0, min(w-1, int(x2*w)))))
        y1, y2 = sorted((max(0, min(h-1, int(y1*h))), max(0, min(h-1, int(y2*h)))))
        if x2 <= x1 or y2 <= y1:
            return frame
        color = (0, 0, 255) if event.get('type') != 'weapon' else (255, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        subtype = str(event.get('subtype', event.get('type', 'candidate'))).upper()
        label = f'REVIEW {subtype} {float(event["confidence"]):.0%}'
        cv2.putText(frame, label, (x1, max(20, y1 - 9)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 2, cv2.LINE_AA)
    except (ValueError, TypeError):
        return frame
    return frame
