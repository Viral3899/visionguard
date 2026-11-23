"""Person detection counter with 5-minute appearance & track ID deduplication.

Ensures that individuals passing through or present in the frame are counted,
while preventing the exact same individual from being counted again within a
5-minute (300-second) cooldown window.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def extract_person_signature(frame: np.ndarray, bbox: list[float] | tuple[float, ...]) -> np.ndarray | None:
    """Extract an appearance color signature (split upper/lower body HSV histogram) from a person bbox."""
    if frame is None or len(frame.shape) < 2 or not bbox or len(bbox) != 4:
        return None
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        # Check if coordinates are normalized (<= 1.0) or pixel values
        if max(x1, y1, x2, y2) <= 1.05 and (x2 <= 1.0 or y2 <= 1.0):
            px1 = int(max(0, min(w - 1, round(x1 * w))))
            py1 = int(max(0, min(h - 1, round(y1 * h))))
            px2 = int(max(0, min(w, round(x2 * w))))
            py2 = int(max(0, min(h, round(y2 * h))))
        else:
            px1 = int(max(0, min(w - 1, round(x1))))
            py1 = int(max(0, min(h - 1, round(y1))))
            px2 = int(max(0, min(w, round(x2))))
            py2 = int(max(0, min(h, round(y2))))

        if px2 <= px1 + 4 or py2 <= py1 + 8:
            return None

        crop = frame[py1:py2, px1:px2]
        crop_h, crop_w = crop.shape[:2]
        if crop_h < 8 or crop_w < 4:
            return None

        # Convert to HSV color space
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Split into upper body (top 50%) and lower body (bottom 50%)
        mid_y = crop_h // 2
        upper_hsv = hsv[:mid_y, :]
        lower_hsv = hsv[mid_y:, :]

        # 16 Hue bins (0-180), 8 Saturation bins (0-256)
        hist_upper = cv2.calcHist([upper_hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hist_lower = cv2.calcHist([lower_hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])

        cv2.normalize(hist_upper, hist_upper, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_lower, hist_lower, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        sig = np.concatenate([hist_upper.flatten(), hist_lower.flatten()]).astype(np.float32)
        return sig
    except Exception as exc:
        LOGGER.debug("Signature extraction failed: %s", exc)
        return None


def compare_signatures(sig1: np.ndarray | None, sig2: np.ndarray | None) -> float:
    """Compute histogram correlation similarity between two person appearance signatures."""
    if sig1 is None or sig2 is None:
        return 0.0
    try:
        corr = cv2.compareHist(sig1, sig2, cv2.HISTCMP_CORREL)
        return float(corr) if math.isfinite(corr) else 0.0
    except Exception:
        return 0.0


class PersonDeduplicator:
    """Maintains person count with a 5-minute appearance & track ID deduplication window."""

    def __init__(self, cooldown_seconds: float = 300.0, sim_threshold: float = 0.75):
        self.cooldown_seconds = max(10.0, float(cooldown_seconds))
        self.sim_threshold = float(sim_threshold)
        self.current_in_frame: int = 0
        self.total_unique_passers: int = 0
        self._next_profile_id: int = 1
        # Active profiles: profile_id -> dict
        # {
        #   'profile_id': int,
        #   'track_ids': set[int],
        #   'first_seen': float,
        #   'last_seen': float,
        #   'signature': np.ndarray,
        # }
        self.profiles: dict[int, dict[str, Any]] = {}

    def update(self, frame: np.ndarray | None, persons: list[dict], now: float | None = None) -> tuple[int, int]:
        """Update tracker with detections from current frame.

        Parameters:
            frame: Current image frame (BGR)
            persons: List of detected person dicts, e.g. [{'id': 1, 'bbox': [x1, y1, x2, y2]}]
            now: Monotonic timestamp (seconds)

        Returns:
            (current_in_frame, total_unique_passers)
        """
        now = time.monotonic() if now is None else float(now)
        self.current_in_frame = len(persons)

        # 1. Purge profiles whose last sighting is older than the 5-minute cooldown
        expired_ids = [
            pid for pid, prof in self.profiles.items()
            if (now - prof['last_seen']) > self.cooldown_seconds
        ]
        for pid in expired_ids:
            del self.profiles[pid]

        # 2. Process each person in current frame
        for person in persons:
            track_id = int(person.get('id', -1))
            bbox = person.get('bbox')
            sig = extract_person_signature(frame, bbox) if frame is not None and bbox else None

            matched_profile = None

            # Check if this track_id already belongs to an active profile
            for prof in self.profiles.values():
                if track_id in prof['track_ids']:
                    matched_profile = prof
                    break

            # If track_id not found directly, compare visual appearance signature
            # to active profiles within the 5-minute cooldown window
            if matched_profile is None and sig is not None:
                best_sim = 0.0
                best_profile = None
                for prof in self.profiles.values():
                    prof_sig = prof.get('signature')
                    if prof_sig is not None:
                        sim = compare_signatures(prof_sig, sig)
                        if sim > best_sim and sim >= self.sim_threshold:
                            best_sim = sim
                            best_profile = prof

                if best_profile is not None:
                    matched_profile = best_profile
                    if track_id > 0:
                        matched_profile['track_ids'].add(track_id)

            if matched_profile is not None:
                # Existing person within 5 minutes -> update timestamp & running signature, DO NOT count again
                matched_profile['last_seen'] = now
                if sig is not None:
                    old_sig = matched_profile.get('signature')
                    if old_sig is not None:
                        matched_profile['signature'] = (0.7 * old_sig + 0.3 * sig).astype(np.float32)
                    else:
                        matched_profile['signature'] = sig
            else:
                # New individual not seen in the last 5 minutes -> increment unique count!
                self.total_unique_passers += 1
                new_profile = {
                    'profile_id': self._next_profile_id,
                    'track_ids': {track_id} if track_id > 0 else set(),
                    'first_seen': now,
                    'last_seen': now,
                    'signature': sig,
                }
                self.profiles[self._next_profile_id] = new_profile
                self._next_profile_id += 1

        return self.current_in_frame, self.total_unique_passers

    def get_counts(self) -> dict[str, int]:
        return {
            'in_frame': self.current_in_frame,
            'unique_5m': self.total_unique_passers,
        }

    def reset(self):
        self.current_in_frame = 0
        self.total_unique_passers = 0
        self.profiles.clear()
        self._next_profile_id = 1
