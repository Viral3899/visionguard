"""Conservative, auditable weapon-candidate suppression for fixed-camera CCTV.

This is *not* a second trained classifier: geometry and person context only gate
an existing YOLO hypothesis. Rejecting an event never changes model confidence.
All constants are deployment heuristics; calibrate on labeled real positives and
negatives. Suppression trades sensitivity for reduced false alarms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThreatPolicy:
    enabled: bool = True
    require_person: bool = True
    # A weapon's center should be inside/near a person box in a people-focused
    # camera. This misses unattended weapons on desks: disable explicitly there.
    person_margin: float = 0.08
    # Values refer to *raw YOLO confidence*, never fabricated certainty.
    thresholds: dict[str, float] = field(default_factory=lambda: {
        'gun': 0.85,
        'knife': 0.75,
        'weapon': 0.90,  # includes vague 'grenade'/'explosive' categories
    })
    # Normalized predicted box area (fraction of full frame).
    max_areas: dict[str, float] = field(default_factory=lambda: {
        'gun': 0.15,
        'knife': 0.10,
        'weapon': 0.12,
    })
    # Small background pixels are often false positives in generic threat nets.
    min_area: float = 0.00003

    def review(self, event: dict, frame_shape: tuple, persons: list[dict]) -> tuple[bool, str]:
        """Return (passes, reason). Always fail closed for invalid security boxes."""
        if not self.enabled or event.get('type') != 'weapon':
            return True, 'not_applicable'
        subtype = str(event.get('subtype', 'weapon')).lower().strip()
        if subtype not in self.thresholds:
            return False, 'unrecognized_weapon_subtype'
        try:
            conf = float(event['confidence'])
        except (ValueError, TypeError, KeyError):
            return False, 'invalid_confidence'
        if not math.isfinite(conf) or not 0 <= conf <= 1:
            return False, 'invalid_confidence'
        if conf < self.thresholds[subtype]:
            return False, 'below_class_threshold'
        box = event.get('bbox')
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return False, 'invalid_bbox'
        try:
            x1, y1, x2, y2 = map(float, box)
        except (TypeError, ValueError):
            return False, 'invalid_bbox'
        if (not all(math.isfinite(v) and 0 <= v <= 1 for v in (x1,y1,x2,y2))
                or x1 >= x2 or y1 >= y2):
            return False, 'invalid_bbox'
        area = (x2-x1)*(y2-y1)
        if area < self.min_area or area > self.max_areas[subtype]:
            return False, 'implausible_bbox_size'
        if self.require_person:
            h, w = frame_shape[:2]
            cx, cy = (x1+x2)/2, (y1+y2)/2
            associated = False
            for person in persons or ():
                pbox = person.get('bbox')
                if not isinstance(pbox, (list, tuple)) or len(pbox) != 4:
                    continue
                try:
                    px1, py1, px2, py2 = (float(pbox[0])/w, float(pbox[1])/h,
                                          float(pbox[2])/w, float(pbox[3])/h)
                except (ValueError, TypeError, ZeroDivisionError):
                    continue
                if (px1-self.person_margin <= cx <= px2+self.person_margin and
                        py1-self.person_margin <= cy <= py2+self.person_margin):
                    associated = True
                    break
            if not associated:
                return False, 'no_person_near_weapon'
        return True, 'passed'

    def filter(self, events: list[dict], frame_shape: tuple, persons: list[dict]):
        accepted, suppressed = [], []
        for event in events:
            allowed, reason = self.review(event, frame_shape, persons)
            if allowed:
                accepted.append(event)
            else:
                # Preserve raw predictions in diagnostic reports, but never
                # show a suppressed hypothesis as an actual incident.
                suppressed.append({**event, 'suppressed_reason': reason})
        return accepted, suppressed

    def decide(self, candidate: dict, persons: list[dict], frame_shape: tuple = None):
        """Simplified interface for DetectionManager: returns object with .allow and .reason.
        
        Args:
            candidate: Weapon candidate with type, confidence, bbox (normalized [0,1])
            persons: Person tracks with bbox in PIXEL coordinates
            frame_shape: (height, width) of frame for normalizing person bboxes. If None, assumes persons are already normalized.
        """
        from types import SimpleNamespace
        # Create a mock event with the candidate info
        event = {
            'type': 'weapon',
            'subtype': candidate.get('type', 'weapon'),
            'confidence': candidate.get('confidence', 0.0),
            'bbox': candidate.get('bbox', [0, 0, 0, 0]),
        }
        # Use provided frame_shape or default to (1, 1) for backward compatibility
        shape = frame_shape if frame_shape is not None else (1, 1)
        allowed, reason = self.review(event, shape, persons)
        return SimpleNamespace(allow=allowed, reason=reason)
