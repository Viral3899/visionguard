"""VisionGuard Enterprise Business Suite.

Addresses the five core business and commercial surveillance challenges:
1. Inventory Shrinkage & Shoplifting (LossPreventionEngine)
2. Perimeter Security Trespassing & Tailgating (PerimeterTrespassEngine)
3. Worker Accidents, Slips & Prolonged Immobility (WorkerSafetyEngine)
4. Foot Traffic Blindspots & Spatial Dwell Heatmap (FootTrafficAnalytics)
5. Automated IT Deployment Diagnostics (High IT Cost Elimination)
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# Classes associated with retail merchandise and theft-sensitive inventory
RETAIL_HIGH_VALUE_CLASSES = frozenset({
    'cell phone', 'laptop', 'mouse', 'keyboard', 'bottle', 'cup',
    'handbag', 'backpack', 'suitcase', 'book', 'umbrella'
})


def box_iou(box_a: list | tuple, box_b: list | tuple) -> float:
    """Compute Intersection-over-Union between two boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter_area = iw * ih
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def box_contains_point(box: list | tuple, pt: tuple[float, float]) -> bool:
    """Check if point (px, py) is inside bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    px, py = pt
    return (x1 <= px <= x2) and (y1 <= py <= y2)


class LossPreventionEngine:
    """Detects inventory shrinkage, merchandise concealment, and grab-and-conceal patterns.

    Tracks high-value inventory items in relation to nearby individuals. If an item
    was in close proximity/overlap with a person or bag and vanishes from detection while
    the person remains in the scene, a concealment event is flagged.
    """

    def __init__(self, proximity_threshold: float = 120.0, concealment_frames: int = 3, cooldown_seconds: float = 8.0):
        self.proximity_threshold = proximity_threshold
        self.concealment_frames = concealment_frames
        self.cooldown_seconds = cooldown_seconds
        # track item key -> {'last_seen': float, 'last_box': list, 'class_name': str, 'near_person_id': int, 'missed_frames': int}
        self.active_items: dict[str, dict[str, Any]] = {}
        # person_id -> timestamp of last alert
        self.last_alerts: dict[int, float] = {}

    def process(
        self,
        frame_shape: tuple[int, int],
        detected_objects: list[dict],
        persons: list[dict],
        now: float | None = None
    ) -> list[dict]:
        """Analyze frame for suspicious concealment and inventory shrinkage."""
        if now is None:
            now = time.monotonic()
        h, w = frame_shape[:2]
        events: list[dict] = []

        current_item_keys = set()

        for obj in detected_objects:
            cname = str(obj.get('class_name', '')).strip().lower()
            if cname not in RETAIL_HIGH_VALUE_CLASSES:
                continue

            raw_bbox = obj.get('bbox')
            if not raw_bbox or len(raw_bbox) != 4:
                continue

            # Normalized or absolute coordinates
            x1, y1, x2, y2 = raw_bbox
            if max(x1, x2) <= 1.0 and max(y1, y2) <= 1.0:
                abs_box = [x1 * w, y1 * h, x2 * w, y2 * h]
            else:
                abs_box = [float(x1), float(y1), float(x2), float(y2)]

            obj_cx = (abs_box[0] + abs_box[2]) / 2.0
            obj_cy = (abs_box[1] + abs_box[3]) / 2.0
            obj_key = f"{cname}_{int(obj_cx // 40)}_{int(obj_cy // 40)}"
            current_item_keys.add(obj_key)

            # Check association with any person in scene
            associated_person_id = None
            for p in persons:
                p_box = p.get('bbox')
                if not p_box or len(p_box) != 4:
                    continue
                px1, py1, px2, py2 = p_box
                if max(px1, px2) <= 1.0 and max(py1, py2) <= 1.0:
                    p_abs = [px1 * w, py1 * h, px2 * w, py2 * h]
                else:
                    p_abs = [float(px1), float(py1), float(px2), float(py2)]

                p_cx = (p_abs[0] + p_abs[2]) / 2.0
                p_cy = (p_abs[1] + p_abs[3]) / 2.0
                dist = math.hypot(obj_cx - p_cx, obj_cy - p_cy)

                iou = box_iou(abs_box, p_abs)
                if iou > 0.05 or dist < self.proximity_threshold:
                    associated_person_id = p.get('id', 0)
                    break

            if obj_key in self.active_items:
                self.active_items[obj_key]['last_seen'] = now
                self.active_items[obj_key]['last_box'] = abs_box
                self.active_items[obj_key]['missed_frames'] = 0
                if associated_person_id is not None:
                    self.active_items[obj_key]['near_person_id'] = associated_person_id
            else:
                self.active_items[obj_key] = {
                    'class_name': cname,
                    'last_seen': now,
                    'last_box': abs_box,
                    'near_person_id': associated_person_id,
                    'missed_frames': 0
                }

        # Check items that vanished while near an active person
        keys_to_delete = []
        for key, item in self.active_items.items():
            if key not in current_item_keys:
                item['missed_frames'] += 1
                person_id = item.get('near_person_id')

                # If item disappeared while associated with person and missed frames >= threshold
                if person_id is not None and item['missed_frames'] >= self.concealment_frames:
                    # Verify person is still present
                    active_person = next((p for p in persons if p.get('id') == person_id), None)
                    if active_person is not None:
                        last_alert_time = self.last_alerts.get(person_id, 0.0)
                        if (now - last_alert_time) >= self.cooldown_seconds:
                            self.last_alerts[person_id] = now
                            p_box = active_person.get('bbox', item['last_box'])
                            events.append({
                                'type': 'inventory_shrinkage',
                                'severity': 'high',
                                'confidence': 0.88,
                                'person_id': person_id,
                                'item': item['class_name'],
                                'details': f"Suspected shoplifting / shrinkage: {item['class_name']} concealed by Person #{person_id}",
                                'timestamp': now,
                                'bbox': p_box
                            })
                    keys_to_delete.append(key)
                elif item['missed_frames'] > 15:
                    keys_to_delete.append(key)

        for key in keys_to_delete:
            self.active_items.pop(key, None)

        return events


class PerimeterTrespassEngine:
    """Monitors perimeter boundaries for directional trespass and tailgating (piggybacking).

    - Directional Tripwire: Differentiates authorized egress from unauthorized ingress.
    - Tailgating: Flags when multiple individuals cross a boundary or access point within < 2.0s.
    """

    def __init__(self, tailgating_gap_seconds: float = 2.0, cooldown_seconds: float = 5.0):
        self.tailgating_gap_seconds = tailgating_gap_seconds
        self.cooldown_seconds = cooldown_seconds
        # track person_id -> history of centroids
        self.person_positions: dict[int, list[tuple[float, float, float]]] = {}
        # record crossings: list of (timestamp, person_id, pt)
        self.recent_crossings: list[tuple[float, int, tuple[float, float]]] = []
        self.last_tailgate_alert: float = 0.0
        self.last_trespass_alerts: dict[int, float] = {}

    def _crosses_line(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        q1: tuple[float, float],
        q2: tuple[float, float]
    ) -> tuple[bool, float]:
        """Check if segment (p1, p2) intersects segment (q1, q2).

        Returns (intersected, cross_product_direction).
        """
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        intersect = (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))
        # Direction: vector (p2 - p1) cross line vector (q2 - q1)
        dir_cross = (p2[0] - p1[0]) * (q2[1] - q1[1]) - (p2[1] - p1[1]) * (q2[0] - q1[0])
        return intersect, dir_cross

    def process(
        self,
        frame_shape: tuple[int, int],
        persons: list[dict],
        tripwire: list | tuple | None = None,
        now: float | None = None
    ) -> list[dict]:
        """Analyze perimeter security, directional vector crossings, and tailgating."""
        if now is None:
            now = time.monotonic()
        h, w = frame_shape[:2]
        events: list[dict] = []

        if not tripwire or len(tripwire) < 2:
            return events

        # Parse tripwire coordinates in normalized or absolute form
        (t1x, t1y), (t2x, t2y) = tripwire[0], tripwire[1]
        line_p1 = (t1x * w if t1x <= 1.0 else t1x, t1y * h if t1y <= 1.0 else t1y)
        line_p2 = (t2x * w if t2x <= 1.0 else t2x, t2y * h if t2y <= 1.0 else t2y)

        for person in persons:
            p_id = person.get('id', 0)
            foot = person.get('foot')
            if not foot:
                bbox = person.get('bbox', [0, 0, 0, 0])
                foot = ((bbox[0] + bbox[2]) / 2.0, bbox[3])

            fx = foot[0] * w if foot[0] <= 1.0 else foot[0]
            fy = foot[1] * h if foot[1] <= 1.0 else foot[1]
            current_pt = (fx, fy)

            hist = self.person_positions.setdefault(p_id, [])
            hist.append((now, current_pt[0], current_pt[1]))
            # Keep up to 30 past points
            if len(hist) > 30:
                hist.pop(0)

            if len(hist) >= 2:
                prev_t, prev_x, prev_y = hist[-2]
                crossed, dir_cross = self._crosses_line((prev_x, prev_y), current_pt, line_p1, line_p2)
                if crossed:
                    last_alert = self.last_trespass_alerts.get(p_id, 0.0)
                    if (now - last_alert) >= self.cooldown_seconds:
                        self.last_trespass_alerts[p_id] = now
                        direction_str = "ingress" if dir_cross > 0 else "egress"
                        events.append({
                            'type': 'perimeter_trespass',
                            'severity': 'critical',
                            'confidence': 0.90,
                            'person_id': p_id,
                            'direction': direction_str,
                            'details': f"Perimeter breach: Person #{p_id} {direction_str} crossing boundary",
                            'timestamp': now,
                            'bbox': person.get('bbox')
                        })
                        self.recent_crossings.append((now, p_id, current_pt))

        # Prune crossing log older than 10 seconds
        self.recent_crossings = [(t, pid, pt) for t, pid, pt in self.recent_crossings if now - t <= 10.0]

        # Check for Tailgating (Piggybacking)
        # If two distinct persons crossed line within tailgating_gap_seconds
        if len(self.recent_crossings) >= 2 and (now - self.last_tailgate_alert) >= self.cooldown_seconds:
            for i in range(len(self.recent_crossings) - 1):
                t_a, id_a, pt_a = self.recent_crossings[i]
                t_b, id_b, pt_b = self.recent_crossings[i + 1]
                if id_a != id_b and (t_b - t_a) <= self.tailgating_gap_seconds:
                    self.last_tailgate_alert = now
                    events.append({
                        'type': 'tailgating_detected',
                        'severity': 'critical',
                        'confidence': 0.92,
                        'details': f"Tailgating breach: Person #{id_b} piggybacked Person #{id_a} ({t_b - t_a:.2f}s gap)",
                        'person_ids': [id_a, id_b],
                        'timestamp': now
                    })
                    break

        return events


class WorkerSafetyEngine:
    """Detects industrial worker accidents, slips, falls, and prolonged immobility.

    - Impact & Fall Detection: Rapid downward displacement followed by horizontal aspect ratio inversion.
    - Prolonged Immobility: Escalates alert to unresponsive worker if fallen person remains recumbent > 5.0s.
    """

    def __init__(
        self,
        fall_aspect_ratio_min: float = 1.15,
        fall_velocity_threshold: float = 80.0,
        immobility_seconds: float = 5.0,
        cooldown_seconds: float = 10.0
    ):
        self.fall_aspect_ratio_min = fall_aspect_ratio_min
        self.fall_velocity_threshold = fall_velocity_threshold
        self.immobility_seconds = immobility_seconds
        self.cooldown_seconds = cooldown_seconds
        # person_id -> {'history': [(t, cy, w, h)], 'fallen_since': float | None, 'last_alert': float}
        self.person_state: dict[int, dict[str, Any]] = {}

    def process(
        self,
        frame_shape: tuple[int, int],
        persons: list[dict],
        now: float | None = None
    ) -> list[dict]:
        """Evaluate worker postures, downward velocities, and post-fall immobility."""
        if now is None:
            now = time.monotonic()
        h, w = frame_shape[:2]
        events: list[dict] = []

        seen_ids = set()

        for person in persons:
            p_id = person.get('id', 0)
            seen_ids.add(p_id)
            bbox = person.get('bbox')
            if not bbox or len(bbox) != 4:
                continue

            bx1, by1, bx2, by2 = bbox
            if max(bx1, bx2) <= 1.0 and max(by1, by2) <= 1.0:
                abs_box = [bx1 * w, by1 * h, bx2 * w, by2 * h]
            else:
                abs_box = [float(bx1), float(by1), float(bx2), float(by2)]

            bw = max(1.0, abs_box[2] - abs_box[0])
            bh = max(1.0, abs_box[3] - abs_box[1])
            cy = (abs_box[1] + abs_box[3]) / 2.0
            aspect_ratio = bw / bh

            state = self.person_state.setdefault(p_id, {
                'history': [],
                'fallen_since': None,
                'last_alert': 0.0,
                'last_immobility_alert': 0.0,
                'last_cy': cy
            })

            hist = state['history']
            hist.append((now, cy, bw, bh))
            if len(hist) > 20:
                hist.pop(0)

            # Check for sudden fall kinematics
            is_fallen_posture = aspect_ratio >= self.fall_aspect_ratio_min

            # Velocity check from previous frames
            vertical_drop_velocity = 0.0
            if len(hist) >= 2:
                prev_t, prev_cy, _, _ = hist[-2]
                dt = max(0.001, now - prev_t)
                vertical_drop_velocity = (cy - prev_cy) / dt

            # If sudden drop or recumbent posture occurs
            if (is_fallen_posture and vertical_drop_velocity > self.fall_velocity_threshold) or (is_fallen_posture and state['fallen_since'] is None):
                if state['fallen_since'] is None:
                    state['fallen_since'] = now
                    if (now - state['last_alert']) >= self.cooldown_seconds:
                        state['last_alert'] = now
                        events.append({
                            'type': 'worker_slip_fall',
                            'severity': 'critical',
                            'confidence': 0.90,
                            'person_id': p_id,
                            'details': f"Worker slip/fall detected for Person #{p_id} (AR={aspect_ratio:.2f})",
                            'timestamp': now,
                            'bbox': abs_box
                        })
            elif not is_fallen_posture:
                # Recovered or standing back up
                state['fallen_since'] = None

            # Prolonged immobility / unresponsiveness
            if state['fallen_since'] is not None:
                duration_down = now - state['fallen_since']
                if duration_down >= self.immobility_seconds:
                    if (now - state['last_immobility_alert']) >= self.cooldown_seconds:
                        state['last_immobility_alert'] = now
                        events.append({
                            'type': 'unresponsive_worker',
                            'severity': 'critical',
                            'confidence': 0.95,
                            'person_id': p_id,
                            'duration_seconds': round(duration_down, 1),
                            'details': f"EMERGENCY: Worker #{p_id} unresponsive on floor for {duration_down:.1f}s",
                            'timestamp': now,
                            'bbox': abs_box
                        })

        # Clean up stale persons
        stale_ids = [pid for pid in self.person_state if pid not in seen_ids]
        for pid in stale_ids:
            self.person_state.pop(pid, None)

        return events


class FootTrafficAnalytics:
    """Foot traffic density tracker, dwell-time accumulator, and blindspot classifier.

    Divides camera view into an 8x8 spatial occupancy grid.
    Accumulates dwell seconds per cell, flags unmonitored blindspot deadzones,
    and generates translucent thermal heatmap overlays.
    """

    def __init__(self, grid_size: tuple[int, int] = (8, 8), decay_factor: float = 0.999):
        self.rows, self.cols = grid_size
        self.decay_factor = decay_factor
        self.occupancy_grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.last_update_time: float = time.monotonic()
        self.total_foot_traffic: int = 0

    def update(self, frame_shape: tuple[int, int], persons: list[dict], now: float | None = None) -> np.ndarray:
        """Update occupancy grid with active person locations."""
        if now is None:
            now = time.monotonic()
        dt = max(0.01, min(1.0, now - self.last_update_time))
        self.last_update_time = now

        # Slight exponential decay to reflect recent foot traffic trends
        self.occupancy_grid *= (self.decay_factor ** dt)

        h, w = frame_shape[:2]
        for person in persons:
            foot = person.get('foot')
            if not foot:
                bbox = person.get('bbox', [0, 0, 0, 0])
                foot = ((bbox[0] + bbox[2]) / 2.0, bbox[3])

            fx = foot[0] if foot[0] <= 1.0 else foot[0] / max(1, w)
            fy = foot[1] if foot[1] <= 1.0 else foot[1] / max(1, h)

            c = min(self.cols - 1, max(0, int(fx * self.cols)))
            r = min(self.rows - 1, max(0, int(fy * self.rows)))
            self.occupancy_grid[r, c] += float(dt)
            self.total_foot_traffic += 1

        return self.occupancy_grid

    def get_blindspots(self, threshold_ratio: float = 0.15) -> list[dict]:
        """Identify cold-zone blindspots that lack normal pedestrian coverage."""
        total = float(np.sum(self.occupancy_grid))
        if total <= 0.05:
            return []

        mean_val = float(np.mean(self.occupancy_grid))
        cold_threshold = mean_val * threshold_ratio
        blindspots = []

        for r in range(self.rows):
            for c in range(self.cols):
                val = float(self.occupancy_grid[r, c])
                # Check corners and perimeter cells with low traffic
                if val <= cold_threshold:
                    blindspots.append({
                        'grid_coord': (r, c),
                        'dwell_seconds': round(val, 2),
                        'is_corner': (r in (0, self.rows - 1) and c in (0, self.cols - 1))
                    })
        return blindspots

    def get_summary(self) -> dict:
        """Return structured summary of foot traffic analytics."""
        max_dwell = float(np.max(self.occupancy_grid))
        avg_dwell = float(np.mean(self.occupancy_grid))
        blindspots = self.get_blindspots()
        return {
            'grid_size': (self.rows, self.cols),
            'max_dwell_seconds': round(max_dwell, 1),
            'average_dwell_seconds': round(avg_dwell, 1),
            'blindspot_cells_count': len(blindspots),
            'blindspots': blindspots[:6]
        }

    def render_heatmap(self, frame: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        """Generate and blend a thermal JET colormap heatmap onto the video frame."""
        if frame is None:
            return frame
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return frame

        max_val = np.max(self.occupancy_grid)
        if max_val > 0.01:
            norm_grid = (self.occupancy_grid / max_val * 255.0).astype(np.uint8)
        else:
            norm_grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # Upsample grid smoothly to frame resolution
        upscaled = cv2.resize(norm_grid, (w, h), interpolation=cv2.INTER_CUBIC)
        color_heatmap = cv2.applyColorMap(upscaled, cv2.COLORMAP_JET)

        # Create mask so zero-traffic regions remain fully transparent
        mask = (upscaled > 12).astype(np.uint8)
        heatmap_masked = cv2.bitwise_and(color_heatmap, color_heatmap, mask=mask)

        # Blend
        blended = frame.copy()
        cv2.addWeighted(heatmap_masked, alpha, blended, 1.0 - (alpha * 0.5), 0, blended)

        # Draw subtle grid boundary guides for store layout awareness
        for r in range(1, self.rows):
            gy = int(r * (h / self.rows))
            cv2.line(blended, (0, gy), (w, gy), (80, 80, 80), 1, cv2.LINE_4)
        for c in range(1, self.cols):
            gx = int(c * (w / self.cols))
            cv2.line(blended, (gx, 0), (gx, h), (80, 80, 80), 1, cv2.LINE_4)

        return blended


class BusinessAnalyticsManager:
    """Unified coordinator for the VisionGuard Enterprise Business Suite."""

    def __init__(self):
        self.loss_prevention = LossPreventionEngine()
        self.perimeter_trespass = PerimeterTrespassEngine()
        self.worker_safety = WorkerSafetyEngine()
        self.foot_traffic = FootTrafficAnalytics()

    def process_frame(
        self,
        frame: np.ndarray,
        persons: list[dict],
        detected_objects: list[dict] | None = None,
        tripwire: list | tuple | None = None,
        now: float | None = None
    ) -> list[dict]:
        """Execute all business intelligence analyzers on the incoming frame."""
        if now is None:
            now = time.monotonic()
        h, w = frame.shape[:2]
        events: list[dict] = []

        # 1. Foot traffic density & dwell accumulation
        self.foot_traffic.update((h, w), persons, now=now)

        # 2. Inventory Shrinkage & Shoplifting
        if detected_objects:
            shrinkage_events = self.loss_prevention.process((h, w), detected_objects, persons, now=now)
            events.extend(shrinkage_events)

        # 3. Perimeter Security Trespassing & Tailgating
        if tripwire:
            perimeter_events = self.perimeter_trespass.process((h, w), persons, tripwire=tripwire, now=now)
            events.extend(perimeter_events)

        # 4. Worker Accidents, Slips & Prolonged Immobility
        worker_events = self.worker_safety.process((h, w), persons, now=now)
        events.extend(worker_events)

        return events

    def get_summary(self) -> dict:
        """Provide comprehensive business telemetry and operational health report."""
        return {
            'foot_traffic': self.foot_traffic.get_summary(),
            'active_tracked_merchandise': len(self.loss_prevention.active_items),
            'recent_perimeter_crossings': len(self.perimeter_trespass.recent_crossings),
            'active_worker_monitors': len(self.worker_safety.person_state)
        }


def draw_business_overlays(
    frame: np.ndarray,
    events: list[dict],
    foot_traffic: FootTrafficAnalytics | None = None,
    show_heatmap: bool = False
) -> np.ndarray:
    """Render high-contrast business incident badges and optional thermal occupancy heatmap."""
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return frame

    # 1. Thermal heatmap overlay if enabled
    if show_heatmap and foot_traffic is not None:
        frame = foot_traffic.render_heatmap(frame, alpha=0.32)

    # 2. Render prominent event notification banners
    business_event_types = {
        'inventory_shrinkage': ('[!] INVENTORY SHRINKAGE / CONCEALMENT', (0, 140, 255)),
        'perimeter_trespass': ('[!] PERIMETER TRESPASS BREACH', (0, 0, 255)),
        'tailgating_detected': ('[!] TAILGATING / PIGGYBACKING BREACH', (0, 0, 255)),
        'worker_slip_fall': ('[!] WORKER SLIP / ACCIDENT DETECTED', (0, 69, 255)),
        'unresponsive_worker': ('[!] CRITICAL: UNRESPONSIVE WORKER', (0, 0, 255))
    }

    badge_y = 65
    for evt in events:
        etype = evt.get('type')
        if etype in business_event_types:
            label, color = business_event_types[etype]
            details = str(evt.get('details', label))

            (lw, lh), _ = cv2.getTextSize(details, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (12, badge_y - 18), (24 + lw, badge_y + 6), (14, 20, 30), -1)
            cv2.rectangle(frame, (12, badge_y - 18), (24 + lw, badge_y + 6), color, 2)
            cv2.putText(frame, details, (18, badge_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw highlight box around incident location if bbox provided
            bbox = evt.get('bbox')
            if bbox and len(bbox) == 4:
                bx1, by1, bx2, by2 = bbox
                if max(bx1, bx2) <= 1.0 and max(by1, by2) <= 1.0:
                    px1, py1, px2, py2 = int(bx1 * w), int(by1 * h), int(bx2 * w), int(by2 * h)
                else:
                    px1, py1, px2, py2 = int(bx1), int(by1), int(bx2), int(by2)
                px1 = max(0, min(w - 1, px1))
                py1 = max(0, min(h - 1, py1))
                px2 = max(0, min(w - 1, px2))
                py2 = max(0, min(h - 1, py2))
                if px2 > px1 and py2 > py1:
                    cv2.rectangle(frame, (px1, py1), (px2, py2), color, 3, cv2.LINE_AA)

            badge_y += 32

    return frame
