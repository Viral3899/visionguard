"""Motion Trajectory Trails and Dwell Time Tracker."""
import collections
import time
from typing import Optional, List, Tuple, Dict
import cv2
import numpy as np


class TrajectoryTracker:
    """Maintains historical motion trajectories and dwell times for tracked entities."""

    def __init__(self, max_history: int = 25, max_idle_seconds: float = 5.0):
        self.max_history = max(2, int(max_history))
        self.max_idle_seconds = float(max_idle_seconds)
        self._histories: Dict[int, collections.deque] = {}
        self._first_seen: Dict[int, float] = {}
        self._last_seen: Dict[int, float] = {}

    def update(self, tracks: List[dict], now: Optional[float] = None) -> None:
        """Update trajectories with new detections/tracks."""
        if now is None:
            now = time.monotonic()

        seen_ids = set()
        for track in tracks:
            if not isinstance(track, dict):
                continue
            tid = track.get('id')
            if tid is None:
                continue
            try:
                tid = int(tid)
            except (ValueError, TypeError):
                continue

            bbox = track.get('bbox')
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox]
            foot_x = int(round((x1 + x2) / 2.0))
            foot_y = int(round(y2))

            if tid not in self._histories:
                self._histories[tid] = collections.deque(maxlen=self.max_history)
                self._first_seen[tid] = now

            self._histories[tid].append((foot_x, foot_y))
            self._last_seen[tid] = now
            seen_ids.add(tid)

        # Prune stale tracks exceeding max_idle_seconds
        stale_ids = [
            tid for tid, last_t in self._last_seen.items()
            if (now - last_t) > self.max_idle_seconds
        ]
        for tid in stale_ids:
            self._histories.pop(tid, None)
            self._first_seen.pop(tid, None)
            self._last_seen.pop(tid, None)

    def get_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        """Return list of (x, y) coordinates for a given track."""
        if track_id in self._histories:
            return list(self._histories[track_id])
        return []

    def get_dwell_time(self, track_id: int, now: Optional[float] = None) -> float:
        """Return duration in seconds that a track has been active."""
        if now is None:
            now = time.monotonic()
        if track_id in self._first_seen:
            return max(0.0, now - self._first_seen[track_id])
        return 0.0

    def draw(
        self,
        frame: np.ndarray,
        color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
        default_color: Tuple[int, int, int] = (20, 220, 167)
    ) -> np.ndarray:
        """Render fading motion trajectory tails on the video frame."""
        if frame is None or not self._histories:
            return frame

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return frame

        overlay = frame.copy()
        for tid, points in self._histories.items():
            if len(points) < 2:
                continue

            color = color_map.get(tid, default_color) if color_map else default_color
            num_pts = len(points)

            for i in range(1, num_pts):
                p1 = points[i - 1]
                p2 = points[i]

                # Clamp within frame bounds
                if not (0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h):
                    continue

                # Progressively thicker and more opaque towards newest point
                progress = i / float(num_pts)
                thick = 1 if progress < 0.6 else 2
                cv2.line(overlay, p1, p2, color, thick, cv2.LINE_AA)

            # Draw leading position dot
            head = points[-1]
            if 0 <= head[0] < w and 0 <= head[1] < h:
                cv2.circle(overlay, head, 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, head, 5, color, 1, cv2.LINE_AA)

        # Blend smooth antialiased overlay
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        return frame
