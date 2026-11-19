"""Camera-local intrusion, occupancy and bidirectional tripwire analytics.

Coordinates are normalized to (0..1), and each person's bottom-center
is the approximate ground-contact point.
"""
from dataclasses import dataclass
from math import hypot
from time import monotonic


def point_in_polygon(point, polygon):
    try:
        x, y = point
    except (ValueError, TypeError):
        return False
    inside = False
    for i in range(len(polygon)):
        try:
            x1, y1 = polygon[i - 1]
            x2, y2 = polygon[i]
        except (ValueError, TypeError, IndexError):
            continue
        if (y1 > y) != (y2 > y):
            cross_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < cross_x:
                inside = not inside
    return inside


def crossing_direction(previous, current, line):
    """Return entry/exit if a person's motion crosses a *finite* directed line."""
    if len(line) != 2 or previous == current:
        return None
    try:
        (ax, ay), (bx, by) = line
    except (ValueError, TypeError):
        return None
    dx, dy = bx - ax, by - ay
    if dx * dx + dy * dy < 1e-8 or hypot(current[0]-previous[0], current[1]-previous[1]) > 0.35:
        return None
    def side(p): return dx * (p[1] - ay) - dy * (p[0] - ax)
    prev_side, cur_side = side(previous), side(current)
    if prev_side * cur_side >= -1e-10:
        return None
    # Linear motion interpolation: find where the track path crosses the line.
    fraction = prev_side / (prev_side - cur_side)
    ix = previous[0] + fraction * (current[0] - previous[0])
    iy = previous[1] + fraction * (current[1] - previous[1])
    t = ((ix - ax) * dx + (iy - ay) * dy) / (dx * dx + dy * dy)
    if not 0 <= t <= 1:
        return None
    return 'entry' if prev_side < 0 < cur_side else 'exit'


@dataclass
class TrackState:
    point: tuple[float, float] | None = None
    active_inside: bool = False
    inside_frames: int = 0
    outside_frames: int = 0
    last_seen: int = 0
    last_intrusion: float = -1e12
    last_crossing: float = -1e12


class Analytics:
    def __init__(self, zone=None, tripwire=None, cooldown=20):
        # Validate and sanitize zone: keep only valid [x, y] pairs, need at least 3 for polygon
        if isinstance(zone, list):
            self.zone = [p for p in zone if isinstance(p, (list, tuple)) and len(p) == 2]
            if len(self.zone) < 3:
                self.zone = []
        else:
            self.zone = []
        
        # Validate and sanitize tripwire: must be exactly 2 points, each [x, y]
        if isinstance(tripwire, list) and len(tripwire) == 2:
            p1, p2 = tripwire[0], tripwire[1]
            if (isinstance(p1, (list, tuple)) and len(p1) == 2 and
                isinstance(p2, (list, tuple)) and len(p2) == 2):
                self.tripwire = [p1, p2]
            else:
                self.tripwire = []
        else:
            self.tripwire = []
        
        self.cooldown = cooldown
        self.tracks: dict[int, TrackState] = {}
        self.frame_number = 0
        self.current_people = 0

    def process(self, persons: list[dict], now=None):
        """Persons: [{id: int, foot: (normalized x,y)}]. Returns event dicts."""
        now = monotonic() if now is None else now
        self.frame_number += 1
        events = []
        self.current_people = len(persons)
        for p in persons:
            track_id = int(p['id'])
            point = tuple(p['foot'])
            state = self.tracks.setdefault(track_id, TrackState())
            state.last_seen = self.frame_number
            if len(self.zone) >= 3:
                in_zone = point_in_polygon(point, self.zone)
                if in_zone:
                    state.inside_frames += 1
                    state.outside_frames = 0
                    if not state.active_inside and state.inside_frames >= 2:
                        state.active_inside = True
                        if now - state.last_intrusion >= self.cooldown:
                            events.append({'kind': 'intrusion', 'track_id': track_id,
                                           'message': f'Person #{track_id} entered restricted area'})
                            state.last_intrusion = now
                else:
                    state.outside_frames += 1
                    state.inside_frames = 0
                    if state.outside_frames >= 2:
                        state.active_inside = False
            if state.point is not None and len(self.tripwire) == 2:
                direction = crossing_direction(state.point, point, self.tripwire)
                if direction and now - state.last_crossing >= max(2.0, self.cooldown / 2):
                    events.append({'kind': direction, 'track_id': track_id,
                                   'message': f'Person #{track_id} crossed counting line ({direction})'})
                    state.last_crossing = now
            state.point = point
        # Expire IDs that disappeared, including when tracker reuses numbers later.
        for track_id, state in list(self.tracks.items()):
            if self.frame_number - state.last_seen > 90:
                del self.tracks[track_id]
        return events
