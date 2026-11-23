"""Intelligent Traffic & Road Safety Enforcement Suite.

Features:
- Motorcycle Overcapacity / Triple-Riding Detection
- Red Light / Traffic Signal Jumping Violation Detection
- Driver / Passenger Seatbelt Compliance Detection
- License Plate Localization and ANPR OCR
"""
import math
import re
import time
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np


class TripleRidingDetector:
    """Detects 3 or more riders mounted on a single motorcycle/two-wheeler."""

    def __init__(self, horizontal_margin: float = 0.15):
        self.horizontal_margin = float(horizontal_margin)

    def detect_violations(self, motorcycles: List[dict], persons: List[dict]) -> List[dict]:
        violations = []
        if not motorcycles or not persons:
            return violations

        for moto in motorcycles:
            if not isinstance(moto, dict) or 'bbox' not in moto:
                continue
            m_box = moto['bbox']
            if len(m_box) != 4:
                continue

            mx1, my1, mx2, my2 = [float(v) for v in m_box]
            mw = max(1.0, mx2 - mx1)
            mh = max(1.0, my2 - my1)

            # Define motorcycle riding envelope
            left_bound = mx1 - self.horizontal_margin * mw
            right_bound = mx2 + self.horizontal_margin * mw

            associated_riders = []
            for p in persons:
                if not isinstance(p, dict) or 'bbox' not in p:
                    continue
                p_box = p['bbox']
                if len(p_box) != 4:
                    continue

                px1, py1, px2, py2 = [float(v) for v in p_box]
                pcx = (px1 + px2) / 2.0

                # Check horizontal containment
                if left_bound <= pcx <= right_bound:
                    # Check vertical overlap (rider sits above or inside motorcycle envelope)
                    if py2 >= my1 - 0.2 * mh and py1 <= my2:
                        associated_riders.append(p)

            # Violation if 3 or more riders mounted
            if len(associated_riders) >= 3:
                m_id = moto.get('id')
                violations.append({
                    'type': 'triple_riding',
                    'severity': 'high',
                    'rider_count': len(associated_riders),
                    'motorcycle_id': m_id,
                    'motorcycle_bbox': [mx1, my1, mx2, my2],
                    'rider_bboxes': [r['bbox'] for r in associated_riders],
                    'confidence': 0.90,
                    'message': f"Triple Riding: {len(associated_riders)} persons on motorcycle" + (f" #{m_id}" if m_id is not None else ""),
                })

        return violations


def classify_traffic_light_color(crop: np.ndarray) -> str:
    """Classify traffic light crop into 'red', 'yellow', 'green', or 'unknown'."""
    if crop is None or crop.size == 0:
        return 'unknown'
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return 'unknown'

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Red masks (wraps around 0 and 180 in HSV)
    mask_r1 = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([12, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([168, 60, 60]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)

    # Yellow mask
    mask_yellow = cv2.inRange(hsv, np.array([15, 60, 60]), np.array([34, 255, 255]))

    # Green mask
    mask_green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([90, 255, 255]))

    # Position weights: top 45% favors red, bottom 45% favors green
    y_indices = np.arange(h).reshape(-1, 1)
    red_pos_weight = np.where(y_indices < h * 0.45, 1.4, 0.7)
    green_pos_weight = np.where(y_indices > h * 0.55, 1.4, 0.7)

    red_score = np.sum((mask_red > 0) * red_pos_weight)
    yellow_score = float(np.sum(mask_yellow > 0))
    green_score = np.sum((mask_green > 0) * green_pos_weight)

    scores = {'red': red_score, 'yellow': yellow_score, 'green': green_score}
    best_color, best_score = max(scores.items(), key=lambda item: item[1])

    if best_score >= 4:
        return best_color
    return 'unknown'


class SignalViolationDetector:
    """Detects vehicles breaching the stop line during a RED traffic light state."""

    def __init__(self):
        self._reported_violations: Dict[str, float] = {}

    def check_violations(
        self,
        vehicles: List[dict],
        signal_state: str,
        stop_line: Optional[List[list]],
        frame_shape: Tuple[int, int],
        now: Optional[float] = None
    ) -> List[dict]:
        violations = []
        if str(signal_state).strip().lower() != 'red':
            return violations
        if not stop_line or len(stop_line) < 2 or not vehicles:
            return violations

        h, w = frame_shape[:2]
        if h <= 0 or w <= 0:
            return violations

        p1, p2 = stop_line[0], stop_line[1]
        line_y = ((p1[1] + p2[1]) / 2.0) * h

        if now is None:
            now = time.monotonic()

        for v in vehicles:
            if not isinstance(v, dict) or 'bbox' not in v:
                continue
            box = v['bbox']
            if len(box) != 4:
                continue

            vx1, vy1, vx2, vy2 = [float(val) for val in box]
            vid = v.get('id')

            # Check if vehicle has breached/crossed the stop line
            if vy2 >= line_y and vy1 <= line_y + (vy2 - vy1) * 0.8:
                v_key = f"red_light:{vid if vid is not None else f'{int(vx1)}_{int(vy1)}'}"
                last_time = self._reported_violations.get(v_key, float('-inf'))
                if now - last_time < 5.0:  # 5-second per-vehicle cooldown
                    continue

                self._reported_violations[v_key] = now
                violations.append({
                    'type': 'red_light_violation',
                    'severity': 'critical',
                    'vehicle_id': vid,
                    'vehicle_bbox': [vx1, vy1, vx2, vy2],
                    'signal_state': 'red',
                    'confidence': 0.95,
                    'message': f"Red Light Violation: Vehicle" + (f" #{vid}" if vid is not None else "") + " breached stop line on RED",
                })

        return violations


class SeatbeltDetector:
    """Detects driver/passenger seatbelt compliance using diagonal torso strap analysis."""

    def __init__(self, min_angle: float = 25.0, max_angle: float = 75.0):
        self.min_angle = float(min_angle)
        self.max_angle = float(max_angle)

    def check_seatbelt(self, driver_crop: np.ndarray) -> dict:
        """Analyze upper-torso crop for diagonal seatbelt strap."""
        if driver_crop is None or driver_crop.size == 0:
            return {'wearing_seatbelt': False, 'confidence': 0.0}

        h, w = driver_crop.shape[:2]
        if h < 20 or w < 20:
            return {'wearing_seatbelt': False, 'confidence': 0.0}

        gray = cv2.cvtColor(driver_crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 35, 110)

        min_len = int(min(h, w) * 0.35)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=18, minLineLength=max(15, min_len), maxLineGap=12)

        has_seatbelt = False
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                angle = math.degrees(math.atan2(dy, max(1, dx)))
                if self.min_angle <= angle <= self.max_angle:
                    length = math.hypot(dx, dy)
                    if length >= min_len:
                        has_seatbelt = True
                        break

        return {
            'wearing_seatbelt': has_seatbelt,
            'confidence': 0.88 if has_seatbelt else 0.82
        }


def clean_plate_text(text: str) -> str:
    """Normalize and clean license plate string."""
    if not text:
        return ""
    cleaned = re.sub(r'[^A-Za-z0-9]', '', str(text)).upper().strip()
    return cleaned


class LicensePlateOCR:
    """Localizes license plates and extracts alphanumeric text."""

    def __init__(self, use_easyocr: bool = True):
        self.use_easyocr = use_easyocr
        self._easyocr_reader = None
        if self.use_easyocr:
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            except Exception:
                self._easyocr_reader = None

        # Build built-in glyph template library for pure OpenCV optical recognition
        self._glyph_templates: Dict[str, np.ndarray] = {}
        chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for ch in chars:
            img = np.zeros((32, 24), dtype=np.uint8)
            cv2.putText(img, ch, (3, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
            self._glyph_templates[ch] = img

    def _ocr_template_match(self, plate_crop: np.ndarray) -> str:
        """Lightweight template-matching OCR fallback for license plate characters."""
        if plate_crop is None or plate_crop.size == 0:
            return ""
        ph, pw = plate_crop.shape[:2]
        if ph < 15 or pw < 30:
            return ""

        # Crop slightly inward to remove plate outer border
        margin_y = max(1, int(ph * 0.06))
        margin_x = max(1, int(pw * 0.04))
        inner_crop = plate_crop[margin_y:ph-margin_y, margin_x:pw-margin_x]
        if inner_crop.size == 0:
            inner_crop = plate_crop

        iph, ipw = inner_crop.shape[:2]
        gray = cv2.cvtColor(inner_crop, cv2.COLOR_BGR2GRAY) if len(inner_crop.shape) == 3 else inner_crop
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Invert if majority of image is white
        if np.mean(thresh) > 128:
            thresh = cv2.bitwise_not(thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_boxes = []
        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            # Skip the full container or tiny noise
            if cw >= 0.85 * ipw or ch >= 0.95 * iph:
                continue
            # Filter character dimensions
            if (0.20 * iph <= ch <= 0.95 * iph) and (3 <= cw <= 0.40 * ipw) and (cw * ch >= 15):
                char_boxes.append((cx, cy, cw, ch))

        # Sort characters from left to right
        char_boxes.sort(key=lambda b: b[0])
        extracted = []
        for cx, cy, cw, ch in char_boxes:
            char_roi = thresh[cy:cy+ch, cx:cx+cw]
            char_resized = cv2.resize(char_roi, (24, 32), interpolation=cv2.INTER_AREA)

            best_char, best_val = "X", -999.0
            for g_char, g_tmpl in self._glyph_templates.items():
                res = cv2.matchTemplate(char_resized, g_tmpl, cv2.TM_CCOEFF_NORMED)
                val = float(res[0][0])
                if val > best_val:
                    best_val = val
                    best_char = g_char

            extracted.append(best_char)

        return "".join(extracted)

    def extract_plates(self, frame: np.ndarray, vehicle_box: Optional[list] = None) -> List[dict]:
        """Detect license plate candidates and extract text."""
        candidates = []
        if frame is None or frame.size == 0:
            return candidates

        fh, fw = frame.shape[:2]
        if vehicle_box and len(vehicle_box) == 4:
            vx1, vy1, vx2, vy2 = [max(0, int(round(v))) for v in vehicle_box]
            vx1, vx2 = min(fw - 1, vx1), min(fw, vx2)
            vy1, vy2 = min(fh - 1, vy1), min(fh, vy2)
            if vx2 <= vx1 or vy2 <= vy1:
                search_region = frame
                offset_x, offset_y = 0, 0
            else:
                # Lower 60% of vehicle is where plates reside
                search_y1 = vy1 + int((vy2 - vy1) * 0.40)
                search_region = frame[search_y1:vy2, vx1:vx2]
                offset_x, offset_y = vx1, search_y1
        else:
            search_region = frame
            offset_x, offset_y = 0, 0

        sh, sw = search_region.shape[:2]
        if sh < 20 or sw < 40:
            return candidates

        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blurred, 50, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        total_area = sh * sw

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(max(1, h))
            area = w * h

            # License plates have aspect ratios between 1.8 and 5.5
            if (1.8 <= aspect_ratio <= 5.5) and (250 <= area <= 0.45 * total_area):
                crop = search_region[y:y+h, x:x+w]
                plate_text = ""

                # Try EasyOCR if available
                if self._easyocr_reader:
                    try:
                        ocr_res = self._easyocr_reader.readtext(crop)
                        if ocr_res:
                            plate_text = "".join([r[1] for r in ocr_res])
                    except Exception:
                        pass

                # Fallback to OpenCV template matching OCR
                if not plate_text:
                    plate_text = self._ocr_template_match(crop)

                clean = clean_plate_text(plate_text)
                if len(clean) >= 4:
                    abs_x1 = offset_x + x
                    abs_y1 = offset_y + y
                    abs_x2 = abs_x1 + w
                    abs_y2 = abs_y1 + h
                    candidates.append({
                        'bbox': [abs_x1, abs_y1, abs_x2, abs_y2],
                        'plate_text': clean,
                        'confidence': 0.88,
                    })

        return candidates


class TrafficEnforcementManager:
    """Coordinates ANPR plate OCR, signal violations, triple-riding, and seatbelt detection."""

    def __init__(self):
        self.triple_riding = TripleRidingDetector()
        self.signal_detector = SignalViolationDetector()
        self.seatbelt_detector = SeatbeltDetector()
        self.plate_ocr = LicensePlateOCR()

    def process_traffic(
        self,
        frame: np.ndarray,
        detected_objects: List[dict],
        persons: List[dict],
        stop_line: Optional[List[list]] = None,
        now: Optional[float] = None
    ) -> Tuple[List[dict], List[dict]]:
        """Run all traffic compliance checks and plate OCR."""
        violations = []
        plates = []
        if frame is None or frame.size == 0:
            return violations, plates

        h, w = frame.shape[:2]

        # Categorize detected objects
        motorcycles = []
        vehicles = []
        traffic_lights = []

        for obj in (detected_objects or []):
            if not isinstance(obj, dict):
                continue
            cid = obj.get('class_id')
            cname = str(obj.get('class_name', '')).strip().lower()

            if cid == 3 or cname in ('motorcycle', 'motorbike', 'bike'):
                motorcycles.append(obj)
                vehicles.append(obj)
            elif cid in (2, 5, 7) or cname in ('car', 'bus', 'truck', 'van', 'vehicle'):
                vehicles.append(obj)
            elif cid == 9 or 'traffic light' in cname:
                traffic_lights.append(obj)

        # 1. Check Motorcycle Overcapacity / Triple-Riding
        if motorcycles and persons:
            triple_events = self.triple_riding.detect_violations(motorcycles, persons)
            violations.extend(triple_events)

        # 2. Check Traffic Signal State and Stop-Line Jumping
        signal_state = 'unknown'
        for tl in traffic_lights:
            tl_box = tl.get('bbox')
            if tl_box and len(tl_box) == 4:
                x1, y1, x2, y2 = [max(0, int(round(v))) for v in tl_box]
                crop = frame[y1:y2, x1:x2]
                state = classify_traffic_light_color(crop)
                if state in ('red', 'green', 'yellow'):
                    signal_state = state
                    break

        if stop_line and vehicles:
            signal_events = self.signal_detector.check_violations(
                vehicles=vehicles,
                signal_state=signal_state,
                stop_line=stop_line,
                frame_shape=(h, w),
                now=now
            )
            violations.extend(signal_events)

        # 3. Check Driver Seatbelt Compliance for passenger cars
        for veh in vehicles:
            v_box = veh.get('bbox')
            if not v_box or len(v_box) != 4:
                continue
            # Look for persons overlapping upper half of vehicle
            vx1, vy1, vx2, vy2 = [float(v) for v in v_box]
            vw = vx2 - vx1
            vh = vy2 - vy1
            if vh < 60 or vw < 60:
                continue

            for p in persons:
                p_box = p.get('bbox')
                if not p_box or len(p_box) != 4:
                    continue
                px1, py1, px2, py2 = [float(v) for v in p_box]
                # Check if person is inside driver windshield region
                if (vx1 <= (px1 + px2) / 2.0 <= vx2) and (vy1 <= py1 <= vy1 + vh * 0.7):
                    driver_crop = frame[max(0, int(round(py1))):min(h, int(round(py2))),
                                        max(0, int(round(px1))):min(w, int(round(px2)))]
                    if driver_crop.size > 0:
                        sb_res = self.seatbelt_detector.check_seatbelt(driver_crop)
                        if not sb_res['wearing_seatbelt']:
                            violations.append({
                                'type': 'seatbelt_violation',
                                'severity': 'medium',
                                'vehicle_id': veh.get('id'),
                                'driver_id': p.get('id'),
                                'driver_bbox': [px1, py1, px2, py2],
                                'confidence': sb_res['confidence'],
                                'message': f"No Seatbelt Detected: Driver in vehicle" + (f" #{veh.get('id')}" if veh.get('id') is not None else "")
                            })

        # 4. Extract License Plates (ANPR) on candidate vehicles
        for veh in vehicles[:4]:  # Focus on prominent vehicles
            v_box = veh.get('bbox')
            found_plates = self.plate_ocr.extract_plates(frame, vehicle_box=v_box)
            plates.extend(found_plates)

        return violations, plates


def draw_traffic_overlays(
    frame: np.ndarray,
    violations: List[dict],
    plates: List[dict]
) -> np.ndarray:
    """Render high-visibility traffic violation boxes and ANPR plate badges."""
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return frame

    # 1. Render Traffic Violations
    for viol in (violations or []):
        v_type = viol.get('type', '')
        box = viol.get('motorcycle_bbox') or viol.get('vehicle_bbox') or viol.get('driver_bbox')
        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        if v_type == 'triple_riding':
            color = (0, 69, 255)  # Neon Orange/Red
            riders = viol.get('rider_count', 3)
            label = f"[!] VIOLATION: TRIPLE RIDING ({riders} RIDERS)"
        elif v_type == 'red_light_violation':
            color = (0, 0, 255)  # High-Alert Red
            label = "[!] VIOLATION: RED LIGHT JUMP"
        elif v_type == 'seatbelt_violation':
            color = (255, 0, 255)  # Neon Magenta
            label = "[!] VIOLATION: NO SEATBELT"
        else:
            color = (0, 140, 255)
            label = f"[!] VIOLATION: {v_type.upper()}"

        # Draw thick alert bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

        # Corner accents
        clen = max(8, min(24, (x2 - x1) // 4, (y2 - y1) // 4))
        cv2.line(frame, (x1, y1), (x1 + clen, y1), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + clen), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2 - clen, y1), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + clen), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1 + clen, y2), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - clen), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - clen, y2), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - clen), (255, 255, 255), 2, cv2.LINE_AA)

        # Label badge
        font_scale = 0.48
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        pad = 4
        tag_y1 = max(0, y1 - th - 2 * pad)
        tag_y2 = tag_y1 + th + 2 * pad
        tag_x1 = x1
        tag_x2 = min(w - 1, x1 + tw + 2 * pad)

        cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), (10, 16, 26), -1)
        cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), color, 1)
        cv2.putText(frame, label, (tag_x1 + pad, tag_y2 - pad - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Render ANPR License Plates
    for plate in (plates or []):
        p_box = plate.get('bbox')
        p_text = plate.get('plate_text', '')
        conf = float(plate.get('confidence', 0.88))
        if not p_box or len(p_box) != 4 or not p_text:
            continue

        px1, py1, px2, py2 = [int(round(v)) for v in p_box]
        px1, px2 = max(0, min(w - 1, px1)), max(0, min(w - 1, px2))
        py1, py2 = max(0, min(h - 1, py1)), max(0, min(h - 1, py2))
        if px2 <= px1 or py2 <= py1:
            continue

        plate_color = (255, 220, 0)  # Bright Cyan / Yellow
        cv2.rectangle(frame, (px1, py1), (px2, py2), plate_color, 2, cv2.LINE_AA)

        p_label = f"[ANPR] {p_text} | {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(p_label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        pad = 3
        badge_y1 = min(h - th - 2 * pad, py2 + 2)
        badge_y2 = badge_y1 + th + 2 * pad
        badge_x1 = px1
        badge_x2 = min(w - 1, px1 + tw + 2 * pad)

        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (12, 18, 28), -1)
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), plate_color, 1)
        cv2.putText(frame, p_label, (badge_x1 + pad, badge_y2 - pad - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    return frame
