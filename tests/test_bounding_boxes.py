import numpy as np
import pytest
from app.detectors import draw_security_bounding_boxes


def test_draw_weapon_bounding_box():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    events = [
        {
            'type': 'weapon',
            'subtype': 'knife',
            'confidence': 0.88,
            'bbox': [0.1, 0.2, 0.3, 0.5]  # normalized
        }
    ]

    annotated = draw_security_bounding_boxes(frame.copy(), events)

    # Frame should have non-zero pixels inside the bounding box and label area
    # Bbox in pixels: x: 60..180, y: 80..200
    sub_region = annotated[80:200, 60:180]
    assert np.any(sub_region > 0), "Weapon bounding box was not drawn"


def test_draw_irregular_activity_bounding_boxes():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    events = [
        {
            'type': 'violence',
            'subtype': 'fighting',
            'confidence': 0.82,
            'bbox': [0.4, 0.2, 0.8, 0.7]
        },
        {
            'type': 'fall_detected',
            'track_id': 2,
            'confidence': None,
            'bbox': [0.1, 0.6, 0.4, 0.9]
        },
        {
            'type': 'loitering',
            'track_id': 3,
            'confidence': 0.7,
            'bbox': [0.7, 0.6, 0.9, 0.9]
        }
    ]

    annotated = draw_security_bounding_boxes(frame.copy(), events)

    # Verify pixels are modified in all 3 regions
    region_violence = annotated[80:280, 240:480]
    region_fall = annotated[240:360, 60:240]
    region_loitering = annotated[240:360, 420:540]

    assert np.any(region_violence > 0), "Violence bounding box not drawn"
    assert np.any(region_fall > 0), "Fall bounding box not drawn"
    assert np.any(region_loitering > 0), "Loitering bounding box not drawn"


def test_draw_intrusion_with_person_fallback():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    events = [
        {
            'type': 'intrusion',
            'track_id': 5,
            'confidence': None
            # No direct bbox in event
        }
    ]
    persons = [
        {
            'id': 5,
            'bbox': [100, 100, 200, 300]  # pixel coordinates
        }
    ]

    annotated = draw_security_bounding_boxes(frame.copy(), events, persons=persons)
    sub_region = annotated[100:300, 100:200]
    assert np.any(sub_region > 0), "Intrusion bounding box from person track was not drawn"
