import os
os.environ['ADMIN_PASSWORD'] = ''
os.environ['SESSION_SECRET'] = 'test-secret-for-testing-only'
os.environ['DASHBOARD_PASSWORD'] = ''

import pytest
from unittest.mock import MagicMock, patch
import camera_utils

def test_get_connected_camera_devices():
    devices = camera_utils.get_connected_camera_devices()
    assert isinstance(devices, list)
    for dev in devices:
        assert 'name' in dev
        assert 'status' in dev

def test_detect_cameras_mocked():
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    import numpy as np
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_cap.read.return_value = (True, fake_frame)
    
    with patch('camera_utils.open_capture', return_value=fake_cap):
        cams = camera_utils.detect_cameras(max_index=2)
        assert len(cams) >= 1
        cam = cams[0]
        assert 'index' in cam
        assert 'source' in cam
        assert 'width' in cam
        assert 'height' in cam
        assert 'is_phone_candidate' in cam
        assert cam['width'] == 640
        assert cam['height'] == 480

def test_is_phone_device():
    assert camera_utils.is_phone_device("Connected Camera") is True
    assert camera_utils.is_phone_device("Pixel 8 Camera") is True
    assert camera_utils.is_phone_device("Galaxy S23") is True
    assert camera_utils.is_phone_device("Integrated Webcam") is False
    assert camera_utils.is_phone_device("USB Video Device") is False
