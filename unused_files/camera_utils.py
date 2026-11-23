"""Camera and connected device detection utility for VisionGuard.

Supports built-in webcams, USB cameras, and mobile connected devices
such as Windows 11 Phone Link ("Connected Camera"), DroidCam, and Wi-Fi streams.
"""
import os
import sys
import logging
from typing import List, Dict, Optional, Any
import cv2

try:
    if hasattr(cv2, 'utils') and hasattr(cv2.utils, 'logging'):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

LOGGER = logging.getLogger(__name__)

PHONE_KEYWORDS = (
    'phone', 'connected camera', 'link to windows', 'mobile',
    'galaxy', 'samsung', 'pixel', 'xiaomi', 'oneplus', 'iphone', 'android',
    'droidcam', 'iriun', 'epoccam', 'camo', 'virtual camera'
)


class silence_stderr:
    """Context manager to suppress low-level C stderr outputs from OpenCV backend."""
    def __enter__(self):
        try:
            self.null_fd = os.open(os.devnull, os.O_WRONLY)
            self.saved_stderr_fd = os.dup(2)
            os.dup2(self.null_fd, 2)
        except Exception:
            self.null_fd = None
            self.saved_stderr_fd = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.saved_stderr_fd is not None:
                os.dup2(self.saved_stderr_fd, 2)
                os.close(self.saved_stderr_fd)
            if self.null_fd is not None:
                os.close(self.null_fd)
        except Exception:
            pass


def is_phone_device(name: Optional[str]) -> bool:
    """Check if a device name suggests a phone link or mobile virtual camera."""
    if not name:
        return False
    lower = name.lower()
    return any(keyword in lower for keyword in PHONE_KEYWORDS)


def get_connected_camera_devices() -> List[Dict[str, Any]]:
    """Query Windows WMI/PnP for physical and virtual camera devices."""
    devices = []
    if os.name != 'nt':
        return devices
    try:
        import win32com.client
        wmi = win32com.client.GetObject('winmgmts:')
        query = "SELECT Name, DeviceID, Status, Description, PNPClass FROM Win32_PnPEntity WHERE PNPClass = 'Camera' OR PNPClass = 'Image'"
        records = wmi.ExecQuery(query)
        for r in records:
            name = getattr(r, 'Name', '') or getattr(r, 'Description', '') or 'Camera Device'
            status = getattr(r, 'Status', 'OK')
            dev_id = getattr(r, 'DeviceID', '')
            devices.append({
                'name': name,
                'status': status,
                'device_id': dev_id,
                'is_phone_candidate': is_phone_device(name)
            })
    except Exception as exc:
        LOGGER.debug("WMI camera query skipped or failed: %s", exc)
    return devices


def open_capture(source: Any) -> cv2.VideoCapture:
    """Open camera source trying DirectShow first on Windows, then falling back to MSMF."""
    value = int(source) if isinstance(source, str) and source.isdecimal() else source
    if isinstance(value, int) and os.name == 'nt' and hasattr(cv2, 'CAP_DSHOW'):
        cap = cv2.VideoCapture(value, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(value)


def detect_cameras(max_index: int = 8) -> List[Dict[str, Any]]:
    """Detect available connected cameras and correlate with Windows friendly names."""
    connected_pnp = get_connected_camera_devices()
    cameras = []
    
    for i in range(max_index):
        with silence_stderr():
            cap = open_capture(i)
        try:
            if not cap.isOpened():
                continue
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                status = 'working'
            else:
                w, h = 0, 0
                status = 'no_frame'
            
            # Associate friendly name from WMI if index is within pnp count
            friendly_name = connected_pnp[i]['name'] if i < len(connected_pnp) else f'Camera {i}'
            is_phone = is_phone_device(friendly_name) or (i > 0 and len(connected_pnp) > 1 and is_phone_device(connected_pnp[min(i, len(connected_pnp)-1)]['name']))
            
            cam_entry = {
                'index': i,
                'source': str(i),
                'name': f"{friendly_name} ({w}x{h})" if w > 0 else friendly_name,
                'friendly_name': friendly_name,
                'width': w,
                'height': h,
                'status': status,
                'is_phone_candidate': is_phone,
                'type': 'phone' if is_phone else ('built_in' if i == 0 and 'integrated' in friendly_name.lower() else 'usb_external')
            }
            cameras.append(cam_entry)
        finally:
            cap.release()
            
    return cameras


def get_default_camera() -> Optional[int]:
    """Get the default working camera index."""
    cameras = detect_cameras()
    working = [c for c in cameras if c['width'] > 0]
    return working[0]['index'] if working else None


def get_phone_cameras() -> List[Dict[str, Any]]:
    """Filter and return only candidate phone/connected cameras."""
    cameras = detect_cameras()
    return [c for c in cameras if c.get('is_phone_candidate') and c['width'] > 0]


def print_camera_menu(cameras: List[Dict]):
    """Print camera selection menu."""
    print("\n" + "=" * 55)
    print("AVAILABLE CAMERAS & CONNECTED DEVICES")
    print("=" * 55)
    if not cameras:
        print("No cameras detected!")
        print("\nOptions:")
        print("  - Check webcam or phone link connection")
        print("  - Ensure Windows camera permissions are granted")
        print("  - For Phone Link: Windows Settings > Bluetooth & devices > Mobile devices > Manage devices")
        print("  - Try RTSP or Wi-Fi IP Webcam stream URL instead")
        return
    
    for i, cam in enumerate(cameras):
        badge = " [PHONE LINK/MOBILE]" if cam.get('is_phone_candidate') else ""
        status = "[OK] WORKING" if cam['width'] > 0 else "[!] OPENS BUT NO FRAME"
        default_mark = " (DEFAULT)" if cam['index'] == 0 else ""
        print(f"  [{i}] {cam['name']}{badge}{default_mark} - {status}")
    
    print("\n  [R] RTSP / Wi-Fi IP Stream URL (e.g. http://phone-ip:8080/video)")
    print("  [F] Local Video File")
    print("  [Q] Quit")


def select_camera() -> Optional[str]:
    """Interactive camera selection."""
    cameras = detect_cameras()
    if not cameras:
        print("No cameras detected!")
        choice = input("Enter RTSP URL, Phone IP stream, or video file path: ").strip()
        return choice if choice else None
    
    print_camera_menu(cameras)
    while True:
        choice = input("\nSelect camera [0-9, R, F, Q]: ").strip().upper()
        if choice == 'Q':
            return None
        elif choice == 'R':
            url = input("Enter RTSP/HTTP URL (e.g., http://192.168.1.50:8080/video): ").strip()
            if url:
                return url
        elif choice == 'F':
            path = input("Enter video file path: ").strip()
            if path:
                return path
        elif choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(cameras):
                return str(cameras[idx]['index'])
        print("Invalid choice. Please try again.")


def test_camera(source: str) -> bool:
    """Test if a camera source works."""
    try:
        cap = open_capture(source)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    except Exception:
        return False


if __name__ == '__main__':
    print("VisionGuard Camera & Connected Device Detection")
    print("-" * 50)
    cams = detect_cameras()
    print_camera_menu(cams)
    
    phones = get_phone_cameras()
    if phones:
        print(f"\n✓ Detected {len(phones)} candidate phone camera(s):")
        for p in phones:
            print(f"  -> Index {p['index']}: {p['friendly_name']} ({p['width']}x{p['height']})")
