"""Camera detection and selection utility for VisionGuard."""
import cv2
import sys
from typing import List, Dict, Optional

def detect_cameras(max_index: int = 5) -> List[Dict]:
    """Detect available cameras and return their info."""
    cameras = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to verify it works
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                cameras.append({
                    'index': i,
                    'width': w,
                    'height': h,
                    'name': f'Camera {i} ({w}x{h})'
                })
            else:
                cameras.append({
                    'index': i,
                    'width': 0,
                    'height': 0,
                    'name': f'Camera {i} (opens but no frame)'
                })
            cap.release()
        else:
            # Not available
            pass
    return cameras


def get_default_camera() -> Optional[int]:
    """Get the default working camera index (prefers first working camera)."""
    cameras = detect_cameras()
    working = [c for c in cameras if c['width'] > 0]
    if not working:
        return None
    # Return first working camera
    return working[0]['index']


def print_camera_menu(cameras: List[Dict]):
    """Print camera selection menu."""
    print("\n" + "=" * 50)
    print("AVAILABLE CAMERAS")
    print("=" * 50)
    if not cameras:
        print("No cameras detected!")
        print("\nOptions:")
        print("  - Check webcam connection")
        print("  - Ensure camera permissions are granted")
        print("  - Try RTSP stream URL instead")
        return
    
    for i, cam in enumerate(cameras):
        status = "[OK] WORKING" if cam['width'] > 0 else "[!] OPENS BUT NO FRAME"
        default_mark = " (DEFAULT)" if cam['index'] == 0 else ""
        print(f"  [{i}] {cam['name']}{default_mark} - {status}")
    
    print("\n  [R] RTSP Stream URL")
    print("  [F] Local Video File")
    print("  [Q] Quit")


def select_camera() -> Optional[str]:
    """Interactive camera selection. Returns camera source string."""
    cameras = detect_cameras()
    
    if not cameras:
        print("No cameras detected!")
        print("\nTrying RTSP or file input...")
        choice = input("Enter RTSP URL or video file path (or press Enter to quit): ").strip()
        return choice if choice else None
    
    # If only one working camera, auto-select it
    working_cameras = [c for c in cameras if c['width'] > 0]
    if len(working_cameras) == 1:
        cam = working_cameras[0]
        print(f"\nAuto-selecting: Camera {cam['index']} ({cam['width']}x{cam['height']})")
        return str(cam['index'])
    
    # Multiple cameras - show menu
    print_camera_menu(cameras)
    
    while True:
        choice = input("\nSelect camera [0-9, R, F, Q]: ").strip().upper()
        
        if choice == 'Q':
            return None
        elif choice == 'R':
            url = input("Enter RTSP URL (e.g., rtsp://user:pass@ip:554/stream): ").strip()
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
        value = int(source) if source.isdecimal() else source
        cap = cv2.VideoCapture(value)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    except Exception:
        return False


if __name__ == '__main__':
    print("VisionGuard Camera Detection")
    print("-" * 30)
    
    cameras = detect_cameras()
    print_camera_menu(cameras)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Auto mode: return default camera
        default = get_default_camera()
        if default is not None:
            print(f"\nAuto-selected: Camera {default}")
            sys.exit(0)
        else:
            print("\nNo camera available for auto-selection")
            sys.exit(1)
    
    source = select_camera()
    if source:
        print(f"\nSelected source: {source}")
        if test_camera(source):
            print("✓ Camera test PASSED")
        else:
            print("✗ Camera test FAILED - source may not work")
    else:
        print("\nNo camera selected.")