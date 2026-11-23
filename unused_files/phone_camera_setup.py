#!/usr/bin/env python3
"""VisionGuard Phone Camera & Connected Device Setup Tool.

Easily connects Microsoft Phone Link ('Connected Camera'), USB webcams,
or Wi-Fi IP camera streams (e.g. IP Webcam app) into VisionGuard AI.
"""
import argparse
import sys
import time
from pathlib import Path

from app.config import settings, ROOT
from app.database import Database
from app.schemas import CameraCreate
import camera_utils


def print_banner():
    print("=" * 60)
    print(" VisionGuard AI - Phone Camera & Connected Device Setup")
    print("=" * 60)


def list_devices():
    """List all detected camera devices."""
    print("\nScanning system for connected cameras and devices...")
    cams = camera_utils.detect_cameras(max_index=8)
    if not cams:
        print("\n[!] No active cameras detected.")
        print("    If using Phone Link:")
        print("    1. Open Windows 11 Settings -> Bluetooth & devices -> Mobile devices")
        print("    2. Turn ON 'Allow this PC to access your mobile devices'")
        print("    3. Click 'Manage devices' and toggle ON 'Use as a connected camera'")
        print("    4. Accept the video stream prompt on your phone.")
        return []

    print(f"\nFound {len(cams)} camera device(s):")
    for cam in cams:
        badge = " [PHONE LINK / MOBILE]" if cam['is_phone_candidate'] else (" [BUILT-IN]" if cam['type'] == 'built_in' else " [EXTERNAL/USB]")
        status = "WORKING" if cam['width'] > 0 else "NO FRAMES"
        print(f"  [{cam['index']}] {cam['friendly_name']} ({cam['width']}x{cam['height']}){badge} - {status}")
    return cams


def test_camera_feed(source: str, duration: int = 4):
    """Test capture and measure FPS."""
    print(f"\nTesting camera source '{source}' for {duration} seconds...")
    cap = camera_utils.open_capture(source)
    if not cap.isOpened():
        print("✗ Failed to open camera feed.")
        return False

    start = time.time()
    frames = 0
    w, h = 0, 0
    while time.time() - start < duration:
        ret, frame = cap.read()
        if ret and frame is not None:
            frames += 1
            h, w = frame.shape[:2]
        time.sleep(0.01)
    cap.release()

    elapsed = time.time() - start
    fps = frames / elapsed if elapsed > 0 else 0
    if frames > 0:
        print(f"✓ Stream OK: {frames} frames captured ({w}x{h} @ {fps:.1f} FPS)")
        return True
    else:
        print("✗ Camera opened but zero valid frames received.")
        return False


def register_camera(name: str, source: str, db: Database) -> bool:
    """Save camera to VisionGuard database."""
    try:
        # Check if source already exists
        existing = [c for c in db.list_cameras(safe=False) if c.get('source') == source]
        if existing:
            print(f"[!] Source '{source}' is already configured as '{existing[0]['name']}' (ID: {existing[0]['id']})")
            return True

        cam = CameraCreate(
            name=name,
            source=source,
            enabled=True,
            features={
                'fire_enabled': True,
                'violence_enabled': True,
                'anomaly_enabled': True,
                'object_enabled': True,
                'telegram_enabled': True,
                'tts_enabled': True,
                'sensitivity': 0.75
            }
        )
        created = db.create_camera(cam)
        print(f"✓ Successfully registered '{name}' (Source: {source}, ID: {created['id']})")
        return True
    except Exception as exc:
        print(f"✗ Failed to register camera: {exc}")
        return False


def interactive_menu(db: Database):
    print_banner()
    cams = list_devices()

    phones = [c for c in cams if c.get('is_phone_candidate') and c['width'] > 0]
    other_working = [c for c in cams if not c.get('is_phone_candidate') and c['width'] > 0]

    print("\nActions:")
    if phones:
        print(f"  [1] Add Phone Link Camera ({phones[0]['friendly_name']} on index {phones[0]['index']})")
    else:
        print("  [1] Scan / Add a Connected Camera (Index 0, 1, 2...)")
    print("  [2] Add Wi-Fi IP Camera / Phone Stream (e.g. IP Webcam app URL)")
    print("  [3] Test an existing camera source")
    print("  [Q] Quit")

    choice = input("\nEnter choice [1-3, Q]: ").strip().upper()
    if choice == '1':
        if phones:
            p = phones[0]
            name = input(f"Enter camera name [{p['friendly_name']}]: ").strip() or p['friendly_name']
            if test_camera_feed(p['source']):
                register_camera(name, p['source'], db)
        else:
            if not cams:
                print("No cameras available. Connect your device and re-run.")
                return
            idx = input(f"Enter camera index [0-{len(cams)-1}]: ").strip()
            if idx.isdigit() and int(idx) < len(cams):
                c = cams[int(idx)]
                name = input(f"Enter camera name [{c['friendly_name']}]: ").strip() or c['friendly_name']
                if test_camera_feed(c['source']):
                    register_camera(name, c['source'], db)
            else:
                print("Invalid index.")
    elif choice == '2':
        print("\nWi-Fi Phone Camera Setup (e.g. Android 'IP Webcam' app from Play Store):")
        print("Start the server in the app, then enter the video stream URL below.")
        print("Common format: http://192.168.1.X:8080/video or rtsp://192.168.1.X:8080/h264_pcm.sdp")
        url = input("\nEnter stream URL: ").strip()
        if url:
            name = input("Enter camera name [Phone Wi-Fi Camera]: ").strip() or "Phone Wi-Fi Camera"
            if test_camera_feed(url):
                register_camera(name, url, db)
    elif choice == '3':
        src = input("Enter camera source to test (index or URL): ").strip()
        if src:
            test_camera_feed(src)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true', help='List all detected camera devices and exit')
    p.add_argument('--test', type=str, help='Test a camera source (e.g. 0, 1, rtsp://...)')
    p.add_argument('--add', nargs=2, metavar=('NAME', 'SOURCE'), help='Register camera by name and source (e.g. "Phone" 1)')
    p.add_argument('--auto-phone', action='store_true', help='Automatically register detected phone camera if available')
    args = p.parse_args()

    db = Database(settings.data_dir / 'visionguard.db')

    if args.list:
        list_devices()
        return

    if args.test:
        test_camera_feed(args.test)
        return

    if args.add:
        name, source = args.add
        if test_camera_feed(source):
            register_camera(name, source, db)
        return

    if args.auto_phone:
        phones = camera_utils.get_phone_cameras()
        if not phones:
            print("No phone cameras detected via Phone Link.")
            sys.exit(1)
        p = phones[0]
        register_camera(p['friendly_name'], p['source'], db)
        return

    interactive_menu(db)


if __name__ == '__main__':
    main()
