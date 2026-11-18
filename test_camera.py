#!/usr/bin/env python
"""Quick camera test for VisionGuard - shows available cameras and tests them."""
import sys
sys.path.insert(0, '.')

from camera_utils import detect_cameras, get_default_camera, test_camera

def main():
    print("=" * 50)
    print("VisionGuard Camera Test")
    print("=" * 50)
    
    # Detect cameras
    print("\nDetecting cameras...")
    cameras = detect_cameras()
    
    if not cameras:
        print("No cameras found!")
        return 1
    
    print(f"\nFound {len(cameras)} camera(s):")
    for cam in cameras:
        status = "WORKS" if cam['width'] > 0 else "NO FRAME"
        print(f"  Camera {cam['index']}: {cam['width']}x{cam['height']} - {status}")
    
    # Get default
    default = get_default_camera()
    if default is not None:
        print(f"\nDefault camera: {default}")
        if test_camera(str(default)):
            print(f"Camera {default} test: PASSED")
        else:
            print(f"Camera {default} test: FAILED")
    else:
        print("\nNo working camera available")
        return 1
    
    # Test all
    print("\nTesting all cameras...")
    for cam in cameras:
        if cam['width'] > 0:
            if test_camera(str(cam['index'])):
                print(f"  Camera {cam['index']}: OK")
            else:
                print(f"  Camera {cam['index']}: FAIL")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())