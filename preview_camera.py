#!/usr/bin/env python3
"""VisionGuard Live Camera Preview.

Pops up a real-time on-screen window showing the camera feed
with FPS, resolution, and status overlay. Press 'Q' or 'ESC' to close.
"""
import argparse
import sys
import time
import os
import cv2
import camera_utils


def preview(source="1", window_title="VisionGuard - Camera Live Preview"):
    print(f"\nOpening camera source: {source}...")
    cap = camera_utils.open_capture(source)
    
    if not cap.isOpened():
        print(f"✗ Failed to open source '{source}'. Trying fallback...")
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            print(f"✗ Could not open camera {source}.")
            return False

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 960, 540)

    fps_count = 0
    fps_start = time.time()
    current_fps = 0.0

    print(f"✓ Camera connected! Live window opened on screen.")
    print("  -> Press 'Q' or 'ESC' in the preview window to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # show waiting frame
                import numpy as np
                frame = np.zeros((540, 960, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for video frame...", (40, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            else:
                h, w = frame.shape[:2]
                fps_count += 1
                if time.time() - fps_start >= 1.0:
                    current_fps = fps_count / (time.time() - fps_start)
                    fps_count = 0
                    fps_start = time.time()

                # Overlay status bar
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (20, 20, 20), -1)
                cv2.putText(frame, f"VisionGuard Phone / Camera [{source}] | {w}x{h} @ {current_fps:.1f} FPS | Press 'Q' to Exit",
                            (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)

            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                break
            # Check if window was closed by user clicking X
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Preview closed.")

    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='1', help='Camera index (0, 1) or stream URL (default: 1)')
    args = p.parse_args()
    preview(args.source)
