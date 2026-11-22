# Using Your Smartphone as a VisionGuard Security Camera

VisionGuard AI natively supports using Android and iOS smartphones as AI surveillance cameras alongside standard USB webcams and RTSP security streams.

---

## Method 1: Windows 11 Phone Link ("Connected Camera") — Recommended

If you have a Windows 11 PC and an Android phone, you can use your phone's camera wirelessly with zero extra third-party desktop apps.

### Prerequisites:
- **Windows 11** (Build 22621.3235 or newer, 23H2+)
- **Android Phone** (Android 9.0 or newer)
- Both devices connected to the same Wi-Fi network and Bluetooth enabled.

### Step-by-Step Setup:
1. **Enable Mobile Devices in Windows:**
   - Open Windows **Settings** (`Win + I`).
   - Go to **Bluetooth & devices** > **Mobile devices**.
   - Turn **ON** *"Allow this PC to access your mobile devices"*.
2. **Link Your Phone:**
   - Click **Manage devices** and scan the QR code using the **Link to Windows** app on your phone.
3. **Turn on Connected Camera:**
   - In the **Manage devices** window, locate your phone and toggle **ON** *"Use as a connected camera"*.
   - If prompted, accept the camera streaming permission on your smartphone screen.
4. **Add to VisionGuard:**
   - Double-click **`CONNECT_PHONE_CAMERA_WINDOWS.bat`** (or run `python phone_camera_setup.py`).
   - VisionGuard will automatically detect your phone (labeled `[PHONE LINK / MOBILE]`).
   - Press `1` to test the live stream and register it into VisionGuard.
   - Start the server (`run.bat`) and view your phone's live AI detections on `http://127.0.0.1:8000`.

---

## Method 2: Wi-Fi IP Camera App (Android / iPhone / Windows 10)

If you use Windows 10, an iPhone, or want a dedicated wireless RTSP/HTTP camera stream:

### Step-by-Step Setup:
1. **Install a streaming app on your phone:**
   - **Android:** Download **"IP Webcam"** by Pavel Khlebovich from the Google Play Store.
   - **iPhone:** Download **"RTSP Camera"** or **"Live-Reporter Security Camera"** from the App Store.
2. **Start the stream:**
   - Open the app and tap **Start Server**.
   - Note the Wi-Fi address displayed on the screen (e.g. `http://192.168.1.45:8080/video` or `rtsp://192.168.1.45:8080/h264_pcm.sdp`).
3. **Register in VisionGuard:**
   - Run **`CONNECT_PHONE_CAMERA_WINDOWS.bat`** and select Option **[2]** (*Add Wi-Fi IP Camera*).
   - Enter your phone's stream URL and a friendly name (e.g., *"Living Room Phone"*).
   - VisionGuard validates the live feed and adds it to your active surveillance roster.

---

## Method 3: DroidCam / Iriun Webcam (USB or Wi-Fi Virtual Device)

1. Install **DroidCam** or **Iriun Webcam** on both your PC and phone.
2. Connect via Wi-Fi or USB cable.
3. The app creates a virtual webcam on your Windows PC (typically Camera index `1` or `2`).
4. VisionGuard will automatically list it under `CONNECT_PHONE_CAMERA_WINDOWS.bat`.

---

## CLI & Automation Reference

```powershell
# List all detected physical and connected phone cameras
python phone_camera_setup.py --list

# Test a camera feed for 4 seconds and measure real FPS
python phone_camera_setup.py --test 1

# Register a camera directly into VisionGuard database
python phone_camera_setup.py --add "Front Gate Phone" 1

# Automatically register any detected Phone Link camera
python phone_camera_setup.py --auto-phone
```

---

## Troubleshooting

- **Black frame or "NO FRAMES":**
  Ensure your phone screen is unlocked and camera permissions are granted. For Phone Link, tap the "Pause/Resume" button in the Windows Connected Camera overlay.
- **High latency on Wi-Fi:**
  In the phone app, set the resolution to `1280x720` or `1920x1080` at `30 FPS` with H.264 or MJPEG encoding.
- **Multiple cameras:**
  VisionGuard automatically limits concurrent active cameras based on `MAX_ACTIVE_CAMERAS` in `.env` (default: 4) to ensure your GPU memory is never overloaded.
