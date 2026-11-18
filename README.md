# VisionGuard AI — Multi-Model Surveillance Intelligence

A **local-first** AI-powered surveillance platform with multi-model detection, real-time alerts, and voice sirens.

## 🚀 Key Features

### Multi-Model Detection Pipeline
| Model | Purpose | Classes |
|-------|---------|---------|
| **YOLOv8n** | Person detection + ByteTrack tracking | Person (COCO class 0) |
| **YOLOv8n-Pose** | Fall detection, loitering, pose anomalies | 17 COCO keypoints |
| **Custom Fire Model** | Fire & smoke detection | Fire, Smoke |
| **Custom Violence Model** | Fight, assault, weapon detection | Violence, Fight, Weapon, Knife, Gun |
| **YOLOv8n (General)** | All COCO objects | 80 classes (vehicles, animals, items) |

### Alert Channels
- 🔊 **Voice Siren (TTS)** — Offline Windows SAPI text-to-speech alerts
- 🖥️ **Real-time WebSocket** — Live dashboard alerts
- Telegram is disabled and no bot token or chat ID is required. Model-generated detections below 50% confidence are discarded before event creation or alerting.

### Detection Capabilities
- ✅ **Zone Intrusion** — Polygon restricted areas
- ✅ **Bidirectional Line Counting** — Entry/exit totals
- ✅ **Fire & Smoke** — Early fire detection
- ✅ **Violence & Weapons** — Fight, assault, knife, gun detection
- ✅ **Fall Detection** — Person collapse via pose estimation
- ✅ **Loitering** — Suspicious dwelling in area
- ✅ **Proximity Alert** — Close person-to-person proximity (potential altercation)
- ✅ **Heat Source Detection** — Pre-fire safety (ovens, heaters, electronics)
- ✅ **General Objects** — Vehicles, animals, packages, etc.

## 📦 Quick Start (Windows)

### 1. Prerequisites
- **Python 3.11+** (recommended)
- **NVIDIA GPU with CUDA** (for best performance) or CPU
- **Windows 10/11**

### 2. Install
```cmd
# Double-click or run from command prompt
install_windows.bat
```

This will:
- Create virtual environment (`.venv/`)
- Install PyTorch with CUDA 12.1 support
- Install all Python dependencies
- Download base YOLO models (person, pose)
- Create `.env` from template
- Create data directories

### 3. Configure
Edit `.env` before the first run. Set a strong unique admin password and session signing secret:

```env
ADMIN_USERNAME=admin
ADMIN_PASSWORD=replace-with-a-long-unique-password
SESSION_SECRET=replace-with-a-long-random-secret
APP_TIMEZONE=Asia/Kolkata
TTS_ENABLED=true
```

### 4. Run
```cmd
run.bat
```

### 5. Access Dashboard
Open **http://127.0.0.1:8000** and sign in with `admin` plus the password configured in `.env`.

### 6. Separate admin and client workspaces
- Administrators add cameras, edit RTSP/webcam sources, test connections, and assign each camera to a client.
- Administrators create client accounts under **Client workspaces**. Client usernames and passwords are separate.
- Clients only see cameras, event history, alerts, and streams assigned to their account. They cannot add/delete cameras or change camera source URLs.
- Client passwords are stored as salted PBKDF2 hashes. Set `ADMIN_PASSWORD` before creating clients.
- This is a local single-server deployment; use a VPN/reverse proxy with HTTPS and firewall rules before remote access.

## 🔧 Configuration Guide

### Camera Setup
1. Sign in as administrator and click **+ Add camera**.
2. Enter a camera name and source: RTSP URL, HTTP/MJPEG URL, webcam index (`0`), or local video file.
3. Assign the camera to a client or leave it under the administrator workspace.
4. Select the camera and use **Test connection**. The dashboard shows connecting/online/reconnecting status and the live MJPEG preview.
5. Use **Edit source** to change the stream URL; VisionGuard restarts the camera worker automatically.
6. Select camera → **Draw zone** (3+ points for intrusion) or **Draw line** (2 points for counting).

### Detection Modules (per-camera)
Go to **⚙ Detection config** tab to enable/disable:
- 🔥 **Fire & Smoke** — Requires `models/yolov8n-fire.pt`
- ⚔ **Violence & Weapons** — Requires `models/yolov8n-violence.pt`
- 🧍 **Anomaly (Pose)** — Uses `yolov8n-pose.pt` (auto-downloaded)
- 📦 **General Objects** — Uses `yolov8n.pt` (all COCO classes)

### Voice Siren (TTS)
- Uses **pyttsx3** with Windows SAPI (offline, no internet)
- Configure in **🔔 Alert settings** tab
- Test with **Test Voice Alert** button
- Triggers on: fire, smoke, violence, weapon, fight, assault, intrusion

### Custom Model Training
For fire/violence detection, train your own models:

```bash
# Fire detection example
yolo detect train data=fire_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
# Save as models/yolov8n-fire.pt

# Violence detection example
yolo detect train data=violence_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
# Save as models/yolov8n-violence.pt
```

**Datasets:** Search Roboflow Universe for "fire", "smoke", "violence", "weapon", "fight"

## 📁 Project Structure

```
visionguard/
├── app/
│   ├── main.py           # FastAPI app, REST API, WebSocket
│   ├── engine.py         # Camera workers, detection pipeline
│   ├── config.py         # Settings from .env
│   ├── database.py       # SQLite WAL persistence
│   ├── analytics.py      # Zone intrusion, line counting
│   ├── detectors.py      # Fire, Violence, Anomaly, Object detectors
│   ├── alerts.py         # TTS alert service
│   ├── schemas.py        # Pydantic models
│   └── static/           # Dashboard HTML/CSS/JS
├── models/               # YOLO .pt model files
├── data/                 # SQLite DB, snapshots
├── tests/                # Unit tests
├── .env                  # Your configuration (create from .env.example)
├── requirements.txt      # Python dependencies
├── install_windows.bat   # Windows installer
├── run.bat               # Windows launcher
├── download_models.py    # Model downloader
└── README.md
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/cameras` | List cameras with telemetry & config |
| POST | `/api/cameras` | Add camera |
| PATCH | `/api/cameras/{id}` | Update camera |
| DELETE | `/api/cameras/{id}` | Delete camera |
| GET | `/api/cameras/{id}/config` | Get detection config |
| PATCH | `/api/cameras/{id}/config` | Update detection config |
| GET | `/api/events` | Query events (filter by kind, camera, limit) |
| GET | `/api/events.csv` | Export all events as CSV |
| GET | `/api/alert-log` | Alert delivery log |
| GET | `/api/stats` | System statistics |
| GET | `/api/detection-types` | Supported event types |
| GET | `/api/system/info` | System & GPU info |
| POST | `/api/alerts/test` | Test all alert channels |
| GET | `/stream/{camera_id}` | MJPEG live stream |
| WS | `/ws/alerts` | Real-time event WebSocket |

## 🎯 Event Types & Severity

| Category | Event Types | Severity |
|----------|-------------|----------|
| Core | `intrusion`, `entry`, `exit` | high, low, low |
| Fire | `fire_detected`, `smoke`, `heat_source` | critical, high, low |
| Violence | `violence`, `weapon`, `fight`, `assault`, `proximity_alert` | critical, critical, critical, high, medium |
| Anomaly | `fall_detected`, `loitering` | high, medium |

## 🛠️ Advanced Configuration

### Performance Tuning (.env)
```env
# Inference settings
DEVICE=0              # GPU index (0=first), 'cpu', or 'auto'
IMGSZ=640             # Inference resolution (416/640/1280)
FRAME_SKIP=2          # Process every Nth frame (1=every frame)
MAX_FRAME_WIDTH=960   # Resize wider frames
CONFIDENCE=0.25       # Detection confidence threshold

# Alert cooldowns
EVENT_COOLDOWN_SECONDS=20      # Per-track event cooldown
SIREN_DURATION_SECONDS=10      # TTS alert duration
SIREN_REPEAT_COUNT=3           # TTS repeat count
```

### GPU Requirements
- **NVIDIA GPU** with 4GB+ VRAM (8GB+ recommended for multi-model)
- **CUDA 12.1** compatible driver
- PyTorch CUDA 12.1 wheels (installed by `install_windows.bat`)

## 🧪 Testing

```cmd
# Install dev dependencies
.venv\Scripts\pip.exe install -r requirements-dev.txt

# Run tests
.venv\Scripts\pytest.exe -q
```

## 📊 Dashboard Features

### Live Overview
- Real-time camera grid with annotated video
- Live statistics (cameras, people, intrusions, crossings, fire, violence)
- Detection badges on video feed

### Event History
- Filterable event log with severity badges
- Snapshot viewer
- CSV export

### Detection Config
- Per-camera module toggles
- Sensitivity adjustment
- Local voice siren enable/disable

### Alert Settings
- Offline TTS voice alert test
- No Telegram bot token or chat ID required

## ⚠️ Important Disclaimers

> **This is NOT a certified security or life-safety system.**
> - False alarms and missed detections WILL occur
> - Only monitor cameras/spaces you own or are authorized to access
> - Comply with all applicable privacy laws
> - Inform affected people appropriately
> - Test thoroughly before relying on alerts

## 🔒 Security Notes

- Default binds to **localhost only** (`HOST=127.0.0.1`)
- Set `ADMIN_PASSWORD` and a unique `SESSION_SECRET` in `.env`; legacy `DASHBOARD_PASSWORD` is supported
- Use **HTTPS + VPN + Firewall** for remote access
- RTSP credentials stored in local SQLite (plaintext); protect the data directory and backups
- No cloud connectivity — all processing local

## 📄 License

- VisionGuard code: MIT License
- Ultralytics YOLO: AGPL-3.0 (review before commercial use)
- PyTorch: BSD-3-Clause

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new detection types
4. Submit PR with description

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera "reconnecting" | Verify RTSP URL in VLC first; check network/firewall |
| "Loading model" stuck | First run downloads models — needs internet |
| No people detected | Lower `CONFIDENCE` to 0.15; ensure person visible |
| Slow inference | Reduce `IMGSZ` to 416; increase `FRAME_SKIP`; verify GPU |
| TTS not speaking | Verify `pyttsx3` installed; test in Alert settings tab |
| CUDA not detected | Reinstall PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |

## 📚 Resources

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [COCO Dataset Classes](https://cocodataset.org/#home)
- [Roboflow Universe](https://universe.roboflow.com/) (datasets for custom models)
- [pyttsx3 Documentation](https://pyttsx3.readthedocs.io/)

---

**VisionGuard AI** — Local-first surveillance intelligence for the modern era. 🛡️

## Camera setup and troubleshooting

1. Sign in as administrator and choose **+ Add camera**.
2. Use **Discover USB cameras** to scan local camera indexes, or enter an RTSP URL / local video path.
3. Choose **Test source** before saving. A failed test does not block saving an offline camera; the camera will appear with an Offline status and retry automatically.
4. Select **Add & connect**. The camera should appear in the sidebar. Select it to view the live MJPEG preview.
5. If the sidebar says it cannot load cameras, use Retry and inspect the error shown below the button.

The backend now initializes the SQLite schema before optional USB-camera discovery, preventing first-start discovery from running against a database whose tables do not yet exist. Camera creation rejects duplicate source strings. Windows USB capture prefers DirectShow and falls back to OpenCV's default backend.

## Confidence policy

All detector model confidence settings are clamped to at least 0.50. Any event carrying a confidence value below 0.50 is dropped before snapshots, database event insertion, WebSocket broadcast, or alert processing. Zone/line analytics events without model confidence continue to work. Telegram delivery is disabled; local TTS is optional.

## Upload a video for offline AI testing

Open the dashboard and use **Upload video for AI testing**. Supported formats are MP4, AVI, MOV, MKV, WEBM and M4V; maximum upload size is 250 MB. The uploaded clip is processed locally with the configured fire/smoke, violence/weapon, pose/anomaly and object detectors. The dashboard returns a detection summary and an annotated MP4. The upload itself is deleted after analysis; the annotated output remains under `data/test_videos/`.

- Minimum confidence floor: **50%**. Detections below 0.50 are discarded.
- Local TTS alarm: a detection type must recur on at least two processed frames in the uploaded clip before a test alarm is triggered.
- Telegram: disabled; no bot token or chat ID is required.
- Smoke is an early warning signal; this software does not predict ignition or guarantee fire prevention. Test results depend on the installed model weights and camera/video quality.
- For weapon/fight/fire model validation, use both positive clips and negative/normal clips. Do not treat one successful sample as proof of reliable real-world detection.

### Public sample videos and datasets for testing

Use the providers' own download instructions and follow each dataset's licence/registration terms:

- **Smoke / fire:** MIVIA Smoke Detection Dataset (149 videos, challenging smoke/cloud negatives): https://mivia.unisa.it/datasets/video-analysis-datasets/smoke-detection-dataset/
- **Fire + false-positive negatives:** MIVIA Fire Detection Dataset (31 clips; includes non-fire red objects, smoke and clouds): https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/
- **Fireground / public safety:** NIST Consumer Digital Video Library; register at CDVL and search PSCR Fireground / Event Identification samples: https://www.nist.gov/ctl/pscr/consumer-digital-video-library
- **Fire/smoke surveillance samples:** D-Fire project page includes surveillance-video links and dataset resources: https://gaia-solutions-on-demand.github.io/DFireDataset/
- **Fight / assault / shooting / normal anomaly clips:** UCF-Crime-X3D dataset mirror on Hugging Face; review the dataset card and licence before use: https://huggingface.co/datasets/CosmicAstro/UCF-Crime-X3D
- **Firearm actions (video clips with labels):** Mendeley firearm-related action-recognition dataset. Licence is CC BY-NC 3.0; commercial use requires prior permission: https://data.mendeley.com/datasets/bbzpxhd22j/2
- **Knife / handgun / rifle detection research datasets:** OD-WeaponDetection project page, including separate knife and pistol resources: https://github.com/ari-dasci/OD-WeaponDetection

Some of these are full datasets rather than one-click MP4 samples; extract individual video files before uploading. The weapon dataset licence may restrict commercial use. Never use downloaded footage as proof of detector performance without checking the ground-truth labels.


## Telegram security alerts (optional)

Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`. Telegram sends only configured suspicious/security event types when the event has an explicit model confidence strictly greater than 75% (default threshold `TELEGRAM_MIN_CONFIDENCE=0.76`). Events at 75% or below, and events without a model confidence score, are never sent to Telegram. Enable `telegram_enabled` for each camera in its camera configuration. `TELEGRAM_COOLDOWN_SECONDS` limits repeat notifications per camera and event type. Do not share your bot token; keep `.env` private.
