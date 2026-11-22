> **Historical documentation:** this original README refers to Python 3.11. The CURRENT installer requires Python 3.12 x64; follow README.md and START_HERE.txt for installation.

# VisionGuard AI — RTSP surveillance, unusual-event photos and Telegram

A local FastAPI + YOLO video surveillance application for Windows, RTSP and webcams. **This updated release saves still photos only for confirmed, scored, unusual incidents — not every normal or stationary frame.** Video-test output is still a *full annotated MP4* so you can review missed events.

## What changed in this release

- **No loitering detection or alerts**, regardless of an old `.env` value.
- **50% minimum model confidence** for incident snapshots and Telegram; raw model scores are not inflated by repeated smoke detections or nearby people.
- **Two consecutive analyzed frames** are required for an incident. One snapshot per continuing incident. Smoke can escalate to a separately saved confirmed fire incident. New incidents after an absence can be saved again.
- **Only** `fire_detected`, `smoke`, `weapon`, `violence`, `unusual_activity` or a genuinely classed `fall_detected` can create a security photo. Zone crossings, normal people, proximity, loitering, heat-source labels and pending fire candidates do not create pictures. Normal analytics may still be retained as text in event history.
- **Video-test snapshots** have user-scoped URLs and are displayed under **Saved incident photos**; the annotated MP4 retains full frames for auditing.
- **Improved small-object inference:** larger `FIRE_IMGSZ`, `OBJECT_IMGSZ`, optional overlapping-quadrant `KNIFE_TILE_SCAN`, optional specialist knife checkpoint, more frequent testing of threat models, and correct real detection boxes.
- Telegram sends **actual** confirmed model events if `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set. Video tests have Telegram on but local voice sirens off.
- `verify_models.py` exposes which actual checkpoints and labels are ready. Missing specialist models are **not silently counted as working**.

See [FIX_REPORT.md](FIX_REPORT.md) for the provided-video inspection and limitations.

## Models that are actually in this ZIP

| Checkpoint | Present | Capabilities |
|---|---|---|
| `models/yolov8n-fire.pt` | Yes | Custom **fire and smoke** model; labels verified from checkpoint metadata |
| `models/yolov8n.pt` | Yes | COCO people and **knife** (class 43), but small knives can be missed |
| `models/yolov8n-pose.pt` | Yes | Person pose landmarks, not a general unusual-action classifier |
| `models/fight-yolov8-nano.pt` | **No** | Install requested Musawer14 fight/assault checkpoint using installer |
| `models/cs2-yolo12-weapon-detection.pt` | **No** | Install requested Counter-Strike 2 **game gun** checkpoint (NOT real knives) |

**Important:** A generic YOLOv8n COCO knife label is not sufficient to guarantee detection of a tiny blade in a moving or blurry video. Higher resolution and tiling increase the chances but do **not** guarantee detection. Generic abnormal movements (knife trick, assault, etc.) require suitable *trained action/weapon detection weights*. A pose model's confidence is for a **person**, not the fall/action; VisionGuard will not misrepresent it as a 50%+ action-detection score.

## Setup (Windows 10/11, Python 3.11 recommended)

1. Unzip to a new project folder. Do not replace your active `.env` or event database without first backing them up.
2. Double-click `install_windows.bat`, or create a virtual environment and install `requirements.txt` yourself. Installation needs internet access to install Python packages.
3. If `.env` does not already exist, it is copied from `.env.example`. Otherwise **manually merge the new settings** from `.env.example` into your existing `.env`.
4. Put your Telegram bot token and chat ID in `.env` (keep these private), then run `run.bat`.
5. Open <http://127.0.0.1:8000> and enable the camera's Telegram switch in the Detection config tab. On a new installation, sign in using the configured admin password (set `ADMIN_PASSWORD` and `SESSION_SECRET` in `.env`).

Model preflight:

```cmd
.venv\Scripts\python.exe verify_models.py
.venv\Scripts\python.exe check_fire.py
```

To install **your own trusted specialist knife model** obtained from a legitimate source and trained with actual `knife`/`weapon` labels:

```cmd
.venv\Scripts\python.exe setup_hf_models.py --weapon "C:\Downloads\best.pt"
```

The script checks for compatible labels, copies the checkpoint to `models/cs2-yolo12-weapon-detection.pt`, and prints its SHA-256 digest. Do not load untrusted `.pt` files: PyTorch checkpoints may contain executable serialized objects. You can also install a dedicated fight model via `--violence "C:\Downloads\fight.pt"`. Both are optional; **no specialist knife/fight weights are bundled**. Validate any custom model on representative labelled footage before relying on it for safety monitoring.

## Recommended `.env` for catching small knives / fire

```env
FRAME_SKIP=1
INCIDENT_MIN_CONFIDENCE=0.50
INCIDENT_CONFIRMATION_FRAMES=2
ENABLE_LOITERING_DETECTION=false
FIRE_MODEL_PATH=models/yolov26-fire-smoke.pt
FIRE_CONFIDENCE=0.30
FIRE_IMGSZ=768
OBJECT_MODEL_PATH=yolov8n.pt
OBJECT_CLASSES=43
OBJECT_CONFIDENCE=0.40
OBJECT_IMGSZ=960
OBJECT_EVERY_N_FRAMES=3
KNIFE_TILE_SCAN=true
WEAPON_MODEL_PATH=models/cs2-yolo12-weapon-detection.pt
WEAPON_IMGSZ=960
WEAPON_EVERY_N_FRAMES=2
TELEGRAM_MIN_CONFIDENCE=0.50
TELEGRAM_COOLDOWN_SECONDS=60
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

`KNIFE_TILE_SCAN=true` performs several extra YOLO passes: **much slower** on CPU, helpful only when there is a capable knife checkpoint and adequate video detail. Turn it off for speed or keep the original object sampling rate for high camera counts. Higher `FIRE_IMGSZ` also increases compute demand.

**The 0.30 / 0.40 detector candidate thresholds do not bypass the 0.50 incident threshold.** They allow early multi-frame model evidence to be assessed; only the model's unboosted score at/above 0.50 qualifies for photos and Telegram.

If the uploaded test contains only a momentary single frame of a tiny knife, change `INCIDENT_CONFIRMATION_FRAMES=1` temporarily for diagnosis, but expect more false positives. For operational monitoring, use `2` or validate a stronger model.

## Test a video and view photos

In the dashboard, choose **Video testing**, select a video and click **Upload & analyse**. The response includes:

- `events`: counts of model-detected threat frames scoring at least 50% (not numbers of saved photos).
- `confirmed_events`: one entry per incident (kind, confidence, timestamp, bounding box and `snapshot_url`).
- `annotated_video_url`: full MP4, with bounding boxes for model detections.
- `models_loaded`, `warnings`, and `fire_detector`: diagnostics. Never interpret a missing-model empty video as a successful negative test.

For API use, first obtain an authenticated dashboard session (the endpoint is not public). Example after logging in through your browser:

```bash
curl -b cookies.txt -F "file=@C:/Videos/knife-test.mp4" http://127.0.0.1:8000/api/test-videos
```

For a terminal-only login you can create the cookie file with your credentials:

```bash
curl -c cookies.txt -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"YOUR_ADMIN_PASSWORD"}' \
  http://127.0.0.1:8000/api/login
```

The full video remains available even when **zero** threats are confirmed. Only incident JPGs are stored as test photos. Old video-test JPGs are cleaned up using the existing event retention policy. Never place Telegram tokens or passwords in uploaded video filenames or public logs.

## Automated checks

Run:

```cmd
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

The regression tests cover a missing fire model, fire confirmation timing, model label mapping, confidence floors, incident deduplication, Telegram policy, snapshots and authenticated test-video photo URLs. **Mocks validate the program logic, not the quality of trained model weights on your specific videos.**

## Limitations and safety

- Fire / smoke detection can produce false positives and miss flames; do not use this alone in place of fire alarms.
- A bare pose model **does not** recognize arbitrary unusual actions. For genuine `unusual_activity` alerts you need trained weights producing that label (or trained fight, fall or weapon labels with model confidence).
- A small knife is not guaranteed detectable even with tiled inference if the input lacks sufficient pixels, illumination or focus.
- Telegram requires working internet, bot permissions, chat ID and the camera's Telegram opt-in. No actual Telegram delivery can be verified without your private credentials.
- This ZIP includes the source and bundled original weights, but it does not include your `.env`, Telegram secrets, running database or uploaded videos.
- Observe relevant privacy, safety and licensing requirements (the bundled fire checkpoint is documented under `models/FIRE-MODEL.md`).


## Fix: Uploaded fire/smoke video reports 0 evaluated frames

The `SalahALHaismawi/yolov26-fire-detection` model has an actual checkpoint named **best.pt** (not `<weights>.pt`) and contains **fire**, **smoke**, and **other** classes. The installer now fetches it using `huggingface_hub` and verifies both fire and smoke labels before installing `models/yolov26-fire-smoke.pt`. The stock YOLO person detector is **not** a substitute for fire/smoke.

Run `install_windows.bat` again, or, inside the existing virtual environment:

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe install_fire_smoke_model.py
.venv\Scripts\python.exe check_fire.py
.venv\Scripts\python.exe run_fire_smoke_video.py --input "your-fire-video.mp4" --output "fire-smoke-detected.mp4"
```

If Hugging Face is blocked, download `best.pt` manually from https://huggingface.co/SalahALHaismawi/yolov26-fire-detection/tree/main and use `python install_fire_smoke_model.py --local "C:\path\best.pt"`.

`run_fire_smoke_video.py` examines **every decoded frame** (no skipping). It draws boxes for all real fire/smoke predictions >=20% for inspection, while confirmed incident photos are only saved from predictions >=50% on 2 consecutive frames. A JSON incident report accompanies the MP4. If the required model is missing, it exits with an error instead of silently producing an empty video. No synthetic detections are generated.

This community model's reported performance is on its own validation data, not a guarantee for matchstick flames or your camera footage. Model files are NOT bundled here because Hugging Face could not be accessed from the build environment. Run the installer on a network-connected system.

## Added specialist weapon + fight detection (18 November 2026)

Two user-selected, **external** checkpoints are integrated with the real-time
pipeline and uploaded-video tests. They are NOT in this ZIP: download them on a
machine with internet access, after reviewing the trust risk of PyTorch `.pt`
checkpoints. The code verifies the published SHA256 before loading a remote
checkpoint and validates the actual model class names.

| Role | Hugging Face repository | Actual downloadable filename | Local installed filename |
|---|---|---|---|
| CS2 weapon | `jparedesDS/cs2-yolo12m-weapon-detection` | `cs2-yolo12m-weapon-detection.pt` | `models/cs2-yolo12-weapon-detection.pt` |
| Fighting/assault | `Musawer14/fight_detection_yolov8` | `Yolo_nano_weights.pt` | `models/fight-yolov8-nano.pt` |

**Warning:** The CS2 model detects 19 **Counter-Strike 2 game gun classes** and
contains **no knife class**. It does not establish accuracy for real CCTV guns.
For tiny **real knives**, the separately retained stock COCO knife detector can
help, but a trained CCTV knife model is needed for reliable small-blade results.
Do not equate a pistol-shaped game object with an actual firearm.

### Install the two trained checkpoints on Windows

Double-click `DOWNLOAD_SECURITY_MODELS_WINDOWS.bat` and confirm that you trust
the checkpoint publishers, or run:

```bat
.venv\Scripts\python.exe install_security_models.py --accept-checkpoint-risk
.venv\Scripts\python.exe verify_models.py
```

Alternatively download the two files manually from their Hugging Face repos
and use `--local-weapon C:\path\cs2-yolo12m-weapon-detection.pt`
and `--local-fight C:\path\Yolo_nano_weights.pt` with the installer.
The checkpoints are loaded **only after** explicit acknowledgement because
ordinary PyTorch `.pt` files can run serialized code.

The `.env.example` defaults now select the installed names. If reusing an older
`.env`, the installer migrates the two old default paths but **does not replace
custom paths**. Restart the running server after installing weights.

### Test every video frame

Drag an MP4 onto `RUN_SECURITY_TEST_WINDOWS.bat`, or run:

```bat
.venv\Scripts\python.exe run_security_video.py --input "C:\Videos\test.mp4" --output security-annotated.mp4
```

Outputs: `security-annotated.mp4` (all decoded frames annotated),
`security-annotated.json` (which models loaded, confidence and incident data),
and an incident-only `security-annotated/` JPG folder.

The dashboard's **Upload & analyse** workflow now evaluates every decoded frame
for both trained specialist models. If **all** models are missing, the upload
fails with an explicit configuration error instead of showing a false "no
detections" result. If one specialist is missing, other available detectors
continue running and the result includes warnings.

A fight classification must have a *positive* model label such as `Violence`,
`Fight`, or `Assault`. `NoViolence`, `NoFight`, normal people, and unknown class
IDs do not create fight alerts. The video alert rule remains **two qualifying
frames with raw confidence >=50%**, one picture per continuous incident and
Telegram alerts only when configured. Upload tests have no local sirens;
standalone `run_security_video.py` intentionally makes no Telegram calls.

**Troubleshooting:** For a Windows CUDA GPU, configure `DEVICE=0` in `.env`;
for CPU, `DEVICE=cpu`. All sample code defaults to `DEVICE=auto` when the GPU
is unavailable. Loading the YOLO12 specialist requires a recent Ultralytics
release (`ultralytics>=8.3.80`). Reported training-set performance cannot prove
accuracy on the user's uploaded videos. Real end-to-end inference could not be
verified in this build environment (Ultralytics and model weights unavailable).
