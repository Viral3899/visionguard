# VisionGuard — Python 3.12 Multi-Camera GPU Deployment + Auto-Training

This version adds **site camera import, GPU-aware multi-camera scheduling, real online training datasets for 3 separate tasks, explicit model activation, and a dedicated inference launcher** to the existing VisionGuard surveillance dashboard.

## Start here (Windows)

1. Install **Python 3.12 64-bit** and a current NVIDIA driver. Extract to a short local folder path.
2. **One-click full setup and automatic training:** `ONE_CLICK_SETUP_TRAIN_DEPLOY.bat` (license-review prompts and internet required; may need many GB disk and long GPU processing).
3. **Quick pretrained-model setup without training:** `SETUP_MODELS_ONLY.bat`.
4. **Configure multiple cameras:** `CONFIGURE_CAMERAS_WINDOWS.bat`, or create `cameras.json` from `cameras.example.json` and set `${CAMERA_NAME}` values in `.env`.
5. **Normal live inference, no training:** `INFERENCE_MULTI_CAMERA.bat`. **Offline video inference:** `INFERENCE_VIDEO.bat` or `python inference.py --mode video --input clip.mp4`.
6. **Train separately any time:** `TRAIN_ALL_MODELS.bat`.
7. **Readiness checks:** `VERIFY_MULTI_CAMERA.bat`.
8. **Optional trained-model activation after real, separately labeled CCTV tests:** `ACTIVATE_TRAINED_MODELS_AFTER_REVIEW.bat`.

Open dashboard at `http://127.0.0.1:8000` by default. Generated admin credentials are saved in `.env` on the first install. Never expose the dashboard or camera credentials to the public internet.

For the complete step-by-step multi-site configuration, source datasets, licenses, model roles, troubleshooting, GPU memory limits, validation rules and cautionary notes, read **`MULTI_CAMERA_DEPLOYMENT_GUIDE.md`**.

> **Not a certified security system.** Trained weights and datasets are downloaded at first run, not bundled. A successful training run or high confidence cannot guarantee accurate live detection. Real independent site-camera evaluation is required; false positives and missed detections remain possible. The YOLO fight model is pretrained only unless actual bounding-box fighting data is supplied. Labeled fight clips instead train the video-action R3D-18 classifier. The original Keras 3DCNN is downloaded but its unknown output-label polarity is not treated as trained/activated automatically.

---

# VisionGuard — False-Positive Hardened Detection + GPU Training (Python 3.12)

One project, based on the **original VisionGuard FastAPI dashboard**, with fixes merged from the user-supplied `setup_all(1).zip` and `DOWNLOAD_SECURITY_MODELS_WINDOWS(1).zip`. The latter ZIPs contained overlapping installers and incompatible detection manager revisions. The stable Python 3.12 version is the foundation here; the buggy practice of feeding annotated video frames back into later detectors is **not** carried forward.

**Included in the ZIP:** Python source, dashboard, Windows launchers, model manifest, data converters, training scripts, video testing, unit/integration tests and docs. **Not included:** model weights, datasets, CUDA runtime wheels, or any fabricated accuracy figures. Those are large and download on your own machine during setup, after you approve community model checkpoint downloads. This project has not undergone real NVIDIA/CCTV validation in the packaging environment.

## 1. Install on Windows 10/11

Requirements: **Python 3.12 64-bit**, Windows 10/11, internet for first install, adequate disk space, and NVIDIA GPU + current NVIDIA driver if GPU acceleration is desired. No CUDA Toolkit is required for precompiled PyTorch wheels. Modern PyTorch CUDA wheels are selected using `nvidia-smi`, then checked by an actual GPU tensor calculation. CPU operation is supported if there is no NVIDIA GPU.

1. Extract the ZIP to a writable directory without spaces that require administrator rights (e.g. `C:\VisionGuard`).
2. Double-click **`ONE_CLICK_SETUP_AND_RUN.bat`**. It builds `.venv` with Python **3.12 x64**, installs PyTorch (CUDA if supported) and application requirements, creates `.env` without overwriting secrets, asks approval for the three community YOLO checkpoints, and starts the dashboard.
3. Open **http://127.0.0.1:8000** and sign in. First-time admin credentials are printed by the initializer and stored locally in `.env`.
4. On later runs double-click **`run.bat`**. Nothing is retrained on ordinary startup.

To verify:

```bat
.venv\Scripts\python.exe verify_runtime.py --require-gpu
.venv\Scripts\python.exe verify_models.py
```

If you don't have NVIDIA CUDA, remove `--require-gpu` for the runtime check. If a detected NVIDIA GPU cannot initialize, setup **fails clearly**; it does not claim GPU acceleration when PyTorch is actually on CPU.

### Security of downloaded weights

Community PyTorch `.pt` model files may contain pickle payloads which can execute Python code when loaded. **Only approve downloads if you trust their publishers.** The installer uses explicit filenames, downloads them to a staging file, loads them locally, checks task/class names and promotes weights only after successful checks. For maximum security use only independently audited weights. No checkpoint was embedded in this ZIP.

## 2. Model inventory: detection vs classification

| Feature | Hugging Face repository / weights | Role |
|---|---|---|
| Fire + smoke | `SalahALHaismawi/yolov26-fire-detection` → `best.pt` | YOLO bounding boxes; class labels must say fire/smoke |
| Fight/violence | `Musawer14/fight_detection_yolov8` → `Yolo_nano_weights.pt` | YOLO bounding boxes; negative labels `normal`/`non-violence` are never converted to fights |
| Guns/knives/threats | `Subh775/Threat-Detection-YOLOv8n` → `weights/best.pt` | YOLO bounding boxes; labels are taken from the model, not guessed from numeric ID |
| Optional source 3D CNN | `Musawer14/fight_detection_3DCNN` → `fight_detection_model.h5` | **Sequence classifier** only; disabled until input normalization and class ordering are independently verified |
| New optional Windows GPU fight classifier | **Locally trained** `models/visionguard-r3d18-fight.pt` | TorchVision R3D-18 binary *whole-video-window* classifier, trained only from explicitly labeled video clips |
| Synthetic knife train data | `syzenzz/cctv-knife-detection-dataset` | **YOLO dataset**, not a checkpoint; 114 synthetic CCTV sample images |
| Knife/pistol train data | `shravya11/weapon-detection` | **Dataset**, not a checkpoint; download only annotated detection subsets (`Knife_detection/`, `Pistol detection/`) |

**Important:** A 3D CNN's probability does not identify where an assault occurred, so its event is reported as *whole-frame classification evidence*, not a fake object box. User-trained R3D-18 uses the labels `normal` (0) and `fight` (1), stored in checkpoint metadata. The downloaded publisher `.h5` is still an **opt-in legacy research model**, requiring separate TensorFlow installation and verified class polarity. Native Windows TensorFlow 2.11+ does not support NVIDIA GPU; the new TorchVision classifier can use PyTorch CUDA on Windows.

**Game-specific CS2 weapon class labels** (AK47, M4A1, etc.) are intentionally **not** treated as proof of a real gun in security footage. The pretrained `Subh775` model uses generic threat labels; optional trained weights are loaded independently. Unrecognized labels result in no incident and a model warning, never a guessed class ID.

## 3. Diagnosis: why does nothing detect on my video?

Use the **same detection manager** the live dashboard uses. Drag an MP4/AVI/MKV file onto **`ANALYZE_VIDEO_WINDOWS.bat`**, or:

```bat
.venv\Scripts\python.exe analyze_video.py --input "C:\videos\test.mp4" --every 1
```

Three files appear next to the input:

- `test_visionguard.mp4` — video with **every source frame** saved; threat boxes are drawn **only after policy and multi-frame confirmation**, never directly from raw YOLO output.
- `test_visionguard.events.jsonl` — one line per analyzed frame with raw hypotheses, accepted candidates, suppressed candidates **with reasons**, and confirmed events.
- `test_visionguard.summary.json` — model names, loaded/missing models, person-context status, raw/accepted/suppressed candidate counts, and confirmed incidents.

For **debugging low-confidence candidates only** (does *not* reduce the alert policy):

```bat
.venv\Scripts\python.exe analyze_video.py --input "C:\videos\test.mp4" --only fire,weapon --diagnostic-conf 0.15
```

A 15% raw YOLO prediction is logged for diagnosis but **is not drawn as an alert box**. The score is not a calibrated chance that an item is present. Default alert eligibility remains a separate **50% raw model score**, plus temporal and spatial confirmation. Results are grouped by source event and location, so a knife appearing at unrelated positions on consecutive frames doesn't become one confirmed incident. Model confidence is not accuracy; adjust thresholds only with verified negative/positive camera examples. Do not assume that `0 incidents` means the camera area is safe.

For the web dashboard, upload a video in **Video testing** and inspect the downloadable annotated MP4, per-model status, and separate raw/confirmed counts. Missing required specialist models are reported instead of pretending a security test passed. Camera feeds may differ in light, resolution, perspective, motion blur, and compression.

## 4. Real weapon detector training (optional)

**Double-click `TRAIN_AND_RUN.bat`.** It verifies NVIDIA CUDA, then asks:

- `[1]` fast functional sample with synthetic CCTV knife images (~114 images; **not production quality**)
- `[2]` merge the synthetic CCTV set with `shravya11`'s *annotated* knife/pistol VOC subsets (larger download; review Shravya's separate license file)

It then downloads raw images/labels to `data/datasets/raw/`, converts valid YOLO/VOC boxes, rejects conflicting duplicate image annotations, keeps source image/label pairs in the same dataset part, excludes image classification-only folders, deduplicates by exact image SHA256, and splits **train and validation image files**. It can also use user-verified object-free negative images. You can extract candidate frames from a **clean, human-reviewed weapon-free video** by dragging it onto `EXTRACT_VERIFIED_NEGATIVES_WINDOWS.bat` (first 30 seconds, one frame every two seconds), or run `collect_verified_negatives.py --input "C:\videos\normal.mp4" --start 0 --end 30 --interval 2 --confirm-no-weapons`. Re-review the extracted stills. **Do not use your annotated screenshots as negatives**: their pink/green boxes contaminate training. Then train with:

```bat
.venv\Scripts\python.exe train_models.py --source both --epochs 35 --base yolov8n --verified-negatives "C:\videos\negative_stills"
```

**Critical fix:** Class ID mappings in the *source dataset* are not the model's class IDs. The trainer creates classes only for objects with actual boxes, then remaps all labels to a contiguous zero-based index. The synthetic sample uses source classes `person` and `knife`, but **person is deliberately excluded** from the weapon checkpoint (a separate COCO person detector handles people). If there are no gun labels, the new model will **not** claim to detect guns. Other pretrained threat weights remain separately loaded for gun detection.

This edition defaults to **Ultralytics `yolov8n.pt`** as the fine-tuning base. Verified hard-negative CCTV backgrounds (walls, window grills, doors, shirt textures) can reduce false positives if properly sampled, but training on these is not a substitute for positive examples. Starting from the 4-class pretrained threat checkpoint for a new 1–3-class dataset would require reinitializing its classification head. Use `--base threat` only if you understand that class mismatch; `--base yolov8s` offers a larger COCO pretrained starting point. The script audits each train/val class, trains with GPU, validates on held-out images, and promotes `best.pt` to `models/visionguard-weapon-trained.pt` only on valid training/validation completion. Existing checkpoints are not replaced. Per-run mAP is saved in `models/training_report.json`. A basic held-out mAP@50 sanity gate (default 0.05) blocks promotion of an obviously non-learning checkpoint; this **does not establish real-world reliability**.

**Validation limitation:** Image-disjoint is not the same as camera/scene-disjoint. Similar frames, backgrounds, and synthetic generation can leak across splits. A separate real CCTV test set with independently annotated positive examples *and actual negative scenes* is needed to measure missed objects, false alarms, precision, recall and per-camera thresholds. There is no guaranteed accuracy improvement from a single fine-tuning run.

Manual commands:

```bat
.venv\Scripts\python.exe prepare_datasets.py --source both
.venv\Scripts\python.exe train_models.py --source both --epochs 35 --skip-download
.venv\Scripts\python.exe verify_models.py
```

To fine-tune **fire or YOLO fight** detection with your own annotated YOLO images and boxes, use `fine_tune_yolo.py --task fire|fight --data "C:\dataset\data.yaml" --epochs 35 --activate`. That is a different task from action video classification.

## 5. Real fight video classification training (optional)

YOLO fight *detection* can operate with the downloaded pretrained model. To add a real Windows GPU *video action classifier*, collect independently labeled **video clips**, not just frames. Create:

```
data/fight_clips/
    normal/
        clip_001.mp4
        clip_002.mp4
        ... (at least 5 different clips)
    fight/
        clip_001.mp4
        clip_002.mp4
        ... (at least 5 different clips)
```

Double-click **`TRAIN_FIGHT_CLASSIFIER_WINDOWS.bat`** or:

```bat
.venv\Scripts\python.exe train_fight_classifier.py --data data\fight_clips --epochs 10 --batch 2 --activate
```

This downloads **TorchVision R3D-18 Kinetics pretrained weights** (first run), fine-tunes its binary classification head on the labeled *clips*, uses disjoint video files for a small holdout check, saves labels/preprocessing in a weights-only-safe PyTorch checkpoint and writes a report. Activation sets `ENABLE_FIGHT_CLASSIFIER=true` in `.env`; **restart** the dashboard to use it. The classifier processes **16 real BGR→RGB frames** per window at 112×112, requires two distinct positive windows above `FIGHT_CLASSIFIER_CONFIDENCE` (default 85%) and then adds a *classification* event, not a YOLO bounding box.

Five videos per class only proves the training pipeline can run; it is **not enough** to claim a capable violence detector. A short clip split from the same long surveillance recording can make a deceptive holdout score. Diverse real negative actions (dancing, hugging, arguing, running, sports, crowded scenes), real positive fights, camera separation, consent and correct labels are essential. The model is deliberately disabled until you train and activate it. This option uses **PyTorch GPU on native Windows** rather than TensorFlow GPU.

### Optional publisher 3D CNN (advanced research only)

```bat
.venv\Scripts\python.exe install_security_models.py --accept-checkpoint-risk --only 3dcnn
.venv\Scripts\python.exe -m pip install -r requirements-3dcnn.txt
```

Inspect the publisher's data pipeline, class order and normalization first. Only then set `ENABLE_3DCNN=true`, `CNN3D_POSITIVE_CLASS=0|1` accurately in `.env`. On native Windows, TensorFlow uses CPU for modern versions; WSL2/Linux is recommended if CUDA is required. Treat this model as experimental until verified.

## 6. Evaluate false alarms and missed events on your own clips

Create `cases.json` with independent positive and negative videos. Use human-reviewed event ground truth, not model-generated labels:

```json
{"cases": [
  {"video": "C:/clips/knife_test.mp4", "expected": ["weapon"]},
  {"video": "C:/clips/quiet_corridor.mp4", "expected": []},
  {"video": "C:/clips/fight_test.mp4", "expected": ["violence"]}
]}
```

Run:

```bat
.venv\Scripts\python.exe evaluate_video_cases.py --manifest cases.json
```

The evaluator runs the shared inference pipeline on each video and writes `visionguard_evaluation/benchmark_report.json` with **video-level** true positives, false positives, false negatives, precision and recall per event, and individual failure cases. These are **measured against your labels** and are not calculated until you supply videos. Video-level metrics do not evaluate bounding-box localization. Use separate source recordings/cameras for testing.

## 7. Why your room screenshots gave false gun/knife/grenade/fire boxes

The uploaded frames show **large magenta gun/knife boxes on a window and wall fixtures**, **grenade boxes overlapping a person's shirt/body**, and a **73% fire hypothesis** over a handheld object. Most person boxes look reasonable. A confidence score is a ranking of what the pretrained model *believes*, not proof of a weapon. The community threat checkpoint appears poorly matched to this room/camera setting. No threshold can make an unsuitable checkpoint accurate.

This edition fixes the **application error** of drawing predictions before confirmation. By default the strict policy:

- Uses independent raw model thresholds gun **0.85**, knife **0.75**, vague weapon/grenade **0.90**; fire **0.80**.
- Rejects weapon boxes covering implausibly large portions of the image and object boxes not near a detected person.
- Requires repeated spatially consistent evidence before drawing boxes, capturing a threat snapshot, or sending an alert.
- Logs **every rejected proposal and reason** in offline video JSONL for diagnostics; it does not increase model scores or claim verified accuracy.
- Keeps a positive match visible while it remains confirmed; no raw threat labels are drawn.

These filters **can miss real weapons**, especially a gun close to the camera, an unattended knife on a table, or an object detected at a lower score. For unattended-object scenarios change `WEAPON_REQUIRE_PERSON=false` in `.env` and calibrate the other gates from your own labeled video. To debug a missing valid object, use `--diagnostic-conf 0.15` and inspect `raw_candidates`, `accepted_candidates`, `suppressed`, and the reasons in JSONL. For camera-specific calibration and hard-negative training, read **`FALSE_POSITIVE_FIX_GUIDE.md`**.

## 8. Running tests and inspecting configuration

```bat
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe verify_models.py
.venv\Scripts\python.exe verify_runtime.py
```

Configuration lives in `.env` generated from `.env.example`. Stop and restart the server after installing a new model or editing `.env`; cameras load model instances on startup. Keep `.env`, RTSP passwords, annotated recordings, training clips, database, and incident images private. Dashboard binds to `127.0.0.1` by default. Do not publicly expose the service without HTTPS, strong authentication and network isolation.

## 9. What was tested, and what is not tested

**Locally tested in the packaging environment:** Python source syntax and automated tests, original API policies, data box/label conversion, class filtering, spatial confirmation, classification tensor preprocessing, deterministic file split, duplicate-label detection, per-file dataset pairing, and writing/reading a complete mock-inference annotated video.

**Not run here:** first-run Windows Python 3.12 installation, real NVIDIA CUDA tensor verification on user hardware, third-party model downloads/loading, live camera RTSP streams, actual fine-tuning jobs, independent real-world benchmark clips. These need to run on the user's Windows GPU machine before claiming model performance. The project is an experimental security analytics aid, **not a certified fire/security alarm**. For emergencies, rely on certified detectors and human review.

**Sources:** Official model/dataset repositories are linked by their Hugging Face IDs above. Review their licenses and security disclosures, especially third-party serialized checkpoints. Earlier historical docs are archived in `README_LEGACY.md`, `ARCHITECTURE.md` and `UPDATE_NOTES.md`; when those disagree, this README describes the current implementation.
