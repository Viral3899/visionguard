# VisionGuard site deployment — Python 3.12 + NVIDIA GPU

## Fast start (Windows)

1. Install **Python 3.12 x64** and update your NVIDIA driver (`nvidia-smi` must work). Extract the ZIP to a short local path with **at least 15–25 GB free** if downloading full datasets; high-resolution data and training runs may need more.
2. Run `ONE_CLICK_SETUP_TRAIN_DEPLOY.bat`. It first installs the Python environment and known model weights. After reviewing repository licenses and the risks of untrusted `.pt` checkpoints, it downloads labeled training data and executes up to **three distinct training tasks**. It starts inference afterward. Re-running the installer is supported but training reruns unless you use the inference-only launcher.
3. For camera setup, run `CONFIGURE_CAMERAS_WINDOWS.bat`: change `cameras.json`, put RTSP credentials in `.env`, then import. You can alternatively add cameras via dashboard. Example `.env` entries:

   ```dotenv
   CAMERA_GATE_RTSP=rtsp://operator:password@192.168.1.100:554/Streaming/Channels/101
   CAMERA_WAREHOUSE_RTSP=rtsp://operator:password@192.168.1.101:554/Streaming/Channels/101
   MAX_ACTIVE_CAMERAS=4
   INFERENCE_PARALLELISM=1
   FRAME_SKIP=2
   WEAPON_EVERY_N_FRAMES=3
   OBJECT_EVERY_N_FRAMES=5
   ```

   These paths are generic examples; Hikvision, Dahua, ONVIF and other NVRs vary. Confirm the real RTSP path in VLC first. Use a lower-resolution **substream** for AI if available. Multiple simultaneous streams need enough camera/NVR RTSP sessions, bandwidth, RAM and GPU VRAM. Camera feeds must be authorized. Never post URLs/passwords online.

4. Run `VERIFY_MULTI_CAMERA.bat` to test CUDA, model files, disk, credentials, camera count and a frame from each configured stream. Then run `INFERENCE_MULTI_CAMERA.bat` for the normal day-to-day operation; **no training is done in this script**.
5. Open the dashboard at `http://127.0.0.1:8000` from the local machine. For access on a trusted LAN, change `HOST=0.0.0.0` and set strong `ADMIN_PASSWORD` and `SESSION_SECRET`. Protect port 8000 behind a VPN or HTTPS reverse proxy and a firewall; **do not expose it directly to the public internet**. If accessed across a LAN, use `http://<server-lan-ip>:8000` only for trusted internal traffic. Prefer TLS even on LAN.
6. Optional `REGISTER_AUTOSTART_WINDOWS.bat` creates an on-logon Windows Scheduled Task. Verify its working directory, Windows session, environment, restarting-on-failure, and USB camera session behavior before relying on it unattended.

## Camera scaling

* `MAX_ACTIVE_CAMERAS` defaults to **4**. Additional cameras can be configured in the dashboard, but report `capacity_limit` until worker slots are increased. A 4-camera default is a safety resource limit, **not a certified throughput**.
* `SHARE_SPECIALIST_MODELS=true` shares trained model objects among the camera workers, while temporal incident state remains per camera. The person tracking model remains per camera to avoid mixing track IDs across feeds. `INFERENCE_PARALLELISM=1` serializes GPU-heavy calls to reduce concurrency spikes. Bounded inference trades latency against GPU contention; measure actual latency and FPS.
* CPU/network load increases with frame size, FPS, and inference frequency. Start with 2 cameras at 640–960px, measure GPU VRAM using `nvidia-smi`, then increase. For many dozens of cameras, deploy **multiple inference nodes** or a dedicated batched inference server; this single Windows process is a site pilot, not a horizontally scaled orchestration platform.
* The worker automatically retries streams after read errors. Camera/NVR network connectivity, RTSP credentials, and camera clock synchronization still need correct site setup.
* `models/` and `data/` are local. Back up configuration and the database; reserve storage for snapshots and logs. Your organization must define retention, consent, and privacy policies.

## Training and activation

**One-click automation does not invent labels.**

| Task | Online dataset | Training output | How inference uses it |
|---|---|---|---|
| Weapon bounding boxes | `syzenzz/cctv-knife-detection-dataset` (114 synthetic images) + `shravya11/weapon-detection` knife/pistol detection XML | `models/visionguard-weapon-trained.pt` | Opt-in `ENABLE_TRAINED_WEAPON_MODEL=true` **after** independent CCTV evaluation |
| Fire/smoke bounding boxes | `hanvithSai/gavin-dfire-raw`, original D-Fire ~21k annotated images, CC BY 4.0 | `models/visionguard-fire-trained.pt` | Opt-in `FIRE_MODEL_PATH=models/visionguard-fire-trained.pt` **after** independent CCTV evaluation |
| Fight video classification | `OmarNabill/event-detection`, source-labeled `Fighting` and `Other` videos | `models/visionguard-r3d18-fight.pt` | Opt-in `ENABLE_FIGHT_CLASSIFIER=true` **after** labeled camera evaluation |

The project also downloads the separately pretrained **YOLOv26 fire**, **YOLOv8 fight**, **YOLOv8n threat**, **COCO person**, and **pose** weights, plus the small optional **Musawer14 3DCNN Keras `.h5`** checkpoint (stored as `downloaded_unverified_classes`, not claimed deployable). The YOLO fight detector is **not** automatically retrained: video-level class labels cannot be used as frame-level bounding boxes. The optional unknown-label TensorFlow 3DCNN is **not activated** by default and is not silently trained from the video classes. Modern TensorFlow NVIDIA GPU on native Windows is not supported; this project uses PyTorch R3D for the trained temporal classifier.

Run `TRAIN_ALL_MODELS.bat` without restarting the dashboard, or call:

```powershell
.venv\Scripts\python.exe setup_train_all.py --accept-checkpoint-risk --accept-data-licenses --fight-license-reviewed
```

Use `--fire-max-images 2000 --fight-max-per-class 40` for a smaller first **development** run; defaults use all available labeled data. For manual steps: `prepare_fire_dataset.py`, `train_fire.py`, `prepare_datasets.py`, `train_models.py`, `download_fight_dataset.py`, `train_fight_classifier.py`.

Training results are saved in `models/*report.json`, `data/training_pipeline_status.json`, and `runs/`. Failed stages remain explicitly **failed**; the app does not claim them as trained.

**Site validation is required** before switching to fine-tuned weights. Use separate real site videos containing both actual fire/weapon/fight incidents **and** normal no-threat footage. Label genuine positives/negatives and measure missed detections, false alerts per camera-hour, and processing delay. Use `ANALYZE_VIDEO_WINDOWS.bat` to save annotated MP4 and `.events.jsonl`. After validation, run `ACTIVATE_TRAINED_MODELS_AFTER_REVIEW.bat` to apply gates and opt-in; restart the server afterward. Its built-in model-metric gates alone cannot prove real-world suitability.

## Key safety caveats

* High YOLO confidence does **not** guarantee a correct object. The previously reported window/gun and shirt/grenade errors require real negative examples from your site and testing on separate clips.
* Strict weapon filtering can **miss unattended weapons** because `WEAPON_REQUIRE_PERSON=true` by default; calibrate to use case.
* Training many images will not fix wrong annotation IDs, incorrect labels, duplicate scenes, CCTV domain shift or weak target negatives. Use class-specific quality checks and a truly independent camera test set.
* A poor model can generate both missed security events and false alerts. VisionGuard is a decision-support system, **not** a substitute for trained operators, fire alarms, emergency response systems or other safeguards.

See `README.md` and `FALSE_POSITIVE_FIX_GUIDE.md` for further tuning, data preparation and troubleshooting.
