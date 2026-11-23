# VisionGuard | Architecture and operations

## Reading the reported fire test

The client uploaded 2,536,151 bytes. OpenCV read 375 frames in 15 seconds (25 fps), and the upload loop selected 187. With the original scheduling, the fire detector was called every second selected frame, giving it 93 opportunities. The earlier response counted selected frames even when the model failed. It cannot establish whether the detector recognized the footage.

## Windows upgrade and first run

1. Stop the server. Back up the entire existing project, including data, models and .env, while it is stopped. Extract this release into a new folder. Copy your existing data and trusted model weights there. Keep the old installation for rollback.
2. On a new installation, run install_windows.bat with Python 3.12 x64 available. Compare your existing settings to .env.example. Set ADMIN_PASSWORD to a unique password. Generate SESSION_SECRET with the command below. Keep HOST=127.0.0.1 for local use. Changing the secret invalidates existing sessions.
3. The release includes a fire-and-smoke checkpoint at models/yolov8n-fire.pt. Preserve that file when migrating other weights. Set FIRE_MODEL_PATH and FIRE_CLASSES to match its actual path and class IDs. An empty FIRE_CLASSES enables all classes. See models/FIRE-MODEL.md for provenance and checksum. Set MODEL_PATH separately for person tracking. Configure the other detector paths for the capabilities you intend to use.
4. From the project directory in UCRT64 run the following commands, then open the dashboard at http://127.0.0.1:8000 and sign in. Use its video test form for authenticated uploads.

```bash
source .venv/Scripts/activate
python -c "import secrets; print(secrets.token_hex(32))"
python check_fire.py
python -m app.main
```

The preflight performs a single inference on a blank frame. Exit code 0 requires successful inference and an enabled fire or flame label. A smoke-only model can evaluate smoke but fails the flame preflight. A successful check establishes model compatibility. Accuracy requires labeled footage.

| Upload fire_detector.status | Interpretation and operator action |
|---|---|
| model_unavailable | Inspect server logs for the loading exception. Check the path, checkpoint and runtime. |
| unsupported_labels | Compare enabled IDs with checkpoint metadata. Use trained weights with named fire, flame or smoke classes. Numeric labels need corrected metadata from the model supplier. |
| not_sampled | No frame reached the fire inference schedule. Use a longer clip or adjust FRAME_SKIP. |
| inference_failed / partial_failure | Inspect successful_frames and failed_frames. Check runtime logs and reproduce on CPU to isolate device issues. |
| ready | Read enabled_labels and confidence_threshold. If events is empty, review annotations against labeled positive and negative clips before changing thresholds. |

## HLD — single-host deployment

| Component | Responsibility and boundary |
|---|---|
| Browser and FastAPI | Dashboard, session cookies, client-scoped API, MJPEG previews and WebSocket events. |
| CameraManager and CameraWorker | One worker per enabled source. Capture, reconnect, track people and run scheduled detectors. CPU/GPU and disk bandwidth limit capacity. |
| SQLite and local files | Store camera configuration, client accounts, events, counts and alert records. Snapshots and annotated clips occupy local disk. |
| AlertManager | Applies channel policy and invokes local speech or optional Telegram delivery. Validate external delivery separately. |
| Clip analysis | Reads an uploaded file in a worker thread, instantiates its own models and produces an annotated MP4. One upload analysis at a time per process. |

Deploy one application process per data directory. Multiple workers would independently start camera threads, allocate models and emit alerts. For remote operation use authenticated HTTPS through a reverse proxy with WebSocket forwarding, explicit request size and time limits, and restricted access to the backend. Configure authentication even with a loopback backend behind a proxy.

## LLD — runtime paths

**Live camera.** app/engine.py opens the configured source, samples frames, scales large images, runs person tracking, and passes normalized foot positions to app/analytics.py for zone and directed-line crossings. app/detectors.py schedules fire, violence, anomaly and general object inference every 2, 4, 5 and 8 processed frames. The camera worker applies a 0.50 scored-event floor. It publishes a JPEG preview and event updates. On read failure it enters reconnecting state.

**Fire classification.** FireDetector checks the checkpoint's enabled class names, preserves their IDs and recognizes fire and flame as flame evidence. The standard confirmed path uses recent smoke or at least two fire observations within a four-second monotonic-clock window. Small-flame and smoke candidates have separate rules. Uploaded fire confirmation uses decoded frame index divided by the source frame rate. Live cameras use monotonic capture-processing time. Variable-frame-rate uploads approximate presentation time using average FPS. Fire-only and smoke-only checkpoints have different coverage.

**Upload contract.** POST /api/test-videos accepts multipart field file, up to 250 MiB of copied media and at most 12,000 decoded frames. The response reports read, processed and successful fire-inference counts. A 0.30 event filter operates after individual model thresholds. Two distinct processed frames containing the same kind are required for an upload alarm. The annotated download uses a random token and an owner-scoped filename. The server removes input files after processing. Daily retention includes completed annotated clips. A busy analyser returns HTTP 429. Multipart parsing happens before this application limit, so enforce a body limit at ingress as well.

**Persistence and access.** app/database.py uses SQLite WAL, per-operation connections and transactions. Cameras link to client accounts, camera_config and totals. Events reference cameras and optional snapshot files. Alert logs record delivery outcomes. app/main.py signs expiring session cookies and restricts client views to assigned resources. Legacy annotated files stored before owner scoping are available to administrators. A generated session key lasts until the process restarts if no private key was configured.

## Production acceptance and rollback

| Check on target hardware | Evidence to retain |
|---|---|
| Model qualification | Model source, hash, class mapping, device and thresholds. Run preflight, then representative day/night fire, smoke and negative clips. Record missed incidents, false alarms per camera-hour and time to first alert. Agree on acceptable limits with the operator before enabling notifications. |
| Full-load operation | At the intended camera count, measure FPS, end-to-end delay, GPU memory and disk growth during a concurrent upload. Disconnect and reconnect a camera, then restart the server. Verify previews and event delivery recover. |
| Isolation and channels | Test two client logins, assigned cameras, alert stream and downloads. Use designated test recipients to verify Telegram and check the actual Windows audio output. |
| Backup and recovery | Stop the process, copy data, configuration and weights, then restore into a separate directory. Verify users, assignments, event history and snapshot retrieval. |
| Release environment | Install requirements-dev.txt and run python -m pytest -q. Record Python and package versions, GPU driver and checkpoint hashes for the accepted build. The supplied requirements specify ranges. |

For rollback, stop the new process and start the saved application with its matching pre-upgrade data and configuration. Preserve any events captured after upgrade separately. The health endpoint reports process and camera status. Use the model check and labeled footage to qualify detection independently.

## Verification of this package

The automated suite exercises API and authentication, analytics, the camera worker with a substitute tracker, alert policy, dashboard/WebSocket delivery, fire-label handling, and an encoded upload through annotated MP4 download with a missing model. Windows CUDA, site-specific checkpoints, RTSP cameras, speaker and Telegram delivery require tests on the deployment host.

### CPU integration results with bundled weights

The normal upload endpoint was exercised with Ultralytics 8.4.215, PyTorch 2.8.0+cpu, a 0.50 fire confidence threshold and FRAME_SKIP=2. Each input contained 24 frames at 12 fps. The two positive inputs repeated individual public photographs in MP4 containers. The control contained black frames.

| Input | Successful fire evaluations | Confirmed alerts | Annotated MP4 download |
|---|---:|---|---|
| Burning stove | 6 of 6 | fire_detected | HTTP 200 |
| Outdoor bonfire | 6 of 6 | fire_detected | HTTP 200 |
| Black frames | 6 of 6 | None | HTTP 200 |

The two positive clips each produced one fire_watch and five fire_detected candidates. The endpoint loaded the person, object, pose and fire models. The optional violence checkpoint was absent and reported as unavailable. These trials exercise decoding, neural inference, confirmation, encoding and retrieval. Motion, smoke-only performance, operating thresholds at the deployment site and live alert delivery require separate qualification.

Test image sources: [Burning stove fire](https://commons.wikimedia.org/wiki/File:Burning_stove_fire_(55026908190).jpg) and [Large bonfire](https://commons.wikimedia.org/wiki/File:Large_bonfire.jpg).


## Multi-camera deployment extension (November 2026)

* `import_cameras.py` imports validated multi-camera JSON without deleting historical records. Each camera retains its own intrusion zone, tripwire, event tracker and client assignment.
* `app/engine.py` enforces a configurable max worker count and bounded inference slots. Model-sharing in `app/detectors.py` reuses specialist networks behind a lock; the person tracker remains per camera because persistent track state cannot be shared across streams.
* `inference.py` is the dedicated entry point: `--mode cameras` for the FastAPI dashboard or `--mode video` for annotated MP4 evidence. Neither path trains models.
* `setup_train_all.py` manages independent pipelines and writes machine-readable success/failure records. `prepare_datasets.py` handles YOLO/VOC weapon boxes, `prepare_fire_dataset.py` handles D-Fire YOLO images with separate test split, and `download_fight_dataset.py` preserves supplied whole-video labels and predefined train/val/test splits.
* `train_models.py` fine-tunes a weapon YOLO detector; `train_fire.py` fine-tunes a fire/smoke YOLO detector; `train_fight_classifier.py` trains TorchVision R3D-18 clip classifier. Unknown-label 3DCNN checkpoints remain opt-in and cannot be treated as trained classifiers with known class polarity.
* `activate_models.py` requires explicit site evaluation approval and checks held-out metric gates before enabling trained weights, but **cannot itself verify** the site's measured false-positive/false-negative rates.
* Horizontal scaling across several physical GPU servers is not provided; configure several independent sites/nodes when one server cannot serve all cameras.
