# Fixing the false gun / knife / grenade / fire annotations

## What your screenshots demonstrate

In your CCTV room, the generic threat model is labeling a window (gun ~67–80%), a sidewall/fixtures (knife ~33–54%) and a person's shirt/torso (grenade ~54–79%). Another frame shows a 73% fire prediction near an object held by a person. The **person** detections are more plausible. These are *model predictions*, not confirmed real objects. Static backgrounds and novel textures can fool a pretrained detector even with high confidence.

## What the code now does

1. All models infer on the original clean source image; no boxes painted by one detector are re-used by another.
2. Each weapon prediction receives a `ThreatPolicy.review(...)` result: explicit label support, raw class confidence, box validity, normalized area, and optional person proximity.
3. Rejected predictions are retained for **diagnostics** (`suppressed` with `suppressed_reason`) but cannot create alerts or be drawn as threats.
4. Accepted predictions still require multi-frame confirmation from `IncidentTracker`, including spatial consistency. **Only** active confirmed predictions are drawn in live video, uploaded video tests, and `analyze_video.py` output.
5. Fire predictions under the stricter `FIRE_ALERT_MIN_SCORE=.80` stay unconfirmed candidates; small flames require three repeated frames. This reduces single-frame fire alarms but does not prove that a high-score prediction is actually fire.
6. The 3D fight classifier is no longer recreated every frame and the optional legacy temporal classifier is no longer reset after every frame.

## Tune with real positive AND negative clips

`.env` defaults (change one setting at a time and restart):

```ini
STRICT_WEAPON_POLICY=true
WEAPON_REQUIRE_PERSON=true
WEAPON_GUN_MIN_SCORE=0.85
WEAPON_KNIFE_MIN_SCORE=0.75
WEAPON_OTHER_MIN_SCORE=0.90
WEAPON_GUN_MAX_AREA=0.15
WEAPON_KNIFE_MAX_AREA=0.10
WEAPON_OTHER_MAX_AREA=0.12
FIRE_ALERT_MIN_SCORE=0.80
SMALL_FLAME_ALERT_MIN_SCORE=0.75
INCIDENT_CONFIRMATION_FRAMES=2
```

If cameras monitor a **knife on a table**, set `WEAPON_REQUIRE_PERSON=false` (otherwise that object may be suppressed). The large-box limits can also suppress a real gun in a close-up. Do not rely on these defaults for safety-critical automatic responses. Underlying model quality and independent measurements are essential.

## 1. Run a complete original video without overlays

```bat
ANALYZE_VIDEO_WINDOWS.bat "C:\videos\room_original.mp4"
```

Read the generated `room_original_visionguard.events.jsonl`. Each analyzed frame has `raw_candidates`, `accepted_candidates`, `suppressed` (with rejection reason), and `confirmed`. Video boxes mean *temporal/model confirmation*, not human-verified truth. The summary shows suppression counts.

For detailed candidate investigation only:

```bat
.venv\Scripts\python.exe analyze_video.py --input "C:\videos\room_original.mp4" --only fire,weapon --diagnostic-conf 0.15
```

If `person_context_available=false`, make sure the official base YOLO person checkpoint is installed. A missing person detector in strict mode can suppress even legitimate weapons.

## 2. Extract true hard negatives for retraining

Use an **unannotated** source video with no weapons in the specified time segment. Never use pink-box screenshots; they are model overlays rather than clean images. Watch the entire video and confirm no guns, knives, grenades, or explosives are present. Then:

```bat
.venv\Scripts\python.exe collect_verified_negatives.py --input "C:\videos\normal_room.mp4" --start 0 --end 30 --interval 2 --confirm-no-weapons
```

This extracts 15 stills into `data\verified_negatives\<clip-name>\`. Inspect all exported images. If you find a genuine weapon or ambiguous object, remove that image. Keep private footage private. Use diverse normal scenes, lighting and camera angles; thousands of nearly identical frames are not independent training examples.

Then:

```bat
.venv\Scripts\python.exe train_models.py --source both --epochs 35 --base yolov8n --verified-negatives data\verified_negatives
```

Or double-click `TRAIN_AND_RUN.bat` and approve inclusion of negatives only after verifying the images. Training adds **empty YOLO label files** for confirmed negatives. Real positive boxes from licensed datasets are still required. The weapon fine-tuner intentionally trains **knife/gun only**, never the redundant `person` class, so weapon-free frames containing people can be proper negatives. Review the dataset split for near-duplicate scene leakage; do not treat holdout mAP as a real-camera security benchmark.

## 3. Measure instead of guessing

Create separate human-reviewed clips: clear knife/gun positives, weak small-object positives, empty room negatives, people in patterned clothes, windows/grills, TV-like screens, boxes/phones/lighters. Hold out whole cameras or recording sessions. Populate `cases.json` and run `evaluate_video_cases.py` as explained in `README.md`. Track both **false alarm rate** and **missed genuine threats**, not only a single accuracy number.

## What is not proven

The code was tested with **synthetic model outputs** matching the screenshot label/box patterns, without loading third-party weights. No claim is made that false alarms are eliminated or real-world weapon/fire recall improves. A proper evaluation requires your original videos and verified ground-truth annotations. Keep certified fire/security systems independent of this experimental application.
