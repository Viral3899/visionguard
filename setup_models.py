#!/usr/bin/env python3
"""Automated model downloader and verifier for VisionGuard."""
import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / 'models'

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    'yolov8n.pt': {
        'type': 'ultralytics',
        'source': 'yolov8n.pt',
        'target': 'models/yolov8n.pt',
        'min_bytes': 5_000_000,
        'desc': 'Core Object Detection (80 COCO classes)',
    },
    'yolov8n-pose.pt': {
        'type': 'ultralytics',
        'source': 'yolov8n-pose.pt',
        'target': 'models/yolov8n-pose.pt',
        'min_bytes': 5_000_000,
        'desc': 'Human Pose & Fall Detection (17 keypoints)',
    },
    'yolov26-fire-smoke.pt': {
        'type': 'huggingface',
        'repo': 'SalahALHaismawi/yolov26-fire-detection',
        'filename': 'best.pt',
        'target': 'models/yolov26-fire-smoke.pt',
        'min_bytes': 10_000_000,
        'desc': 'Fire & Smoke Detection',
    },
    'threat-yolov8n.pt': {
        'type': 'huggingface',
        'repo': 'Subh775/Threat-Detection-YOLOv8n',
        'filename': 'weights/best.pt',
        'target': 'models/threat-yolov8n.pt',
        'min_bytes': 5_000_000,
        'desc': 'Specialist Weapon & Threat Detection',
    },
    'fight-yolov8-nano.pt': {
        'type': 'huggingface',
        'repo': 'Musawer14/fight_detection_yolov8',
        'filename': 'Yolo_nano_weights.pt',
        'target': 'models/fight-yolov8-nano.pt',
        'min_bytes': 5_000_000,
        'desc': 'Violence & Fight Detection',
    },
}


def check_model_file(path: Path) -> bool:
    """Verify that a model file exists, is non-empty, and loads properly."""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        if p.stat().st_size < 100_000:
            return False
        # Quick check using ultralytics or torch
        from ultralytics import YOLO
        model = YOLO(str(p))
        return bool(model and hasattr(model, 'names'))
    except Exception:
        try:
            import torch
            checkpoint = torch.load(str(p), map_location='cpu', weights_only=False)
            return isinstance(checkpoint, dict) and ('model' in checkpoint or 'state_dict' in checkpoint)
        except Exception:
            return False


def install_spec(spec_key: str, spec: dict, models_dir: Path, force: bool = False) -> bool:
    """Download and verify a single model specification."""
    models_dir.mkdir(parents=True, exist_ok=True)
    target_path = models_dir / Path(spec['target']).name
    root_candidate = ROOT / Path(spec['target']).name

    # Check if already present and valid
    if not force:
        if check_model_file(target_path):
            print(f"  [CACHED] {spec_key:<24} ({spec['desc']})")
            return True
        if check_model_file(root_candidate):
            shutil.copyfile(root_candidate, target_path)
            print(f"  [COPIED] {spec_key:<24} (from project root)")
            return True

    print(f"  [DOWNLOADING] {spec_key} ({spec['desc']})...")
    spec_type = spec.get('type')
    staging_file = models_dir / f".staging-{spec_key}"

    try:
        if spec_type == 'ultralytics':
            from ultralytics import YOLO
            # Downloading via Ultralytics creates the file in the working directory
            temp_model = YOLO(spec['source'])
            downloaded = Path(temp_model.ckpt_path if hasattr(temp_model, 'ckpt_path') else spec['source'])
            if downloaded.is_file():
                shutil.copyfile(downloaded, staging_file)
            else:
                raise FileNotFoundError(f"Ultralytics failed to download {spec['source']}")

        elif spec_type == 'huggingface':
            from huggingface_hub import hf_hub_download
            hf_cache = hf_hub_download(repo_id=spec['repo'], filename=spec['filename'])
            shutil.copyfile(hf_cache, staging_file)

        else:
            raise ValueError(f"Unknown model type: {spec_type}")

        # Validate downloaded file before promoting
        if not check_model_file(staging_file):
            raise RuntimeError(f"Integrity check failed for {spec_key}")

        # Atomically promote file
        staging_file.replace(target_path)

        # Also copy to root if it is core yolov8n or yolov8n-pose
        if spec_key in ('yolov8n.pt', 'yolov8n-pose.pt'):
            shutil.copyfile(target_path, ROOT / spec_key)

        print(f"  [READY]   {spec_key:<24} ({target_path.stat().st_size // 1024 // 1024} MB)")
        return True

    except Exception as exc:
        print(f"  [FAILED]  {spec_key}: {exc}", file=sys.stderr)
        if staging_file.is_file():
            staging_file.unlink(missing_ok=True)
        return False


def verify_and_download_all_models(models_dir: Path, force: bool = False, check_only: bool = False) -> dict[str, bool]:
    """Verify or download all required models."""
    results = {}
    print("\n" + "=" * 65)
    print(" VisionGuard AI Model Verification & Provisioning")
    print("=" * 65)

    for spec_key, spec in MODEL_SPECS.items():
        target_path = models_dir / Path(spec['target']).name
        root_path = ROOT / Path(spec['target']).name
        is_ok = check_model_file(target_path) or check_model_file(root_path)

        if check_only:
            status = "[OK]" if is_ok else "[MISSING]"
            print(f"  {status:<10} {spec_key:<24} ({spec['desc']})")
            results[spec_key] = is_ok
        else:
            ok = install_spec(spec_key, spec, models_dir, force=force)
            results[spec_key] = ok

    print("=" * 65)
    return results


def main():
    parser = argparse.ArgumentParser(description="Download and verify VisionGuard AI models.")
    parser.add_argument("--force", action="store_true", help="Force redownload even if models exist.")
    parser.add_argument("--check-only", action="store_true", help="Only verify existing models without downloading.")
    args = parser.parse_args()

    results = verify_and_download_all_models(MODELS_DIR, force=args.force, check_only=args.check_only)
    failed = [k for k, ok in results.items() if not ok]
    if failed:
        print(f"\n[!] WARNING: {len(failed)} model(s) could not be verified: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\n[✓] All required models are verified and ready for live inference!\n")
        sys.exit(0)


if __name__ == '__main__':
    main()
