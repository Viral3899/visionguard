"""VisionGuard Automated Deployment Readiness & Hardware Diagnostic.

Eliminates high IT deployment and maintenance costs by automatically auditing
environment dependencies, GPU acceleration, AI model weights, live tensor throughput,
and storage headroom.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("DeploymentDiagnostic")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

REQUIRED_MODELS = [
    ("yolov8n.pt", 6_000_000),           # Primary general detector (~6.2 MB)
    ("yolov8n-pose.pt", 6_000_000),      # Pose estimation (~6.5 MB)
]

OPTIONAL_MODELS = [
    ("yolov8n-knife.pt", 5_000_000),     # Weapon detection
    ("yolov8s-fire-smoke.pt", 15_000_000), # Fire & hazard detection
]


def audit_system_environment() -> dict[str, Any]:
    """Check operating system, Python architecture, and memory."""
    is_64bit = sys.maxsize > 2**32
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    valid_py = sys.version_info >= (3, 10)

    disk_usage = shutil.disk_usage(BASE_DIR)
    free_gb = round(disk_usage.free / (1024**3), 2)
    total_gb = round(disk_usage.total / (1024**3), 2)

    status = "READY" if (is_64bit and valid_py and free_gb >= 1.0) else "WARNING"

    return {
        "status": status,
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": py_ver,
        "is_64bit": is_64bit,
        "free_disk_gb": free_gb,
        "total_disk_gb": total_gb,
        "in_venv": sys.prefix != sys.base_prefix
    }


def audit_pytorch_gpu() -> dict[str, Any]:
    """Audit PyTorch, CUDA runtime, and GPU acceleration capabilities."""
    try:
        import torch
        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"
        vram_gb = 0.0
        if cuda_avail:
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            device_target = "cuda:0"
        else:
            device_target = "cpu"

        return {
            "status": "READY",
            "torch_version": torch_ver,
            "cuda_available": cuda_avail,
            "gpu_device": gpu_name,
            "vram_gb": vram_gb,
            "recommended_device": device_target
        }
    except ImportError:
        return {
            "status": "FAILED",
            "torch_version": None,
            "cuda_available": False,
            "gpu_device": "None",
            "vram_gb": 0.0,
            "recommended_device": "cpu",
            "error": "PyTorch is not installed"
        }


def audit_model_weights() -> dict[str, Any]:
    """Verify presence and file sizes of all AI model checkpoints."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models_status = {}
    all_required_present = True

    for filename, min_bytes in REQUIRED_MODELS:
        path = MODELS_DIR / filename
        if path.is_file() and path.stat().st_size >= min_bytes:
            models_status[filename] = {"present": True, "size_mb": round(path.stat().st_size / (1024**2), 2), "required": True}
        else:
            models_status[filename] = {"present": False, "size_mb": 0.0, "required": True}
            all_required_present = False

    for filename, min_bytes in OPTIONAL_MODELS:
        path = MODELS_DIR / filename
        if path.is_file() and path.stat().st_size >= min_bytes:
            models_status[filename] = {"present": True, "size_mb": round(path.stat().st_size / (1024**2), 2), "required": False}
        else:
            models_status[filename] = {"present": False, "size_mb": 0.0, "required": False}

    return {
        "status": "READY" if all_required_present else "WARNING",
        "models": models_status,
        "all_required_present": all_required_present
    }


def benchmark_inference_throughput(num_warmup: int = 2, num_iters: int = 5) -> dict[str, Any]:
    """Measure inference latency and FPS on dummy frames to ensure smooth surveillance."""
    import numpy as np
    try:
        from ultralytics import YOLO
        model_file = MODELS_DIR / "yolov8n.pt"
        if not model_file.exists():
            return {"status": "SKIPPED", "reason": "yolov8n.pt not found for benchmark"}

        model = YOLO(str(model_file))
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Warmup
        for _ in range(num_warmup):
            model(dummy_frame, verbose=False, imgsz=480)

        # Timed benchmark
        t0 = time.perf_counter()
        for _ in range(num_iters):
            model(dummy_frame, verbose=False, imgsz=480)
        t1 = time.perf_counter()

        elapsed = max(0.001, t1 - t0)
        fps = round(num_iters / elapsed, 1)
        latency_ms = round((elapsed / num_iters) * 1000.0, 1)

        return {
            "status": "READY",
            "benchmark_fps": fps,
            "average_latency_ms": latency_ms,
            "iterations": num_iters
        }
    except Exception as exc:
        return {
            "status": "WARNING",
            "error": str(exc),
            "benchmark_fps": 0.0,
            "average_latency_ms": 0.0
        }


def run_deployment_diagnostics(save_report: bool = True) -> dict[str, Any]:
    """Execute complete diagnostic evaluation and save readiness report."""
    LOGGER.info("Starting VisionGuard Automated Deployment Readiness Diagnostic...")

    env_audit = audit_system_environment()
    gpu_audit = audit_pytorch_gpu()
    model_audit = audit_model_weights()
    perf_audit = benchmark_inference_throughput()

    overall_status = "READY"
    issues = []

    if env_audit["status"] != "READY":
        overall_status = "WARNING"
        issues.append("Operating environment or disk space warning")

    if gpu_audit["status"] == "FAILED":
        overall_status = "FAILED"
        issues.append("PyTorch runtime is not installed")
    elif not gpu_audit.get("cuda_available"):
        issues.append("GPU CUDA not detected; running in CPU inference mode")

    if not model_audit.get("all_required_present"):
        overall_status = "WARNING"
        issues.append("Some required model weights are missing; run setup_models.py")

    report = {
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_readiness": overall_status,
        "issues": issues,
        "environment": env_audit,
        "hardware_acceleration": gpu_audit,
        "model_inventory": model_audit,
        "performance_benchmark": perf_audit
    }

    if save_report:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        report_path = DATA_DIR / "deployment_readiness.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        LOGGER.info(f"Deployment readiness report saved to: {report_path}")

    return report


def main():
    report = run_deployment_diagnostics(save_report=True)
    print("\n" + "=" * 65)
    print("      VisionGuard Deployment Readiness & Diagnostic Report")
    print("=" * 65)
    print(f" Status:              [{report['overall_readiness']}]")
    print(f" Operating System:    {report['environment']['os']}")
    print(f" Python Environment:  {report['environment']['python_version']} (64-bit: {report['environment']['is_64bit']})")
    print(f" Hardware Engine:     {report['hardware_acceleration']['gpu_device']}")
    if report['hardware_acceleration']['cuda_available']:
        print(f" VRAM Headroom:       {report['hardware_acceleration']['vram_gb']} GB")
    print(f" Free Disk Space:     {report['environment']['free_disk_gb']} GB")
    if report['performance_benchmark'].get('benchmark_fps'):
        print(f" Inference Latency:   {report['performance_benchmark']['average_latency_ms']} ms ({report['performance_benchmark']['benchmark_fps']} FPS)")
    print("-" * 65)
    if report['issues']:
        print(" Actionable Advisories:")
        for issue in report['issues']:
            print(f"  * {issue}")
    else:
        print(" [OK] All enterprise systems and AI pipelines certified ready.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
