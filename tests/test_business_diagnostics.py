import json
from pathlib import Path
import pytest
from diagnose_deployment import (
    audit_system_environment,
    audit_pytorch_gpu,
    audit_model_weights,
    run_deployment_diagnostics
)


def test_audit_system_environment():
    env = audit_system_environment()
    assert env['status'] in ('READY', 'WARNING')
    assert isinstance(env['is_64bit'], bool)
    assert env['python_version'].count('.') == 2
    assert env['free_disk_gb'] > 0.0


def test_audit_pytorch_gpu():
    gpu = audit_pytorch_gpu()
    assert gpu['status'] in ('READY', 'FAILED')
    assert 'cuda_available' in gpu
    assert 'recommended_device' in gpu


def test_audit_model_weights():
    model_inv = audit_model_weights()
    assert 'status' in model_inv
    assert 'models' in model_inv
    assert 'yolov8n.pt' in model_inv['models']


def test_run_deployment_diagnostics_generates_json():
    report = run_deployment_diagnostics(save_report=True)
    assert report['overall_readiness'] in ('READY', 'WARNING', 'FAILED')
    assert 'environment' in report
    assert 'hardware_acceleration' in report
    assert 'model_inventory' in report

    report_file = Path("data/deployment_readiness.json")
    assert report_file.is_file()
    with open(report_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded['overall_readiness'] == report['overall_readiness']
