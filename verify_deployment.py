"""Readiness report for multi-camera deployment (no training or webcam modifications).

Strict mode requires actual valid weights, CUDA, credentials, capacity and probeable streams.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
from app.config import settings,local_model_path, ROOT
from app.database import Database


def readiness(db_path=None,probe=False,require_gpu=False):
    issues=[]; warnings=[]
    detected=(sys.version_info[:2]==(3,12) and sys.maxsize>2**32)
    if not detected: issues.append('Python 3.12 64-bit runtime required')
    cuda=False;gpu_name=''
    try:
        import torch
        cuda=torch.cuda.is_available()
        if cuda:
            x=torch.ones(4,device='cuda')
            cuda=bool((x*x).sum().item()==4.0)
            gpu_name=torch.cuda.get_device_name(0)
    except Exception as exc:
        warnings.append('PyTorch/CUDA check failed: '+type(exc).__name__)
    if require_gpu and not cuda: issues.append('CUDA tensor execution unavailable')
    required={'person':settings.model_path,'fire':settings.fire_model_path,
              'fight':settings.violence_model_path,'weapon':settings.weapon_model_path}
    present={name:Path(local_model_path(file)).is_file() for name,file in required.items()}
    issues += [f'Missing {name} model: {file}' for name,file in required.items() if not present[name]]
    if not settings.admin_password and not settings.dashboard_password:
        issues.append('ADMIN_PASSWORD missing; refuse site deployment')
    if settings.session_secret.startswith(('CHANGE_THIS','change-this')):
        issues.append('SESSION_SECRET not configured')
    free_gb=shutil.disk_usage(ROOT).free/(1024**3)
    if free_gb<5: warnings.append('Less than 5 GB free storage')
    db=Database(db_path or settings.data_dir/'visionguard.db');db.init()
    cameras=db.list_cameras(safe=False)
    enabled=[c for c in cameras if c['enabled']]
    if not enabled: warnings.append('No enabled camera sources configured')
    if len(enabled)>settings.max_active_cameras:
        issues.append(f'{len(enabled)} cameras enabled but only {settings.max_active_cameras} worker slots. '
                      'Increase MAX_ACTIVE_CAMERAS after capacity testing.')
    probes=[]
    if probe:
        from app.engine import CameraWorker
        for cam in enabled:
            cap=None; success=False; message=''
            try:
                cap=CameraWorker._capture(cam['source'])
                if not cap.isOpened():
                    message='Capture could not open'
                else:
                    success,frame=cap.read()
                    if not success or frame is None:message='Capture opened but no decodable frame'
            except Exception as exc:
                message=type(exc).__name__  # don't expose camera credentials
            finally:
                if cap is not None:cap.release()
            probes.append({'name':cam['name'],'ok':bool(success),'detail':message})
            if not success: issues.append('Camera probe failed: '+cam['name'])
    report={'ready_for_smoke_test':not issues,'gpu_available':cuda,'gpu_name':gpu_name,
            'python':sys.version.split()[0],'models_present':present,
            'enabled_cameras':len(enabled),'max_active_cameras':settings.max_active_cameras,
            'inference_parallelism':settings.inference_parallelism,
            'free_space_gb':round(free_gb,1),'camera_probes':probes,
            'errors':issues,'warnings':warnings,
            'disclaimer':'Smoke-test readiness is NOT detection accuracy verification or production certification.'}
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',action='store_true')
    p.add_argument('--require-gpu',action='store_true')
    p.add_argument('--strict',action='store_true')
    a=p.parse_args()
    result=readiness(probe=a.probe,require_gpu=a.require_gpu)
    print(json.dumps(result,indent=2))
    if a.strict and result['errors']:raise SystemExit(1)
