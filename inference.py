"""Standalone deployment/inference entry point; NO downloading or training.

Usage: py -3.12 inference.py --mode cameras [--config cameras.json]
       py -3.12 inference.py --mode video --input test.mp4
"""
import argparse
import os
from pathlib import Path

from app.config import ROOT


def run(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['cameras','video'], default='cameras')
    p.add_argument('--config',type=Path,default=ROOT/'cameras.json')
    p.add_argument('--input',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--every',type=int,default=1)
    p.add_argument('--allow-missing-models',action='store_true')
    args=p.parse_args(argv)
    if args.mode=='video':
        if args.input is None or not args.input.is_file():
            p.error('--input is required and must exist for video mode')
        from analyze_video import analyze
        return analyze(args.input,args.output,'fire,fight,weapon,anomaly,object',args.every,None,0,'auto')
    # Missing models are a deployment warning, NOT a reason to silently say they are ready.
    if args.config.is_file():
        from import_cameras import import_cameras
        import_cameras(args.config,update_existing=False)
    if not args.allow_missing_models:
        from app.config import settings, local_model_path
        required=('model_path','fire_model_path','violence_model_path','weapon_model_path')
        missing=[key for key in required if not Path(local_model_path(getattr(settings,key))).is_file()]
        if missing:
            raise SystemExit(f'Not starting: missing required model weights {missing}. Run installer/download models first. '
                             'Use --allow-missing-models only for diagnostics.')
    import uvicorn
    from app.config import settings
    uvicorn.run('app.main:app',host=settings.host,port=settings.port,workers=1,reload=False)


if __name__=='__main__':
    run()
