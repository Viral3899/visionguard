#!/bin/bash
cd "$(dirname "$0")"
echo "============================================"
echo "VisionGuard AI Surveillance - Starting"
echo "============================================"
echo

if [ ! -f .venv/bin/python ]; then
  echo "Virtual environment not found."
  echo "Run python setup_all.py first."
  exit 1
fi

echo "Checking CUDA availability..."
.venv/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"
echo

echo "Starting VisionGuard server..."
echo "Dashboard: http://127.0.0.1:8000"
echo "API Docs:  http://127.0.0.1:8000/docs"
echo
echo "Press Ctrl+C to stop"

.venv/bin/python -m app.main
