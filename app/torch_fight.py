"""Optional Windows CUDA temporal fight classification using torchvision R3D-18.

Different from YOLO object detection: reports a *whole-clip action hypothesis*,
not a fight bounding box. Requires user-trained clips with known class labels.
A downloaded unknown-label Keras 3DCNN is not activated by this module.
"""
from collections import deque
from pathlib import Path
import threading

# R3D weights are shared across cameras; buffers and confidence history are not.
_R3D_POOL = {}
_R3D_POOL_MUTEX = threading.RLock()

import cv2
import numpy as np

FRAMES = 16
SIZE = 112
MEAN = (0.43216, 0.394666, 0.37645)
STD = (0.22803, 0.22145, 0.216989)


def preprocess_frames(frames):
    """BGR uint8 frames -> FloatTensor (C,T,112,112), Kinetics-style RGB."""
    import torch
    if len(frames) != FRAMES:
        raise ValueError(f'Expected {FRAMES} frames, got {len(frames)}')
    images = []
    for frame in frames:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Expected BGR video frames')
        rgb = cv2.cvtColor(cv2.resize(frame, (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
        images.append(rgb)
    array = np.stack(images, axis=0).astype('float32') / 255.0
    array = (array - np.array(MEAN, dtype='float32')) / np.array(STD, dtype='float32')
    return torch.from_numpy(array.copy()).permute(3, 0, 1, 2).contiguous()


def build_r3d(pretrained=False):
    from torchvision.models.video import r3d_18, R3D_18_Weights
    from torch import nn
    net = r3d_18(weights=R3D_18_Weights.DEFAULT if pretrained else None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    return net


class FightClipClassifier:
    """Opt-in classifier; requires verified 2-class Torch checkpoint metadata.

    Two positive consecutive *distinct* windows are required for temporal event.
    This avoids treating one frame or one repeated score as two confirmations.
    """
    def __init__(self, checkpoint, threshold=0.85, interval=8, device='auto'):
        import torch
        checkpoint = Path(checkpoint)
        # weights_only=True prevents executing arbitrary objects in saved state.
        obj = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if obj.get('format') != 'visionguard-r3d18-v1' or obj.get('classes') != ['normal','fight']:
            raise ValueError('Missing verified normal/fight metadata; not loading action classifier')
        if obj.get('frames') != FRAMES or obj.get('size') != SIZE:
            raise ValueError('Incompatible frame preprocessing metadata')
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if str(self.device).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available')
        key = (str(checkpoint.resolve()), str(self.device))
        with _R3D_POOL_MUTEX:
            record = _R3D_POOL.get(key)
            if record is None:
                model = build_r3d(False)
                model.load_state_dict(obj['state_dict'], strict=True)
                model.to(self.device).eval()
                record = (model, threading.RLock())
                _R3D_POOL[key] = record
            self.model, self._inference_lock = record
        self.buffer = deque(maxlen=FRAMES)
        self.threshold = float(threshold)
        self.interval = max(1, int(interval))
        self.counter = 0
        self.positive_windows = 0

    def analyze(self, frame):
        import torch
        self.buffer.append(frame.copy())
        self.counter += 1
        if len(self.buffer) != FRAMES or (self.counter - FRAMES) % self.interval:
            return None
        clip = preprocess_frames(self.buffer).unsqueeze(0).to(self.device)
        with self._inference_lock, torch.inference_mode():
            confidence = float(self.model(clip).softmax(1)[0, 1].item())
        self.positive_windows = self.positive_windows + 1 if confidence >= self.threshold else 0
        if self.positive_windows < 2:
            return None
        return {
            'type':'violence', 'severity':'high', 'subtype':'temporal_fight_classifier',
            'confidence':confidence, 'bbox':[0.0,0.0,1.0,1.0],
            'source_model':'torch_r3d18_fight', 'temporal_confirmed':True,
            'message':f'Temporal clip classified as possible fight ({confidence:.0%}); no spatial box',
        }
