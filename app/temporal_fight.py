"""Opt-in 3D CNN classification over a *sequence* of real frames.

Never treat single frames as temporal evidence. Class polarity is not known
from an .h5 file; user must confirm positive class mapping before enable.
"""
from collections import deque
import logging
from pathlib import Path
import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

class TemporalFightDetector:
    def __init__(self, path, positive_index=None, confidence=0.75, interval=8):
        if positive_index not in (0,1):
            raise ValueError('Confirm label mapping: set CNN3D_POSITIVE_CLASS=0 or 1')
        from tensorflow.keras.models import load_model
        self.model=load_model(str(path),compile=False)
        shape=self.model.input_shape
        if isinstance(shape,list):shape=shape[0]
        if len(shape)!=5 or any(not isinstance(x,int) or x<=0 for x in shape[1:]):
            raise ValueError(f'Unsupported temporal model shape: {shape}')
        self.timesteps,self.height,self.width,self.channels=shape[1:]
        if self.channels not in (1,3): raise ValueError('Unsupported channel count')
        self.buffer=deque(maxlen=self.timesteps)
        self.positive_index=positive_index
        self.confidence=confidence
        self.counter=0
        self.interval=max(1,interval)

    def analyze(self, frame):
        resized=cv2.resize(frame,(self.width,self.height))
        if self.channels==3:resized=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
        else:resized=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)[...,None]
        self.buffer.append(resized.astype('float32') / 255.0)
        self.counter+=1
        if len(self.buffer)<self.timesteps or self.counter%self.interval:
            return None
        pred=np.asarray(self.model.predict(np.expand_dims(np.stack(self.buffer),0),verbose=0)).ravel()
        if pred.size==1:
            p=float(pred[0]);p=p if self.positive_index==1 else 1-p
        elif pred.size==2:
            p=float(pred[self.positive_index])
        else:raise ValueError(f'Unexpected classifier outputs: {pred.shape}')
        if not 0<=p<=1:raise ValueError('Model output is not a probability')
        if p>=self.confidence:
            return {'type':'violence','severity':'critical','subtype':'3dcnn_temporal',
                    'confidence':p,'bbox':[0.,0.,1.,1.],
                    'message':f'3D CNN sequence classified as fight ({p:.0%})',
                    'source_model':'3dcnn'}
        return None
