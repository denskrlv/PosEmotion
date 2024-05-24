# !/usr/bin/env python3

import cv2
import numpy as np
from tools.keypoints import Keypoints
import mediapipe as mp
from ultralytics import YOLO


Emotions = {
    "Happy"     : 0,
    "Sad"       : 1,
    "Fear"      : 2,
    "Neutral"   : 3,
    "Surprise"  : 4,
    "Disgust"   : 5,
    "Anger"     : 6
}


def detect_poses(image, model, resize=(1280, 720)):
    results = model(image)[0]
    keypoints = results.keypoints.xy.numpy().tolist()[0]

    if keypoints != []:
        return Keypoints(image=image, keys=keypoints)
    else:
        return Keypoints(image=image, keys=_empty_keypoints())


def _empty_keypoints():
    return [[0, 0]] * 17
