# !/usr/bin/env python3

import cv2
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


def detect_poses(target, model, resize=(1280, 720)):
    img = cv2.imread(target)
    keypoints = []

    if model == 'yolo':
        model = YOLO("/Users/deniskrylov/Developer/PosEmotion/models/yolo-pose.pt")
        results = model(target)
        print(results)
        # for result in results:
        #     for _, keypoint in enumerate(result.keypoints.xy.numpy().tolist()[0]):
        #         keypoints.append(keypoint)
    else:
        raise ValueError('Choose a valid model: yolo, openpose or alpha-pose!')
    
    return Keypoints(image=img, keys=keypoints)
