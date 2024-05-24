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
    # img = cv2.imread(target)
    keypoints = []

    if model == 'yolo':
        model = YOLO("/Users/deniskrylov/Developer/PosEmotion/models/yolo-pose.pt")
        results = model(image)[0]
        for result in results:
            for _, keypoint in enumerate(result.keypoints.xy.numpy().tolist()[0]):
                keypoints.append(keypoint)
    else:
        raise ValueError('Choose a valid model: yolo, openpose or alpha-pose!')
    
    return Keypoints(image=image, keys=keypoints)


def mask_image(target, x, y, w, h):
    image = cv2.imread(target)
    height, width, _ = image.shape
    # mask = np.zeros(image.shape[:2], dtype="uint8")

    x = int((float(x)/100) * width)
    y = int((float(y)/100) * height)
    w = int((float(w)/100) * width)
    h = int((float(h)/100) * height)

    # left, upper, right, lower = x, y, x + w, y + h
    # mask[upper:lower, left:right] = 255

    # black_background = np.zeros_like(image)

    # result_image = cv2.bitwise_and(image, image, mask=mask)
    # black_background = cv2.bitwise_and(black_background, black_background, mask=cv2.bitwise_not(mask))
    # final_image = cv2.add(result_image, black_background)
    cv2.rectangle(image, (x, y), (x + w, y + h), (128,128,128), 2)

    return image
