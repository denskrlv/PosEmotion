# !/usr/bin/env python3

import cv2
from keypoints import Keypoints
import mediapipe as mp
from ultralytics import YOLO


class Detector:

    def __init__(self, directory: str):
        self.directory = directory
        self.keypoints = Keypoints()


    def detect_pose(self, target, model, resize=(1280, 720)):
        keys = Keypoints()
        model = YOLO('/Users/deniskrylov/Developer/PosEmotion/models/yolo-pose.pt')

        img = cv2.imread('/Users/deniskrylov/Developer/PosEmotion/assets/frames/aJKL0ahn1Dk_19532.jpg')
        results = model('/Users/deniskrylov/Developer/PosEmotion/assets/frames/aJKL0ahn1Dk_19532.jpg')[0]

        for result in results:
            print(result.keypoints.xy.numpy().tolist()[0])
            for keypoint_idx, keypoint in enumerate(result.keypoints.xy.numpy().tolist()[0]):
                print(keypoint_idx, keypoint)
                # cv2.putText(img, str(keypoint_idx), (int(keypoint[0]), int(keypoint[1])),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv2.imshow('image', img)
        # cv2.waitKey(0)

# keys = Keypoints()
model = YOLO('/Users/deniskrylov/Developer/PosEmotion/models/yolo-pose.pt')

img = cv2.imread('/Users/deniskrylov/Developer/PosEmotion/assets/frames/aJKL0ahn1Dk_19532.jpg')
results = model('/Users/deniskrylov/Developer/PosEmotion/assets/frames/aJKL0ahn1Dk_19532.jpg')[0]

for result in results:
    print(result.keypoints.xy.numpy().tolist()[0])
    for keypoint_idx, keypoint in enumerate(result.keypoints.xy.numpy().tolist()[0]):
        print(keypoint_idx, keypoint)
