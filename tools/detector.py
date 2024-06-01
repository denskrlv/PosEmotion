# !/usr/bin/env python3

import os

from tools.structures import Skeleton
from ultralytics import YOLO


def _empty_keypoints(count=17):
        return [[0, 0]] * count


""" 
Detector should use multiple detection models to detect poses in images combined.
"""


class Detector:

    CORE_DIR = os.path.dirname(os.path.dirname(__file__))

    def __init__(self, model_path):
        self.model = YOLO(os.path.join(self.CORE_DIR, model_path))

    def detect_poses(self, file_path):
        file_path = os.path.join(self.CORE_DIR, file_path)

        results = self.model(file_path)[0]
        keypoints = results.keypoints.xy.numpy().tolist()[0]

        if keypoints != []:
            return Skeleton(joints=keypoints, image=file_path)
        else:
            return Skeleton(joints=_empty_keypoints())
