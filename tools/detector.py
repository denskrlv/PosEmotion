# !/usr/bin/env python3

import os
from tools.structures import Keypoints
from ultralytics import YOLO


def _empty_keypoints(count=17):
        return [[0, 0]] * count


class Detector:

    CORE_DIR = os.path.dirname(os.path.dirname(__file__))

    def __init__(self, model_path):
        self.model = YOLO(os.path.join(self.CORE_DIR, model_path))

    def detect_poses(self, file_path):
        file_path = os.path.join(self.CORE_DIR, file_path)

        results = self.model(file_path)[0]
        keypoints = results.keypoints.xy.numpy().tolist()[0]

        if keypoints != []:
            return Keypoints(image=file_path, keys=keypoints)
        else:
            return Keypoints(image=file_path, keys=_empty_keypoints())
