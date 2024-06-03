# !/usr/bin/env python3

import cv2
import os
import mediapipe as mp
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tools.structures import Skeleton
from ultralytics import YOLO


CORE_DIR = os.path.dirname(os.path.dirname(__file__))

def _empty_keypoints(count=17):
        return [[0, 0]] * count


class YoloDetector:

    def __init__(self, model_path):
        self.model = YOLO(os.path.join(CORE_DIR, model_path))

    def detect(self, file_path):
        file_path = os.path.join(CORE_DIR, file_path)

        results = self.model(file_path)[0]
        keypoints = results.keypoints.xy.numpy().tolist()[0]

        if keypoints != []:
            return Skeleton(joints=keypoints, image=file_path)
        else:
            return Skeleton(joints=_empty_keypoints(), image=file_path)
        
    def detect_multi(self, df, frames_path):
        keypoints = []

        for index, row in df.iterrows():
            result = self.detect(f"{frames_path}/{row['Video Tag']}_{index}.jpg")
            keypoints.append(result.to_series())
            print("Progress: {}/{}".format(index+1, len(df)))
        
        return keypoints
    

class MoveNetDetector:

    def __init__(self, model):
        if model == "lightning":
            self.model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        else:
            raise Exception("Model is incorrect!")
        
    def detect(self, file_path, conf=0.3):
        file_path = os.path.join(CORE_DIR, file_path)

        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_image = tf.image.resize(image_rgb, (192, 192))
        input_image = tf.expand_dims(input_image, axis=0)
        input_image = tf.cast(input_image, dtype=tf.int32)

        outputs = self.model.signatures["serving_default"](input_image)
        keypoints = outputs['output_0'].numpy().tolist()[0][0]

        if keypoints == []:
            return Skeleton(joints=_empty_keypoints(), image=file_path)
        
        keypoints = [
            [x, y] if confidence >= conf else [np.nan, np.nan]
            for y, x, confidence in keypoints
        ]

        original_height, original_width, _ = image.shape
        adjusted_keypoints = []

        for keypoint in keypoints:
            x, y = keypoint
            if not np.isnan(x) and not np.isnan(y):
                x *= original_width
                y *= original_height
                adjusted_keypoints.append([x, y])
            else:
                adjusted_keypoints.append([np.nan, np.nan])
        
        return Skeleton(joints=adjusted_keypoints, image=file_path)
    
    def detect_multi(self, df, frames_path, conf=0.3):
        keypoints = []

        for index, row in df.iterrows():
            result = self.detect(f"{frames_path}/{row['Video Tag']}_{index}.jpg", conf)
            keypoints.append(result.to_series())
            print("Progress: {}/{}".format(index+1, len(df)))
        
        return keypoints


class PoseLandmarkerDetector:

    def __init__(self, model_path):
        self.model_path = os.path.join(CORE_DIR, model_path)
        self.base_options = mp.tasks.BaseOptions
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker
        self.pose_landmarker_options = mp.tasks.vision.PoseLandmarkerOptions
        self.vision_running_mode = mp.tasks.vision.RunningMode
        self.options = self.pose_landmarker_options(
            base_options = self.base_options(model_asset_path=self.model_path),
            running_mode = self.vision_running_mode.IMAGE
        )

    def detect(self, file_path, conf=0.3):
        keypoints = []
        depths = []

        mp_image = mp.Image.create_from_file(os.path.join(CORE_DIR, file_path))
        indices = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

        with self.pose_landmarker.create_from_options(self.options) as landmarker:
            results = landmarker.detect(mp_image)
            
            if results.pose_landmarks == []:
                return Skeleton(joints=_empty_keypoints(), image=file_path), depths

            results = results.pose_landmarks[0]
            results = [results[i] for i in indices]
            for r in results:
                if r.visibility >= conf:
                    keypoints.append([r.x, r.y])
                    depths.append(r.z)
                else:
                    keypoints.append([np.nan, np.nan])
                    depths.append(np.nan)

        original_height = mp_image.height
        original_width = mp_image.width
        adjusted_keypoints = []

        for keypoint in keypoints:
            x, y = keypoint
            if not np.isnan(x) and not np.isnan(y):
                x *= original_width
                y *= original_height
                adjusted_keypoints.append([x, y])
            else:
                adjusted_keypoints.append([np.nan, np.nan])
        
        return Skeleton(joints=adjusted_keypoints, image=file_path), depths
    
    def detect_multi(self, df, frames_path, conf=0.3):
        keypoints = []

        for index, row in df.iterrows():
            result, _ = self.detect(f"{frames_path}/{row['Video Tag']}_{index}.jpg", conf)
            keypoints.append(result.to_series())
            print("Progress: {}/{}".format(index+1, len(df)))
        
        return keypoints
