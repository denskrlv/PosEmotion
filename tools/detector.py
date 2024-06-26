# !/usr/bin/env python3

import os
import mediapipe as mp
import numpy as np
import pandas as pd

from tqdm import tqdm
from tools.structures import Skeleton


CORE_DIR = os.path.dirname(os.path.dirname(__file__))


def _empty_keypoints(count: int=33) -> list:
    """
    Only for internal use. Returns a list of empty keypoints.

    Parameters:
    count (int): The number of keypoints to generate. Default is 33.

    Returns:
        list: A list of empty keypoints, where each keypoint is represented by [np.nan, np.nan, np.nan].
    
    """
    return [[np.nan, np.nan, np.nan]] * count


def _align_skeleton(skeleton: Skeleton) -> Skeleton:
    """
    Only for internal use. Aligns the given skeleton by rotating it around the vertical axis.

    Args:
        skeleton (Skeleton): The input skeleton to be aligned.

    Returns:
        Skeleton: The aligned skeleton.

    """
    joints = skeleton.joints
    l_hip = joints["left_hip"]
    r_hip = joints["right_hip"]
    
    v = np.array(l_hip) - np.array(r_hip)
    u = np.cross(v, [1, 0, 0])
    u = u / np.linalg.norm(u)

    cos_theta = v.dot([1, 0, 0]) / np.linalg.norm(v)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    K = np.array([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0]
    ])

    # Rodrigues' rotation formula
    I = np.eye(3)
    R = I + sin_theta * K + (1 - cos_theta) * np.dot(K, K)

    rotated_joints = {}
    for key, point in joints.items():
        rotated_joints[key] = np.dot(R, point)

    return Skeleton(joints=rotated_joints, image=skeleton.image)


class PoseLandmarkerDetector:
    """
    Class for detecting pose landmarks in images using a pre-trained model.

    Args:
        model_path (str): The path to the pre-trained model.

    Attributes:
        model_path (str): The path to the pre-trained model.
        base_options: The base options for the pose landmarker.
        pose_landmarker: The pose landmarker task.
        pose_landmarker_options: The options for the pose landmarker.
        vision_running_mode: The running mode for the vision task.
        options: The options for the pose landmarker task.

    """
    def __init__(self, model_path: str) -> None:
        self.model_path = os.path.join(CORE_DIR, model_path)
        self.base_options = mp.tasks.BaseOptions
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker
        self.pose_landmarker_options = mp.tasks.vision.PoseLandmarkerOptions
        self.vision_running_mode = mp.tasks.vision.RunningMode
        self.options = self.pose_landmarker_options(
            base_options=self.base_options(model_asset_path=self.model_path),
            running_mode=self.vision_running_mode.IMAGE
        )

    def detect(self, file_path: str, conf: float = 0.3, depth: bool = True, align: bool = False) -> Skeleton:
        """
        Detects pose landmarks in an image.

        Args:
            file_path (str): The path to the image file.
            conf (float, optional): The confidence threshold for landmark visibility. Defaults to 0.3.
            depth (bool, optional): Whether to include depth information in the keypoints. Defaults to True.
            align (bool, optional): Whether to align the detected skeleton. Defaults to False.

        Returns:
            Skeleton: The detected skeleton.

        """
        keypoints = []
        mp_image = mp.Image.create_from_file(os.path.join(CORE_DIR, file_path))

        with self.pose_landmarker.create_from_options(self.options) as landmarker:
            results = landmarker.detect(mp_image)

            if results.pose_landmarks == []:
                return Skeleton(joints=_empty_keypoints(), image=file_path)

            results = results.pose_landmarks[0]
            for r in results:
                if r.visibility >= conf:
                    keypoints.append([r.x, r.y, r.z]) if depth else keypoints.append([r.x, r.y, 0])
                else:
                    keypoints.append([np.nan, np.nan, np.nan]) if depth else keypoints.append([np.nan, np.nan, 0])

        if align:
            return _align_skeleton(Skeleton(joints=keypoints, image=file_path))
        else:
            return Skeleton(joints=keypoints, image=file_path)

    def detect_multi(self, df: pd.DataFrame, frames_path: str, conf: float = 0.3, depth: bool = True,
                     align: bool = False) -> list:
        """
        Detects pose landmarks in multiple images.

        Args:
            df (pd.DataFrame): The DataFrame containing information about the images.
            frames_path (str): The path to the directory containing the image frames.
            conf (float, optional): The confidence threshold for landmark visibility. Defaults to 0.3.
            depth (bool, optional): Whether to include depth information in the keypoints. Defaults to True.
            align (bool, optional): Whether to align the detected skeletons. Defaults to False.

        Returns:
            list: A list of detected skeletons.

        """
        keypoints = []

        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Detecting"):
            result = self.detect(f"{frames_path}/{row['Video Tag']}_{index}.jpg", conf, depth, align)
            keypoints.append(result.to_series())

        return keypoints
