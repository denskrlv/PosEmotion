# !/usr/bin/env python3

import os

from IPython.display import display, Image


class Segment:

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def indices(self):
        return (self.df.index[0], self.df.index[-1])


class Keypoints:

    CORE_DIR = os.path.dirname(os.path.dirname(__file__))

    def __init__(self, keypoints):
        self.nose = keypoints[0]
        self.left_eye = keypoints[1]
        self.right_eye = keypoints[2]
        self.left_ear = keypoints[3]
        self.right_ear = keypoints[4]
        self.left_shoulder = keypoints[5]
        self.right_shoulder = keypoints[6]
        self.left_elbow = keypoints[7]
        self.right_elbow = keypoints[8]
        self.left_wrist = keypoints[9]
        self.right_wrist = keypoints[10]
        self.left_hip = keypoints[11]
        self.right_hip = keypoints[12]
        self.left_knee = keypoints[13]
        self.right_knee = keypoints[14]
        self.left_ankle = keypoints[15]
        self.right_ankle = keypoints[16]

    def __str__(self):
        return (
            "Keypoints:"
            f"\nNose: {self.nose}"
            f"\nLeft Eye: {self.left_eye}"
            f"\nRight Eye: {self.right_eye}"
            f"\nLeft Ear: {self.left_ear}"
            f"\nRight Ear: {self.right_ear}"
            f"\nLeft Shoulder: {self.left_shoulder}"
            f"\nRight Shoulder: {self.right_shoulder}"
            f"\nLeft Elbow: {self.left_elbow}"
            f"\nRight Elbow: {self.right_elbow}"
            f"\nLeft Wrist: {self.left_wrist}"
            f"\nRight Wrist: {self.right_wrist}"
            f"\nLeft Hip: {self.left_hip}"
            f"\nRight Hip: {self.right_hip}"
            f"\nLeft Knee: {self.left_knee}"
            f"\nRight Knee: {self.right_knee}"
            f"\nLeft Ankle: {self.left_ankle}"
            f"\nRight Ankle: {self.right_ankle}"
        )
    
    def get_head(self):
        return {
            "nose": self.nose,
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
            "left_ear": self.left_ear,
            "right_ear": self.right_ear
        }
    
    def get_torso(self):
        return {
            "left_shoulder": self.left_shoulder,
            "right_shoulder": self.right_shoulder,
            "left_hip": self.left_hip,
            "right_hip": self.right_hip
        }
    
    def get_left_arm(self):
        return {
            "left_shoulder": self.left_shoulder,
            "left_elbow": self.left_elbow,
            "left_wrist": self.left_wrist
        }
    
    def get_right_arm(self):
        return {
            "right_shoulder": self.right_shoulder,
            "right_elbow": self.right_elbow,
            "right_wrist": self.right_wrist
        }
    
    def get_left_leg(self):
        return {
            "left_hip": self.left_hip,
            "left_knee": self.left_knee,
            "left_ankle": self.left_ankle
        }
    
    def get_right_leg(self):
        return {
            "right_hip": self.right_hip,
            "right_knee": self.right_knee,
            "right_ankle": self.right_ankle
        }
    
    def to_list(self):
        return [
            self.nose, self.left_eye, self.right_eye, self.left_ear, self.right_ear,
            self.left_shoulder, self.right_shoulder, self.left_elbow, self.right_elbow,
            self.left_wrist, self.right_wrist, self.left_hip, self.right_hip,
            self.left_knee, self.right_knee, self.left_ankle, self.right_ankle
        ]
    
    def to_dict(self):
        keys = {}
        for var_name, var_value in vars(self).items():
            keys[var_name + "_X"] = var_value[0]
            keys[var_name + "_Y"] = var_value[1]
        return keys
