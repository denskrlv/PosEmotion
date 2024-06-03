# !/usr/bin/env python3

import numpy as np
import pandas as pd


class Segment:

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def indices(self):
        return (self.df.index[0], self.df.index[-1])


class Skeleton:

    def __init__(self, joints, image: str=None):
        if isinstance(joints, dict):
            self.joints = joints
        elif isinstance(joints, list):
            self.joints = self.to_dict(joints)
        else:
            raise TypeError("joints must be a list or a dictionary")
        self.image = image

    def __str__(self):
        return (
            "Joints:"
            f"\nImage: {self.image}"
            f"\nNose: {self.joints['nose']}"
            f"\nLeft Eye: {self.joints['left_eye']}"
            f"\nRight Eye: {self.joints['right_eye']}"
            f"\nLeft Ear: {self.joints['left_ear']}"
            f"\nRight Ear: {self.joints['right_ear']}"
            f"\nLeft Shoulder: {self.joints['left_shoulder']}"
            f"\nRight Shoulder: {self.joints['right_shoulder']}"
            f"\nLeft Elbow: {self.joints['left_elbow']}"
            f"\nRight Elbow: {self.joints['right_elbow']}"
            f"\nLeft Wrist: {self.joints['left_wrist']}"
            f"\nRight Wrist: {self.joints['right_wrist']}"
            f"\nLeft Hip: {self.joints['left_hip']}"
            f"\nRight Hip: {self.joints['right_hip']}"
            f"\nLeft Knee: {self.joints['left_knee']}"
            f"\nRight Knee: {self.joints['right_knee']}"
            f"\nLeft Ankle: {self.joints['left_ankle']}"
            f"\nRight Ankle: {self.joints['right_ankle']}"
        )
    
    def to_dict(self, joints):
        return {
            "nose": _standardize(joints[0]),
            "left_eye": _standardize(joints[1]),
            "right_eye": _standardize(joints[2]),
            "left_ear": _standardize(joints[3]),
            "right_ear": _standardize(joints[4]),
            "left_shoulder": _standardize(joints[5]),
            "right_shoulder": _standardize(joints[6]),
            "left_elbow": _standardize(joints[7]),
            "right_elbow": _standardize(joints[8]),
            "left_wrist": _standardize(joints[9]),
            "right_wrist": _standardize(joints[10]),
            "left_hip": _standardize(joints[11]),
            "right_hip": _standardize(joints[12]),
            "left_knee": _standardize(joints[13]),
            "right_knee": _standardize(joints[14]),
            "left_ankle": _standardize(joints[15]),
            "right_ankle": _standardize(joints[16])
        }
    
    def to_list(self):
        return [
            self.joints["nose"], self.joints["left_eye"], self.joints["right_eye"],
            self.joints["left_ear"], self.joints["right_ear"], self.joints["left_shoulder"],
            self.joints["right_shoulder"], self.joints["left_elbow"], self.joints["right_elbow"],
            self.joints["left_wrist"], self.joints["right_wrist"], self.joints["left_hip"],
            self.joints["right_hip"], self.joints["left_knee"], self.joints["right_knee"],
            self.joints["left_ankle"], self.joints["right_ankle"]
        ]
    
    def to_series(self):
        transformed_dict = {}
        for key, value in self.joints.items():
            transformed_dict[f"{key}_X"] = value[0]
            transformed_dict[f"{key}_Y"] = value[1]
        return pd.Series(transformed_dict)
    
    @classmethod
    def from_series(cls, series, image=None):
        joints = {}
        for key in series.index:
            if key.endswith('_X'):
                joint_name = key[:-2]
                x = series[key]
                y = series.get(f"{joint_name}_Y", None)
                joints[joint_name] = [x, y]
        return cls(joints=joints, image=image)


def _standardize(joint):
    return joint if joint != [0, 0] else [np.nan, np.nan]
