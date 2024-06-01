# !/usr/bin/env python3

import os
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

    CORE_DIR = os.path.dirname(os.path.dirname(__file__))

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
            "nose": _standartize(joints[0]),
            "left_eye": _standartize(joints[1]),
            "right_eye": _standartize(joints[2]),
            "left_ear": _standartize(joints[3]),
            "right_ear": _standartize(joints[4]),
            "left_shoulder": _standartize(joints[5]),
            "right_shoulder": _standartize(joints[6]),
            "left_elbow": _standartize(joints[7]),
            "right_elbow": _standartize(joints[8]),
            "left_wrist": _standartize(joints[9]),
            "right_wrist": _standartize(joints[10]),
            "left_hip": _standartize(joints[11]),
            "right_hip": _standartize(joints[12]),
            "left_knee": _standartize(joints[13]),
            "right_knee": _standartize(joints[14]),
            "left_ankle": _standartize(joints[15]),
            "right_ankle": _standartize(joints[16])
        }
    
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


def _standartize(joint):
    return joint if joint != [0, 0] else [np.nan, np.nan]

        
if __name__ == "__main__":
    joints_list = [
        [0, 0], [3, 4], [5, 6], [7, 8], [0, 0], [11, 12], [13, 14], [15, 16], [17, 18],
        [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32], [33, 34]
    ]
    skeleton = Skeleton(joints_list, image="image.jpg")
    print(skeleton)

    series = skeleton.to_series()
    print("Series:")
    print(series)

    new_skeleton = Skeleton.from_series(series)
    print("New Skeleton joints:")
    print(new_skeleton)

