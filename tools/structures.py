# !/usr/bin/env python3

import ast
import numpy as np
import pandas as pd
from rdp import rdp


def find_column_pairs(columns):
    pairs = []
    for col in columns:
        if col.endswith('_X'):
            pair_name = col[:-2]
            if f"{pair_name}_Y" in columns:
                pairs.append(pair_name)

    return pairs


def _standardize(joint):
    return joint if joint != [0, 0] else [np.nan, np.nan]


class Segment:

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def indices(self):
        return (self.df.index[0], self.df.index[-1])
    
    def get_column_names(self):
        return list(self.df.columns)

    def count_emotions(self):
        base_emotions = ["Happy", "Sad", "Fear", "Surprise", "Disgust", "Anger", "Neutral"]
        emotion_count = {emotion: 0 for emotion in base_emotions}

        emotions = self.df['Labels'].values
        emotions = [ast.literal_eval(emotion) for emotion in emotions]

        for subarray in emotions:
            for inner_array in subarray:
                if inner_array != "No annotation":
                    for label in inner_array:
                        if label in emotion_count:
                            emotion_count[label] += 1
                        else:
                            emotion_count[label] = 1

        return emotion_count
    
    def get_vectors(self, pairs=[]):
        if not pairs:
            pairs = find_column_pairs(self.get_column_names())
        
        emotions = self.count_emotions()
        vectors = {
            "Label": max(emotions, key=emotions.get)
        }

        for pair in pairs:
            features = self.vector_features(pair)
            vectors.update(features)
        
        return vectors
    
    def vector_features(self, pair):
        features = {}
        coordinates = self.df[[f"{pair}_X", f"{pair}_Y"]].values

        # simplified_coordinates = rdp(coordinates, epsilon=epsilon)  # simplify curve

        for i in range(1, len(coordinates)):
            v = coordinates[i] - coordinates[i - 1]
            features[f"Magnitude_{pair}_{i}"] = np.linalg.norm(v)
            features[f"Direction_{pair}_{i}"] = np.arctan2(v[1], v[0])
            features[f"Slope_{pair}_{i}"] = v[1] / v[0] if v[0] != 0 else 0  # np.inf won't work with clustering

        return features


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
    
    def __len__(self):
        return len(self.joints)
    
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
