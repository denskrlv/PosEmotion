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
                pairs.append([f"{pair_name}_X", f"{pair_name}_Y"])

    return pairs


def _standardize(joint):
    return joint if joint != [0, 0] else [np.nan, np.nan]


# probs = []

#     for _, row in df.iterrows():
#         nested_list = ast.literal_eval(row[labels_column])
#         if not preserve:
#             nested_list = _remove_empty(nested_list)
#         prob_single = np.zeros(len(Emotions))
#         n_label_size = _real_size(nested_list)
#         for label in nested_list:
#             if label != [""]:
#                 prob_single[Emotions[label[0]]] += 1 / n_label_size
#         probs.append(prob_single)

#     return np.round(probs, 2)


class Segment:

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def indices(self):
        return (self.df.index[0], self.df.index[-1])
    
    def get_column_names(self):
        return list(self.df.columns)
    
    def simplify_curve(self, pair, epsilon):
        coordinates = self.df[pair].values

        return rdp(coordinates, epsilon=epsilon)

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
    
    def avg_torso_angle(self):
        torso_angles = []
        shoulder_coords = self.df[['left_shoulder_X', 'left_shoulder_Y', 'right_shoulder_X', 'right_shoulder_Y']].values
        hip_coords = self.df[['left_hip_X', 'left_hip_Y', 'right_hip_X', 'right_hip_Y']].values

        for i in range(len(shoulder_coords)):
            shoulder_mid = ((shoulder_coords[i][0] + shoulder_coords[i][2]) / 2, (shoulder_coords[i][1] + shoulder_coords[i][3]) / 2)
            hips_mid = ((hip_coords[i][0] + hip_coords[i][2]) / 2, (hip_coords[i][1] + hip_coords[i][3]) / 2)

            dx = hips_mid[0] - shoulder_mid[0]
            dy = hips_mid[1] - shoulder_mid[1]

            rad = np.arctan2(dy, dx)
            deg = np.rad2deg(rad)
            torso_angles.append(deg)
        
        return np.mean(torso_angles)
    
    def avg_head_position(self):
        head_positions = []
        nose_coords = self.df[['nose_X', 'nose_Y']].values
        left_eye_coords = self.df[['left_eye_X', 'left_eye_Y']].values
        right_eye_coords = self.df[['right_eye_X', 'right_eye_Y']].values
        left_ear_coords = self.df[['left_ear_X', 'left_ear_Y']].values
        right_ear_coords = self.df[['right_ear_X', 'right_ear_Y']].values

        for i in range(len(nose_coords)):
            head_position = [(nose_coords[i][0] + left_eye_coords[i][0] + right_eye_coords[i][0] + left_ear_coords[i][0] + right_ear_coords[i][0]) / 5,
                             (nose_coords[i][1] + left_eye_coords[i][1] + right_eye_coords[i][1] + left_ear_coords[i][1] + right_ear_coords[i][1]) / 5]
            if not np.isnan(head_position).any():
                head_positions.append(head_position)

        mean_head_positions = np.mean(head_positions, axis=0)

        return mean_head_positions[0], mean_head_positions[1]
    
    def avg_head_tilt(self):
        head_tilts = []
        left_eye_coords = self.df[['left_eye_X', 'left_eye_Y']].values
        right_eye_coords = self.df[['right_eye_X', 'right_eye_Y']].values

        for i in range(len(left_eye_coords)):
            dx = right_eye_coords[i][0] - left_eye_coords[i][0]
            dy = right_eye_coords[i][1] - left_eye_coords[i][1]

            rad = np.arctan2(dy, dx)
            deg = np.rad2deg(rad)
            head_tilts.append(deg)
        
        return np.mean(head_tilts)
    
    def avg_velocity(self, pair, fps=24):
        coodinates = self.df[pair].values
        frame_numbers = self.df["Frame Number"].values

        dist = np.linalg.norm(coodinates[1:] - coodinates[:-1], axis=1)
        time = (frame_numbers[0] - frame_numbers[-1]) / fps

        return abs(np.sum(dist) / time)
    
    def avg_acceleration(self, pair, fps=24):
        coodinates = self.df[pair].values
        frame_numbers = self.df["Frame Number"].values

        dist = np.linalg.norm(coodinates[1:] - coodinates[:-1], axis=1)
        time = (frame_numbers[0] - frame_numbers[-1]) / fps
        velocity = abs(np.sum(dist) / time)

        return velocity / time
    
    def avg_jerk(self, pair, fps=24):
        coodinates = self.df[pair].values
        frame_numbers = self.df["Frame Number"].values

        dist = np.linalg.norm(coodinates[1:] - coodinates[:-1], axis=1)
        time = (frame_numbers[0] - frame_numbers[-1]) / fps
        velocity = abs(np.sum(dist) / time)
        acceleration = velocity / time

        return acceleration / time
    
    def avg_hands_position(self):
        left_wrist_coords = self.df[['left_wrist_X', 'left_wrist_Y']].values
        right_wrist_coords = self.df[['right_wrist_X', 'right_wrist_Y']].values

        mean_left_wrist_coords = np.mean(left_wrist_coords, axis=0)
        mean_right_wrist_coords = np.mean(right_wrist_coords, axis=0)

        return (mean_left_wrist_coords[0], mean_left_wrist_coords[1]), (mean_right_wrist_coords[0], mean_right_wrist_coords[1])
    
    def avg_dist_to_body(self):
        left_wrist_coords = self.df[['left_wrist_X', 'left_wrist_Y']].values
        right_wrist_coords = self.df[['right_wrist_X', 'right_wrist_Y']].values
        left_shoulder_coords = self.df[['left_shoulder_X', 'left_shoulder_Y']].values
        right_shoulder_coords = self.df[['right_shoulder_X', 'right_shoulder_Y']].values
        left_hip_coords = self.df[['left_hip_X', 'left_hip_Y']].values
        right_hip_coords = self.df[['right_hip_X', 'right_hip_Y']].values

        left_wrist_to_body = np.linalg.norm(left_wrist_coords - left_shoulder_coords, axis=1) + np.linalg.norm(left_wrist_coords - left_hip_coords, axis=1)
        right_wrist_to_body = np.linalg.norm(right_wrist_coords - right_shoulder_coords, axis=1) + np.linalg.norm(right_wrist_coords - right_hip_coords, axis=1)

        return np.mean(left_wrist_to_body), np.mean(right_wrist_to_body)
    
    def avg_posture_symmetry(self):
        left_shoulder_coords = self.df[['left_shoulder_X', 'left_shoulder_Y']].values
        right_shoulder_coords = self.df[['right_shoulder_X', 'right_shoulder_Y']].values
        left_hip_coords = self.df[['left_hip_X', 'left_hip_Y']].values
        right_hip_coords = self.df[['right_hip_X', 'right_hip_Y']].values

        shoulder_symmetry = np.linalg.norm(left_shoulder_coords - right_shoulder_coords, axis=1)
        hip_symmetry = np.linalg.norm(left_hip_coords - right_hip_coords, axis=1)

        return np.mean(shoulder_symmetry), np.mean(hip_symmetry)


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
