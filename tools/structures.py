# !/usr/bin/env python3

import ast
import numpy as np
import pandas as pd


def find_column_pairs(columns):
    pairs = []
    for col in columns:
        if col.endswith('_X'):
            pair_name = col[:-2]
            if f"{pair_name}_Y" in columns:
                pairs.append(pair_name)

    return pairs


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
            distances = self.vector_dist(pair)
            angles = self.vector_angle(pair)
            vectors.update(distances)
            vectors.update(angles)
        
        return vectors
    
    def vector_dist(self, pair):
        distances = {}
        coordinates = self.df[[f"{pair}_X", f"{pair}_Y", f"{pair}_Z"]].values
        other_pairs = find_column_pairs(self.get_column_names())
        other_pairs.remove(pair)

        for other_pair in other_pairs:
            other_coordinates = self.df[[f"{other_pair}_X", f"{other_pair}_Y", f"{other_pair}_Z"]].values
            for i in range(len(coordinates)):
                distances[f"d_{pair}_{other_pair}_{i}"] = np.linalg.norm(coordinates[i] - other_coordinates[i])

        return distances
    
    def vector_angle(self, pair):
        angles = {}
        coordinates = self.df[[f"{pair}_X", f"{pair}_Y", f"{pair}_Z"]].values
        
        left_hip = self.df[["left_hip_X", "left_hip_Y", "left_hip_Z"]].values
        right_hip = self.df[["right_hip_X", "right_hip_Y", "right_hip_Z"]].values
        hips_vector = right_hip - left_hip

        for i in range(len(coordinates)):
            dot_product = np.dot(coordinates[i], hips_vector[i])
            norm_product = np.linalg.norm(coordinates[i]) * np.linalg.norm(hips_vector[i])
            angles[f"a_{pair}_hips_{i}"] = np.arccos(dot_product / norm_product)

        return angles


class Skeleton:
    def __init__(self, joints, image: str=None):
        if isinstance(joints, dict):
            self.joints = joints
        elif isinstance(joints, list):
            self.joints = self.to_dict(joints)
        else:
            raise TypeError("joints must be a list or a dictionary")
        self.image = image
    
    def __len__(self):
        return len(self.joints)
    
    def to_dict(self, joints):
        joint_names = [
            "nose", "left_reye", "left_eye", "left_leye", "right_leye", "right_eye", "right_reye", 
            "left_ear", "right_ear", "left_mouth", "right_mouth", "left_shoulder", "right_shoulder", 
            "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_f1", "right_f1", 
            "left_f2", "right_f2", "left_f3", "right_f3", "left_hip", "right_hip", "left_knee", 
            "right_knee", "left_ankle", "right_ankle", "left_hiel", "right_hiel", "left_teen", "right_teen"
        ]
    
        return {name: joints[i] for i, name in enumerate(joint_names)}

    
    def to_ndarray(self):
        joints_list = [self.joints[f"point_{i}"] for i in range(33)]

        return np.array(joints_list)
    
    def to_series(self):
        transformed_dict = {}
        for key, value in self.joints.items():
            transformed_dict[f"{key}_X"] = value[0]
            transformed_dict[f"{key}_Y"] = value[1]
            transformed_dict[f"{key}_Z"] = value[2]

        return pd.Series(transformed_dict)
