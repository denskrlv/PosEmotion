# !/usr/bin/env python3

import ast
import numpy as np
import pandas as pd


def _find_column_pairs(columns: list) -> list:
    """
    Only for internal use. Find pairs of columns in the given list.

    Args:
        columns (list): A list of column names.

    Returns:
        list: A list of column names that have a corresponding pair.

    Example:
        >>> columns = ['A_X', 'B_X', 'C_Y', 'D_X', 'E_Y']
        >>> _find_column_pairs(columns)
        ['A', 'B', 'D']

    """
    pairs = []
    for col in columns:
        if col.endswith('_X'):
            pair_name = col[:-2]
            if f"{pair_name}_Y" in columns:
                pairs.append(pair_name)

    return pairs


class Segment:
    """
    Represents a segment of data.

    Args:
        df (pd.DataFrame): The DataFrame containing the segment data.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the segment data.

    Methods:
        __len__(): Returns the length of the segment.
        indices(): Returns the indices of the segment.
        get_column_names(): Returns a list of column names in the segment DataFrame.
        count_emotions(): Counts the occurrences of different emotions in the segment.
        get_vectors(pairs: list=[]): Returns a dictionary of vectors calculated from the segment data.
        vector_dist(pair: str): Calculates the distances between vectors in the segment data.

    """
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def indices(self) -> tuple:
        """
        Returns the indices of the first and last elements in the DataFrame.

        Returns:
            tuple: A tuple containing the index of the first element and the index of the last element.
        
        """
        return (self.df.index[0], self.df.index[-1])
    
    def get_column_names(self) -> list:
        """
        Returns a list of column names in the DataFrame.

        Returns:
            list: A list of column names.
        
        """
        return list(self.df.columns)

    def count_emotions(self) -> dict:
        """
        Counts the occurrences of different emotions in the dataset.

        Returns:
            emotion_count (dict): A dictionary containing the count of each emotion.
                The keys are the emotion names and the values are the corresponding counts.

        """
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
    
    def get_vectors(self, pairs: list=[]) -> dict:
        """
        Calculates the vectors for the given pairs of columns or finds the best pairs automatically if none are provided.

        Args:
            pairs (list, optional): A list of column pairs to calculate vectors for. If not provided, the method will automatically find the best pairs based on column names. Defaults to an empty list.

        Returns:
            dict: A dictionary containing the calculated vectors, with the "Label" key representing the label with the highest count of emotions.

        """
        if not pairs:
            pairs = _find_column_pairs(self.get_column_names())
        
        emotions = self.count_emotions()
        vectors = {
            "Label": max(emotions, key=emotions.get)
        }

        for pair in pairs:
            distances = self.vector_dist(pair)
            vectors.update(distances)
        
        return vectors
    
    def vector_dist(self, pair: str) -> dict:
        """
        Calculate the vector distances between the given pair and all other pairs in the dataframe.

        Args:
            pair (str): The pair for which to calculate the vector distances.

        Returns:
            dict: A dictionary containing the calculated vector distances, with keys in the format 'd_{pair}_{other_pair}_fft'.
        
        """
        distances = {}
        coordinates = self.df[[f"{pair}_X", f"{pair}_Y", f"{pair}_Z"]].values
        other_pairs = _find_column_pairs(self.get_column_names())
        other_pairs.remove(pair)

        for other_pair in other_pairs:
            other_coordinates = self.df[[f"{other_pair}_X", f"{other_pair}_Y", f"{other_pair}_Z"]].values
            for i in range(len(coordinates)):
                pair_dist = []
                pair_dist.append(np.linalg.norm(coordinates[i] - other_coordinates[i]))
            fft_result = np.real(np.fft.fft(pair_dist)[0])
            distances[f"d_{pair}_{other_pair}_fft"] = fft_result

        return distances


class Skeleton:
    """
    Represents a skeleton with joints and an associated image.

    Args:
        joints (dict or list): The joints of the skeleton. If a dictionary is provided, it should have joint names as keys and joint coordinates as values. If a list is provided, it should contain joint coordinates in the same order as the `joint_names` list.
        image (str, optional): The path to the associated image.

    Raises:
        TypeError: If `joints` is neither a list nor a dictionary.

    Attributes:
        joints (dict): The joints of the skeleton represented as a dictionary with joint names as keys and joint coordinates as values.
        image (str): The path to the associated image.

    Methods:
        __len__(): Returns the number of joints in the skeleton.
        to_dict(joints): Converts a list of joint coordinates to a dictionary representation.
        to_ndarray(): Converts the joint coordinates to a NumPy ndarray.
        to_series(): Converts the joint coordinates to a pandas Series.

    """
    def __init__(self, joints, image: str=None) -> None:
        if isinstance(joints, dict):
            self.joints = joints
        elif isinstance(joints, list):
            self.joints = self.to_dict(joints)
        else:
            raise TypeError("joints must be a list or a dictionary")
        self.image = image
    
    def __len__(self):
        return len(self.joints)
    
    def to_dict(self, joints: list) -> dict:
        """
        Converts a list of joint coordinates to a dictionary representation.

        Args:
            joints (list): The joint coordinates in the same order as the `joint_names` list.

        Returns:
            dict: A dictionary representation of the joint coordinates with joint names as keys and joint coordinates as values.

        """
        joint_names = [
            "nose", "left_reye", "left_eye", "left_leye", "right_leye", "right_eye", "right_reye", 
            "left_ear", "right_ear", "left_mouth", "right_mouth", "left_shoulder", "right_shoulder", 
            "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_f1", "right_f1", 
            "left_f2", "right_f2", "left_f3", "right_f3", "left_hip", "right_hip", "left_knee", 
            "right_knee", "left_ankle", "right_ankle", "left_hiel", "right_hiel", "left_teen", "right_teen"
        ]
    
        return {name: joints[i] for i, name in enumerate(joint_names)}

    
    def to_ndarray(self) -> np.ndarray:
        """
        Converts the joint coordinates to a NumPy ndarray.

        Returns:
            np.ndarray: An ndarray representation of the joint coordinates.

        """
        joints_list = [self.joints[f"point_{i}"] for i in range(33)]

        return np.array(joints_list)
    
    def to_series(self) -> pd.Series:
        """
        Converts the joint coordinates to a pandas Series.

        Returns:
            pd.Series: A Series representation of the joint coordinates.

        """
        transformed_dict = {}

        for key, value in self.joints.items():
            transformed_dict[f"{key}_X"] = value[0]
            transformed_dict[f"{key}_Y"] = value[1]
            transformed_dict[f"{key}_Z"] = value[2]

        return pd.Series(transformed_dict)
