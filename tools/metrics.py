# !/usr/bin/env python3

import ast
import numpy as np
import os
import pandas as pd

from tools.structures import Skeleton, Segment


CORE_DIR = os.path.dirname(os.path.dirname(__file__))

Emotions = {
    "Happy"     : 0,
    "Sad"       : 1,
    "Fear"      : 2,
    "Neutral"   : 3,
    "Surprise"  : 4,
    "Disgust"   : 5,
    "Anger"     : 6
}


def label_probabilities(df: pd.DataFrame, labels_column: str="Labels", preserve: bool=False) -> list[float]:
    """
    Calculate the probabilities of each emotion label in the given DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        labels_column (str, optional): The name of the column containing the emotion labels. Defaults to "Labels".
        preserve (bool, optional): Whether to preserve empty labels in the calculation. Defaults to False.

    Returns:
        list: A list of probabilities for each emotion label.

    This function iterates over the rows of the DataFrame and calculates the probabilities of each emotion label.
    It first converts the string representation of the labels into a nested list using `ast.literal_eval`.
    If `preserve` is False, it removes empty labels from the nested list using the `_remove_empty` function.
    It then initializes an array of zeros with the length of the `Emotions` dictionary.
    For each non-empty label in the nested list, it increments the corresponding index in the probability array.
    Finally, it appends the probability array to the list of probabilities.

    The function returns the list of probabilities rounded to two decimal places.
    """
    probs = []

    for _, row in df.iterrows():
        nested_list = ast.literal_eval(row[labels_column])
        if not preserve:
            nested_list = _remove_empty(nested_list)
        prob_single = np.zeros(len(Emotions))
        n_label_size = _real_size(nested_list)
        for label in nested_list:
            if label != [""]:
                prob_single[Emotions[label[0]]] += 1 / n_label_size
        probs.append(prob_single)

    return np.round(probs, 2)


def segmentate(df: pd.DataFrame) -> list[Segment]:
    """
    This function segments the input DataFrame into different segments based on changes in 'Video Tag', 'Clip Id', and 'Person Id'.

    Args:
        df (pandas.DataFrame): The input DataFrame which contains the columns 'Video Tag', 'Clip Id', 'Person Id' 
        and possibly 'X', 'Y', 'Width', 'Height'.

    Returns:
        list: A list of Segment objects. Each Segment object represents a segment of the input DataFrame where 
        'Video Tag', 'Clip Id', and 'Person Id' are constant.

    The function first drops the columns 'X', 'Y', 'Width', 'Height' if they exist in the DataFrame. 
    Then it iterates over the rows of the DataFrame. When it detects a change in 'Video Tag', 'Clip Id', or 'Person Id', 
    it creates a new Segment with the rows up to that point and starts a new segment. 
    After iterating over all rows, it creates a Segment with the remaining rows if any are left.
    """
    segments = []
    base = (None, None, None)
    start_i = 0
    num = 0

    # columns_to_drop = ["X", "Y", "Width", "Height"]
    # for column in columns_to_drop:
    #     if column in df.columns:
    #         df = df.drop(columns=column)

    for index, row in df.iterrows():
        if base == (None, None, None):
            base = (row['Video Tag'], row['Clip Id'], row['Person Id'])
            continue
        if (row['Video Tag'], row['Clip Id'], row['Person Id']) != base:
            base = (row['Video Tag'], row['Clip Id'], row['Person Id'])
            segments.append(Segment(df.loc[start_i:index-1]))
            start_i = index
            num += 1
    
    if start_i < df.shape[0]:
        segments.append(Segment(df.loc[start_i:]))

    return segments


def normalize_segment(segment: Segment, target_size: int=10) -> Segment:
    """
    This function normalizes the length of a segment to a target size by either removing or duplicating rows.

    Args:
        segment (Segment): The input Segment object which contains a DataFrame to be normalized.
        target_size (int, optional): The target number of rows for the segment. Defaults to 10.
        after (str, optional): The name of the column in the DataFrame after which empty keypoints are removed. Defaults to "Anger".

    Returns:
        Segment: A new Segment object with the normalized DataFrame.

    The function first removes empty keypoints from the DataFrame and resets its index. Then it calculates the start and end indices 
    of the DataFrame and the length of the DataFrame.

    If the length of the DataFrame is greater than the target size, it calculates the indices of redundant rows and 
    drops them from the DataFrame. If the length of the DataFrame is less than the target size, it calculates 
    the indices of rows to be duplicated and inserts duplicates of these rows into the DataFrame.

    Finally, it returns a new Segment with the normalized DataFrame.
    """
    df = segment.df.reset_index(drop=True)
    start, end = 0, len(df)-1
    length = end - start + 1
    rows_inserted = 0

    segment_pivots = list(np.linspace(start, end, target_size, dtype=int))
    if length > target_size:
        redundant_indices = _get_redundant_indices(start, end, segment_pivots)
        df = df.drop(index=redundant_indices).reset_index(drop=True)
    elif length < target_size:
        duplicate_rows = _get_duplicate_indices(segment_pivots)
        for index in duplicate_rows:
            adjusted_index = index + rows_inserted
            duplicated_row = df.iloc[[adjusted_index]]
            upper = df.iloc[:adjusted_index+1]
            lower = df.iloc[adjusted_index+1:]
            df = pd.concat([upper, duplicated_row, lower]).reset_index(drop=True)
            rows_inserted += 1
    
    return Segment(df)


def normalize_skeleton(skeleton: Skeleton, box: tuple[float, float, float, float]) -> Skeleton:
    """
    This function normalizes the keypoints based on the bounding box.

    Args:
        keypoints (Keypoints): The input Keypoints object which contains the keypoints to be normalized.
        box (tuple): A tuple containing the x, y coordinates and the width and height of the bounding box.

    Returns:
        Keypoints: A new Keypoints object with the normalized keypoints.

    The function first extracts the x, y coordinates and the width and height from the bounding box. 
    Then it converts the keypoints to a list and initializes an empty list for the normalized keypoints.

    It then iterates over the keypoints. For each keypoint, if both coordinates are not None, it subtracts the x, y coordinates 
    of the bounding box from the keypoint coordinates and divides the result by the width and height of the bounding box, respectively. 
    The normalized keypoint is then added to the list of normalized keypoints. If either coordinate of the keypoint is None, 
    a keypoint of (0, 0) is added to the list of normalized keypoints.

    Finally, it returns a new Keypoints object with the normalized keypoints and the same image as the input keypoints.
    """
    x, y, w, h = box
    joints = skeleton.joints
    norm_skeleton = []

    for _, value in joints.items():
        if value != [np.nan, np.nan]:
            value = np.array(value)
            value[0] = (value[0] - x) / w
            value[1] = (value[1] - y) / h
            norm_skeleton.append([value[0], value[1]])
        else:
            norm_skeleton.append([0, 0])
    
    return Skeleton(joints=norm_skeleton)


def _get_redundant_indices(start: int, end: int, segment_pivots: list[int]) -> list[int]:
    """
    Returns a list of redundant indices within the given range.

    Args:
        start (int): The starting index of the range.
        end (int): The ending index of the range.
        segment_pivots (list): A list of indices that are considered as pivots.

    Returns:
        list: A list of redundant indices within the given range.
    """
    segment_indices = set(range(start, end+1))
    segment_pivots = set(segment_pivots)
    diff = segment_indices - segment_pivots
    return list(diff)


def _get_duplicate_indices(array: list[int]) -> list[int]:
    """
    Returns a list of duplicate elements in the given array.

    Args:
        array (list): The input array.

    Returns:
        list: A list of duplicate elements in the array.
    """
    duplicates = set()
    result = []
    for elem in array:
        if elem in duplicates:
            result.append(elem)
        else:
            duplicates.add(elem)
    return result


def _remove_empty(array: list) -> list:
    """
    Removes empty elements from the given array.

    Args:
        array (list): The input array.

    Returns:
        list: The array with empty elements removed.
    """
    for i in range(len(array)):
        if array[i] == "No annotation":
            array[i] = [""]
    
    return array


def _real_size(array: list) -> int:
    """
    Calculates the number of non-empty labels in the given array.

    Parameters:
    array (list): The array containing labels.

    Returns:
    int: The count of non-empty labels in the array.
    """
    count = 0
    for label in array:
        if label != [""]:
            count += 1
    
    return count
