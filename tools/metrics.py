# !/usr/bin/env python3

import numpy as np
import os
import pandas as pd

from scipy.interpolate import UnivariateSpline
from tools.structures import Skeleton, Segment


CORE_DIR = os.path.dirname(os.path.dirname(__file__))


def segment(df: pd.DataFrame) -> list[Segment]:
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


def interpolate(df, columns, method="spline", order=3):
    if method == "spline":
        for column in columns:
            valid_indices = df[column].notna()
            x = np.arange(len(df)) 
            if np.sum(valid_indices) > order:
                spline = UnivariateSpline(x[valid_indices], df.loc[valid_indices, column], k=order, s=0)
                df[column] = spline(x)
            else:
                print(f"Not enough data to interpolate {column} with spline of order {order}.")
    else:
        raise ValueError(f"Interpolation method {method} is not supported.")
    
    return df


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
