# !/usr/bin/env python3

import numpy as np
import os
import pandas as pd

from scipy.interpolate import UnivariateSpline
from tools.structures import Segment


CORE_DIR = os.path.dirname(os.path.dirname(__file__))


def segment(df: pd.DataFrame) -> list[Segment]:
    """
    Segments the given DataFrame based on changes in the 'Video Tag', 'Clip Id', and 'Person Id' columns.

    Args:
        df (pd.DataFrame): The DataFrame to be segmented.

    Returns:
        list[Segment]: A list of Segment objects representing the segmented portions of the DataFrame.

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
    Normalize the given segment to the target size by either dropping redundant rows or duplicating rows.

    Args:
        segment (Segment): The segment to be normalized.
        target_size (int, optional): The desired size of the normalized segment. Defaults to 10.

    Returns:
        Segment: The normalized segment.

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


def interpolate(df: pd.DataFrame, columns: list, method: str="spline", order: int=3):
    """
    Interpolates missing values in the specified columns of a DataFrame using the specified method.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be interpolated.
        columns (list): A list of column names to interpolate.
        method (str, optional): The interpolation method to use. Defaults to "spline".
        order (int, optional): The order of the spline interpolation. Only applicable if method is "spline". Defaults to 3.

    Returns:
        pd.DataFrame: The DataFrame with missing values interpolated.

    Raises:
        ValueError: If the specified interpolation method is not supported.

    """
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
    Only for internal use. Get the redundant indices between the given start and end range, excluding the segment pivots.

    Args:
        start (int): The start index of the range.
        end (int): The end index of the range.
        segment_pivots (list[int]): The list of segment pivots.

    Returns:
        list[int]: The list of redundant indices.

    """
    segment_indices = set(range(start, end+1))
    segment_pivots = set(segment_pivots)
    diff = segment_indices - segment_pivots
    
    return list(diff)


def _get_duplicate_indices(array: list[int]) -> list[int]:
    """
    Only for internal use. Returns a list of indices of duplicate elements in the given array.

    Args:
        array (list[int]): The input array.

    Returns:
        list[int]: A list of indices of duplicate elements in the array.

    """
    duplicates = set()
    result = []
    for elem in array:
        if elem in duplicates:
            result.append(elem)
        else:
            duplicates.add(elem)

    return result
