# !/usr/bin/env python3

import ast
import numpy as np
import pandas as pd
from tools.structures import Segment


Emotions = {
    "Happy"     : 0,
    "Sad"       : 1,
    "Fear"      : 2,
    "Neutral"   : 3,
    "Surprise"  : 4,
    "Disgust"   : 5,
    "Anger"     : 6
}


def label_probabilities(df, labels_column="Labels", preserve=False):
    probs = []

    for _, row in df.iterrows():
        nested_list = ast.literal_eval(row[labels_column])
        nested_list = _remove_empty(nested_list)
        prob_single = np.zeros(len(Emotions))
        n_label_size = _real_size(nested_list)
        for label in nested_list:
            if label != [""]:
                prob_single[Emotions[label[0]]] += 1 / n_label_size
        probs.append(prob_single)

    probs = np.round(probs, 2)
    probs_df = pd.DataFrame(probs, columns=list(Emotions.keys()))

    if not preserve:
        df = df.drop(columns=[labels_column])

    return pd.concat([df, probs_df], ignore_index=False, axis=1)


def segmentate(df):
    segments = []
    base = (None, None, None)
    start_i = 0

    df = df.drop(columns=["X", "Y", "Width", "Height"])

    for index, row in df.iterrows():
        if base == (None, None, None):
            base = (row['Video Tag'], row['Clip Id'], row['Person Id'])
            continue
        if (row['Video Tag'], row['Clip Id'], row['Person Id']) != base:
            base = (row['Video Tag'], row['Clip Id'], row['Person Id'])
            segments.append(Segment(df.loc[start_i:index-1]))
            start_i = index
    
    if start_i < df.shape[0]:
        segments.append(Segment(df.loc[start_i:]))

    return segments


def normalize_segments(df, target_size=10, after="Anger"):
    df = remove_empty_keypoints(df, after)

    segments = segmentate(df)
    rows_inserted = 0

    to_drop = []
    to_duplicate = []
    for start, end in segments:
        length = end - start + 1
        if length < target_size:
            to_duplicate.append((start, end))
        elif length > target_size:
            to_drop.append((start, end))

    for start, end in to_drop:
        segment_pivots = list(np.linspace(start, end, target_size, dtype=int))
        redundant_indices = _get_redundant_indices(start, end, segment_pivots)
        df = df.drop(index=redundant_indices)
        df = df.reset_index(drop=True)

    # for start, end in segments:
    #     length = end - start + 1
    #     segment_pivots = list(np.linspace(start, end, target_size, dtype=int))
    #     if length > target_size:
    #         redundant_indices = _get_redundant_indices(start, end, segment_pivots)
    #         df = df.drop(index=redundant_indices)
    #     elif length < target_size:
            # duplicate_rows = _get_duplicate_indices(segment_pivots)
            # for index in duplicate_rows:
            #     adjusted_index = index + rows_inserted
            #     duplicated_row = df.iloc[[adjusted_index]]
            #     upper = df.iloc[:adjusted_index+1]
            #     lower = df.iloc[adjusted_index+1:]
            #     df = pd.concat([upper, duplicated_row, lower]).reset_index(drop=True)
            #     rows_inserted += 1
    
    return df


def remove_empty_keypoints(df, after="Anger"):
    col_index = df.columns.get_loc(after)
    keypoints_columns = df.columns[col_index + 1:]
    df = df.dropna(subset=keypoints_columns, how="all")
    
    return df


def _get_redundant_indices(start, end, segment_pivots):
    segment_indices = set(range(start, end+1))
    segment_pivots = set(segment_pivots)
    diff = segment_indices - segment_pivots
    return list(diff)


def _get_duplicate_indices(array):
    duplicates = set()
    result = []
    for elem in array:
        if elem in duplicates:
            result.append(elem)
        else:
            duplicates.add(elem)
    return result


def _remove_empty(array):
    for i in range(len(array)):
        if array[i] == "No annotation":
            array[i] = [""]
    
    return array


def _real_size(array):
    count = 0
    for label in array:
        if label != [""]:
            count += 1
    
    return count


if __name__ == "__main__":
    yolo_df = pd.read_csv("/Users/deniskrylov/Developer/PosEmotion/assets/annotations/yolo_annotations.csv")
    print(_get_duplicate_indices([32, 32, 33, 34, 35, 35, 36, 37, 38, 39]))
