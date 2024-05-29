# !/usr/bin/env python3

import ast
import cv2
import math
import numpy as np
import os
import pandas as pd
from PIL import Image
from tools.structures import Keypoints, Segment


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


def prepare_base_skeleton(size=(1280, 720), skeleton_path="assets/base_skeleton.png"):
    background = Image.new('RGBA', size, (0, 0, 0, 255))
    foreground = Image.open(skeleton_path).convert("RGBA")

    bg_width, bg_height = background.size
    fg_width, fg_height = foreground.size
    position = ((bg_width - fg_width) // 2, (bg_height - fg_height) // 2)

    background.paste(foreground, position, foreground)
    return background


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
    num = 0

    columns_to_drop = ["X", "Y", "Width", "Height"]
    for column in columns_to_drop:
        if column in df.columns:
            df = df.drop(columns=column)

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


def normalize_segment(segment, target_size=10, after="Anger"):
    df = remove_empty_keypoints(segment.df, after).reset_index(drop=True)
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


def normalize_skeleton(keypoints, base):
    keypoints = align_with(base, keypoints)
    
    angle, z_best, s1, s2 = _find_optimal_rotation(keypoints, base)
    keypoints = _rotate_around_y(keypoints, angle, z_best, s1, s2)
    
    return keypoints


def align_with(base, points):
    keypoints = points.to_list()
    base_points = base.to_list()
    result_points = []

    src_points = np.array(keypoints, dtype=np.float32)
    dst_points = np.array(base_points, dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
    transformed_keypoints = cv2.transform(np.array([src_points]), M)[0]

    for (x, y) in transformed_keypoints:
        if np.isnan(x):
            x = 0
        if np.isnan(y):
            y = 0
        result_points.append([x, y])

    return Keypoints(image=points.image, keys=result_points)


def euclidean_dist_2d(point1, point2):
    x1, y1, = point1
    x2, y2, = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def euclidean_dist_3d(point1, point2):
    x1, y1, _ = point1
    x2, y2, _ = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _rotate_around_y(keypoints, angle, z_best, s1, s2):
    kl = keypoints.to_list()
    rotated_keypoints = [kl[0]]
    
    for i in range(1, len(kl)):
        if i % 2 == 1:
            if kl[i] != [None, None]:
                left = kl[i] + [s1 * z_best]
                rotated_left = _rotate_around_axis_3d([left], np.radians(angle))
                rotated_keypoints.append(list(rotated_left[0][:2]))
            else:
                rotated_keypoints.append([None, None])
        else:
            if kl[i] != [None, None]:
                right = kl[i] + [s2 * z_best]
                rotated_right = _rotate_around_axis_3d([right], np.radians(angle))
                rotated_keypoints.append(list(rotated_right[0][:2]))
            else:
                rotated_keypoints.append([None, None])
    
    return Keypoints(image=keypoints.image, keys=rotated_keypoints)


def remove_empty_keypoints(df, after="Anger"):
    col_index = df.columns.get_loc(after)
    keypoints_columns = df.columns[col_index + 1:]
    df = df.dropna(subset=keypoints_columns, how="all")
    
    return df


def _determine_rotation_direction(keypoints):
    kl = keypoints.to_list()
    nose_x, _ = kl[0]
    left_eye_x, _ = kl[1]
    right_eye_x, _ = kl[2]
    
    # Calculate the average x position of the eyes
    eye_center_x = (left_eye_x + right_eye_x) / 2
    
    # Determine direction based on nose and eye center position
    if nose_x < eye_center_x:
        return 1
    else:
        return -1


def _find_optimal_rotation(points, base_points):
    angle = 0
    z_best = 0
    rest = 10000000
    prev_rest = 10000000

    points_base = np.array(points.extract_base())
    base_points_base = np.array(base_points.extract_base())

    target_dist = euclidean_dist_2d(base_points_base[0], base_points_base[1])
    side = _determine_rotation_direction(points)
    if side == 1:
        s1, s2 = 1, -1
    else:
        s1, s2 = -1, 1

    for z in range(int(target_dist)):
        left_shoulder_3d = np.append(points_base[0], s1 * z)
        right_shoulder_3d = np.append(points_base[1], s2 * z)
    
        for i in range(90):
            rotated_points = _rotate_around_axis_3d([left_shoulder_3d, right_shoulder_3d], np.radians(i))
            dist = euclidean_dist_3d(rotated_points[0], rotated_points[1])
            rest = abs(target_dist - dist)
            if rest < prev_rest:
                prev_rest = rest
                angle = i
                z_best = z

    return angle, z_best, s1, s2
    

def _rotate_around_axis_3d(points, theta):
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    rotated_points = np.dot(points, rotation_matrix.T)

    return rotated_points


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
