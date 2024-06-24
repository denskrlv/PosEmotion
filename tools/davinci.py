# !/usr/bin/env python3

import cv2
import numpy as np
import os

from IPython.display import display, Image
from matplotlib import pyplot as plt


def draw(skeleton, labels=False, lines=True):
    try:
        img = cv2.imread(skeleton.image)
        joints = dict()
        
        original_joints = skeleton.joints
        original_height, original_width, _ = img.shape

        for key in original_joints:
            x, y, _ = original_joints[key]
            if not np.isnan(x) and not np.isnan(y):
                joints[key] = [x * original_width, y * original_height]
            else:
                joints[key] = [np.nan, np.nan]
    except Exception as e:
        raise Exception(f"Error: {e}")
    
    if labels:
        _add_labels(img, joints)
    
    if lines:
        _add_lines(img, joints)

    _, encoded_image = cv2.imencode('.jpg', img)
    encoded_image_bytes = encoded_image.tobytes()
    return encoded_image_bytes


def visualize(skeleton, ax=None, text=False):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    joints = skeleton.joints
    
    # Transform coordinates by switching Y and Z and inverting the new Z
    transformed_joints = {key: [x, z, -y] for key, (x, y, z) in joints.items()}
    
    # Plot the transformed joints
    for key in transformed_joints:
        x, y, z = transformed_joints[key]
        ax.scatter(x, y, z, marker='o', s=20)
        if text:
            ax.text(x, y, z, key)  # Add label to each marker
    
    connections = [
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        ("left_eye", "left_ear"),
        ("right_eye", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_hip", "right_hip"),
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle")
    ]
    
    # Plot the connections between the transformed joints
    for start, end in connections:
        if start in transformed_joints and end in transformed_joints:
            start_pos = transformed_joints[start]
            end_pos = transformed_joints[end]
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], 'r')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z')  # Y axis now represents the original Z values
    ax.set_zlabel('Y')  # Z axis now represents the original Y values
    ax.view_init(elev=20., azim=-45)  # Adjust viewing angle for better visualization
    
    return ax


def _add_lines(img, joints):
    connections = [
        (joints["nose"], joints["left_eye"]),
        (joints["nose"], joints["right_eye"]),
        (joints["left_eye"], joints["left_ear"]),
        (joints["right_eye"], joints["right_ear"]),
        (joints["left_shoulder"], joints["right_shoulder"]),
        (joints["left_shoulder"], joints["left_elbow"]),
        (joints["right_shoulder"], joints["right_elbow"]),
        (joints["left_elbow"], joints["left_wrist"]),
        (joints["right_elbow"], joints["right_wrist"]),
        (joints["left_shoulder"], joints["left_hip"]),
        (joints["right_shoulder"], joints["right_hip"]),
        (joints["left_hip"], joints["left_knee"]),
        (joints["right_hip"], joints["right_knee"]),
        (joints["left_hip"], joints["right_hip"]),
        (joints["left_knee"], joints["left_ankle"]),
        (joints["right_knee"], joints["right_ankle"])
    ]

    for start, end in connections:
        if not np.isnan(start).any() and not np.isnan(end).any():
            cv2.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 0, 0), 2)

    for _, var_value in joints.items():
        if not np.isnan(var_value).any():
            cv2.circle(img, (int(var_value[0]), int(var_value[1])), 5, (0, 255, 0), -1)

def _add_labels(img, joints):
    for var_name, var_value in joints.items():
        if not np.isnan(var_value).any():
            cv2.putText(img, str(var_name), (int(var_value[0]), int(var_value[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
