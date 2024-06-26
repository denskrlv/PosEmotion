# !/usr/bin/env python3

import cv2
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tools.structures import Skeleton


def draw(skeleton: Skeleton, labels: bool=False, lines: bool=True) -> bytes:
    """
    Draw the skeleton on the image and return the encoded image bytes.

    Args:
        skeleton (Skeleton): The skeleton object containing the image and joint information.
        labels (bool, optional): Whether to add labels to the joints. Defaults to False.
        lines (bool, optional): Whether to add lines connecting the joints. Defaults to True.

    Returns:
        bytes: The encoded image bytes.

    Raises:
        Exception: If there is an error while processing the image.

    """
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


def visualize(skeleton: Skeleton, ax: Axes=None, text: bool=False) -> Axes:
    """
    Visualizes a skeleton in a 3D plot.

    Args:
        skeleton (Skeleton): The skeleton object to visualize.
        ax (Axes, optional): The matplotlib Axes object to plot on. If not provided, a new figure and Axes will be created.
        text (bool, optional): Whether to display text labels for each joint. Defaults to False.

    Returns:
        Axes: The matplotlib Axes object containing the plot.

    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    joints = skeleton.joints
    transformed_joints = {key: [x, z, -y] for key, (x, y, z) in joints.items()}
    
    for key in transformed_joints:
        x, y, z = transformed_joints[key]
        ax.scatter(x, y, z, marker='o', s=20)
        if text:
            ax.text(x, y, z, key)
    
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

    for start, end in connections:
        if start in transformed_joints and end in transformed_joints:
            start_pos = transformed_joints[start]
            end_pos = transformed_joints[end]
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], 'r')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.view_init(elev=20., azim=-45)
    
    return ax


def _add_lines(img, joints) -> None:
    """
    Only for internal use. Draw lines and circles on the image based on the given joints.

    Parameters:
    - img: The image on which the lines and circles will be drawn.
    - joints: A dictionary containing the joint positions.

    Returns:
        None

    """
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


def _add_labels(img, joints) -> None:
    """
    Only for internal use. Add labels to the image based on the given joints.

    Args:
        img (numpy.ndarray): The image to add labels to.
        joints (dict): A dictionary containing joint names as keys and their corresponding coordinates as values.

    Returns:
        None

    """
    for var_name, var_value in joints.items():
        if not np.isnan(var_value).any():
            cv2.putText(img, str(var_name), (int(var_value[0]), int(var_value[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
