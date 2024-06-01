# !/usr/bin/env python3

import cv2
import numpy as np
import os

from IPython.display import display, Image


def draw(skeleton, labels=False, lines=True, inline=False):
    try:
        img = cv2.imread(skeleton.image)
        img_name = os.path.basename(skeleton.image)
        joints = skeleton.joints
    except Exception as e:
        print(f"Error: {e}")
        return
    
    if labels:
        _add_labels(img, joints)
    
    if lines:
        _add_lines(img, joints)

    if inline:
        _, encoded_image = cv2.imencode('.jpg', img)
        encoded_image_bytes = encoded_image.tobytes()
        display(Image(data=encoded_image_bytes))
    else:
        cv2.imshow(img_name, img)
        cv2.waitKey(0)

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
        (joints["left_knee"], joints["left_ankle"]),
        (joints["right_knee"], joints["right_ankle"])
    ]

    for start, end in connections:
        if start != [np.nan, np.nan] and end != [np.nan, np.nan]:
            cv2.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 0, 0), 2)

    for _, var_value in joints.items():
        if var_value != [np.nan, np.nan]:
            cv2.circle(img, (int(var_value[0]), int(var_value[1])), 5, (0, 255, 0), -1)

def _add_labels(img, joints):
    for var_name, var_value in joints.items():
        if var_value != [np.nan, np.nan]:
            cv2.putText(img, str(var_name), (int(var_value[0]), int(var_value[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
