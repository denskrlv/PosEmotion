# !/usr/bin/env python3

import cv2
import os

from IPython.display import display, Image


def draw(keypoints, image_path, labels=False, lines=True, inline=False):
    try:
        img = cv2.imread(image_path)
        img_name = os.path.basename(image_path)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    if labels:
        _add_labels(img, keypoints)
    
    if lines:
        _add_lines(img, keypoints)

    if inline:
        _, encoded_image = cv2.imencode('.jpg', img)
        encoded_image_bytes = encoded_image.tobytes()
        display(Image(data=encoded_image_bytes))
    else:
        cv2.imshow(img_name, img)
        cv2.waitKey(0)

def _add_lines(img, keypoints):
    connections = [
        (keypoints.nose, keypoints.left_eye), (keypoints.nose, keypoints.right_eye),
        (keypoints.left_eye, keypoints.left_ear), (keypoints.right_eye, keypoints.right_ear),
        (keypoints.left_shoulder, keypoints.right_shoulder),
        (keypoints.left_shoulder, keypoints.left_elbow), (keypoints.right_shoulder, keypoints.right_elbow),
        (keypoints.left_elbow, keypoints.left_wrist), (keypoints.right_elbow, keypoints.right_wrist),
        (keypoints.left_hip, keypoints.right_hip),
        (keypoints.left_shoulder, keypoints.left_hip), (keypoints.right_shoulder, keypoints.right_hip),
        (keypoints.left_hip, keypoints.left_knee), (keypoints.right_hip, keypoints.right_knee),
        (keypoints.left_knee, keypoints.left_ankle), (keypoints.right_knee, keypoints.right_ankle)
    ]

    for start, end in connections:
        if start != [0, 0] and end != [0, 0]:
            cv2.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 0, 0), 2)

    for _, var_value in vars(keypoints).items():
        if var_value != [0, 0]:
            cv2.circle(img, (int(var_value[0]), int(var_value[1])), 5, (0, 255, 0), -1)

def _add_labels(img, keypoints):
    for var_name, var_value in vars(keypoints).items():
        if var_value != [0, 0]:
            cv2.putText(img, str(var_name), (int(var_value[0]), int(var_value[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
