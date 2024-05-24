# !/usr/bin/env python3

from IPython.display import display, Image
import cv2


class Keypoints():
    """Class to store and show the keypoints of a person detected in an image.
    """

    def __init__(self, image, keys):
        self.image = image
        self.nose = keys[0]
        self.left_eye = keys[1]
        self.right_eye = keys[2]
        self.left_ear = keys[3]
        self.right_ear = keys[4]
        self.left_shoulder = keys[5]
        self.right_shoulder = keys[6]
        self.left_elbow = keys[7]
        self.right_elbow = keys[8]
        self.left_wrist = keys[9]
        self.right_wrist = keys[10]
        self.left_hip = keys[11]
        self.right_hip = keys[12]
        self.left_knee = keys[13]
        self.right_knee = keys[14]
        self.left_ankle = keys[15]
        self.right_ankle = keys[16]

    def __str__(self):
        return (
            "Keypoints:"
            f"\nNose: {self.nose}"
            f"\nLeft Eye: {self.left_eye}"
            f"\nRight Eye: {self.right_eye}"
            f"\nLeft Ear: {self.left_ear}"
            f"\nRight Ear: {self.right_ear}"
            f"\nLeft Shoulder: {self.left_shoulder}"
            f"\nRight Shoulder: {self.right_shoulder}"
            f"\nLeft Elbow: {self.left_elbow}"
            f"\nRight Elbow: {self.right_elbow}"
            f"\nLeft Wrist: {self.left_wrist}"
            f"\nRight Wrist: {self.right_wrist}"
            f"\nLeft Hip: {self.left_hip}"
            f"\nRight Hip: {self.right_hip}"
            f"\nLeft Knee: {self.left_knee}"
            f"\nRight Knee: {self.right_knee}"
            f"\nLeft Ankle: {self.left_ankle}"
            f"\nRight Ankle: {self.right_ankle}"
        )
    
    def to_dict(self):
        keys = {}
        for var_name, var_value in vars(self).items():
            if var_name != 'image':
                keys[var_name + "_X"] = var_value[0]
                keys[var_name + "_Y"] = var_value[1]
        return keys
    
    def draw(self):
        self._add_labels()          
        cv2.imshow('image', self.image)
        cv2.waitKey(0)
    
    def draw_ipython(self):
        self._add_labels()          
        _, encoded_image = cv2.imencode('.jpg', self.image)
        encoded_image_bytes = encoded_image.tobytes()
        display(Image(data=encoded_image_bytes))

    def _add_labels(self):
        if self.image is None:
            raise ValueError(f"Unable to read the image!")
        
        for var_name, var_value in vars(self).items():
            if var_name != 'image' and var_value != [0.0, 0.0]:
                cv2.putText(self.image, str(var_name), (int(var_value[0]), int(var_value[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
