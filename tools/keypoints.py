# !/usr/bin/env python3

from IPython.display import display, Image
import cv2


class Keypoints():
    """Class to store and show the keypoints of a person detected in an image.
    """

    def __init__(self, image, keys):
        self.image = image
        self.nose = self._extract_with_index(keys, 0)
        self.left_eye = self._extract_with_index(keys, 1)
        self.right_eye = self._extract_with_index(keys, 2)
        self.left_ear = self._extract_with_index(keys, 3)
        self.right_ear = self._extract_with_index(keys, 4)
        self.left_shoulder = self._extract_with_index(keys, 5)
        self.right_shoulder = self._extract_with_index(keys, 6)
        self.left_elbow = self._extract_with_index(keys, 7)
        self.right_elbow = self._extract_with_index(keys, 8)
        self.left_wrist = self._extract_with_index(keys, 9)
        self.right_wrist = self._extract_with_index(keys, 10)
        self.left_hip = self._extract_with_index(keys, 11)
        self.right_hip = self._extract_with_index(keys, 12)
        self.left_knee = self._extract_with_index(keys, 13)
        self.right_knee = self._extract_with_index(keys, 14)
        self.left_ankle = self._extract_with_index(keys, 15)
        self.right_ankle = self._extract_with_index(keys, 16)

    def __str__(self):
        return (
            "Keypoints:"
            f"\nImage: {self.image}"
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
        img = cv2.imread(self.image)
        self._add_labels(img)          
        cv2.imshow('image', img)
        cv2.waitKey(0)
    
    def draw_ipython(self):
        img = cv2.imread(self.image)
        self._add_labels(img)          
        _, encoded_image = cv2.imencode('.jpg', img)
        encoded_image_bytes = encoded_image.tobytes()
        display(Image(data=encoded_image_bytes))

    def _add_labels(self, img):
        if img is None:
            raise ValueError(f"Unable to read the image!")
        
        for var_name, var_value in vars(self).items():
            if var_name != 'image' and var_value != [0.0, 0.0]:
                cv2.putText(img, str(var_name), (int(var_value[0]), int(var_value[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
    def _extract_with_index(self, keys, index):
        if keys[index] == [0.0, 0.0]:
            return [None, None]
        else:
            return keys[index]
