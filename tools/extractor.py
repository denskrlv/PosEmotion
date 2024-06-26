# !/usr/bin/env python3

import cv2
import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


CORE_DIR = os.path.dirname(os.path.dirname(__file__))


class Extractor:
    """
    A class that provides methods for extracting frames from videos and performing image transformations.

    Args:
        annotations (str): The path to the annotations file.
        extract_from (str): The directory containing the videos to extract frames from.
        extract_to (str): The directory to save the extracted frames to.

    Attributes:
        annotations (str): The path to the annotations file.
        extract_from (str): The directory containing the videos to extract frames from.
        extract_to (str): The directory to save the extracted frames to.
    
    """

    def __init__(self, annotations: str, extract_from: str, extract_to: str) -> None:
        self.annotations = os.path.join(CORE_DIR, annotations)
        self.extract_from = os.path.join(CORE_DIR, extract_from)
        self.extract_to = os.path.join(CORE_DIR, extract_to)

    def flip_frames(self, target: str, axis: int=0) -> None:
        """
        Flip frames horizontally or vertically and save them to the target directory.

        Args:
            target (str): The directory to save the flipped frames to.
            axis (int, optional): The axis along which to flip the frames. 
                0 for flipping vertically (default), 1 for flipping horizontally.

        Raises:
            Exception: If the extracted frames directory does not exist.

        Note:
            This method uses OpenCV's `cv2.flip()` function to flip the frames.

        """
        target = os.path.join(CORE_DIR, target)

        if not os.path.exists(self.extract_to):
            raise Exception("Error: Extracted frames not found.")
        
        if not os.path.exists(target):
            os.makedirs(target)

        jpg_files = glob.glob(os.path.join(self.extract_to, "*.jpg"))

        for file_path in jpg_files:
            frame = cv2.imread(file_path)
            frame = cv2.flip(frame, axis)
            cv2.imwrite(os.path.join(target, os.path.basename(file_path)), frame)

    def random_rotate_frames(self, target: str, minr: int=30, maxr: int=330) -> None:
        """
        Randomly rotate frames within a specified range and save them to the target directory.

        Args:
            target (str): The directory to save the rotated frames to.
            minr (int, optional): The minimum rotation angle in degrees (default: 30).
            maxr (int, optional): The maximum rotation angle in degrees (default: 330).

        Raises:
            Exception: If the extracted frames directory does not exist.
            Exception: If the rotation range is invalid.

        Note:
            This method uses OpenCV's `cv2.getRotationMatrix2D()` and `cv2.warpAffine()` functions to rotate the frames.

        """
        target = os.path.join(CORE_DIR, target)

        if not os.path.exists(self.extract_to):
            raise Exception("Error: Extracted frames not found.")
        
        if not os.path.exists(target):
            os.makedirs(target)

        if minr < 0 or maxr > 360 or minr > maxr:
            raise Exception("Error: Invalid rotation range. Allowed rotation range is [0, 360].")

        jpg_files = glob.glob(os.path.join(self.extract_to, "*.jpg"))

        for file_path in jpg_files:
            frame = cv2.imread(file_path)
            angle = np.random.randint(minr, maxr)
            frame = self._rotate_frame(frame, angle)
            cv2.imwrite(os.path.join(target, os.path.basename(file_path)), frame)

    def extract_frames(self) -> None:
        """
        Extract frames from videos based on the annotations file and save them to the extract_to directory.

        Raises:
            Exception: If the extract_to directory does not exist.

        Note:
            This method uses OpenCV's `cv2.VideoCapture()` function to read the videos and extract frames.

        """
        if not os.path.exists(self.extract_to):
            os.makedirs(self.extract_to)

        df = pd.read_csv(self.annotations)

        total = len(df)
        for index, row in tqdm(df.iterrows(), total=total, desc="Progress"):
            video_tag, frame_no = row['Video Tag'], row['Frame Number']
            x, y, w, h = row['X'], row['Y'], row['Width'], row['Height']
            x, y, w, h = float(x), float(y), float(w), float(h)

            frame = self._extract_frame(os.path.join(self.extract_from, f"{video_tag}.mp4"), int(frame_no), x, y, w, h)
            cv2.imwrite(os.path.join(self.extract_to, f"{video_tag}_{index}.jpg"), frame)

    def _extract_frame(self, video_path: str, frame_no: int, x: float, y: float, w: float, h: float) -> np.ndarray:
        """
        Only for internal use. Extract a single frame from a video based on the frame number and bounding box coordinates.

        Args:
            video_path (str): The path to the video file.
            frame_no (int): The frame number to extract.
            x (float): The x-coordinate of the top-left corner of the bounding box.
            y (float): The y-coordinate of the top-left corner of the bounding box.
            w (float): The width of the bounding box.
            h (float): The height of the bounding box.

        Returns:
            np.ndarray: The extracted frame as a NumPy array.

        Raises:
            Exception: If the frame cannot be read from the video.

        Note:
            This method uses OpenCV's `cv2.VideoCapture()` function to read the video and extract the frame.

        """
        cam = cv2.VideoCapture(video_path)
        cam.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        
        ret, frame = cam.read()
        if not ret:
            raise Exception("Error: Couldn't read the frame.")

        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))

        x = int((x / 100) * width)
        y = int((y / 100) * height)
        w = int((w / 100) * width)
        h = int((h / 100) * height)

        return frame[y:y + h, x:x + w]
    
    def _rotate_frame(self, frame, angle):
        """
        Only for internal use. Rotate a frame by a specified angle.

        Args:
            frame (np.ndarray): The frame to rotate.
            angle (int): The rotation angle in degrees.

        Returns:
            np.ndarray: The rotated frame as a NumPy array.

        Note:
            This method uses OpenCV's `cv2.getRotationMatrix2D()` and `cv2.warpAffine()` functions to rotate the frame.

        """
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        return cv2.warpAffine(frame, rotation_matrix, (width, height))

    def _mask_frame(self, frame, x, y, w, h):
        """
        Only for internal use. Apply a mask to a frame based on the specified bounding box coordinates.

        Args:
            frame (np.ndarray): The frame to mask.
            x (int): The x-coordinate of the top-left corner of the bounding box.
            y (int): The y-coordinate of the top-left corner of the bounding box.
            w (int): The width of the bounding box.
            h (int): The height of the bounding box.

        Returns:
            np.ndarray: The masked frame as a NumPy array.

        Note:
            This method uses NumPy's array indexing and bitwise operations to apply the mask.

        """
        mask = np.zeros(frame.shape[:2], dtype="uint8")

        left, upper, right, lower = x, y, x + w, y + h
        mask[upper:lower, left:right] = 255

        black_background = np.zeros_like(frame)

        result_frame = cv2.bitwise_and(frame, frame, mask=mask)
        black_background = cv2.bitwise_and(black_background, black_background, mask=cv2.bitwise_not(mask))
        masked_frame = cv2.add(result_frame, black_background)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (128,128,128), 2)

        return masked_frame

    def _remove_duplicates(self, columns):
        """
        Only for internal use. Remove duplicate rows from the annotations DataFrame based on the specified columns.

        Args:
            columns (list): The columns to consider when checking for duplicates.

        Returns:
            pd.DataFrame: The DataFrame with duplicate rows removed.

        Note:
            This method uses pandas' `drop_duplicates()` function to remove duplicate rows.

        """
        df = pd.read_csv(self.annotations)
        df = df.drop_duplicates(subset=columns)

        return df
