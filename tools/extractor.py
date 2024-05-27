# !/usr/bin/env python3

import cv2
import os
import numpy as np
import pandas as pd


class Extractor:

    CORE_DIR = os.path.dirname(os.path.dirname(__file__))

    def __init__(self, annotations, extract_from, extract_to):
        self.annotations = os.path.join(self.CORE_DIR, annotations)
        self.extract_from = os.path.join(self.CORE_DIR, extract_from)
        self.extract_to = os.path.join(self.CORE_DIR, extract_to)

    def extract_frames(self):
        if not os.path.exists(self.extract_to):
            os.makedirs(self.extract_to)

        df = pd.read_csv(self.annotations)

        progress = 0
        total = len(df)
        for index, row in df.iterrows():
            video_tag, clip_id, label, frame_no = row['Video Tag'], row['Clip Id'], row['Labels'], row['Frame Number']
            x, y, w, h, person_id = row['X'], row['Y'], row['Width'], row['Height'], row['Person Id']

            frame = self._extract_frame(os.path.join(self.extract_from, video_tag + ".mp4"), frame_no, x, y, w, h)
            cv2.imwrite(os.path.join(self.extract_to, video_tag + "_" + str(index) + ".jpg"), frame)
            progress += 1
            print("Progress: {}/{}".format(progress, total))

    def _extract_frame(self, video_path, frame_no, x, y, w, h):
        cam = cv2.VideoCapture(video_path)
        cam.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        
        ret, frame = cam.read()
        if not ret:
            raise Exception("Error: Couldn't read the frame.")

        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))

        x = int((float(x)/100) * width)
        y = int((float(y)/100) * height)
        w = int((float(w)/100) * width)
        h = int((float(h)/100) * height)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (128,128,128), 2)
        frame = self._mask_frame(frame, x, y, w, h)

        return frame

    def _mask_frame(self, frame, x, y, w, h):
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
        df = pd.read_csv(self.annotations)
        df = df.drop_duplicates(subset=columns)

        return df
