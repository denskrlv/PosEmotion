# !/usr/bin/env python3

import cv2
import os
import pandas as pd


class Extractor:

    def __init__(self, annotations, extract_from=None, extract_to=None):
        self.annotations = annotations
        self.extract_from = extract_from
        self.extract_to = extract_to

    def extract_segments(self):
        df = pd.read_csv(self.annotations)
        df = df.drop(columns=["X", "Y", "Width", "Height"])
        
        base = (None, None, None)
        start_i = 0
        segments = []

        for index, row in df.iterrows():
            if base == (None, None, None):
                base = (row['Video Tag'], row['Clip Id'], row['Person Id'])
                continue
            if (row['Video Tag'], row['Clip Id'], row['Person Id']) != base:
                base = (row['Video Tag'], row['Clip Id'], row['Person Id'])
                segments.append((start_i, index-1))
                start_i = index

        return segments

    def extract_frames(self, columns=["Video Tag", "Frame Number", "Person Id"], drop_duplicates=False):
        if not os.path.exists(self.extract_to):
            os.makedirs(self.extract_to)

        if drop_duplicates:
            df = self._remove_duplicates(columns)
        else:
            df = pd.read_csv(self.annotations)

        progress = 0
        total = len(df)
        for _, row in df.iterrows():
            video_tag = row[columns[0]]
            frame_number = row[columns[1]]
            person_id = row[columns[2]]
            frame = self._extract_frame(os.path.join(self.extract_from, video_tag + ".mp4"), frame_number)
            cv2.imwrite(os.path.join(self.extract_to, video_tag + "_" + str(frame_number) + "_" + str(person_id) + ".jpg"), frame)
            progress += 1
            print("Progress: {}/{}".format(progress, total))


    def _extract_frame(self, video_path, frame_number):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        _, frame = cap.read()
        cap.release()

        return frame


    def _remove_duplicates(self, columns):
        df = pd.read_csv(self.annotations)
        df = df.drop_duplicates(subset=columns)

        return df
