# !/usr/bin/env python3

import cv2
import os
import pandas as pd
from tools.segment import Segment


class Extractor:

    def __init__(self, annotations, extract_from=None, extract_to=None):
        self.annotations = annotations
        self.extract_from = extract_from
        self.extract_to = extract_to

    def segmentate(self):
        """
        Segments the annotations.csv file into ranges of the same "Video Tag", "Clip Id" and "Person Id".
        :return: a list of ranges.
        """
        df = pd.read_csv(self.annotations)
        base = (None, None, None)
        start_i = 0
        segments = []

        for index, row in df.iterrows():
            if base == (None, None, None):
                base = (row['Video Tag'], row['Clip Id'], row['Person Id'])
                continue
            if (row['Video Tag'], row['Clip Id'], row['Person Id']) != base:
                base = (row['Video Tag'], row['Clip Id'], row['Person Id'])
                group = df.iloc[start_i:index-1]
                segment = Segment(name=base, group=group, rows=(start_i, index-1))
                segments.append(segment)
                start_i = index

        return segments

    def extract_frames(self, columns=["Video Tag", "Frame Number"]):
        """
        Goes through the csv file with Video Tag and extracts frames from
        the video according to the Frame Number.
        :param annotations: csv file with "Video Tag" and "Frame Number" columns.
        :param extract_from: directory path to the folder with videos.
        :param extract_to: directory path to the folder where frames will be extracted.
        :param columns: list of columns that are used to find duplicates.
        """
        if not os.path.exists(self.extract_to):
            os.makedirs(self.extract_to)

        df = self._remove_duplicates(columns)

        progress = 0
        total = len(df)
        for _, row in df.iterrows():
            video_tag = row[columns[0]]
            frame_number = row[columns[1]]
            frame = self._extract_frame(os.path.join(self.extract_from, video_tag + ".mp4"), frame_number)
            cv2.imwrite(os.path.join(self.extract_to, video_tag + "_" + str(frame_number) + ".jpg"), frame)
            progress += 1
            print("Progress: {}/{}".format(progress, total))


    def _extract_frame(self, video_path, frame_number):
        """
        Extracts frame from a video.
        :param video_path: a path to the video file.
        :param frame_number: a frame number that needed to be extracted.
        :return: an extracted frame.
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        _, frame = cap.read()
        cap.release()

        return frame


    def _remove_duplicates(self, columns):
        """
        Removes all duplicate frames from annotations.csv file that have the same "Video Tag" and "Frame Number" values.
        :param annotations: csv file that contains "Video Tag" and "Frame Number" columns.
        :return: data frame with no rows with the same "Video Tag" and "Frame Number" values.
        """
        df = pd.read_csv(self.annotations)
        df = df.drop_duplicates(subset=columns)

        return df
