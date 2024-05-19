# !/usr/bin/env python3

import ast
import numpy as np
import pandas as pd


Emotions = {
    "Happy"     : 0,
    "Sad"       : 1,
    "Fear"      : 2,
    "Neutral"   : 3,
    "Surprise"  : 4,
    "Disgust"   : 5,
    "Anger"     : 6
}


class Segment:

    def __init__(self, name=None, group=None, rows=None, labels="Labels"):
        self.name = name
        self.video_tag = name[0]
        self.clip_id = name[1]
        self.person_id = name[2]
        self.group = group
        self.rows = rows
        self.labels = labels
        self.probs = None
        self.speeds = None

    def __str__(self):
        return "Segment {}\nVideo Tag: {}\nClip Id: {}\nPerson Id: {}\nSize: {}\n".format(
            self.rows, self.video_tag, self.clip_id, self.person_id, len(self.group))

    def __len__(self):
        return len(self.group)

    def labels_to_probs(self, unite=False, preserve=False, update=True):
        df = self.group[self.labels]
        data = []
        for val in df:
            nested_list = ast.literal_eval(val)
            data.append(nested_list)

        data = self._remove_empty(data)
        prob_labels = None

        if unite:
            n_data_size = self._real_size_all(data)
            prob_labels = np.zeros(len(Emotions))
            for label in data:
                for l in label:
                    if l != [""]:
                        prob_labels[Emotions[l[0]]] += 1 / n_data_size
        else:
            prob_labels = []
            for label in data:
                prob_single = np.zeros(len(Emotions))
                n_label_size = self._real_size(label)
                for l in label:
                    if l != [""]:
                        prob_single[Emotions[l[0]]] += 1 / n_label_size
                prob_labels.append(prob_single)

        if not preserve:
            self.group = self.group.drop(columns=[self.labels])

        self.probs = np.round(prob_labels, 2)

        if update:
            self._update()

    def movement_speed(self, base_speed=60, update=True):
        """Computes how fast movement was made from the segment. 
        The speed based on base_speed that represents the speed of the video in frames per second.

        Args:
            base_speed (int, optional): frames per second in the video. Defaults to 60.
        """
        df = self.group["Frame Number"]
        start_frame = None
        self.speeds = [0]

        for index, row in df.items():
            if index == self.rows[0]:
                start_frame = row
            else:
                speed = round((row - start_frame) / base_speed, 3)
                self.speeds.append(speed)
                start_frame = row
        
        if update:
            self._update()
    
    def _update(self):
        """Updates the data frame of the segment. Call it after you called any function that computes something in segment.
        """
        if self.probs is not None:           
            columns = list(Emotions.keys())
            for i, col_name in enumerate(columns):
                self.group[col_name] = self.probs[:, i]
        
        if self.speeds is not None:
            self.group["Speed (seconds)"] = self.speeds
    
    def _remove_empty(self, data):
        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[i][j] == "No annotation":
                    data[i][j] = [""]
        return data
    
    def _real_size_all(self, data):
        count = 0
        for label in data:
            for l in label:
                if l != "":
                    count += 1
        return count
    
    def _real_size(self, data):
        count = 0
        for label in data:
            if label != "":
                count += 1
        return count
