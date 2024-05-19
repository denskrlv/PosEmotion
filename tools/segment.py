# !/usr/bin/env python3

import ast
import numpy as np


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

    def __str__(self):
        return "Segment {}\nVideo Tag: {}\nClip Id: {}\nPerson Id: {}\nSize: {}\n".format(
            self.rows, self.video_tag, self.clip_id, self.person_id, len(self.group))

    def __len__(self):
        return len(self.group)

    def labels_to_probs(self, unite=False):
        """
        Returns data structure of probabilistic labels from the array of labels.
        The indices in the output are the same as the labels in the enumerator class Labels.
        """
        df = self.group[self.labels]
        data = []
        for val in df:
            nested_list = ast.literal_eval(val)
            data.append(nested_list)

        data = self._remove_empty(data)
        n_data_size = self._real_size(data)
        prob_labels = np.zeros(len(Emotions))

        for label in data:
            for l in label:
                if l != [""]:
                    prob_labels[Emotions[l[0]]] += 1 / n_data_size

        self.probs = np.round(prob_labels, 2)
        return self.probs
    
    def to_csv(self, path):
        self.group.to_csv(path, index=False)
    
    def _remove_empty(self, data):
        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[i][j] == "No annotation":
                    data[i][j] = [""]
        return data
    
    def _real_size(self, data):
        count = 0
        for label in data:
            for l in label:
                if l != "":
                    count += 1
        return count
