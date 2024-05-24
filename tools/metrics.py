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


def label_probabilities(df, labels_column="Labels", preserve=False):
    probs = []

    for _, row in df.iterrows():
        nested_list = ast.literal_eval(row[labels_column])
        nested_list = _remove_empty(nested_list)
        prob_single = np.zeros(len(Emotions))
        n_label_size = _real_size(nested_list)
        for label in nested_list:
            if label != [""]:
                prob_single[Emotions[label[0]]] += 1 / n_label_size
        probs.append(prob_single)

    probs = np.round(probs, 2)
    probs_df = pd.DataFrame(probs, columns=list(Emotions.keys()))

    if not preserve:
        df = df.drop(columns=[labels_column])

    return pd.concat([df, probs_df], ignore_index=False, axis=1)


def _remove_empty(array):
    for i in range(len(array)):
        if array[i] == "No annotation":
            array[i] = [""]
    return array


def _real_size(array):
    count = 0
    for label in array:
        if label != [""]:
            count += 1
    return count
