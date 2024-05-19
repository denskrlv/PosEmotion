import pandas as pd
from tools.Segment import Segment
import unittest


class TestMetrics(unittest.TestCase):

    def setUp(self):
        df = pd.read_csv("assets/annotations/annotations.csv")
        df = df.drop(["X", "Y", "Width", "Height"], axis=1)
        grouped = df.groupby(["Video Tag", "Clip Id", "Person Id"])
        self.segment = Segment()
        for name, group in grouped:
            self.segment.name = name
            self.segment.group = group
            break

    def test_probs(self):
        self.assertTrue((self.segment.probs(
            [[["Happy"], ["Happy"], ["Sad"]], [["Happy"], ["Sad"], ["Neutral"]]]
        ) == [0.5, 0.33, 0, 0.17, 0, 0, 0]).all())
        self.assertTrue((self.segment.probs(
            [[["Happy"], ["Sad"], ["Neutral"]], [["Fear"], ["Surprise"], ["Anger"]]]
        ) == [0.17, 0.17, 0.17, 0.17, 0.17, 0, 0.17]).all())
        self.assertTrue((self.segment.probs(
            [[["Sad"], ["Neutral"], "No annotation"], [["Sad"], ["Neutral"], ["Sad"]]]
        ) == [0, 0.6, 0, 0.4, 0, 0, 0]).all())


if __name__ == '__main__':
    unittest.main()
