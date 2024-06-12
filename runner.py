import pandas as pd

from tools.detector import PoseLandmarkerDetector


if __name__ == '__main__':
    df = pd.read_csv("assets/annotations/annotations.csv")
    poselandmarker_detector = PoseLandmarkerDetector("models/pose_landmarker_full.task")

    def apply_plm(df, target, output_csv):
        keypoints = poselandmarker_detector.detect_multi(df, target)
        poselandmarker_df = pd.concat(keypoints, axis=1, ignore_index=True).T
        poselandmarker_df.to_csv(output_csv, index=False)

    # Uncomment lines below to apply MoveNet detector to the frames
    apply_plm(
        df,
        "assets/frames",
        "assets/annotations/keypoints_plm.csv"
    )