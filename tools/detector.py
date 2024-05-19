import cv2
import mediapipe as mp

from ultralytics import YOLO


class Detector:

    def __init__(self, directory: str):
        self.directory = directory


    def detect_pose(self, target, model, resize=(1280, 720)):
        mp_draw = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

        image = cv2.imread(target)
        if image is None:
            print("Failed to load image.")
            return None
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = pose.process(img_rgb)

        if result.pose_landmarks:
            mp_draw.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for id, lm in enumerate(result.pose_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            cv2.imshow("Image", image)
            cv2.waitKey(0)
