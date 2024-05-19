import cv2
import mediapipe as mp

from ultralytics import YOLO


def detect_face(target, model, resize=(1280, 720)) -> dict:
    """
    Detects faces on the image and returns the boundary boxes.
    :param target: the path to the image.
    :param model: the path to the AI model.
    :param resize: dimensions of the resized image.
    :return: boundary boxes of the detected faces.
    """
    boxes = dict()
    i = 0

    model = YOLO(model)
    if model.ckpt is None:
        print("Failed to load model.")
        return boxes

    image = cv2.imread(target)
    if image is not None:
        image = cv2.resize(image, resize)
        face_result = model.predict(image, conf=0.40)
        parameters = face_result[0].boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = y2 - y1, x2 - x1
            boxes[str(i)] = [x1, y1, h, w]
            i += 1
    else:
        print("Failed to load image.")

    return boxes


def detect_pose(target, model, resize=(1280, 720)):
    """
    Detects poses on the image and returns the boundary boxes.
    :param target: the path to the image.
    :param model: the path to the AI model.
    :param resize: dimensions of the resized image.
    :return: boundary boxes of the detected faces.
    """
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


print(detect_pose("/Users/deniskrylov/Developer/EiLAPython/assets/frames/Bqb2wT_eP_4_44761.jpg", ""))
