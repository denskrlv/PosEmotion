import cv2
from pydantic import BaseModel
from ultralytics import YOLO


class Keypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16


keys = Keypoint()
model = YOLO('/Users/deniskrylov/Developer/PosEmotion/models/yolo-pose.pt')

img = cv2.imread('/Users/deniskrylov/Developer/PosEmotion/assets/frames/aJKL0ahn1Dk_19532.jpg')
results = model('/Users/deniskrylov/Developer/PosEmotion/assets/frames/aJKL0ahn1Dk_19532.jpg')[0]

for result in results:
    print(result.keypoints.xy.numpy().tolist()[0])
    for keypoint_idx, keypoint in enumerate(result.keypoints.xy.numpy().tolist()[0]):
        cv2.putText(img, str(keypoint_idx), (int(keypoint[0]), int(keypoint[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow('image', img)
cv2.waitKey(0)
