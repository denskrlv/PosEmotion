# PosEmotion: Emotion Recognition from Skeletal Movements

![](https://github.com/denskrlv/PosEmotion/blob/main/media/logo.png)

The accurate interpretation of emotions is crucial in enhancing various fields, such as assistive technologies and healthcare. People may mask their true emotions, as facial expressions are not always reliable indicators. This study explores the efficacy of using skeletal movements for emotion recognition. The research focuses on two primary questions: the effectiveness of clustering skeletal movements from the EiLA dataset into the seven basic emotions, and the accuracy of different models in predicting emotions based on these movements. The methodology involves extracting frames from video data and using the PoseLandmarker algorithm to obtain normalized 3D coordinates of key skeletal points. The skeletal movements are normalized, truncated for consistency, and converted into feature vectors. These vectors are then clustered and used to train various models to determine their performance in emotion recognition.

## Progress
|Task|Status|
|:---|:----:|
|...|...|
|Added supervised learning (SVM)|✅|
|Changing features type to vectors that represent movements and its lengths|✅|
|Augment the data by rotating and flipping the pictures|✅|
|"Cut" overrepresented classes, so the number of samples will be equal|✅|
|Switch to PoseLandmarker and compute 3D vectors (instead of 2D) with estimated depths|✅|
<<<<<<< HEAD
|Clean the code and add documentation||
=======
|Add quaternion interpolation to skeletons|🟡|
|Clean the code and add documentation|🟡|
>>>>>>> 1706ef091b43fb0e030960a965701d7f58f1aace

## Requirements
Before running the project, install all necessary packages using <code>pip install -r requirements.txt</code>.

For this project EiLA (Latin-American) dataset was used. The data should have the the following structure:
```
│posemotion/
├── assets/
│   ├── annotations
│   │   ├── annotations.csv
│   │   ├── README.md
│   ├── frames
│   │   ├── aJKL0ahn1Dk_19532.jpg
│   │   ├── aJKL0ahn1Dk_19538.jpg
│   │   ├── ......
│   ├── videos
│   │   ├── aJKL0ahn1Dk.mp4
│   │   ├── Bqb2wT_eP_4.mp4
│   │   ├── ......
│   ├── ......
├── models/
│   ├── yolo-face.pt
│   ├── yolo-pose.pt
│......
```

## Pre-trained Models
These pre-trained models were used:
|Model|Input size (pixels)|Keypoints|
|:----|:-----------------:|:-------:|
|[YOLOv8x-pose-p6](https://docs.ultralytics.com/tasks/pose/)|1280|17|
|[PoseLandmarker](https://ai.google.dev/edge/api/mediapipe/java/com/google/mediapipe/tasks/vision/poselandmarker/PoseLandmarker)|256|32|
|[MoveNet](https://www.tensorflow.org/hub/tutorials/movenet)|192|17|

## Demo
To see the demo, open and run the <code>posemotion.ipynb</code> file.

## Results
