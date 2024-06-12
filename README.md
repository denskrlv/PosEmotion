# PosEmotion

![](https://github.com/denskrlv/PosEmotion/blob/main/media/LOGO.gif)

Abstract

## Progress
|Task|Status|
|:---|:----:|
|Added supervised learning (SVM)|✅|
|Changing features type to vectors that represent movements and its lengths|✅|
|Switch to PoseLandmarker and compute 3D vectors (instead of 2D) with estimated depths|🟡|
|Augment the data by rotating, flipping and adding noise to pictures|🟡|
|Apply simplification curve while calculating vector features, ensuring consistency in vector sizes||
|Clean the code and add documentation||

## Contents

## Introduction

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

## Acknowledgement
