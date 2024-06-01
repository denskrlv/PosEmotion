# PosEmotion

Logo

Abstract

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
|Model|Size (pixels)|Params (M)|Keypoints|
|:----|:-----------:|:--------:|:-------:|
|[YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-pose-p6.pt)|1280|99.1|17|
|Pose-ResNet18-Body|256|15|17(?)|

## Demo
To see the demo, open and run the <code>posemotion.ipynb</code> file.

## Results

## BibTex

## Acknowledgement
