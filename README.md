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
|Clean the code and add documentation||

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
These pre-trained model was used:
|Model|Input size (pixels)|Keypoints|Returns|
|:----|:-----------------:|:-------:|:-----:|
|[PoseLandmarker](https://ai.google.dev/edge/api/mediapipe/java/com/google/mediapipe/tasks/vision/poselandmarker/PoseLandmarker)|256|33|Normalized $(x,y,z)$ coordinates

## Demo
To see the demo, open and run the <code>posemotion.ipynb</code> file.

## Results
| **Model**         | **Accuracy (Mean)**       | **F1-Score (Mean)**      |
|:-----------------:|:-------------------------:|:------------------------:|
| SVM               | _0.5623 $\pm$ 0.0448_     | _0.4706 $\pm$ 0.0536_    |
| Random Forest     | 0.5438                    | 0.4357                   |
| Neural Network    | 0.4834                    | 0.4692                   |
