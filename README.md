# PosEmotion: Emotion Recognition from Skeletal Movements

![](https://github.com/denskrlv/PosEmotion/blob/main/media/logo.png)

The accurate interpretation of emotions is crucial in enhancing various fields, such as assistive technologies and healthcare. People may mask their true emotions, as facial expressions are not always reliable indicators. This study explores the efficacy of using skeletal movements for emotion recog- nition. The research focuses on two primary questions. First, it evaluates the provided labels in the EiLA dataset by clustering skeletal movements into the seven basic emotions. Second, it examines the accuracy of different models in predicting emotions based on these movements. The methodology involves (1) extracting frames from video data, (2) using the PoseLandmarker algorithm to obtain normalized 3D coordinates of key skeletal points, (3) normalizing and truncating skeletal movements for consistency, and (4) converting them into feature vectors. These vectors are then clustered and used to train various models to determine their performance in emotion recognition. The average linkage method proved most effective for cluster- ing skeletal movements into the seven basic emotions. However, qualitative analysis revealed challenges related to overlap and ambiguity in emotion labeling. Among the models evaluated, the Support Vector Machine (SVM) achieved the highest accuracy but exhibited moderate precision and recall, indicating difficulty in handling class imbalances. In contrast, the Random Forest model demonstrated more robust performance with the highest F1- Score, effectively identifying true positive emotions.

## Progress
|Task|Status|
|:---|:----:|
|...|...|
|Added supervised learning (SVM)|✅|
|Changing features type to vectors that represent movements and its lengths|✅|
|Augment the data by rotating and flipping the pictures|✅|
|"Cut" overrepresented classes, so the number of samples will be equal|✅|
|Switch to PoseLandmarker and compute 3D vectors (instead of 2D) with estimated depths|✅|
|Clean the code and add documentation|✅|

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

## Pre-trained Model
|Model|Input size (pixels)|Keypoints|Returns|
|:----|:-----------------:|:-------:|:-----:|
|[PoseLandmarker (Full)](https://ai.google.dev/edge/api/mediapipe/java/com/google/mediapipe/tasks/vision/poselandmarker/PoseLandmarker)|256|33|Normalized $(x,y,z)$ coordinates

## Demo
To see the demo, open and run the <code>posemotion.ipynb</code> file.

## Results
### Clustering
| **Clusters** | **Metric** | **Linkage** | **Silhouette** | **Davies-Bouldin** |
|--------------|------------|-------------|----------------|--------------------|
| 7            | euclidean  | ward        | 0.1507         | 1.5080             |
| 5            | euclidean  | ward        | _0.1663_       | _1.4382_           |
| 3            | euclidean  | ward        | 0.1603         | 1.5024             |
| 7            | euclidean  | average     | _0.5648_       | _0.5397_           |
| 5            | euclidean  | average     | 0.6225         | 0.6528             |
| 3            | euclidean  | average     | 0.7240         | 0.1723             |
| 7            | euclidean  | complete    | _0.5137_       | _0.9460_           |
| 5            | euclidean  | complete    | 0.5270         | 0.9960             |
| 3            | euclidean  | complete    | _0.7309_       | _0.8466_           |


### Classification
| **Model**        | **Accuracy (Mean)**         | **F1-Score (Mean)**        |
|------------------|-----------------------------|----------------------------|
| SVM              | _0.5816 ± 0.0310_           | _0.4672 ± 0.04476_         |
| Random Forest    | 0.5539 ± 0.0563             | 0.5114 ± 0.0608            |
| Neural Network   | 0.5055 ± 0.0670             | 0.4174 ± 0.0300            |
