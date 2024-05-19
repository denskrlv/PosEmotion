# PosEmotion

Introduction

## Requirements
For this project EiLA (Latin-American) dataset was used. To run scripts succesfully, all data should have the the following structure:
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

## Models
These pre-trained models were used:
| Model      | Resolution |   Params    |
|:-----------|:----------:|:-----------:|
| Alice      |  30        |   New York  |
| Bob        |  25        | Los Angeles |
| Charlie    |  35        |    Chicago  |

## Demo
To see the demo, open and run the <code>main.ipynb</code> file.

## Results

## BibTex

## Acknowledgement
