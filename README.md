[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/mrpoulpe/cubes-and-cylinders)

## YOLOvsCustomCNN
<img align="right" src="https://github.com/elias-utf8/YOLOvsCustomCNN/blob/main/assets/media.gif" width="375px">

The goal of this project was to conduct a supervised learning performance audit comparing YOLO and a custom CNN.
In order to benefit from 3D localisation of elements,  it is necessary to have a RealSense camera.

Dataset is available on [Kaggle](https://www.kaggle.com/datasets/mrpoulpe/cubes-and-cylinders/data).

<br><br><br><br><br><br><br>

## Model architecture
```mermaid
graph TD
    Input["Input Image\n224×224"] --> Conv1

    subgraph Backbone
        Conv1["conv1 (3→16)\nBatchNorm 16\nReLU\nMaxPool 2"]
        Conv1 --> Conv2["conv2 (16→32)\nBatchNorm 32\nReLU\nMaxPool 2"]
        Conv2 --> Conv3["conv3 (32→64)\nBatchNorm 64\nReLU\nMaxPool 2"]
        Conv3 --> Conv4["conv4 (64→128)\nBatchNorm 128\nReLU\nMaxPool 2"]
        Conv4 --> Flatten["Flatten"]
    end

    Flatten --> cls_head
    Flatten --> reg_head

    subgraph Head
        subgraph Classification
            cls_head["cls_head\nLinear\nReLU\nLinear"]
            cls_head --> cls_output["cls_output\nCylindre ou Cube"]
        end

        subgraph Regression
            reg_head["reg_head\nLinear\nReLU\nLinear"]
            reg_head --> reg_output["reg_output\nBounding Box\n(x1,y1,x2,y2)\n× nb classes"]
        end
    end
```

## 3D localisation

Detected objects are located in 3D using an Intel RealSense depth camera and an ArUco marker as world reference. The bounding box center is deprojected to 3D camera coordinates, then transformed to world coordinates via `solvePnP`.

## Software

PyQt6 desktop app with three tabs: **Detection** (real-time inference), **Training** (model training), and **Calibration** (camera calibration via chessboard). All paths and parameters are configured in `config.ini`.

The two main modules are ModelManager, which enables efficient management of model inference, and RealSense, which enables acquisition of the various camera streams via an optimised pipeline.


## Comparisons

### IoU and confusion matrix

| Model | Results |
|:-----:|:-------:|
| **Custom CNN** | <img width="800" alt="Custom CNN results" src="https://github.com/user-attachments/assets/5e5f703c-58b6-4d0b-b5f7-5bc4409369da" /> |
| **YOLO** | <img width="800" alt="YOLO results" src="https://github.com/user-attachments/assets/ef51d0b1-0a91-43b8-abeb-8ef839cdd7d5" /> |

### Classification

| Model | Results |
|:-----:|:-------:|
| **Custom CNN** | <img width="1372" height="388" alt="image" src="https://github.com/user-attachments/assets/d9a474b8-f530-4e84-8bd0-1d3088df2718"/> |
| **YOLO** | <img width="1360" height="345" alt="image" src="https://github.com/user-attachments/assets/f32046cf-1af6-4cf6-9052-58b3ebcaae75" /> |


## Run

```bash
git clone https://github.com/elias-utf8/YOLOvsCustomCNN.git
cd YOLOvsCustomCNN
uv sync
uv run python app.py
```

## Resources

- [Documentation YOLOv11](https://docs.ultralytics.com/models/yolo11/)
- [Tutoriel OpenCV Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Cours Deep Learning](https://melodiedaniel.github.io/deep_learning/)
- [Label Studio](https://labelstud.io/guide/)

## Contributors 
- [Mathieu Jay](https://github.com/Arkww) (CNN development)
- [Anh Tin Nguyen](https://github.com/atnguyen14) (3D localisation)
- [Elias Gauthier](https://github.com/elias-utf8) (Software architecture & GUI)
