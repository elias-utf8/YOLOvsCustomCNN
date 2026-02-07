[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/mrpoulpe/cubes-and-cylinders)

## YOLOvsCustomCNN
<img align="right" src="https://github.com/elias-utf8/YOLOvsCustomCNN/blob/main/assets/media.gif" width="350px">

The goal of this project was to conduct a supervised learning performance audit comparing YOLO and a custom CNN.
<br><br><br><br><br><br><br><br><br>

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

## Software

## Resources

- [Documentation YOLOv11](https://docs.ultralytics.com/models/yolo11/)
- [Tutoriel OpenCV Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Cours Deep Learning](https://melodiedaniel.github.io/deep_learning/)
- [Label Studio](https://labelstud.io/guide/)

## Contributors 
- Mathieu Jay (CNN development)
- Anh Tin Nguyen (3D localisation)
- Elias Gauthier (Software architecture & GUI)
