# Advanced Vehicle Collision Avoidance System (AVCAS)

A computer vision system for vehicle detection, lane detection, and distance estimation to improve road safety. This is the final project of the course CSCI931 - Deep Learning instructed by Prof. Ashis Biswas @ Univeristy of Colorado Denver.

## Overview

AVCAS combines multiple computer vision components to create a comprehensive driver assistance system:

- Object detection and tracking using YOLOv11
- Lane detection using UltrafastLaneDetector
- Distance estimation using monocular camera techniques
- Trajectory visualization and collision warnings

The system processes video input (dashcam footage) and produces an annotated output with bounding boxes around detected objects, estimated distances, lane markings, and warning indicators.

## Contributors

This project was made possible by the contributions of the following individuals:

- **Anh-Huy Dinh (@anhhuy-dinh)**
- **Tanvir Ahmad (@tanvir23165)**

## Features

- Real-time object detection and tracking (vehicles, pedestrians, cyclists)
- Lane line detection and visualization
- Distance estimation to objects in the scene
- Trajectory visualization for tracked objects
- Visual and textual warnings for close objects
- Performance metrics (FPS, processing times)

## Directory Structure

```
AVCAS/
├── inference.py                  # Main inference script
├── model/
│   ├── yolov11/                  # YOLO models
│   │   └── yolo11_bdd100k.pt     # Fine-tuned YOLO model
│   └── lane_detection/
│       └── culane.pth            # Lane detection model
├── utils/
│   ├── distance_utils.py         # Distance estimation utilities
│   ├── lane_utils.py             # Lane detection utilities
│   ├── vis_utils.py              # Visualization utilities
│   └── yolo_utils.py             # YOLO utilities
├── ultrafastLaneDetector/        # Lane detection model
│   ├── __init__.py
│   ├── backbone.py
│   ├── model.py
│   └── ultrafastLaneDetector.py
├── data/
│   ├── videos/                   # Input videos
│   ├── bdd100k/                  # BDD100K dataset (for training)
│   └── CULane/                   # CULane dataset (for lane detection)
└── output/
    └── videos/                   # Output videos
```

## Installation

Please see [INSTALL.md](INSTALL.md)

## Usage

### Run Inference

```
python run_inference.py
```

This will process the default video (`data/videos/test_1.mp4`) and save the output to `output/videos/output_test_1.mp4`.

### Custom Configuration

You can modify the following parameters in `run_inference.py`:

```python
# Paths
LANE_MODEL_PATH = "model/lane_detection/culane.pth"
YOLO_MODEL_PATH = "model/yolov11/yolo11_bdd100k.pt"
VIDEO_PATH = "data/videos/test_1.mp4"
OUTPUT_PATH = "output/videos/output_test_1.mp4"

# Distance threshold for warnings (meters)
DISTANCE_THRESHOLD = 0.5
```

## Training Your Own Models

### Fine-tuning YOLOv11

1. Prepare the BDD100K dataset:
   ```
   python preprocess_bdd100k_yolo.py
   ```

2. Fine-tune the model:
   ```
   python finetune_yolo.py
   ```

3. Test the fine-tuned model:
   ```
   python test_yolo.py
   ```

### Training UltrafastLaneDetector

For the lane detection model, we modified the original UltrafastLaneDetector implementation with the following enhancements:

1. Clone the original UltrafastLaneDetector repository:
   ```
   git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection.git
   cd Ultra-Fast-Lane-Detection
   ```

2. Create a conda environment for lane detection:
   ```
   conda create -n lane-det python=3.7 -y
   conda activate lane-det
   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
   pip install -r requirements.txt
   ```

3. Apply our modifications to improve accuracy:
   - Increase the griding number to 400 in `configs/culane.py`:
     ```python
     # Change this line
     griding_num = 400  # Originally was 200
     ```
   - Modify the row anchors to use 39 points in `data/constant.py`:
     ```python
     # Modify the row_anchor to include 39 points
     row_anchor = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 
                  150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 
                  200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 
                  250, 255, 260, 265, 270, 275, 280, 285, 287]
     ```

4. Train the model:
   ```
   # Prepare the CULane dataset according to INSTALL.md
   python train.py configs/culane_resnet18.py
   ```

5. Convert the trained model for inference:
   ```
   # After training is complete, copy the model to your AVCAS project
   cp /path/to/trained/model.pth ../AVCAS/model/lane_detection/culane.pth
   ```

These modifications improved the lane detection accuracy, especially for curved lanes and complex road scenarios. The increased griding number (400) provides finer granularity for lane detection, while the increased number of row anchors (39 points instead of 18) helps better detect curved lanes.

## System Components

### 1. Object Detection (YOLOv11)

The system uses YOLOv11 fine-tuned on the BDD100K dataset to detect and track:
- Cars
- Trucks
- Buses
- People
- Motorcycles
- Bicycles

### 2. Lane Detection (UltrafastLaneDetector)

The UltrafastLaneDetector identifies lane markings to:
- Determine lane boundaries
- Create visual lane overlays
- Provide context for vehicle positioning

Our modified implementation includes increased griding number (400) and more row anchors (39 points) for better accuracy.

### 3. Distance Estimation

The system implements a monocular camera distance estimation approach:
- Uses known object dimensions as references
- Applies perspective principles to estimate distances
- Provides real-time distance measurements in meters

## Visualization

The system creates a detailed visualization with:
- Bounding boxes around detected objects
- Color-coded by distance (red for close, green for far)
- Lane markings overlay
- Distance measurements to each object
- Trajectory visualization for tracked objects
- Warning indicators for close objects
- Performance metrics (FPS)

## Acknowledgements

- The YOLOv11 implementation is based on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- The lane detection model is based on [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection), which we modified to improve accuracy by increasing the griding number and row anchors
- The BDD100K dataset is provided by [Berkeley DeepDrive](https://bdd-data.berkeley.edu/)
- The CULane dataset is used for lane detection training
