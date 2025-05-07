# Install

## Prerequisites

- Python 3.11.11
- PyTorch 2.6.0
- CUDA 12.4
- cudnn 9

## Project Setup

1. Clone the project
    ```Shell
    git clone https://github.com/your-username/AVCAS.git
    cd AVCAS
    ```

2. Create a conda virtual environment and activate it
    ```Shell
    conda create -n avcas python=3.11 -y
    conda activate avcas
    ```

3. Install dependencies
    ```Shell
    # If you don't have PyTorch
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    pip install -r requirements.txt
    ```

4. Download pre-trained models
    ```Shell
    # Create model directories if they don't exist
    mkdir -p model/yolov11
    mkdir -p model/lane_detection
    
    # Option 1: Using the Ultralytics pip package (recommended)
    pip install ultralytics
    
    # This will automatically download the model to ~/.cache/ultralytics/
    # when you first use it in your code
    
    # Option 2: Manual download from Ultralytics GitHub releases
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt -O model/yolov11/yolo11n.pt
    ```
    The pre-trained models used in the project are available here: [GoogleDrive](https://drive.google.com/drive/folders/1O8AhWSGKE4euaFaVJdb6ug4GQgRXTM74?usp=sharing)

## Data Preparation

### BDD100K Dataset

1. Download [BDD100K](https://bdd-data.berkeley.edu/)
    - Create an account and accept the license agreement
    - Download the following files:
      - Images (10K subset): `bdd100k_images_10k.zip`
      - Labels (Detection): `bdd100k_labels_release.zip`

2. Extract the files to the proper directories
    ```Shell
    mkdir -p data/bdd100k/images/10k
    mkdir -p data/bdd100k/labels
    
    # Extract image files
    unzip bdd100k_images_10k.zip -d data/bdd100k/images/10k
    
    # Extract label files
    unzip bdd100k_labels_release.zip -d data/bdd100k/labels
    ```

3. Preprocess the dataset for YOLOv11
    ```Shell
    python preprocess_bdd100k_yolo.py
    ```

### CULane Dataset

1. Download [CULane](https://xingangpan.github.io/projects/CULane.html)
    - Download the following files:
      - Training & Validation Images:
        - `driver_23_30frame.tar.gz`
        - `driver_161_90frame.tar.gz`
        - `driver_182_30frame.tar.gz`
      - Testing Images:
        - `driver_37_30frame.tar.gz`
        - `driver_100_30frame.tar.gz`
        - `driver_193_90frame.tar.gz`
      - Annotations:
        - `annotations_new.tar.gz`
        - `list.tar.gz`

2. Extract the files to the proper directory
    ```Shell
    mkdir -p data/CULane
    
    # Extract all tar.gz files
    tar -xzf driver_23_30frame.tar.gz -C data/CULane/
    tar -xzf driver_161_90frame.tar.gz -C data/CULane/
    tar -xzf driver_182_30frame.tar.gz -C data/CULane/
    tar -xzf driver_37_30frame.tar.gz -C data/CULane/
    tar -xzf driver_100_30frame.tar.gz -C data/CULane/
    tar -xzf driver_193_90frame.tar.gz -C data/CULane/
    tar -xzf annotations_new.tar.gz -C data/CULane/
    tar -xzf list.tar.gz -C data/CULane/
    ```

The directory arrangement of CULane should look like:
```
data/CULane
|──driver_100_30frame
|──driver_161_90frame
|──driver_182_30frame
|──driver_193_90frame
|──driver_23_30frame
|──driver_37_30frame
|──laneseg_label_w16
|──list
```

## Prepare Test Videos

Place your test videos in the `data/videos/` directory:
```Shell
mkdir -p data/videos
# Copy your test videos to this directory
```

## Create Output Directory

Create the output directory for the processed videos:
```Shell
mkdir -p output/videos
```
