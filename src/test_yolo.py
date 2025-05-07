from ultralytics import YOLO
import cv2
import time
import supervision as sv
import numpy as np

from utils.yolo_utils import visualize_detections

# Paths
MODEL_PATH = "model/yolov11/yolo11_bdd100k.pt"
VIDEO_PATH = "data/videos/test_3.mp4"
OUTPUT_PATH = "output/videos/test3_yolo_output.mp4"

def test_yolo():
    # Load model
    model = YOLO(MODEL_PATH)

    # Load video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_count = 0
    total_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        # Inference
        results = model(frame, conf=0.5, classes=np.range(1,11))  # Relevant classes
        detections = sv.Detections.from_ultralytics(results[0])

        # Visualize
        frame = visualize_detections(frame, detections)
        out.write(frame)

        # Log time
        frame_time = time.time() - start_time
        total_time += frame_time
        frame_count += 1
        print(f"Frame {frame_count}, Processing Time: {frame_time:.4f}s")

    cap.release()
    out.release()
    print(f"Total Frames: {frame_count}, Total Time: {total_time:.2f}s, Avg FPS: {frame_count/total_time:.2f}")
    print("YOLOv11 object detection test completed.")

if __name__ == "__main__":
    test_yolo()