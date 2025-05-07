import cv2
import torch
import time
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from utils.distance_utils import SingleCamDistanceMeasure, RectInfo, get_distances_from_detections
from utils.vis_utils import modified_dashcam_visualization
import sys
import os

# Import the UltrafastLaneDetector
from ultrafastLaneDetector.ultrafastLaneDetector import UltrafastLaneDetector, ModelType

# Paths - updated to be relative to root directory
LANE_MODEL_PATH = "model/lane_detection/culane.pth"
YOLO_MODEL_PATH = "model/yolov11/yolo11_bdd100k.pt"
VIDEO_PATH = "data/videos/test_7.mp4"
OUTPUT_PATH = "output/videos/output_test_7.mp4"

# Distance threshold for warnings (meters)
DISTANCE_THRESHOLD = 0.5

# Consistent class mapping for our system
CONSISTENT_CLASS_NAMES = {
    0: 'car',
    1: 'truck', 
    2: 'bus',
    3: 'person',
    5: 'motorbike',  # Named "motorcycle" in YOLO
    6: 'bicycle'
}

def inference():
    # Load models
    lane_detector = UltrafastLaneDetector(LANE_MODEL_PATH, 
                                         ModelType.TUSIMPLE if 'tusimple' in LANE_MODEL_PATH.lower() else ModelType.CULANE, 
                                         use_gpu=True)

    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    # Get class names from YOLO model
    class_names = yolo_model.names
    print("Class names:", class_names)
    
    # Initialize distance measure object
    distance_measure = SingleCamDistanceMeasure()
    
    # Load video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Create output video writer
    out = cv2.VideoWriter(
        OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    
    # Dictionary to store trajectory points
    trajectory_history = defaultdict(list)
    MAX_HISTORY = 10  # Keep last 10 points for each object

    frame_count = 0
    total_time = 0
    lane_times = []
    yolo_times = []
    distance_times = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Lane detection
        lane_start = time.time()
        lane_frame = lane_detector.detect_lanes(frame)
        lane_times.append(time.time() - lane_start)

        # YOLOv11 detection with tracking
        yolo_start = time.time()
        results = yolo_model.track(
            frame,
            conf=0.5, 
            # Updated classes to match the ones we want to use
            classes=[0, 1, 2, 3, 5, 6],  # car, truck, bus, person, motorcycle, bicycle
            persist=True,
            tracker="bytetrack.yaml"
        )
        yolo_times.append(time.time() - yolo_start)
        
        # Convert detections to the format we need
        detections = []
        
        # Create RectInfo objects for distance calculation
        rect_infos = []
        
        # Update trajectory history for each tracked object
        distance_start = time.time()
        current_ids = set()
        
        for box in results[0].boxes:
            # Print class name for debugging in first frame
            cls_id = int(box.cls.item())
            if frame_count == 1:  
                print(f"Class ID: {cls_id}, Class Name: {results[0].names[cls_id]}")
            
            # Prepare detection data for visualization
            box_data = []
            box_data.append(box.xyxy[0].tolist())  # box (x1, y1, x2, y2)
            box_data.append(None)  # placeholder for keypoints
            box_data.append(float(box.conf.item()))  # confidence
            box_data.append(int(box.cls.item()))  # class id
            
            # Get class name for SingleCamDistanceMeasure
            class_name = class_names.get(cls_id, "unknown")
            
            # Convert YOLO classes to SingleCamDistanceMeasure classes
            # Using our consistent naming
            if class_name in ["car", "truck", "bus", "person", "bicycle"]:
                measure_class = class_name
            elif class_name == "motorcycle":
                # Map motorcycle to motorbike for consistent naming
                measure_class = "motorbike"
            else:
                # For any other classes, assign an appropriate default
                if cls_id in CONSISTENT_CLASS_NAMES:
                    measure_class = CONSISTENT_CLASS_NAMES[cls_id]
                else:
                    # If not in our mapping, use car as default for distance measurement
                    measure_class = "car"
            
            # Process tracking info
            track_id = -1
            if box.id is not None:
                track_id = int(box.id.item())
                box_data.append(None)  # placeholder
                box_data.append(track_id)  # track id
                current_ids.add(track_id)
                
                # Get bounding box for trajectory
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Add RectInfo for distance measurement - now with track_id
                rect_info = RectInfo(x1, y1, x2, y2, label=measure_class, track_id=track_id)
                rect_infos.append(rect_info)
                
                # Calculate centroid for trajectory
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                trajectory_history[track_id].append((centroid_x, centroid_y))
                
                # Limit history length
                if len(trajectory_history[track_id]) > MAX_HISTORY:
                    trajectory_history[track_id].pop(0)
            
            detections.append(box_data)
        
        # Remove inactive trajectories
        trajectory_history = defaultdict(list, {k: v for k, v in trajectory_history.items() if k in current_ids})
        
        # Update distances using SingleCamDistanceMeasure
        distance_measure.updateDistance(rect_infos)
        
        # Get distance points for visualization
        distance_points = distance_measure.distance_points
        
        # Convert to format needed for warnings
        # Now using the improved distance_by_track_id dictionary
        distances = []
        for track_id in current_ids:
            # Get distance directly from the distance measure
            distance = distance_measure.getDistanceForTrackId(track_id)
            if distance is not None:
                distances.append((track_id, distance))
            else:
                # Fallback - find the closest distance point to this box
                for box_index, box in enumerate(results[0].boxes):
                    if box.id is not None and int(box.id.item()) == track_id:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        bottom_center_x = (x1 + x2) // 2
                        
                        # Find closest distance point to this box
                        closest_distance = None
                        min_distance = float('inf')
                        
                        for dp_x, dp_y, distance in distance_points:
                            dist = abs(dp_x - bottom_center_x)
                            if dist < min_distance:
                                min_distance = dist
                                closest_distance = distance
                        
                        if closest_distance is not None:
                            distances.append((track_id, closest_distance))
                        else:
                            # If still no distance, use a fallback based on position
                            normalized_y = min(1.0, max(0.1, y2 / height))
                            fallback_distance = 50 * (1 - normalized_y)
                            distances.append((track_id, fallback_distance))
        
        distance_times.append(time.time() - distance_start)
        
        # Calculate current FPS
        frame_fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 30.0
        
        # Use modified dashcam visualization with SingleCamDistanceMeasure
        # Now including the grid visualization flag and type
        result_frame = modified_dashcam_visualization(
            frame, 
            lane_frame, 
            detections, 
            distances, 
            trajectory_history,
            distance_points,
            DISTANCE_THRESHOLD,
            CONSISTENT_CLASS_NAMES,  # Using our consistent class names
            frame_fps,
        )
        
        # Write to output
        out.write(result_frame)

        # Log time
        frame_time = time.time() - start_time
        total_time += frame_time
        frame_count += 1
        print(f"Frame {frame_count}, Total Time: {frame_time:.4f}s, "
              f"Lane: {lane_times[-1]:.4f}s, YOLO: {yolo_times[-1]:.4f}s, "
              f"Distance: {distance_times[-1]:.4f}s, FPS: {1/frame_time:.2f}")

    cap.release()
    out.release()

    # Log summary
    print(f"\nSummary:")
    print(f"Total Frames: {frame_count}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average FPS: {frame_count/total_time:.2f}")
    print(f"Average Lane Detection Time: {np.mean(lane_times):.4f}s")
    print(f"Average YOLO Detection Time: {np.mean(yolo_times):.4f}s")
    print(f"Average Distance Estimation Time: {np.mean(distance_times):.4f}s")
    print(f"Output saved to {OUTPUT_PATH}")
    print("Inference completed.")

if __name__ == "__main__":
    inference()