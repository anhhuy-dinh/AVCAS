import cv2
import supervision as sv

# Updated consistent class mapping
CLASS_NAMES = {
    0: 'car',
    1: 'truck',
    2: 'bus',
    3: 'person',
    5: 'motorbike',  # Originally "motorcycle" in YOLO
    6: 'bicycle'
}

def visualize_detections(frame, detections, class_names=None):
    """
    Draw bounding boxes and IDs for detected objects
    
    Args:
        frame: Input image
        detections: Detections from YOLOv11 tracker
        class_names: Dictionary of class names
        
    Returns:
        Annotated frame
    """
    # Use provided class names or fallback to default mapping
    if class_names is None:
        class_names = CLASS_NAMES
    
    for det in detections:
        # Based on provided example: (bbox, None, conf, class_id, None, {'class_name': 'car'})
        if len(det) < 6:
            continue
            
        bbox = det[0]  # First element is bbox
        conf = det[2]  # Third element is confidence
        class_id = det[3]  # Fourth element is class_id
        
        # Check if we have valid data
        if bbox is None or class_id is None:
            continue
            
        # Extract additional info from the dictionary (if available)
        track_id = -1
        if det[5] is not None and isinstance(det[5], dict) and 'id' in det[5]:
            track_id = det[5]['id']
        elif isinstance(det[5], int):
            track_id = det[5]
            
        # Get custom class name if available in the dictionary
        custom_class_name = None
        if det[5] is not None and isinstance(det[5], dict) and 'class_name' in det[5]:
            custom_class_name = det[5]['class_name']
        
        # Get coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get class name (prioritize custom name if available)
        if custom_class_name is not None:
            class_name = custom_class_name
        else:
            class_name = class_names.get(class_id, f"class_{class_id}")
            
            # Map "motorcycle" to "motorbike" for consistency if needed
            if class_name == "motorcycle":
                class_name = "motorbike"
        
        # Create label
        if track_id >= 0:
            label = f"{class_name} ID:{track_id} {conf:.2f}"
        else:
            label = f"{class_name} {conf:.2f}"
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return frame