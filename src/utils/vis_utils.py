import cv2
import numpy as np
from collections import defaultdict
import time

# Aesthetically pleasing color scheme
COLORS = {
    'lane': (0, 255, 0),         # Green
    'trajectory': (255, 255, 0),  # Yellow
    'bounding_box': {
        'car': (255, 0, 0),       # Blue
        'truck': (255, 0, 127),   # Purple
        'person': (0, 165, 255),  # Orange
        'bicycle': (0, 255, 255), # Yellow
        'motorbike': (0, 255, 255), # Yellow (same as bicycle)
        'bus': (255, 0, 127),     # Purple (same as truck)
    },
    'warning': (0, 0, 255),       # Red
    'safe': (0, 255, 0),          # Green
    'text_outline': (0, 0, 0)     # Black
}

# Updated consistent class name mapping
CLASS_NAMES = {
    0: 'car',
    1: 'truck',
    2: 'bus',
    3: 'person',
    5: 'motorbike',  # Originally "motorcycle" in YOLO
    6: 'bicycle'
}

def draw_tracking_lines(frame, detections, trajectory_history=None, fade_effect=True, vehicle_distances=True):
    """
    Draw trajectory lines for tracked objects with fade effect
    
    Args:
        frame: Input image
        detections: Detections from YOLOv11 tracker
        trajectory_history: Dictionary of trajectory points per object ID
        fade_effect: Whether to apply fading effect to trajectory lines
        vehicle_distances: Whether to show distances between vehicles
        
    Returns:
        Annotated frame
    """
    if trajectory_history is None:
        # Simplified; draw center point of each detection
        for det in detections:
            if len(det) > 0 and det[0] is not None:
                x1, y1, x2, y2 = det[0]
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                cv2.circle(frame, center, 5, COLORS['trajectory'], -1)
    else:
        # Get current positions of vehicles
        vehicle_positions = {}
        vehicle_boxes = {}
        
        # First pass - collect positions
        for det in detections:
            if len(det) <= 4 or det[4] is None:
                continue
                
            try:
                cls = int(det[4])
            except (TypeError, ValueError):
                continue
                
            # Update to include bus class (2)
            if cls in [0, 1, 2]:  # For cars, trucks, buses
                track_id = -1
                if len(det) > 5:
                    if isinstance(det[5], int):
                        track_id = det[5]
                    elif isinstance(det[5], dict) and 'id' in det[5]:
                        track_id = det[5]['id']
                
                if track_id >= 0 and det[0] is not None:
                    x1, y1, x2, y2 = [int(v) for v in det[0]]
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    vehicle_positions[track_id] = center
                    vehicle_boxes[track_id] = (x1, y1, x2, y2)
        
        # Draw trajectories from history with fade effect
        for track_id, points in trajectory_history.items():
            if len(points) > 1:
                for i in range(1, len(points)):
                    # Apply fade effect - more recent lines are brighter
                    if fade_effect:
                        alpha = 0.3 + 0.7 * (i / len(points))
                        color = tuple([int(c * alpha) for c in COLORS['trajectory']])
                    else:
                        color = COLORS['trajectory']
                    
                    # Draw thicker lines for better visibility
                    cv2.line(
                        frame,
                        points[i - 1],
                        points[i],
                        color=color,
                        thickness=2
                    )
                    
                # Draw a circle at the current position
                if points:
                    cv2.circle(frame, points[-1], 5, COLORS['trajectory'], -1)
        
        # Draw distances between vehicles
        if vehicle_distances and len(vehicle_positions) > 1:
            # Create list of vehicle ids
            vehicle_ids = list(vehicle_positions.keys())
            
            # Calculate and draw distances between vehicles
            for i in range(len(vehicle_ids)):
                for j in range(i+1, len(vehicle_ids)):
                    id1, id2 = vehicle_ids[i], vehicle_ids[j]
                    pos1, pos2 = vehicle_positions[id1], vehicle_positions[id2]
                    
                    # Calculate pixel distance between centers
                    pixel_dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    # Skip if too far apart (clutters the display)
                    # Increased from 300 to 500 to show more distances
                    if pixel_dist > 500:
                        continue
                    
                    # Calculate midpoint for text placement
                    mid_x = (pos1[0] + pos2[0]) // 2
                    mid_y = (pos1[1] + pos2[1]) // 2
                    
                    # Draw line between centers
                    cv2.line(frame, pos1, pos2, (180, 180, 180), 1, cv2.LINE_AA)
                    
                    # Estimate real-world distance using bounding box width
                    # This is an approximation based on the average size of vehicles
                    box1, box2 = vehicle_boxes[id1], vehicle_boxes[id2]
                    width1 = box1[2] - box1[0]
                    width2 = box2[2] - box2[0]
                    
                    # Use average of both vehicles to estimate distance
                    avg_width = (width1 + width2) / 2
                    avg_real_width = 1.8  # meters
                    
                    # Use the pixel distance and estimated scale to calculate real-world meters
                    estimated_distance = pixel_dist * (avg_real_width / avg_width)
                    
                    # Draw distance text with background for better visibility
                    distance_text = f"{estimated_distance:.1f}m"
                    
                    # Get text size
                    (text_width, text_height), _ = cv2.getTextSize(
                        distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    
                    # Draw background rectangle
                    cv2.rectangle(
                        frame,
                        (mid_x - text_width//2 - 5, mid_y - text_height//2 - 5),
                        (mid_x + text_width//2 + 5, mid_y + text_height//2 + 5),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        frame,
                        distance_text,
                        (mid_x - text_width//2, mid_y + text_height//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )
    
    return frame

def draw_warnings(frame, distances, threshold, show_all_distances=False):
    """
    Draw stylish warning messages for objects that are too close
    
    Args:
        frame: Input image
        distances: List of (track_id, distance) tuples
        threshold: Distance threshold for warnings (meters)
        show_all_distances: Whether to show distances for all objects
        
    Returns:
        Annotated frame
    """
    # Create a semi-transparent overlay for warnings at the top
    overlay = frame.copy()
    warning_height = 80 if len(distances) > 0 else 0
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], warning_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # Display current time stamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (frame.shape[1] - 200, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add warning/safe indicator
    warning_objects = [d for d in distances if d[1] < threshold]
    if warning_objects:
        cv2.putText(frame, "WARNING", (20, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['warning'], 2)
    else:
        cv2.putText(frame, "SAFE", (20, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['safe'], 2)
    
    # Display warning messages
    warning_y = 60
    warning_count = 0
    
    # Sort distances to show closest objects first
    sorted_distances = sorted(distances, key=lambda x: x[1])
    
    for tracker_id, distance in sorted_distances:
        if distance < threshold or show_all_distances:
            # Color based on distance (red for close, green for far)
            if distance < threshold:
                color = COLORS['warning']
                prefix = "WARN:"
            else:
                color = COLORS['safe']
                prefix = "DIST:"
            
            warning_text = f"{prefix} ID {int(tracker_id)}: {distance:.1f}m"
            
            # Draw text with black outline for better visibility
            x_pos = 20 + (warning_count % 3) * 300
            y_pos = warning_y + (warning_count // 3) * 40
            
            # Text outline
            cv2.putText(frame, warning_text, (x_pos-1, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text_outline'], 3)
            # Text
            cv2.putText(frame, warning_text, (x_pos, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            warning_count += 1
    
    return frame

def visualize_detections(frame, detections, class_names=None):
    """
    Draw enhanced bounding boxes and IDs for detected objects
    
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
        if len(det) < 4:
            continue
            
        bbox = det[0]  # First element is bbox
        conf = det[2] if len(det) > 2 and det[2] is not None else 0.0  # Third element is confidence
        cls = det[3] if len(det) > 3 and det[3] is not None else None  # Fourth element is class_id
        
        # Check if we have valid data
        if bbox is None or cls is None:
            continue
            
        # Extract track_id properly
        track_id = -1
        if len(det) > 5 and det[5] is not None:
            if isinstance(det[5], int):
                track_id = det[5]
            elif isinstance(det[5], dict) and 'id' in det[5]:
                track_id = det[5]['id']
            
        # Get coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get class name (prioritize custom name if available)
        class_idx = int(cls)
        class_name = class_names.get(class_idx, f"class_{class_idx}")
        
        # Map "motorcycle" to "motorbike" for consistency if needed
        if class_name == "motorcycle":
            class_name = "motorbike"
        
        # Create label
        if track_id >= 0:
            label = f"{class_name} ID:{track_id} {conf:.2f}"
        else:
            label = f"{class_name} {conf:.2f}"
        
        # Determine color based on class
        color = COLORS['bounding_box'].get(class_name.lower(), (255, 0, 0))
        
        # Draw filled rectangle for text background
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        cv2.rectangle(
            frame, 
            (x1, y1 - text_height - 10), 
            (x1 + text_width + 10, y1), 
            color, 
            -1
        )
        
        # Draw white text for better visibility
        cv2.putText(
            frame, 
            label, 
            (x1 + 5, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            2
        )
        
        # Draw bounding box with thicker line
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
    return frame

def modified_dashcam_visualization(frame, lane_frame, detections, distances, trajectory_history, distance_points, threshold=15.0, class_names=None, frame_fps=15.0):
    """
    Create a modified dashcam visualization with:
    - Lane detection overlay (less transparent)
    - Object detection with distances using SingleCamDistanceMeasure
    - Simple safe/warning indicator
    - Trajectory lines
    - Corner frames with class colors
    - Label background with class colors
    
    Args:
        frame: Original input frame
        lane_frame: Lane detection visualization frame
        detections: Object detections
        distances: List of (track_id, distance) tuples
        trajectory_history: Dictionary of trajectory points
        distance_points: List of [x, y, distance] points from SingleCamDistanceMeasure
        threshold: Warning threshold distance
        class_names: Dictionary of class names
        frame_fps: Current FPS for display
        
    Returns:
        Modified dashcam visualization
    """
    try:
        from utils.distance_utils import putText_shadow
        have_shadow_text = True
    except ImportError:
        have_shadow_text = False
    
    # Create a copy of the frame
    output = frame.copy()
    frame_height, frame_width = output.shape[:2]
    
    # Overlay lane detection with less transparency
    if lane_frame is not None:
        lane_resized = cv2.resize(lane_frame, (frame_width, frame_height))
        output = cv2.addWeighted(output, 0.5, lane_resized, 0.5, 0)  # More visible lanes (50/50 blend)
    
    # Use provided class names or fallback to default mapping
    if class_names is None:
        class_names = CLASS_NAMES
    
    # Process each detection
    for det in detections:
        # Extract bbox and class
        if len(det) < 4 or det[0] is None or det[3] is None:
            continue
            
        bbox = det[0]  # First element is bbox
        conf = det[2] if len(det) > 2 and det[2] is not None else 0.0  # Confidence
        try:
            class_id = int(det[3])  # Fourth element is class_id
        except (TypeError, ValueError):
            continue
        
        # Extract track ID if available
        track_id = -1
        if len(det) > 5 and det[5] is not None:
            if isinstance(det[5], dict) and 'id' in det[5]:
                track_id = det[5]['id']
            elif isinstance(det[5], int):
                track_id = det[5]
                
        # Get class name
        class_name = class_names.get(class_id, f"class_{class_id}")
        
        # Map "motorcycle" to "motorbike" for consistency if needed
        if class_name == "motorcycle":
            class_name = "motorbike"
        
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # First get default color based on class
        class_color = COLORS['bounding_box'].get(class_name.lower(), (255, 155, 0))  # Default orange-ish if class not found
        
        # Then determine distance-based color if available
        object_distance = None
        for id_, distance in distances:
            if id_ == track_id:
                object_distance = distance
                break
                
        # Set color based on distance, ONLY if we have distance info
        if object_distance is not None:
            if object_distance < threshold:
                distance_color = (0, 0, 255)  # Red for close objects
            elif object_distance < threshold * 1.5:
                distance_color = (0, 165, 255)  # Orange for medium distance
            else:
                distance_color = (0, 255, 0)  # Green for far objects
        else:
            # If no distance info, use class color instead of white
            distance_color = class_color
            
        # Create semi-transparent box with distance-based color
        overlay = output.copy()
        alpha = 0.3  # Transparency factor
        cv2.rectangle(overlay, (x1, y1), (x2, y2), distance_color, -1)
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
        
        # Draw solid border with distance-based color
        cv2.rectangle(output, (x1, y1), (x2, y2), distance_color, 2)
        
        # Add corner frames with class color
        corner_length = int(min(x2-x1, y2-y1) * 0.2)  # 20% of the box size
        thickness = 3  # Thickness of the corner lines
        
        # Draw corners with class color
        # Top-left corner
        cv2.line(output, (x1, y1), (x1 + corner_length, y1), class_color, thickness)
        cv2.line(output, (x1, y1), (x1, y1 + corner_length), class_color, thickness)
        
        # Top-right corner
        cv2.line(output, (x2, y1), (x2 - corner_length, y1), class_color, thickness)
        cv2.line(output, (x2, y1), (x2, y1 + corner_length), class_color, thickness)
        
        # Bottom-left corner
        cv2.line(output, (x1, y2), (x1 + corner_length, y2), class_color, thickness)
        cv2.line(output, (x1, y2), (x1, y2 - corner_length), class_color, thickness)
        
        # Bottom-right corner
        cv2.line(output, (x2, y2), (x2 - corner_length, y2), class_color, thickness)
        cv2.line(output, (x2, y2), (x2, y2 - corner_length), class_color, thickness)
        
        # Create label with class and ID
        if track_id >= 0:
            label = f"{class_name} ID:{track_id}"
        else:
            label = f"{class_name}"
            
        # Draw label above the box
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = max(y1 - 10, label_size[1])
        
        # Draw label background with class color instead of distance color
        cv2.rectangle(output, (x1, label_y - label_size[1]), 
                    (x1 + label_size[0], label_y + 5), class_color, -1)
        
        # Draw label text
        cv2.putText(output, label, (x1, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw trajectory lines
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # If tracking history is available, draw trajectory
        if track_id in trajectory_history and len(trajectory_history[track_id]) > 1:
            points = trajectory_history[track_id]
            for i in range(1, len(points)):
                # Fade older points
                alpha = 0.3 + 0.7 * (i / len(points))
                line_color = tuple([int(c * alpha) for c in distance_color])
                
                # Draw line segments
                cv2.line(
                    output,
                    points[i-1],
                    points[i],
                    line_color,
                    2
                )
            
            # Draw circle at current position
            cv2.circle(output, points[-1], 5, distance_color, -1)
    
    # Draw distance measurements using SingleCamDistanceMeasure visualization
    if distance_points:
        for x, y, d in distance_points:
            unit = 'm'
            if d < 0:
                text = ' {} {}'.format("unknown", unit)
            else:
                text = ' {:.2f} {}'.format(d, unit)
            
            fontScale = max(0.4, min(1, 1/d if d > 0 else 1))
            # get coords based on boundary
            textsize = cv2.getTextSize(text, 0, fontScale=fontScale, thickness=3)[0]
            textX = int((x - textsize[0]/2))
            textY = int((y + textsize[1]))
            
            # Draw distance text with shadow
            if have_shadow_text:
                try:
                    putText_shadow(output, text, (textX + 1, textY + 5), 
                               fontFace=cv2.FONT_HERSHEY_TRIPLEX,  fontScale=fontScale,  
                               color=(255, 255, 255), thickness=1, shadow_color=(150, 150, 150))
                except:
                    have_shadow_text = False  # Fallback if it fails
            
            # Fallback if putText_shadow is not available
            if not have_shadow_text:
                cv2.putText(output, text, (textX + 1, textY + 5), 
                            cv2.FONT_HERSHEY_TRIPLEX, fontScale, (0, 0, 0), 3)  # Shadow
                cv2.putText(output, text, (textX + 1, textY + 5), 
                            cv2.FONT_HERSHEY_TRIPLEX, fontScale, (255, 255, 255), 1)  # Text
    
    # Add simple safe/warning indicator
    warning_objects = [d for _, d in distances if d < threshold]
    if warning_objects:
        cv2.putText(output, "WARNING", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    else:
        cv2.putText(output, "SAFE", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    # Add FPS counter
    fps_text = f"FPS: {frame_fps:.1f}"
    cv2.putText(output, fps_text, (frame_width - 150, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return output