import cv2
import numpy as np
import typing

# Simple RectInfo class to mimic your original implementation
class RectInfo:
    def __init__(self, xmin, ymin, xmax, ymax, label="unknown", track_id=-1):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.label = label
        self.track_id = track_id  # Added track_id to keep association
    
    def tolist(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]

# Function to add shadowed text on frame
def putText_shadow(frame, text, position, fontFace, fontScale, color, thickness, shadow_color):
    # Draw shadow
    cv2.putText(frame, text, (position[0]+1, position[1]+1), fontFace, fontScale, shadow_color, thickness)
    # Draw text
    cv2.putText(frame, text, position, fontFace, fontScale, color, thickness)

class SingleCamDistanceMeasure(object):
    # 1 cm = 0.39 inch, original size h x w 
    INCH = 0.39
    # Reference sizes for different object classes
    RefSizeDict = { 
                    "person" : (160*INCH, 50*INCH), 
                    "bicycle" : (98*INCH, 65*INCH),
                    "motorbike" : (100*INCH, 100*INCH),
                    "car" : (150*INCH, 180*INCH),
                    "bus" : (319*INCH, 250*INCH), 
                    "truck" : (346*INCH, 250*INCH), 
                 }
    
    def __init__(self, object_list: list = ["person", "bicycle", "car", "motorbike", "bus", "truck"], focal_length=100):
        self.object_list = object_list
        self.f = focal_length  # focal length
        self.distance_points = []
        self.distance_by_track_id = {}  # New dictionary to store distances by track ID
        
    def __isInsidePolygon(self, pt: tuple, poly: np.ndarray) -> bool:
        """
        Judgment point is within the polygon range.
        Args:
            pt: the object points.
            poly: is a polygonal points. [[x1, y1], [x2, y2], [x3, y3] ... [xn, yn]]
        Returns:
            True if point is inside polygon, False otherwise.
        """
        c = False
        i = -1
        l = len(poly)
        j = l - 1
        while i < l - 1:
            i += 1
            if((poly[i][0]<=pt[0] and pt[0] < poly[j][0])or(
                poly[j][0]<=pt[0] and pt[0]<poly[i][0] )):
                if(pt[1]<(poly[j][1]-poly[i][1]) * (pt[0]-poly[i][0])/(
                    poly[j][0]-poly[i][0])+poly[i][1]):
                    c = not c
            j=i
        return c
        
    def updateDistance(self, boxes: typing.List[RectInfo]) -> None:
        """
        Update the distance of the target object through the size of pixels.
        Args:
            boxes: coordinate information and labels of the target object.
        Returns:
        """
        self.distance_points = []
        self.distance_by_track_id = {}  # Clear previous distances
        
        if (len(boxes) != 0):
            for box in boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                label = box.label
                track_id = box.track_id
                
                # Check if the object is in our list of interest and not too low in the frame
                # Increased the ymax threshold to 720 to include more objects
                if label in self.object_list and ymax <= 720:
                    point_x = (xmax + xmin) // 2
                    point_y = ymax
                    
                    try:
                        # Get reference height for this object class
                        ref_height = self.RefSizeDict[label][0]
                        
                        # Calculate distance using the formula: (real_size * focal_length) / apparent_size
                        object_height = ymax - ymin
                        if object_height > 0:  # Avoid division by zero
                            distance = (ref_height * self.f) / object_height
                            distance = distance / 12 * 0.3048  # Convert from feet to meters (1ft = 0.3048 m)
                            
                            # Add to distance points list
                            self.distance_points.append([point_x, point_y, distance])
                            
                            # Also store by track ID for easier access
                            if track_id >= 0:
                                self.distance_by_track_id[track_id] = distance
                        else:
                            # Fallback for zero height (should be rare)
                            if track_id >= 0:
                                self.distance_by_track_id[track_id] = 999.9  # Large distance as fallback
                    except Exception as e:
                        print(f"Error calculating distance for {label}: {e}")
                        # If calculation fails, still try to provide a fallback distance based on position
                        # Objects lower in the frame are typically closer
                        if track_id >= 0:
                            normalized_y = min(1.0, max(0.1, ymax / 720))  # Normalize y position (0.1 to 1.0)
                            fallback_distance = 50 * (1 - normalized_y)  # Simple heuristic: objects at bottom are closer
                            self.distance_by_track_id[track_id] = fallback_distance
 
    def getDistanceForTrackId(self, track_id):
        """
        Get the distance for a specific tracked object.
        Args:
            track_id: ID of the tracked object
        Returns:
            Distance in meters or None if not available
        """
        return self.distance_by_track_id.get(track_id, None)
        
    def calcCollisionPoint(self, poly: np.ndarray) -> typing.Union[list, None]:
        """
        Determine whether the target object is within the main lane lines.
        Args:
            poly: is a polygonal points. [[x1, y1], [x2, y2], [x3, y3] ... [xn, yn]]
        Returns:
            [Xcenter, Ybottom, distance]
        """
        if (len(self.distance_points) != 0 and len(poly)):
            sorted_distance_points = sorted(self.distance_points, key=lambda arr: arr[2])
            for x, y, d in sorted_distance_points:
                status = True if cv2.pointPolygonTest(poly, ((x, y)), False) >= 0 else False
                # status = self.__isInsidePolygon((x, y), np.squeeze(poly))  # also can use it.
                if (status):
                    return [x, y, d]
        return None
        
    def DrawDetectedOnFrame(self, frame_show: cv2) -> None: 
        if (len(self.distance_points) != 0):
            for x, y, d in self.distance_points:
                cv2.circle(frame_show, (x, y), 4, (255, 255, 255), thickness=-1)
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
                putText_shadow(frame_show, text, (textX + 1, textY + 5), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=fontScale,
                         color=(255, 255, 255), thickness=1, shadow_color=(150, 150, 150))

# Updated consistent class mapping
CLASS_NAMES = {
    0: 'car',
    1: 'truck',
    2: 'bus',
    3: 'person',
    5: 'motorbike',  # Originally "motorcycle" in YOLO
    6: 'bicycle'
}

# Function to extract distance information from detections
def get_distances_from_detections(detections, class_names=None):
    """
    Calculate distances for detected objects using SingleCamDistanceMeasure
    
    Args:
        detections: List of detections from YOLO
        class_names: Dictionary mapping class IDs to class names
        
    Returns:
        Tuple of (distances list, distance_points)
    """
    # Create a distance measure object
    distance_measure = SingleCamDistanceMeasure()
    
    # Convert detections to RectInfo objects
    rect_infos = []
    track_ids = []
    
    for det in detections:
        if len(det) < 4 or det[0] is None or det[3] is None:
            continue
            
        bbox = det[0]  # First element is bbox
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
        
        # Get class name with consistent mapping
        if class_names and class_id in class_names:
            class_name = class_names[class_id]
        else:
            # Use our updated consistent mapping
            class_name = CLASS_NAMES.get(class_id, "unknown")
            
        # Map "motorcycle" to "motorbike" for consistency
        if class_name == "motorcycle":
            class_name = "motorbike"
            
        # Create RectInfo object
        x1, y1, x2, y2 = map(int, bbox)
        rect_info = RectInfo(x1, y1, x2, y2, label=class_name, track_id=track_id)
        rect_infos.append(rect_info)
        track_ids.append(track_id)
    
    # Update distances
    distance_measure.updateDistance(rect_infos)
    
    # Extract distances using track IDs
    distances = []
    for track_id in distance_measure.distance_by_track_id:
        if track_id >= 0:  # Valid track ID
            distance = distance_measure.distance_by_track_id[track_id]
            distances.append((track_id, distance))
    
    return distances, distance_measure.distance_points

# Enhanced function to visualize distances on frame
def visualize_distances(frame, distance_points, threshold=15.0):
    """
    Draw distance information on frame using SingleCamDistanceMeasure
    
    Args:
        frame: Input image
        distance_points: List of [x, y, distance] from distance calculation
        threshold: Warning threshold distance in meters
        
    Returns:
        Frame with visualized distances
    """
    result_frame = frame.copy()
    
    if len(distance_points) > 0:
        for x, y, d in distance_points:
            # Circle at the bottom of the object
            if d < threshold:
                # Red circle for close objects
                circle_color = (0, 0, 255)
            else:
                # Green circle for far objects
                circle_color = (0, 255, 0)
                
            cv2.circle(result_frame, (x, y), 6, circle_color, thickness=-1)
            
            # Distance text
            unit = 'm'
            if d < 0:
                text = ' {} {}'.format("unknown", unit)
            else:
                text = ' {:.2f} {}'.format(d, unit)
            
            # Dynamic font scaling based on distance
            fontScale = max(0.4, min(1, 1/d if d > 0 else 1))
            
            # Get text size for positioning
            textsize = cv2.getTextSize(text, 0, fontScale=fontScale, thickness=3)[0]
            textX = int((x - textsize[0]/2))
            textY = int((y + textsize[1]))
            
            # Draw text with shadow effect
            cv2.putText(result_frame, text, (textX + 1, textY + 5 + 1), 
                         cv2.FONT_HERSHEY_TRIPLEX, fontScale, (0, 0, 0), 3)  # Shadow
            cv2.putText(result_frame, text, (textX + 1, textY + 5), 
                         cv2.FONT_HERSHEY_TRIPLEX, fontScale, 
                         (255, 255, 255) if d >= threshold else (50, 200, 255), 
                         1)  # Text
    
    return result_frame

# Modified distance estimation that handles more edge cases
def estimate_distance(bbox_width, bbox_height, object_class="car", focal_length=1000):
    """
    Enhanced distance estimation function
    
    Args:
        bbox_width: Width of bounding box in pixels
        bbox_height: Height of bounding box in pixels
        object_class: Class of the object
        focal_length: Camera focal length
        
    Returns:
        Distance in meters
    """
    # Reference sizes for different object classes (in meters)
    ref_sizes = {
        "car": (1.8, 1.5),      # width, height
        "truck": (2.5, 2.5),
        "bus": (2.5, 3.0),
        "person": (0.5, 1.7),
        "bicycle": (0.6, 1.0),
        "motorbike": (0.8, 1.2)
    }
    
    # Use a more robust approach by considering both width and height
    if bbox_width == 0 or bbox_height == 0:
        return float("inf")
    
    # Get reference size for this class or default to car
    ref_width, ref_height = ref_sizes.get(object_class, ref_sizes["car"])
    
    # Calculate distance using both dimensions
    distance_by_width = (ref_width * focal_length) / bbox_width
    distance_by_height = (ref_height * focal_length) / bbox_height
    
    # Average the two estimates with a preference for height-based estimate
    # (height is often more reliable for distance estimation)
    distance = 0.4 * distance_by_width + 0.6 * distance_by_height
    
    return distance