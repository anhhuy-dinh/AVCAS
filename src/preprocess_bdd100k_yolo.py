import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# Paths
BDD100K_ROOT = "data/bdd100k"
BDD100K_LABELS = "data/bdd100k_labels"
OUTPUT_YOLO = "data/bdd100k/yolo_processed"
SPLITS = ["train", "val"]  # Splits to process

# BDD100K class names without mapping
BDD_CLASSES = [
    "car", "truck", "bus", "person", "rider", "motorcycle", "bicycle", 
    "traffic light", "traffic sign", "train"
]

# Create mapping from class name to ID
CLASS_NAMES = {name: idx for idx, name in enumerate(BDD_CLASSES)}

def preprocess_yolo_data():
    """Preprocess BDD100K for YOLOv11."""
    
    # Create output directories
    os.makedirs(OUTPUT_YOLO, exist_ok=True)
    
    for split in SPLITS:
        # Create split directories
        split_img_dir = os.path.join(OUTPUT_YOLO, split, "images")
        split_label_dir = os.path.join(OUTPUT_YOLO, split, "labels")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_label_dir, exist_ok=True)
        
        # Get input paths
        image_dir = os.path.join(BDD100K_ROOT, split)
        label_dir = os.path.join(BDD100K_LABELS, split)
        
        # Get all image files
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        
        for img_file in tqdm(image_files, desc=f"Processing {split} data for YOLO"):
            # Get image path
            img_path = os.path.join(image_dir, img_file)
            
            # Get corresponding label file
            label_file = os.path.join(label_dir, img_file.replace('.jpg', '.json').replace('.png', '.json'))
            
            # Skip if label file doesn't exist
            if not os.path.exists(label_file):
                continue
            
            # Load image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            # Copy image to destination
            shutil.copy(img_path, os.path.join(split_img_dir, img_file))
            
            # Load and process label
            with open(label_file, 'r') as f:
                data = json.load(f)
            
            # Create YOLO format label file
            yolo_label_path = os.path.join(split_label_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            with open(yolo_label_path, 'w') as f:
                # Process all objects in the first frame
                if 'frames' in data and len(data['frames']) > 0:
                    objects = data['frames'][0].get('objects', [])
                else:
                    # Old format directly has objects
                    objects = data.get('labels', [])
                
                for obj in objects:
                    if 'category' not in obj:
                        continue
                        
                    category = obj['category']
                    
                    # Skip if category is not in our class mapping
                    if category not in CLASS_NAMES:
                        continue
                    
                    # Get class ID
                    class_id = CLASS_NAMES[category]
                    
                    # Get bounding box coordinates
                    if 'box2d' in obj:
                        box = obj['box2d']
                        x1, y1 = box['x1'], box['y1']
                        x2, y2 = box['x2'], box['y2']
                        
                        # Convert to YOLO format: center_x, center_y, width, height (normalized)
                        center_x = (x1 + x2) / 2 / w
                        center_y = (y1 + y2) / 2 / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        
                        # Write to file
                        f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
    
    # Create data.yaml for YOLO
    classes = len(set(CLASS_NAMES.values()))
    class_list = [k for k, v in sorted(CLASS_NAMES.items(), key=lambda item: item[1])]
    
    data_yaml = f"""
train: {os.path.abspath(os.path.join(OUTPUT_YOLO, 'train', 'images'))}
val: {os.path.abspath(os.path.join(OUTPUT_YOLO, 'val', 'images'))}
nc: {classes}
names: {str(class_list)}
    """
    
    with open(os.path.join(OUTPUT_YOLO, "data.yaml"), "w") as f:
        f.write(data_yaml)
    
    print(f"YOLO preprocessing completed. Data saved to {OUTPUT_YOLO}")

if __name__ == "__main__":
    preprocess_yolo_data()