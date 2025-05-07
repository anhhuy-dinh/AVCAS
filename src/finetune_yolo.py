from ultralytics import YOLO
import os
import sys
sys.path.append(".")  # Add current directory to path

# Paths
DATA_PATH = "data/bdd100k/yolo_processed"
MODEL_PATH = "model/yolov11/yolo11n.pt"  # Use nano model for speed, change to s/m/l for better accuracy
OUTPUT_MODEL = "model/yolov11/yolo11_bdd100k.pt"

def finetune():
    # Verify data.yaml exists
    data_yaml = os.path.join(DATA_PATH, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}. Run preprocess_bdd100k_yolo.py first.")
    
    # Load model
    model = YOLO(MODEL_PATH)
    
    # Fine-tune
    results = model.train(
        data=data_yaml,
        epochs=100,           # More epochs for better results
        patience=15,          # Early stopping patience
        batch=16,             # Batch size, adjust based on GPU memory
        imgsz=640,            # Training image size
        device=0,             # GPU device ID, use 'cpu' for CPU training
        workers=4,            # Number of data loader workers
        project="runs/train", # Project name
        name="yolov11_bdd",   # Run name
        exist_ok=True,        # Overwrite existing runs
        pretrained=True,      # Use pretrained weights
        optimizer="Adam",     # Optimizer
        lr0=0.001,            # Initial learning rate
        lrf=0.01,             # Final learning rate factor
        momentum=0.937,       # Momentum
        weight_decay=0.0005,  # Weight decay
        warmup_epochs=3.0,    # Warmup epochs
        warmup_momentum=0.8,  # Warmup momentum
        warmup_bias_lr=0.1,   # Warmup bias learning rate
        close_mosaic=10,      # Disable mosaic augmentation for final epochs
        amp=True,             # Automatic mixed precision
        resume=False          # Resume training from last checkpoint
    )
    
    # Save model
    model.save(OUTPUT_MODEL)
    print(f"Fine-tuning completed. Model saved to {OUTPUT_MODEL}")

if __name__ == "__main__":
    finetune()