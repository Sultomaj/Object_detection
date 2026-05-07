from ultralytics import YOLO

# Load the nano model (fastest for Mac/Real-time)
model = YOLO('yolo11n.pt') 

print("🚀 Starting Training...")

model.train(
    data='data.yaml',
    epochs=100,      # Long enough to learn the shapes
    imgsz=640,
    device='mps',    # Use Mac GPU
    batch=16,
    
    # --- The "Strong Model" Augmentations ---
    mosaic=1.0,      # Cuts images and mixes them (Fixes Center Bias)
    degrees=15,      # Rotates image +/- 15 degrees (Hands are never perfectly straight)
    scale=0.5,       # Zooms in/out (Helps if hand is far away)
    fliplr=0.5       # Flips left/right (Learns both Left and Right hands)
)