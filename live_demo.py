import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = '/opt/homebrew/runs/detect/train3/weights/best.pt'
CONFIDENCE_THRESHOLD = 0.4  # Lowered to 0.4 to catch those shy Scissors!

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error: Could not load model at {MODEL_PATH}")
    exit()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("Press 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get screen dimensions
    height, width, _ = frame.shape

    # --- DEFINE THE GAME ZONE ---
    # We define a box in the center-right of the screen
    # (x1, y1) is top-left, (x2, y2) is bottom-right
    zone_size = 600
    zone_x1 = int((width - zone_size) / 2)
    zone_y1 = int((height - zone_size) / 2) + 50
    zone_x2 = zone_x1 + zone_size
    zone_y2 = zone_y1 + zone_size

    # Draw the Game Zone (Blue Box)
    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 0), 3)
    cv2.putText(frame, "PUT HAND HERE", (zone_x1, zone_y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Run detection on the WHOLE frame (easier than cropping)
    results = model(frame, stream=True, conf=CONFIDENCE_THRESHOLD)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Calculate the center of the detected object
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # --- THE "MAGIC" FILTER ---
            # Check if the CENTER of the object is inside the Game Zone
            is_in_zone = (zone_x1 < center_x < zone_x2) and (zone_y1 < center_y < zone_y2)

            if is_in_zone:
                # If it's in the zone, it's valid! Draw it green.
                color = (0, 255, 0) # Green
                
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls_id]} {conf:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                # (Optional) Draw ignored objects in Red so you can see them being rejected
                # This helps you confirm your head is being detected but ignored
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imshow("Rock Paper Scissors - Game Zone", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()