import cv2
import time
import os
from ultralytics import YOLO

MODEL_PATH = '/opt/homebrew/runs/detect/train3/weights/best.pt'
CONFIDENCE = 0.5

# Cooldown to prevent volume jumping too fast
last_action_time = 0
ACTION_COOLDOWN = 0.2 # Seconds between volume changes

try:
    model = YOLO(MODEL_PATH)
except:
    print("Error: Check path")
    exit()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("✌️  Paper = Volume UP")
print("✊  Rock  = Volume DOWN")
print("✌️  Scissors = MUTE")
print("Press 'Q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Run detection
    results = model(frame, stream=True, conf=CONFIDENCE, verbose=False)
    
    current_gesture = None
    
    for result in results:
        for box in result.boxes:
            # We pick the gesture with highest confidence
            cls_id = int(box.cls[0])
            current_gesture = model.names[cls_id]
            
            # Draw for feedback
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, current_gesture, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # --- ACTION LOGIC ---
    if current_gesture and (time.time() - last_action_time > ACTION_COOLDOWN):
        
        if current_gesture == "Paper":
            # Volume Up (+5%)
            os.system("osascript -e 'set volume output volume (output volume of (get volume settings) + 5)'")
            print("🔊 Volume UP")
            last_action_time = time.time()
            
        elif current_gesture == "Rock":
            # Volume Down (-5%)
            os.system("osascript -e 'set volume output volume (output volume of (get volume settings) - 5)'")
            print("🔉 Volume DOWN")
            last_action_time = time.time()
            
        elif current_gesture == "Scissors":
            # Mute (Toggle)
            os.system("osascript -e 'set volume output muted true'")
            print("Is Muted")
            last_action_time = time.time() + 1.0 # Longer cooldown for mute

    cv2.imshow("Gesture Volume Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()