import cv2
import time
import random
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = '/opt/homebrew/runs/detect/train3/weights/best.pt'
CONFIDENCE = 0.5
ZONE_SIZE = 600

# Game States
STATE_WAITING = "waiting"
STATE_COUNTDOWN = "countdown"
STATE_RESULT = "result"

try:
    model = YOLO(MODEL_PATH)
except:
    print("Error: Check your model path!")
    exit()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Game Variables
current_state = STATE_WAITING
countdown_start_time = 0
result_start_time = 0
ai_move = ""
user_move = ""
winner = ""
message = "Press SPACE to Start!"

def get_winner(u, a):
    if u == a: return "Tie"
    if (u == "Rock" and a == "Scissors") or \
       (u == "Paper" and a == "Rock") or \
       (u == "Scissors" and a == "Paper"):
        return "You Win!"
    return "AI Wins!"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Flip frame for mirror effect (more natural for gaming)
    frame = cv2.flip(frame, 1)
    
    height, width, _ = frame.shape
    
    # --- DRAW UI ---
    # Draw Game Zone (Center)
    z_x1 = int((width - ZONE_SIZE) / 2)
    z_y1 = int((height - ZONE_SIZE) / 2) + 50
    z_x2 = z_x1 + ZONE_SIZE
    z_y2 = z_y1 + ZONE_SIZE
    cv2.rectangle(frame, (z_x1, z_y1), (z_x2, z_y2), (255, 255, 0), 3)

    # --- DETECTION LOGIC ---
    # Only detect if we are in counting or result mode
    detected_class = None
    
    # Run model
    results = model(frame, stream=True, conf=CONFIDENCE, verbose=False)
    for result in results:
        for box in result.boxes:
            # Check if center of box is in zone
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            if z_x1 < cx < z_x2 and z_y1 < cy < z_y2:
                # Valid move found
                cls_id = int(box.cls[0])
                detected_class = model.names[cls_id]
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, detected_class, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- GAME STATE MACHINE ---
    
    if current_state == STATE_WAITING:
        cv2.putText(frame, "Press SPACE to Play", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Check for key press
        key = cv2.waitKey(1)
        if key == 32: # SPACE bar
            current_state = STATE_COUNTDOWN
            countdown_start_time = time.time()

    elif current_state == STATE_COUNTDOWN:
        elapsed = time.time() - countdown_start_time
        
        if elapsed < 1:
            display_text = "3..."
        elif elapsed < 2:
            display_text = "2..."
        elif elapsed < 3:
            display_text = "1..."
        else:
            # TIME IS UP! Capture the move
            user_move = detected_class if detected_class else "Nothing"
            ai_move = random.choice(["Rock", "Paper", "Scissors"])
            
            if user_move == "Nothing":
                winner = "No Move Detected"
            else:
                winner = get_winner(user_move, ai_move)
            
            current_state = STATE_RESULT
            result_start_time = time.time()
            display_text = "SHOOT!"

        # Show countdown text big in the middle
        if current_state == STATE_COUNTDOWN:
            cv2.putText(frame, display_text, (width//2 - 100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 165, 255), 5)

    elif current_state == STATE_RESULT:
        # Show results for 4 seconds
        elapsed = time.time() - result_start_time
        
        # Display Who Played What
        cv2.putText(frame, f"You: {user_move}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"AI: {ai_move}", (width - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display Winner (Big)
        color = (0, 255, 0) if "You Win" in winner else (0, 0, 255)
        cv2.putText(frame, winner, (width//2 - 200, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        
        if elapsed > 4:
            current_state = STATE_WAITING

    cv2.imshow("Rock Paper Scissors AI Battle", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()