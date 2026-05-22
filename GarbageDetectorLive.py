import cv2
import math
import cvzone
from ultralytics import YOLO
import os
import sys

model = YOLO("best.pt")

class_names = ['Aerosol', 'Aluminium blister pack', 'Aluminium foil', 'Battery', 'Broken glass', 'Carded blister pack', 'Cigarette', 'Clear plastic bottle', 'Corrugated carton', 'Crisp packet', 'Disposable food container', 'Disposable plastic cup', 'Drink can', 'Drink carton', 'Egg carton', 'Foam cup', 'Foam food container', 'Food Can', 'Food waste', 'Garbage bag', 'Glass bottle', 'Glass cup', 'Glass jar', 'Magazine paper', 'Meal carton', 'Metal bottle cap', 'Metal lid', 'Normal paper', 'Other carton', 'Other plastic', 'Other plastic bottle', 'Other plastic container', 'Other plastic cup', 'Other plastic wrapper', 'Paper bag', 'Paper cup', 'Paper straw', 'Pizza box', 'Plastic bottle cap', 'Plastic film', 'Plastic glooves', 'Plastic lid', 'Plastic straw', 'Plastic utensils', 'Polypropylene bag', 'Pop tab', 'Rope & strings', 'Scrap metal', 'Shoe', 'Single-use carrier bag', 'Six pack rings', 'Spread tub', 'Squeezable tube', 'Styrofoam piece', 'Tissues', 'Toilet tube', 'Tupperware', 'Unlabeled litter', 'Wrapping paper']

#target_filename = "trashAroundLondon.mp4"
#target_filename = "beach.mp4"
target_filename = "trashOnWater.mp4"
#target_filename = "garbage.mp4"

project_root = "/Users/meimei/Documents/CTU/Project/GarbageDetector"
video_path = None

for root, dirs, files in os.walk(project_root):
    if target_filename in files:
        video_path = os.path.join(root, target_filename)
        break

if video_path is None:
    print("\n" + "!" * 60)
    print(f"The video file '{target_filename}' does not exist inside:")
    print(f"--> {project_root}")
    sys.exit()
else:
    print(f"Found video at absolute path: {video_path}")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file stream at {video_path}")
    sys.exit()

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("\nVideo playback finished or stream interrupted.")
        break

    results = model.track(frame, device="mps", conf=0.15, iou=0.45, persist=True, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf > 0.15:
                label = f'{class_names[cls]} {conf:.2f}'
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                (t_w, t_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_y = max(t_h + 10, y1)
                
                cv2.rectangle(frame, (x1, text_y - t_h - 5), (x1 + t_w + 5, text_y + baseline), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1 + 2, text_y - 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Real-Time Garbage Detection Video Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nDemo stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)