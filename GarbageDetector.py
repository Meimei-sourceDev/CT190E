import cv2
import math
import cvzone
from ultralytics import YOLO
import os
import sys


model = YOLO("bestv2.pt")

class_names = ['Aerosol', 'Aluminium blister pack', 'Aluminium foil', 'Battery', 'Broken glass', 'Carded blister pack', 'Cigarette', 'Clear plastic bottle', 'Corrugated carton', 'Crisp packet', 'Disposable food container', 'Disposable plastic cup', 'Drink can', 'Drink carton', 'Egg carton', 'Foam cup', 'Foam food container', 'Food Can', 'Food waste', 'Garbage bag', 'Glass bottle', 'Glass cup', 'Glass jar', 'Magazine paper', 'Meal carton', 'Metal bottle cap', 'Metal lid', 'Normal paper', 'Other carton', 'Other plastic', 'Other plastic bottle', 'Other plastic container', 'Other plastic cup', 'Other plastic wrapper', 'Paper bag', 'Paper cup', 'Paper straw', 'Pizza box', 'Plastic bottle cap', 'Plastic film', 'Plastic glooves', 'Plastic lid', 'Plastic straw', 'Plastic utensils', 'Polypropylene bag', 'Pop tab', 'Rope & strings', 'Scrap metal', 'Shoe', 'Single-use carrier bag', 'Six pack rings', 'Spread tub', 'Squeezable tube', 'Styrofoam piece', 'Tissues', 'Toilet tube', 'Tupperware', 'Unlabeled litter', 'Wrapping paper']

target_filename = "garbage_1.jpg"  
project_root = "/Users/meimei/Documents/CTU/Project/GarbageDetector/Media"
image_path = None

for root, dirs, files in os.walk(project_root):
    if target_filename in files:
        image_path = os.path.join(root, target_filename)
        break

if image_path is None:
    print("\n" + "!" * 60)
    print(f"The file '{target_filename}' does not exist anywhere inside:")
    print(f"--> {project_root}")
    sys.exit()
else:
    print(f"\nFound file at absolute path: {image_path}")

img = cv2.imread(image_path)
results = model(img, device="mps", conf=0.15, iou=0.45)

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

        if conf > 0.15:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f'{class_names[cls]} {conf:.2f}'
            
            font_scale = min(max(w / 200, 0.4), 0.7)
            thickness = 1 if font_scale < 0.6 else 2
            
            (t_w, t_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            text_y = max(t_h + 10, y1)

            cv2.rectangle(img, (x1, text_y - t_h - 5), (x1 + t_w + 5, text_y + baseline), (0, 0, 0), -1)
            cv2.putText(img, label, (x1 + 2, text_y - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
cv2.imshow("Image", img)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cv2.waitKey(1)