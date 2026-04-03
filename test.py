# save as test_yolo2.py
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("data/input_videos/long.mp4")

# Try ALL classes, low confidence, more frames
for fi in range(95, 160, 2):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret: continue
    results = model(frame, conf=0.10, verbose=False)  # very low threshold, all classes
    boxes = results[0].boxes
    if len(boxes) > 0:
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            name = model.names[cls]
            print(f"frame={fi}: {name}(cls={cls}) conf={conf:.2f} at ({int((x1+x2)/2)},{int((y1+y2)/2)})")

cap.release()
print("done")