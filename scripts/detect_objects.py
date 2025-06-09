from ultralytics import YOLO
import cv2
import os
import pandas as pd
from tqdm import tqdm

# Updated paths according to Flickd submission structure
frames_dir = 'frames'                        # Extracted frames directory
output_dir = 'outputs/detections'            # Detections output directory
os.makedirs(output_dir, exist_ok=True)

# Load YOLOv8 model from the models/ directory
model = YOLO('models/yolov8n.pt')  # Use 'yolov8s.pt' for better accuracy if available

# Loop through each video's frames
for video_name in os.listdir(frames_dir):
    video_frames_dir = os.path.join(frames_dir, video_name)
    if not os.path.isdir(video_frames_dir):
        continue

    results_list = []

    frame_files = sorted([f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')])
    for frame_file in tqdm(frame_files, desc=f"Detecting in {video_name}"):
        frame_path = os.path.join(video_frames_dir, frame_file)
        frame_idx = int(frame_file.split('_')[-1].split('.')[0])
        img = cv2.imread(frame_path)

        results = model(img)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.model.names[cls]
                results_list.append({
                    "frame_file": frame_file,
                    "frame_idx": frame_idx,
                    "class": class_name,
                    "confidence": conf,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2
                })

    # Save detections for this video
    df = pd.DataFrame(results_list)
    df.to_csv(os.path.join(output_dir, f"{video_name}_detections.csv"), index=False)
    print(f"Detections saved for {video_name}")

# This script assumes you run it from the root of your submission folder.
# It reads frames from 'frames/', writes detections to 'outputs/detections/',
# and loads model weights from 'models/'.
