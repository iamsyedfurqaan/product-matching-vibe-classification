from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import shutil
import uuid

import cv2
import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image
from ultralytics import YOLO
from sklearn.neighbors import NearestNeighbors

# --- Load all models and data once at startup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
yolo_model = YOLO('models/yolov8n.pt')  # Use your model weights in models/

# Load catalog embeddings and IDs (from root)
embeddings = np.load('catalog_clip_embeddings.npy')
catalog_ids = pd.read_csv('catalog_product_ids.csv')['Product ID'].tolist()
nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(embeddings)

# Vibe keywords (expand as needed)
vibe_keywords = {
    "Coquette": ["lace", "bow", "feminine", "pink", "romantic"],
    "Clean Girl": ["minimal", "sleek", "neutral", "simple", "clean"],
    "Cottagecore": ["floral", "nature", "cottage", "pastel", "vintage"],
    "Streetcore": ["urban", "street", "baggy", "sneaker", "graffiti"],
    "Y2K": ["retro", "2000s", "butterfly", "glitter", "low-rise"],
    "Boho": ["bohemian", "fringe", "earthy", "flowy", "gypsy"],
    "Party Glam": ["sparkle", "glam", "party", "metallic", "bold"]
}

app = FastAPI()

def extract_frames(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frame_paths = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_file = os.path.join(frames_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_file, frame)
        frame_paths.append(frame_file)
        frame_idx += 1
    cap.release()
    return frame_paths

def detect_objects(frame_path):
    img = cv2.imread(frame_path)
    results = yolo_model(img)
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            class_name = yolo_model.model.names[cls]
            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
    return detections

def match_product(frame_path, bbox):
    img = Image.open(frame_path).convert('RGB')
    x1, y1, x2, y2 = bbox
    cropped = img.crop((x1, y1, x2, y2))
    image_input = preprocess(cropped).unsqueeze(0).to(device)
    with torch.no_grad():
        query_feat = clip_model.encode_image(image_input)
        query_feat /= query_feat.norm(dim=-1, keepdim=True)
        query_np = query_feat.cpu().numpy().astype(np.float32)
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)
        distances, indices = nn.kneighbors(query_np, n_neighbors=1)
        sim = 1 - float(distances[0][0])
        idx = int(indices[0][0])
        match_type = "no_match"
        if sim > 0.9:
            match_type = "exact"
        elif sim > 0.75:
            match_type = "similar"
        return catalog_ids[idx], sim, match_type

def classify_vibe(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    matched_vibes = []
    for vibe, keywords in vibe_keywords.items():
        for kw in keywords:
            if kw in text:
                matched_vibes.append(vibe)
                break
    return matched_vibes[:3]

@app.post("/analyze/")
async def analyze_video(file: UploadFile = File(...), caption: str = ""):
    # Save uploaded video to a temp file
    video_id = str(uuid.uuid4())
    temp_dir = f"/tmp/flickd_{video_id}"
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, file.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1. Extract frames
    frames_dir = os.path.join(temp_dir, "frames")
    frame_paths = extract_frames(video_path, frames_dir)

    # 2. Run detection and matching
    products = []
    for frame_path in frame_paths:
        detections = detect_objects(frame_path)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            prod_id, sim, match_type = match_product(frame_path, [x1, y1, x2, y2])
            products.append({
                "frame_file": os.path.basename(frame_path),
                "type": det["class"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "matched_product_id": prod_id,
                "similarity": sim,
                "match_type": match_type
            })

    # 3. Vibe classification
    vibes = classify_vibe(caption)

    # 4. Build output
    output = {
        "video_id": video_id,
        "caption": caption,
        "vibes": vibes,
        "products": products
    }

    # Clean up temp files if desired
    shutil.rmtree(temp_dir)

    return JSONResponse(content=output)
