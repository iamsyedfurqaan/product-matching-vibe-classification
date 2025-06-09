import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image
import os
from sklearn.neighbors import NearestNeighbors

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load catalog embeddings and IDs from root
embeddings = np.load('catalog_clip_embeddings.npy')
catalog_ids = pd.read_csv('catalog_product_ids.csv')['Product ID'].tolist()

# Build NearestNeighbors index
nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(embeddings)

def match_detection(frame_path, bbox):
    if not os.path.exists(frame_path):
        print(f"Frame file not found: {frame_path}")
        return None, 0.0, "file_not_found"
    img = Image.open(frame_path).convert('RGB')
    x1, y1, x2, y2 = map(int, bbox)
    width, height = img.size
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        print(f"Invalid crop for {frame_path}: {bbox}")
        return None, 0.0, "invalid_crop"
    cropped = img.crop((x1, y1, x2, y2))
    image_input = preprocess(cropped).unsqueeze(0).to(device)
    with torch.no_grad():
        query_feat = clip_model.encode_image(image_input)
        query_feat /= query_feat.norm(dim=-1, keepdim=True)
        query_np = query_feat.cpu().numpy().astype(np.float32)
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)
        distances, indices = nn.kneighbors(query_np, n_neighbors=1)
        sim = 1 - float(distances[0][0])  # cosine similarity
        idx = int(indices[0][0])
        match_type = "no_match"
        if sim > 0.9:
            match_type = "exact"
        elif sim > 0.75:
            match_type = "similar"
        return catalog_ids[idx], sim, match_type

# Process all detections for a video
video_name = "video1"  # <-- Change this to the actual video name or loop over all videos
detections = pd.read_csv(f'outputs/detections/{video_name}_detections.csv')
results = []
for _, row in detections.iterrows():
    frame_file = row['frame_file']
    bbox = [row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']]
    frame_path = f"frames/{video_name}/{frame_file}"
    try:
        prod_id, sim, match_type = match_detection(frame_path, bbox)
        results.append({**row, 'matched_product_id': prod_id, 'similarity': sim, 'match_type': match_type})
    except Exception as e:
        print(f"Error processing {frame_file} with bbox {bbox}: {e}")
        results.append({**row, 'matched_product_id': None, 'similarity': 0.0, 'match_type': 'error'})

results_df = pd.DataFrame(results)
os.makedirs('outputs/matched', exist_ok=True)
results_df.to_csv(f'outputs/matched/{video_name}_matched.csv', index=False)
print(f"Matching complete for {video_name}.")

# Notes:
# - To process all videos, wrap the above in a loop over all detection CSVs in 'outputs/detections/'.
# - Run from the root of your submission folder.
