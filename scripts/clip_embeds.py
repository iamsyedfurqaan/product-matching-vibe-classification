import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
import clip
import torch
import numpy as np

# Use your actual CSV name
catalog_csv_path = 'images.csv'  # <-- Use images.csv as you mentioned

image_dir = 'catalog_images'                       # catalog images folder in root
embeddings_path = 'catalog_clip_embeddings.npy'    # embeddings saved in root
product_ids_path = 'catalog_product_ids.csv'       # product ids saved in root

# Create image_dir if it doesn't exist
os.makedirs(image_dir, exist_ok=True)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load images.csv
catalog = pd.read_csv(catalog_csv_path)

embeddings = []
product_ids = []

for idx, row in catalog.iterrows():
    img_url = row['image_url']
    prod_id = row['id'] if 'id' in row else row['product_id']
    img_path = os.path.join(image_dir, f"{prod_id}.jpg")
    
    # Download image if not already present
    if not os.path.exists(img_path):
        try:
            r = requests.get(img_url, timeout=10)
            img = Image.open(BytesIO(r.content)).convert('RGB')
            img.save(img_path)
        except Exception as e:
            print(f"Error downloading {img_url}: {e}")
            continue
    else:
        img = Image.open(img_path).convert('RGB')
    
    # Preprocess and embed
    image_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        embeddings.append(image_features.cpu().numpy())
        product_ids.append(prod_id)

if embeddings:
    embeddings = np.concatenate(embeddings, axis=0)
    np.save(embeddings_path, embeddings)
    pd.DataFrame({'Product ID': product_ids}).to_csv(product_ids_path, index=False)
    print("Catalog embeddings and IDs saved.")
else:
    print("No embeddings were created. Please check your catalog and image URLs.")

# Notes:
# - Assumes images.csv has columns 'image_url' and either 'id' or 'product_id'.
# - Run from the root of your submission folder.
# - Images are saved in 'catalog_images/', embeddings and IDs in root.
