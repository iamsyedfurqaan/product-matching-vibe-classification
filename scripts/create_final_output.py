import pandas as pd
import json
import os

# Paths
matched_dir = 'outputs/matched'
vibes_path = 'outputs/vibe_classification.csv'
output_dir = 'outputs'

# Load vibe classifications
vibes = pd.read_csv(vibes_path)

# For each video, process its matched detections and save a JSON
for _, vibe_row in vibes.iterrows():
    video_id = vibe_row['video_id']
    matched_path = os.path.join(matched_dir, f"{video_id}_matched.csv")
    output_path = os.path.join(output_dir, f"{video_id}.json")

    if not os.path.exists(matched_path):
        print(f"Matched file not found for {video_id}: {matched_path}")
        continue

    video_matched = pd.read_csv(matched_path)

    # Build products list, filter similarity < 0.75 and "no_match"
    products = []
    for _, det in video_matched.iterrows():
        if det['similarity'] < 0.75 or det['match_type'] == "no_match":
            continue
        products.append({
            "type": det['class'],  # You may want to map this to catalog type
            "color": "",           # Fill this from catalog if needed
            "matched_product_id": det['matched_product_id'],
            "match_type": det['match_type'],
            "confidence": round(det['similarity'], 2)
        })

    # Parse vibes (handle both string and list)
    vibes_list = vibe_row['vibes']
    if isinstance(vibes_list, str):
        try:
            vibes_list = eval(vibes_list)
        except:
            vibes_list = [vibes_list]

    # Build output dict
    output = {
        "video_id": video_id,
        "vibes": vibes_list,
        "products": products
    }

    # Save as per-video JSON
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Final output saved to {output_path}")

# Notes:
# - This script assumes you run it from the root of your submission folder.
# - It reads matched CSVs from 'outputs/matched/' and vibe classification from 'outputs/vibe_classification.csv'.
# - It writes one JSON per video to 'outputs/' as required by Flickd.
# - To fill in 'color', you may need to join with catalog.csv using 'matched_product_id'.
