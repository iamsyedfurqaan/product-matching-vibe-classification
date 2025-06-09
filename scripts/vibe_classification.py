import pandas as pd
import json
import os

# Load vibes list from JSON (must be in root as 'vibes_list.json')
vibes_path = 'vibes_list.json'
with open(vibes_path, 'r') as f:
    vibes = json.load(f)

# Define keywords for each vibe (expand as needed)
vibe_keywords = {
    "Coquette": ["lace", "bow", "feminine", "pink", "romantic"],
    "Clean Girl": ["minimal", "sleek", "neutral", "simple", "clean"],
    "Cottagecore": ["floral", "nature", "cottage", "pastel", "vintage"],
    "Streetcore": ["urban", "street", "baggy", "sneaker", "graffiti"],
    "Y2K": ["retro", "2000s", "butterfly", "glitter", "low-rise"],
    "Boho": ["bohemian", "fringe", "earthy", "flowy", "gypsy"],
    "Party Glam": ["sparkle", "glam", "party", "metallic", "bold"]
}

def classify_vibe(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    matched_vibes = []
    for vibe in vibes:
        keywords = vibe_keywords.get(vibe, [])
        for kw in keywords:
            if kw in text:
                matched_vibes.append(vibe)
                break
    return matched_vibes[:3]  # Up to 3 vibes

# Load captions CSV
captions_path = 'captions.csv'  # Captions should be in data/
output_path = 'outputs/vibe_classification.csv'

if not os.path.exists(captions_path):
    print(f"Captions file not found: {captions_path}")
else:
    captions = pd.read_csv(captions_path)  # Columns: video_id, caption
    captions['vibes'] = captions['caption'].apply(classify_vibe)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    captions.to_csv(output_path, index=False)
    print(f"Vibe classification complete! Output saved to {output_path}")

# Notes:
# - Run this script from the root of your submission folder.
# - Make sure 'vibes_list.json' is in the root, and 'data/captions.csv' exists.
