---

# Flickd AI Hackathon Submission

## Overview

This repository contains my end-to-end solution for the Flickd AI Hackathon.  
My pipeline processes short-form fashion videos, detects products, matches them to the provided catalog, and classifies the vibe for each video.  
The solution is modular, reproducible, and follows all Flickd special instructions.

---

## Folder Structure

```
submission/
├── api/                 # FastAPI code for pipeline API
├── scripts/             # Batch processing scripts
├── videos/              # Raw input videos
├── catalog.csv          # Provided product catalog (CSV)
├── vibes_list.json      # Provided list of possible vibes
├── outputs/             # Per-video JSON outputs (final results only)
│   ├── reel_001.json
│   └── reel_002.json
├── models/              # Model weights (e.g., yolov8n.pt)
├── README.md
├── requirements.txt
└── demo.mp4             # (Or Loom link, see below)
```

---

## Setup Instructions

1. **Clone and Prepare Environment**
    ```bash
    git clone 
    cd submission
    python3.10 -m venv venv310
    source venv310/bin/activate
    pip install -r requirements.txt
    ```

2. **Download Model Weights**
    - Place `yolov8n.pt` (or your YOLO weights) in the `models/` directory.

3. **Prepare Data**
    - Place all videos to process in the `videos/` folder.
    - Ensure `catalog.csv` and `vibes_list.json` are present in the root.
    - (Optional) Place `captions.csv` in `data/` if running batch vibe classification.

---

## Pipeline Usage

### A. Batch Processing (Recommended for Bulk Videos)

1. **Extract Frames:**
    ```bash
    python scripts/extract_frames.py
    ```
2. **Object Detection:**
    ```bash
    python scripts/detect_objects.py
    ```
3. **Generate Catalog Embeddings (run once):**
    ```bash
    python scripts/clip_embeds.py
    ```
4. **Product Matching:**
    ```bash
    python scripts/batch_match.py
    ```
5. **Vibe Classification:**
    ```bash
    python scripts/vibe_classification.py
    ```
6. **Create Final Output JSONs:**
    ```bash
    python scripts/create_final_output.py
    ```

- **Outputs:**  
  One JSON per video in `outputs/`, matching Flickd’s required format.

---

### B. API Usage (Optional, for Single Video/Interactive Use)

1. **Start the API:**
    ```bash
    uvicorn api.main:app --reload
    ```
2. **Open your browser at:**  
    [http://localhost:8000/docs](http://localhost:8000/docs)
3. **Upload a video and enter a caption** at the `/analyze/` endpoint.
4. **Output:**  
    JSON result in the Flickd-required format.

---

## Model Versions Used

- **YOLO:** ultralytics/yolov8n
- **CLIP:** ViT-B/32 (OpenAI)
- **scikit-learn:** NearestNeighbors
- **Python:** 3.10+

---

## Special Instructions Compliance

- Only the provided catalog.csv is used.
- No extra data or scraping.
- Catalog embeddings are precomputed and reused.
- Products with similarity < 0.75 are excluded from outputs.
- Multiple products per video are included if detected.
- Vibe classification returns 1–3 relevant vibes per video.
- All code (API and batch scripts) is included and documented.
- README includes all setup and usage instructions.
- No Whisper (audio transcription) was used.

---

## Demo Video - https://your-loom-link.com](https://drive.google.com/file/d/1VHvEv2txkURPEwK-E_jbAtZZdhhoSnDv/view?usp=sharing  
  *(Or see `demo.txt` in this repo)*

---

## Example Output Format

```json
{
  "video_id": "reel_001",
  "vibes": ["Coquette", "Brunchcore"],
  "products": [
    {
      "type": "top",
      "color": "white",
      "matched_product_id": "prod_002",
      "match_type": "exact",
      "confidence": 0.93
    },
    {
      "type": "earrings",
      "color": "gold",
      "matched_product_id": "prod_003",
      "match_type": "similar",
      "confidence": 0.85
    }
  ]
}
```

---

## Notes

- All code is in `api/` (API) and `scripts/` (batch processing).
- All outputs are in `outputs/` as per-video JSONs.
- If you encounter any issues, please check the file paths and ensure all dependencies are installed.

---

## Contact

For any questions, please contact:  
**Syed Furqaan Ahmed**  
**iamsyedfurqaan@gmail.com**

---

**Thank you for reviewing my submission!**

---
