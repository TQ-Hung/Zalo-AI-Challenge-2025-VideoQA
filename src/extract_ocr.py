# extract_ocr.py
import os
import cv2
import numpy as np
from tqdm import tqdm
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import json

# --- Config ---
VIDEO_DIR = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/videos"
TRAIN_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/train.json"
OUTPUT_DIR = "features_v2/ocr"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load VietOCR (nhẹ, nhanh) ---
config = Cfg()
config['weights'] = 'https://drive.google.com/uc?id=1m4gL0yLpganL6gfX--v1gP7wY4v0i8i7'  # vgg_transformer
config['cnn']['pretrained'] = False
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
config['predictor']['beamsearch'] = False
predictor = Predictor(config)

# --- Load support_frames ---
with open(TRAIN_JSON) as f:
    data = json.load(f)["data"]

video_to_supports = {}
for item in data:
    basename = os.path.splitext(os.path.basename(item["video_path"]))[0]
    video_to_supports[basename] = [float(t) for t in item.get("support_frames", []) if t]

# --- Extract OCR ---
def get_ocr_text(video_path, timestamps):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    texts = []
    for ts in timestamps:
        idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            text = predictor.predict(pil_img)
            if text.strip():
                texts.append(text.strip())
    cap.release()
    return " | ".join(texts) if texts else ""

# --- Main ---
from PIL import Image
import torch

for item in tqdm(data, desc="Extracting OCR"):
    basename = os.path.splitext(os.path.basename(item["video_path"]))[0]
    save_path = os.path.join(OUTPUT_DIR, f"{basename}.txt")
    if os.path.exists(save_path):
        continue
    video_path = os.path.join(VIDEO_DIR, f"{basename}.mp4")
    timestamps = video_to_supports.get(basename, [])
    ocr_text = get_ocr_text(video_path, timestamps)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(ocr_text)

print(f"OCR DONE! Saved to {OUTPUT_DIR}")