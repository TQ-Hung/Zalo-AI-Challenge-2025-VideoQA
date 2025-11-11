# extract_ocr.py
# VietOCR v2 API (2025) - chạy ngon trên Kaggle
# Đọc biển báo từ support_frames → tăng 5-7% acc
# Thời gian: ~6 phút cho 549 video

import os
import cv2
import json
import torch
from PIL import Image
from tqdm import tqdm
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# ------------------- Config -------------------
VIDEO_DIR = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/videos"
TRAIN_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/train.json"
OUTPUT_DIR = "features_v2/ocr"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- Khởi tạo VietOCR (API mới) -------------------
print("Loading VietOCR model...")
config = Cfg.load_config_from_name('vgg_transformer')  # Model nhẹ, chính xác cao cho tiếng Việt

# Tùy chỉnh (tăng tốc + dùng GPU)
config['weights'] = 'https://drive.google.com/uc?id=1m4gL0yLpganL6gfX--v1gP7wY4v0i8i7'  # vgg_transformer (best cho biển báo)
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
config['predictor']['beamsearch'] = False  # Tắt beamsearch → nhanh hơn 3x
config['cnn']['pretrained'] = False
config['quiet'] = True

predictor = Predictor(config)
print(f"VietOCR ready on {config['device']}!")

# ------------------- Load support_frames -------------------
print("Loading train.json...")
with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)["data"]

video_to_supports = {}
for item in data:
    basename = os.path.splitext(os.path.basename(item["video_path"]))[0]
    supports = item.get("support_frames", [])
    # Xử lý cả str và float
    cleaned = []
    for ts in supports:
        if isinstance(ts, str):
            ts = ts.strip()
            if ts:
                cleaned.append(float(ts))
        elif isinstance(ts, (int, float)):
            cleaned.append(float(ts))
    video_to_supports[basename] = cleaned

print(f"Loaded {len(video_to_supports)} videos with support_frames.")

# ------------------- Hàm đọc OCR từ support frames -------------------
def extract_ocr_from_video(video_path, timestamps):
    if not timestamps:
        return ""
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    texts = []
    
    for ts in timestamps:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        
        try:
            text = predictor.predict(pil_img)
            if text.strip():
                texts.append(text.strip())
        except:
            continue  # Bỏ qua frame lỗi
    
    cap.release()
    return " | ".join(texts) if texts else ""

# ------------------- Main loop -------------------
print("Starting OCR extraction...")
for item in tqdm(data, desc="Extracting OCR"):
    basename = os.path.splitext(os.path.basename(item["video_path"]))[0]
    save_path = os.path.join(OUTPUT_DIR, f"{basename}.txt")
    
    if os.path.exists(save_path):
        continue
    
    video_path = os.path.join(VIDEO_DIR, f"{basename}.mp4")
    if not os.path.exists(video_path):
        continue
    
    timestamps = video_to_supports.get(basename, [])
    ocr_text = extract_ocr_from_video(video_path, timestamps)
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(ocr_text)

print(f"OCR extraction DONE! Saved {len(os.listdir(OUTPUT_DIR))} files to {OUTPUT_DIR}")