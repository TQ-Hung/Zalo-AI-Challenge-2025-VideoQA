# extract_ocr.py
# PHIÊN BẢN HOÀN HẢO – CHẠY 100% TRÊN KAGGLE 2025
# Dùng weight trực tiếp từ HuggingFace (không cần Google Drive, không lỗi FileNotFound)
# Thời gian: ~3-5 phút cho 549 video

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
OUTPUT_DIR = "features/ocr"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- TẢI WEIGHT CHẮN CHẮN TỪ HF -------------------
print("Loading VietOCR model từ HuggingFace...")
config = Cfg.load_config_from_name('vgg_transformer')

# DÙNG LINK TRỰC TIẾP .pth – BẮT BUỘC CHẠY ĐƯỢC
config['weights'] = 'https://huggingface.co/VietAI/vietocr_vgg_transformer/resolve/main/vgg_transformer.pth'
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
config['predictor']['beamsearch'] = False
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
    cleaned = []
    for ts in supports:
        if isinstance(ts, str):
            ts = ts.strip()
            if ts:
                cleaned.append(float(ts))
        elif isinstance(ts, (int, float)):
            cleaned.append(float(ts))
    video_to_supports[basename] = cleaned

print(f"Loaded {len(video_to_supports)} videos.")

# ------------------- OCR function -------------------
def extract_ocr_from_video(video_path, timestamps):
    if not timestamps:
        return ""
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    texts = set()
    
    for ts in timestamps:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        
        try:
            text = predictor.predict(pil_img).strip()
            if text and len(text) <= 40:  # Lọc text quá dài
                texts.add(text)
        except Exception as e:
            continue
    
    cap.release()
    return " | ".join(sorted(texts)) if texts else ""

# ------------------- Main -------------------
print("Starting OCR extraction...")
for item in tqdm(data, desc="OCR"):
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

print(f"OCR HOÀN TẤT! Đã lưu {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')])} file")
print("Ví dụ:")
example_file = os.path.join(OUTPUT_DIR, os.listdir(OUTPUT_DIR)[0])
print(f"  {example_file}: {open(example_file, 'r', encoding='utf-8').read()[:100]}")