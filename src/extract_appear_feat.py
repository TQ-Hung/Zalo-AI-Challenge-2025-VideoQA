# src/extract_appear_feat_dino.py
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "features/appearance"
VIDEO_DIR = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model & Processor ---
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
model = AutoModel.from_pretrained("facebook/dinov2-base", token=HF_TOKEN)
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", token=HF_TOKEN)
model.eval()

# --- Utility: đọc khung hình đều nhau ---
def read_frames(video_path, num=16):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, total - 1, num).astype(int)
    frames = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

# --- Trích xuất đặc trưng ---
@torch.inference_mode()
def extract_appearance(video_path):
    frames = read_frames(video_path, num=16)
    if len(frames) == 0:
        return None
    inputs = processor(images=frames, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    feats = outputs.last_hidden_state.mean(1)  # mean pooling over patch tokens
    return feats.cpu().numpy()  # (16, 768)

# --- Main loop ---
if __name__ == "__main__":
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    for file in tqdm(videos, desc="Extracting appearance features"):
        save_path = os.path.join(OUTPUT_DIR, file.replace(".mp4", ".npy"))
        if os.path.exists(save_path):
            continue  # bỏ qua file đã xử lý
        video_path = os.path.join(VIDEO_DIR, file)
        feats = extract_appearance(video_path)
        if feats is not None:
            np.save(save_path, feats)
