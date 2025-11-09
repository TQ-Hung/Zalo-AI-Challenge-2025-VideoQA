# extract_motion_feat_v2.py
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from transformers import VideoMAEFeatureExtractor, VideoMAEModel
import json

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "features_v2/motion"  # Thư mục mới để phân biệt với bản cũ
VIDEO_DIR = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/videos"
TRAIN_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/train.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load train.json để lấy support_frames ---
print("Loading train.json to get support_frames...")
with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    train_data = json.load(f)["data"]

# Tạo dict: video_basename -> list support_timestamps (giây)
video_to_supports = {}
for item in train_data:
    vid_path = item["video_path"]
    basename = os.path.splitext(os.path.basename(vid_path))[0]
    supports = item.get("support_frames", [])
    # Chuyển sang float, bỏ giá trị rỗng
    video_to_supports[basename] = [float(ts) for ts in supports if ts]

print(f"Loaded support_frames for {len(video_to_supports)} videos.")

# --- Model ---
print("Loading VideoMAE model...")
processor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kineticsyed").to(DEVICE)
model.eval()

# --- HÀM ƯU TIÊN SUPPORT FRAMES ---
def read_prioritized_frames(video_path, num=16):
    basename = os.path.splitext(os.path.basename(video_path))[0]
    support_timestamps = video_to_supports.get(basename, [])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    frames = []
    indices_used = set()

    # 1. Lấy support frames trước
    for ts in support_timestamps:
        frame_idx = int(ts * fps)
        frame_idx = max(0, min(frame_idx, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            indices_used.add(frame_idx)

    # 2. Bổ sung frame đều nhau nếu chưa đủ
    remaining = num - len(frames)
    if remaining > 0:
        # Lấy các frame chưa dùng
        step = max(1, (total_frames - 1) // remaining)
        idx = 0
        added = 0
        while added < remaining and idx < total_frames:
            if idx not in indices_used:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    added += 1
            idx += step

    cap.release()
    return frames[:num]  # Đảm bảo đúng 16 frame

# --- Trích xuất motion features ---
@torch.inference_mode()
def extract_motion(video_path):
    frames = read_prioritized_frames(video_path, num=16)
    if len(frames) == 0:
        return None
    inputs = processor(frames, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    feats = outputs.last_hidden_state.mean(1)  # mean pooling -> (16, 768)
    return feats.cpu().numpy()

# --- Main loop ---
if __name__ == "__main__":
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    print(f"Found {len(videos)} videos. Starting extraction...")

    for file in tqdm(videos, desc="Extracting motion features (v2 - support_frames)"):
        save_path = os.path.join(OUTPUT_DIR, file.replace(".mp4", ".npy"))
        if os.path.exists(save_path):
            continue  # Bỏ qua nếu đã extract

        video_path = os.path.join(VIDEO_DIR, file)
        feats = extract_motion(video_path)
        if feats is not None:
            np.save(save_path, feats)

    print(f"Done! Motion features saved to {OUTPUT_DIR}")