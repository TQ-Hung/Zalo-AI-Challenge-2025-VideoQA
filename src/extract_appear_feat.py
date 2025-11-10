# extract_motion_feat_v2.py — PHIÊN BẢN SIÊU TỐC ĐỘ 2 GPU
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from transformers import VideoMAEFeatureExtractor, VideoMAEModel
import json
from torch import nn

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "features_v2/motion"
VIDEO_DIR = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/videos"
TRAIN_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/train.json"
BATCH_SIZE = 16  # Tăng batch để tận dụng GPU
NUM_WORKERS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load support_frames ---
print("Loading train.json...")
with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    train_data = json.load(f)["data"]

video_to_supports = {}
for item in train_data:
    basename = os.path.splitext(os.path.basename(item["video_path"]))[0]
    supports = item.get("support_frames", [])
    video_to_supports[basename] = [float(ts) for ts in supports if ts]

print(f"Loaded support info for {len(video_to_supports)} videos.")

# --- Model + Multi-GPU ---
print("Loading VideoMAE with DataParallel...")
processor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
    model = nn.DataParallel(model)  # THÊM DÒNG NÀY

model = model.to(DEVICE)
model.eval()

# --- Hàm đọc frame ưu tiên support ---
def read_prioritized_frames(video_path, num=16):
    basename = os.path.splitext(os.path.basename(video_path))[0]
    support_timestamps = video_to_supports.get(basename, [])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    frames = []
    indices_used = set()

    # Support frames
    for ts in support_timestamps:
        idx = int(ts * fps)
        idx = max(0, min(idx, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            indices_used.add(idx)

    # Bổ sung đều
    remaining = num - len(frames)
    if remaining > 0:
        step = max(1, total_frames // remaining)
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
    return frames[:num]

# --- Batch processing ---
def extract_batch(video_paths):
    all_frames = []
    valid_paths = []
    for vp in video_paths:
        frames = read_prioritized_frames(vp, num=16)
        if len(frames) == 16:
            all_frames.extend(frames)
            valid_paths.append(vp)
        else:
            print(f"Skip {vp}: only {len(frames)} frames")

    if len(all_frames) == 0:
        return []

    inputs = processor(all_frames, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    feats = outputs.last_hidden_state.mean(1)  # (N*16, 768)
    feats = feats.cpu().numpy()

    results = []
    for i, path in enumerate(valid_paths):
        start = i * 16
        end = start + 16
        feat = feats[start:end]  # (16, 768)
        results.append((path, feat))
    return results

# --- Main loop ---
if __name__ == "__main__":
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    videos = [os.path.join(VIDEO_DIR, f) for f in videos]
    videos.sort()

    print(f"Total videos: {len(videos)} | Batch size: {BATCH_SIZE}")

    results = []
    with tqdm(total=len(videos), desc="Extracting motion (2 GPU)") as pbar:
        for i in range(0, len(videos), BATCH_SIZE):
            batch_paths = videos[i:i+BATCH_SIZE]
            batch_results = extract_batch(batch_paths)
            for vp, feat in batch_results:
                basename = os.path.splitext(os.path.basename(vp))[0]
                save_path = os.path.join(OUTPUT_DIR, f"{basename}.npy")
                if not os.path.exists(save_path):
                    np.save(save_path, feat)
            pbar.update(len(batch_paths))

    print(f"Done! Saved to {OUTPUT_DIR}")
    print(f"GPU usage: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")