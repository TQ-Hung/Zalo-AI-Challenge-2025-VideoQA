# extract_motion_feat_v2.py
# FINAL VERSION – CHẠY ỔN ĐỊNH 2 GPU + support_frames + fix all errors
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from transformers import VideoMAEImageProcessor, VideoMAEModel
import json
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "features_test/motion"
VIDEO_DIR = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/videos"
TRAIN_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- Load support_frames (FIX LỖI FLOAT) -------------------
print("Loading train.json...")
with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    train_data = json.load(f)["data"]

video_to_supports = {}
for item in train_data:
    basename = os.path.splitext(os.path.basename(item["video_path"]))[0]
    supports = item.get("support_frames", [])
    # FIX: xử lý cả string lẫn float
    cleaned = []
    for ts in supports:
        if isinstance(ts, str):
            ts = ts.strip()
            if ts:  # bỏ rỗng
                cleaned.append(float(ts))
        elif isinstance(ts, (int, float)):
            cleaned.append(float(ts))
    video_to_supports[basename] = cleaned

print(f"Loaded support_frames for {len(video_to_supports)} videos.")

# ------------------- Model -------------------
print("Loading VideoMAE...")
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(DEVICE)
model.eval()

# ------------------- Read frames -------------------
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

    for ts in support_timestamps:
        idx = int(ts * fps)
        idx = max(0, min(idx, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            indices_used.add(idx)

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

# ------------------- Extract single -------------------
@torch.inference_mode()
def extract_single(video_path):
    frames = read_prioritized_frames(video_path, num=16)
    if len(frames) != 16:
        return None

    inputs = processor(frames, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = model(**inputs)
    feats = outputs.last_hidden_state.mean(dim=1)
    return feats.cpu().numpy()

# ------------------- Main -------------------
if __name__ == "__main__":
    videos = sorted([os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])
    print(f"Found {len(videos)} videos.")

    skipped = 0
    for vp in tqdm(videos, desc="Motion (2 GPU)"):
        basename = os.path.splitext(os.path.basename(vp))[0]
        save_path = os.path.join(OUTPUT_DIR, f"{basename}.npy")
        if os.path.exists(save_path):
            continue
        feats = extract_single(vp)
        if feats is not None:
            np.save(save_path, feats)
        else:
            skipped += 1

    print(f"MOTION DONE! Saved {len(videos)-skipped}, skipped {skipped}")
    if torch.cuda.is_available():
        print(f"GPU memory peak: {torch.cuda.max_memory_allocated()/1024**3:.1f} GB")