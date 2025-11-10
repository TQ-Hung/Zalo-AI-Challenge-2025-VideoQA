# extract_motion_feat_v2.py
# Phiên bản: FULL GPU 2x + support_frames + batch processing
# Thời gian chạy trên Kaggle 2xT4: ~12-18 phút

import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from transformers import VideoMAEFeatureExtractor, VideoMAEModel
import json
from torch import nn

# ------------------- Config -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "features_v2/motion"          # Thư mục lưu .npy
VIDEO_DIR = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/videos"
TRAIN_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/train.json"

BATCH_SIZE = 8          # 8 video/lần → tận dụng hết GPU memory
NUM_WORKERS = 4         # Không dùng DataLoader nhưng vẫn để lại cho tương lai

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- Load support_frames -------------------
print("Loading train.json để lấy support_frames...")
with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    train_data = json.load(f)["data"]

video_to_supports = {}
for item in train_data:
    basename = os.path.splitext(os.path.basename(item["video_path"]))[0]
    supports = item.get("support_frames", [])
    video_to_supports[basename] = [float(ts) for ts in supports if ts]

print(f"Đã load support_frames cho {len(video_to_supports)} video.")

# ------------------- Model + Multi-GPU -------------------
print("Loading VideoMAE model...")
processor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

# QUAN TRỌNG: Dùng DataParallel để tận dụng 2 GPU
if torch.cuda.device_count() > 1:
    print(f"Sử dụng {torch.cuda.device_count()} GPU với DataParallel!")
    model = nn.DataParallel(model)

model = model.to(DEVICE)
model.eval()

# ------------------- Hàm ưu tiên support frames -------------------
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

    # 1. Lấy support frames trước
    for ts in support_timestamps:
        idx = int(ts * fps)
        idx = max(0, min(idx, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            indices_used.add(idx)

    # 2. Bổ sung frame đều nếu chưa đủ
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
    return frames[:num]   # Đảm bảo luôn 16 frame

# ------------------- Batch extraction (tận dụng GPU) -------------------
def extract_batch(video_paths):
    all_frames = []
    valid_paths = []

    for vp in video_paths:
        frames = read_prioritized_frames(vp, num=16)
        if len(frames) == 16:               # Chỉ lấy video đủ 16 frame
            all_frames.extend(frames)
            valid_paths.append(vp)
        else:
            print(f"Skip {os.path.basename(vp)}: chỉ có {len(frames)} frame")

    if len(all_frames) == 0:
        return []

    # Processor tự động chia thành (N*16, C, H, W)
    inputs = processor(all_frames, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
    feats = outputs.last_hidden_state.mean(dim=1)   # (N*16, 768)
    feats = feats.cpu().numpy()

    results = []
    for i, vp in enumerate(valid_paths):
        start = i * 16
        end = start + 16
        feat = feats[start:end]                 # (16, 768)
        results.append((vp, feat))
    return results

# ------------------- Main loop -------------------
if __name__ == "__main__":
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    video_paths = [os.path.join(VIDEO_DIR, f) for f in video_files]
    video_paths.sort()

    print(f"Tìm thấy {len(video_paths)} video. Bắt đầu extract với batch_size={BATCH_SIZE}")

    with tqdm(total=len(video_paths), desc="Extracting motion (2 GPU + support_frames)") as pbar:
        for i in range(0, len(video_paths), BATCH_SIZE):
            batch_paths = video_paths[i:i + BATCH_SIZE]
            batch_results = extract_batch(batch_paths)

            for vp, feat in batch_results:
                basename = os.path.splitext(os.path.basename(vp))[0]
                save_path = os.path.join(OUTPUT_DIR, f"{basename}.npy")
                if not os.path.exists(save_path):
                    np.save(save_path, feat)

            pbar.update(len(batch_paths))

    # In thông tin GPU để kiểm tra
    if torch.cuda.is_available():
        print(f"GPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        for gpu_id in range(torch.cuda.device_count()):
            print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    print(f"HOÀN TẤT! Motion features đã lưu tại: {OUTPUT_DIR}")