# src/extract_motion_feat_videomae.py
import os, torch, cv2
import numpy as np
from tqdm import tqdm
from transformers import VideoMAEFeatureExtractor, VideoMAEModel

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "features/motion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model ---
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").to(DEVICE)
processor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model.eval()

# --- Utility ---
def read_frames(video_path, num=16):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, num).astype(int)
    frames = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret: break
        if i in idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

def extract_motion(video_path):
    frames = read_frames(video_path, num=16)
    inputs = processor(frames, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        feats = outputs.last_hidden_state.mean(1)  # mean pooling
    return feats.cpu().numpy()  # (16, 768)

# --- Main loop ---
VIDEO_DIR = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/train/videos"
for file in tqdm(os.listdir(VIDEO_DIR)):
    if not file.endswith(".mp4"): continue
    path = os.path.join(VIDEO_DIR, file)
    feats = extract_motion(path)
    np.save(os.path.join(OUTPUT_DIR, file.replace(".mp4", ".npy")), feats)
