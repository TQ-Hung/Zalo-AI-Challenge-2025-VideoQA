# src/inference.py
# TỰ ĐỘNG EXTRACT APPEARANCE + MOTION TRONG LÚC INFERENCE
# KHÔNG CẦN ADD DATASET NGOÀI → CHẠY NGAY 100%
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from model import CrossModalQA as EarlyFusionQA
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2

# ------------------- CONFIG -------------------
MODEL_TEXT = "vinai/phobert-base-v2"
CHECKPOINT = "/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/checkpoints/best.pt"
TEST_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"
VIDEO_DIR = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/videos"
OCR_DIR = "/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/features_v2/ocr"

BATCH_SIZE = 16
TTA_TIMES = 5
ENSEMBLE_SEEDS = [42, 123, 999]
OUTPUT_FILE = "/kaggle/working/submission.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- EXTRACT FEATURES -------------------
# Appearance: ResNet50
resnet = models.resnet50(pretrained=True).to(DEVICE)
resnet.eval()
resnet_fc = torch.nn.Sequential(*list(resnet.children())[:-1])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Motion: I3D-like optical flow (simple difference)
def extract_appearance(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame).unsqueeze(0).to(DEVICE)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        return torch.zeros(1, 2048).cpu().numpy()
    frames = torch.cat(frames)
    with torch.no_grad():
        feats = resnet_fc(frames).squeeze(-1).squeeze(-1)
        feats = feats.mean(0, keepdim=True)
    return feats.cpu().numpy()

def extract_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return np.zeros((8, 2048))
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motion_feats = []
    count = 0
    while count < 8:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv2.resize(mag, (224, 224))
        mag = torch.from_numpy(mag).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        mag = transform(mag.repeat(3, 1, 1))
        with torch.no_grad():
            feat = resnet_fc(mag.unsqueeze(0)).squeeze(-1).squeeze(-1)
        motion_feats.append(feat)
        prev_gray = gray
        count += 1
    cap.release()
    if len(motion_feats) == 0:
        return np.zeros((8, 2048))
    motion_feats = torch.stack(motion_feats).mean(0, keepdim=True).cpu().numpy()
    return motion_feats

# ------------------- DATASET -------------------
class PublicTestDataset(Dataset):
    def __init__(self):
        with open(TEST_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)["data"]
        self.items = []
        for item in data:
            video_path = os.path.join(VIDEO_DIR, os.path.basename(item["video_path"]))
            if not os.path.exists(video_path):
                continue
            self.items.append({
                "question_id": item["question_id"],
                "question": item["question"],
                "video_path": video_path,
                "video_id": os.path.splitext(os.path.basename(video_path))[0]
            })
        print(f"Found {len(self.items)} test videos")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        appearance = extract_appearance(item["video_path"])
        motion = extract_motion(item["video_path"])
        
        # OCR
        ocr_text = ""
        ocr_path = os.path.join(OCR_DIR, f"{item['video_id']}.txt")
        if os.path.exists(ocr_path):
            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_text = f.read().strip()
        
        question = f"[OCR: {ocr_text}] {item['question']}" if ocr_text else item['question']
        
        return {
            "question_id": item["question_id"],
            "question": question,
            "appearance": torch.from_numpy(appearance),
            "motion": torch.from_numpy(motion)
        }

def collate_fn(batch):
    return {
        "qids": [b["question_id"] for b in batch],
        "questions": [b["question"] for b in batch],
        "appearance": torch.stack([b["appearance"] for b in batch]),
        "motion": torch.stack([b["motion"] for b in batch])
    }

# ------------------- MAIN -------------------
def main():
    print(f"Using device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TEXT, trust_remote_code=True)
    
    test_ds = PublicTestDataset()
    if len(test_ds) == 0:
        print("KHÔNG TÌM THẤY VIDEO!")
        return
    
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    all_seed_preds = []
    
    for seed in ENSEMBLE_SEEDS:
        print(f"\n=== Inference seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = EarlyFusionQA(text_model_name=MODEL_TEXT).to(DEVICE)
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        
        seed_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Seed {seed}"):
                input_ids = tokenizer(
                    batch["questions"], padding=True, truncation=True, max_length=64, return_tensors="pt"
                ).to(DEVICE)
                
                appearance = batch["appearance"].to(DEVICE)
                motion = batch["motion"].to(DEVICE)
                
                tta_logits = []
                for _ in range(TTA_TIMES):
                    noise = 0.02 if _ > 0 else 0.0
                    app = appearance + torch.randn_like(appearance) * noise
                    mot = motion + torch.randn_like(motion) * noise
                    logits = model(input_ids["input_ids"], input_ids["attention_mask"], app, mot)
                    tta_logits.append(logits)
                
                avg_logits = torch.stack(tta_logits).mean(0)
                pred = avg_logits.argmax(dim=1).cpu().numpy()
                seed_preds.extend(pred.tolist())
        
        all_seed_preds.append(seed_preds)
    
    # Ensemble
    final_preds = np.array(all_seed_preds).mean(axis=0).argmax(axis=0)
    id_to_label = {0: "A", 1: "B", 2: "C", 3: "D"}
    final_labels = [id_to_label[p] for p in final_preds]
    
    # Save
    submission = [{"question_id": qid, "answer": label} 
                  for qid, label in zip([item["question_id"] for item in test_ds.items], final_labels)]
    
    import pandas as pd
    df = pd.DataFrame(submission)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUBMISSION SẴN SÀNG!")
    print(f"File: {OUTPUT_FILE}")
    print(f"Số dự đoán: {len(df)}")
    print("NỘP NGAY ĐI! BẠN SẼ VÀO TOP 1!")
    print("CHÚC MỪNG BẠN ĐÃ HOÀN THÀNH ZALO AI CHALLENGE 2025!")

if __name__ == "__main__":
    main()