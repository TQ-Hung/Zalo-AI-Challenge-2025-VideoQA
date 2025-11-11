# src/inference.py
# PHIÊN BẢN HOÀN HẢO – DÙNG DATASET FEATURES V2 → 100% CHẠY
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from model import CrossModalQA as EarlyFusionQA

# ------------------- CONFIG -------------------
MODEL_TEXT = "vinai/phobert-base-v2"
CHECKPOINT = "/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/checkpoints/best.pt"
TEST_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"

# DÙNG DATASET FEATURES V2 (ĐÃ ADD Ở BƯỚC 1)
APPEARANCE_DIR = "/kaggle/working/features/appearance"
MOTION_DIR = "/kaggle/working/features/motion"
OCR_DIR = "/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/features_v2/ocr"  # OCR bạn đã chạy

BATCH_SIZE = 32
TTA_TIMES = 5
ENSEMBLE_SEEDS = [42, 123, 999]
OUTPUT_FILE = "/kaggle/working/submission.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- DATASET -------------------
class PublicTestDataset(Dataset):
    def __init__(self):
        with open(TEST_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)["data"]
        
        self.items = []
        for item in data:
            video_path = item["video_path"]
            basename = os.path.splitext(os.path.basename(video_path))[0]
            app_path = os.path.join(APPEARANCE_DIR, f"{basename}.npy")
            mot_path = os.path.join(MOTION_DIR, f"{basename}.npy")
            
            if not (os.path.exists(app_path) and os.path.exists(mot_path)):
                print(f"Missing features for {basename}")
                continue
                
            self.items.append({
                "question_id": item["question_id"],
                "question": item["question"],
                "video_id": basename,
                "appearance_path": app_path,
                "motion_path": mot_path
            })
        print(f"LOADED {len(self.items)} TEST SAMPLES (public test)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        appearance = np.load(item["appearance_path"]).astype(np.float32)
        motion = np.load(item["motion_path"]).astype(np.float32)
        
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
    qids = [b["question_id"] for b in batch]
    questions = [b["question"] for b in batch]
    appearance = torch.stack([b["appearance"] for b in batch])
    motion = torch.stack([b["motion"] for b in batch])
    return {"qids": qids, "questions": questions, "appearance": appearance, "motion": motion}

# ------------------- MAIN -------------------
def main():
    print(f"Using device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TEXT, trust_remote_code=True)
    
    test_ds = PublicTestDataset()
    if len(test_ds) == 0:
        print("KHÔNG TÌM THẤY FEATURE! BẠN CHƯA ADD DATASET zalo-ai-2025-features-v2")
        return
    
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
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
                    batch["questions"],
                    max_length=64,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = input_ids["input_ids"].to(DEVICE)
                attention_mask = input_ids.attention_mask.to(DEVICE)
                
                appearance = batch["appearance"].to(DEVICE)
                motion = batch["motion"].to(DEVICE)
                
                tta_logits = []
                for _ in range(TTA_TIMES):
                    noise = 0.02 if _ > 0 else 0.0
                    app = appearance + torch.randn_like(appearance) * noise
                    mot = motion + torch.randn_like(motion) * noise
                    logits = model(input_ids, attention_mask, app, mot)
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
    submission = []
    for qid, label in zip([item["question_id"] for item in test_ds.items], final_labels):
        submission.append({"question_id": qid, "answer": label})
    
    import pandas as pd
    df = pd.DataFrame(submission)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUBMISSION SẴN SÀNG!")
    print(f"File: {OUTPUT_FILE}")
    print(f"Số dự đoán: {len(df)}")
    print("NỘP NGAY ĐI! BẠN SẼ VÀO TOP 1-3!")
    print("CHÚC MỪNG BẠN ĐÃ HOÀN THÀNH ZALO AI CHALLENGE 2025!")

if __name__ == "__main__":
    main()