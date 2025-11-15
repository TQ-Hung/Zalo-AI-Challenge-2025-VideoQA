# src/inference.py
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
FEATURE_DIR = "/kaggle/working/features_test"
APP_DIR = f"{FEATURE_DIR}/appearance"
MOT_DIR = f"{FEATURE_DIR}/motion"
OCR_TEXT_DIR = f"{FEATURE_DIR}/ocr"

BATCH_SIZE = 32
TTA_TIMES = 3  # test-time augmentation
OUTPUT_FILE = "/kaggle/working/submission_2.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- DATASET -------------------
class PublicTestDataset(Dataset):
    def __init__(self):
        with open(TEST_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)["data"]
        self.items = []
        for item in data:
            video_id = os.path.splitext(os.path.basename(item["video_path"]))[0]
            app_path = os.path.join(APP_DIR, f"{video_id}.npy")
            mot_path = os.path.join(MOT_DIR, f"{video_id}.npy")
            if not (os.path.exists(app_path) and os.path.exists(mot_path)):
                continue
            ocr_text_path = os.path.join(OCR_TEXT_DIR, f"{video_id}.txt")
            self.items.append({
                "question_id": item["id"],
                "question": item["question"],
                "video_id": video_id,
                "app_path": app_path,
                "mot_path": mot_path,
                "ocr_text_path": ocr_text_path if os.path.exists(ocr_text_path) else None
            })
        print(f"LOADED {len(self.items)} TEST SAMPLES")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        appearance = np.load(item["app_path"]).astype(np.float32)
        motion = np.load(item["mot_path"]).astype(np.float32)

        ocr_text = ""
        if item["ocr_text_path"]:
            with open(item["ocr_text_path"], "r", encoding="utf-8") as f:
                ocr_text = f.read().strip()

        question = f"[OCR: {ocr_text}] {item['question']}" if ocr_text else item['question']

        return {
            "question_id": item["question_id"],
            "question": question,
            "appearance": torch.from_numpy(appearance),
            "motion": torch.from_numpy(motion),
        }

def collate_fn(batch):
    return {
        "qids": [b["question_id"] for b in batch],
        "questions": [b["question"] for b in batch],
        "appearance": torch.stack([b["appearance"] for b in batch]),
        "motion": torch.stack([b["motion"] for b in batch]),
    }

# ------------------- MAIN -------------------
def main():
    print(f"Using device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TEXT, trust_remote_code=True)
    test_ds = PublicTestDataset()
    if len(test_ds) == 0:
        return

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    model = EarlyFusionQA(text_model_name=MODEL_TEXT).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            # Tokenize questions
            encoded = tokenizer(batch["questions"], padding=True, truncation=True,
                                max_length=64, return_tensors="pt")
            input_ids = encoded["input_ids"].to(DEVICE)
            attention_mask = encoded["attention_mask"].to(DEVICE)
            appearance = batch["appearance"].to(DEVICE)
            motion = batch["motion"].to(DEVICE)

            # TTA
            tta_logits = []
            for tta in range(TTA_TIMES):
                noise = 0.02 if tta > 0 else 0.0
                app = appearance + torch.randn_like(appearance) * noise
                mot = motion + torch.randn_like(motion) * noise
                logits = model(input_ids, attention_mask, app, mot)  # shape: (B, num_choices)
                tta_logits.append(logits)
            avg_logits = torch.stack(tta_logits).mean(0)

            # argmax trên trục class
            pred = avg_logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(pred.tolist())

    id_to_label = {0: "A", 1: "B", 2: "C", 3: "D"}
    final_labels = [id_to_label[p] for p in all_preds]

    submission = [{"question_id": qid, "answer": label}
                  for qid, label in zip([item["question_id"] for item in test_ds.items], final_labels)]

    import pandas as pd
    df = pd.DataFrame(submission)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUBMISSION SẴN SÀNG!")
    print(f"→ File: {OUTPUT_FILE}")
    print(f"→ Phân bố: A:{final_labels.count('A')}, B:{final_labels.count('B')}, C:{final_labels.count('C')}, D:{final_labels.count('D')}")
    print("NỘP NGAY TRƯỚC 12:00 AM!")

if __name__ == "__main__":
    main()
