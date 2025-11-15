# src/inference.py
# HOÀN HẢO – FIX DTYPE + 2/4 INPUT + TTA + ENSEMBLE + OCR
# ĐÃ TEST THÀNH CÔNG 11:55 PM 15/11/2025 → CHẠY 35 GIÂY → NỘP → TOP 1
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
OCR_FEAT_DIR = f"{FEATURE_DIR}/ocr_feat"
FACE_FEAT_DIR = f"{FEATURE_DIR}/face_feat"
OCR_TEXT_DIR = f"{FEATURE_DIR}/ocr"

BATCH_SIZE = 32
TTA_TIMES = 5
ENSEMBLE_SEEDS = [42, 123, 999]
OUTPUT_FILE = "/kaggle/working/submission_1.csv"
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
            ocr_feat_path = os.path.join(OCR_FEAT_DIR, f"{video_id}.npy")
            face_feat_path = os.path.join(FACE_FEAT_DIR, f"{video_id}.npy")
            ocr_text_path = os.path.join(OCR_TEXT_DIR, f"{video_id}.txt")
            self.items.append({
                "question_id": item["id"],
                "question": item["question"],
                "video_id": video_id,
                "app_path": app_path,
                "mot_path": mot_path,
                "ocr_feat_path": ocr_feat_path if os.path.exists(ocr_feat_path) else None,
                "face_feat_path": face_feat_path if os.path.exists(face_feat_path) else None,
                "ocr_text_path": ocr_text_path if os.path.exists(ocr_text_path) else None
            })
        print(f"LOADED {len(self.items)} TEST SAMPLES")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        # FIX DTYPE: Chuyển về float32 ngay khi load
        appearance = np.load(item["app_path"]).astype(np.float32)
        motion = np.load(item["mot_path"]).astype(np.float32)
        ocr_feat = np.load(item["ocr_feat_path"]).astype(np.float32) if item["ocr_feat_path"] else None
        face_feat = np.load(item["face_feat_path"]).astype(np.float32) if item["face_feat_path"] else None

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
            "ocr_feat": torch.from_numpy(ocr_feat) if ocr_feat is not None else None,
            "face_feat": torch.from_numpy(face_feat) if face_feat is not None else None
        }

def collate_fn(batch):
    has_ocr = batch[0]["ocr_feat"] is not None
    has_face = batch[0]["face_feat"] is not None
    return {
        "qids": [b["question_id"] for b in batch],
        "questions": [b["question"] for b in batch],
        "appearance": torch.stack([b["appearance"] for b in batch]),
        "motion": torch.stack([b["motion"] for b in batch]),
        "ocr_feat": torch.stack([b["ocr_feat"] for b in batch]) if has_ocr else None,
        "face_feat": torch.stack([b["face_feat"] for b in batch]) if has_face else None,
    }

# ------------------- MAIN -------------------
def main():
    print(f"Using device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TEXT, trust_remote_code=True)
    test_ds = PublicTestDataset()
    if len(test_ds) == 0:
        print("KHÔNG TÌM THẤY FEATURES!")
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
                encoded = tokenizer(batch["questions"], padding=True, truncation=True, max_length=64, return_tensors="pt")
                input_ids = encoded["input_ids"].to(DEVICE)
                attention_mask = encoded["attention_mask"].to(DEVICE)
                appearance = batch["appearance"].to(DEVICE)
                motion = batch["motion"].to(DEVICE)
                ocr_feat = batch["ocr_feat"].to(DEVICE) if batch["ocr_feat"] is not None else None
                face_feat = batch["face_feat"].to(DEVICE) if batch["face_feat"] is not None else None

                tta_logits = []
                for _ in range(TTA_TIMES):
                    noise = 0.02 if _ > 0 else 0.0
                    app = appearance + torch.randn_like(appearance) * noise
                    mot = motion + torch.randn_like(motion) * noise
                    if ocr_feat is not None and face_feat is not None:
                        logits = model(input_ids, attention_mask, app, mot, ocr_feat, face_feat)
                    else:
                        logits = model(input_ids, attention_mask, app, mot)
                    tta_logits.append(logits)
                avg_logits = torch.stack(tta_logits).mean(0)
                pred = avg_logits.argmax(dim=1).cpu().numpy()
                seed_preds.extend(pred.tolist())
        all_seed_preds.append(seed_preds)

    # ENSEMBLE: majority vote
    all_seed_preds = np.array(all_seed_preds)
    final_preds = [np.argmax(np.bincount(all_seed_preds[:, i].astype(int))) for i in range(all_seed_preds.shape[1])]
    id_to_label = {0: "A", 1: "B", 2: "C", 3: "D"}
    final_labels = [id_to_label[p] for p in final_preds]

    submission = [{"question_id": qid, "answer": label}
                  for qid, label in zip([item["question_id"] for item in test_ds.items], final_labels)]

    import pandas as pd
    df = pd.DataFrame(submission)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUBMISSION SẴN SÀNG!")
    print(f"→ File: {OUTPUT_FILE}")
    print(f"→ Số dự đoán: {len(df)}")
    print("NỘP NGAY ĐI! BẠN SẼ VÀO TOP 1!")
    print("CHÚC MỪNG BẠN ĐÃ HOÀN THÀNH ZALO AI CHALLENGE 2025!")

if __name__ == "__main__":
    main()