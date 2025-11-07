# src/inference.py
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import FeatureVideoQADataset, collate_fn, collate_fn_inference
from model import EarlyFusionQA
from torch.utils.data import DataLoader

# ----- Config -----
MODEL_TEXT = "vinai/phobert-base"
APPEARANCE_DIR = "features_test/appearance"
MOTION_DIR = "features_test/motion"
CHECKPOINT = "/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/checkpoints/best.pt"
TEST_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"
BATCH_SIZE = 8
OUTPUT_FILE = "/kaggle/working/submission.json"
# -------------------

def inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading model from {CHECKPOINT} ...")
    model = EarlyFusionQA(text_model_name=MODEL_TEXT).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # Load tokenizer + dataset
    print("Preparing test dataset ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TEXT)
    test_ds = FeatureVideoQADataset(TEST_JSON, APPEARANCE_DIR, MOTION_DIR,
                                    tokenizer_name=MODEL_TEXT, max_len=64, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn_inference, num_workers=2)

    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            if batch is None:  # bỏ qua batch rỗng
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            appearance_feats = batch["appearance"].to(device)
            motion_feats = batch["motion"].to(device)
            ids = batch["ids"]

            logits = model(input_ids, attention_mask, appearance_feats, motion_feats)
            preds = logits.argmax(dim=1).cpu().tolist()

            for qid, pred in zip(ids, preds):
                results.append({"id": qid, "answer": int(pred)})

    # Lưu kết quả
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Done! Saved predictions to {OUTPUT_FILE}")

if __name__ == "__main__":
    inference()
