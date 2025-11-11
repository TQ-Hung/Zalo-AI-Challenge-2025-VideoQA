# src/inference.py
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import FeatureVideoQADataset, collate_fn_inference
from model import CrossModalQA as EarlyFusionQA
from torch.utils.data import DataLoader
import random

# ----- Config -----
MODEL_TEXT = "vinai/phobert-base-v2"
APPEARANCE_DIR = "features/appearance"  # Đã update từ extract mới
MOTION_DIR = "features/motion"
CHECKPOINT = "/kaggle/working/Zalo-AI-Challenge-2025-VideoQA/checkpoints/best.pt"
TEST_JSON = "/kaggle/input/zalo-ai-challenge-2025-roadbuddy/traffic_buddy_train+public_test/public_test/public_test.json"
BATCH_SIZE = 8
OUTPUT_FILE = "/kaggle/working/submission.json"
TTA_TIMES = 3  # THÊM: số lần TTA

# -------------------

def inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {CHECKPOINT} ...")
    model = EarlyFusionQA(text_model_name=MODEL_TEXT).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    print("Preparing test dataset ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_TEXT,
        use_auth_token=os.getenv("HUGGINGFACE_TOKEN", None),
        trust_remote_code=True,
        ignore_chat_template_errors=True
    )

    test_ds = FeatureVideoQADataset(TEST_JSON, APPEARANCE_DIR, MOTION_DIR,
                                    tokenizer_name=MODEL_TEXT, max_len=64, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn_inference, num_workers=2)

    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference + TTA"):
            if batch is None:
                continue
            ids = batch["ids"]
            all_logits = []

            # THÊM TTA: Chạy nhiều lần với noise nhỏ
            for _ in range(TTA_TIMES):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                # Thêm noise nhẹ vào features (aug test-time)
                noise_level = 0.03 if _ > 0 else 0.0  # Lần đầu không noise
                appearance_feats = batch["appearance"].to(device) + torch.randn_like(batch["appearance"], device=device) * noise_level
                motion_feats = batch["motion"].to(device) + torch.randn_like(batch["motion"], device=device) * noise_level

                logits = model(input_ids, attention_mask, appearance_feats, motion_feats)
                all_logits.append(logits)

            # Vote: average logits rồi argmax
            avg_logits = torch.mean(torch.stack(all_logits), dim=0)
            preds = avg_logits.argmax(dim=1).cpu().tolist()

            for qid, pred in zip(ids, preds):
                results.append({"id": qid, "answer": int(pred)})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Done! Saved predictions to {OUTPUT_FILE}")

if __name__ == "__main__":
    inference()